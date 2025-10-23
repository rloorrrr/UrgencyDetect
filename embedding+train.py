# =========================================================
# 🚀 Whisper + KoELECTRA (자동 분할 + 성능 개선 버전)
#   - Cross-Attention (2층, head=8)
#   - Text: CLS+Mean 하이브리드 풀링
#   - Audio: GroupNorm(RMSNorm 대체)
#   - FocalLoss(γ=1.8) + WeightedSampler
#   - Unfreeze: Whisper 4층, KoELECTRA 9층
#   - CosineAnnealingWarmRestarts + EarlyStop(F1)
#   - 자동 분할(Group/일반) 선택
# =========================================================

# !pip install -q transformers accelerate torch torchaudio librosa scikit-learn

import os, re, random, numpy as np, pandas as pd
import torch, torch.nn as nn, torch.optim as optim
import librosa
from tqdm.auto import tqdm
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.metrics import f1_score, classification_report
from transformers import WhisperFeatureExtractor, WhisperModel, AutoTokenizer, AutoModel

# ---------------------------------------------------------
# ⚙️ 기본 설정
# ---------------------------------------------------------
SEED = 42
random.seed(SEED); np.random.seed(SEED)
torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
os.environ["HF_HUB_DISABLE_XET"] = "1"
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("✅ Device:", device)

# ---------------------------------------------------------
# 파일 경로 및 파라미터
# ---------------------------------------------------------
DF_PATH = "./preprocess_data/preprocess.csv"
assert os.path.exists(DF_PATH), f"❌ 파일 없음: {DF_PATH}"

AUDIO_SR = 16000
WHISPER_NAME = "openai/whisper-small"
TEXT_NAME    = "monologg/koelectra-base-v3-discriminator"

BATCH_SIZE = 16
LR = 1e-5
EPOCHS = 3
FOCAL_GAMMA = 1.8
GRAD_CLIP = 1.0
PATIENCE = 5
HIDDEN_DIM = 512
NUM_CLASSES = 3
DROPOUT_FUSION = 0.3
UNFREEZE_AUDIO_LAYERS = 4
UNFREEZE_TEXT_LAYERS  = 9

# ---------------------------------------------------------
# 데이터 로드 및 그룹 컬럼 생성
# ---------------------------------------------------------
df_full = pd.read_csv(DF_PATH)
df = df_full[:1500]
assert all(c in df.columns for c in ["audio_path","all_text","label"]), "CSV 컬럼 누락"

def get_group(x):
    d = os.path.basename(os.path.dirname(str(x)))
    return d if d else os.path.basename(str(x))

df["group_id"] = df["audio_path"].apply(get_group)
print("✅ 데이터 로드 완료:", df.shape)

# ---------------------------------------------------------
# 자동 분할 로직
# ---------------------------------------------------------
unique_groups = df["group_id"].nunique()
print(f"그룹 개수: {unique_groups}")

if unique_groups >= 5:
    print("📦 Group 기반 분할 사용")
    outer = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED)
    tr_idx, te_idx = next(outer.split(df, df["label"], df["group_id"]))
    tr_df_full = df.iloc[tr_idx].reset_index(drop=True)
    te_df = df.iloc[te_idx].reset_index(drop=True)

    inner = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED)
    tr2_idx, va_idx = next(inner.split(tr_df_full, tr_df_full["label"], tr_df_full["group_id"]))
    train_df = tr_df_full.iloc[tr2_idx].reset_index(drop=True)
    val_df   = tr_df_full.iloc[va_idx].reset_index(drop=True)
else:
    print("⚙️ 일반 분할(train_test_split) 사용")
    train_df, te_df = train_test_split(df, test_size=0.2, random_state=SEED, stratify=df["label"])
    train_df, val_df = train_test_split(train_df, test_size=0.2, random_state=SEED, stratify=train_df["label"])

print(f"✅ 분할 완료 → train={len(train_df)}, val={len(val_df)}, test={len(te_df)}")

# ---------------------------------------------------------
# Processor / Tokenizer
# ---------------------------------------------------------
feature_extractor = WhisperFeatureExtractor.from_pretrained(WHISPER_NAME)
tok_text = AutoTokenizer.from_pretrained(TEXT_NAME)

# ---------------------------------------------------------
# Dataset & Collate
# ---------------------------------------------------------
class MultiModalDataset(Dataset):
    def __init__(self, df):
        self.paths  = df["audio_path"].tolist()
        self.texts  = df["all_text"].tolist()
        self.labels = df["label"].tolist()
    def __len__(self): return len(self.paths)
    def __getitem__(self, idx):
        y, sr = librosa.load(self.paths[idx], sr=AUDIO_SR, mono=True)
        a = feature_extractor(y, sampling_rate=AUDIO_SR, return_tensors="pt")
        t = tok_text(self.texts[idx], truncation=True, padding="max_length", max_length=128, return_tensors="pt")
        return {
            "audio_inputs": {k: v.squeeze(0) for k,v in a.items()},
            "text_inputs" : {k: v.squeeze(0) for k,v in t.items()},
            "label": torch.tensor(self.labels[idx], dtype=torch.long)
        }

def collate_fn(batch):
    audio = torch.stack([b["audio_inputs"]["input_features"] for b in batch])
    input_ids = torch.stack([b["text_inputs"]["input_ids"] for b in batch])
    attention_mask = torch.stack([b["text_inputs"]["attention_mask"] for b in batch])
    labels = torch.stack([b["label"] for b in batch])
    return (
        {"input_features": audio},
        {"input_ids": input_ids, "attention_mask": attention_mask},
        labels
    )

# ---------------------------------------------------------
# Audio/Text Encoder + Fusion
# ---------------------------------------------------------
class AudioSeqEncoder(nn.Module):
    def __init__(self, whisper_encoder, in_dim=768, hdim=HIDDEN_DIM):
        super().__init__()
        self.base = whisper_encoder
        self.proj = nn.Linear(in_dim, hdim)
        self.rms = nn.GroupNorm(1, hdim)
        self.norm = nn.LayerNorm(hdim)
    def forward(self, x):
        out = self.base(**x).last_hidden_state
        h = self.proj(out)
        h = self.rms(h.transpose(1,2)).transpose(1,2)
        return self.norm(h)

class TextSeqEncoder(nn.Module):
    def __init__(self, text_encoder, in_dim=768, hdim=HIDDEN_DIM):
        super().__init__()
        self.base = text_encoder
        self.proj = nn.Linear(in_dim*2, hdim)
        self.norm = nn.LayerNorm(hdim)
    def forward(self, x):
        out = self.base(**x).last_hidden_state
        cls, mean = out[:,0,:], out.mean(dim=1)
        h = torch.cat([cls, mean], dim=1)
        return self.norm(self.proj(h)).unsqueeze(1)

class CrossAttentionFusion(nn.Module):
    def __init__(self, hdim=HIDDEN_DIM, heads=8, dropout=DROPOUT_FUSION):
        super().__init__()
        self.a2t = nn.ModuleList([nn.MultiheadAttention(hdim, heads, batch_first=True) for _ in range(2)])
        self.t2a = nn.ModuleList([nn.MultiheadAttention(hdim, heads, batch_first=True) for _ in range(2)])
        self.norm = nn.LayerNorm(hdim*4)
        self.bn = nn.BatchNorm1d(hdim*4)
        self.drop = nn.Dropout(dropout)
    def forward(self, A, T):
        for attn in self.a2t: A,_ = attn(A,T,T)
        for attn in self.t2a: T,_ = attn(T,A,A)
        fused = torch.cat([A.mean(1),A.max(1).values,T.mean(1),T.max(1).values],1)
        fused = self.norm(fused)
        fused = self.drop(fused)
        return self.bn(fused)

# ---------------------------------------------------------
# 전체 모델
# ---------------------------------------------------------
class MultiModalClassifier(nn.Module):
    def __init__(self, whisper_name, text_name):
        super().__init__()
        whisper = WhisperModel.from_pretrained(whisper_name).encoder
        text = AutoModel.from_pretrained(text_name)
        for p in whisper.parameters(): p.requires_grad = False
        for p in text.parameters(): p.requires_grad = False
        for l in whisper.layers[-UNFREEZE_AUDIO_LAYERS:]:
            for p in l.parameters(): p.requires_grad = True
        for l in text.encoder.layer[-UNFREEZE_TEXT_LAYERS:]:
            for p in l.parameters(): p.requires_grad = True
        self.audio = AudioSeqEncoder(whisper)
        self.text = TextSeqEncoder(text)
        self.fusion = CrossAttentionFusion()
        self.head = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(HIDDEN_DIM*4,384),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(384,NUM_CLASSES)
        )
    def forward(self,a,t):
        A,T = self.audio(a),self.text(t)
        return self.head(self.fusion(A,T))

# ---------------------------------------------------------
# Focal Loss / Loader
# ---------------------------------------------------------
class FocalLoss(nn.Module):
    def __init__(self,gamma=1.8): super().__init__(); self.g=gamma
    def forward(self,logits,y):
        ce=torch.nn.functional.cross_entropy(logits,y,reduction='none')
        pt=torch.exp(-ce)
        return (((1-pt)**self.g)*ce).mean()

train_labels = train_df["label"].values
cls_counts = np.bincount(train_labels, minlength=NUM_CLASSES)
w = 1/(cls_counts+1e-9); w = w/w.sum()
sample_weights = w[train_labels]
sampler = WeightedRandomSampler(torch.from_numpy(sample_weights).float(), len(sample_weights), replacement=True)

train_loader = DataLoader(MultiModalDataset(train_df), batch_size=BATCH_SIZE, sampler=sampler, collate_fn=collate_fn)
val_loader = DataLoader(MultiModalDataset(val_df), batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)
test_loader = DataLoader(MultiModalDataset(te_df), batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

# ---------------------------------------------------------
# 학습 세팅
# ---------------------------------------------------------
model = MultiModalClassifier(WHISPER_NAME,TEXT_NAME).to(device)
criterion = FocalLoss()
optimizer = optim.AdamW(filter(lambda p:p.requires_grad,model.parameters()), lr=LR)
scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer,T_0=3,T_mult=2)
scaler = torch.cuda.amp.GradScaler(enabled=(device.type=="cuda"))

# ---------------------------------------------------------
# 평가 함수
# ---------------------------------------------------------
@torch.no_grad()
def evaluate(dl):
    model.eval(); preds,trues=[],[]
    for a,t,l in dl:
        a={k:v.to(device) for k,v in a.items()}
        t={k:v.to(device) for k,v in t.items()}
        out=model(a,t)
        preds+=out.argmax(1).cpu().tolist()
        trues+=l.tolist()
    return f1_score(trues,preds,average="macro"),classification_report(trues,preds,digits=4)

# ---------------------------------------------------------
# 학습 루프
# ---------------------------------------------------------
best,bstate,pat=0,None,0
for e in range(1,EPOCHS+1):
    model.train(); loop=tqdm(train_loader,desc=f"Epoch {e}/{EPOCHS}")
    for a,t,l in loop:
        a={k:v.to(device) for k,v in a.items()}
        t={k:v.to(device) for k,v in t.items()}
        l=l.to(device)
        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=(device.type=="cuda")):
            out=model(a,t); loss=criterion(out,l)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(),GRAD_CLIP)
        scaler.step(optimizer); scaler.update()
        loop.set_postfix(loss=f"{loss.item():.4f}")
    scheduler.step(e)
    val_f1,_=evaluate(val_loader)
    print(f"[VAL] Macro-F1: {val_f1:.4f}")
    if val_f1>best:
        best,bstate,pat=val_f1,{k:v.cpu() for k,v in model.state_dict().items()},0
        print("🔥 Best updated!")
        SAVE_PATH = "/content/drive/MyDrive/test_data/model/best_model.pt"
        os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)
        torch.save(bstate, SAVE_PATH)
        print(f"🔥 Best updated & saved → {SAVE_PATH}")
    else:
        pat+=1
        if pat>=PATIENCE: print("⛔ Early stop"); break

if bstate: model.load_state_dict(bstate)
test_f1,rep=evaluate(test_loader)
print("\n================ [TEST RESULT] ================")
print(f"Macro-F1: {test_f1:.4f}")
print(rep)

# ---------------------------------------------------------
# 예측 함수
# ---------------------------------------------------------
id2label={0:"하",1:"중",2:"상"}
@torch.no_grad()
def predict_urgency(audio_path,text):
    model.eval()
    y,_=librosa.load(audio_path,sr=AUDIO_SR,mono=True)
    a=feature_extractor(y,sampling_rate=AUDIO_SR,return_tensors="pt")
    t=tok_text(text,truncation=True,padding="max_length",max_length=128,return_tensors="pt")
    a={k:v.to(device) for k,v in a.items()}
    t={k:v.to(device) for k,v in t.items()}
    logits=model(a,t)
    p=torch.softmax(logits,1)[0].cpu().numpy()
    return {"pred_label":id2label[int(p.argmax())],
            "probability":{"하":float(p[0]),"중":float(p[1]),"상":float(p[2])}}
