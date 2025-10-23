# =========================================================
# 1. 라이브러리 로드
# =========================================================
import os
import numpy as np
import pandas as pd
import torch
import random
import copy
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from sklearn.metrics import f1_score, classification_report, accuracy_score
from collections import Counter
from sklearn.model_selection import train_test_split

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"✅ Device: {device}")

# =========================================================
# 2. 데이터 로드
# =========================================================
SAVE_DIR = "/content/drive/MyDrive/embeddings_backup"

meta_path = os.path.join(SAVE_DIR, "metadata.csv")
emb_path  = os.path.join(SAVE_DIR, "embeddings.npz")

df_meta = pd.read_csv(meta_path)
data = np.load(emb_path)

Xa = data["audio"]
Xt = data["text"]
y  = df_meta["urgency_y"].to_numpy()

print(f"오디오 임베딩: {Xa.shape}")
print(f"텍스트 임베딩: {Xt.shape}")
print(f"라벨: {y.shape}")

# =========================================================
# 3. 학습 데이터 분할 및 Tensor 변환
# =========================================================
Xa_tr, Xa_val, Xt_tr, Xt_val, y_tr, y_val = train_test_split(
    Xa, Xt, y, test_size=0.2, stratify=y, random_state=42
)

Xa_tr_t = torch.FloatTensor(Xa_tr)
Xt_tr_t = torch.FloatTensor(Xt_tr)
Xa_val_t = torch.FloatTensor(Xa_val)
Xt_val_t = torch.FloatTensor(Xt_val)
y_tr_t = torch.LongTensor(y_tr)
y_val_t = torch.LongTensor(y_val)

# =========================================================
# 4. 시드 고정
# =========================================================
def set_seed(seed=123):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
set_seed(123)

# =========================================================
# 5. 데이터로더 생성
# =========================================================
class PairDataset(Dataset):
    def __init__(self, a, t, y):
        self.a, self.t, self.y = a, t, y
    def __len__(self):
        return len(self.y)
    def __getitem__(self, idx):
        return self.a[idx], self.t[idx], self.y[idx]

class_counts = Counter(y_tr_t.cpu().numpy())
total = len(y_tr_t)
weights = torch.sqrt(torch.FloatTensor([
    total / (3 * class_counts[0]),
    total / (3 * class_counts[1]),
    total / (3 * class_counts[2])
])).to(device)

sample_weights = torch.DoubleTensor([weights.cpu()[y_i].item() for y_i in y_tr_t])
sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)

train_loader = DataLoader(PairDataset(Xa_tr_t, Xt_tr_t, y_tr_t), batch_size=32, sampler=sampler)
val_loader = DataLoader(PairDataset(Xa_val_t, Xt_val_t, y_val_t), batch_size=32, shuffle=False)

# =========================================================
# 6. 모델 정의
# =========================================================
class CrossAttentionFusionOrdinal(nn.Module):
    def __init__(self, audio_dim, text_dim=768, hidden_dim=768, num_classes=3, nhead=8):
        super().__init__()
        self.audio_proj = nn.Sequential(nn.Linear(audio_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU(), nn.Dropout(0.4))
        self.text_proj = nn.Sequential(nn.Linear(text_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU(), nn.Dropout(0.4))
        enc_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=nhead, dim_feedforward=2048, dropout=0.2, activation='gelu', batch_first=True)
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=3)
        fusion_dim = hidden_dim * 4
        self.classifier = nn.Sequential(nn.Linear(fusion_dim, 512), nn.LayerNorm(512), nn.GELU(), nn.Dropout(0.4), nn.Linear(512, num_classes))

    def forward(self, a, t):
        a = self.audio_proj(a); t = self.text_proj(t)
        x = torch.stack([a, t], dim=1)
        h = self.encoder(x)
        pooled = torch.cat([h[:,0,:], h[:,1,:], torch.abs(h[:,0,:]-h[:,1,:]), h[:,0,:]*h[:,1,:]], dim=-1)
        return self.classifier(pooled)

# =========================================================
# 7. Loss 및 Optimizer
# =========================================================
class OrdinalFocalLoss(nn.Module):
    def __init__(self, num_classes=3, gamma=2.0, alpha=None):
        super().__init__()
        self.num_classes = num_classes; self.gamma = gamma; self.alpha = alpha
    def forward(self, logits, targets):
        ce_loss = F.cross_entropy(logits, targets, reduction='none')
        probs = F.softmax(logits, dim=-1)
        pt = probs.gather(1, targets.unsqueeze(1)).squeeze()
        focal_weight = (1 - pt) ** self.gamma
        pred_class = logits.argmax(dim=1)
        distance = torch.abs(targets - pred_class).float()
        ordinal_weight = 1.0 + distance
        class_weight = self.alpha[targets] if self.alpha is not None else 1.0
        loss = focal_weight * ordinal_weight * class_weight * ce_loss
        return loss.mean()

def create_soft_labels(targets, num_classes=3, smoothing=0.2):
    batch_size = targets.size(0)
    soft_labels = torch.zeros(batch_size, num_classes).to(targets.device)
    for i in range(batch_size):
        label = targets[i].item()
        soft_labels[i, label] = 1.0 - smoothing
        if label > 0:
            soft_labels[i, label - 1] = smoothing / 2
        if label < num_classes - 1:
            soft_labels[i, label + 1] = smoothing / 2
    return soft_labels

# =========================================================
# 8. 학습 설정
# =========================================================
EPOCHS = 50
SAVE_DIR = "/content/drive/MyDrive/urgency_checkpoints_final"
os.makedirs(SAVE_DIR, exist_ok=True)

SAVE_PATH = os.path.join(SAVE_DIR, "best_model_final.pt")

model = CrossAttentionFusionOrdinal(audio_dim=Xa.shape[1]).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-5, weight_decay=0.01)
kl_criterion = nn.KLDivLoss(reduction='batchmean')
ordinal_criterion = OrdinalFocalLoss(num_classes=3, gamma=2.0, alpha=weights)
ce_criterion = nn.CrossEntropyLoss(weight=weights)

from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)

best_f1 = 0
patience, patience_limit = 0, 10
best_state = None

# =========================================================
# 9. 학습 루프
# =========================================================
for epoch in range(1, EPOCHS+1):
    model.train()
    for a, t, yb in train_loader:
        a, t, yb = a.to(device), t.to(device), yb.to(device)
        optimizer.zero_grad()
        logits = model(a, t)

        rand_val = np.random.rand()
        if rand_val < 0.25:
            soft_labels = create_soft_labels(yb, smoothing=0.2)
            loss = kl_criterion(F.log_softmax(logits, dim=-1), soft_labels)
        elif rand_val < 0.65:
            loss = ordinal_criterion(logits, yb)
        else:
            loss = ce_criterion(logits, yb)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

    scheduler.step()

    # ===== Validation =====
    model.eval(); preds, trues = [], []
    with torch.no_grad():
        for a, t, yb in val_loader:
            a, t, yb = a.to(device), t.to(device), yb.to(device)
            out = model(a, t)
            preds += out.argmax(1).cpu().tolist()
            trues += yb.cpu().tolist()

    f1 = f1_score(trues, preds, average="macro")
    acc = accuracy_score(trues, preds)
    print(f"Epoch {epoch:02d}/{EPOCHS} | Val F1: {f1:.4f} | Acc: {acc:.4f}", end=" ")

    if f1 > best_f1:
        best_f1 = f1
        best_state = copy.deepcopy(model.state_dict())
        patience = 0
        print("🔥 Best Updated")
    else:
        patience += 1
        print(f"({patience}/{patience_limit})")

        if patience >= patience_limit:
            print("⏹️ Early stopping!")
            break

# =========================================================
# 10. 전체 모델 저장
# =========================================================
if best_state is not None:
    # best_state 로 모델 갱신
    model.load_state_dict(best_state)

    full_save_path = os.path.join(SAVE_DIR, "best_model_full.pt")
    torch.save(model, full_save_path)
    print(f"\n✅ 전체 모델 저장 완료: {full_save_path}")
