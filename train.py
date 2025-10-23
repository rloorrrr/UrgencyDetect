import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
from transformers import ElectraTokenizer, ElectraModel
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.utils.class_weight import compute_class_weight
from tqdm.auto import tqdm
import numpy as np
import pandas as pd

# -----------------------------
# 0️⃣ GPU 세팅
# -----------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"

# -----------------------------
# 1️⃣ 데이터 로드
# -----------------------------
embedding_file = "./wav2vec2_embeddings_full.npy"
csv_file = "./광주_구급_full_final.csv"

df = pd.read_csv(csv_file)
embeddings = np.load(embedding_file, allow_pickle=True)
df["wav2vec2_embedding"] = list(embeddings)

# -----------------------------
# 2️⃣ Label Encoding
# -----------------------------
le_sent = LabelEncoder()
le_urg = LabelEncoder()
df['sentiment_label'] = le_sent.fit_transform(df['sentiment'])
df['urgency_label'] = le_urg.fit_transform(df['urgencyLevel'])

# -----------------------------
# 2-1️⃣ Class Imbalance 처리 (Class Weights 계산)
# -----------------------------
sent_weights = compute_class_weight(
    'balanced',
    classes=np.unique(df['sentiment_label']),
    y=df['sentiment_label']
)
urg_weights = compute_class_weight(
    'balanced',
    classes=np.unique(df['urgency_label']),
    y=df['urgency_label']
)

sent_weights = torch.FloatTensor(sent_weights).to(device)
urg_weights = torch.FloatTensor(urg_weights).to(device)

print(f"Sentiment class weights: {sent_weights}")
print(f"Urgency class weights: {urg_weights}")

# -----------------------------
# 3️⃣ Dataset 정의
# -----------------------------
class MultiModalDataset(Dataset):
    def __init__(self, df, tokenizer, max_len=512):
        self.texts = df['text'].tolist()
        self.embeddings = np.stack(df['wav2vec2_embedding'].to_numpy()).astype(np.float32)
        self.sentiments = df['sentiment_label'].tolist()
        self.urgencies = df['urgency_label'].tolist()
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            truncation=True,
            padding='max_length',
            max_length=self.max_len,
            return_tensors='pt'
        )
        wav_emb = torch.from_numpy(self.embeddings[idx]).float()
        return {
            "input_ids": encoding['input_ids'].squeeze(0),
            "attention_mask": encoding['attention_mask'].squeeze(0),
            "wav_emb": wav_emb,
            "sentiment": torch.tensor(self.sentiments[idx], dtype=torch.long),
            "urgency": torch.tensor(self.urgencies[idx], dtype=torch.long)
        }

# -----------------------------
# 4️⃣ Train / Val split
# -----------------------------
train_df, val_df = train_test_split(
    df, test_size=0.15, random_state=42, stratify=df['urgencyLevel']
)

print(f"Train size: {len(train_df)}, Val size: {len(val_df)}")

# -----------------------------
# 5️⃣ Tokenizer & DataLoader
# -----------------------------
MODEL_NAME = "monologg/koelectra-base-v3-discriminator"
tokenizer = ElectraTokenizer.from_pretrained(MODEL_NAME)

train_dataset = MultiModalDataset(train_df, tokenizer)
val_dataset   = MultiModalDataset(val_df, tokenizer)

BATCH_SIZE = 16
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True)
val_loader   = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)

# -----------------------------
# 6️⃣ Cross-Attention 모듈
# -----------------------------
class CrossAttention(nn.Module):
    def __init__(self, hidden_dim, num_heads=4, dropout=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, query, key, value):
        attn_output, _ = self.attn(query=query, key=key, value=value)
        out = self.norm(query + self.dropout(attn_output))
        return out

# -----------------------------
# 7️⃣ 멀티모달 모델 정의 (개선버전)
# -----------------------------
class KoElectraCrossModal(nn.Module):
    def __init__(self, model_name, wav_emb_size, num_sent, num_urg, hidden_dim=768, dropout=0.1):
        super().__init__()
        self.text_encoder = ElectraModel.from_pretrained(model_name)
        self.wav_fc = nn.Linear(wav_emb_size, hidden_dim)
        self.cross_attn = CrossAttention(hidden_dim, num_heads=4, dropout=dropout)
        self.dropout = nn.Dropout(dropout)
        self.sent_head = nn.Linear(hidden_dim, num_sent)
        self.urg_head  = nn.Linear(hidden_dim, num_urg)

    def forward(self, input_ids, attention_mask, wav_emb):
        # Text embedding
        text_out = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls_emb = text_out.last_hidden_state[:, 0:1, :]  # [B,1,H]

        # Wav embedding
        wav_emb = torch.relu(self.wav_fc(wav_emb)).unsqueeze(1)  # [B,1,H]

        # Cross-attention
        cross_out = self.cross_attn(cls_emb, wav_emb, wav_emb).squeeze(1)
        cross_out = self.dropout(cross_out)

        # Multi-task heads
        sent_logits = self.sent_head(cross_out)
        urg_logits  = self.urg_head(cross_out)
        return sent_logits, urg_logits

    def multitask_loss(self, sent_logits, urg_logits, sent_labels, urg_labels,
                       sent_weights, urg_weights):
        """
        개선된 Multi-task Loss
        - Uncertainty loss 제거
        - Class imbalance를 위한 weighted loss 적용
        - Task별 가중치 적용 (urgency 더 중요)
        """
        criterion_sent = nn.CrossEntropyLoss(weight=sent_weights)
        criterion_urg = nn.CrossEntropyLoss(weight=urg_weights)

        loss_sent = criterion_sent(sent_logits, sent_labels)
        loss_urg = criterion_urg(urg_logits, urg_labels)

        # Weighted sum (urgency가 더 중요하므로 0.6 가중치)
        alpha = 0.4  # sentiment
        beta = 0.6   # urgency
        total_loss = alpha * loss_sent + beta * loss_urg

        return total_loss, loss_sent.item(), loss_urg.item()

# -----------------------------
# 8️⃣ 모델, optimizer, scheduler 정의
# -----------------------------
wav_emb_size = df['wav2vec2_embedding'][0].shape[0]
num_sent = len(le_sent.classes_)
num_urg  = len(le_urg.classes_)

model = KoElectraCrossModal(MODEL_NAME, wav_emb_size, num_sent, num_urg).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)

# Learning Rate Scheduler 추가
EPOCHS = 5
scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)

print(f"\nModel parameters: {sum(p.numel() for p in model.parameters()):,}")
print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

# -----------------------------
# 9️⃣ 평가 함수
# -----------------------------
def evaluate(model, dataloader, device):
    model.eval()
    preds_sent, trues_sent = [], []
    preds_urg, trues_urg = [], []

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            wav_emb = batch['wav_emb'].to(device)
            sentiment = batch['sentiment'].to(device)
            urgency = batch['urgency'].to(device)

            sent_logits, urg_logits = model(input_ids, attention_mask, wav_emb)
            preds_sent.extend(torch.argmax(sent_logits, dim=1).cpu().numpy())
            trues_sent.extend(sentiment.cpu().numpy())
            preds_urg.extend(torch.argmax(urg_logits, dim=1).cpu().numpy())
            trues_urg.extend(urgency.cpu().numpy())

    sent_acc = accuracy_score(trues_sent, preds_sent)
    urg_acc  = accuracy_score(trues_urg, preds_urg)
    sent_f1 = f1_score(trues_sent, preds_sent, average='macro')
    urg_f1  = f1_score(trues_urg, preds_urg, average='macro')

    return {
        "sent_acc": sent_acc, "sent_f1": sent_f1,
        "urg_acc": urg_acc, "urg_f1": urg_f1,
        "preds_sent": preds_sent, "trues_sent": trues_sent,
        "preds_urg": preds_urg, "trues_urg": trues_urg
    }

# -----------------------------
# 10️⃣ 학습 루프 (개선버전)
# -----------------------------
best_val_f1 = 0.0
save_dir = "YOUR_SAVE_DIR"
os.makedirs(save_dir, exist_ok=True)

for epoch in range(1, EPOCHS+1):
    model.train()
    total_loss = 0
    total_sent_loss = 0
    total_urg_loss = 0

    pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS}")
    for batch in pbar:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        wav_emb = batch['wav_emb'].to(device)
        sentiment = batch['sentiment'].to(device)
        urgency = batch['urgency'].to(device)

        optimizer.zero_grad()
        sent_logits, urg_logits = model(input_ids, attention_mask, wav_emb)

        # Loss 계산 (class weights 적용)
        loss, sent_loss, urg_loss = model.multitask_loss(
            sent_logits, urg_logits, sentiment, urgency,
            sent_weights, urg_weights
        )

        loss.backward()

        # Gradient Clipping 추가
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        total_loss += loss.item()
        total_sent_loss += sent_loss
        total_urg_loss += urg_loss

        pbar.set_postfix({
            "loss": f"{total_loss/(pbar.n+1):.4f}",
            "sent": f"{total_sent_loss/(pbar.n+1):.4f}",
            "urg": f"{total_urg_loss/(pbar.n+1):.4f}",
            "lr": f"{scheduler.get_last_lr()[0]:.2e}"
        })

    # Learning Rate Scheduler Step
    scheduler.step()

    # Validation
    val_metrics = evaluate(model, val_loader, device)
    avg_val_f1 = (val_metrics['sent_f1'] + val_metrics['urg_f1']) / 2

    print(f"\n{'='*60}")
    print(f"Epoch {epoch}/{EPOCHS} Summary")
    print(f"{'='*60}")
    print(f"Train Loss: {total_loss/len(train_loader):.4f} | "
          f"Sent Loss: {total_sent_loss/len(train_loader):.4f} | "
          f"Urg Loss: {total_urg_loss/len(train_loader):.4f}")
    print(f"Val Sent | Acc: {val_metrics['sent_acc']:.4f} | F1: {val_metrics['sent_f1']:.4f}")
    print(f"Val Urg  | Acc: {val_metrics['urg_acc']:.4f} | F1: {val_metrics['urg_f1']:.4f}")
    print(f"Avg Val F1: {avg_val_f1:.4f} | LR: {scheduler.get_last_lr()[0]:.2e}")

    if avg_val_f1 > best_val_f1:
        best_val_f1 = avg_val_f1
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'best_val_f1': best_val_f1,
            'val_metrics': val_metrics
        }, os.path.join(save_dir, "best_model.pt"))
        print("✅ Best model saved!")
    print(f"{'='*60}\n")

# -----------------------------
# 11️⃣ 최종 평가
# -----------------------------
# Best model 로드
checkpoint = torch.load(os.path.join(save_dir, "best_model.pt"))
model.load_state_dict(checkpoint['model_state_dict'])
print(f"\n🔥 Loaded best model from epoch {checkpoint['epoch']} (F1: {checkpoint['best_val_f1']:.4f})")

res = evaluate(model, val_loader, device)
print("\n" + "="*60)
print("=== FINAL VALIDATION REPORT ===")
print("="*60)
print(f"Sentiment | Acc: {res['sent_acc']:.4f} | Macro-F1: {res['sent_f1']:.4f}")
print(f"Urgency   | Acc: {res['urg_acc']:.4f} | Macro-F1: {res['urg_f1']:.4f}")
print(f"Average F1: {(res['sent_f1'] + res['urg_f1'])/2:.4f}")
print("="*60 + "\n")

print("📊 Sentiment Classification Report:")
print(classification_report(res['trues_sent'], res['preds_sent'],
                          target_names=le_sent.classes_, zero_division=0))

print("\n📊 Urgency Classification Report:")
print(classification_report(res['trues_urg'], res['preds_urg'],
                          target_names=le_urg.classes_, zero_division=0))