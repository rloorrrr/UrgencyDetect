import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, f1_score
from tqdm import tqdm
import random
from transformers import AutoTokenizer, AutoModel, Wav2Vec2Processor, Wav2Vec2Model
import torchaudio
import warnings
warnings.filterwarnings('ignore')

# ==============================
# 1️⃣ 설정
# ==============================
DATA_CSV = "./processed_data.csv"
BATCH_SIZE = 2  # 메모리 고려해서 작게 설정
EPOCHS = 5
LR = 1e-5
SEED = 42
SAVE_PATH = "./checkpoints/emergency_model.pth"
HIDDEN_DIM = 512
DROPOUT = 0.3
MAX_TEXT_LENGTH = 256
TARGET_SAMPLE_RATE = 16000
MAX_AUDIO_LENGTH = 16000 * 25  # 최대 30초

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)

# 재현성
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

print(f"Using device: {device}")

# ==============================
# 2️⃣ 데이터 로드
# ==============================
print("Loading data...")
df = pd.read_csv(DATA_CSV)
print(f"Total samples: {len(df)}")
print("\nUrgency Level distribution:")
print(df['urgencyLevel'].value_counts())

# ==============================
# 3️⃣ 모델 및 프로세서 로드
# ==============================
print("\nLoading KoELECTRA and Wav2Vec2 models...")

# KoELECTRA
koelectra_model_name = "monologg/koelectra-base-v3-discriminator"
tokenizer = AutoTokenizer.from_pretrained(koelectra_model_name)
text_encoder = AutoModel.from_pretrained(koelectra_model_name)

# Wav2Vec2 (한국어 특화)
wav2vec2_model_name = "kresnik/wav2vec2-large-xlsr-korean"
audio_processor = Wav2Vec2Processor.from_pretrained(wav2vec2_model_name)
audio_encoder = Wav2Vec2Model.from_pretrained(wav2vec2_model_name)

print("Models loaded successfully!")

# ==============================
# 4️⃣ Dataset 정의
# ==============================
class EmergencyDataset(Dataset):
    def __init__(self, df, tokenizer, audio_processor):
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.audio_processor = audio_processor
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        # 텍스트 처리
        text = str(row['text'])
        text_encoded = self.tokenizer(
            text,
            max_length=MAX_TEXT_LENGTH,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # 오디오 처리
        audio_path = row['audio_path']
        
        try:
            waveform, sample_rate = torchaudio.load(audio_path)
            
            # 리샘플링
            if sample_rate != TARGET_SAMPLE_RATE:
                resampler = torchaudio.transforms.Resample(sample_rate, TARGET_SAMPLE_RATE)
                waveform = resampler(waveform)
            
            # 모노로 변환
            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)
            
            waveform = waveform.squeeze(0)
            
            # 길이 조절 (너무 길면 자르고, 짧으면 패딩)
            if waveform.shape[0] > MAX_AUDIO_LENGTH:
                waveform = waveform[:MAX_AUDIO_LENGTH]
            elif waveform.shape[0] < TARGET_SAMPLE_RATE:
                # 너무 짧은 오디오는 패딩
                padding = TARGET_SAMPLE_RATE - waveform.shape[0]
                waveform = torch.nn.functional.pad(waveform, (0, padding))
            
            # Wav2Vec2 입력 형식으로 변환
            audio_input = self.audio_processor(
                waveform.numpy(),
                sampling_rate=TARGET_SAMPLE_RATE,
                return_tensors="pt",
                padding=True,
                max_length=MAX_AUDIO_LENGTH,
                truncation=True
            )
            
            audio_values = audio_input['input_values'].squeeze(0)
            
        except Exception as e:
            print(f"Error loading audio {audio_path}: {e}")
            # 오류 시 무음으로 대체
            audio_values = torch.zeros(TARGET_SAMPLE_RATE)
        
        label = int(row['urgency_label'])
        
        return {
            "input_ids": text_encoded['input_ids'].squeeze(0),
            "attention_mask": text_encoded['attention_mask'].squeeze(0),
            "audio_input": audio_values,
            "label": torch.tensor(label, dtype=torch.long)
        }

# ==============================
# 5️⃣ Train/Val/Test Split
# ==============================
print("\nSplitting data...")
train_df, temp_df = train_test_split(
    df, test_size=0.3, random_state=SEED, stratify=df['urgency_label']
)
val_df, test_df = train_test_split(
    temp_df, test_size=0.5, random_state=SEED, stratify=temp_df['urgency_label']
)

print(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

train_dataset = EmergencyDataset(train_df, tokenizer, audio_processor)
val_dataset = EmergencyDataset(val_df, tokenizer, audio_processor)
test_dataset = EmergencyDataset(test_df, tokenizer, audio_processor)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

# ==============================
# 6️⃣ 멀티모달 분류 모델 정의
# ==============================
class EmergencyClassifier(nn.Module):
    def __init__(self, text_encoder, audio_encoder, hidden_dim=HIDDEN_DIM, dropout=DROPOUT, num_classes=3):
        super().__init__()
        self.text_encoder = text_encoder
        self.audio_encoder = audio_encoder
        
        # 인코더 파인튜닝 활성화 (마지막 몇 레이어만)
        # 텍스트 인코더의 마지막 2개 레이어만 학습
        for param in self.text_encoder.parameters():
            param.requires_grad = False
        for param in self.text_encoder.encoder.layer[-2:].parameters():
            param.requires_grad = True
        
        # 오디오 인코더의 마지막 레이어만 학습
        for param in self.audio_encoder.parameters():
            param.requires_grad = False
        for param in self.audio_encoder.encoder.layers[-2:].parameters():
            param.requires_grad = True
        
        # 차원 정보
        text_dim = text_encoder.config.hidden_size  # 768
        audio_dim = audio_encoder.config.hidden_size  # 1024
        
        # 프로젝션 레이어
        self.text_proj = nn.Sequential(
            nn.Linear(text_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        self.audio_proj = nn.Sequential(
            nn.Linear(audio_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Attention 메커니즘 (간단한 버전)
        self.attention = nn.MultiheadAttention(hidden_dim, num_heads=8, dropout=dropout, batch_first=True)
        
        # 최종 분류기
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes)
        )
    
    def forward(self, input_ids, attention_mask, audio_input):
        # 텍스트 인코딩
        text_output = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)
        text_emb = text_output.last_hidden_state[:, 0, :]  # [CLS] token
        
        # 오디오 인코딩
        audio_output = self.audio_encoder(audio_input)
        audio_emb = torch.mean(audio_output.last_hidden_state, dim=1)  # 평균 풀링
        
        # 프로젝션
        text_feat = self.text_proj(text_emb)  # [batch, hidden_dim]
        audio_feat = self.audio_proj(audio_emb)  # [batch, hidden_dim]
        
        # Cross-modal attention (optional, 간단하게 concat으로도 가능)
        combined = torch.cat([text_feat, audio_feat], dim=1)  # [batch, hidden_dim*2]
        
        # 분류
        logits = self.classifier(combined)
        
        return logits

model = EmergencyClassifier(text_encoder, audio_encoder).to(device)

# 클래스 가중치 계산 (불균형 데이터 대응)
class_counts = train_df['urgency_label'].value_counts().sort_index()
class_weights = 1.0 / torch.tensor(class_counts.values, dtype=torch.float)
class_weights = class_weights / class_weights.sum() * len(class_counts)
class_weights = class_weights.to(device)

criterion = nn.CrossEntropyLoss(weight=class_weights)
optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)

# Learning rate scheduler
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='max', factor=0.5, patience=2, verbose=True
)

# ==============================
# 7️⃣ 학습 루프
# ==============================
best_val_f1 = 0.0

print("\n" + "="*50)
print("Starting training...")
print("="*50)

for epoch in range(1, EPOCHS + 1):
    # Training
    model.train()
    train_losses, train_preds, train_labels = [], [], []
    
    for batch in tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS} Train"):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        audio_input = batch["audio_input"].to(device)
        labels_batch = batch["label"].to(device)
        
        optimizer.zero_grad()
        logits = model(input_ids, attention_mask, audio_input)
        loss = criterion(logits, labels_batch)
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        train_losses.append(loss.item())
        preds = torch.argmax(logits, dim=1).cpu().numpy()
        train_preds.extend(preds)
        train_labels.extend(labels_batch.cpu().numpy())
    
    train_acc = accuracy_score(train_labels, train_preds)
    train_f1 = f1_score(train_labels, train_preds, average='weighted')
    avg_train_loss = np.mean(train_losses)
    
    # Validation
    model.eval()
    val_preds, val_labels_list = [], []
    val_losses = []
    
    with torch.no_grad():
        for batch in tqdm(val_loader, desc=f"Epoch {epoch}/{EPOCHS} Val"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            audio_input = batch["audio_input"].to(device)
            labels_batch = batch["label"].to(device)
            
            logits = model(input_ids, attention_mask, audio_input)
            loss = criterion(logits, labels_batch)
            val_losses.append(loss.item())
            
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            val_preds.extend(preds)
            val_labels_list.extend(labels_batch.cpu().numpy())
    
    val_acc = accuracy_score(val_labels_list, val_preds)
    val_f1 = f1_score(val_labels_list, val_preds, average='weighted')
    avg_val_loss = np.mean(val_losses)
    
    print(f"\nEpoch {epoch}/{EPOCHS}")
    print(f"Train Loss: {avg_train_loss:.4f} | Train Acc: {train_acc:.4f} | Train F1: {train_f1:.4f}")
    print(f"Val Loss: {avg_val_loss:.4f} | Val Acc: {val_acc:.4f} | Val F1: {val_f1:.4f}")
    
    # Learning rate scheduling
    scheduler.step(val_f1)
    
    # 모델 저장 (F1 score 기준)
    if val_f1 > best_val_f1:
        best_val_f1 = val_f1
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_f1': val_f1,
            'val_acc': val_acc
        }, SAVE_PATH)
        print(f"✅ New best model saved (val_f1={val_f1:.4f})")

# ==============================
# 8️⃣ 테스트 평가
# ==============================
print("\n" + "="*50)
print("Loading best model for testing...")
print("="*50)

checkpoint = torch.load(SAVE_PATH)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

test_preds, test_labels_list = [], []

with torch.no_grad():
    for batch in tqdm(test_loader, desc="Testing"):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        audio_input = batch["audio_input"].to(device)
        labels_batch = batch["label"].to(device)
        
        logits = model(input_ids, attention_mask, audio_input)
        preds = torch.argmax(logits, dim=1).cpu().numpy()
        test_preds.extend(preds)
        test_labels_list.extend(labels_batch.cpu().numpy())

test_acc = accuracy_score(test_labels_list, test_preds)
test_f1 = f1_score(test_labels_list, test_preds, average='weighted')

print("\n" + "="*50)
print("=== Test Results ===")
print("="*50)
print(f"Test Accuracy: {test_acc:.4f}")
print(f"Test F1 Score: {test_f1:.4f}")
print("\nDetailed Classification Report:")
print(classification_report(
    test_labels_list, 
    test_preds, 
    target_names=['하', '중', '상'],
    digits=4
))