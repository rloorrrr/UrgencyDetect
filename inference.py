# =========================================================
# inference.py — Whisper + KoELECTRA 긴급도 예측 (JSON + WAV 입력)
# =========================================================
# -*- coding: utf-8 -*-

import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import librosa
import numpy as np
from transformers import WhisperFeatureExtractor, WhisperModel, AutoTokenizer, AutoModel

# ==============================
# 1️⃣ 설정
# ==============================
SAVE_PATH = "/content/drive/MyDrive/test_data/model/1500_model.pt"
AUDIO_SR = 16000
HIDDEN_DIM = 512
DROPOUT = 0.3
NUM_CLASSES = 3

WHISPER_NAME = "openai/whisper-small"
TEXT_NAME = "monologg/koelectra-base-v3-discriminator"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"✅ Using device: {device}")

# JSON + WAV 파일 경로
CUSTOM_JSON_PATH = "/content/drive/MyDrive/data/Training/2.라벨링데이터/TL_광주_구급/6509537abc5846983ad2efb3_20230523072543.json"
CUSTOM_AUDIO_PATH = "/content/drive/MyDrive/data/Training/1.원천데이터/TS_광주_구급/6509537abc5846983ad2efb3_20230523072543.wav"

# ==============================
# 2️⃣ 모델 정의 (학습 코드와 동일)
# ==============================
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
    def __init__(self, hdim=HIDDEN_DIM, heads=8, dropout=DROPOUT):
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

class MultiModalClassifier(nn.Module):
    def __init__(self, whisper_name, text_name):
        super().__init__()
        whisper = WhisperModel.from_pretrained(whisper_name).encoder
        text = AutoModel.from_pretrained(text_name)
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

# ==============================
# 3️⃣ 모델 및 토크나이저 로드
# ==============================
print("🔹 모델 및 토크나이저 로드 중...")

feature_extractor = WhisperFeatureExtractor.from_pretrained(WHISPER_NAME)
tokenizer = AutoTokenizer.from_pretrained(TEXT_NAME)

model = MultiModalClassifier(WHISPER_NAME, TEXT_NAME).to(device)
checkpoint = torch.load(SAVE_PATH, map_location=device)
model.load_state_dict(checkpoint)
model.eval()
print("✅ 모델 로드 완료!")

# ==============================
# 4️⃣ JSON + WAV 예측 함수
# ==============================
def load_text_from_json(json_path):
    """JSON에서 발화 텍스트 전부 합치기"""
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"❌ JSON 파일을 찾을 수 없습니다: {json_path}")

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if "utterances" not in data:
        raise ValueError("❌ JSON에 'utterances' 키가 없습니다.")

    utterances = [u.get("text", "") for u in data["utterances"] if isinstance(u, dict)]
    full_text = " ".join(utterances).strip()
    if not full_text:
        full_text = " "  # 비어 있으면 공백이라도 전달
    return full_text

def predict_from_json_and_audio(json_path, audio_path):
    # 1. JSON → 텍스트 변환
    text = load_text_from_json(json_path)

    # 2. 오디오 로드
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"❌ 오디오 파일을 찾을 수 없습니다: {audio_path}")

    y, _ = librosa.load(audio_path, sr=AUDIO_SR, mono=True)

    # 3. Feature 변환
    audio_inputs = feature_extractor(y, sampling_rate=AUDIO_SR, return_tensors="pt")
    text_inputs = tokenizer(text, truncation=True, padding="max_length", max_length=128, return_tensors="pt")

    a = {k:v.to(device) for k,v in audio_inputs.items()}
    t = {k:v.to(device) for k,v in text_inputs.items()}

    # 4. 추론
    with torch.no_grad():
        logits = model(a, t)
        probs = torch.softmax(logits, dim=1)[0].cpu().numpy()
        pred_label = int(np.argmax(probs))

    id2label = {0:"하", 1:"중", 2:"상"}

    print(f"\n🗣️ JSON 텍스트 예시: {text[:100]}...")
    print(f"🎧 오디오 파일: {os.path.basename(audio_path)}")
    print(f"✅ 예측된 긴급도: {id2label[pred_label]}")
    print(f"📊 확률분포: {probs}")

    return {"pred_label": id2label[pred_label], "probabilities": probs.tolist()}

# ==============================
# 5️⃣ 실행
# ==============================
if __name__ == "__main__":
    result = predict_from_json_and_audio(CUSTOM_JSON_PATH, CUSTOM_AUDIO_PATH)
    print(f"\n최종 예측 결과: {result}")