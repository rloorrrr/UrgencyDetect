# infer_emergency.py
# -*- coding: utf-8 -*-

import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import numpy as np
from transformers import AutoTokenizer, AutoModel, Wav2Vec2Processor, Wav2Vec2Model

# ==============================
# 1️⃣ 설정
# ==============================
SAVE_PATH = "./checkpoints/YOUR_PT"
TARGET_SAMPLE_RATE = 16000
MAX_TEXT_LENGTH = 256
MAX_AUDIO_LENGTH = TARGET_SAMPLE_RATE * 25  # 25초 제한
HIDDEN_DIM = 512
DROPOUT = 0.3

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"✅ Using device: {device}")

# ✅ 오디오 & JSON 경로 지정
CUSTOM_AUDIO_PATH = "YOUR_WAV"
CUSTOM_JSON_PATH = "YOUR_JSON"

# ==============================
# 2️⃣ 모델 정의 (학습 코드와 동일)
# ==============================
class EmergencyClassifier(nn.Module):
    def __init__(self, text_encoder, audio_encoder, hidden_dim=HIDDEN_DIM, dropout=DROPOUT, num_classes=3):
        super().__init__()
        self.text_encoder = text_encoder
        self.audio_encoder = audio_encoder

        # 파라미터 동결 정책은 인퍼런스에서는 불필요하지만 유지
        for param in self.text_encoder.parameters():
            param.requires_grad = False
        for param in self.audio_encoder.parameters():
            param.requires_grad = False

        text_dim = text_encoder.config.hidden_size  # 768
        audio_dim = audio_encoder.config.hidden_size  # 1024

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

        self.attention = nn.MultiheadAttention(hidden_dim, num_heads=8, dropout=dropout, batch_first=True)

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
        text_emb = text_output.last_hidden_state[:, 0, :]  # [CLS]

        # 오디오 인코딩
        audio_output = self.audio_encoder(audio_input)
        audio_emb = torch.mean(audio_output.last_hidden_state, dim=1)

        # 임베딩 투영
        text_feat = self.text_proj(text_emb)
        audio_feat = self.audio_proj(audio_emb)

        combined = torch.cat([text_feat, audio_feat], dim=1)
        logits = self.classifier(combined)
        return logits


# ==============================
# 3️⃣ 모델 및 토크나이저 로드
# ==============================
print("🔹 모델 및 토크나이저 로드 중...")

koelectra_model_name = "monologg/koelectra-base-v3-discriminator"
wav2vec2_model_name = "kresnik/wav2vec2-large-xlsr-korean"

tokenizer = AutoTokenizer.from_pretrained(koelectra_model_name)
audio_processor = Wav2Vec2Processor.from_pretrained(wav2vec2_model_name)
text_encoder = AutoModel.from_pretrained(koelectra_model_name)
audio_encoder = Wav2Vec2Model.from_pretrained(wav2vec2_model_name)

model = EmergencyClassifier(text_encoder, audio_encoder).to(device)

checkpoint = torch.load(SAVE_PATH, map_location=device)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

print("✅ 모델 로드 완료!")


# ==============================
# 4️⃣ 예측 함수
# ==============================
def predict_from_json_and_audio(json_path, audio_path):
    # JSON 로드
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # 텍스트 추출
    utterances = [u["text"] for u in data["utterances"] if "text" in u]
    full_text = " ".join(utterances).strip()
    if not full_text:
        full_text = " "

    # 오디오 로드
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    waveform, sr = torchaudio.load(audio_path)

    # 리샘플링
    if sr != TARGET_SAMPLE_RATE:
        waveform = torchaudio.transforms.Resample(sr, TARGET_SAMPLE_RATE)(waveform)

    # 모노 변환
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)
    waveform = waveform.squeeze(0)

    # 길이 조정
    if waveform.shape[0] > MAX_AUDIO_LENGTH:
        waveform = waveform[:MAX_AUDIO_LENGTH]
    elif waveform.shape[0] < TARGET_SAMPLE_RATE:
        pad = TARGET_SAMPLE_RATE - waveform.shape[0]
        waveform = F.pad(waveform, (0, pad))

    # 토큰화
    text_inputs = tokenizer(
        full_text,
        max_length=MAX_TEXT_LENGTH,
        padding="max_length",
        truncation=True,
        return_tensors="pt"
    )

    audio_inputs = audio_processor(
        waveform.numpy(),
        sampling_rate=TARGET_SAMPLE_RATE,
        return_tensors="pt",
        padding=True,
        max_length=MAX_AUDIO_LENGTH,
        truncation=True
    )

    with torch.no_grad():
        input_ids = text_inputs["input_ids"].to(device)
        attention_mask = text_inputs["attention_mask"].to(device)
        audio_input = audio_inputs["input_values"].to(device)

        logits = model(input_ids, attention_mask, audio_input)
        probs = torch.softmax(logits, dim=1)
        pred_label = torch.argmax(probs, dim=1).item()

    label_map = {0: "하", 1: "중", 2: "상"}

    print(f"\n🗣️ 텍스트 예시: {full_text[:100]}...")
    print(f"🎧 오디오 파일: {audio_path}")
    print(f"✅ 예측된 긴급도: {label_map[pred_label]}")
    print(f"📊 확률분포: {probs.squeeze().cpu().numpy()}")

    return label_map[pred_label]


# ==============================
# 5️⃣ 실행
# ==============================
if __name__ == "__main__":
    result = predict_from_json_and_audio(CUSTOM_JSON_PATH, CUSTOM_AUDIO_PATH)
    print(f"\n최종 예측 결과: {result}")
