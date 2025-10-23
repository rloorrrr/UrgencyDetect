# infer_emergency_crossmodal.py
# -*- coding: utf-8 -*-

import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import librosa
import numpy as np
from transformers import ElectraTokenizer, ElectraModel, Wav2Vec2Processor, Wav2Vec2Model

# ==============================
# 1️⃣ 설정
# ==============================
# CHECKPOINT_PATH = "./checkpoints/best_model.pt"  # 학습된 KoElectra+Wav2Vec2 모델
CHECKPOINT_PATH = "./checkpoints/koelectra_multi_ds_model.pt"


TARGET_SR = 16000
MAX_TEXT_LEN = 512

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"✅ Using device: {device}")

# ==============================
# 2️⃣ CrossAttention + 모델 정의
# ==============================
class CrossAttention(nn.Module):
    def __init__(self, hidden_dim, num_heads=4, dropout=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, query, key, value):
        attn_output, _ = self.attn(query=query, key=key, value=value)
        return self.norm(query + self.dropout(attn_output))


class KoElectraCrossModal(nn.Module):
    def __init__(self, model_name, wav_emb_size, num_sent, num_urg, hidden_dim=768, dropout=0.1):
        super().__init__()
        self.text_encoder = ElectraModel.from_pretrained(model_name)
        self.wav_fc = nn.Linear(wav_emb_size, hidden_dim)
        self.cross_attn = CrossAttention(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.sent_head = nn.Linear(hidden_dim, num_sent)
        self.urg_head  = nn.Linear(hidden_dim, num_urg)

    def forward(self, input_ids, attention_mask, wav_emb):
        text_out = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls_emb = text_out.last_hidden_state[:, 0:1, :]
        wav_emb = torch.relu(self.wav_fc(wav_emb)).unsqueeze(1)
        cross_out = self.cross_attn(cls_emb, wav_emb, wav_emb).squeeze(1)
        cross_out = self.dropout(cross_out)
        sent_logits = self.sent_head(cross_out)
        urg_logits  = self.urg_head(cross_out)
        return sent_logits, urg_logits

# ==============================
# 3️⃣ 토크나이저 + Wav2Vec2 processor
# ==============================
ELECTRA_MODEL_NAME = "monologg/koelectra-base-v3-discriminator"
W2V_MODEL_NAME     = "kresnik/wav2vec2-large-xlsr-korean"

tokenizer = ElectraTokenizer.from_pretrained(ELECTRA_MODEL_NAME)
wav_processor = Wav2Vec2Processor.from_pretrained(W2V_MODEL_NAME)
wav_model = Wav2Vec2Model.from_pretrained(W2V_MODEL_NAME).to(device)
wav_model.eval()

# ==============================
# 4️⃣ 모델 로드
# ==============================

NUM_SENT = 4
NUM_URG  = 3
WAV_EMB_SIZE = 1024  # wav2vec2-large hidden size

model = KoElectraCrossModal(ELECTRA_MODEL_NAME, WAV_EMB_SIZE, NUM_SENT, NUM_URG).to(device)
checkpoint = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=False)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()
print(f"🔥 Loaded model from {CHECKPOINT_PATH}")

# ==============================
# 5️⃣ 단일 json + wav 예측 함수
# ==============================
def predict_from_json_and_audio(json_path, wav_path, start_ms=None, end_ms=None):
    # 1) JSON 로드
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    utterances = [u["text"] for u in data.get("utterances", []) if "text" in u]
    full_text = " ".join(utterances).strip()
    if not full_text:
        full_text = " "

    # 2) 오디오 로드 및 segment
    waveform, sr = librosa.load(wav_path, sr=TARGET_SR, mono=True)
    if start_ms is not None and end_ms is not None:
        waveform = waveform[int(start_ms/1000*TARGET_SR):int(end_ms/1000*TARGET_SR)]
    input_values = wav_processor(waveform, sampling_rate=TARGET_SR, return_tensors="pt").input_values.to(device)

    # 3) Wav2Vec2 embedding (mean pooling)
    with torch.no_grad():
        wav_emb = wav_model(input_values).last_hidden_state.squeeze(0).mean(dim=0).unsqueeze(0)

    # 4) 텍스트 토큰화
    encoding = tokenizer(full_text, truncation=True, padding="max_length",
                         max_length=MAX_TEXT_LEN, return_tensors="pt")
    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)

    # 5) 모델 forward
    with torch.no_grad():
        sent_logits, urg_logits = model(input_ids, attention_mask, wav_emb)
        pred_sent = torch.argmax(sent_logits, dim=1).item()
        pred_urg  = torch.argmax(urg_logits, dim=1).item()

    return pred_sent, pred_urg

def decode_labels(pred_sent, pred_urg):
    SENTIMENT_MAP = {
        0: "기타부정",
        1: "당황/난처",
        2: "불안/걱정",
        3: "중립"
    }

    URGENCY_MAP = {
        0: "상",
        1: "중",
        2: "하"
    }

    sent_label = SENTIMENT_MAP.get(pred_sent, f"Unknown({pred_sent})")
    urg_label  = URGENCY_MAP.get(pred_urg, f"Unknown({pred_urg})")

    return sent_label, urg_label
# ==============================
# 6️⃣ 실행 예시
# ==============================
if __name__ == "__main__":
    CUSTOM_AUDIO_PATH = "YOUR_WAV"
    CUSTOM_JSON_PATH = "YOUR_JSON"
    pred_sent, pred_urg = predict_from_json_and_audio(CUSTOM_JSON_PATH, CUSTOM_AUDIO_PATH)
    sent_label, urg_label = decode_labels(pred_sent, pred_urg)


    print(f"\n✅ Predicted Sentiment: {sent_label}")
    print(f"✅ Predicted Urgency:   {urg_label}")