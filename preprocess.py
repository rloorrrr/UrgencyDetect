# =========================================================
# 1. 라이브러리 로드 및 경로 설정
# =========================================================
import os, json, glob, re, torch
import pandas as pd
import numpy as np
import librosa
from librosa.util import normalize
from tqdm.auto import tqdm
from transformers import AutoProcessor, AutoModel, AutoTokenizer
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from sklearn.metrics import f1_score, classification_report
from collections import Counter
import random
import copy


device = "cuda" if torch.cuda.is_available() else "cpu"

json_root = "/content/drive/MyDrive/urgency_extracted/TL_광주_구조"
wav_root  = "/content/drive/MyDrive/urgency_extracted/TS_광주_구조"


# =========================================================
# 2. JSON 로드 및 텍스트 전처리
# =========================================================
json_files = glob.glob(os.path.join(json_root, "**/*.json"), recursive=True)
print(f"JSON 파일 수: {len(json_files):,}")

data = []
for f in tqdm(json_files, desc="Loading JSON"):
    try:
        with open(f, "r", encoding="utf-8") as j:
            d = json.load(j)
            d["file_id"] = os.path.splitext(os.path.basename(f))[0]
            data.append(d)
    except Exception as e:
        print(f"⚠️ {f} 읽기 실패: {e}")

df = pd.json_normalize(data)
print(f"총 {len(df):,}행 로드 완료")

urgency_map = {"하": 0, "중": 1, "상": 2}
df["urgency_y"] = df["urgencyLevel"].astype(str).map(urgency_map)

def extract_text(u):
    try:
        if isinstance(u, str):
            u = json.loads(u)
        if isinstance(u, list):
            texts = [x.get("text", "") for x in u if isinstance(x, dict)]
            return " ".join(t for t in texts if t)
    except:
        return ""
    return ""

df["text"] = df["utterances"].apply(extract_text)
df = df[["file_id", "text", "urgency_y"]].dropna(subset=["urgency_y"])
df = df[df["text"].str.len() > 0].reset_index(drop=True)

# =========================================================
# 3. WAV 파일 로드 및 병합 + 오디오 전처리
# =========================================================
wav_files = glob.glob(os.path.join(wav_root, "**/*.wav"), recursive=True)
df_wav = pd.DataFrame({"audio_path": wav_files})
df_wav["file_id"] = df_wav["audio_path"].apply(lambda x: os.path.splitext(os.path.basename(x))[0])

print(f"🎧 WAV 파일 수: {len(df_wav):,}")

# 오디오 기본 피처 추출 함수
def extract_audio_features(path):
    try:
        y, sr = librosa.load(path, sr=16000)
        duration = librosa.get_duration(y=y, sr=sr)

        # 무성구간 제외한 발화 비율
        non_silent = librosa.effects.split(y, top_db=25)
        voiced_time = sum((end - start) for start, end in non_silent) / sr
        speech_ratio = voiced_time / duration if duration > 0 else 0

        # 평균 에너지 (RMS²)
        energy_mean = np.mean(y ** 2)

        # 피크 진폭
        peak_amp = np.max(np.abs(y))

        # RMS (루트 평균 제곱)
        rms = np.sqrt(energy_mean)
        peak_to_rms = peak_amp / rms if rms > 0 else 0

        return pd.Series({
            "duration_sec": duration,
            "speech_ratio": speech_ratio,
            "energy_mean": energy_mean,
            "peak_amp": peak_amp,
            "peak_to_rms": peak_to_rms
        })
    except Exception as e:
        print(f"⚠️ {path} 처리 중 오류: {e}")
        return pd.Series({
            "duration_sec": np.nan,
            "speech_ratio": np.nan,
            "energy_mean": np.nan,
            "peak_amp": np.nan,
            "peak_to_rms": np.nan
        })

# 전체 wav 파일에 적용
df_features = df_wav["audio_path"].apply(extract_audio_features)
df_wav = pd.concat([df_wav, df_features], axis=1)

print("✅ 오디오 전처리 피처 추가 완료")
print(df_wav[["file_id", "duration_sec", "speech_ratio", "energy_mean", "peak_amp", "peak_to_rms"]].head())

#  JSON과 병합
df_all = pd.merge(df, df_wav, on="file_id", how="inner")
print(f"병합 완료: {len(df_all):,}행")

# =========================================================
# 4. 오디오 임베딩 (Data2Vec-Audio)
# =========================================================

AUDIO_MODEL = "facebook/data2vec-audio-base"

from transformers import AutoProcessor, AutoModel
processor = AutoProcessor.from_pretrained(AUDIO_MODEL)
audio_model = AutoModel.from_pretrained(AUDIO_MODEL).to(device).eval()

def extract_audio_emb(path):
    try:
        y, sr = librosa.load(path, sr=16000)
        y = normalize(y)
        inputs = processor(y, sampling_rate=sr, return_tensors="pt", padding=True)
        with torch.no_grad():
            out = audio_model(**{k: v.to(device) for k, v in inputs.items()})
        emb = out.last_hidden_state.mean(dim=1).cpu().numpy().flatten()
        return emb / (np.linalg.norm(emb) + 1e-12)
    except Exception as e:
        print(f"[Audio ERR] {path}: {e}")
        return None

audio_embs = [extract_audio_emb(p) for p in tqdm(df_all["audio_path"], desc="Audio Embedding")]
df_all["audio_emb"] = audio_embs

# =========================================================
# 5. 텍스트 임베딩 생성
# =========================================================


TEXT_MODEL = "jhgan/ko-sroberta-multitask"

# 모델 로드
tokenizer = AutoTokenizer.from_pretrained(TEXT_MODEL)
text_model = AutoModel.from_pretrained(TEXT_MODEL).to(device).eval()

def extract_text_emb(text):
    try:
        if not text or len(text.strip()) == 0:
            return None

        inputs = tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=384,
            return_tensors="pt"
        )

        with torch.no_grad():
            out = text_model(**{k: v.to(device) for k, v in inputs.items()})

        # [CLS] 토큰의 임베딩 추출
        emb = out.last_hidden_state[:, 0, :]

        # Layer Normalization
        emb = F.layer_norm(emb, emb.shape[-1:])

        # L2 정규화
        emb = emb.cpu().numpy().flatten()
        emb = emb / (np.linalg.norm(emb) + 1e-12)

        return emb
    except Exception as e:
        print(f"⚠️ 텍스트 임베딩 실패: {e}")
        return None

# 배치 처리로 속도 개선
print(f"\n📝 {len(df_all)}개 텍스트 임베딩 생성 중...")
text_embs = []

for text in tqdm(df_all["text"], desc="Text Embedding"):
    emb = extract_text_emb(text)
    text_embs.append(emb)

# df_all에 업데이트
df_all["text_emb"] = text_embs

# 오디오 임베딩 및 피처 결합
audio_embs = np.vstack(df_all["audio_emb"].to_numpy())
audio_feat_cols = ["duration_sec", "speech_ratio", "energy_mean", "peak_amp", "peak_to_rms"]

if all(col in df_all.columns for col in audio_feat_cols):
    audio_feats = df_all[audio_feat_cols].fillna(0).to_numpy()
    Xa = np.hstack([audio_embs, audio_feats])
else:
    Xa = audio_embs

# 텍스트 임베딩 및 레이블
Xt = np.vstack(df_all["text_emb"].to_numpy())
y = df_all["urgency_y"].to_numpy()

# 드라이브 저장 경로
save_dir = "/content/drive/MyDrive/embeddings_backup"
os.makedirs(save_dir, exist_ok=True)

# numpy 배열로 변환 (None 값 필터링)
valid_df = df_all[df_all["audio_emb"].apply(lambda x: isinstance(x, np.ndarray)) &
                  df_all["text_emb"].apply(lambda x: isinstance(x, np.ndarray))].reset_index(drop=True)

audio_embs = np.vstack(valid_df["audio_emb"].to_numpy())
text_embs  = np.vstack(valid_df["text_emb"].to_numpy())

# 메타데이터 저장
meta_path = os.path.join(save_dir, "metadata.csv")
valid_df.drop(columns=["audio_emb", "text_emb"]).to_csv(meta_path, index=False)

# 임베딩 압축 저장
emb_path = os.path.join(save_dir, "embeddings.npz")
np.savez_compressed(emb_path, audio=audio_embs, text=text_embs)

print(f"✅ 저장 완료!")
print(f"📁 메타데이터: {meta_path}")
print(f"📦 임베딩: {emb_path}")
