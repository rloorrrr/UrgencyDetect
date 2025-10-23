# -*- coding: utf-8 -*-
"""
긴급도분류모델_데이터전처리
- WAV 파일 및 JSON 라벨 데이터를 불러와 DataFrame 생성
- Wav2Vec2 pretrained 모델로 음성 구간 임베딩 추출
- 결과 저장
"""

import os
import glob
import re
import json
import zipfile
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
import numpy as np
import librosa
import torch
from tqdm import tqdm
from transformers import Wav2Vec2Processor, Wav2Vec2Model

# =========================================================
# 0️⃣ Google Drive Mount (Colab 환경에서만 필요)
# =========================================================
from google.colab import drive
drive.mount('/content/drive')

# =========================================================
# 1️⃣ ZIP 파일 압축 해제
# =========================================================
def unzip_file(zip_path, extract_dir):
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_dir)
    print(f"✅ 압축 해제 완료: {extract_dir}")

# WAV 파일
wav_zip = "/content/drive/MyDrive/urgency_data/Validation/1.원천데이터/VS_광주_구급.zip"
wav_extract_dir = "/content/audio_dataset/Validation/Wave"
unzip_file(wav_zip, wav_extract_dir)

# JSON 라벨 파일
label_zip = "/content/drive/MyDrive/urgency_data/Validation/2.라벨링데이터/VL_광주_구급.zip"
label_extract_dir = "/content/audio_dataset/Validation/Label"
unzip_file(label_zip, label_extract_dir)

# =========================================================
# 2️⃣ WAV 파일 스캔 및 ID → 경로 매핑
# =========================================================
def build_wav_map(base_path):
    all_wav_files = glob.glob(os.path.join(base_path, "**/*.wav"), recursive=True)
    print(f"🔍 탐색된 WAV 파일 수: {len(all_wav_files)}")

    wav_map = {}
    for wav in all_wav_files:
        basename = os.path.basename(wav)
        name = os.path.splitext(basename)[0]

        match = re.match(r"^([A-Za-z0-9]+)_[0-9]{8}", name)
        file_id = match.group(1) if match else name.split("_")[0]
        wav_map[str(file_id)] = wav

    print(f"✅ 추출된 wav ID 개수: {len(wav_map)}")
    return wav_map

wav_map = build_wav_map(wav_extract_dir)

# =========================================================
# 3️⃣ JSON 파일 읽고 DataFrame 생성
# =========================================================
def build_dataframe(json_dir, wav_map):
    json_files = glob.glob(os.path.join(json_dir, '**/*.json'), recursive=True)
    all_rows = []

    for file in tqdm(json_files, desc="Reading JSON"):
        with open(file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        json_id = str(data.get('_id'))
        wav_path = wav_map.get(json_id, None)

        common_labels = {
            '_id': json_id,
            'sentiment': data.get('sentiment'),
            'urgencyLevel': data.get('urgencyLevel'),
            'disasterLarge': data.get('disasterLarge'),
            'disasterMedium': data.get('disasterMedium'),
            'triage': data.get('triage'),
            'gender': data.get('gender'),
            'mediaType': data.get('mediaType'),
            'address': data.get('address'),
            'symptom': ','.join(data.get('symptom', [])),
            'wav_file_path': wav_path
        }

        for u in data['utterances']:
            all_rows.append({
                **common_labels,
                'utterance_id': u['id'],
                'startAt': u['startAt'],
                'endAt': u['endAt'],
                'text': u['text'],
                'speaker': u['speaker']
            })

    df = pd.DataFrame(all_rows)
    df = df[df["wav_file_path"].notna()]
    df = df[df["wav_file_path"].apply(os.path.exists)]
    print(f"✅ 유효한 wav 파일 수: {len(df)}")
    return df

df_u = build_dataframe(label_extract_dir, wav_map)

# =========================================================
# 4️⃣ Wav2Vec2 모델 로드
# =========================================================
MODEL_NAME = "kresnik/wav2vec2-large-xlsr-korean"
device = "cuda" if torch.cuda.is_available() else "cpu"

processor = Wav2Vec2Processor.from_pretrained(MODEL_NAME)
model = Wav2Vec2Model.from_pretrained(MODEL_NAME).to(device)
model.eval()
target_sr = 16000

# =========================================================
# 5️⃣ 오디오 구간별 로딩 함수
# =========================================================
def load_waveform_segment(wav_path, start_ms, end_ms, sr=target_sr, min_sec=0.5):
    waveform, _ = librosa.load(wav_path, sr=sr, mono=True)
    start_sample = int(start_ms / 1000 * sr)
    end_sample = int(end_ms / 1000 * sr)
    segment = waveform[start_sample:end_sample]
    return segment if len(segment) >= int(min_sec * sr) else None

# =========================================================
# 6️⃣ Streaming 방식 임베딩 추출
# =========================================================
def extract_embeddings(df, batch_size=32, num_workers=8):
    embeddings = []

    with torch.no_grad():
        for i in tqdm(range(0, len(df), batch_size), desc="Extracting"):
            batch_rows = df.iloc[i:i+batch_size]

            # CPU 병렬로 waveform 로드
            with ThreadPoolExecutor(max_workers=num_workers) as executor:
                waveforms = list(executor.map(
                    lambda r: load_waveform_segment(r.wav_file_path, r.startAt, r.endAt),
                    batch_rows.itertuples(index=False)
                ))

            valid = [(wf, idx) for idx, wf in enumerate(waveforms) if wf is not None]
            if not valid:
                embeddings.extend([None]*len(batch_rows))
                continue

            batch_embeddings = [None]*len(batch_rows)
            for wf, idx in valid:
                input_values = processor(wf, sampling_rate=target_sr, return_tensors="pt").input_values.to(device)
                outputs = model(input_values)
                hidden_states = outputs.last_hidden_state.squeeze(0).cpu().numpy()
                batch_embeddings[idx] = np.mean(hidden_states, axis=0)

            embeddings.extend(batch_embeddings)
    return embeddings

embeddings = extract_embeddings(df_u)
df_u["wav2vec2_embedding"] = embeddings
df_u = df_u[df_u["wav2vec2_embedding"].notna()].reset_index(drop=True)

# =========================================================
# 7️⃣ 결과 저장
# =========================================================
np.save(
    "/content/drive/MyDrive/train_embedding_split_utter/valid_wav2vec2_embeddings_split.npy",
    np.stack(df_u["wav2vec2_embedding"].to_numpy())
)
df_u.to_csv(
    "/content/drive/MyDrive/train_embedding_split_utter/valid_광주_구급_split.csv",
    index=False
)

print("✅ Streaming 방식 Embedding 추출 완료 및 저장 완료")
