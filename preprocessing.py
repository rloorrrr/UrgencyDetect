# -*- coding: utf-8 -*-
"""
긴급도분류모델_데이터전처리 

"""

import os
import json
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
import numpy as np
import librosa
import torch
from tqdm import tqdm
from transformers import Wav2Vec2Processor, Wav2Vec2Model

# =========================================================
# 설정
# =========================================================
WAV_ZIP = "/content/drive/MyDrive/urgency_data/Validation/1.원천데이터/VS_광주_구급.zip"
JSON_ZIP = "/content/drive/MyDrive/urgency_data/Validation/2.라벨링데이터/VL_광주_구급.zip"
EXTRACT_DIR = "/content/audio_dataset/Validation"
OUTPUT_CSV = "/content/drive/MyDrive/train_embedding_split_utter/valid_광주_구급_split.csv"
OUTPUT_NPY = "/content/drive/MyDrive/train_embedding_split_utter/valid_wav2vec2_embeddings_split.npy"
MODEL_NAME = "kresnik/wav2vec2-large-xlsr-korean"
TARGET_SR = 16000

# =========================================================
# 함수 정의
# =========================================================
def unzip_file(zip_path, extract_dir):
    from zipfile import ZipFile
    with ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_dir)
    print(f"✅ 압축 해제 완료: {extract_dir}")

def build_wav_map(wav_dir):
    wav_files = list(Path(wav_dir).rglob("*.wav"))
    wav_map = {wav.stem.split("_")[0]: str(wav) for wav in wav_files}
    print(f"🔍 WAV 파일 수: {len(wav_files)}, ID 수: {len(wav_map)}")
    return wav_map

def build_dataframe(json_dir, wav_map):
    json_files = list(Path(json_dir).rglob("*.json"))
    rows = []
    for file in tqdm(json_files, desc="Reading JSON"):
        try:
            with open(file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            wav_path = wav_map.get(str(data.get('_id')))
            if not wav_path: 
                continue
            common = {
                '_id': data.get('_id'),
                'sentiment': data.get('sentiment'),
                'urgencyLevel': data.get('urgencyLevel'),
                'disasterMedium': data.get('disasterMedium'),
                'wav_file_path': wav_path
            }
            for u in data.get('utterances', []):
                rows.append({**common,
                             'utterance_id': u['id'],
                             'startAt': u['startAt'],
                             'endAt': u['endAt'],
                             'text': u['text']})
        except Exception as e:
            print(f"⚠️ JSON 처리 오류 {file}: {e}")
    df = pd.DataFrame(rows)
    df = df[df["wav_file_path"].apply(os.path.exists)].reset_index(drop=True)
    print(f"✅ 유효한 샘플 수: {len(df)}")
    return df

def load_waveform_segment(wav_path, start_ms, end_ms, sr=TARGET_SR, min_sec=0.5):
    waveform, _ = librosa.load(wav_path, sr=sr, mono=True)
    seg = waveform[int(start_ms/1000*sr):int(end_ms/1000*sr)]
    return seg if len(seg) >= int(min_sec*sr) else None

def extract_embeddings(df, processor, model, batch_size=32, num_workers=8):
    embeddings = []
    with torch.no_grad():
        for i in tqdm(range(0, len(df), batch_size), desc="Extracting Embeddings"):
            batch = df.iloc[i:i+batch_size]
            with ThreadPoolExecutor(max_workers=num_workers) as executor:
                waveforms = list(executor.map(
                    lambda r: load_waveform_segment(r.wav_file_path, r.startAt, r.endAt),
                    batch.itertuples(index=False)
                ))
            batch_emb = [None]*len(batch)
            for idx, wf in enumerate(waveforms):
                if wf is not None:
                    input_values = processor(wf, sampling_rate=TARGET_SR, return_tensors="pt").input_values.to(model.device)
                    output = model(input_values)
                    batch_emb[idx] = output.last_hidden_state.squeeze(0).cpu().numpy().mean(axis=0)
            embeddings.extend(batch_emb)
    return embeddings

# =========================================================
# main()
# =========================================================
def main():
    # Google Drive Mount
    from google.colab import drive
    drive.mount('/content/drive')

    # 1️⃣ ZIP 해제
    unzip_file(WAV_ZIP, EXTRACT_DIR+"/Wave")
    unzip_file(JSON_ZIP, EXTRACT_DIR+"/Label")

    # 2️⃣ WAV → ID 매핑
    wav_map = build_wav_map(EXTRACT_DIR+"/Wave")

    # 3️⃣ DataFrame 생성
    df = build_dataframe(EXTRACT_DIR+"/Label", wav_map)

    # 4️⃣ Wav2Vec2 모델 로드
    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = Wav2Vec2Processor.from_pretrained(MODEL_NAME)
    model = Wav2Vec2Model.from_pretrained(MODEL_NAME).to(device)
    model.eval()

    # 5️⃣ 임베딩 추출
    df["wav2vec2_embedding"] = extract_embeddings(df, processor, model)
    df = df[df["wav2vec2_embedding"].notna()].reset_index(drop=True)

    # 6️⃣ 통계 출력
    print("\n=== 데이터 통계 ===")
    print(df['urgencyLevel'].value_counts())
    print(df['disasterMedium'].value_counts())
    print(df['sentiment'].value_counts())

    # 긴급도 레이블 숫자 매핑
    urgency_mapping = {'상': 2, '중': 1, '하': 0}
    df['urgency_label'] = df['urgencyLevel'].map(urgency_mapping).astype(int)

    # 7️⃣ CSV 및 NPY 저장
    df.to_csv(OUTPUT_CSV, index=False, encoding='utf-8-sig')
    np.save(OUTPUT_NPY, np.stack(df["wav2vec2_embedding"].to_numpy()))
    print(f"\n✅ 완료: CSV → {OUTPUT_CSV}, NPY → {OUTPUT_NPY}")
    print(f"총 샘플 수: {len(df)}")
    print(df[['text', 'urgencyLevel', 'urgency_label']].head())

if __name__ == "__main__":
    main()
