
!pip install -q koreanize-matplotlib

# 라이브러리 불러오기
import matplotlib.pyplot as plt
import koreanize_matplotlib
import pandas as pd
import numpy as np
import json
import glob
import os
import re
import librosa


# 구글 드라이브와 마운트
from google.colab import drive
drive.mount('/content/drive')

# 📂 드라이브 경로 지정 (샘플 데이터)
base_path = "/content/drive/MyDrive/data/Training/2.라벨링데이터/TL_광주_구급"
base_path_wav = "/content/drive/MyDrive/data/Training/1.원천데이터/TS_광주_구급"

# 모든 하위 폴더의 .json, wav 파일 찾기
json_files = glob.glob(os.path.join(base_path, "**/*.json"), recursive=True)
print(f"총 {len(json_files)}개의 JSON 파일 발견")
wav_files = glob.glob(os.path.join(base_path_wav, "**/*.wav"), recursive=True)
print(f"총 {len(wav_files)}개의 WAV 파일 발견")

# JSON 파일 읽기
data = []
for f in json_files:
    try:
        with open(f, "r", encoding="utf-8") as j:
            data.append(json.load(j))
    except Exception as e:
        print(f"⚠️ {f} 읽기 실패: {e}")

# JSON, wav → DataFrame 변환
df = pd.json_normalize(data)
df_wav = pd.DataFrame({"audio_path": wav_files})

# 파일명, 상위폴더명 컬럼 추가 (분석용)
df_wav["filename"] = df_wav["audio_path"].apply(lambda x: os.path.basename(x))
df_wav["parent_folder"] = df_wav["audio_path"].apply(lambda x: os.path.basename(os.path.dirname(x)))

# 기본 정보 확인
print(f"총 {len(df)}개의 행 로드 완료")
print(f"총 {len(df_wav)}개의 파일 로드 완료")


# ==========================================

# ✅ 1️⃣ WAV 파일명에서 _id 부분만 추출 (언더바 앞)
df_wav["id_key"] = df_wav["filename"].apply(lambda x: x.split("_")[0])

# ✅ 2️⃣ JSON의 _id를 문자열로 통일
df["_id"] = df["_id"].astype(str)
df["id_key"] = df["_id"]

# ✅ 3️⃣ 병합 (_id <-> filename 앞부분)
df_merged = pd.merge(
    df,
    df_wav[["audio_path", "id_key"]],
    on="id_key",
    how="left"
)

print("✅ 병합 완료:", len(df_merged))
print(df_merged[["audio_path", "_id", "urgencyLevel"]].head())


# ============================================

urgency_map = {"하": 0, "중": 1, "상": 2}
df_merged["urgency_encoded"] = df_merged["urgencyLevel"].map(urgency_map)


# ==========================================================


# utterances 파싱 (문자열이면 JSON으로 변환)
def parse_json(x):
    if isinstance(x, str):
        try:
            return json.loads(x)
        except:
            return []
    return x

df_merged["utterances"] = df_merged["utterances"].apply(parse_json)

# ✅ 파일 단위 데이터 구성 (utterances 전체를 하나의 텍스트로 합치기)
def parse_json(x):
    if isinstance(x, str):
        try:
            return json.loads(x)
        except:
            return []
    return x

df_merged["utterances"] = df_merged["utterances"].apply(parse_json)

# 각 파일의 모든 발화를 하나의 문자열로 합치기
df_merged["all_text"] = df_merged["utterances"].apply(
    lambda x: " ".join([u["text"] for u in x if isinstance(u, dict) and "text" in u])
)

# 최종 파일 단위 데이터프레임 구성
df_final = df_merged[["audio_path", "all_text", "urgencyLevel", "urgency_encoded"]].dropna()
df_final.rename(columns={"urgency_encoded": "label"}, inplace=True)

print("🎯 파일 단위 데이터 샘플:")
print(df_final.head())
print(f"파일 수: {len(df_final)}")
print(f"레이블 분포:\n{df_final['label'].value_counts().sort_index()}")

# CSV 저장
save_path = "/content/drive/MyDrive/test_data/df_final_file_level_구급_clean.csv"
df_final.to_csv(save_path, index=False, encoding="utf-8-sig")
print("💾 파일 단위 CSV 저장 완료:", save_path)