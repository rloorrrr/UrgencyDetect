import os
import json
import pandas as pd
from pathlib import Path
from tqdm import tqdm

# ==============================
# 데이터 전처리: JSON + WAV 매칭
# ==============================

# 경로 설정
RAW_AUDIO_DIR = "./data/1.원천데이터/광주_구급"  # WAV 파일들이 있는 폴더
LABEL_JSON_DIR = "./data/2.라벨링데이터/광주_구급"  # JSON 파일들이 있는 폴더
OUTPUT_CSV = "./processed_data.csv"

def extract_data_from_json(json_path):
    """JSON 파일에서 필요한 정보 추출"""
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # utterances의 모든 텍스트 결합
    texts = []
    for utterance in data.get('utterances', []):
        text = utterance.get('text', '').strip()
        # 개인정보 마스킹된 부분 제외
        if text and '***[개인정보]' not in text:
            texts.append(text)
    
    combined_text = ' '.join(texts)
    
    # 레이블 추출
    disaster_medium = data.get('disasterMedium', '')
    urgency_level = data.get('urgencyLevel', '')
    sentiment = data.get('sentiment', '')
    
    return {
        'combined_text': combined_text,
        'disasterMedium': disaster_medium,
        'urgencyLevel': urgency_level,
        'sentiment': sentiment
    }

def match_audio_json_files():
    """WAV 파일과 JSON 파일 매칭"""
    audio_dir = Path(RAW_AUDIO_DIR)
    json_dir = Path(LABEL_JSON_DIR)
    
    # WAV 파일 목록
    wav_files = list(audio_dir.glob('**/*.wav'))
    print(f"Found {len(wav_files)} WAV files")
    
    # JSON 파일 목록
    json_files = list(json_dir.glob('**/*.json'))
    print(f"Found {len(json_files)} JSON files")
    
    # 파일명 기준으로 매칭 (확장자 제외)
    json_dict = {}
    for json_path in json_files:
        basename = json_path.stem  # 확장자 제외한 파일명
        json_dict[basename] = json_path
    
    matched_data = []
    
    for wav_path in tqdm(wav_files, desc="Matching files"):
        basename = wav_path.stem
        
        # 매칭되는 JSON 파일 찾기
        if basename in json_dict:
            json_path = json_dict[basename]
            
            try:
                # JSON 데이터 추출
                json_data = extract_data_from_json(json_path)
                
                # 텍스트가 비어있지 않은 경우만 포함
                if json_data['combined_text']:
                    matched_data.append({
                        'audio_path': str(wav_path),
                        'text': json_data['combined_text'],
                        'disasterMedium': json_data['disasterMedium'],
                        'urgencyLevel': json_data['urgencyLevel'],
                        'sentiment': json_data['sentiment']
                    })
            except Exception as e:
                print(f"Error processing {basename}: {e}")
                continue
    
    return matched_data

def main():
    print("Starting data preprocessing...")
    
    # 데이터 매칭
    matched_data = match_audio_json_files()
    
    if not matched_data:
        print("No matched data found!")
        return
    
    # DataFrame 생성
    df = pd.DataFrame(matched_data)
    
    print(f"\nTotal matched samples: {len(df)}")
    print("\n=== Data Statistics ===")
    print(f"\nUrgency Level distribution:")
    print(df['urgencyLevel'].value_counts())
    print(f"\nDisaster Medium distribution:")
    print(df['disasterMedium'].value_counts())
    print(f"\nSentiment distribution:")
    print(df['sentiment'].value_counts())
    
    # 긴급도 레벨을 숫자로 매핑
    urgency_mapping = {'상': 2, '중': 1, '하': 0}
    df['urgency_label'] = df['urgencyLevel'].map(urgency_mapping)
    
    # 매핑되지 않은 값 확인
    unmapped = df[df['urgency_label'].isna()]
    if len(unmapped) > 0:
        print(f"\n⚠️ Warning: {len(unmapped)} samples with unmapped urgency levels")
        print("Unique unmapped values:", unmapped['urgencyLevel'].unique())
        # 매핑되지 않은 값 제거
        df = df.dropna(subset=['urgency_label'])
    
    df['urgency_label'] = df['urgency_label'].astype(int)
    
    # CSV 저장
    df.to_csv(OUTPUT_CSV, index=False, encoding='utf-8-sig')
    print(f"\n✅ Processed data saved to {OUTPUT_CSV}")
    print(f"Final dataset size: {len(df)}")
    
    # 샘플 데이터 출력
    print("\n=== Sample Data ===")
    print(df[['text', 'urgencyLevel', 'urgency_label']].head())

if __name__ == "__main__":
    main()