# UrgencySense - 은우

- 본 프로젝트는 멋쟁이사자처럼 AI 자연어처리(NLP)단기 심화 교육에서 진행한 긴급도 판단 모델 구축 프로젝트입니다.

<br>

## 전처리 
1️⃣ WAV 파일과 JSON 파일 매칭
RAW_AUDIO_DIR 폴더 내 WAV 파일과 LABEL_JSON_DIR 폴더 내 JSON 파일을 파일명 기준으로 매칭합니다.
JSON 파일 내 utterances 항목에서 텍스트를 추출하며, 개인정보(***[개인정보])가 포함된 텍스트는 제거합니다.

2️⃣ 텍스트 전처리
JSON 내 여러 발화(utterances) 텍스트를 공백 기준으로 연결하여 하나의 combined_text 생성
공백 제거 및 불필요한 문자열 제거
긴급도(urgencyLevel) 값은 학습용 정수 레이블로 매핑: {'상': 2, '중': 1, '하': 0}

3️⃣ 데이터 불균형 완화
긴급도 클래스(urgencyLevel)별 데이터 수 불균형을 완화하기 위해 샘플링 기법 적용
오버샘플링: 데이터가 적은 클래스 샘플 수를 가장 많은 클래스 수에 맞춰 복제
샘플링 후 데이터는 랜덤 셔플링하여 학습 데이터셋 구성

4️⃣ 최종 데이터셋
전처리 완료 후 CSV 파일(processed_data_balanced.csv)로 저장

CSV 파일 컬럼:
audio_path: WAV 파일 경로
text: 결합된 텍스트
disasterMedium: 재난 매체 정보
urgencyLevel: 원본 긴급도 라벨
urgency_label: 정수형 긴급도 라벨
sentiment: 감정 라벨
균형 조정 완료 후 클래스별 샘플 수 확인 가능

<br>

## 학습 방식 
```
flowchart TD
    A[원천 데이터: WAV + JSON] --> B[전처리]
    B --> B1[JSON에서 텍스트 추출 및 개인정보 제거]
    B --> B2[WAV 파일 매칭, 리샘플링 및 패딩/자르기]
    B --> B3[CSV 저장 - audio_path, text, disasterMedium, urgencyLevel, sentiment, urgency_label]

    B3 --> C[데이터셋 준비]
    C --> C1[Train/Val/Test split 7-1.5-1.5]
    C --> C2[EmergencyDataset 클래스: 텍스트 + 오디오 처리]
    C --> C3[DataLoader 생성]

    C3 --> D[멀티모달 모델 - EmergencyClassifier]
    D --> D1[텍스트 인코더 - KoELECTRA (마지막 2개 레이어만 학습)]
    D --> D2[오디오 인코더 - Wav2Vec2 (마지막 레이어만 학습)]
    D --> D3[텍스트 + 오디오 임베딩 결합 → 분류기]

    D3 --> E[학습]
    E --> E1[손실함수 - CrossEntropy + 클래스 가중치]
    E --> E2[옵티마이저 - AdamW + LR 스케줄러]
    E --> E3[FP16 mixed precision 학습]

    E3 --> F[모델 평가]
    F --> F1[Test Accuracy, Weighted F1, Classification Report]

```
<br>

## 결과

Test Accuracy: 0.6982
Test F1 Score: 0.6995
