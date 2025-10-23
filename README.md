# UrgencySense 

- 본 프로젝트는 멋쟁이사자처럼 AI 자연어처리(NLP)단기 심화 교육에서 진행한 긴급도 판단 모델 구축 프로젝트입니다.

<br>

## 프로젝트 설명 
- 음성 파일(.wav)과 발화된 음성을 변환한 텍스트를 사용해 긴급도(urgencyLevel)를 판별하는 멀티 모달리티, 멀티 태스크 모델을 구현하고자 하였습니다. 
- AI허브의 위급상황 음성/음향 (고도화) - 119 지능형 신고접수 음성 인식 데이터를 사용했습니다.

## 데이터 구조 
```
UrgencyDetect/
├─ preprocess_data/
│ └─ preprocess_data.csv
├─ README.md
├─ inference.py
├─ preprocess.py
├─ requirement.txt
└─ train.py
```

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


## inference

1. [googleDrive](https://drive.google.com/drive/folders/11xbdv4FtbQaRsEnJWC1eDQBCFI6krnYy?usp=drive_link) 에서 해당하는 체크포인트 파일을 불러온다
2. SAVE_PATH = "./checkpoints/YOUR_PT" 를 해당 체크포인트 이름으로 바꿔준다 
3. inference.py 실행


## 결과

Test Accuracy: 0.6982

Test F1 Score: 0.6995
