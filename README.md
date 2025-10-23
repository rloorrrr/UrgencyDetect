# UrgencySense

- 본 프로젝트는 멋쟁이사자처럼 AI 자연어처리(NLP)단기 심화 교육에서 진행한 긴급도 판단 모델 구축 프로젝트입니다.

<br>

## 폴더 구조

## 실행

🧩 데이터 전처리 (📄 preprocess.py)
1️⃣ 데이터 로드 및 통합

입력 데이터

JSON: /TL_광주_구조
WAV: /TS_광주_구조
JSON 내 utterances.text를 추출하여 텍스트 문장 생성
file_id 기준으로 오디오와 텍스트 병합
urgencyLevel을 다음과 같이 숫자 레이블로 매핑:
{"하": 0, "중": 1, "상": 2}


2️⃣ 오디오 전처리

Librosa 기반 음향 피처 추출:
duration_sec: 전체 길이(초)
speech_ratio: 발화(비무성) 구간 비율
energy_mean: 평균 에너지
peak_amp: 최대 진폭
peak_to_rms: 피크 대비 RMS 비율
음성 신호를 정규화(normalize) 후 Data2Vec-Audio 모델 입력으로 사용

3️⃣ 임베딩 생성
🎧 오디오 임베딩
모델: facebook/data2vec-audio-base
방법: last_hidden_state 평균(mean pooling)
벡터 정규화 (L2 normalization)

🗣️ 텍스트 임베딩
모델: jhgan/ko-sroberta-multitask
입력 최대 길이: 384
[CLS] 토큰 임베딩 사용 → LayerNorm + L2 정규화

4️⃣ 피처 결합 및 저장
오디오 임베딩 + 피처(duration_sec 등)를 수평 결합 (np.hstack)
텍스트 임베딩과 레이블(urgency_y) 병합
모든 유효 샘플(np.ndarray 형태만) 필터링
다음 경로에 저장:

/content/drive/MyDrive/embeddings_backup/
├── metadata.csv
└── embeddings.npz  (audio, text)




🧠 학습 파이프라인 (📄 train.py)

1️⃣ 데이터 로드
위에서 저장된 metadata.csv 및 embeddings.npz 불러오기
학습/검증 8:2 비율로 분할 (stratify by urgency)

2️⃣ 데이터로더 구성
PairDataset을 통해 (오디오, 텍스트, 라벨) 형태로 구성
클래스 불균형 완화를 위해 WeightedRandomSampler 적용
→ 각 클래스의 샘플 수에 반비례하는 가중치 사용

3️⃣ 모델 구조
⚙️ CrossAttentionFusionOrdinal
오디오/텍스트 각각 별도 projection (Linear + LayerNorm + GELU)
Cross-Attention Transformer Encoder (3 layers)
두 modality의 interaction feature 생성:
concat([audio, text, |audio-text|, audio*text])

최종 Classifier:
Linear(hidden*4 → 512) → GELU → Dropout → Linear(512 → 3)


4️⃣ 손실 함수 (Loss)

세 가지 방식을 확률적으로 혼합하여 학습 안정화 및 일반화 향상:

Loss 종류	설명	적용 확률
KLDivLoss	soft label (label smoothing=0.2)	25%
Ordinal Focal Loss	계층적 거리 반영, 클래스 불균형 대응	40%
CrossEntropyLoss	기본 분류 기준	35%


5️⃣ 학습 설정

Optimizer: AdamW (lr=3e-5, weight_decay=0.01)
Scheduler: CosineAnnealingWarmRestarts
Early Stopping: 10 epoch patience
Gradient Clipping: max_norm=1.0
Epochs: 50


6️⃣ 모델 저장

검증 F1 점수가 향상될 때마다 모델 파라미터 저장
최종 Best 모델 전체 저장:
/content/drive/MyDrive/urgency_checkpoints_final/best_model_full.pt


🎯 핵심 특징 요약
구분	내용
오디오 모델	facebook/data2vec-audio-base
텍스트 모델	jhgan/ko-sroberta-multitask
결합 구조	Cross-Attention Transformer
Loss 전략	KL + Focal + CE 혼합
불균형 대응	Weighted Sampler + Focal Loss
성능 지표	Macro-F1
출력 클래스	긴급도 3단계 (하/중/상)


🧾 참고 실험 기록
모델	오디오 임베딩	텍스트 임베딩	Fusion	최고 F1
Baseline	AST + KcELECTRA	단순 Concat	Late Fusion	0.47
개선	Data2Vec + SroBERTa	Cross-Attention	Early Fusion	0.61~0.64