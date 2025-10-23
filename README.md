# UrgencyDetect -119 신고 전화 WAV 파일, 라벨데이터 기반 긴급도/감정 분류 모델 학습  
* 본 프로젝트는 멋쟁이사자처럼 AI 자연어처리(NLP)단기 심화 교육에서 진행한 긴급도 판단 모델 구축 프로젝트입니다.


## 📂 전처리 단계

### 1️⃣ 데이터 압축 해제

* 원천데이터(WAV)와 라벨링데이터(JSON)가 각각 ZIP 파일로 제공되며, 지정된 디렉터리(EXTRACT_DIR)에 압축 해제됩니다.
* `Wave` 폴더에는 음성 파일(.wav), `Label` 폴더에는 JSON 라벨 파일이 저장됩니다.

### 2️⃣ WAV–JSON 매핑

* WAV 파일명(`_id`)을 기준으로 JSON 내 동일한 ID와 매칭합니다.
* 매칭되지 않거나 존재하지 않는 파일은 자동으로 제외됩니다.

### 3️⃣ JSON 파싱 및 텍스트 추출

* 각 JSON 파일의 `utterances` 항목을 반복 처리하여 발화 구간(`startAt`, `endAt`)과 텍스트(`text`)를 추출합니다.
* 해당 발화에 연결된 긴급도(`urgencyLevel`), 감정(`sentiment`), 재난 매체(`disasterMedium`) 등 메타 정보를 함께 저장합니다.

### 4️⃣ 오디오 세그먼트 로딩

* `startAt`, `endAt` 구간 단위로 오디오를 잘라 로딩하며, 최소 0.5초 이상인 구간만 유효 데이터로 인정합니다.
* 샘플링 레이트는 16kHz(TARGET_SR)로 통일합니다.

### 5️⃣ Wav2Vec2 임베딩 추출

* 사전학습된 `kresnik/wav2vec2-large-xlsr-korean` 모델을 활용해 각 음성 세그먼트의 임베딩 벡터를 추출합니다.
* 추출된 벡터는 평균 풀링(mean pooling)을 적용하여 고정 길이(1024차원) 임베딩으로 변환됩니다.
* 임베딩은 `numpy` 배열로 저장되며, 학습에 직접 사용됩니다.

### 6️⃣ 레이블 매핑

* 긴급도 레이블을 정수형으로 매핑합니다.

  ```
  {'상': 2, '중': 1, '하': 0}
  ```
* `urgency_label` 컬럼에 정수형 라벨을 저장하여 모델 학습 시 바로 활용할 수 있도록 합니다.

### 7️⃣ 결과 저장

* 전처리 완료된 데이터프레임을 CSV(`processed_data.csv`)와 NPY(`wav2vec2_embeddings.npy`) 형식으로 저장합니다.
* CSV에는 오디오 경로, 텍스트, 긴급도, 감정, 재난 매체 등의 정보가 포함됩니다.

---

## 🧾 최종 산출물 예시

| 컬럼명                  | 설명              |
| -------------------- | --------------- |
| `_id`                | 샘플 고유 ID        |
| `wav_file_path`      | WAV 파일 경로       |
| `utterance_id`       | 발화 구간 ID        |
| `text`               | 발화 텍스트          |
| `startAt`, `endAt`   | 발화 구간(ms 단위)    |
| `sentiment`          | 감정 라벨           |
| `urgencyLevel`       | 원본 긴급도 (상/중/하)  |
| `urgency_label`      | 정수형 긴급도 (2/1/0) |
| `wav2vec2_embedding` | 음성 임베딩 벡터       |

---
## 📂 학습 단계

### 1️⃣ 환경 설정
- **GPU 사용 여부 확인**: `cuda` 사용 가능 시 GPU 활용
- **라이브러리**
  - PyTorch, Transformers, scikit-learn, tqdm, pandas, numpy

---

### 2️⃣ 데이터 로드 및 전처리
1. **CSV 로드**
   - `광주_구급_full_final.csv` 불러오기
   - WAV2Vec2 임베딩(`wav2vec2_embeddings_full.npy`) 로드 후 DataFrame에 추가
2. **Label Encoding**
   - `sentiment` → `sentiment_label`
   - `urgencyLevel` → `urgency_label`
3. **Class Imbalance 처리**
   - `compute_class_weight`로 클래스별 가중치 계산
   - GPU Tensor로 변환 후 CrossEntropyLoss에 적용

---

### 3️⃣ Dataset 정의
- `MultiModalDataset` 클래스 사용
  - 텍스트: tokenizer → `input_ids`, `attention_mask`
  - 오디오: Wav2Vec2 임베딩
  - 레이블: Sentiment, Urgency
- DataLoader 생성
  - Train / Val split = 85% / 15%
  - Batch size = 16

---

### 4️⃣ 멀티모달 모델 구조

#### 4-1. Cross-Attention 모듈
- MultiheadAttention 기반
- Query: 텍스트 CLS 임베딩
- Key/Value: 오디오 임베딩
- LayerNorm + Dropout 적용

#### 4-2. KoElectraCrossModal
- **텍스트 인코더:** KoELECTRA (`ElectraModel.from_pretrained`)  
  - CLS 토큰 `[0]` 추출
- **오디오 인코더:** Linear → ReLU
- **Cross-Attention:** 텍스트 CLS ↔ 오디오 임베딩
- **분류 헤드**
  - Sentiment head: Linear → 클래스 개수
  - Urgency head: Linear → 클래스 개수
- **Loss 계산**
  - CrossEntropyLoss + 클래스 가중치
  - Multi-task weighted sum: `loss_total = 0.4 * loss_sent + 0.6 * loss_urg`

---

### 5️⃣ 학습 설정
- **Optimizer:** AdamW (lr=2e-5, weight_decay=0.01)
- **Scheduler:** CosineAnnealingLR (T_max=EPOCHS, eta_min=1e-6)
- **Epochs:** 5
- **Gradient Clipping:** max_norm=1.0

---

### 6️⃣ 학습 루프
1. **Train 단계**
   - 모델 train 모드
   - 배치별 forward → loss 계산 → backward → optimizer step
   - CrossEntropyLoss + 클래스 가중치 적용
   - tqdm progress bar로 loss, LR 모니터링
2. **Validation 단계**
   - 모델 eval 모드
   - Sentiment & Urgency accuracy, macro-F1 계산
3. **모델 저장**
   - Validation 평균 F1 기준 최고 모델 저장
   - checkpoint: model_state_dict, optimizer_state_dict, scheduler_state_dict 포함

---

### 7️⃣ 평가
- **최종 모델 로드 후 Validation**
- Metrics:
  - Sentiment: Accuracy, Macro-F1
  - Urgency: Accuracy, Macro-F1
  - Average F1
- **Classification Report** 출력
  - Sentiment, Urgency 별 precision, recall, f1-score

---

### 8️⃣ 학습 파이프라인 요약

| 단계 | 주요 특징 |
|------|-----------|
| 데이터 | CSV + Wav2Vec2 embedding, Label Encoding, Class Weights |
| 모델 | KoELECTRA + Wav2Vec2 + Cross-Attention + Multi-task Head |
| 학습 | AdamW + CosineAnnealingLR, Gradient Clipping, Multi-task Loss |
| 평가 | Accuracy, Macro-F1, Classification Report, Best Model 저장 |

---

