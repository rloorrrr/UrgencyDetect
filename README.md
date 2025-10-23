# 🚨 Urgency – 긴급도 판별 AI

본 프로젝트는 멋쟁이사자처럼 AI 자연어처리(NLP) 단기 심화 과정에서 진행한  
음성·텍스트 멀티모달 기반 긴급도 판단 모델 구축 프로젝트입니다.

## 🧩 데이터 전처리

1️⃣ WAV–JSON 매칭  
- `1. 원천 데이터` 내 `.wav` 파일과 `2. 라벨링 데이터` 내 `.json` 파일을 파일명 기준으로 매칭  
- JSON의 `utterances.text`를 추출하고, `[개인정보]` 문구 제거  
- `id_key`를 기준으로 병합하여 오디오–라벨링 매칭 완료

2️⃣ 텍스트 전처리  
- 여러 발화를 하나로 연결하여 `all_text` 컬럼 생성  
- 공백 및 특수문자 제거  
- 긴급도(`urgencyLevel`)를 정수 라벨로 매핑: {'하': 0, '중': 1, '상': 2}

3️⃣ 데이터 불균형 완화  
- 클래스 불균형 해소를 위해 WeightedRandomSampler 사용  
- 학습 시 각 클래스에 역비례 가중치를 적용

4️⃣ 최종 데이터셋 컬럼  
| 컬럼명 | 설명 |
|----------------|-------------------------------------|
| `audio_path`   | WAV 파일 경로                        |
| `all_text`     | 전처리된 통합 텍스트                  |
| `urgencyLevel` | 원본 긴급도 라벨                      |
| `label`        | 정수형 긴급도 라벨 (0:하, 1:중, 2:상) |

- 최종 데이터 CSV: `./preprocess_data/preprocess.csv`

## 🧠 학습 방식

프로세스 단계
1. AI-Hub 음성 + JSON 데이터 수집  
2. 전처리 (오디오–라벨 매칭, 텍스트 정제)  
3. Dataset 구성 및 DataLoader 생성  
4. Whisper + KoELECTRA 멀티모달 모델 설계  
5. Cross-Attention 기반 임베딩 융합  
6. 긴급도(하·중·상) 분류 모델 학습  
7. Validation F1 Score 기준 EarlyStopping  
8. 최종 추론(Inference) 및 성능 평가

---

- 오디오 인코더: Whisper Encoder (`openai/whisper-small`)  
- 텍스트 인코더: KoELECTRA (`monologg/koelectra-base-v3-discriminator`)  
- 결합 구조: Cross-Attention Layer ×2 (head=8) + Hybrid Pooling (CLS·Mean Concatenation)  
- 정규화: Dual Normalization (GroupNorm + LayerNorm)  
- 손실 함수: FocalLoss(γ=1.8) + Weighted Sampler  
- 스케줄러: CosineAnnealingWarmRestarts  
- EarlyStopping: Validation F1 Score 기준


## ⚙️ 모델 구조 요약

```
음성(WAV)  → Whisper Encoder → AudioSeqEncoder
텍스트(JSON) → KoELECTRA Encoder → TextSeqEncoder
↓
Cross-Attention Fusion
↓
Classifier (Linear + ReLU + Dropout)
↓
긴급도 출력 (하 / 중 / 상)
```

- 모델 학습 코드: `embedding+train.py`  
- 추론 코드: `inference.py` (JSON + WAV 입력 지원)

## 🔍 추론 (Inference)

```bash
python inference.py
```

**예시 출력:**
```
🗣️ JSON 텍스트 예시: 119입니다. 네. 네, 저 뭐지? 엄마가 허리를 지금 다치셔가지고. 전혀 움직이지를 못하셔서 병원을 가봐야 될 것 같은데 와주실 수 있나요? 어. 뭐 허리를 다치셨어요, 어떻게 ...
🎧 오디오 파일: 6509537abc5846983ad2efb3_20230523072543.wav
✅ 예측된 긴급도: 하
📊 확률분포: [0.37645307 0.34889665 0.27465025]
```

모델 가중치(`.pt`) 파일은 `/content/drive/MyDrive/test_data/model/1500_model.pt` 위치에 저장됩니다.

## 📊 모델 성과

Test Accuracy     : 0.2767
Weighted F1 Score : 0.2776 
