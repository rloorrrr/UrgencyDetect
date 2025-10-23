# =========================================================
# 1. 라이브러리 로드
# =========================================================
import json, torch, librosa, numpy as np, os
from librosa.util import normalize
from transformers import AutoProcessor, AutoModel, AutoTokenizer
import torch.nn as nn
import torch.nn.functional as F

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"✅ Device: {device}")

# =========================================================
# 2. 오디오 / 텍스트 임베딩 도구 로드
# =========================================================
AUDIO_MODEL = "facebook/data2vec-audio-base"
TEXT_MODEL  = "jhgan/ko-sroberta-multitask"

print("📥 모델 다운로드 중...")
audio_processor = AutoProcessor.from_pretrained(AUDIO_MODEL)
audio_model = AutoModel.from_pretrained(AUDIO_MODEL).to(device).eval()

tokenizer = AutoTokenizer.from_pretrained(TEXT_MODEL)
text_model = AutoModel.from_pretrained(TEXT_MODEL).to(device).eval()
print("✅ 임베딩 모델 로드 완료\n")

# =========================================================
# 3. 모델 정의 (학습 코드와 동일)
# =========================================================
class CrossAttentionFusionOrdinal(nn.Module):
    def __init__(self, audio_dim, text_dim=768, hidden_dim=768, num_classes=3, nhead=8):
        super().__init__()
        self.audio_proj = nn.Sequential(
            nn.Linear(audio_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.4)
        )
        self.text_proj = nn.Sequential(
            nn.Linear(text_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.4)
        )
        enc_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=nhead,
            dim_feedforward=2048,
            dropout=0.2,
            activation='gelu',
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=3)
        fusion_dim = hidden_dim * 4
        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(0.4),
            nn.Linear(512, num_classes)
        )

    def forward(self, a, t):
        a = self.audio_proj(a)
        t = self.text_proj(t)
        x = torch.stack([a, t], dim=1)
        h = self.encoder(x)
        pooled = torch.cat([
            h[:,0,:],
            h[:,1,:],
            torch.abs(h[:,0,:]-h[:,1,:]),
            h[:,0,:]*h[:,1,:]
        ], dim=-1)
        return self.classifier(pooled)

# =========================================================
# 4. 모델 로드 (자동 차원 감지)
# =========================================================
SAVE_PATH = "/content/drive/MyDrive/urgency_checkpoints_final/best_model_full.pt"

print(f"📦 모델 불러오는 중: {SAVE_PATH}")

if not os.path.exists(SAVE_PATH):
    raise FileNotFoundError(f"❌ 모델 파일을 찾을 수 없습니다: {SAVE_PATH}")

# 먼저 모델을 로드해서 audio_dim 확인
loaded = torch.load(SAVE_PATH, map_location=device, weights_only=False)

# audio_dim 자동 감지
audio_dim = None

if isinstance(loaded, nn.Module):
    # 전체 모델이 저장된 경우
    print("🔍 전체 모델 감지")
    try:
        # audio_proj의 첫 번째 Linear 레이어에서 입력 차원 추출
        first_linear = loaded.audio_proj[0]
        audio_dim = first_linear.in_features
        print(f"✅ 감지된 audio_dim: {audio_dim}")
    except:
        print("⚠️ audio_dim 자동 감지 실패")
    model = loaded
else:
    # state_dict가 저장된 경우
    print("🔍 state_dict 감지")
    state_dict = loaded.get("model_state_dict", loaded)

    # audio_proj.0.weight의 shape에서 입력 차원 추출
    first_layer_key = "audio_proj.0.weight"
    if first_layer_key in state_dict:
        audio_dim = state_dict[first_layer_key].shape[1]
        print(f"✅ 감지된 audio_dim: {audio_dim}")
    else:
        print("⚠️ audio_dim 자동 감지 실패, 기본값 768 사용")
        audio_dim = 768

    # 감지된 차원으로 모델 생성
    model = CrossAttentionFusionOrdinal(audio_dim=audio_dim).to(device)
    model.load_state_dict(state_dict)
    print("✅ state_dict 로드 완료")

model.eval()
print(f"✅ 모델 준비 완료 (audio_dim={audio_dim})\n")

# 전역 변수로 저장 (임베딩 함수에서 사용)
MODEL_AUDIO_DIM = audio_dim

# =========================================================
# 5. 텍스트 및 오디오 임베딩 함수
# =========================================================
def extract_text_from_json(json_path):

    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        utterances = data.get("utterances", [])
        texts = [u.get("text", "").strip() for u in utterances if isinstance(u, dict)]
        return " ".join([t for t in texts if t])
    except Exception as e:
        print(f"❌ JSON 로드 실패: {json_path}")
        print(f"   에러: {e}")
        return ""

def get_audio_emb(path, include_features=True):

    try:
        # 1. Data2Vec 임베딩 (768차원)
        y, sr = librosa.load(path, sr=16000)
        y = normalize(y)
        inputs = audio_processor(y, sampling_rate=sr, return_tensors="pt", padding=True)
        with torch.no_grad():
            out = audio_model(**{k: v.to(device) for k, v in inputs.items()})
        emb = out.last_hidden_state.mean(dim=1).cpu().numpy().flatten()

        if include_features:
            # 2. 오디오 피처 5개 추출
            duration = librosa.get_duration(y=y, sr=sr)

            # 무성구간 제외한 발화 비율
            non_silent = librosa.effects.split(y, top_db=25)
            voiced_time = sum((end - start) for start, end in non_silent) / sr
            speech_ratio = voiced_time / duration if duration > 0 else 0

            # 평균 에너지
            energy_mean = np.mean(y ** 2)

            # 피크 진폭
            peak_amp = np.max(np.abs(y))

            # RMS 및 Peak-to-RMS
            rms = np.sqrt(energy_mean)
            peak_to_rms = peak_amp / rms if rms > 0 else 0

            # 피처 배열 생성
            feats = np.array([duration, speech_ratio, energy_mean, peak_amp, peak_to_rms])

            # 3. 임베딩 + 피처 결합 (773차원)
            full_emb = np.concatenate([emb, feats])
        else:
            # 피처 없이 768차원만
            full_emb = emb

        # 4. L2 정규화
        return full_emb / (np.linalg.norm(full_emb) + 1e-12)

    except Exception as e:
        print(f"❌ 오디오 임베딩 추출 실패: {path}")
        print(f"   에러: {e}")
        raise

def get_text_emb(text):
    """텍스트 임베딩 추출 (768차원)"""
    try:
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
        return emb / (np.linalg.norm(emb) + 1e-12)

    except Exception as e:
        print(f"❌ 텍스트 임베딩 추출 실패")
        print(f"   에러: {e}")
        raise

# =========================================================
# 6. 추론 함수
# =========================================================
label_map = {0: "하", 1: "중", 2: "상"}

def predict_urgency_from_pair(json_path, wav_path, verbose=True):

    # 파일 존재 확인
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"JSON 파일을 찾을 수 없습니다: {json_path}")
    if not os.path.exists(wav_path):
        raise FileNotFoundError(f"WAV 파일을 찾을 수 없습니다: {wav_path}")

    # 텍스트 추출
    text = extract_text_from_json(json_path)
    if not text.strip():
        raise ValueError(f"⚠️ {json_path}에 유효한 텍스트가 없습니다.")

    # 모델이 기대하는 차원에 맞춰 임베딩 추출
    include_features = (MODEL_AUDIO_DIM == 773)

    if verbose:
        if include_features:
            print("🎵 오디오 임베딩 추출 중 (773차원: 임베딩 + 피처)...")
        else:
            print("🎵 오디오 임베딩 추출 중 (768차원: 임베딩만)...")

    audio_emb = get_audio_emb(wav_path, include_features=include_features)

    if verbose:
        print("📝 텍스트 임베딩 추출 중...")
    text_emb = get_text_emb(text)

    # 텐서 변환
    a = torch.FloatTensor(audio_emb).unsqueeze(0).to(device)
    t = torch.FloatTensor(text_emb).unsqueeze(0).to(device)

    if verbose:
        print(f"🔍 오디오 임베딩 shape: {a.shape}")
        print(f"🔍 텍스트 임베딩 shape: {t.shape}")

    # 추론
    with torch.no_grad():
        logits = model(a, t)
        pred = logits.argmax(1).item()
        probs = torch.softmax(logits, dim=-1).cpu().numpy().flatten()

    return label_map[pred], probs, text

# =========================================================
# 7. 예시 실행
# =========================================================
print("=" * 60)
print("🚀 추론 시작")
print("=" * 60)

json_path = "YOUR_JSON"
wav_path  = "YOUR_WAV"

try:
    label, probs, text = predict_urgency_from_pair(json_path, wav_path, verbose=True)

    print("\n" + "=" * 60)
    print("📊 추론 결과")
    print("=" * 60)
    print(f"🗣 텍스트: {text[:200]}{'...' if len(text) > 200 else ''}")
    print(f"\n🎯 예측 결과: {label}")
    print(f"📊 클래스 확률:")
    print(f"   - 하: {probs[0]:.4f} ({probs[0]*100:.2f}%)")
    print(f"   - 중: {probs[1]:.4f} ({probs[1]*100:.2f}%)")
    print(f"   - 상: {probs[2]:.4f} ({probs[2]*100:.2f}%)")
    print("=" * 60)

except Exception as e:
    print("\n" + "=" * 60)
    print(f"❌ 추론 실패: {e}")
    print("=" * 60)
    import traceback
    traceback.print_exc()
