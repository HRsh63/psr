"""FAKE FACE, REAL RIOT - configuration constants."""

from pathlib import Path

# --- Paths ---
BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"
DEMO_VIDEOS_DIR = BASE_DIR / "demo_videos"
TEMP_DIR = BASE_DIR / "temp"
for _d in (MODELS_DIR, DEMO_VIDEOS_DIR, TEMP_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# --- Upload constraints ---
SUPPORTED_FORMATS = [".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v"]
MAX_DURATION_SEC = 60
MAX_UPLOAD_MB = 200

# --- Audio ---
AUDIO_SAMPLE_RATE = 16000
AUDIO_ANALYSIS_WINDOW_SEC = 0.04
AUDIO_HOP_SEC = 0.02
N_FFT = 1024

# --- Video / frames ---
FRAME_SAMPLE_FPS = 10

# --- Risk thresholds (0-100) ---
LOW_RISK_MAX = 35
SUSPICIOUS_MAX = 65

# --- Fusion weights (must sum to 1.0) ---
FUSION_WEIGHTS = {
    "lip_sync": 0.45,
    "audio": 0.40,
    "compression": 0.15,
}

# --- Audio anomaly z-score thresholds ---
AUDIO_FLUX_ZSCORE = 3.0
AUDIO_RMS_ZSCORE = 3.5
AUDIO_ZCR_ZSCORE = 3.5
AUDIO_MFCC_ZSCORE = 3.0

# --- Compression ---
COMPRESSION_GENERATIONS = 3
COMPRESSION_CRF = 30

# --- Confidence bounds ---
MIN_CONFIDENCE = 0.25
MAX_CONFIDENCE = 0.95
