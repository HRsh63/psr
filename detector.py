"""FAKE FACE, REAL RIOT - deterministic forensic detector.

This module intentionally uses explainable, media-dependent heuristics.
It does not claim to replace a trained deepfake classifier. The important
contract is that the same input produces the same output and different media
properties produce different evidence.
"""

from __future__ import annotations

import os
import math
import hashlib
from pathlib import Path
from typing import Dict, Any, List, Optional, Callable, Tuple

import numpy as np
from scipy import signal as scipy_signal
from scipy.fftpack import dct

import ffrr_config as config
import video_utils as vu


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

def _cfg(name: str, default: Any) -> Any:
    return getattr(config, name, default)


def _analysis_frame_fps() -> float:
    """FPS used consistently by extraction, lip-sync and visual timestamps."""
    return max(8.0, float(_cfg("FRAME_SAMPLE_FPS", 4.0)))


def _robust_z(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    if x.size == 0:
        return x
    med = float(np.median(x))
    mad = float(np.median(np.abs(x - med)))
    # Use both robust MAD and ordinary spread. A tiny MAD alone can
    # massively exaggerate normal speech/frame-to-frame variation.
    scale = max(
        1.4826 * mad,
        float(np.std(x)) * 0.50,
        float(np.percentile(np.abs(x - med), 75)) * 0.75,
        1e-8,
    )
    return (x - med) / scale


def _normalised_corr(a: np.ndarray, b: np.ndarray) -> float:
    n = min(len(a), len(b))
    if n < 4:
        return 0.0
    a = np.asarray(a[:n], dtype=np.float64)
    b = np.asarray(b[:n], dtype=np.float64)
    a = a - np.mean(a)
    b = b - np.mean(b)
    den = math.sqrt(float(np.sum(a * a) * np.sum(b * b)))
    if den <= 1e-12:
        return 0.0
    return float(np.sum(a * b) / den)


def _safe_interp(times: np.ndarray, values: np.ndarray,
                 target: np.ndarray) -> np.ndarray:
    if len(times) == 0 or len(values) == 0:
        return np.zeros_like(target, dtype=np.float64)
    if len(times) == 1:
        return np.full_like(target, float(values[0]), dtype=np.float64)
    order = np.argsort(times)
    t = np.asarray(times)[order]
    v = np.asarray(values)[order]
    return np.interp(target, t, v)


def _stft_mag(y: np.ndarray, sr: int, n_fft: int, hop: int):
    n_fft = max(128, int(n_fft))
    hop = max(1, min(int(hop), n_fft - 1))
    if len(y) < n_fft:
        y = np.pad(y, (0, n_fft - len(y)))
    f, t, z = scipy_signal.stft(
        y,
        fs=sr,
        nperseg=n_fft,
        noverlap=n_fft - hop,
        boundary=None,
        padded=False,
    )
    return f, t, np.abs(z)


def _mel_filterbank(n_mels: int, n_fft: int, sr: int) -> np.ndarray:
    f_max = sr / 2.0
    mel_pts = np.linspace(
        0.0, 1127.0 * np.log1p(f_max / 700.0), n_mels + 2
    )
    hz_pts = 700.0 * (np.exp(mel_pts / 1127.0) - 1.0)
    bins = np.floor((n_fft + 1) * hz_pts / sr).astype(int)
    n_freq = n_fft // 2 + 1
    fb = np.zeros((n_mels, n_freq), dtype=np.float32)

    for m in range(1, n_mels + 1):
        lo = max(0, min(n_freq - 1, bins[m - 1]))
        mid = max(lo + 1, min(n_freq, bins[m]))
        hi = max(mid + 1, min(n_freq, bins[m + 1]))

        for k in range(lo, mid):
            fb[m - 1, k] = (k - lo) / max(1, mid - lo)
        for k in range(mid, hi):
            fb[m - 1, k] = (hi - k) / max(1, hi - mid)

    return fb


def _mfcc(y: np.ndarray, sr: int, n_fft: int, hop: int,
          n_mels: int = 20, n_mfcc: int = 13):
    f, t, mag = _stft_mag(y, sr, n_fft, hop)
    fb = _mel_filterbank(n_mels, n_fft, sr)
    log_mel = np.log(fb @ (mag + 1e-10) + 1e-10)
    mfcc = dct(log_mel, axis=0, type=2, norm="ortho")[:n_mfcc]
    return f, t, mfcc


# ---------------------------------------------------------------------------
# AUDIO
# ---------------------------------------------------------------------------

def analyze_audio(audio_path: str) -> Dict[str, Any]:
    base = {
        "score": 0.0,
        "status": "unavailable",
        "anomalies": [],
        "anomaly_count": 0,
        "features": None,
        "spectrogram": None,
        "waveform": None,
        "sample_rate": 0,
        "duration": 0.0,
        "silent": True,
        "available": False,
    }

    if not audio_path or not os.path.exists(audio_path):
        return base

    try:
        sr, y = vu.read_wav(audio_path)
        sr = int(sr)
        y = np.asarray(y, dtype=np.float64).reshape(-1)
    except Exception as exc:
        base["status"] = "error"
        base["message"] = str(exc)
        return base

    if sr <= 0 or len(y) == 0:
        base["status"] = "empty"
        return base

    y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)

    peak = float(np.max(np.abs(y))) if len(y) else 0.0
    if peak > 1.5:
        y = y / peak

    duration = len(y) / sr

    base.update({
        "sample_rate": sr,
        "duration": float(duration),
        "available": True,
    })

    n_fft = int(_cfg("N_FFT", 1024))
    hop = max(1, int(float(_cfg("AUDIO_HOP_SEC", 0.025)) * sr))
    frame_len = max(
        hop * 2,
        int(float(_cfg("AUDIO_ANALYSIS_WINDOW_SEC", 0.05)) * sr)
    )

    if len(y) < max(sr * 0.25, frame_len):
        base["status"] = "too_short"
        base["silent"] = bool(peak < 0.01)
        return base

    # Frame-level time-domain features.
    starts = np.arange(
        0, max(1, len(y) - frame_len + 1), hop
    )

    rms = np.empty(len(starts), dtype=np.float64)
    zcr = np.empty(len(starts), dtype=np.float64)

    for j, start in enumerate(starts):
        seg = y[start:start + frame_len]
        rms[j] = math.sqrt(float(np.mean(seg * seg)) + 1e-12)
        signs = np.signbit(seg)
        zcr[j] = (
            float(np.mean(signs[1:] != signs[:-1]))
            if len(seg) > 1 else 0.0
        )

    rms_time = (starts + frame_len / 2) / sr

    f, t, mag = _stft_mag(y, sr, n_fft, hop)
    mag_sum = np.sum(mag, axis=0) + 1e-12
    centroid = (f[:, None] * mag).sum(axis=0) / mag_sum

    if mag.shape[1] > 1:
        diff = np.diff(mag, axis=1)
        flux = np.sum(np.maximum(diff, 0.0) ** 2, axis=0)
        flux = np.concatenate(([0.0], flux))
    else:
        flux = np.zeros(mag.shape[1], dtype=np.float64)

    _, _, mfcc = _mfcc(y, sr, n_fft, hop)

    mfcc_dist = np.zeros(mfcc.shape[1], dtype=np.float64)
    if mfcc.shape[1] > 1:
        mfcc_dist[1:] = np.linalg.norm(
            np.diff(mfcc, axis=1), axis=0
        )

    rms_stft = _safe_interp(rms_time, rms, t)
    zcr_stft = _safe_interp(rms_time, zcr, t)

    # Relative changes are much more useful than absolute speech values.
    log_rms = np.log(rms_stft + 1e-5)
    rms_jump = np.abs(np.diff(log_rms, prepend=log_rms[:1]))
    centroid_jump = np.abs(
        np.diff(centroid, prepend=centroid[:1])
    )
    zcr_jump = np.abs(
        np.diff(zcr_stft, prepend=zcr_stft[:1])
    )

    zr = _robust_z(rms_jump)
    zc = _robust_z(centroid_jump)
    zz = _robust_z(zcr_jump)
    zf = _robust_z(flux)
    zm = _robust_z(mfcc_dist)

    # Normal speech has lots of acoustic variation. A discontinuity is only
    # considered meaningful when multiple independent features jump together.
    candidates: List[Tuple[int, float]] = []

    for i in range(1, len(t)):
        values = [zr[i], zc[i], zz[i], zf[i], zm[i]]
        # Normal speech/music produces many spectral changes. Require
        # either several independent signals to agree strongly, or a
        # genuinely extreme jump in multiple signals.
        votes = sum(v > 7.0 for v in values)

        ordered = sorted(values, reverse=True)
        strength = ordered[0]
        second = ordered[1]
        third = ordered[2]

        if (
            votes >= 3
            or (
                strength > 11.0
                and second > 8.0
                and third > 6.5
            )
        ):
            candidates.append((
                i,
                float(
                    np.clip(
                        (strength - 6.0) / 12.0,
                        0.0,
                        1.0,
                    )
                )
            ))

    # Cluster nearby candidate frames so one edit does not become dozens of
    # separate anomalies.
    clusters: List[List[Tuple[int, float]]] = []
    frame_dt = (
        float(t[1] - t[0]) if len(t) > 1 else 0.025
    )
    min_gap = max(1, int(round(0.30 / max(frame_dt, 1e-3))))

    for item in candidates:
        if (
            not clusters
            or item[0] - clusters[-1][-1][0] > min_gap
        ):
            clusters.append([item])
        else:
            clusters[-1].append(item)

    anomalies: List[Dict[str, Any]] = []

    for cluster in clusters:
        idx = max(cluster, key=lambda x: x[1])[0]
        severity = max(x[1] for x in cluster)

        anomalies.append({
            "timestamp": float(t[idx]),
            "type": "acoustic_discontinuity",
            "description": f"Acoustic discontinuity near {t[idx]:.2f}s",
            "severity": float(severity),
        })

    # Duration-normalized score.
    # A handful of natural speech transitions should stay low. High scores
    # require both density AND strong independent evidence.
    anomaly_density = len(anomalies) / max(duration, 1.0)
    strongest = max(
        (a["severity"] for a in anomalies),
        default=0.0
    )

    density_component = min(
        1.0,
        anomaly_density / 0.80,
    )
    strength_component = strongest ** 1.35

    risk = 100.0 * (
        0.45 * density_component
        + 0.55 * strength_component
    )

    # Do not call ordinary acoustic variation suspicious just because one
    # candidate exists.
    if len(anomalies) <= 1 and strongest < 0.75:
        risk *= 0.25

    risk = float(np.clip(risk, 0.0, 100.0))

    if not anomalies:
        risk = 0.0

    silent = bool(float(np.percentile(rms, 95)) < 0.008)

    if silent:
        risk = 0.0
        status = "silent"
    elif len(anomalies) == 0:
        status = "normal"
    elif risk < 35:
        status = "mild_anomalies"
    else:
        status = "suspicious"

    step = max(1, len(y) // 4000)
    wave_ds = y[::step]
    wave_t = np.arange(len(wave_ds)) * step / sr

    return {
        "score": float(np.clip(risk, 0, 100)),
        "status": status,
        "anomalies": anomalies[:20],
        "anomaly_count": len(anomalies),
        "features": {
            "rms": rms.tolist(),
            "rms_time": rms_time.tolist(),
            "flux": flux.tolist(),
            "flux_time": t.tolist(),
            "spec_centroid": centroid.tolist(),
            "zcr": zcr.tolist(),
            "mfcc_dist": mfcc_dist.tolist(),
        },
        "spectrogram": {
            "f": f.tolist(),
            "t": t.tolist(),
            "mag_db": (
                20 * np.log10(mag + 1e-10)
            ).tolist(),
        },
        "waveform": {
            "y": wave_ds.tolist(),
            "t": wave_t.tolist(),
        },
        "sample_rate": sr,
        "duration": float(duration),
        "silent": silent,
        "available": True,
    }


# ---------------------------------------------------------------------------
# LIP SYNC
# ---------------------------------------------------------------------------

_YUNET = None
_YUNET_TRIED = False

_YUNET_URL = (
    "https://huggingface.co/pollen-robotics/"
    "face_detection_yunet_2023mar/resolve/main/"
    "face_detection_yunet_2023mar.onnx?download=true"
)

_YUNET_SHA256 = (
    "8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4"
)


def _yunet_model_path() -> Path:
    root = Path(__file__).resolve().parent
    model_dir = root / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    return model_dir / "face_detection_yunet_2023mar.onnx"


def _ensure_yunet():
    """Load YuNet; download the small MIT model once if it is absent.

    If the environment is offline, return None and let the Haar fallback
    handle detection instead of pretending a face was found.
    """
    global _YUNET, _YUNET_TRIED

    if _YUNET_TRIED:
        return _YUNET

    _YUNET_TRIED = True

    try:
        import cv2
        if not hasattr(cv2, "FaceDetectorYN"):
            return None

        model_path = _yunet_model_path()

        if not model_path.exists() or model_path.stat().st_size < 100_000:
            tmp = model_path.with_suffix(".download")
            mirrors = [
                _YUNET_URL,
                (
                    "https://github.com/opencv/opencv_zoo/raw/"
                    "refs/heads/main/models/face_detection_yunet/"
                    "face_detection_yunet_2023mar.onnx"
                ),
            ]

            downloaded = False
            try:
                import urllib.request

                for url in mirrors:
                    try:
                        req = urllib.request.Request(
                            url,
                            headers={"User-Agent": "FAKE-FACE-REAL-RIOT/1.0"},
                        )
                        with urllib.request.urlopen(req, timeout=60) as response:
                            with open(tmp, "wb") as fh:
                                while True:
                                    chunk = response.read(1024 * 1024)
                                    if not chunk:
                                        break
                                    fh.write(chunk)

                        digest = hashlib.sha256(
                            tmp.read_bytes()
                        ).hexdigest()

                        if digest == _YUNET_SHA256:
                            downloaded = True
                            break

                        tmp.unlink(missing_ok=True)
                    except Exception:
                        tmp.unlink(missing_ok=True)

                if not downloaded:
                    return None

                tmp.replace(model_path)
            except Exception:
                try:
                    tmp.unlink(missing_ok=True)
                except Exception:
                    pass
                return None

        detector = cv2.FaceDetectorYN.create(
            str(model_path),
            "",
            (320, 320),
            0.55,
            0.30,
            5000,
        )

        _YUNET = detector
        return _YUNET

    except Exception:
        return None


def _load_face_cascades():
    import cv2

    paths = [
        os.path.join(
            cv2.data.haarcascades,
            "haarcascade_frontalface_default.xml",
        ),
        os.path.join(
            cv2.data.haarcascades,
            "haarcascade_frontalface_alt2.xml",
        ),
        os.path.join(
            cv2.data.haarcascades,
            "haarcascade_frontalface_alt.xml",
        ),
        os.path.join(
            cv2.data.haarcascades,
            "haarcascade_profileface.xml",
        ),
    ]

    cascades = []
    for path in paths:
        if os.path.exists(path):
            c = cv2.CascadeClassifier(path)
            if not c.empty():
                cascades.append(c)

    return cascades


def _detect_face(gray, cascades):
    """Return (bbox, mouth_corners).

    YuNet is preferred because it directly provides the mouth-corner
    landmarks. Haar remains a deterministic offline fallback.
    """
    import cv2

    if gray is None or gray.size == 0:
        return None, None

    h, w = gray.shape[:2]

    # ------------------------- YuNet -------------------------
    yunet = _ensure_yunet()

    if yunet is not None:
        try:
            bgr = cv2.cvtColor(
                gray,
                cv2.COLOR_GRAY2BGR,
            )

            yunet.setInputSize((w, h))
            _, faces = yunet.detect(bgr)

            if faces is not None and len(faces):
                candidates = []

                for row in faces:
                    x, y, fw, fh = [
                        float(v) for v in row[:4]
                    ]
                    score = float(row[-1])

                    if score < 0.45:
                        continue

                    x = max(
                        0,
                        min(int(round(x)), w - 1),
                    )
                    y = max(
                        0,
                        min(int(round(y)), h - 1),
                    )
                    fw = max(
                        1,
                        min(int(round(fw)), w - x),
                    )
                    fh = max(
                        1,
                        min(int(round(fh)), h - y),
                    )

                    # YuNet row layout:
                    # bbox + right eye + left eye + nose +
                    # right mouth corner + left mouth corner + score
                    if len(row) >= 14:
                        mouth = np.asarray(
                            row[9:13],
                            dtype=np.float64,
                        ).reshape(2, 2)
                    else:
                        mouth = None

                    area = fw * fh

                    candidates.append(
                        (
                            area,
                            score,
                            (x, y, fw, fh),
                            mouth,
                        )
                    )

                if candidates:
                    _, _, bbox, mouth = max(
                        candidates,
                        key=lambda item: (
                            item[0],
                            item[1],
                        ),
                    )
                    return bbox, mouth

        except Exception:
            pass

    # ------------------------- Haar fallback -------------------------
    if not cascades:
        return None, None

    scale = 2.0
    enlarged = cv2.resize(
        gray,
        None,
        fx=scale,
        fy=scale,
        interpolation=cv2.INTER_CUBIC,
    )

    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8),
    )

    variants = [
        enlarged,
        clahe.apply(enlarged),
        cv2.equalizeHist(enlarged),
    ]

    min_side = max(
        24,
        int(min(w, h) * 0.045 * scale),
    )

    candidates = []

    for image in variants:
        for cascade in cascades:
            try:
                found = cascade.detectMultiScale(
                    image,
                    scaleFactor=1.04,
                    minNeighbors=2,
                    minSize=(min_side, min_side),
                    maxSize=(
                        int(w * 0.95 * scale),
                        int(h * 0.95 * scale),
                    ),
                )
            except Exception:
                found = ()

            for x, y, fw, fh in found:
                x = int(x / scale)
                y = int(y / scale)
                fw = int(fw / scale)
                fh = int(fh / scale)

                x = max(0, min(x, w - 1))
                y = max(0, min(y, h - 1))
                fw = max(1, min(fw, w - x))
                fh = max(1, min(fh, h - y))

                area_ratio = (
                    fw * fh
                    / max(1, w * h)
                )

                if (
                    area_ratio >= 0.003
                    and area_ratio <= 0.90
                    and 0.35 <= fw / max(fh, 1) <= 2.2
                ):
                    candidates.append(
                        (fw * fh, (x, y, fw, fh))
                    )

    if not candidates:
        return None, None

    _, bbox = max(
        candidates,
        key=lambda item: item[0],
    )

    return bbox, None


def _mouth_motion_from_frames(
    frame_paths: List[Path],
    fps: float,
):
    import cv2

    cascades 
