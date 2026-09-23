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

    cascades = _load_face_cascades()

    motions: List[float] = []
    face_counts: List[int] = []
    face_areas: List[float] = []
    face_boxes: List[Optional[Tuple[int, int, int, int]]] = []

    prev_mouth = None
    prev_face = None

    for fp in frame_paths:
        img = cv2.imread(
            str(fp),
            cv2.IMREAD_GRAYSCALE,
        )

        if img is None:
            motions.append(0.0)
            face_counts.append(0)
            face_areas.append(0.0)
            face_boxes.append(None)
            prev_mouth = None
            prev_face = None
            continue

        face, mouth_corners = _detect_face(
            img,
            cascades,
        )

        # Center-crop fallback for videos with large letterbox borders.
        if face is None:
            h, w = img.shape[:2]
            margin_x = int(0.08 * w)
            margin_y = int(0.04 * h)

            crop = img[
                margin_y:h - margin_y,
                margin_x:w - margin_x,
            ]

            crop_face, crop_mouth = _detect_face(
                crop,
                cascades,
            )

            if crop_face is not None:
                cx, cy, cw, ch = crop_face
                face = (
                    cx + margin_x,
                    cy + margin_y,
                    cw,
                    ch,
                )

                if crop_mouth is not None:
                    mouth_corners = (
                        crop_mouth
                        + np.asarray(
                            [margin_x, margin_y],
                            dtype=np.float64,
                        )
                    )

        if face is None:
            motions.append(0.0)
            face_counts.append(0)
            face_areas.append(0.0)
            face_boxes.append(None)
            prev_mouth = None
            prev_face = None
            continue

        face_counts.append(1)
        face_boxes.append(face)

        x, y, fw, fh = [
            int(v) for v in face
        ]

        face_areas.append(
            float(fw * fh)
            / max(
                1,
                img.shape[0] * img.shape[1],
            )
        )

        # ---------------- Mouth ROI ----------------
        if (
            mouth_corners is not None
            and mouth_corners.shape == (2, 2)
            and np.all(np.isfinite(mouth_corners))
        ):
            p1, p2 = mouth_corners
            mouth_center = (
                p1 + p2
            ) / 2.0

            mouth_width = max(
                8.0,
                float(
                    np.linalg.norm(
                        p2 - p1
                    )
                ),
            )

            roi_w = 2.0 * mouth_width
            roi_h = 1.25 * mouth_width

            mx1 = int(
                mouth_center[0]
                - roi_w / 2.0
            )
            my1 = int(
                mouth_center[1]
                - roi_h * 0.35
            )
            mx2 = int(
                mouth_center[0]
                + roi_w / 2.0
            )
            my2 = int(
                mouth_center[1]
                + roi_h * 0.65
            )

            # Keep ROI safely inside face.
            fx1 = x + int(0.08 * fw)
            fx2 = x + int(0.92 * fw)
            fy1 = y + int(0.48 * fh)
            fy2 = y + int(0.96 * fh)

            mx1 = max(fx1, mx1)
            mx2 = min(fx2, mx2)
            my1 = max(fy1, my1)
            my2 = min(fy2, my2)

        else:
            # Haar fallback: normalized lower-face ROI.
            mx1 = x + int(0.12 * fw)
            mx2 = x + int(0.88 * fw)
            my1 = y + int(0.55 * fh)
            my2 = y + int(0.95 * fh)

        if (
            mx2 <= mx1
            or my2 <= my1
        ):
            motions.append(0.0)
            prev_mouth = None
            prev_face = None
            continue

        mouth = img[
            my1:my2,
            mx1:mx2,
        ]

        if mouth.size == 0:
            motions.append(0.0)
            prev_mouth = None
            prev_face = None
            continue

        mouth = cv2.resize(
            mouth,
            (96, 64),
            interpolation=cv2.INTER_AREA,
        )
        mouth = cv2.GaussianBlur(
            mouth,
            (5, 5),
            0,
        )
        mouth = cv2.equalizeHist(mouth)

        # Normalize local illumination.
        mouth = (
            mouth.astype(np.float32)
            / 255.0
        )

        if prev_mouth is None:
            motion = 0.0
        else:
            raw = float(
                np.mean(
                    np.abs(
                        mouth - prev_mouth
                    )
                )
            )

            cur_edges = cv2.Canny(
                (mouth * 255).astype(
                    np.uint8
                ),
                30,
                100,
            )
            prev_edges = cv2.Canny(
                (prev_mouth * 255).astype(
                    np.uint8
                ),
                30,
                100,
            )

            edge_motion = float(
                np.mean(
                    np.abs(
                        cur_edges.astype(
                            np.float32
                        )
                        - prev_edges.astype(
                            np.float32
                        )
                    )
                )
                / 255.0
            )

            # Optical-flow magnitude adds sensitivity to actual local
            # mouth movement while suppressing pure brightness changes.
            try:
                flow = cv2.calcOpticalFlowFarneback(
                    (prev_mouth * 255).astype(
                        np.uint8
                    ),
                    (mouth * 255).astype(
                        np.uint8
                    ),
                    None,
                    0.5,
                    2,
                    15,
                    2,
                    5,
                    1.2,
                    0,
                )
                flow_mag = float(
                    np.mean(
                        np.sqrt(
                            flow[..., 0] ** 2
                            + flow[..., 1] ** 2
                        )
                    )
                )
                flow_component = min(
                    1.0,
                    flow_mag / 2.5,
                )
            except Exception:
                flow_component = 0.0

            motion = (
                0.50 * raw
                + 0.25 * edge_motion
                + 0.25 * flow_component
            )

        motions.append(
            float(motion)
        )
        prev_mouth = mouth
        prev_face = face

    return (
        motions,
        face_counts,
        face_areas,
        face_boxes,
    )


def analyze_lip_sync(
    frames_dir: str,
    audio_features: Optional[Dict],
) -> Dict[str, Any]:

    unavailable = {
        "score": 0.0,
        "status": "unavailable",
        "anomalies": [],
        "faces_detected": 0,
        "best_correlation": 0.0,
        "offset": 0.0,
        "available": False,
        "message": "Lip-sync analysis unavailable.",
    }

    if not getattr(vu, "HAS_CV2", False):
        unavailable["message"] = "OpenCV not available."
        return unavailable

    frame_paths = sorted(
        list(Path(frames_dir).glob("frame_*.jpg"))
        + list(Path(frames_dir).glob("frame_*.jpeg"))
        + list(Path(frames_dir).glob("frame_*.png"))
    )

    if len(frame_paths) < 5:
        unavailable["status"] = "no_frames"
        unavailable["message"] = (
            "Not enough frames for face/lip analysis."
        )
        return unavailable

    fps = _analysis_frame_fps()

    try:
        motions, face_counts, face_areas, face_boxes = (
            _mouth_motion_from_frames(
                frame_paths,
                fps,
            )
        )
    except Exception as exc:
        unavailable["status"] = "error"
        unavailable["message"] = (
            f"Face analysis failed: {exc}"
        )
        return unavailable

    max_faces = max(face_counts, default=0)
    faces_seen = sum(
        1 for c in face_counts if c > 0
    )

    if (
        max_faces == 0
        or faces_seen < min(3, len(frame_paths))
    ):
        return {
            "score": 0.0,
            "status": "no_face",
            "anomalies": [],
            "faces_detected": 0,
            "best_correlation": 0.0,
            "offset": 0.0,
            "available": False,
            "mouth_motion": motions,
            "mouth_time": (
                np.arange(len(motions)) / fps
            ).tolist(),
            "message": (
                "No stable face detected; "
                "lip-sync evidence not available."
            ),
            "face_detector": "YuNet" if _ensure_yunet() is not None else "Haar fallback",
        }

    if not audio_features or not audio_features.get("available"):
        return {
            "score": 0.0,
            "status": "no_audio",
            "anomalies": [],
            "faces_detected": max_faces,
            "best_correlation": 0.0,
            "offset": 0.0,
            "available": False,
            "mouth_motion": motions,
            "mouth_time": (
                np.arange(len(motions)) / fps
            ).tolist(),
            "message": (
                "No usable audio track for "
                "lip-sync comparison."
            ),
        }

    ar = np.asarray(
        audio_features.get(
            "features", {}
        ).get("rms", []),
        dtype=np.float64,
    )

    ar_t = np.asarray(
        audio_features.get(
            "features", {}
        ).get("rms_time", []),
        dtype=np.float64,
    )

    if len(ar) < 5:
        return {
            "score": 0.0,
            "status": "insufficient_audio",
            "anomalies": [],
            "faces_detected": max_faces,
            "best_correlation": 0.0,
            "offset": 0.0,
            "available": False,
            "mouth_motion": motions,
            "mouth_time": (
                np.arange(len(motions)) / fps
            ).tolist(),
            "message": (
                "Not enough audio features "
                "for lip-sync comparison."
            ),
        }

    mm = np.asarray(
        motions,
        dtype=np.float64,
    )

    mt = (
        np.arange(len(mm), dtype=np.float64)
        / max(fps, 1e-6)
    )

    if len(mm) >= 5:
        mm = scipy_signal.medfilt(
            mm,
            kernel_size=5,
        )

    mm = np.maximum(
        mm - np.percentile(mm, 20),
        0.0,
    )

    if np.max(mm) <= 1e-8:
        return {
            "score": 0.0,
            "status": "insufficient_mouth_motion",
            "anomalies": [],
            "faces_detected": max_faces,
            "best_correlation": 0.0,
            "offset": 0.0,
            "available": True,
            "mouth_motion": mm.tolist(),
            "mouth_time": mt.tolist(),
            "message": (
                "Face detected, but insufficient "
                "measurable mouth motion."
            ),
        }

    mm = mm / (
        np.percentile(mm, 95) + 1e-8
    )
    mm = np.clip(mm, 0, 1)

    ar = np.maximum(
        ar - np.percentile(ar, 20),
        0.0,
    )

    if np.max(ar) <= 1e-8:
        return {
            "score": 0.0,
            "status": "silent",
            "anomalies": [],
            "faces_detected": max_faces,
            "best_correlation": 0.0,
            "offset": 0.0,
            "available": True,
            "mouth_motion": mm.tolist(),
            "mouth_time": mt.tolist(),
            "message": (
                "Audio is effectively silent; "
                "lip-sync cannot be assessed."
            ),
        }

    ar = ar / (
        np.percentile(ar, 95) + 1e-8
    )
    ar = np.clip(ar, 0, 1)

    audio_at_mouth = _safe_interp(
        ar_t,
        ar,
        mt,
    )

    # Search a realistic AV offset window.
    max_lag_frames = max(
        1,
        int(round(0.55 * fps)),
    )

    correlations = []

    for lag in range(
        -max_lag_frames,
        max_lag_frames + 1,
    ):
        if lag < 0:
            a, b = (
                mm[:lag],
                audio_at_mouth[-lag:],
            )
        elif lag > 0:
            a, b = (
                mm[lag:],
                audio_at_mouth[:-lag],
            )
        else:
            a, b = (
                mm,
                audio_at_mouth,
            )

        correlations.append((
            lag,
            max(
                0.0,
                _normalised_corr(a, b),
            ),
        ))

    best_lag, best_corr = max(
        correlations,
        key=lambda x: x[1],
    )

    # Conservative mapping. Ordinary imperfect lip tracking should not
    # automatically become a deepfake verdict.
    if best_corr >= 0.55:
        risk = 0.0
        status = "normal"
    elif best_corr >= 0.35:
        risk = (
            (0.55 - best_corr)
            / 0.20
            * 35.0
        )
        status = "mild_inconsistency"
    else:
        risk = (
            35.0
            + (0.35 - best_corr)
            / 0.35
            * 65.0
        )
        status = "suspicious"

    anomalies: List[Dict[str, Any]] = []

    window = max(
        7,
        int(round(1.5 * fps)),
    )

    if len(mm) >= window:
        local = []
        local_t = []

        step = max(
            1,
            int(fps * 0.25),
        )

        for i in range(
            0,
            len(mm) - window + 1,
            step,
        ):
            c = max(
                0.0,
                _normalised_corr(
                    mm[i:i + window],
                    audio_at_mouth[
                        i:i + window
                    ],
                ),
            )

            local.append(c)
            local_t.append(
                float(
                    mt[
                        i + window // 2
                    ]
                )
            )

        if local:
            local = np.asarray(
                local,
                dtype=np.float64,
            )

            med = float(
                np.median(local)
            )

            threshold = (
                med
                - max(
                    0.12,
                    0.75 * float(
                        np.std(local)
                    ),
                )
            )

            bad = (
                (local < 0.22)
                & (local < threshold)
            )

            for idx in np.flatnonzero(bad):
                idx = int(idx)

                anomalies.append({
                    "timestamp": local_t[idx],
                    "type": "lip_sync",
                    "description": (
                        "Possible AV sync inconsistency "
                        f"near {local_t[idx]:.2f}s"
                    ),
                    "severity": float(
                        np.clip(
                            (0.35 - local[idx])
                            / 0.35,
                            0,
                            1,
                        )
                    ),
                })

            # Merge adjacent windows.
            compact: List[Dict[str, Any]] = []

            for anomaly in anomalies:
                if (
                    not compact
                    or anomaly["timestamp"]
                    - compact[-1]["timestamp"]
                    > 0.6
                ):
                    compact.append(anomaly)
                elif (
                    anomaly["severity"]
                    > compact[-1]["severity"]
                ):
                    compact[-1] = anomaly

            anomalies = compact[:12]

    if anomalies:
        risk = min(
            100.0,
            risk + min(
                25.0,
                len(anomalies) * 4.0,
            ),
        )

    return {
        "score": float(
            np.clip(risk, 0, 100)
        ),
        "status": status,
        "anomalies": anomalies,
        "faces_detected": max_faces,
        "best_correlation": float(
            best_corr
        ),
        "offset": float(
            best_lag / max(fps, 1e-6)
        ),
        "available": True,
        "face_detector": "YuNet" if _ensure_yunet() is not None else "Haar fallback",
        "mouth_motion": mm.tolist(),
        "mouth_time": mt.tolist(),
    }


# ---------------------------------------------------------------------------
# COMPRESSION ROBUSTNESS
# ---------------------------------------------------------------------------

def analyze_compression(
    video_path: str,
    base_lip_score: float,
    base_audio_score: float,
    progress: Optional[Callable] = None,
) -> Dict[str, Any]:
    """Measure stability after deterministic re-encoding.

    This is only a secondary signal. Stable compression behavior is NOT proof
    of manipulation.
    """

    generations: List[Dict[str, Any]] = []
    temp_files: List[str] = []
    current = str(video_path)

    n_gen = int(
        _cfg(
            "COMPRESSION_GENERATIONS",
            2,
        )
    )

    if n_gen <= 0:
        return {
            "score": 0.0,
            "status": "not_run",
            "generations": 0,
            "stability": 0.0,
            "avg_lip_diff": 0.0,
            "avg_audio_diff": 0.0,
        }

    temp_dir = Path(
        _cfg(
            "TEMP_DIR",
            Path("."),
        )
    )
    temp_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    try:
        for gen in range(
            1,
            n_gen + 1,
        ):
            if progress:
                progress(
                    f"Compression check "
                    f"{gen}/{n_gen}"
                )

            out_path = str(
                temp_dir
                / f"recomp_gen{gen}.mp4"
            )

            crf = int(
                _cfg(
                    "COMPRESSION_CRF",
                    28,
                )
            )

            ok, _ = vu.recompress_video(
                current,
                out_path,
                crf=crf,
            )

            if (
                not ok
                or not os.path.exists(out_path)
            ):
                break

            temp_files.append(out_path)

            frames_dir = str(
                temp_dir
                / f"gen{gen}_frames"
            )

            audio_out = str(
                temp_dir
                / f"gen{gen}_audio.wav"
            )

            ok_f, _, frame_files = (
                vu.extract_frames(
                    out_path,
                    frames_dir,
                    _analysis_frame_fps(),
                )
            )

            ok_a, _ = vu.extract_audio(
                out_path,
                audio_out,
                int(
                    _cfg(
                        "AUDIO_SAMPLE_RATE",
                        16000,
                    )
                ),
            )

            gen_audio = 0.0
            gen_lip = 0.0
            af = None

            if (
                ok_a
                and os.path.exists(audio_out)
            ):
                af = analyze_audio(
                    audio_out
                )
                gen_audio = float(
                    af.get(
                        "score",
                        0.0,
                    )
                )

            if ok_f and frame_files:
                lf = analyze_lip_sync(
                    frames_dir,
                    af,
                )
                gen_lip = float(
                    lf.get(
                        "score",
                        0.0,
                    )
                )

            generations.append({
                "generation": gen,
                "lip_sync_score": gen_lip,
                "audio_score": gen_audio,
            })

            vu.cleanup_path(
                frames_dir
            )
            vu.cleanup_path(
                audio_out
            )

            current = out_path

    finally:
        for f in temp_files:
            vu.cleanup_path(f)

    if not generations:
        return {
            "score": 0.0,
            "status": "failed",
            "generations": 0,
            "stability": 0.0,
            "avg_lip_diff": 0.0,
            "avg_audio_diff": 0.0,
        }

    lip_diffs = [
        abs(
            g["lip_sync_score"]
            - base_lip_score
        )
        for g in generations
    ]

    audio_diffs = [
        abs(
            g["audio_score"]
            - base_audio_score
        )
        for g in generations
    ]

    avg_lip_diff = float(
        np.mean(lip_diffs)
    )
    avg_audio_diff = float(
        np.mean(audio_diffs)
    )

    stability = 1.0 - min(
        1.0,
        (
            avg_lip_diff
            + avg_audio_diff
        ) / 60.0,
    )

    # Compression is secondary. It can amplify an already-measured signal,
    # but it cannot create a high risk score by itself.
    existing_signal = max(
        base_lip_score,
        base_audio_score,
    ) / 100.0

    comp_risk = (
        100.0
        * existing_signal
        * stability
        * 0.45
    )

    if comp_risk < 15:
        status = "normal"
    elif comp_risk < 40:
        status = "stable_secondary_signal"
    else:
        status = "persistent_signal"

    return {
        "score": float(
            np.clip(
                comp_risk,
                0,
                100,
            )
        ),
        "status": status,
        "generations": len(
            generations
        ),
        "stability": float(
            stability
        ),
        "avg_lip_diff": avg_lip_diff,
        "avg_audio_diff": avg_audio_diff,
    }


# ---------------------------------------------------------------------------
# VISUAL ANALYSIS
# ---------------------------------------------------------------------------

def analyze_visual_frames(
    frames_dir: str,
) -> Dict[str, Any]:

    frame_paths = sorted(
        Path(frames_dir).glob(
            "frame_*.jpg"
        )
    )

    if (
        not frame_paths
        or not getattr(
            vu,
            "HAS_CV2",
            False,
        )
    ):
        return {
            "frames_analyzed": 0,
            "visual_anomalies": [],
            "visual_risk": 0.0,
        }

    import cv2

    anomalies: List[Dict[str, Any]] = []
    prev = None
    sharpness_values = []
    diff_values = []

    for idx, fp in enumerate(
        frame_paths
    ):
        img = cv2.imread(
            str(fp),
            cv2.IMREAD_GRAYSCALE,
        )

        if img is None:
            continue

        img_small = cv2.resize(
            img,
            (256, 144),
            interpolation=cv2.INTER_AREA,
        )

        blur = float(
            cv2.Laplacian(
                img_small,
                cv2.CV_64F,
            ).var()
        )

        sharpness_values.append(
            blur
        )

        if prev is not None:
            diff_values.append(
                float(
                    np.mean(
                        cv2.absdiff(
                            img_small,
                            prev,
                        )
                    )
                )
            )

        prev = img_small

    if not sharpness_values:
        return {
            "frames_analyzed": 0,
            "visual_anomalies": [],
            "visual_risk": 0.0,
        }

    # Distribution-aware temporal anomaly detection.
    if len(diff_values) >= 4:
        dz = _robust_z(
            np.asarray(
                diff_values
            )
        )

        for j in np.flatnonzero(
            dz > 8.0
        ):
            frame_idx = int(j + 1)

            ts = (
                frame_idx
                / max(
                    _analysis_frame_fps(),
                    1e-6,
                )
            )

            anomalies.append({
                "frame": frame_idx,
                "timestamp": float(ts),
                "type": "temporal_inconsistency",
                "severity": float(
                    np.clip(
                        (dz[j] - 4.0)
                        / 5.0,
                        0,
                        1,
                    )
                ),
                "description": (
                    "Unusually large frame "
                    f"transition near {ts:.2f}s"
                ),
            })

    # Detect unusual sharpness drops relative to this video's own baseline.
    sz = _robust_z(
        np.asarray(
            sharpness_values
        )
    )

    for j in np.flatnonzero(
        sz < -8.0
    ):
        ts = (
            int(j)
            / max(
                _analysis_frame_fps(),
                1e-6,
            )
        )

        anomalies.append({
            "frame": int(j),
            "timestamp": float(ts),
            "type": "sharpness_drop",
            "severity": float(
                np.clip(
                    (-sz[j] - 4.0)
                    / 5.0,
                    0,
                    1,
                )
            ),
            "description": (
                "Sharpness dropped unusually "
                f"near {ts:.2f}s"
            ),
        })

    anomalies.sort(
        key=lambda a: a["severity"],
        reverse=True,
    )

    anomalies = anomalies[:12]

    risk = 0.0

    if anomalies:
        density = (
            len(anomalies)
            / max(
                len(frame_paths),
                1,
            )
        )

        strongest = max(
            a["severity"]
            for a in anomalies
        )

        risk = 100.0 * (
            0.35
            * min(
                1.0,
                density / 0.18,
            )
            + 0.65
            * (strongest ** 1.35)
        )

    return {
        "frames_analyzed": len(
            sharpness_values
        ),
        "visual_anomalies": anomalies,
        "visual_risk": float(
            np.clip(
                risk,
                0,
                100,
            )
        ),
        "sharpness_mean": float(
            np.mean(
                sharpness_values
            )
        ),
        "frame_difference_mean": (
            float(np.mean(diff_values))
            if diff_values
            else 0.0
        ),
    }


# ---------------------------------------------------------------------------
# MAIN PIPELINE
# ---------------------------------------------------------------------------

def analyze_single_pass(
    video_path: str,
    progress: Optional[Callable] = None,
) -> Dict[str, Any]:

    if progress:
        progress(
            "Loading video & extracting assets"
        )

    temp_dir = Path(
        _cfg(
            "TEMP_DIR",
            Path("."),
        )
    )

    temp_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    stem = Path(
        video_path
    ).stem

    safe_stem = "".join(
        c
        if c.isalnum() or c in "-_"
        else "_"
        for c in stem
    )

    audio_out = str(
        temp_dir
        / f"audio_{safe_stem}.wav"
    )

    ok_a, _ = vu.extract_audio(
        video_path,
        audio_out,
        int(
            _cfg(
                "AUDIO_SAMPLE_RATE",
                16000,
            )
        ),
    )

    audio_result: Dict[str, Any] = {
        "score": 0.0,
        "status": "unavailable",
        "anomalies": [],
        "anomaly_count": 0,
        "available": False,
        "silent": True,
    }

    if (
        ok_a
        and os.path.exists(audio_out)
    ):
        if progress:
            progress(
                "Analyzing audio"
            )

        audio_result = analyze_audio(
            audio_out
        )

    if progress:
        progress(
            "Extracting frames"
        )

    frames_dir = str(
        temp_dir
        / f"frames_{safe_stem}"
    )

    ok_f, _, frame_files = (
        vu.extract_frames(
            video_path,
            frames_dir,
            _analysis_frame_fps(),
        )
    )

    lip_result: Dict[str, Any] = {
        "score": 0.0,
        "status": "unavailable",
        "anomalies": [],
        "faces_detected": 0,
        "best_correlation": 0.0,
        "offset": 0.0,
        "available": False,
    }

    if ok_f and frame_files:
        if progress:
            progress(
                "Face & lip analysis"
            )

        lip_result = analyze_lip_sync(
            frames_dir,
            audio_result,
        )

    if progress:
        progress(
            "Visual frame analysis"
        )

    visual_result = (
        analyze_visual_frames(
            frames_dir
        )
        if ok_f
        else {
            "frames_analyzed": 0,
            "visual_anomalies": [],
            "visual_risk": 0.0,
        }
    )

    return {
        "lip_sync": lip_result,
        "audio": audio_result,
        "visual": visual_result,
        "audio_path": (
            audio_out
            if (
                ok_a
                and os.path.exists(
                    audio_out
                )
            )
            else None
        ),
        "frames_dir": (
            frames_dir
            if ok_f
            else None
        ),
    }


def _fuse_scores(
    lip: Dict[str, Any],
    audio: Dict[str, Any],
    comp: Dict[str, Any],
    visual: Dict[str, Any],
) -> Tuple[float, float]:
    """Fuse only available evidence.

    Missing signals are removed from the denominator instead of being
    treated as suspicious evidence.
    """

    components = []

    if lip.get("available"):
        components.append((
            "lip",
            float(
                lip.get(
                    "score",
                    0.0,
                )
            ),
            0.40,
        ))

    if (
        audio.get("available")
        and not audio.get("silent")
    ):
        components.append((
            "audio",
            float(
                audio.get(
                    "score",
                    0.0,
                )
            ),
            0.30,
        ))

    if (
        visual.get(
            "frames_analyzed",
            0,
        ) >= 4
    ):
        components.append((
            "visual",
            float(
                visual.get(
                    "visual_risk",
                    0.0,
                )
            ),
            0.25,
        ))

    if (
        comp.get(
            "generations",
            0,
        ) > 0
    ):
        components.append((
            "compression",
            float(
                comp.get(
                    "score",
                    0.0,
                )
            ),
            0.05,
        ))

    if not components:
        return 0.0, 0.20

    weight_sum = sum(
        w
        for _, _, w in components
    )

    final = (
        sum(
            score * weight
            for _, score, weight
            in components
        )
        / weight_sum
    )

    # Confidence means evidence coverage + agreement. It is NOT the fake
    # probability and does not increase just because a score is high.
    scores = np.asarray(
        [
            s
            for _, s, _
            in components
        ],
        dtype=np.float64,
    )

    agreement = 1.0 - min(
        1.0,
        float(np.std(scores))
        / 45.0,
    )

    coverage = min(
        1.0,
        weight_sum / 0.90,
    )

    confidence = (
        0.35
        + 0.45 * coverage
        + 0.20 * agreement
    )

    return (
        float(
            np.clip(
                final,
                0,
                100,
            )
        ),
        float(
            np.clip(
                confidence,
                0.20,
                0.98,
            )
        ),
    )


def analyze_video(
    video_path: str,
    progress: Optional[Callable] = None,
) -> Dict[str, Any]:
    """Main deterministic forensic pipeline."""

    if (
        not video_path
        or not os.path.exists(video_path)
    ):
        raise FileNotFoundError(
            f"Video not found: {video_path}"
        )

    pass1 = analyze_single_pass(
        video_path,
        progress,
    )

    lip = pass1["lip_sync"]
    audio = pass1["audio"]
    visual = pass1["visual"]

    if progress:
        progress(
            "Compression robustness analysis"
        )

    comp = analyze_compression(
        video_path,
        float(
            lip.get(
                "score",
                0.0,
            )
        ),
        float(
            audio.get(
                "score",
                0.0,
            )
        ),
        progress,
    )

    if progress:
        progress(
            "Fusing evidence"
        )

    final, confidence = _fuse_scores(
        lip,
        audio,
        comp,
        visual,
    )

    low_max = float(
        _cfg(
            "LOW_RISK_MAX",
            30.0,
        )
    )

    suspicious_max = float(
        _cfg(
            "SUSPICIOUS_MAX",
            65.0,
        )
    )

    if final < low_max:
        verdict = "AUTHENTIC"
    elif final < suspicious_max:
        verdict = "SUSPICIOUS"
    else:
        verdict = "MANIPULATED"

    evidence: List[str] = []

    # Evidence is generated from actual measurements only.
    if lip.get("available"):
        corr = float(
            lip.get(
                "best_correlation",
                0.0,
            )
        )

        if lip.get("anomalies"):
            evidence.append(
                "Possible audio-video sync "
                "inconsistency detected "
                f"({len(lip['anomalies'])} "
                "interval(s))."
            )
        elif corr >= 0.55:
            evidence.append(
                "Lip and audio timing showed "
                "consistent correlation "
                f"(r={corr:.2f})."
            )
        else:
            evidence.append(
                "Lip/audio correlation was "
                "inconclusive "
                f"(r={corr:.2f}); no strong "
                "sync anomaly was isolated."
            )
    else:
        evidence.append(
            lip.get(
                "message",
                "Lip-sync evidence unavailable.",
            )
        )

    if not audio.get("available"):
        evidence.append(
            "No usable audio track was available."
        )
    elif audio.get("silent"):
        evidence.append(
            "Audio track appears silent; "
            "acoustic continuity was not scored."
        )
    elif audio.get("anomalies"):
        first = audio["anomalies"][0]

        evidence.append(
            f"{len(audio['anomalies'])} acoustic "
            "discontinuity event(s) detected; "
            "strongest near "
            f"{float(first['timestamp']):.2f}s."
        )
    else:
        evidence.append(
            "No significant acoustic "
            "discontinuity detected."
        )

    if visual.get(
        "visual_anomalies"
    ):
        strongest = max(
            visual["visual_anomalies"],
            key=lambda a: a.get(
                "severity",
                0,
            ),
        )

        evidence.append(
            f"{len(visual['visual_anomalies'])} "
            "visual/temporal outlier(s) "
            "detected; strongest near "
            f"{float(strongest['timestamp']):.2f}s."
        )
    else:
        evidence.append(
            "No major visual frame outliers detected."
        )

    if comp.get(
        "generations",
        0,
    ) > 0:
        stability = float(
            comp.get(
                "stability",
                0.0,
            )
        )

        if comp.get(
            "score",
            0.0,
        ) >= 25:
            evidence.append(
                "A measured forensic signal "
                "persisted across "
                f"{comp['generations']} "
                "recompression generation(s) "
                f"(stability {stability:.2f})."
            )
        else:
            evidence.append(
                "No abnormal compression "
                "persistence detected "
                f"(stability {stability:.2f})."
            )
    else:
        evidence.append(
            "Compression robustness check "
            "could not be completed."
        )

    try:
        meta = vu.get_video_metadata(
            video_path
        )
    except Exception:
        meta = {}

    metadata = {
        "duration": float(
            meta.get(
                "duration",
                audio.get(
                    "duration",
                    0.0,
                ),
            )
            or 0.0
        ),
        "fps": float(
            meta.get(
                "fps",
                0.0,
            )
            or 0.0
        ),
        "resolution": (
            f"{meta.get('width', 0)}x"
            f"{meta.get('height', 0)}"
        ),
        "frames_analyzed": int(
            visual.get(
                "frames_analyzed",
                0,
            )
        ),
    }

    if progress:
        progress(
            "Report generated"
        )

    return {
        "verdict": verdict,
        "overall_score": float(final),
        "confidence": float(
            confidence * 100.0
        ),

        "lip_sync": {
            "score": float(
                lip.get(
                    "score",
                    0.0,
                )
            ),
            "status": lip.get(
                "status",
                "unavailable",
            ),
            "faces_detected": int(
                lip.get(
                    "faces_detected",
                    0,
                )
            ),
            "face_count": int(
                lip.get(
                    "faces_detected",
                    0,
                )
            ),
            "best_correlation": float(
                lip.get(
                    "best_correlation",
                    0.0,
                )
            ),
            "best_corr": float(
                lip.get(
                    "best_correlation",
                    0.0,
                )
            ),
            "offset": float(
                lip.get(
                    "offset",
                    0.0,
                )
            ),
            "face_detector": lip.get(
                "face_detector",
                "unknown",
            ),
            "mouth_motion": lip.get(
                "mouth_motion",
                [],
            ),
            "mouth_time": lip.get(
                "mouth_time",
                [],
            ),
            "anomalies": lip.get(
                "anomalies",
                [],
            ),
        },

        "audio_continuity": {
            "score": float(
                audio.get(
                    "score",
                    0.0,
                )
            ),
            "status": audio.get(
                "status",
                "unavailable",
            ),
            "available": bool(
                audio.get(
                    "available",
                    False,
                )
            ),
            "silent": bool(
                audio.get(
                    "silent",
                    False,
                )
            ),
            "duration": float(
                audio.get(
                    "duration",
                    0.0,
                )
            ),
            "anomalies": audio.get(
                "anomalies",
                [],
            ),
            "anomaly_count": int(
                len(
                    audio.get(
                        "anomalies",
                        [],
                    )
                )
            ),
            "features": audio.get("features"),
            "spectrogram": audio.get("spectrogram"),
            "waveform": audio.get("waveform"),
            "sample_rate": int(
                audio.get("sample_rate", 0)
            ),
        },

        "compression_robustness": {
            "score": float(
                comp.get(
                    "score",
                    0.0,
                )
            ),
            "status": comp.get(
                "status",
                "normal",
            ),
            "generations": int(
                comp.get(
                    "generations",
                    0,
                )
            ),
            "stability": float(
                comp.get(
                    "stability",
                    0.0,
                )
            ),
        },

        "visual_analysis": {
            "frames_analyzed": int(
                visual.get(
                    "frames_analyzed",
                    0,
                )
            ),
            "visual_anomalies": visual.get(
                "visual_anomalies",
                [],
            ),
        },

        "evidence": evidence,
        "metadata": metadata,

        # Backward compatibility with the existing UI.
        "risk_level": verdict,
        "risk_score": float(final),
        "explanation": evidence,
    }


def cleanup_session(
    result: Dict[str, Any]
):
    vu.cleanup_path(
        result.get(
            "audio_path"
        )
    )

    vu.cleanup_path(
        result.get(
            "frames_dir"
        )
    )

    temp_dir = Path(
        _cfg(
            "TEMP_DIR",
            Path("."),
        )
    )

    for pattern in (
        "recomp_gen*.mp4",
        "gen*_frames",
        "gen*_audio.wav",
        "audio_*.wav",
        "frames_*",
        "upload_*",
    ):
        for p in temp_dir.glob(pattern):
            try:
                vu.cleanup_path(
                    str(p)
                )
            except Exception:
                pass
