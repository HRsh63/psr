"""FAKE FACE, REAL RIOT - forensic detector."""

import os
import shutil
from pathlib import Path
from typing import Dict, Any, List, Optional, Callable

import numpy as np
from scipy import signal as scipy_signal
from scipy.fftpack import dct

import ffrr_config as config
import video_utils as vu


# ================= AUDIO =================

def _zscore(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    mu = np.median(x)
    mad = np.median(np.abs(x - mu)) + 1e-10
    sd = mad * 1.4826 + 1e-10
    return (x - mu) / sd


def _stft_mag(y: np.ndarray, sr: int, n_fft: int, hop: int):
    f, t, Z = scipy_signal.stft(
        y, fs=sr, nperseg=n_fft, noverlap=n_fft - hop, boundary=None
    )
    return f, t, np.abs(Z)


def _mel_filterbank(n_mels: int, n_fft: int, sr: int) -> np.ndarray:
    f_max = sr / 2
    mel_pts = np.linspace(0, 1127 * np.log1p(f_max / 700), n_mels + 2)
    hz_pts = 700 * (np.exp(mel_pts / 1127) - 1)
    bins = np.floor((n_fft + 1) * hz_pts / sr).astype(int)
    n_freq = n_fft // 2 + 1
    fb = np.zeros((n_mels, n_freq), dtype=np.float32)
    for m in range(1, n_mels + 1):
        lo, mid, hi = bins[m-1], bins[m], bins[m+1]
        for k in range(lo, mid):
            if mid > lo:
                fb[m-1, k] = (k - lo) / (mid - lo)
        for k in range(mid, hi):
            if hi > mid:
                fb[m-1, k] = (hi - k) / (hi - mid)
    return fb


def _mfcc(y: np.ndarray, sr: int, n_fft: int, hop: int, n_mels: int = 20, n_mfcc: int = 13):
    f, t, mag = _stft_mag(y, sr, n_fft, hop)
    mag = mag + 1e-10
    fb = _mel_filterbank(n_mels, n_fft, sr)
    mel_spec = fb @ mag
    log_mel = np.log(mel_spec + 1e-10)
    mfcc = dct(log_mel, axis=0, type=2, norm='ortho')[:n_mfcc]
    return f, t, mfcc


def analyze_audio(audio_path: str) -> Dict[str, Any]:
    if not audio_path or not os.path.exists(audio_path):
        return {"score": 0.0, "anomalies": [], "features": None,
                "spectrogram": None, "waveform": None,
                "sample_rate": 0, "duration": 0.0,
                "silent": True, "available": False}

    try:
        sr, y = vu.read_wav(audio_path)
    except Exception as e:
        return {"score": 0.0, "anomalies": [], "features": None,
                "spectrogram": None, "waveform": None,
                "sample_rate": 0, "duration": 0.0,
                "silent": True, "available": False, "error": str(e)}

    duration = len(y) / sr
    if len(y) < sr:
        return {"score": 0.0, "anomalies": [], "features": None,
                "spectrogram": None, "waveform": None,
                "sample_rate": int(sr), "duration": float(duration),
                "silent": True, "available": True}

    n_fft = config.N_FFT
    hop = max(1, int(config.AUDIO_HOP_SEC * sr))
    frame_len = max(1, int(config.AUDIO_ANALYSIS_WINDOW_SEC * sr))
    n_frames = max(0, (len(y) - frame_len) // hop + 1)

    rms = np.zeros(n_frames)
    zcr = np.zeros(n_frames)
    for i in range(n_frames):
        seg = y[i * hop: i * hop + frame_len]
        rms[i] = float(np.sqrt(np.mean(seg ** 2) + 1e-10))
        signs = np.sign(seg)
        zcr[i] = float(np.sum(np.abs(np.diff(signs)) > 0) / len(seg))

    rms_time = np.arange(n_frames) * hop / sr

    f, t, mag = _stft_mag(y, sr, n_fft, hop)

    flux = np.zeros(mag.shape[1])
    if mag.shape[1] > 1:
        diff = np.diff(mag, axis=1)
        flux[1:] = np.sum(np.maximum(diff, 0) ** 2, axis=0)
    flux_max = float(np.max(flux)) + 1e-10
    flux = flux / flux_max

    spec_centroid = (f[:, None] * mag).sum(axis=0) / (mag.sum(axis=0) + 1e-10)

    _, _, mfcc = _mfcc(y, sr, n_fft, hop)
    mfcc_delta = np.diff(mfcc, axis=1, prepend=mfcc[:, :1])
    mfcc_dist = np.linalg.norm(mfcc_delta, axis=0)
    md_max = float(np.max(mfcc_dist)) + 1e-10
    mfcc_dist = mfcc_dist / md_max

    flux_z = _zscore(flux)
    rms_z = _zscore(np.abs(np.diff(rms, prepend=rms[0])))
    zcr_z = _zscore(np.abs(np.diff(zcr, prepend=zcr[0])))
    mfcc_z = _zscore(mfcc_dist)

    n_stft = len(t)

    def to_stft(arr):
        if len(rms_time) == 0:
            return np.zeros(n_stft)
        return np.interp(t, rms_time, arr)

    flux_flags = flux_z > config.AUDIO_FLUX_ZSCORE
    rms_flags = to_stft(rms_z) > config.AUDIO_RMS_ZSCORE
    zcr_flags = to_stft(zcr_z) > config.AUDIO_ZCR_ZSCORE
    mfcc_flags = mfcc_z > config.AUDIO_MFCC_ZSCORE

    anomaly_frames = []
    for i in range(1, n_stft):
        cnt = int(sum([flux_flags[i], rms_flags[i], zcr_flags[i], mfcc_flags[i]]))
        if cnt >= 2:
            anomaly_frames.append((i, cnt))

    anomalies: List[Dict[str, Any]] = []
    if anomaly_frames:
        run_start = anomaly_frames[0][0]
        run_end = anomaly_frames[0][0]
        run_cnt = anomaly_frames[0][1]
        for idx, cnt in anomaly_frames[1:]:
            if idx == run_end + 1:
                run_end = idx
                run_cnt = max(run_cnt, cnt)
            else:
                t_mid = float(t[(run_start + run_end) // 2])
                anomalies.append({
                    "timestamp": t_mid,
                    "type": "audio",
                    "description": f"Possible acoustic discontinuity near {t_mid:.2f}s",
                    "intensity": float(run_cnt / 4.0),
                })
                run_start, run_end, run_cnt = idx, idx, cnt
        t_mid = float(t[(run_start + run_end) // 2])
        anomalies.append({
            "timestamp": t_mid,
            "type": "audio",
            "description": f"Possible acoustic discontinuity near {t_mid:.2f}s",
            "intensity": float(run_cnt / 4.0),
        })

    if not anomalies:
        max_flux_z = float(np.max(flux_z)) if len(flux_z) else 0.0
        score = max(0.0, max_flux_z * 5.0)
    else:
        intensity_sum = sum(a["intensity"] for a in anomalies)
        score = 40.0 + min(55.0, intensity_sum * 15.0 + len(anomalies) * 5.0)

    step = max(1, len(y) // 4000)
    wave_ds = y[::step]
    wave_t = np.arange(len(wave_ds)) * step / sr

    return {
        "score": float(np.clip(score, 0, 100)),
        "anomalies": anomalies,
        "features": {
            "rms": rms.tolist(),
            "rms_time": rms_time.tolist(),
            "flux": flux.tolist(),
            "flux_time": t.tolist(),
            "spec_centroid": spec_centroid.tolist(),
            "zcr": zcr.tolist(),
            "mfcc_dist": mfcc_dist.tolist(),
        },
        "spectrogram": {
            "f": f.tolist(),
            "t": t.tolist(),
            "mag_db": (20 * np.log10(mag + 1e-10)).tolist(),
        },
        "waveform": {
            "y": wave_ds.tolist(),
            "t": wave_t.tolist(),
        },
        "sample_rate": int(sr),
        "duration": float(duration),
        "silent": bool(float(np.max(rms)) < 0.01),
        "available": True,
    }


# ================= LIP SYNC =================

def analyze_lip_sync(frames_dir: str, audio_features: Optional[Dict]) -> Dict[str, Any]:
    if not vu.HAS_CV2:
        return {"score": 0.0, "anomalies": [], "face_count": 0, "available": False,
                "best_corr": 0.0, "mouth_motion": [], "mouth_time": [],
                "message": "OpenCV not available — lip-sync analysis skipped."}

    frame_paths = sorted(Path(frames_dir).glob("frame_*.jpg"))
    if not frame_paths:
        return {"score": 0.0, "anomalies": [], "face_count": 0, "available": False,
                "best_corr": 0.0, "mouth_motion": [], "mouth_time": [],
                "message": "No frames extracted."}

    import cv2
    cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )

    mouth_motion: List[float] = []
    face_counts: List[int] = []

    for fp in frame_paths:
        img = cv2.imread(str(fp), cv2.IMREAD_GRAYSCALE)
        if img is None:
            mouth_motion.append(0.0)
            face_counts.append(0)
            continue
        img_eq = cv2.equalizeHist(img)
        try:
            faces = cascade.detectMultiScale(img_eq, 1.2, 5, minSize=(30, 30))
        except Exception:
            faces = []
        face_counts.append(len(faces))
        if len(faces) == 0:
            mouth_motion.append(0.0)
            continue
        largest = max(faces, key=lambda b: b[2] * b[3])
        x, y, w, h = largest
        mx = x
        my = y + int(h * 0.7)
        mw = w
        mh = max(5, int(h * 0.3))
        mouth = img[my:my + mh, mx:mx + mw]
        if mouth.size == 0:
            mouth_motion.append(0.0)
            continue
        motion = float(np.std(mouth.astype(np.float32)))
        mouth_motion.append(motion)

    fps = config.FRAME_SAMPLE_FPS
    mouth_time = (np.arange(len(mouth_motion)) / fps).tolist()

    if not mouth_motion or max(mouth_motion) < 1:
        return {"score": 0.0, "anomalies": [], "face_count": max(face_counts) if face_counts else 0,
                "available": True, "best_corr": 0.0,
                "mouth_motion": mouth_motion, "mouth_time": mouth_time,
                "message": "Insufficient face/mouth signal for lip-sync analysis."}

    mm = np.array(mouth_motion, dtype=np.float32)
    mm = mm - np.median(mm)
    mm = np.maximum(mm, 0)
    if mm.max() > 0:
        mm = mm / mm.max()

    if audio_features and audio_features.get("features"):
        ar = np.array(audio_features["features"]["rms"], dtype=np.float32)
        ar_t = np.array(audio_features["features"]["rms_time"], dtype=np.float32)
    else:
        ar = np.zeros(10, dtype=np.float32)
        ar_t = np.linspace(0, mouth_time[-1] if mouth_time else 1.0, 10)

    if len(ar) > 0 and ar.max() > 0:
        ar = ar / ar.max()

    audio_at_mouth = np.interp(mouth_time, ar_t, ar)

    best_corr = 0.0
    best_lag = 0
    if len(mm) >= 5 and len(audio_at_mouth) >= 5:
        mm_n = mm - np.mean(mm)
        aa_n = audio_at_mouth - np.mean(audio_at_mouth)
        denom = np.sqrt(np.sum(mm_n ** 2) * np.sum(aa_n ** 2)) + 1e-10
        max_lag = min(5, len(mm) // 4)
        for lag in range(-max_lag, max_lag + 1):
            if lag < 0:
                a, b = mm_n[:lag], aa_n[-lag:]
            elif lag > 0:
                a, b = mm_n[lag:], aa_n[:-lag]
            else:
                a, b = mm_n, aa_n
            if len(a) == 0 or len(b) == 0:
                continue
            c = abs(float(np.sum(a * b) / denom))
            if c > best_corr:
                best_corr = c
                best_lag = lag

    lip_score = (1.0 - best_corr) * 100.0

    anomalies: List[Dict[str, Any]] = []
    window = max(5, len(mm) // 4)
    if len(mm) >= window:
        local_corrs = []
        local_times = []
        for i in range(len(mm) - window + 1):
            a = mm[i:i + window] - np.mean(mm[i:i + window])
            b = audio_at_mouth[i:i + window] - np.mean(audio_at_mouth[i:i + window])
            d = np.sqrt(np.sum(a ** 2) * np.sum(b ** 2)) + 1e-10
            c = abs(float(np.sum(a * b) / d))
            local_corrs.append(c)
            local_times.append(mouth_time[i + window // 2])
        local_corrs = np.array(local_corrs)
        if len(local_corrs):
            threshold = max(0.1, float(np.mean(local_corrs) - 0.4 * np.std(local_corrs)))
            bad = local_corrs < threshold
            i = 0
            while i < len(bad):
                if bad[i]:
                    start = i
                    while i < len(bad) and bad[i]:
                        i += 1
                    t_s = float(local_times[start])
                    t_e = float(local_times[min(i - 1, len(local_times) - 1)])
                    anomalies.append({
                        "timestamp": float((t_s + t_e) / 2.0),
                        "type": "lip_sync",
                        "description": f"Possible audio-video synchronization inconsistency between {t_s:.1f}s and {t_e:.1f}s",
                    })
                else:
                    i += 1

    face_count = max(face_counts) if face_counts else 0
    face_note = ""
    if face_count == 0:
        face_note = "No face detected in any frame."
    elif face_count > 1:
        face_note = f"Multiple faces detected in some frames (max {face_count}); analyzing largest."

    return {
        "score": float(np.clip(lip_score, 0, 100)),
        "anomalies": anomalies,
        "face_count": int(face_count),
        "available": True,
        "best_corr": float(best_corr),
        "best_lag": int(best_lag),
        "mouth_motion": mouth_motion,
        "mouth_time": mouth_time,
        "face_note": face_note,
    }


# ================= COMPRESSION =================

def analyze_compression(video_path: str, base_lip_score: float, base_audio_score: float,
                        progress: Optional[Callable] = None) -> Dict[str, Any]:
    generations: List[Dict[str, Any]] = []
    current = str(video_path)
    temp_files: List[str] = []

    for gen in range(1, config.COMPRESSION_GENERATIONS + 1):
        if progress:
            progress(f"Compression generation {gen}/{config.COMPRESSION_GENERATIONS}")
        out_path = str(config.TEMP_DIR / f"recomp_gen{gen}.mp4")
        ok, _ = vu.recompress_video(current, out_path, crf=config.COMPRESSION_CRF)
        if not ok:
            break
        temp_files.append(out_path)

        frames_dir = str(config.TEMP_DIR / f"gen{gen}_frames")
        ok_f, _, frame_files = vu.extract_frames(out_path, frames_dir, config.FRAME_SAMPLE_FPS)
        audio_out = str(config.TEMP_DIR / f"gen{gen}_audio.wav")
        ok_a, _ = vu.extract_audio(out_path, audio_out, config.AUDIO_SAMPLE_RATE)

        gen_audio = 0.0
        af = None
        if ok_a and os.path.exists(audio_out):
            af = analyze_audio(audio_out)
            gen_audio = af["score"]

        gen_lip = 0.0
        if ok_f and frame_files:
            lf = analyze_lip_sync(frames_dir, af)
            gen_lip = lf["score"]

        generations.append({
            "generation": gen,
            "lip_sync_score": float(gen_lip),
            "audio_score": float(gen_audio),
            "risk_score": float(0.5 * gen_lip + 0.5 * gen_audio),
        })

        vu.cleanup_path(frames_dir)
        vu.cleanup_path(audio_out)
        current = out_path

    for f in temp_files:
        vu.cleanup_path(f)

    if not generations:
        return {"score": 50.0, "robustness": 0.0, "generations": [],
                "avg_lip_diff": 0.0, "avg_audio_diff": 0.0,
                "note": "Compression test failed."}

    lip_diffs = [abs(g["lip_sync_score"] - base_lip_score) for g in generations]
    audio_diffs = [abs(g["audio_score"] - base_audio_score) for g in generations]
    avg_lip_diff = float(np.mean(lip_diffs))
    avg_audio_diff = float(np.mean(audio_diffs))

    stability = 1.0 - min(1.0, (avg_lip_diff + avg_audio_diff) / 50.0)

    avg_gen_lip = float(np.mean([g["lip_sync_score"] for g in generations]))
    avg_gen_audio = float(np.mean([g["audio_score"] for g in generations]))

    if base_lip_score > 40 or base_audio_score > 40:
        if avg_gen_lip > 40 or avg_gen_audio > 40:
            comp_score = 70.0 + 25.0 * stability
        else:
            comp_score = 30.0
    else:
        comp_score = 20.0 + 30.0 * (1.0 - stability)

    return {
        "score": float(np.clip(comp_score, 0, 100)),
        "robustness": float(stability),
        "generations": generations,
        "avg_lip_diff": avg_lip_diff,
        "avg_audio_diff": avg_audio_diff,
    }


# ================= MAIN =================

def analyze_single_pass(video_path: str, progress: Optional[Callable] = None) -> Dict[str, Any]:
    if progress: progress("Loading video & extracting assets")

    audio_out = str(config.TEMP_DIR / f"audio_{os.path.basename(video_path)}.wav")
    ok_a, _ = vu.extract_audio(video_path, audio_out, config.AUDIO_SAMPLE_RATE)

    audio_result: Dict[str, Any] = {"score": 0.0, "anomalies": [],
                                     "available": False, "silent": True}
    if ok_a and os.path.exists(audio_out):
        if progress: progress("Analyzing audio")
        audio_result = analyze_audio(audio_out)

    if progress: progress("Extracting frames")
    frames_dir = str(config.TEMP_DIR / f"frames_{os.path.basename(video_path)}")
    ok_f, _, frame_files = vu.extract_frames(video_path, frames_dir, config.FRAME_SAMPLE_FPS)

    if progress: progress("Face & lip analysis")
    lip_result: Dict[str, Any] = {"score": 0.0, "anomalies": [], "available": False}
    if ok_f and frame_files:
        lip_result = analyze_lip_sync(frames_dir, audio_result)

    return {
        "lip_sync": lip_result,
        "audio": audio_result,
        "audio_path": audio_out if (ok_a and os.path.exists(audio_out)) else None,
        "frames_dir": frames_dir if ok_f else None,
    }


def analyze_video(video_path: str, progress: Optional[Callable] = None) -> Dict[str, Any]:
    """Main entry: run full forensic pipeline."""
    pass1 = analyze_single_pass(video_path, progress)
    lip = pass1["lip_sync"]
    audio = pass1["audio"]

    if progress: progress("Compression robustness analysis")
    comp = analyze_compression(video_path, lip.get("score", 0.0),
                                audio.get("score", 0.0), progress)

    if progress: progress("Fusing evidence")
    w = config.FUSION_WEIGHTS
    final = (
        w["lip_sync"] * lip.get("score", 0.0)
        + w["audio"] * audio.get("score", 0.0)
        + w["compression"] * comp["score"]
    )
    final = float(np.clip(final, 0, 100))

    if final < config.LOW_RISK_MAX:
        level = "LOW RISK"
    elif final < config.SUSPICIOUS_MAX:
        level = "SUSPICIOUS"
    else:
        level = "HIGH RISK"

    n_anomalies = len(lip.get("anomalies", [])) + len(audio.get("anomalies", []))
    confidence = min(config.MAX_CONFIDENCE,
                     config.MIN_CONFIDENCE + 0.05 * n_anomalies + 0.5 * (final / 100.0))
    if not lip.get("available"):
        confidence = max(0.20, confidence - 0.30)
    if not audio.get("available"):
        confidence = max(0.20, confidence - 0.30)

    timeline: List[Dict[str, Any]] = []
    for a in lip.get("anomalies", []):
        timeline.append({"timestamp": a["timestamp"], "type": "lip_sync",
                          "description": a["description"]})
    for a in audio.get("anomalies", []):
        timeline.append({"timestamp": a["timestamp"], "type": "audio",
                          "description": a["description"]})
    timeline.sort(key=lambda x: x["timestamp"])

    explanations: List[str] = []
    if not lip.get("available"):
        if lip.get("message"):
            explanations.append(lip["message"])
    elif lip.get("anomalies"):
        explanations.append(
            f"Possible audio-video synchronization inconsistency detected "
            f"({len(lip['anomalies'])} interval(s))."
        )
        for a in lip["anomalies"][:3]:
            explanations.append(a["description"])

    if not audio.get("available"):
        explanations.append("Audio track unavailable — audio analysis skipped.")
    elif audio.get("silent"):
        explanations.append("Audio track appears silent — audio discontinuity analysis not meaningful.")
    elif audio.get("anomalies"):
        explanations.append(
            f"Possible acoustic discontinuity detected ({len(audio['anomalies'])} point(s))."
        )
        for a in audio["anomalies"][:3]:
            explanations.append(a["description"])

    if comp.get("generations"):
        if comp.get("robustness", 0) > 0.7 and (lip.get("score", 0) > 40 or audio.get("score", 0) > 40):
            explanations.append("Suspicious signals remained stable across multiple recompression generations.")
        elif comp.get("robustness", 0) < 0.3:
            explanations.append("Forensic signals were unstable across recompression — interpret with caution.")

    if not explanations:
        explanations.append("No significant forensic anomalies detected in this analysis.")

    if progress: progress("Report generated")

    return {
        "risk_score": final,
        "risk_level": level,
        "confidence": float(confidence),
        "lip_sync": {
            "score": float(lip.get("score", 0.0)),
            "anomalies": lip.get("anomalies", []),
            "available": lip.get("available", False),
            "face_count": lip.get("face_count", 0),
            "best_corr": lip.get("best_corr", 0.0),
            "message": lip.get("message", ""),
            "mouth_motion": lip.get("mouth_motion", []),
            "mouth_time": lip.get("mouth_time", []),
        },
        "audio": {
            "score": float(audio.get("score", 0.0)),
            "anomalies": audio.get("anomalies", []),
            "available": audio.get("available", False),
            "silent": audio.get("silent", True),
            "features": audio.get("features"),
            "spectrogram": audio.get("spectrogram"),
            "waveform": audio.get("waveform"),
            "sample_rate": audio.get("sample_rate", 0),
            "duration": audio.get("duration", 0.0),
        },
        "compression": {
            "score": float(comp["score"]),
            "robustness": float(comp.get("robustness", 0.0)),
            "generations": comp.get("generations", []),
            "avg_lip_diff": float(comp.get("avg_lip_diff", 0.0)),
            "avg_audio_diff": float(comp.get("avg_audio_diff", 0.0)),
        },
        "timeline": timeline,
        "explanation": explanations,
        "audio_path": pass1.get("audio_path"),
        "frames_dir": pass1.get("frames_dir"),
    }


def cleanup_session(result: Dict[str, Any]):
    vu.cleanup_path(result.get("audio_path"))
    vu.cleanup_path(result.get("frames_dir"))
    for p in config.TEMP_DIR.glob("recomp_gen*.mp4"):
        vu.cleanup_path(str(p))
    for p in config.TEMP_DIR.glob("gen*_frames"):
        vu.cleanup_path(str(p))
    for p in config.TEMP_DIR.glob("gen*_audio.wav"):
        vu.cleanup_path(str(p))
    for p in config.TEMP_DIR.glob("audio_*.wav"):
        vu.cleanup_path(str(p))
    for p in config.TEMP_DIR.glob("frames_*"):
        vu.cleanup_path(str(p))
    for p in config.TEMP_DIR.glob("upload_*"):
        vu.cleanup_path(str(p))