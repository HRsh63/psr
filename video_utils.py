"""Video / audio utilities for FAKE FACE, REAL RIOT."""

import os
import re
import shutil
import subprocess
from pathlib import Path
from typing import Dict, Any, List, Tuple

import numpy as np

try:
    import imageio_ffmpeg
    FFMPEG_PATH = imageio_ffmpeg.get_ffmpeg_exe()
    FFMPEG_AVAILABLE = True
except Exception:
    FFMPEG_PATH = shutil.which("ffmpeg") or None
    FFMPEG_AVAILABLE = FFMPEG_PATH is not None

try:
    import cv2
    HAS_CV2 = True
except Exception:
    HAS_CV2 = False

from scipy.io import wavfile

import ffrr_config as config


def _run_ffmpeg(args: List[str]) -> Tuple[bool, str]:
    if not FFMPEG_AVAILABLE:
        return False, "FFmpeg not available"
    cmd = [FFMPEG_PATH, "-y"] + args
    proc = subprocess.run(cmd, capture_output=True, text=True)
    return proc.returncode == 0, proc.stderr


def get_video_metadata(video_path: str) -> Dict[str, Any]:
    """Parse ffmpeg -i stderr for metadata (imageio-ffmpeg ships no ffprobe)."""
    if not FFMPEG_AVAILABLE:
        raise RuntimeError("FFmpeg not available. Install imageio-ffmpeg.")
    proc = subprocess.run(
        [FFMPEG_PATH, "-i", str(video_path)],
        capture_output=True, text=True
    )
    stderr = proc.stderr or ""
    meta: Dict[str, Any] = {
        "duration": None,
        "width": None,
        "height": None,
        "fps": None,
        "video_codec": None,
        "audio": False,
        "audio_codec": None,
        "audio_sample_rate": None,
        "audio_channels": None,
        "bit_rate": None,
        "file_size_mb": os.path.getsize(video_path) / (1024 * 1024),
        "format_ok": True,
    }
    if "Invalid data found" in stderr or "No such file" in stderr or "could not find" in stderr.lower():
        meta["format_ok"] = False
        return meta
    for line in stderr.splitlines():
        line = line.strip()
        if line.startswith("Duration:"):
            for p in line.split(","):
                p = p.strip()
                if p.startswith("Duration:"):
                    t = p.replace("Duration:", "").strip()
                    try:
                        h, m, s = t.split(":")
                        meta["duration"] = float(h) * 3600 + float(m) * 60 + float(s)
                    except Exception:
                        pass
                elif p.startswith("bitrate:"):
                    meta["bit_rate"] = p.replace("bitrate:", "").strip()
        elif "Video:" in line:
            tail = line.split("Video:")[1]
            parts = [x.strip() for x in tail.split(",")]
            if parts:
                meta["video_codec"] = parts[0].split()[0]
            for tok in line.replace(",", " ").split():
                if re.match(r"^\d+x\d+$", tok):
                    try:
                        w, h = tok.split("x")
                        meta["width"] = int(w)
                        meta["height"] = int(h)
                    except Exception:
                        pass
                if tok.endswith("fps") and tok[:-3].replace(".", "").isdigit():
                    meta["fps"] = float(tok[:-3])
                if tok.endswith("tbr") and tok[:-3].replace(".", "").isdigit():
                    if meta["fps"] is None:
                        meta["fps"] = float(tok[:-3])
        elif "Audio:" in line:
            meta["audio"] = True
            tail = line.split("Audio:")[1]
            parts = [x.strip() for x in tail.split(",")]
            if parts:
                meta["audio_codec"] = parts[0].split()[0]
            for tok in line.replace(",", " ").split():
                if tok.endswith("Hz"):
                    try:
                        meta["audio_sample_rate"] = int(tok[:-2])
                    except Exception:
                        pass
                if tok in ("mono", "stereo"):
                    meta["audio_channels"] = tok
    return meta


def validate_upload(file_path: str, file_size_bytes: int, ext: str) -> Tuple[bool, str]:
    ext_lower = ext.lower()
    if ext_lower not in config.SUPPORTED_FORMATS:
        return False, f"Unsupported format '{ext}'. Supported: {', '.join(config.SUPPORTED_FORMATS)}"
    if file_size_bytes > config.MAX_UPLOAD_MB * 1024 * 1024:
        return False, f"File too large. Maximum: {config.MAX_UPLOAD_MB} MB"
    try:
        meta = get_video_metadata(file_path)
    except Exception as e:
        return False, f"Failed to read video metadata: {e}"
    if not meta.get("format_ok", False):
        return False, "Video appears corrupt or unreadable by FFmpeg."
    if meta.get("duration") and meta["duration"] > config.MAX_DURATION_SEC:
        return False, f"Video too long. Max: {config.MAX_DURATION_SEC}s; this video: {meta['duration']:.1f}s"
    return True, ""


def extract_audio(video_path: str, out_wav: str, sr: int = config.AUDIO_SAMPLE_RATE) -> Tuple[bool, str]:
    return _run_ffmpeg(["-i", str(video_path), "-vn", "-ac", "1", "-ar", str(sr), out_wav])


def extract_frames(video_path: str, out_dir: str, fps: int = config.FRAME_SAMPLE_FPS) -> Tuple[bool, str, List[str]]:
    os.makedirs(out_dir, exist_ok=True)
    pattern = os.path.join(out_dir, "frame_%05d.jpg")
    ok, err = _run_ffmpeg(["-i", str(video_path), "-vf", f"fps={fps}", "-q:v", "2", pattern])
    files = sorted(Path(out_dir).glob("frame_*.jpg")) if ok else []
    return ok, err, [str(f) for f in files]


def read_wav(path: str) -> Tuple[int, np.ndarray]:
    sr, samples = wavfile.read(path)
    if samples.dtype == np.int16:
        samples = samples.astype(np.float32) / 32768.0
    elif samples.dtype == np.int32:
        samples = samples.astype(np.float32) / 2147483648.0
    elif samples.dtype == np.uint8:
        samples = (samples.astype(np.float32) - 128) / 128.0
    elif samples.dtype == np.float64:
        samples = samples.astype(np.float32)
    else:
        samples = samples.astype(np.float32)
    if samples.ndim > 1:
        samples = samples.mean(axis=1)
    return sr, samples


def recompress_video(video_path: str, out_path: str, crf: int = config.COMPRESSION_CRF) -> Tuple[bool, str]:
    return _run_ffmpeg([
        "-i", str(video_path),
        "-c:v", "libx264", "-crf", str(crf), "-preset", "veryfast",
        "-c:a", "aac", "-b:a", "96k",
        out_path
    ])


def cleanup_path(path: str):
    if not path:
        return
    try:
        if os.path.isdir(path):
            shutil.rmtree(path, ignore_errors=True)
        elif os.path.exists(path):
            os.remove(path)
    except Exception:
        pass