"""FAKE FACE, REAL RIOT - Streamlit forensic dashboard."""
import os
import time
import hashlib
import subprocess
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import streamlit as st

import ffrr_config as config
import video_utils as vu
import detector



# ---------------------------------------------------------------------------
# Compression Lab helpers
# ---------------------------------------------------------------------------

def _video_id(data: bytes, filename: str) -> str:
    """Stable ID for the currently uploaded file."""
    return hashlib.sha256(
        data + filename.encode("utf-8", errors="ignore")
    ).hexdigest()[:16]


def _compress_generation(
    input_path: str,
    output_path: str,
    crf: int = 28,
    scale: float = 1.0,
) -> tuple[bool, str]:
    """Create one deterministic recompressed MP4 using the local FFmpeg.

    CRF is deliberately moderate. We keep resolution unless the user chooses
    otherwise, so the experiment isolates recompression rather than resizing.
    """
    try:
        import imageio_ffmpeg
        ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    except Exception as exc:
        return False, f"FFmpeg executable unavailable: {exc}"

    vf = []
    if scale != 1.0:
        vf.append(
            "scale=trunc(iw*{0}/2)*2:trunc(ih*{0}/2)*2".format(scale)
        )

    cmd = [
        ffmpeg,
        "-y",
        "-i", input_path,
        "-map", "0:v:0",
        "-map", "0:a?",
        "-c:v", "libx264",
        "-preset", "veryfast",
        "-crf", str(int(crf)),
        "-pix_fmt", "yuv420p",
        "-c:a", "aac",
        "-b:a", "128k",
    ]

    if vf:
        cmd += ["-vf", ",".join(vf)]

    cmd += [
        "-movflags", "+faststart",
        output_path,
    ]

    try:
        proc = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=180,
        )
    except subprocess.TimeoutExpired:
        return False, "Compression timed out after 180 seconds."
    except Exception as exc:
        return False, str(exc)

    if proc.returncode != 0 or not os.path.exists(output_path):
        err = proc.stderr[-1200:] if proc.stderr else "Unknown FFmpeg error"
        return False, err

    return True, "Compression completed."


def _file_size_mb(path: str) -> float:
    try:
        return os.path.getsize(path) / (1024 * 1024)
    except Exception:
        return 0.0


def _ensure_compression_state(video_id: str, original_path: str):
    """Reset the compression chain only when a new upload is detected."""
    if st.session_state.get("compression_video_id") != video_id:
        st.session_state.compression_video_id = video_id
        st.session_state.compression_chain = {
            0: original_path
        }
        st.session_state.compression_results = {}
        st.session_state.compression_errors = {}
        st.session_state.compression_crfs = {}


def _run_generation_analysis(generation: int):
    path = st.session_state.compression_chain.get(generation)
    if not path or not os.path.exists(path):
        st.error(f"Generation {generation} file is missing.")
        return

    progress_bar = st.progress(0.0)
    status = st.empty()

    def update(msg: str):
        status.write(msg)
        progress_bar.progress(
            min(
                0.98,
                progress_bar_value(msg),
            )
        )

    try:
        with st.spinner(
            f"Analyzing Generation {generation}..."
        ):
            result = detector.analyze_video(
                path,
                progress=update,
            )
        st.session_state.compression_results[generation] = result
        try:
            detector.cleanup_session(result)
        except Exception:
            pass
    except Exception as exc:
        st.session_state.compression_errors[generation] = str(exc)
        st.error(f"Generation {generation} analysis failed: {exc}")
    finally:
        progress_bar.progress(1.0)


def progress_bar_value(msg: str) -> float:
    key = msg.lower()
    mapping = [
        ("loading", 0.10),
        ("audio", 0.25),
        ("extracting", 0.40),
        ("face", 0.58),
        ("lip", 0.62),
        ("visual", 0.72),
        ("compression", 0.82),
        ("fusing", 0.92),
        ("report", 0.98),
    ]
    for word, value in mapping:
        if word in key:
            return value
    return 0.50


def _show_generation_result(generation: int, result: dict):
    verdict = str(
        result.get("verdict", "SUSPICIOUS")
    ).upper()
    score = float(
        result.get(
            "overall_score",
            result.get("risk_score", 0.0),
        )
    )
    confidence = float(
        result.get("confidence", 0.0)
    )

    cls = {
        "AUTHENTIC": "risk-low",
        "SUSPICIOUS": "risk-sus",
        "MANIPULATED": "risk-high",
    }.get(verdict, "risk-sus")

    st.markdown(
        f"""
        <div class="card">
            <div class="small-mono">GENERATION {generation}</div>
            <div style="font-size:2rem;font-weight:800">
                {score:.0f} / 100
            </div>
            <div class="risk-badge {cls}">
                {verdict}
            </div>
            <div class="small-mono" style="margin-top:.45rem">
                Confidence: {confidence:.0f}%
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _show_generation_diagnostics(generation: int, result: dict):
    """Show the same useful forensic plots for an analyzed compression generation."""
    lip = result.get("lip_sync", {})
    au = result.get("audio_continuity", result.get("audio", {}))
    visual = result.get("visual_analysis", {})

    st.markdown("##### Generation diagnostics")

    mouth = np.asarray(lip.get("mouth_motion", []), dtype=float)
    mouth_t = np.asarray(lip.get("mouth_time", []), dtype=float)

    feats = au.get("features") or {}
    rms = np.asarray(feats.get("rms", []), dtype=float)
    rms_t = np.asarray(feats.get("rms_time", []), dtype=float)

    if len(mouth) > 2 and len(mouth_t) == len(mouth):
        fig, ax = plt.subplots(figsize=(10, 2.8), facecolor="#0e0e0e")
        ax.plot(
            mouth_t,
            mouth / (np.max(mouth) + 1e-10),
            linewidth=1.4,
            label="Mouth motion",
        )
        if len(rms) > 2 and len(rms_t) == len(rms):
            ax.plot(
                rms_t,
                rms / (np.max(rms) + 1e-10),
                linewidth=1.2,
                label="Audio RMS",
            )
        ax.set_title(
            f"Gen {generation}: lip-sync motion vs audio",
            color="#ddd",
        )
        ax.set_xlabel("Time (s)", color="#aaa")
        ax.set_ylabel("Normalized activity", color="#aaa")
        ax.tick_params(colors="#888")
        ax.set_facecolor("#0e0e0e")
        ax.legend(facecolor="#222", edgecolor="#444", labelcolor="#ddd")
        for spine in ax.spines.values():
            spine.set_color("#444")
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)
    else:
        st.info(
            f"Gen {generation}: lip-sync plot unavailable because mouth motion data was not produced."
        )

    waveform = au.get("waveform")
    spec = au.get("spectrogram")

    if waveform or spec:
        c1, c2 = st.columns(2)

        if waveform:
            wy = np.asarray(waveform.get("y", []), dtype=float)
            wt = np.asarray(waveform.get("t", []), dtype=float)
            with c1:
                if len(wy) and len(wt) == len(wy):
                    fig, ax = plt.subplots(figsize=(6, 2.7), facecolor="#0e0e0e")
                    ax.plot(wt, wy, linewidth=0.55)
                    ax.set_title("Waveform", color="#ddd")
                    ax.tick_params(colors="#888")
                    ax.set_facecolor("#0e0e0e")
                    for a in au.get("anomalies", []):
                        ax.axvline(
                            float(a.get("timestamp", 0)),
                            linestyle="--",
                            alpha=0.55,
                        )
                    for spine in ax.spines.values():
                        spine.set_color("#444")
                    plt.tight_layout()
                    st.pyplot(fig, use_container_width=True)
                    plt.close(fig)

        if spec:
            sf = np.asarray(spec.get("f", []), dtype=float)
            st_ = np.asarray(spec.get("t", []), dtype=float)
            sm = np.asarray(spec.get("mag_db", []), dtype=float)
            with c2:
                if sm.size and len(sf) and len(st_):
                    fig, ax = plt.subplots(figsize=(6, 2.7), facecolor="#0e0e0e")
                    ax.pcolormesh(st_, sf, sm, shading="auto", cmap="magma")
                    ax.set_title("Spectrogram", color="#ddd")
                    ax.set_xlabel("Time (s)", color="#aaa")
                    ax.tick_params(colors="#888")
                    ax.set_facecolor("#0e0e0e")
                    for a in au.get("anomalies", []):
                        ax.axvline(
                            float(a.get("timestamp", 0)),
                            linestyle="--",
                            alpha=0.55,
                        )
                    for spine in ax.spines.values():
                        spine.set_color("#444")
                    plt.tight_layout()
                    st.pyplot(fig, use_container_width=True)
                    plt.close(fig)

    anomalies = []
    for a in au.get("anomalies", []):
        anomalies.append(
            (float(a.get("timestamp", 0)), "audio")
        )
    for a in visual.get("visual_anomalies", []):
        anomalies.append(
            (float(a.get("timestamp", 0)), "visual")
        )
    for a in lip.get("anomalies", []):
        anomalies.append(
            (float(a.get("timestamp", 0)), "lip-sync")
        )

    if anomalies:
        fig, ax = plt.subplots(figsize=(10, 1.9), facecolor="#0e0e0e")
        max_t = max([x[0] for x in anomalies] + [0.0]) + 1.0
        ax.set_xlim(0, max(1.0, max_t))
        ax.set_ylim(-1, 1)
        ax.axhline(0, linewidth=1)
        marker_map = {
            "lip-sync": "^",
            "audio": "o",
            "visual": "s",
        }
        for idx, (ts, kind) in enumerate(sorted(anomalies)):
            ax.scatter(
                ts,
                0,
                s=70,
                marker=marker_map[kind],
                label=kind if idx == 0 else None,
            )
        ax.set_title(
            f"Gen {generation}: anomaly timeline",
            color="#ddd",
        )
        ax.set_yticks([])
        ax.tick_params(colors="#888")
        ax.set_facecolor("#0e0e0e")
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            ax.legend(
                handles,
                labels,
                loc="upper center",
                bbox_to_anchor=(0.5, -0.12),
                ncol=min(3, len(handles)),
                frameon=False,
                labelcolor="#ddd",
            )
        for spine in ax.spines.values():
            spine.set_visible(False)
        plt.subplots_adjust(bottom=0.28)
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)



st.set_page_config(page_title="FAKE FACE, REAL RIOT", layout="wide", page_icon="🎬")

st.markdown("""
<style>
    .main-header {
        font-size: 2.6rem; font-weight: 800;
        background: linear-gradient(90deg, #ff4d4d, #ffb74d);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        letter-spacing: 2px; margin-bottom: 0;
    }
    .subtitle { color: #b0b0b0; font-size: 1.05rem; margin-bottom: 1.5rem; }
    .card { background-color: #1a1a1a; border: 1px solid #2a2a2a;
            border-radius: 8px; padding: 1rem 1.25rem; margin-bottom: 0.5rem; }
    .score-big { font-size: 3.5rem; font-weight: 800; font-family: 'Courier New', monospace; }
    .risk-badge { font-size: 1.4rem; font-weight: 800; padding: 0.45rem 1rem;
                  border-radius: 6px; display: inline-block; margin-top: 0.4rem; }
    .risk-low  { background: #1b3a1b; color: #4ade80; }
    .risk-sus  { background: #3a3018; color: #fbbf24; }
    .risk-high { background: #3a1b1b; color: #f87171; }
    .small-mono { font-family: 'Courier New', monospace; color: #999; font-size: 0.85rem; }
    .evidence-name { font-size: 0.9rem; color: #aaa;
                    text-transform: uppercase; letter-spacing: 1px; }
    .evidence-score { font-size: 2rem; font-weight: 700;
                       font-family: 'Courier New', monospace; }
    .evidence-note { font-size: 0.85rem; color: #b0b0b0; }
    .stButton>button { background-color: #d93025; color: white;
                       border: none; font-weight: 700; }
    .stButton>button:hover { background-color: #b3261e; color: white; }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-header">FAKE FACE, REAL RIOT</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Short-Video Manipulation Detection &amp; Forensic Analysis · 100% Local</div>',
            unsafe_allow_html=True)

with st.sidebar:
    st.markdown("### About")
    st.write("Forensic indicator analysis on short talking-head videos.")
    st.markdown("- 🎥 Lip-sync consistency")
    st.markdown("- 🔊 Audio splicing / discontinuity")
    st.markdown("- 🗜️ Recompression robustness")
    st.write("Results are **indicators**, not absolute verdicts.")
    st.markdown("---")
    st.write(f"Temp dir: `{config.TEMP_DIR}`")
    st.write(f"FFmpeg: {'✅' if vu.FFMPEG_AVAILABLE else '❌'}")
    st.write(f"OpenCV: {'✅' if vu.HAS_CV2 else '⚠️ (lip-sync skipped)'}")

st.markdown("### 1. Upload a short talking-head video")
uploaded = st.file_uploader(
    "Drop your video here (MP4 / MOV / AVI / MKV / WEBM)",
    type=["mp4", "mov", "avi", "mkv", "webm", "m4v"],
    label_visibility="collapsed"
)

if uploaded is not None:
    ext = Path(uploaded.name).suffix.lower()
    uploaded_bytes = uploaded.getvalue()
    video_id = _video_id(uploaded_bytes, uploaded.name)

    # Keep the uploaded source alive across Streamlit reruns so the
    # compression lab can build Generation 1, 2, 3... on demand.
    save_path = str(
        config.TEMP_DIR / f"upload_current_{video_id}{ext}"
    )

    if (
        not os.path.exists(save_path)
        or st.session_state.get("compression_video_id") != video_id
    ):
        with open(save_path, "wb") as f:
            f.write(uploaded_bytes)

    ok, msg = vu.validate_upload(
        save_path,
        uploaded.size,
        ext,
    )
    if not ok:
        st.error(f"❌ {msg}")
        try: os.remove(save_path)
        except Exception: pass
        st.stop()

    _ensure_compression_state(
        video_id,
        save_path,
    )

    col1, col2 = st.columns([3, 2])
    with col1:
        st.markdown("### 2. Video preview")
        with open(save_path, "rb") as f:
            st.video(f.read())
    with col2:
        st.markdown("### Video info")
        try:
            meta = vu.get_video_metadata(save_path)
            st.markdown('<div class="card small-mono">', unsafe_allow_html=True)
            if meta.get("duration") is not None:
                st.write(f"**Duration:** {meta['duration']:.2f}s")
            else:
                st.write("**Duration:** N/A")
            st.write(f"**Resolution:** {meta.get('width','?')} × {meta.get('height','?')}")
            st.write(f"**FPS:** {meta.get('fps','?')}")
            st.write(f"**Video codec:** {meta.get('video_codec','?')}")
            st.write(f"**Audio:** {'yes ('+str(meta.get('audio_codec','?'))+')' if meta.get('audio') else 'no'}")
            if meta.get("audio"):
                st.write(f"**Audio sr:** {meta.get('audio_sample_rate','?')} Hz, {meta.get('audio_channels','?')}")
            st.write(f"**Bitrate:** {meta.get('bit_rate','?')}")
            st.write(f"**File size:** {meta.get('file_size_mb',0):.2f} MB")
            st.markdown('</div>', unsafe_allow_html=True)
        except Exception as e:
            st.error(f"Failed to read metadata: {e}")


    # -----------------------------------------------------------------------
    # COMPRESSION LAB
    # -----------------------------------------------------------------------
    st.markdown("---")
    st.markdown("### 3. Compression Lab")
    st.caption(
        "Analyze the original first, then create recompressed generations "
        "one at a time. Every generation is created from the previous one."
    )

    if st.button(
        "↩ Reset compression chain",
        key="reset_compression_chain",
    ):
        # Delete generated generations, but never delete the original upload.
        for gen, path in list(
            st.session_state.compression_chain.items()
        ):
            if gen > 0:
                try:
                    if os.path.exists(path):
                        os.remove(path)
                except Exception:
                    pass

        st.session_state.compression_chain = {
            0: save_path
        }
        st.session_state.compression_results = {}
        st.session_state.compression_errors = {}
        st.session_state.compression_crfs = {}
        st.rerun()

    original_path = st.session_state.compression_chain[0]

    lab_a, lab_b = st.columns([2, 1])
    with lab_a:
        st.markdown(
            """
            <div class="card">
                <div class="evidence-name">EXPERIMENT MODE</div>
                <div style="font-size:1.15rem;font-weight:700;margin-top:.35rem">
                    Original → G1 → G2 → G3 → G4 → G5
                </div>
                <div class="small-mono" style="margin-top:.4rem">
                    Each compression is generated only when you request it.
                    No automatic chain is created.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with lab_b:
        st.markdown("**Compression settings**")
        crf = st.slider(
            "Compression strength (CRF)",
            min_value=18,
            max_value=38,
            value=28,
            step=1,
            help=(
                "Lower = higher quality / lighter compression. "
                "Higher = stronger compression."
            ),
            key="compression_crf",
        )

    chain_cols = st.columns(min(6, len(st.session_state.compression_chain)))

    for generation in sorted(
        st.session_state.compression_chain.keys()
    ):
        path = st.session_state.compression_chain[generation]
        col = chain_cols[
            generation % len(chain_cols)
        ]

        with col:
            label = "ORIGINAL" if generation == 0 else f"GEN {generation}"
            st.markdown(f"**{label}**")
            crf_label = (
                "Original"
                if generation == 0
                else f"CRF {st.session_state.get('compression_crfs', {}).get(generation, '?')}"
            )
            st.caption(
                f"{_file_size_mb(path):.2f} MB · {crf_label}"
            )

            if os.path.exists(path):
                with open(path, "rb") as vf:
                    st.video(
                        vf.read(),
                    )

            if generation not in st.session_state.compression_results:
                if st.button(
                    "🔍 Analyze",
                    key=f"analyze_generation_{generation}",
                    use_container_width=True,
                ):
                    _run_generation_analysis(generation)
                    st.rerun()
            else:
                _show_generation_result(
                    generation,
                    st.session_state.compression_results[generation],
                )
                _show_generation_diagnostics(
                    generation,
                    st.session_state.compression_results[generation],
                )

    current_gen = max(
        st.session_state.compression_chain.keys()
    )

    if current_gen < 5:
        st.markdown("---")
        next_gen = current_gen + 1

        generation_crf = min(
            51,
            int(crf) + (next_gen - 1) * 6,
        )

        st.caption(
            f"Next generation will use **CRF {generation_crf}** "
            f"(progressively stronger compression)."
        )

        if st.button(
            f"🗜️ Compress → Generation {next_gen}",
            use_container_width=True,
            type="primary",
        ):
            source = st.session_state.compression_chain[current_gen]
            output = str(
                config.TEMP_DIR
                / f"compression_{video_id}_gen{next_gen}.mp4"
            )

            with st.spinner(
                f"Creating compression Generation {next_gen} "
                f"(CRF {generation_crf})..."
            ):
                ok_comp, comp_msg = _compress_generation(
                    source,
                    output,
                    crf=generation_crf,
                )

            if ok_comp:
                st.session_state.compression_chain[
                    next_gen
                ] = output

                if "compression_crfs" not in st.session_state:
                    st.session_state.compression_crfs = {}

                st.session_state.compression_crfs[
                    next_gen
                ] = generation_crf

                st.success(
                    f"Generation {next_gen} created from "
                    f"Generation {current_gen} using CRF "
                    f"{generation_crf}."
                )
                st.rerun()
            else:
                st.error(
                    f"Compression failed: {comp_msg}"
                )

    else:
        st.success(
            "Generation 5 reached. You can analyze any generation above."
        )

    if len(st.session_state.compression_chain) > 1:
        st.markdown("---")
        st.markdown("#### Compression experiment")

        rows = []
        for gen, path in sorted(
            st.session_state.compression_chain.items()
        ):
            r = st.session_state.compression_results.get(
                gen
            )
            rows.append({
                "Generation": (
                    "Original"
                    if gen == 0
                    else f"Gen {gen}"
                ),
                "Size (MB)": round(
                    _file_size_mb(path),
                    2,
                ),
                "Verdict": (
                    r.get("verdict", "Not analyzed")
                    if r else "Not analyzed"
                ),
                "Score": (
                    round(
                        float(
                            r.get(
                                "overall_score",
                                r.get("risk_score", 0),
                            )
                        )
                    )
                    if r else "-"
                ),
                "Confidence": (
                    f"{float(r.get('confidence', 0)):.0f}%"
                    if r else "-"
                ),
            })

        st.dataframe(
            rows,
            use_container_width=True,
            hide_index=True,
        )

        st.markdown("#### Compression verification")
        verified_rows = []

        for gen, path in sorted(
            st.session_state.compression_chain.items()
        ):
            try:
                size_mb = _file_size_mb(path)
                meta = vu.get_video_metadata(path)
                verified_rows.append({
                    "Generation": (
                        "Original"
                        if gen == 0
                        else f"Gen {gen}"
                    ),
                    "File exists": "✓",
                    "Size (MB)": round(size_mb, 2),
                    "Resolution": (
                        f"{meta.get('width', '?')}×"
                        f"{meta.get('height', '?')}"
                    ),
                    "FPS": meta.get("fps", "?"),
                    "Codec": meta.get("video_codec", "?"),
                    "CRF": (
                        "—"
                        if gen == 0
                        else st.session_state.get(
                            "compression_crfs", {}
                        ).get(gen, "?")
                    ),
                })
            except Exception as exc:
                verified_rows.append({
                    "Generation": (
                        "Original"
                        if gen == 0
                        else f"Gen {gen}"
                    ),
                    "File exists": "✓" if os.path.exists(path) else "✗",
                    "Size (MB)": round(
                        _file_size_mb(path),
                        2,
                    ),
                    "Resolution": "N/A",
                    "FPS": "N/A",
                    "Codec": "N/A",
                    "CRF": (
                        "—"
                        if gen == 0
                        else st.session_state.get(
                            "compression_crfs", {}
                        ).get(gen, "?")
                    ),
                })

        st.dataframe(
            verified_rows,
            use_container_width=True,
            hide_index=True,
        )

        st.caption(
            "Each generation is a real FFmpeg H.264 re-encode of the "
            "immediately preceding generation. Progressive CRF makes the "
            "quality loss increasingly visible."
        )

        analyzed_rows = [
            row for row in rows
            if row["Score"] != "-"
        ]

        if len(analyzed_rows) >= 2:
            scores = [
                float(row["Score"])
                for row in analyzed_rows
            ]
            labels = [
                row["Generation"]
                for row in analyzed_rows
            ]

            fig, ax = plt.subplots(
                figsize=(9, 3.2),
                facecolor="#0e0e0e",
            )
            ax.plot(
                labels,
                scores,
                marker="o",
                linewidth=2,
            )
            ax.set_ylim(0, 100)
            ax.set_ylabel(
                "Forensic score",
                color="#aaa",
            )
            ax.set_title(
                "Forensic score across recompression generations",
                color="#ddd",
            )
            ax.tick_params(colors="#aaa")
            ax.set_facecolor("#0e0e0e")
            for spine in ax.spines.values():
                spine.set_color("#444")
            st.pyplot(
                fig,
                use_container_width=True,
            )
            plt.close(fig)

    st.markdown("---")
    st.markdown("### 4. Standard forensic analysis")
    st.caption(
        "This is the normal one-click analysis of the original uploaded video."
    )

    if st.button("🎬 ANALYZE VIDEO", use_container_width=True):
        progress_bar = st.progress(0.0)
        status_box = st.empty()
        phases = [
            "Loading video & extracting assets",
            "Analyzing audio",
            "Extracting frames",
            "Face & lip analysis",
            "Compression robustness",
            "Fusing evidence",
            "Report generated",
        ]
        phase_idx = [0]
        phase_map = {
            "loading": 0,
            "analyzing audio": 1,
            "extracting frames": 2,
            "face": 3, "lip": 3,
            "compression": 4,
            "fusing": 5,
            "report": 6,
        }

        def update(msg: str):
            key = msg.lower()
            for k, idx in phase_map.items():
                if k in key:
                    phase_idx[0] = max(phase_idx[0], idx)
                    break
            progress_bar.progress((phase_idx[0] + 1) / len(phases))
            lines = []
            for i, p in enumerate(phases):
                if i < phase_idx[0]:
                    lines.append(f"✓ {p}")
                elif i == phase_idx[0]:
                    lines.append(f"⋯ {p}")
                else:
                    lines.append(f"○ {p}")
            status_box.markdown("\n".join(lines))

        try:
            with st.spinner("Running forensic pipeline..."):
                result = detector.analyze_video(save_path, progress=update)
        except Exception as e:
            st.error(f"Analysis failed: {e}")
            import traceback
            st.code(traceback.format_exc())
            try: os.remove(save_path)
            except Exception: pass
            st.stop()
        finally:
            progress_bar.progress(1.0)

        detector.cleanup_session(result)
        # Keep save_path alive for the Compression Lab and for subsequent
        # Streamlit reruns. It is keyed by the upload hash.

        # ---------- REPORT ----------
        st.markdown("---")
        st.markdown("## 5. Forensic report")

        # The current detector returns:
        # AUTHENTIC / SUSPICIOUS / MANIPULATED
        # and confidence as 0-100 (not 0-1).
        risk = str(
            result.get(
                "verdict",
                result.get("risk_level", "SUSPICIOUS"),
            )
        ).upper()

        # Backward-compatible labels for older UI terminology.
        risk_display = {
            "LOW RISK": "AUTHENTIC",
            "HIGH RISK": "MANIPULATED",
        }.get(risk, risk)

        css_class = {
            "AUTHENTIC": "risk-low",
            "SUSPICIOUS": "risk-sus",
            "MANIPULATED": "risk-high",
        }.get(risk_display, "risk-sus")

        risk_score = float(
            result.get(
                "overall_score",
                result.get("risk_score", 0.0),
            )
        )
        confidence = float(
            np.clip(result.get("confidence", 0.0), 0.0, 100.0)
        )

        col_a, col_b = st.columns([2, 3])

        with col_a:
            st.markdown("#### OVERALL ASSESSMENT")
            st.markdown(
                f'<div class="card">'
                f'<div class="score-big">{risk_score:.0f} / 100</div>'
                f'<div class="risk-badge {css_class}">{risk_display}</div>'
                f'<div class="small-mono" style="margin-top:0.5rem">'
                f'Confidence: {confidence:.0f}%'
                f'</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

        with col_b:
            st.markdown("#### WHY WAS THIS FLAGGED?")
            for line in result.get(
                "evidence",
                result.get("explanation", []),
            ):
                st.markdown(f"- {line}")

        # Evidence cards
        st.markdown("---")
        st.markdown("#### Evidence breakdown")
        c1, c2, c3 = st.columns(3)

        with c1:
            lip = result.get("lip_sync", {})

            st.markdown(
                '<div class="card">',
                unsafe_allow_html=True,
            )
            st.markdown(
                '<div class="evidence-name">Lip Sync</div>',
                unsafe_allow_html=True,
            )
            st.markdown(
                f'<div class="evidence-score">'
                f'{float(lip.get("score", 0.0)):.0f} / 100'
                f'</div>',
                unsafe_allow_html=True,
            )

            if not lip.get("available", False):
                st.markdown(
                    f'<div class="evidence-note">⚠ '
                    f'{lip.get("message", "Lip-sync analysis unavailable.")}'
                    f'</div>',
                    unsafe_allow_html=True,
                )
            elif float(lip.get("score", 0.0)) > 50:
                st.markdown(
                    '<div class="evidence-note">'
                    '⚠ Possible synchronization inconsistency'
                    '</div>',
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    '<div class="evidence-note">'
                    '🟢 No major sync issue'
                    '</div>',
                    unsafe_allow_html=True,
                )

            st.markdown(
                f'<div class="small-mono">'
                f'Faces: {lip.get("faces_detected", 0)} · '
                f'best-corr: {float(lip.get("best_correlation", 0.0)):.2f}'
                f'</div>',
                unsafe_allow_html=True,
            )
            st.markdown("</div>", unsafe_allow_html=True)

        with c2:
            au = result.get(
                "audio_continuity",
                result.get("audio", {}),
            )

            st.markdown(
                '<div class="card">',
                unsafe_allow_html=True,
            )
            st.markdown(
                '<div class="evidence-name">Audio Continuity</div>',
                unsafe_allow_html=True,
            )
            st.markdown(
                f'<div class="evidence-score">'
                f'{float(au.get("score", 0.0)):.0f} / 100'
                f'</div>',
                unsafe_allow_html=True,
            )

            if not au.get("available", False):
                st.markdown(
                    '<div class="evidence-note">'
                    '⚠ Audio unavailable'
                    '</div>',
                    unsafe_allow_html=True,
                )
            elif au.get("silent", False):
                st.markdown(
                    '<div class="evidence-note">'
                    '⚠ Audio appears silent'
                    '</div>',
                    unsafe_allow_html=True,
                )
            elif float(au.get("score", 0.0)) > 50:
                st.markdown(
                    '<div class="evidence-note">'
                    '🔴 Possible acoustic discontinuity'
                    '</div>',
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    '<div class="evidence-note">'
                    '🟢 No major discontinuity'
                    '</div>',
                    unsafe_allow_html=True,
                )

            st.markdown(
                f'<div class="small-mono">'
                f'Anomalies: {len(au.get("anomalies", []))}'
                f'</div>',
                unsafe_allow_html=True,
            )
            st.markdown("</div>", unsafe_allow_html=True)

        with c3:
            cp = result.get(
                "compression_robustness",
                result.get("compression", {}),
            )

            st.markdown(
                '<div class="card">',
                unsafe_allow_html=True,
            )
            st.markdown(
                '<div class="evidence-name">Compression Robustness</div>',
                unsafe_allow_html=True,
            )
            st.markdown(
                f'<div class="evidence-score">'
                f'{float(cp.get("score", 0.0)):.0f} / 100'
                f'</div>',
                unsafe_allow_html=True,
            )

            if float(cp.get("score", 0.0)) > 60:
                st.markdown(
                    '<div class="evidence-note">'
                    '⚠ Persistent forensic signal'
                    '</div>',
                    unsafe_allow_html=True,
                )
            elif float(cp.get("score", 0.0)) > 25:
                st.markdown(
                    '<div class="evidence-note">'
                    '🟡 Secondary forensic signal'
                    '</div>',
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    '<div class="evidence-note">'
                    '🟢 No abnormal compression persistence'
                    '</div>',
                    unsafe_allow_html=True,
                )

            st.markdown(
                f'<div class="small-mono">'
                f'Generations: {cp.get("generations", 0)} · '
                f'stability: {float(cp.get("stability", 0.0)):.2f}'
                f'</div>',
                unsafe_allow_html=True,
            )
            st.markdown("</div>", unsafe_allow_html=True)

        # Timeline
        st.markdown("---")
        st.markdown("#### Suspicious timeline")

        timeline = []

        for anomaly in lip.get("anomalies", []):
            timeline.append({
                "timestamp": float(anomaly.get("timestamp", 0.0)),
                "type": "lip_sync",
                "description": anomaly.get(
                    "description",
                    "Lip-sync anomaly",
                ),
            })

        for anomaly in au.get("anomalies", []):
            timeline.append({
                "timestamp": float(anomaly.get("timestamp", 0.0)),
                "type": "audio",
                "description": anomaly.get(
                    "description",
                    "Audio anomaly",
                ),
            })

        visual_data = result.get(
            "visual_analysis",
            {},
        )

        for anomaly in visual_data.get(
            "visual_anomalies",
            [],
        ):
            timeline.append({
                "timestamp": float(anomaly.get("timestamp", 0.0)),
                "type": "visual",
                "description": anomaly.get(
                    "description",
                    "Visual anomaly",
                ),
            })

        timeline.sort(
            key=lambda a: a["timestamp"]
        )

        dur = float(
            result.get(
                "metadata",
                {},
            ).get(
                "duration",
                au.get("duration", 10.0),
            )
            or 10.0
        )

        if timeline:
            dur = max(
                dur,
                max(
                    a["timestamp"]
                    for a in timeline
                ) + 1,
            )

        dur = max(dur, 1.0)

        fig, ax = plt.subplots(
            figsize=(10, 1.8),
            facecolor="#0e0e0e",
        )
        ax.set_xlim(0, dur)
        ax.set_ylim(-1, 1)
        ax.axhline(
            0,
            color="#444",
            linewidth=1,
        )

        ticks = np.linspace(
            0,
            dur,
            min(int(dur) + 1, 11),
        )
        ax.set_xticks(ticks)
        ax.set_xticklabels(
            [f"{t:.0f}s" for t in ticks],
            color="#aaa",
        )
        ax.set_yticks([])

        for spine in ax.spines.values():
            spine.set_visible(False)

        marker_colors = {
            "lip_sync": "#ff5757",
            "audio": "#ffb74d",
            "visual": "#60a5fa",
        }

        for anomaly in timeline:
            ax.plot(
                anomaly["timestamp"],
                0,
                marker="^",
                markersize=14,
                color=marker_colors.get(
                    anomaly["type"],
                    "#ffffff",
                ),
                markeredgecolor="white",
            )

        handles = [
            Line2D(
                [0],
                [0],
                marker="^",
                color="w",
                markerfacecolor="#ff5757",
                markersize=12,
                label="lip-sync anomaly",
            ),
            Line2D(
                [0],
                [0],
                marker="^",
                color="w",
                markerfacecolor="#ffb74d",
                markersize=12,
                label="audio anomaly",
            ),
            Line2D(
                [0],
                [0],
                marker="^",
                color="w",
                markerfacecolor="#60a5fa",
                markersize=12,
                label="visual anomaly",
            ),
        ]

        ax.legend(
            handles=handles,
            loc="upper center",
            bbox_to_anchor=(0.5, -0.16),
            ncol=3,
            facecolor="#222",
            edgecolor="#444",
            labelcolor="#ddd",
        )
        ax.set_facecolor("#0e0e0e")
        plt.subplots_adjust(bottom=0.32, top=0.96, left=0.05, right=0.98)
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

        if timeline:
            st.markdown("**Anomaly list:**")
            for anomaly in timeline:
                st.markdown(
                    f"- `{anomaly['timestamp']:.2f}s` "
                    f"({anomaly['type']}): "
                    f"{anomaly['description']}"
                )
        else:
            st.write("No specific anomalies to plot.")

        # Audio visualizations
        st.markdown("---")
        st.markdown("#### Audio visualizations")
        au = result.get(
            "audio_continuity",
            result.get("audio", {}),
        )
        if au.get("available") and au.get("waveform"):
            wy = np.array(au["waveform"]["y"])
            wt = np.array(au["waveform"]["t"])
            fig, axes = plt.subplots(3, 1, figsize=(11, 7), facecolor='#0e0e0e')

            axes[0].plot(wt, wy, color='#4ade80', linewidth=0.5)
            axes[0].set_title("Waveform", color='#ddd')
            axes[0].set_facecolor('#0e0e0e')
            axes[0].tick_params(colors='#888')
            for a in au.get("anomalies", []):
                axes[0].axvline(a["timestamp"], color='#ff5757', alpha=0.55, linestyle='--')

            spec = au.get("spectrogram")
            if spec:
                sf = np.array(spec["f"])
                st_ = np.array(spec["t"])
                sm = np.array(spec["mag_db"])
                axes[1].pcolormesh(st_, sf, sm, shading='auto', cmap='magma')
                axes[1].set_title("Spectrogram", color='#ddd')
                axes[1].set_facecolor('#0e0e0e')
                axes[1].tick_params(colors='#888')
                for a in au.get("anomalies", []):
                    axes[1].axvline(a["timestamp"], color='#ff5757', alpha=0.55, linestyle='--')

            feats = au.get("features")
            if feats:
                ft = np.array(feats["flux_time"])
                flux = np.array(feats["flux"])
                rms = np.array(feats["rms"])
                rms_t = np.array(feats["rms_time"])
                axes[2].plot(ft, flux / (flux.max() + 1e-10), color='#ffb74d',
                             label='spectral flux', linewidth=0.8)
                axes[2].plot(rms_t, rms / (rms.max() + 1e-10), color='#60a5fa',
                             label='RMS', linewidth=0.8)
                axes[2].set_title("Spectral flux & RMS energy", color='#ddd')
                axes[2].set_facecolor('#0e0e0e')
                axes[2].tick_params(colors='#888')
                axes[2].legend(facecolor='#222', edgecolor='#444', labelcolor='#ddd')
                for a in au.get("anomalies", []):
                    axes[2].axvline(a["timestamp"], color='#ff5757', alpha=0.55, linestyle='--')

            for ax in axes:
                for spine in ax.spines.values():
                    spine.set_color('#444')
            plt.tight_layout()
            st.pyplot(fig, use_container_width=True)
        else:
            st.write("Audio unavailable — no audio plots.")

        # Compression stability
        st.markdown("---")
        st.markdown("#### Compression robustness")

        cp = result.get(
            "compression_robustness",
            result.get("compression", {}),
        )

        generations_count = int(
            cp.get("generations", 0) or 0
        )

        if generations_count > 0:
            stability = float(
                cp.get("stability", 0.0)
            )
            avg_lip_diff = float(
                cp.get("avg_lip_diff", 0.0)
            )
            avg_audio_diff = float(
                cp.get("avg_audio_diff", 0.0)
            )

            st.markdown(
                f"""
                <div class="card">
                    <div class="evidence-name">
                        RECOMPRESSION ANALYSIS
                    </div>
                    <div class="small-mono" style="margin-top:0.6rem">
                        Generations tested: {generations_count}
                        &nbsp; · &nbsp;
                        Stability: {stability:.2f}
                        &nbsp; · &nbsp;
                        Avg lip drift: {avg_lip_diff:.1f}
                        &nbsp; · &nbsp;
                        Avg audio drift: {avg_audio_diff:.1f}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            st.progress(
                max(
                    0.0,
                    min(1.0, stability),
                )
            )

            st.caption(
                "Compression robustness is a secondary forensic signal. "
                "Persistence alone does not prove manipulation."
            )
        else:
            st.write(
                "Compression test did not complete."
            )

        with st.expander("Raw forensic data (transparency)"):
            safe = {k: v for k, v in result.items()
                    if k not in ("audio_path", "frames_dir")}
            st.json(safe)

else:
    st.info("👆 Upload a video to begin.")
