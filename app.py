"""FAKE FACE, REAL RIOT - Streamlit forensic dashboard."""

import os
import time
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
    ext = Path(uploaded.name).suffix
    save_path = str(config.TEMP_DIR / f"upload_{int(time.time())}{ext}")
    with open(save_path, "wb") as f:
        f.write(uploaded.getvalue())

    ok, msg = vu.validate_upload(save_path, uploaded.size, ext)
    if not ok:
        st.error(f"❌ {msg}")
        try: os.remove(save_path)
        except Exception: pass
        st.stop()

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

    st.markdown("---")
    st.markdown("### 3. Forensic analysis")
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
        try: os.remove(save_path)
        except Exception: pass

        # ---------- REPORT ----------
        st.markdown("---")
        st.markdown("## 4. Forensic report")

        risk = result["risk_level"]
        css_class = {"LOW RISK": "risk-low", "SUSPICIOUS": "risk-sus", "HIGH RISK": "risk-high"}[risk]

        col_a, col_b = st.columns([2, 3])
        with col_a:
            st.markdown("#### OVERALL ASSESSMENT")
            st.markdown(
                f'<div class="card"><div class="score-big">{result["risk_score"]:.0f} / 100</div>'
                f'<div class="risk-badge {css_class}">{risk}</div>'
                f'<div class="small-mono" style="margin-top:0.5rem">Confidence: {result["confidence"]*100:.0f}%</div>'
                f'</div>', unsafe_allow_html=True)
        with col_b:
            st.markdown("#### WHY WAS THIS FLAGGED?")
            for line in result["explanation"]:
                st.markdown(f"- {line}")

        # Evidence cards
        st.markdown("---")
        st.markdown("#### Evidence breakdown")
        c1, c2, c3 = st.columns(3)
        with c1:
            lip = result["lip_sync"]
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown('<div class="evidence-name">Lip Sync</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="evidence-score">{lip["score"]:.0f} / 100</div>', unsafe_allow_html=True)
            if not lip.get("available"):
                st.markdown(f'<div class="evidence-note">⚠ {lip.get("message","unavailable")}</div>', unsafe_allow_html=True)
            elif lip["score"] > 50:
                st.markdown('<div class="evidence-note">⚠ Possible synchronization inconsistency</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="evidence-note">🟢 No major sync issue</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="small-mono">Faces: {lip.get("face_count",0)} · best-corr: {lip.get("best_corr",0):.2f}</div>',
                        unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
        with c2:
            au = result["audio"]
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown('<div class="evidence-name">Audio Continuity</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="evidence-score">{au["score"]:.0f} / 100</div>', unsafe_allow_html=True)
            if not au.get("available"):
                st.markdown('<div class="evidence-note">⚠ Audio unavailable</div>', unsafe_allow_html=True)
            elif au.get("silent"):
                st.markdown('<div class="evidence-note">⚠ Audio appears silent</div>', unsafe_allow_html=True)
            elif au["score"] > 50:
                st.markdown('<div class="evidence-note">🔴 Possible acoustic discontinuity</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="evidence-note">🟢 No major discontinuity</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="small-mono">Anomalies: {len(au.get("anomalies",[]))}</div>',
                        unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
        with c3:
            cp = result["compression"]
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown('<div class="evidence-name">Compression Robustness</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="evidence-score">{cp["score"]:.0f} / 100</div>', unsafe_allow_html=True)
            if cp["score"] > 60:
                st.markdown('<div class="evidence-note">🟢 Suspicious signal persists</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="evidence-note">No robust suspicious signal</div>', unsafe_allow_html=True)
            st.markdown(
                f'<div class="small-mono">Generations: {len(cp.get("generations",[]))} · stability: {cp.get("robustness",0):.2f}</div>',
                unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

        # Timeline
        st.markdown("---")
        st.markdown("#### Suspicious timeline")
        timeline = result.get("timeline", [])
        dur = result.get("audio", {}).get("duration") or 10
        if timeline:
            dur = max(dur, max(a["timestamp"] for a in timeline) + 1)
        dur = max(dur, 1.0)

        fig, ax = plt.subplots(figsize=(10, 1.8), facecolor='#0e0e0e')
        ax.set_xlim(0, dur)
        ax.set_ylim(-1, 1)
        ax.axhline(0, color='#444', linewidth=1)
        ticks = np.linspace(0, dur, min(int(dur) + 1, 11))
        ax.set_xticks(ticks)
        ax.set_xticklabels([f"{t:.0f}s" for t in ticks], color='#aaa')
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
        for a in timeline:
            color = "#ff5757" if a["type"] == "lip_sync" else "#ffb74d"
            ax.plot(a["timestamp"], 0, marker='^', markersize=14,
                    color=color, markeredgecolor='white')
        handles = [
            Line2D([0], [0], marker='^', color='w', markerfacecolor='#ff5757',
                    markersize=12, label='lip-sync anomaly'),
            Line2D([0], [0], marker='^', color='w', markerfacecolor='#ffb74d',
                    markersize=12, label='audio anomaly'),
        ]
        ax.legend(handles=handles, loc='upper right', facecolor='#222',
                  edgecolor='#444', labelcolor='#ddd')
        ax.set_facecolor('#0e0e0e')
        st.pyplot(fig, use_container_width=True)

        if timeline:
            st.markdown("**Anomaly list:**")
            for a in timeline:
                st.markdown(f"- `{a['timestamp']:.2f}s` ({a['type']}): {a['description']}")
        else:
            st.write("No specific anomalies to plot.")

        # Audio visualizations
        st.markdown("---")
        st.markdown("#### Audio visualizations")
        au = result["audio"]
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
        cp = result["compression"]
        gens = cp.get("generations", [])
        if gens:
            fig, ax = plt.subplots(figsize=(8, 2.5), facecolor='#0e0e0e')
            xs = [g["generation"] for g in gens]
            lip_v = [g["lip_sync_score"] for g in gens]
            aud_v = [g["audio_score"] for g in gens]
            xs_all = [0] + xs
            lip_all = [result["lip_sync"]["score"]] + lip_v
            aud_all = [result["audio"]["score"]] + aud_v
            ax.plot(xs_all, lip_all, marker='o', color='#ff5757', label='lip-sync risk')
            ax.plot(xs_all, aud_all, marker='s', color='#ffb74d', label='audio risk')
            ax.set_xticks(xs_all)
            ax.set_xticklabels(['orig'] + [f'g{i}' for i in xs], color='#aaa')
            ax.tick_params(colors='#888')
            ax.set_title("Score stability across recompression generations", color='#ddd')
            ax.set_facecolor('#0e0e0e')
            ax.legend(facecolor='#222', edgecolor='#444', labelcolor='#ddd')
            for spine in ax.spines.values():
                spine.set_color('#444')
            st.pyplot(fig, use_container_width=True)
            st.markdown(
                f"Stability: `{cp['robustness']:.2f}` · "
                f"avg lip drift: `{cp.get('avg_lip_diff',0):.1f}` · "
                f"avg audio drift: `{cp.get('avg_audio_diff',0):.1f}`"
            )
        else:
            st.write("Compression test did not complete.")

        with st.expander("Raw forensic data (transparency)"):
            safe = {k: v for k, v in result.items()
                    if k not in ("audio_path", "frames_dir")}
            st.json(safe)

else:
    st.info("👆 Upload a video to begin.")