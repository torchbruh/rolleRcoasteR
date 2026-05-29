"""
rolleRcoasteR — The Rollercoaster for Radio Waves That Have Never Existed

A complete, standalone Streamlit laboratory for inventing genuinely new radio waveforms
optimized for civilizational missions.

Run with: streamlit run app.py
"""

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
from matplotlib.colors import LinearSegmentedColormap
import pandas as pd
from datetime import datetime
import os
import json
from pathlib import Path

# Local modules
import core
import llm

# =============================================================================
# PAGE CONFIG & GLOBAL STYLE (Fresh, not copied from anywhere)
# =============================================================================

st.set_page_config(
    page_title="rolleRcoasteR",
    page_icon="🎢",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom dark scientific theme — electric cyan + warm gold on deep space
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&family=Space+Grotesk:wght@500;600&display=swap');

:root {
    --bg: #0a0c12;
    --card: #12151f;
    --accent: #00f0ff;
    --gold: #f4c95d;
    --text: #e6e8ee;
    --muted: #8a8f9e;
}

html, body, [class*="css"] {
    font-family: 'Inter', system_ui, sans-serif;
}

h1, h2, h3 {
    font-family: 'Space Grotesk', 'Inter', sans-serif;
    font-weight: 600;
    letter-spacing: -0.02em;
}

.main-header {
    font-size: 4.6rem;
    font-weight: 800;
    letter-spacing: -0.04em;
    background: linear-gradient(90deg, #00f0ff 0%, #f4c95d 50%, #00f0ff 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 0.05rem;
    line-height: 1.05;
    filter: drop-shadow(0 0 12px rgba(0, 240, 255, 0.25));
}

.tagline {
    font-size: 1.35rem;
    color: #8a8f9e;
    margin-top: -0.25rem;
    margin-bottom: 1.8rem;
    font-weight: 500;
}

.hero-brand {
    background: linear-gradient(180deg, #12151f 0%, #0a0c12 100%);
    border: 1px solid #252a38;
    border-radius: 20px;
    padding: 1.4rem 2rem 1.1rem;
    margin-bottom: 1.8rem;
    text-align: center;
}

.ride-card {
    background: #12151f;
    border: 1px solid #252a38;
    border-radius: 16px;
    padding: 1.25rem 1.35rem;
    margin-bottom: 1rem;
    transition: transform 0.15s ease, border-color 0.15s ease;
}

.ride-card:hover {
    border-color: #00f0ff;
    transform: translateY(-2px);
}

.metric-big {
    font-size: 1.85rem;
    font-weight: 700;
    color: #00f0ff;
    line-height: 1.1;
}

.metric-label {
    font-size: 0.75rem;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: #8a8f9e;
    margin-bottom: 0.15rem;
}

.stButton>button {
    background: linear-gradient(90deg, #00f0ff 0%, #00d4e6 100%);
    color: #0a0c12;
    font-weight: 700;
    border: none;
    border-radius: 10px;
    padding: 0.55rem 1.35rem;
    transition: all 0.2s ease;
}

.stButton>button:hover {
    transform: translateY(-1px);
    box-shadow: 0 0 0 3px rgba(0, 240, 255, 0.2);
}

.stButton>button[kind="secondary"] {
    background: transparent;
    color: #f4c95d;
    border: 1px solid #f4c95d;
}

.wave-name {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 1.45rem;
    font-weight: 700;
    color: #f4c95d;
    margin-bottom: 0.35rem;
}

.disclaimer {
    font-size: 0.78rem;
    color: #8a8f9e;
    border-left: 3px solid #f4c95d;
    padding-left: 0.85rem;
    margin: 1rem 0;
}

.plot-container {
    background: #0d0f17;
    border-radius: 12px;
    padding: 0.6rem;
    border: 1px solid #1f2430;
}
</style>
""", unsafe_allow_html=True)


# =============================================================================
# SESSION STATE
# =============================================================================

if "concepts" not in st.session_state:
    st.session_state.concepts = []
if "current_forge" not in st.session_state:
    st.session_state.current_forge = None
if "forge_history" not in st.session_state:
    st.session_state.forge_history = []
if "last_mission" not in st.session_state:
    st.session_state.last_mission = ""


# =============================================================================
# HELPER: BEAUTIFUL PLOTTING
# =============================================================================

def make_cmap():
    colors = ["#0a0c12", "#0f1a2e", "#1a3a5c", "#00d4ff", "#f4c95d", "#fff"]
    return LinearSegmentedColormap.from_list("rollercoaster", colors)


def plot_waveform(iq: np.ndarray, fs: float, title: str = "Time Domain"):
    fig, ax = plt.subplots(figsize=(7.5, 2.8), facecolor="#0a0c12")
    ax.set_facecolor("#0a0c12")
    t = np.arange(len(iq)) / fs * 1000  # ms
    ax.plot(t, np.real(iq), color="#00f0ff", linewidth=0.85, alpha=0.95, label="I")
    ax.plot(t, np.imag(iq), color="#f4c95d", linewidth=0.85, alpha=0.75, label="Q")
    ax.legend(loc="upper right", frameon=False, fontsize=8, labelcolor="#8a8f9e")
    ax.set_xlabel("Time (ms)", color="#8a8f9e", fontsize=9)
    ax.set_ylabel("Amplitude", color="#8a8f9e", fontsize=9)
    ax.tick_params(colors="#8a8f9e")
    for spine in ax.spines.values():
        spine.set_color("#252a38")
    ax.set_title(title, color="#f4c95d", fontsize=11, pad=8, fontweight=600)
    fig.tight_layout()
    return fig


def plot_spectrogram(iq: np.ndarray, fs: float):
    f, t, Sxx = core.compute_spectrogram(iq, fs, nperseg=256)
    fig, ax = plt.subplots(figsize=(7.5, 3.4), facecolor="#0a0c12")
    ax.set_facecolor("#0a0c12")
    cmap = make_cmap()
    im = ax.pcolormesh(t * 1000, f / 1e6, Sxx, cmap=cmap, shading="gouraud", vmin=np.percentile(Sxx, 2), vmax=np.percentile(Sxx, 98))
    ax.set_xlabel("Time (ms)", color="#8a8f9e", fontsize=9)
    ax.set_ylabel("Frequency (MHz)", color="#8a8f9e", fontsize=9)
    ax.tick_params(colors="#8a8f9e")
    for spine in ax.spines.values():
        spine.set_color("#252a38")
    ax.set_title("Spectrogram — The Rollercoaster Track", color="#f4c95d", fontsize=11, pad=8, fontweight=600)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    cbar.ax.yaxis.set_tick_params(color="#8a8f9e")
    plt.setp(plt.getp(cbar.axes, 'yticklabels'), color="#8a8f9e")
    fig.tight_layout()
    return fig


def plot_psd(iq: np.ndarray, fs: float):
    f, Pxx = core.compute_psd(iq, fs)
    fig, ax = plt.subplots(figsize=(7.5, 2.6), facecolor="#0a0c12")
    ax.set_facecolor("#0a0c12")
    ax.fill_between(f / 1e6, Pxx.min() - 3, Pxx, color="#00f0ff", alpha=0.35)
    ax.plot(f / 1e6, Pxx, color="#00f0ff", linewidth=1.2)
    ax.set_xlabel("Frequency (MHz)", color="#8a8f9e", fontsize=9)
    ax.set_ylabel("PSD (dB)", color="#8a8f9e", fontsize=9)
    ax.tick_params(colors="#8a8f9e")
    for spine in ax.spines.values():
        spine.set_color("#252a38")
    ax.set_title("Power Spectral Density — Notice the Unusual Shape", color="#f4c95d", fontsize=11, pad=8, fontweight=600)
    fig.tight_layout()
    return fig


def plot_ambiguity(iq: np.ndarray):
    acf = core.compute_ambiguity_slice(iq, max_lag=220)
    fig, ax = plt.subplots(figsize=(7.5, 2.6), facecolor="#0a0c12")
    ax.set_facecolor("#0a0c12")
    lags = np.arange(-len(acf)//2, len(acf)//2 + 1)
    ax.plot(lags, 10 * np.log10(acf + 1e-12), color="#f4c95d", linewidth=1.1)
    ax.axhline(y=10 * np.log10(np.max(acf) * 0.05), color="#8a8f9e", linestyle="--", linewidth=0.8, alpha=0.7, label="–13 dB reference")
    ax.legend(frameon=False, fontsize=8, labelcolor="#8a8f9e")
    ax.set_xlabel("Lag (samples)", color="#8a8f9e", fontsize=9)
    ax.set_ylabel("Correlation (dB)", color="#8a8f9e", fontsize=9)
    ax.tick_params(colors="#8a8f9e")
    for spine in ax.spines.values():
        spine.set_color("#252a38")
    ax.set_title("Ambiguity Slice (Range Profile) — Sensing Clarity", color="#f4c95d", fontsize=11, pad=8, fontweight=600)
    fig.tight_layout()
    return fig


def plot_instantaneous(iq: np.ndarray, fs: float):
    phase = np.unwrap(np.angle(iq))
    inst_freq = np.diff(phase) / (2 * np.pi) * fs / 1e6  # MHz
    t = np.arange(len(inst_freq)) / fs * 1000
    fig, ax = plt.subplots(figsize=(7.5, 2.6), facecolor="#0a0c12")
    ax.set_facecolor("#0a0c12")
    ax.plot(t, inst_freq, color="#f4c95d", linewidth=0.9, alpha=0.95)
    ax.set_xlabel("Time (ms)", color="#8a8f9e", fontsize=9)
    ax.set_ylabel("Instantaneous Freq (MHz)", color="#8a8f9e", fontsize=9)
    ax.tick_params(colors="#8a8f9e")
    for spine in ax.spines.values():
        spine.set_color("#252a38")
    ax.set_title("Instantaneous Frequency — The Actual Rollercoaster", color="#f4c95d", fontsize=11, pad=8, fontweight=600)
    fig.tight_layout()
    return fig


# =============================================================================
# UI SECTIONS
# =============================================================================

def render_header():
    st.markdown('<div class="hero-brand">', unsafe_allow_html=True)
    st.markdown('<h1 class="main-header">rolleRcoasteR</h1>', unsafe_allow_html=True)
    st.markdown('<p class="tagline">The rollercoaster for radio waves that have never existed</p>', unsafe_allow_html=True)
    st.markdown('<div style="margin-top:0.4rem; font-size:0.9rem; color:#5f6472; letter-spacing:0.12em;">FORGING NEW RADIO FOR HUMAN CIVILIZATION</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)


def render_mission_section():
    st.markdown('<div style="margin:0.3rem 0 0.1rem;"><span style="font-size:0.78rem; color:#00f0ff; font-weight:700; letter-spacing:0.08em;">rolleRcoasteR MISSION</span></div>', unsafe_allow_html=True)
    st.subheader("1. Name the Human Problem")

    presets = [
        "Resilient low-power mesh for storm-surge threatened coastal villages",
        "Simultaneous wildlife tracking + anti-poaching data relay in dense rainforest",
        "Real-time permafrost thaw monitoring swarm across the Arctic with minimal energy",
        "High-Doppler, eclipse-resilient surface-to-orbit relay for lunar operations",
        "Through-foliage, low-intercept sensor network for humanitarian demining",
    ]

    cols = st.columns(5)
    for i, preset in enumerate(presets):
        if cols[i].button(preset.split()[0] + "…", key=f"preset_{i}", use_container_width=True):
            st.session_state.last_mission = preset
            st.rerun()

    mission = st.text_area(
        "Describe the civilizational challenge in your own words",
        value=st.session_state.get("last_mission", ""),
        height=110,
        placeholder="e.g. Ultra-low power sensors that must also map flooding in real time for river delta communities with almost no infrastructure...",
        key="mission_input",
    )

    col1, col2, col3 = st.columns([1.1, 1.1, 2.8])
    with col1:
        if st.button("🚀 LAUNCH THE RIDE", type="primary", use_container_width=True):
            if len(mission.strip()) < 12:
                st.error("Give the waves a real problem to solve — at least a sentence.")
            else:
                with st.spinner("The waves are dreaming... (this can take 20–40s)"):
                    concepts = llm.dream_waveforms(mission.strip())
                    st.session_state.concepts = concepts
                    st.session_state.last_mission = mission.strip()
                    st.session_state.current_forge = None
                    st.success("Four completely different radical philosophies are ready.")
                    st.rerun()
    with col2:
        if st.button("🎲 Pure Chaos (instant)", use_container_width=True):
            # Skip LLM entirely — forge a completely random exotic right now
            st.session_state.last_mission = mission.strip() or "Maximum surprise"
            with st.spinner("Generating pure mathematical chaos..."):
                iq = core.quick_synthesize_random_exotic(seed=None)
                fake_forge = {
                    "iq": iq,
                    "fs": core.DEFAULT_FS,
                    "duration": core.DEFAULT_DURATION,
                    "n_components": 7,
                    "metrics": core.full_metrics(iq, core.DEFAULT_FS),
                    "mission_weights": {"energy_efficiency": 0.33, "sensing_clarity": 0.34, "frequency_agility": 0.33},
                    "novelty_pressure": 1.8,
                    "concept": {"name": "Pure Mathematical Chaos", "core": "Completely random overlapping nonlinear chirplets with no optimization pressure."},
                    "mission": st.session_state.last_mission,
                    "timestamp": datetime.utcnow().isoformat() + "Z",
                }
                st.session_state.current_forge = fake_forge
                st.session_state.forge_history.append(fake_forge)
            st.rerun()
    with col3:
        st.caption("The system will invent waveform structures that have almost no precedent in the literature or deployed systems.")


def render_concepts_section():
    if not st.session_state.concepts:
        return

    st.markdown('<div style="margin:1.1rem 0 0.1rem;"><span style="font-size:0.78rem; color:#f4c95d; font-weight:700; letter-spacing:0.08em;">rolleRcoasteR WAVEFORMS</span></div>', unsafe_allow_html=True)
    st.subheader("2. Choose Your Car on the Rollercoaster")

    for idx, concept in enumerate(st.session_state.concepts):
        with st.container():
            st.markdown(f'<div class="ride-card">', unsafe_allow_html=True)
            st.markdown(f'<div class="wave-name">{concept.get("name", f"Wave {idx+1}")}</div>', unsafe_allow_html=True)

            c1, c2 = st.columns([3.2, 1])
            with c1:
                st.markdown(f"**Core behavior** — {concept.get('core', '')}")
                st.markdown(f"**Why this mission** — {concept.get('why', '')}")
                st.markdown(f"**How it could be built** — {concept.get('build', '')}")
                st.markdown(f"<span style='color:#f4c95d'>The hard part:</span> {concept.get('dark', '')}", unsafe_allow_html=True)

            with c2:
                if st.button("🎢 Board This Car", key=f"board_{idx}", use_container_width=True):
                    _forge_and_show(concept, idx)
            st.markdown('</div>', unsafe_allow_html=True)


def _get_opt_settings():
    """Return lighter settings when Fast Preview mode is enabled (great for web/Colab)."""
    fast = st.session_state.get("fast_mode", True)
    if fast:
        return {"maxiter": 4, "popsize": 6, "polish": False}
    else:
        return {"maxiter": 8, "popsize": 11, "polish": True}


def _forge_and_show(concept: dict, concept_idx: int):
    """Actually run the optimizer and store the result."""
    mission = st.session_state.last_mission or "General civilizational advancement"

    # Tune weights based on which concept the user picked (adds personality)
    base_weights = {"energy_efficiency": 0.32, "sensing_clarity": 0.41, "frequency_agility": 0.27}
    if "low" in concept.get("name", "").lower() or "whisper" in concept.get("name", "").lower():
        base_weights = {"energy_efficiency": 0.58, "sensing_clarity": 0.27, "frequency_agility": 0.15}
    if "fractal" in concept.get("name", "").lower() or "lattice" in concept.get("name", "").lower():
        base_weights = {"energy_efficiency": 0.25, "sensing_clarity": 0.48, "frequency_agility": 0.27}

    settings = _get_opt_settings()
    spinner_text = "Forging (fast preview mode)..." if st.session_state.get("fast_mode") else f"Forging **{concept['name']}** — this is where the new physics actually appears (15–35s)..."
    
    with st.spinner(spinner_text):
        forge = core.forge_waveform(
            mission_weights=base_weights,
            fs=core.DEFAULT_FS,
            duration=0.0065,
            n_components=8,
            novelty_pressure=1.22,
            maxiter=settings["maxiter"],
            popsize=settings["popsize"],
            polish=settings["polish"],
            seed=42 + concept_idx,
        )

    forge["concept"] = concept
    forge["mission"] = mission
    forge["concept_idx"] = concept_idx

    st.session_state.current_forge = forge
    st.session_state.forge_history.append(forge)
    st.rerun()


def render_analysis_section():
    forge = st.session_state.current_forge
    if not forge:
        return

    st.divider()
    st.markdown(f"""
    <div style="margin-bottom:0.6rem;">
        <span style="background:#1a1f2b; color:#00f0ff; padding:2px 10px; border-radius:999px; font-size:0.72rem; font-weight:700; letter-spacing:0.06em;">rolleRcoasteR</span>
    </div>
    """, unsafe_allow_html=True)
    st.subheader(f"3. The Ride: {forge['concept']['name']}")

    iq = forge["iq"]
    fs = forge["fs"]
    m = forge["metrics"]

    # Big metrics row
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        st.markdown(f'<div class="metric-label">Novelty Score</div><div class="metric-big">{m["novelty"]:.2f}</div>', unsafe_allow_html=True)
    with c2:
        st.markdown(f'<div class="metric-label">Mission Utility</div><div class="metric-big">{m["mission_utility"]:.2f}</div>', unsafe_allow_html=True)
    with c3:
        st.markdown(f'<div class="metric-label">PAPR</div><div class="metric-big">{m["papr_db"]:.1f} dB</div>', unsafe_allow_html=True)
    with c4:
        st.markdown(f'<div class="metric-label">Sidelobe Level</div><div class="metric-big">{m["ambiguity_sidelobe_level"]:.2f}</div>', unsafe_allow_html=True)
    with c5:
        st.markdown(f'<div class="metric-label">Duration</div><div class="metric-big">{m["duration_s"]*1000:.1f} ms</div>', unsafe_allow_html=True)

    st.caption("Higher novelty = more alien to all known modulation families. Utility and sidelobe are proxies — real performance needs real channels.")

    # Visualizations
    tab1, tab2, tab3, tab4 = st.tabs(["Spectrogram", "Instantaneous Freq", "Time Domain + PSD", "Ambiguity (Sensing)"])

    with tab1:
        st.markdown('<div class="plot-container">', unsafe_allow_html=True)
        st.pyplot(plot_spectrogram(iq, fs), use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
        st.caption("This is usually where people say 'holy shit'. The time-frequency structure is the actual new invention.")

    with tab2:
        st.markdown('<div class="plot-container">', unsafe_allow_html=True)
        st.pyplot(plot_instantaneous(iq, fs), use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with tab3:
        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown('<div class="plot-container">', unsafe_allow_html=True)
            st.pyplot(plot_waveform(iq, fs), use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
        with col_b:
            st.markdown('<div class="plot-container">', unsafe_allow_html=True)
            st.pyplot(plot_psd(iq, fs), use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

    with tab4:
        st.markdown('<div class="plot-container">', unsafe_allow_html=True)
        st.pyplot(plot_ambiguity(iq), use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
        st.caption("Clean mainlobe + low integrated sidelobes = good ranging and multipath resistance. This waveform was explicitly optimized for it.")

    # Action buttons
    st.write("")
    colx, coly, colz = st.columns([1.6, 1.6, 2.8])
    with colx:
        if st.button("🔥 Make It Wilder (more novelty)", use_container_width=True):
            _re_optimize(forge, novelty_delta=+0.35)
    with coly:
        if st.button("🧊 Make It More Civilized (better utility)", use_container_width=True):
            _re_optimize(forge, novelty_delta=-0.25)
    with colz:
        if st.button("💾 EXPORT COMPLETE DISCOVERY BUNDLE", type="primary", use_container_width=True):
            _do_export(forge)


def _re_optimize(forge: dict, novelty_delta: float):
    settings = _get_opt_settings()
    spinner_text = "Re-optimizing in fast preview mode..." if st.session_state.get("fast_mode") else "Re-optimizing the same philosophy with new pressure..."
    with st.spinner(spinner_text):
        new_forge = core.forge_waveform(
            mission_weights=forge["mission_weights"],
            fs=forge["fs"],
            duration=forge["duration"],
            n_components=forge["n_components"],
            novelty_pressure=max(0.7, min(2.1, forge["novelty_pressure"] + novelty_delta)),
            maxiter=settings["maxiter"],
            popsize=settings["popsize"],
            polish=settings["polish"],
            seed=None,
        )
    new_forge["concept"] = forge["concept"]
    new_forge["mission"] = forge["mission"]
    new_forge["concept_idx"] = forge.get("concept_idx", 0)
    st.session_state.current_forge = new_forge
    st.session_state.forge_history.append(new_forge)
    st.rerun()


def _do_export(forge: dict):
    path = core.export_discovery(forge, out_dir="exports")
    st.success(f"**rolleRcoasteR** Discovery exported to `{path}`")
    st.info("You now have `iq.npy` (complex64), full metadata JSON, all plots, and a README. This is ready to take to a software-defined radio.")

    # Offer a tiny usage snippet
    with st.expander("Quick Python snippet to load and inspect"):
        st.code(f"""
import numpy as np
iq = np.load("{path}/iq.npy")
print(iq.shape, iq.dtype)
print("Peak amplitude:", np.max(np.abs(iq)))
# Next steps: upconvert, filter, send to HackRF/USRP/Pluto, etc.
""", language="python")


def render_history_and_disclaimer():
    if st.session_state.forge_history:
        with st.expander(f"📜 rolleRcoasteR History — {len(st.session_state.forge_history)} previous rides"):
            for i, f in reversed(list(enumerate(st.session_state.forge_history))):
                name = f.get("concept", {}).get("name", "Unknown")
                nov = f["metrics"]["novelty"]
                if st.button(f"{name} — novelty {nov:.2f}", key=f"hist_{i}"):
                    st.session_state.current_forge = f
                    st.rerun()

    st.markdown("""
<div class="disclaimer">
<b>Critical reality check:</b> These are simulated mathematical objects. They have never been transmitted. 
They may violate your power amplifier, your regulatory license, or the laws of physics in subtle ways once they leave the computer. 
Use them as extremely interesting starting points for real engineering, not as finished products.
</div>
""", unsafe_allow_html=True)


# =============================================================================
# MAIN
# =============================================================================

def main():
    render_header()

    # Sidebar
    with st.sidebar:
        st.markdown("""
        <div style="padding:0.6rem 0 1rem; border-bottom:1px solid #252a38; margin-bottom:1rem;">
            <div style="font-size:1.65rem; font-weight:800; background: linear-gradient(90deg, #00f0ff, #f4c95d); -webkit-background-clip: text; -webkit-text-fill-color: transparent; line-height:1.1;">
                rolleRcoasteR
            </div>
            <div style="font-size:0.72rem; color:#5f6472; letter-spacing:0.08em; margin-top:0.1rem;">NEW RADIO INVENTED HERE</div>
        </div>
        """, unsafe_allow_html=True)

        st.toggle("⚡ Fast preview mode (recommended for web/Colab)", 
                  value=True, 
                  key="fast_mode",
                  help="Reduces optimization steps for much faster rides on CPU-only environments like Colab or Streamlit Cloud.")

        st.markdown("### Controls")
        st.caption("All synthesis runs locally. No data leaves your machine.")
        st.markdown("---")
        st.markdown("**Current defaults**")
        st.code(f"""
fs = {core.DEFAULT_FS/1e6} MHz
duration = {core.DEFAULT_DURATION*1000:.1f} ms
components = {core.DEFAULT_N_COMPONENTS}
        """, language="text")
        st.markdown("---")
        if st.button("Reset Everything", use_container_width=True):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()

        st.markdown("<br><br>", unsafe_allow_html=True)
        st.markdown("### What makes a wave 'new'?")
        st.caption("We optimize against an explicit novelty loss built from spectral shape, instantaneous frequency statistics, and amplitude behavior. The optimizer is heavily penalized for producing anything that looks like classical modulations.")

    render_mission_section()
    render_concepts_section()
    render_analysis_section()
    render_history_and_disclaimer()

    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align:center; font-size:0.95rem; color:#8a8f9e; margin:0.8rem 0;">
        <strong style="color:#f4c95d;">rolleRcoasteR</strong> — New radio waves for human civilization.<br>
        <span style="font-size:0.78rem;">Everything is simulated until you prove otherwise on real hardware.</span>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()