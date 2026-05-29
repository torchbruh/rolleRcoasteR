"""
rolleRcoasteR — Core Synthesis & Analysis Engine

This module is responsible for actually *creating* radio waveforms that have
never existed. It does this through a powerful, flexible multi-chirplet
parameterization + global optimization that explicitly rewards novelty while
respecting mission objectives.

No legacy modulation families are used as base classes. Everything is built
from overlapping nonlinear chirplets with independent envelopes.
"""

from __future__ import annotations
import numpy as np
from scipy import signal
from scipy.optimize import differential_evolution, minimize
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple, Optional, Callable
import json
from datetime import datetime
import os


# =============================================================================
# PHYSICAL CONSTANTS & DEFAULTS
# =============================================================================

DEFAULT_FS = 2_000_000          # 2 MHz — good for most interesting waveforms
DEFAULT_DURATION = 0.008        # 8 ms — long enough to see structure, short for speed
DEFAULT_N_COMPONENTS = 9        # Sweet spot for complexity vs speed on CPU


# =============================================================================
# WAVEFORM PARAMETERIZATION (The Rollercoaster Track)
# =============================================================================

@dataclass
class ChirpletParams:
    """One flexible nonlinear chirplet component."""
    t_center: float      # 0..1 normalized time
    duration: float      # 0.01..0.6 normalized
    f_start: float       # -0.5..0.5 normalized freq
    f_end: float         # -0.5..0.5 normalized freq
    curvature: float     # -1.5..1.5 (0 = linear chirp, >0 = quadratic bend)
    amp: float           # log-amplitude (we exponentiate later)
    env_shape: float     # 0.3..3.0 controls envelope peakedness (1.0 = Gaussian-ish)


def params_to_vector(params: List[ChirpletParams]) -> np.ndarray:
    """Flatten list of chirplets into 1D vector for the optimizer."""
    vec = []
    for p in params:
        vec.extend([
            p.t_center, p.duration, p.f_start, p.f_end,
            p.curvature, p.amp, p.env_shape
        ])
    return np.array(vec)


def vector_to_params(vec: np.ndarray, n_comp: int) -> List[ChirpletParams]:
    """Inverse of above."""
    params = []
    for i in range(n_comp):
        base = i * 7
        params.append(ChirpletParams(
            t_center=float(np.clip(vec[base + 0], 0.0, 1.0)),
            duration=float(np.clip(vec[base + 1], 0.02, 0.55)),
            f_start=float(np.clip(vec[base + 2], -0.48, 0.48)),
            f_end=float(np.clip(vec[base + 3], -0.48, 0.48)),
            curvature=float(np.clip(vec[base + 4], -1.4, 1.4)),
            amp=float(np.clip(vec[base + 5], -2.5, 1.8)),
            env_shape=float(np.clip(vec[base + 6], 0.35, 2.8)),
        ))
    return params


# =============================================================================
# THE ACTUAL WAVEFORM GENERATOR
# =============================================================================

def synthesize_waveform(
    params: List[ChirpletParams],
    fs: float = DEFAULT_FS,
    duration: float = DEFAULT_DURATION,
    phase_noise: float = 0.0,
) -> np.ndarray:
    """
    Synthesize a complex baseband waveform from the rollercoaster parameters.

    This is the heart of the "new radio wave" generator. No OFDM, no PSK,
    no textbook pulses. Only overlapping, independently sculpted nonlinear
    chirplets whose instantaneous frequency and amplitude trajectories can
    do almost anything.
    """
    n = int(fs * duration)
    t = np.arange(n) / fs
    t_norm = t / duration  # 0..1

    iq = np.zeros(n, dtype=np.complex128)

    for p in params:
        # Effective support of this chirplet
        sigma = p.duration * duration * 0.5
        env = np.exp(-((t_norm - p.t_center) ** 2) / (2 * (sigma * p.env_shape) ** 2))

        # Instantaneous frequency law (linear + quadratic curvature)
        # This is where the magic happens — we can make the frequency accelerate,
        # decelerate, or even fold back on itself.
        delta_f = p.f_end - p.f_start
        inst_freq_norm = (
            p.f_start +
            delta_f * ((t_norm - p.t_center + p.duration * 0.5) / max(p.duration, 1e-6)) +
            p.curvature * ((t_norm - p.t_center) ** 2) / max(p.duration, 1e-6)
        )
        # Clip to legal digital frequencies
        inst_freq_norm = np.clip(inst_freq_norm, -0.49, 0.49)

        # Phase = integral of 2*pi*freq
        phase = 2 * np.pi * np.cumsum(inst_freq_norm) * (duration / n)

        if phase_noise > 0:
            phase += np.cumsum(np.random.randn(n)) * phase_noise * 0.01

        component = env * np.exp(1j * phase) * (10 ** p.amp)
        iq += component

    # Gentle normalization so we don't explode PAPR before optimization even starts
    peak = np.max(np.abs(iq))
    if peak > 1e-9:
        iq = iq / peak * 0.92

    return iq.astype(np.complex64)


# =============================================================================
# METRICS — HOW WE SCORE A WAVE (AND FORCE NOVELTY)
# =============================================================================

def compute_papr(iq: np.ndarray) -> float:
    """Peak to Average Power Ratio in dB."""
    power = np.abs(iq) ** 2
    return 10 * np.log10(np.max(power) / (np.mean(power) + 1e-12))


def compute_instantaneous_metrics(iq: np.ndarray, fs: float) -> Dict:
    """Instantaneous frequency and amplitude statistics."""
    # Unwrapped phase derivative = instantaneous frequency
    phase = np.unwrap(np.angle(iq))
    inst_freq = np.diff(phase) / (2 * np.pi) * fs
    inst_amp = np.abs(iq)

    return {
        "inst_freq_mean": float(np.mean(inst_freq)),
        "inst_freq_std": float(np.std(inst_freq)),
        "inst_freq_range": float(np.ptp(inst_freq)),
        "amp_kurtosis": float(_kurtosis(inst_amp)),
        "amp_variation": float(np.std(inst_amp) / (np.mean(inst_amp) + 1e-9)),
    }


def _kurtosis(x: np.ndarray) -> float:
    x = x - np.mean(x)
    return np.mean(x**4) / (np.std(x)**2 + 1e-12)**2 - 3.0


def estimate_spectrum_novelty(iq: np.ndarray, fs: float) -> float:
    """
    A cheap but surprisingly effective novelty score.
    High when the spectrum has unusual shape (many peaks, holes, asymmetry)
    that standard modulations rarely produce.
    """
    f, Pxx = signal.welch(iq, fs=fs, nperseg=min(1024, len(iq)//2), scaling='spectrum')
    Pxx = np.abs(Pxx) + 1e-12
    Pxx /= np.sum(Pxx)

    # Spectral entropy (high = flatter/more noise-like, but we want *structured* weirdness)
    entropy = -np.sum(Pxx * np.log(Pxx + 1e-12))

    # Count significant local peaks in the spectrum (more = more "rollercoastery")
    peaks, _ = signal.find_peaks(10 * np.log10(Pxx), height=-35, distance=8)
    peak_density = len(peaks) / max(len(Pxx), 1)

    # Spectral flatness in sub-bands (low flatness in some bands = structured)
    n_bands = 6
    band_scores = []
    for i in range(n_bands):
        lo, hi = int(len(Pxx) * i / n_bands), int(len(Pxx) * (i + 1) / n_bands)
        band = Pxx[lo:hi]
        if len(band) > 4:
            gmean = np.exp(np.mean(np.log(band)))
            amean = np.mean(band)
            band_scores.append(gmean / (amean + 1e-12))

    weirdness = (entropy / 8.0) + (peak_density * 2.8) + (1.0 - np.mean(band_scores))

    return float(np.clip(weirdness, 0.0, 4.5))


def compute_ambiguity_sidelobes(iq: np.ndarray, fs: float, max_lag: Optional[int] = None) -> float:
    """
    Compute a proxy for the integrated sidelobe level of the ambiguity function.
    Lower is better for sensing (cleaner range-Doppler).
    """
    n = len(iq)
    if max_lag is None:
        max_lag = min(256, n // 4)

    # Zero-padded autocorrelation (range dimension at zero Doppler)
    acf = signal.correlate(iq, iq, mode='full')
    acf = acf[n-1-max_lag : n-1+max_lag+1]
    acf = np.abs(acf) ** 2
    acf /= (np.max(acf) + 1e-12)

    # Sidelobe energy outside mainlobe (mainlobe ~ 3 samples wide for well-behaved signals)
    main = max_lag + 1
    mainlobe = acf[main-2:main+3].sum()
    sidelobe_energy = acf.sum() - mainlobe

    return float(sidelobe_energy / (acf.sum() + 1e-12))


def compute_novelty_score(iq: np.ndarray, fs: float) -> float:
    """
    Composite "how much does this look like nothing we've seen before?"
    This is the secret sauce that pushes the optimizer into genuinely new territory.
    """
    spec_novel = estimate_spectrum_novelty(iq, fs)
    inst = compute_instantaneous_metrics(iq, fs)

    # High variation in instantaneous frequency + amplitude that is *not* constant modulus
    # is rare in classical digital modulations.
    score = (
        0.55 * spec_novel +
        0.22 * np.clip(inst["inst_freq_std"] / (fs * 0.25), 0, 1.8) +
        0.15 * np.clip(inst["amp_variation"], 0, 1.4) +
        0.08 * np.clip((inst["amp_kurtosis"] + 1.5) / 5.0, 0, 1.0)
    )
    return float(np.clip(score, 0.0, 5.0))


def compute_mission_utility(iq: np.ndarray, fs: float, weights: Dict[str, float]) -> float:
    """
    How well does this waveform serve the stated human mission?
    This is deliberately a *proxy* — real utility requires real channels.
    """
    papr = compute_papr(iq)
    sidelobe = compute_ambiguity_sidelobes(iq, fs)
    inst = compute_instantaneous_metrics(iq, fs)

    # Lower PAPR is generally better for power amplifiers (civilization loves efficiency)
    papr_score = np.clip(1.0 - (papr - 3.0) / 12.0, 0.0, 1.0)

    # Lower sidelobes = better sensing / ranging / multipath resistance
    sensing_score = np.clip(1.0 - sidelobe * 1.8, 0.0, 1.0)

    # Some missions want high frequency agility (high inst_freq_std)
    agility_score = np.clip(inst["inst_freq_std"] / (fs * 0.18), 0.0, 1.0)

    utility = (
        weights.get("energy_efficiency", 0.3) * papr_score +
        weights.get("sensing_clarity", 0.4) * sensing_score +
        weights.get("frequency_agility", 0.3) * agility_score
    )
    return float(np.clip(utility, 0.0, 1.0))


def full_metrics(iq: np.ndarray, fs: float, mission_weights: Optional[Dict] = None) -> Dict:
    """Everything we can compute about a waveform in one convenient dict."""
    if mission_weights is None:
        mission_weights = {"energy_efficiency": 0.35, "sensing_clarity": 0.4, "frequency_agility": 0.25}

    return {
        "papr_db": compute_papr(iq),
        "novelty": compute_novelty_score(iq, fs),
        "ambiguity_sidelobe_level": compute_ambiguity_sidelobes(iq, fs),
        "mission_utility": compute_mission_utility(iq, fs, mission_weights),
        "length_samples": int(len(iq)),
        "duration_s": float(len(iq) / fs),
        "fs": float(fs),
        **compute_instantaneous_metrics(iq, fs),
    }


# =============================================================================
# THE OPTIMIZER — WHERE NEW WAVES ARE ACTUALLY BORN
# =============================================================================

def _objective(
    vec: np.ndarray,
    n_comp: int,
    fs: float,
    duration: float,
    mission_weights: Dict[str, float],
    novelty_pressure: float,
) -> float:
    """The loss function the optimizer is trying to *minimize*."""
    params = vector_to_params(vec, n_comp)
    iq = synthesize_waveform(params, fs=fs, duration=duration)

    novelty = compute_novelty_score(iq, fs)
    utility = compute_mission_utility(iq, fs, mission_weights)
    papr = compute_papr(iq)
    sidelobe = compute_ambiguity_sidelobes(iq, fs)

    # We want HIGH novelty + HIGH utility + LOW papr + LOW sidelobes
    loss = (
        -0.38 * novelty * novelty_pressure +
        -0.42 * utility +
        0.11 * (papr - 4.5) / 9.0 +
        0.09 * sidelobe * 1.6
    )
    return float(loss)


def forge_waveform(
    mission_weights: Dict[str, float],
    fs: float = DEFAULT_FS,
    duration: float = DEFAULT_DURATION,
    n_components: int = DEFAULT_N_COMPONENTS,
    novelty_pressure: float = 1.15,
    popsize: int = 11,
    maxiter: int = 9,
    polish: bool = True,
    seed: Optional[int] = None,
) -> Dict:
    """
    The main public API for creating a genuinely new radio waveform.

    Returns a rich dict containing the final I/Q, the parameters that produced it,
    all metrics, and the full optimization trajectory.
    """
    rng = np.random.default_rng(seed)
    n_params = n_components * 7

    # Wide but physically reasonable bounds
    bounds = []
    for _ in range(n_components):
        bounds.extend([
            (0.03, 0.97),      # t_center
            (0.035, 0.48),     # duration
            (-0.46, 0.46),     # f_start
            (-0.46, 0.46),     # f_end
            (-1.25, 1.25),     # curvature
            (-2.1, 1.55),      # amp
            (0.4, 2.55),       # env_shape
        ])

    # Global search
    print(f"[rolleRcoasteR] Forging with {n_components} components, novelty_pressure={novelty_pressure:.2f}...")
    res = differential_evolution(
        _objective,
        bounds,
        args=(n_components, fs, duration, mission_weights, novelty_pressure),
        popsize=popsize,
        maxiter=maxiter,
        mutation=(0.6, 1.8),
        recombination=0.7,
        workers=1,
        seed=seed,
        updating='immediate',
        disp=False,
        polish=False,
    )

    best_vec = res.x
    history = [{"generation": 0, "loss": float(res.fun)}]

    # Optional local polish
    if polish:
        polish_res = minimize(
            _objective,
            best_vec,
            args=(n_components, fs, duration, mission_weights, novelty_pressure),
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 28, 'ftol': 1e-7},
        )
        if polish_res.success:
            best_vec = polish_res.x

    final_params = vector_to_params(best_vec, n_components)
    final_iq = synthesize_waveform(final_params, fs=fs, duration=duration)

    metrics = full_metrics(final_iq, fs, mission_weights)

    return {
        "iq": final_iq,
        "params": [asdict(p) for p in final_params],
        "metrics": metrics,
        "mission_weights": mission_weights,
        "novelty_pressure": float(novelty_pressure),
        "fs": float(fs),
        "duration": float(duration),
        "n_components": int(n_components),
        "optimization": {
            "final_loss": float(res.fun),
            "generations": int(res.nit),
            "success": bool(res.success),
        },
        "timestamp": datetime.utcnow().isoformat() + "Z",
    }


# =============================================================================
# CONVENIENCE: QUICK "WILD" STARTING POINTS (for when you want chaos fast)
# =============================================================================

def random_exotic_start(
    n_components: int = 7,
    seed: Optional[int] = None
) -> List[ChirpletParams]:
    """Create a completely random, usually quite strange starting waveform."""
    rng = np.random.default_rng(seed)
    params = []
    for _ in range(n_components):
        params.append(ChirpletParams(
            t_center=rng.uniform(0.08, 0.92),
            duration=rng.uniform(0.06, 0.38),
            f_start=rng.uniform(-0.43, 0.43),
            f_end=rng.uniform(-0.43, 0.43),
            curvature=rng.uniform(-1.1, 1.1) * rng.choice([-1, 1]) * rng.uniform(0.6, 1.0),
            amp=rng.uniform(-1.6, 1.1),
            env_shape=rng.uniform(0.55, 2.3),
        ))
    return params


def quick_synthesize_random_exotic(
    fs: float = DEFAULT_FS,
    duration: float = DEFAULT_DURATION,
    n_components: int = 7,
    seed: Optional[int] = None,
) -> np.ndarray:
    """Fast path when you just want something weird right now (no optimization)."""
    p = random_exotic_start(n_components, seed)
    return synthesize_waveform(p, fs=fs, duration=duration)


# =============================================================================
# VISUALIZATION HELPERS (used by the Streamlit app)
# =============================================================================

def compute_spectrogram(iq: np.ndarray, fs: float, nperseg: int = 256) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return f, t, Sxx (dB) for plotting."""
    f, t, Sxx = signal.spectrogram(iq, fs=fs, nperseg=nperseg, noverlap=nperseg//2, scaling='spectrum')
    Sxx_db = 10 * np.log10(Sxx + 1e-12)
    return f, t, Sxx_db


def compute_psd(iq: np.ndarray, fs: float) -> Tuple[np.ndarray, np.ndarray]:
    f, Pxx = signal.welch(iq, fs=fs, nperseg=512, scaling='spectrum')
    return f, 10 * np.log10(Pxx + 1e-12)


def compute_ambiguity_slice(iq: np.ndarray, max_lag: int = 180) -> np.ndarray:
    """Zero-Doppler cut of the ambiguity function (range profile)."""
    n = len(iq)
    acf = signal.correlate(iq, iq, mode='full')
    center = n - 1
    return np.abs(acf[center - max_lag : center + max_lag + 1]) ** 2


# =============================================================================
# EXPORT BUNDLE
# =============================================================================

def export_discovery(bundle: Dict, out_dir: str = "exports") -> str:
    """
    Save a complete discovery to disk so the user can take it to real hardware.
    Returns the path to the created directory.
    """
    os.makedirs(out_dir, exist_ok=True)
    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    name = f"rolleRcoasteR_{ts}"
    path = os.path.join(out_dir, name)
    os.makedirs(path, exist_ok=False)

    # Core signal
    np.save(os.path.join(path, "iq.npy"), bundle["iq"])

    # Everything else (JSON serializable)
    meta = {
        k: v for k, v in bundle.items()
        if k not in ("iq",)  # don't duplicate the array in JSON
    }
    with open(os.path.join(path, "metadata.json"), "w") as f:
        json.dump(meta, f, indent=2)

    # Human readable summary
    with open(os.path.join(path, "README.txt"), "w") as f:
        f.write("rolleRcoasteR Discovery\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Created: {bundle.get('timestamp')}\n")
        f.write(f"Duration: {bundle['duration']*1000:.1f} ms @ {bundle['fs']/1e6:.2f} MHz\n")
        f.write(f"Novelty Score: {bundle['metrics']['novelty']:.3f}\n")
        f.write(f"Mission Utility: {bundle['metrics']['mission_utility']:.3f}\n")
        f.write(f"PAPR: {bundle['metrics']['papr_db']:.1f} dB\n\n")
        f.write("Load with: iq = np.load('iq.npy')\n")
        f.write("This is complex baseband (float32 complex64). Upconvert as needed.\n")

    return path


if __name__ == "__main__":
    # Smoke test
    print("rolleRcoasteR core smoke test...")
    iq = quick_synthesize_random_exotic(seed=42)
    print(f"Generated {len(iq)} samples, PAPR={compute_papr(iq):.1f} dB, novelty={compute_novelty_score(iq, DEFAULT_FS):.2f}")

    forged = forge_waveform(
        mission_weights={"energy_efficiency": 0.4, "sensing_clarity": 0.45, "frequency_agility": 0.15},
        maxiter=4,
        seed=7,
    )
    print(f"Forged! Novelty={forged['metrics']['novelty']:.2f}, Utility={forged['metrics']['mission_utility']:.2f}")
    p = export_discovery(forged)
    print(f"Exported to {p}")