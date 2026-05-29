"""
rolleRcoasteR — LLM Ideation Layer

The "Dreamer" that proposes completely new radio waveform philosophies
tailored to a specific civilizational mission.

When Ollama is available it uses it with a very strong prompt.
When not, it falls back to a high-quality curated bank of radical concepts.
"""

from __future__ import annotations
from typing import List, Dict, Optional
import random


# =============================================================================
# THE PROMPT — This is where the magic (or mediocrity) lives
# =============================================================================

SYSTEM_PROMPT = """You are an extremely creative, first-principles radio physicist and humanitarian technologist who has grown bored with every modulation scheme invented in the last 120 years.

Your job is to invent *radically new* radio waveform philosophies — not tweaks, not combinations of existing families, but genuinely alien signal structures that have almost never been deliberately engineered.

You are given a concrete civilizational mission. You must propose 4 completely different waveform "personalities" that could serve that mission better than anything currently deployed, precisely because they were too strange or too hard to analyze with 20th-century mathematics.

Rules for every proposal:
- Never mention OFDM, QAM, PSK, FSK, LoRa, chirp spread spectrum, or any named classical modulation as a base.
- Describe the waveform in terms of its *time-frequency behavior*, energy distribution, and how information or sensing capability is physically encoded in its shape.
- Explain *why* this structure is particularly well suited to the mission (physics reason, not marketing).
- Give it a short, evocative, slightly poetic name (2-4 words max).
- Be specific enough that an engineer could start implementing a version of it tomorrow.
- Acknowledge one serious physical or practical drawback.

Output format (exactly 4 blocks, nothing else before or after):

WAVE 1: <Evocative Name>
Core Idea: <one vivid sentence about the physical behavior>
Why It Wins Here: <2-3 sentences of first-principles reasoning tied to the mission>
How to Build It: <concrete parameterization or synthesis approach>
The Dark Side: <real drawback that must be engineered around>

(Repeat for WAVE 2, 3, 4 — make them as different from each other as possible. One should feel almost biological, one mathematical/exotic, one almost maliciously efficient, one deliberately "wasteful" in a clever way.)"""

USER_PROMPT_TEMPLATE = """Civilizational Mission:
{mission}

Invent four genuinely new radio waveforms for this exact problem. Think like someone who has never read a communications textbook."""


# =============================================================================
# FALLBACK CONCEPT BANK (used when Ollama is unavailable or slow)
# =============================================================================

FALLBACK_CONCEPTS = [
    {
        "name": "Ghost Current",
        "core": "A waveform whose energy exists primarily in the *derivative* of its phase, creating near-zero average power most of the time but massive instantaneous frequency swings that carry information in their timing and curvature.",
        "why": "For ultra-low power sensor meshes in energy-starved environments, the transmitter only expends energy during the rare, violent frequency events. Most of the time the amplifier is nearly off.",
        "build": "Sum of 5-9 extremely short, high-curvature quadratic chirplets whose positions and bend directions encode both data and a built-in ranging sequence.",
        "dark": "Extremely high PAPR and brutal sensitivity to any nonlinearity in the power amplifier.",
    },
    {
        "name": "Fractal Breath",
        "core": "Self-similar frequency trajectories at three nested time scales, so that the same underlying geometry is visible whether you look at 200 µs or 40 ms windows.",
        "why": "In highly time-varying channels (drone swarms, ocean surface, moving foliage), the receiver can lock onto the structure at whatever scale is currently cleanest, giving graceful degradation instead of cliff effects.",
        "build": "Recursive construction: a slow base trajectory that is itself modulated by a medium-speed copy of a similar shape, which is modulated by a fast copy.",
        "dark": "Synchronization is nightmarish; you need a multi-scale PLL or a neural receiver from day one.",
    },
    {
        "name": "Soliton Choir",
        "core": "Several stable, particle-like pulses that interact nonlinearly as they propagate through the synthesized channel model, exchanging energy in ways that encode information in the *interaction pattern* rather than in any individual pulse.",
        "why": "For long-distance, low-power links where the channel itself can be treated as a computational medium. The 'computation' performed by soliton collisions can compress or error-protect data for free.",
        "build": "Carefully tuned sech-shaped envelopes with specific amplitude and velocity relationships so they repeatedly collide inside the observation window.",
        "dark": "Only works well in channels that approximately preserve the soliton property; real hardware nonlinearities can destroy the choreography.",
    },
    {
        "name": "Void Mapper",
        "core": "A waveform that deliberately creates deep, controllable spectral holes at specific frequencies and times, then uses the *shape and motion* of those artificial voids to carry information while the surrounding energy does continuous wideband sensing.",
        "why": "Perfect for cognitive or shared-spectrum humanitarian networks that must not only avoid primary users but actively map them in real time with the same transmission used for data.",
        "build": "Multi-component chirp train with precise, time-varying notches created by destructive superposition of paired components with 180° phase offset in chosen sub-bands.",
        "dark": "Requires extremely linear transmit chain and excellent power control; any compression fills the voids and destroys the sensing mode.",
    },
    {
        "name": "Echo Lattice",
        "core": "A waveform engineered so that its own multipath reflections create a predictable, high-dimensional lattice of delayed and Doppler-shifted copies whose interference pattern at the receiver is the actual information-bearing structure.",
        "why": "In rich-scattering disaster or urban environments, traditional systems fight multipath. This one weaponizes it — the more reflections, the higher the effective dimensionality of the received 'constellation'.",
        "build": "Long, precisely sculpted train of low-amplitude chirplets whose delays and Doppler shifts are chosen so their natural propagation creates a known lattice geometry.",
        "dark": "Channel estimation becomes everything; small changes in the environment rotate the entire received structure. Needs heavy online adaptation.",
    },
    {
        "name": "Thermal Whisper",
        "core": "Information is encoded almost entirely in the *microscopic* statistical fluctuations of an otherwise smooth, low-power noise-like waveform, below the level that conventional energy detectors would notice.",
        "why": "For extremely sensitive ecological or medical-adjacent monitoring where the RF emission itself must not disturb the thing being measured (animals, plants, human tissue).",
        "build": "Very long, extremely low PAPR Gaussian-like envelope whose higher-order statistics (skew, kurtosis trajectories, bispectrum) are slowly modulated.",
        "dark": "Requires long integration times and sophisticated higher-order statistical receivers. Not for low-latency applications.",
    },
]


def get_fallback_concepts(mission: str, k: int = 4) -> List[Dict]:
    """Return k diverse fallback concepts, lightly adapted to the mission."""
    chosen = random.sample(FALLBACK_CONCEPTS, k)
    # Light injection of mission language so it doesn't feel completely generic
    for c in chosen:
        c = c.copy()
        c["why"] = c["why"] + f" This becomes especially powerful when the mission requires {mission.split()[0].lower()}."
    return chosen


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

def dream_waveforms(mission: str, model: str = "llama3.2", timeout: int = 45) -> List[Dict]:
    """
    Ask the LLM (or fallback) for 4 radical new waveform concepts.

    Returns list of dicts with keys: name, core, why, build, dark
    """
    try:
        import ollama  # Lazy import so the package is truly optional
        response = ollama.chat(
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": USER_PROMPT_TEMPLATE.format(mission=mission)},
            ],
            options={"temperature": 0.82, "top_p": 0.95},
            stream=False,
        )
        text = response["message"]["content"]
        return _parse_ollama_output(text)
    except ImportError:
        print("[rolleRcoasteR] 'ollama' package not installed. Using curated radical concepts.")
        return get_fallback_concepts(mission, k=4)
    except Exception as e:
        print(f"[rolleRcoasteR] Ollama unavailable or failed ({e}). Using curated radical concepts.")
        return get_fallback_concepts(mission, k=4)


def _parse_ollama_output(text: str) -> List[Dict]:
    """Very forgiving parser for the strict 4-block format."""
    waves = []
    current = {}
    lines = text.strip().split("\n")

    for line in lines:
        line = line.strip()
        if not line:
            continue
        low = line.lower()
        if low.startswith("wave ") and ":" in line:
            if current:
                waves.append(current)
            current = {"name": line.split(":", 1)[1].strip()}
        elif low.startswith("core idea:"):
            current["core"] = line.split(":", 1)[1].strip()
        elif low.startswith("why it wins"):
            current["why"] = line.split(":", 1)[1].strip()
        elif low.startswith("how to build"):
            current["build"] = line.split(":", 1)[1].strip()
        elif low.startswith("the dark side"):
            current["dark"] = line.split(":", 1)[1].strip()

    if current:
        waves.append(current)

    # Fill any missing keys with something reasonable
    for w in waves:
        for key in ["core", "why", "build", "dark"]:
            w.setdefault(key, "—")

    return waves[:4] if waves else get_fallback_concepts("general purpose", 4)


if __name__ == "__main__":
    print("Testing dreamer...")
    concepts = dream_waveforms("low-power permafrost monitoring mesh in the Arctic")
    for c in concepts:
        print(f"\n{c['name']}\n  {c['core'][:80]}...")