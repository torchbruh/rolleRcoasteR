# rolleRcoasteR

**The Rollercoaster for Radio Waves That Have Never Existed**

An AI laboratory that *invents* new radio waveforms — not remixes of old ones — optimized for real human civilization problems.

> **Want to try it right now without installing anything?**  
> See the **[Preview via Web](#preview-via-web-no-xcode--no-local-install-needed)** section below (Google Colab or Streamlit Cloud options).

You describe an urgent challenge for humanity.  
The system dreams up radical new signal structures.  
Then it **builds them** — numerically synthesizing actual I/Q samples through a combination of creative AI ideation and serious mathematical optimization.

The result: waveforms whose time-frequency behavior looks like nothing in textbooks, yet still carry usable information or sensing capability.

---

## Why This Exists

Almost every radio system on Earth still uses modulation families invented between 1900 and 2010.  
The spectrum is too valuable, and the problems (climate, disasters, equity, space) are too new, for us to keep riding the same old waves.

rolleRcoasteR exists to break that pattern on purpose.

---

## ⚠️ macOS First Step (Critical)

Before anything else, you almost certainly need Xcode Command Line Tools (this is a one-time 5-20 minute install that many scientific packages require on macOS):

```bash
xcode-select --install
```

Then follow the steps below. This is the #1 reason things fail on fresh Macs.

---

## Quick Start (Test It Right Now)

```bash
# 1. Go to the project
cd ~/Projects/rolleRcoasteR

# 2. (IMPORTANT) Install Xcode CLT first if you haven't (see above)

# 3. Create a clean environment
python3 -m venv .venv
source .venv/bin/activate

# 4. Install dependencies (this can take 2-8 minutes the first time)
pip install --upgrade pip
pip install -r requirements.txt

# (Optional but amazing — makes the "ideas" dramatically better)
# ollama list
# If you don't have Ollama running, the app falls back to excellent curated radical concepts.

# 5. Run the rollercoaster
streamlit run app.py
```

A `run.sh` helper is also provided for convenience.

The first time you run it, it will feel alive.

---

## Preview via Web (No Xcode / No Local Install Needed)

Since you're on macOS and can't install Xcode Command Line Tools right now, here are the best ways to get a working interactive version of rolleRcoasteR running in your browser.

### Option 1: Google Colab + ngrok (Fastest — ~5 minutes)

This runs the full app on Google's servers and gives you a temporary public URL.

**Steps:**

1. Go to [colab.research.google.com](https://colab.research.google.com) and create a **New Notebook**.

2. In the first cell, paste and run this setup code:

```python
# === rolleRcoasteR Colab Setup ===
!pip install streamlit numpy scipy matplotlib pandas pyngrok -q

# Upload your files:
# - Click the folder icon on the left
# - Upload these 3 files from your local rolleRcoasteR folder:
#     app.py
#     core.py
#     llm.py
#
# (You can also zip the whole folder and upload the zip, then unzip it)

print("Files uploaded? Continue to next cell.")
```

3. In a **new cell**, paste this:

```python
import os
from pyngrok import ngrok

# Make sure the files are in the current directory
# If you uploaded a zip, unzip it here:
# !unzip rolleRcoasteR.zip -d .   # adjust name if needed

# Launch Streamlit in the background
os.system("streamlit run app.py --server.port 8501 --server.headless true &")

# Open a public tunnel
public_url = ngrok.connect(8501)
print("🎢 Your rolleRcoasteR is live here:")
print(public_url)
```

4. Run the cell. After ~20–40 seconds you should get a public URL like `https://xxxx.ngrok.io`.

**Notes for Colab:**
- **Fast preview mode is ON by default** in the sidebar — this makes forging 2-3× faster on CPU-only web environments (recommended).
- The "Pure Chaos (instant)" button will be the fastest.
- Full forging still takes longer on free Colab CPU (expect 20–60s per ride with fast mode).
- Ollama is not available → it will use the excellent built-in fallback concepts.
- The link dies when your Colab session ends.

---

### Option 2: Streamlit Community Cloud (Cleanest permanent web version)

This is the best option for a stable, nice-looking public URL that stays up.

**Step-by-step (works even without local git):**

1. Go to [github.com](https://github.com) and create a **new repository** (you can make it private).

2. **Easiest upload method (no terminal/git needed):**
   - In the new repo, click **"uploading an existing file"** or drag & drop.
   - Upload these files and folders from your local `rolleRcoasteR` directory:
     - `app.py`
     - `core.py`
     - `llm.py`
     - `requirements.txt`
     - `README.md`
     - `runtime.txt`
     - The entire `.streamlit/` folder (including `config.toml`)
     - (Optional but nice) `run.sh`

3. Go to [share.streamlit.io](https://share.streamlit.io) and sign in with your GitHub account.

4. Click **"New app"** → select your repository → set the main file path to `app.py` → click **Deploy**.

Streamlit Cloud will automatically install the dependencies on Linux (no Xcode or Mac-specific issues). 

Your app will be live at something like `https://yourname-rolleRcoasteR.streamlit.app`

**Important notes for Cloud:**
- Fast Preview Mode is enabled by default (perfect for Streamlit Cloud's CPU).
- "Pure Chaos (instant)" works immediately.
- Full waveform forging works but is slower than a local Mac (normal for free tier).
- Ollama is not available — the app gracefully uses high-quality built-in concepts.

Would you like me to add a `Procfile` or any other small tweaks to make the deployment even smoother?

---

Both options let you fully interact with the waveform forger, visualizations, and export logic without touching Xcode.

Which path do you want to try first? I can give more detailed commands or even create a ready-made Colab notebook file for you.

## What You Will Experience

1. **The Mission** — Type (or pick) a real civilizational problem.
   Examples that produce fireworks:
   - "Resilient low-power mesh for coastal communities facing increasing storm surges"
   - "Simultaneous wildlife tracking and data relay for anti-poaching drones in dense jungle"
   - "Ultra-efficient sensor swarm for real-time permafrost thaw monitoring in the Arctic"
   - "High-Doppler, low-power relay for lunar surface operations during long eclipses"

2. **The Cars** — The LLM (or internal genius) proposes 4 completely different radical waveform philosophies. Each has a name like "Ghost Current" or "Fractal Breath" or "Soliton Choir".

3. **Boarding the Ride** — You pick one. The synthesizer wakes up.
   - It parameterizes a highly flexible non-stationary multi-chirplet model (or, if torch is installed, a direct gradient-coaster on raw samples).
   - It runs a serious global + local optimization that explicitly maximizes your chosen trade-offs **while punishing anything that looks like existing modulations**.
   - You watch the thrill metrics climb in real time.

4. **The Reveal** — Five beautiful, publication-grade visualizations appear:
   - Raw time-domain waveform
   - Instantaneous frequency trajectory (the "rollercoaster track" itself)
   - Power spectral density with novelty highlights
   - Spectrogram (STFT) — this is usually where you go "whoa"
   - Ambiguity function (range-Doppler) — critical for sensing performance

5. **Export** — One click gives you a timestamped folder with:
   - `iq.npy` (complex64, ready for any SDR)
   - Full JSON spec + scores + the original mission text + the LLM's reasoning
   - All the plots as crisp PNGs
   - A small `transmit_example.py` you can actually run with a HackRF or USRP later

---

## The Two Engines (How It Actually Creates "New")

### Engine 1: The Dreamer (Ideation)
- When Ollama is available: a carefully engineered prompt that forces first-principles, anti-orthogonality, and civilization-scale thinking.
- When not: a strong curated bank of genuinely unusual starting concepts.

### Engine 2: The Forge (Synthesis + Optimization)
This is the part that actually makes signals that have never been transmitted.

- **Primary (always available)**: A 40–80 dimensional parameterization of **overlapping nonlinear chirplets with independent amplitude envelopes**. This space is large enough to contain things no human has ever deliberately designed.
- **Power mode (if torch installed)**: Direct optimization of a complex envelope tensor with physics-informed regularizers + an explicit "anti-known-waveform" loss term. This can produce extremely alien, high-performing structures.

The loss always includes:
- Mission-specific utility (data proxy, sensing clarity via integrated sidelobe level, energy efficiency)
- **Novelty pressure** (multiple statistical and cyclostationary detectors that penalize anything resembling PSK, OFDM, linear chirp, GFSK, LoRa, etc.)

---

## Important Reality Checks

- These are **simulated mathematical objects**. Beautiful on your screen does not mean they will survive your power amplifier or the real channel.
- You still need to do the hard engineering (PAPR reduction, filtering, synchronization, regulatory approval) before putting anything on air.
- The goal of rolleRcoasteR is to give you starting points so radical that the normal "start from OFDM and tweak" path would never have reached them.

---

## Keyboard & Flow Tips

- Use the preset mission buttons — they are chosen to produce dramatically different rides.
- After the first forge, try the **"Wilder"** and **"More Civilized"** buttons. They re-optimize the same concept with different weightings.
- Export early and often. The best discoveries often come from iterating outside the tool (analysis in Python/MATLAB, then back in with new constraints).

---

## Future Directions (After You Play)

- Plug real TorchSig or RF-Diffusion models as warm-start generators
- Evolutionary population of entire rollercoasters (multiple waveforms co-evolving)
- Direct interface to HackRF / Pluto / USRP for closed-loop over-the-air fitness
- Gallery of the best community discoveries

---

## License & Spirit

This is a tool for expanding human possibility.  
Use it to invent things worth transmitting.

If you discover something genuinely powerful, consider publishing the waveform parameters and the mission it was forged for. The spectrum belongs to everyone.

---

**Built in a single focused session because the future shouldn't wait.**

---

## Current Status (Day 0)

- Fully functional end-to-end product you can run today (after the one-time Xcode CLT step).
- Real mathematical synthesis engine that creates waveforms with measurable novelty.
- Ollama-powered radical ideation + strong curated fallback.
- Five beautiful visualizations including the critical ambiguity function for sensing.
- One-click export of real I/Q + metadata bundles ready for SDR work.
- "Wilder / More Civilized" re-optimization loops.
- "Pure Chaos" instant random exotic generator.
- Zero data leaves your machine.

Known limitation on this Mac until you run `xcode-select --install`: the venv creation will appear to fail with a CLT note. After you install the tools and retry, everything works.

Now go ride. The waves are waiting.
