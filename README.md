# Somnivex

**Autonomous generative art that lives on your screen.**

Somnivex is an open-source screensaver built on Gray-Scott reaction-diffusion — a mathematical model of two chemicals that react and spread across a grid. The tension between them produces spirals, worms, coral, chaos, and everything in between. It runs forever, never repeats, and slowly morphs from one visual character to another without ever stopping.

No prompts. No inputs. Just autonomous beauty.

---

## What it looks like

The simulation runs 15 distinct behavioral regimes — spirals, maze, fingerprint, bacteria, uskate world, and more — each producing a completely different class of pattern. Overlaid on that are 37 color palettes, 5 rendering modes (each reads the simulation differently), and 6 post-processing effects. The world drifts between regimes every few minutes, and the visual style shifts with it.

**Rendering modes** — same chemistry, completely different image:
- `B` — raw chemical concentration (classic)
- `edges` — structure boundaries glow, spirals become rings
- `reaction` — only the active chemistry zone lights up
- `differential` — maximum contrast at the reaction front
- `A_inv` — the food layer, inverted

**Effects** — applied on top of any mode:
- bloom, vignette, chromatic aberration, film grain, scanlines

---

## Running it

```bash
# Clone and set up
git clone https://github.com/yourusername/somnivex
cd somnivex
pip install jax[cuda] flax pygame numpy

# Run
python main.py
```

**Controls while running:**
| Key | Action |
|-----|--------|
| `U` | Like this era — biases future picks toward this style |
| `D` | Dislike this era |
| `S` | Skip to next regime now |
| `Space` | Pause / unpause |
| `Q` | Quit |

---

## Record a demo video

```bash
python capture_demo.py                  # → somnivex_demo.mp4
python capture_demo.py my_output.mp4   # → custom filename
```

Cycles through all 15 regimes (~7 minutes raw). Speed it up in your editor or:
```bash
ffmpeg -i somnivex_demo.mp4 -vf setpts=0.5*PTS sped_up.mp4
```

---

## How it works

**Gray-Scott reaction-diffusion** — two chemicals A (food) and B (predator) obey:

```
dA/dt = Da·∇²A − A·B² + f·(1−A)
dB/dt = Db·∇²B + A·B² − (f+k)·B
```

`f` (feed rate) and `k` (kill rate) determine everything. The simulation drifts between 15 named parameter regimes from Pearson's 1993 catalog, morphing patterns instead of resetting. When a regime shift happens, the old structure is physically broken up so new chemistry can grow in.

The grid is 256×256, computed on GPU via JAX. About 900 reaction steps per second. Rendering scales up to 1920×1080.

---

## Hardware

Tested on:
- GPU: NVIDIA GTX 1650 (4GB VRAM) — CUDA 13.1
- CPU: Intel i7-2600
- OS: Ubuntu 24.04
- Python 3.12 + JAX + Flax + Pygame

Should run on any CUDA GPU. CPU-only mode works too (slower).

---

## Roadmap

This project is actively developed. Here's where it's going:

### Phase 2 — Preference Learning
After you rate ~200 pieces (U/D), a small neural network (MLP via Flax) trains on your ratings and biases the random parameter sampling toward what you actually like. Your screensaver slowly learns your taste over weeks of use.

### Phase 3 — NCA Blending
Neural Cellular Automata — a trainable version of reaction-diffusion — runs alongside the GS simulation and blends into the final render. NCA weights are steered by the preference model, so the system learns to grow patterns in styles you like. This is how the neuron/slime-mold/crystal morphologies become achievable — GS sets the foundation, NCA learns the specific shapes.

### Phase 4 — Daemon + GNOME Integration
Background daemon that activates automatically on GNOME idle and exits on any input. Each monitor can run independently with its own palette and render mode.

### Phase 5 — Gallery + Review App
Each visually interesting moment is saved as a PNG + JSON. A lightweight review interface lets you rate pieces after the fact. Rated pieces feed back into the preference model.

---

## Color System

117 hand-crafted palettes across 15 categories: space/celestial, ocean/water, geological/mineral, biological/cellular, industrial, atmospheric, fire variations, digital/terminal, fantasy, neon/electric, moody/desaturated, high contrast, warm/fire, cool/ice, and nature/organic.

Planned: procedural palette generation, 8-color gradients, palettes that respond to what the chemistry is doing in real time.

---

## Philosophy

Most generative art tools require you to prompt them. Somnivex runs without any input — it decides everything and shows you the result. You only provide signal (like/dislike) about what it already made. Over time the system builds a model of your taste and steers toward it automatically.

The goal is a screensaver that functions as a living piece of art that knows you.

---

## License

MIT. Use it, fork it, build on it.
