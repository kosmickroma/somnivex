# Somnivex — TODO

## Palettes / Color System

### Procedural palette generation
- Generate palettes algorithmically at runtime instead of only hand-picked ones
- Random hue rotation with locked saturation/brightness relationships
- Complementary, triadic, and analogous color theory schemes
- Perlin-noise-driven palette that slowly shifts hue over time (independent of regime drift)
- Palette seeded from era seed number (every era gets its own unique colors, fully reproducible)

### Richer palette structure
- 8-color palettes instead of 4 (smoother gradations, more depth)
- Non-linear interpolation curves (ease-in, ease-out, sinusoidal) instead of linear lerp
- Per-channel gamma adjustment so darks stay dark and brights actually bloom

### Reactive palettes
- Saturation boost when reaction rate is high (A*B² spikes)
- Hue shift when pattern movement is fast vs slow (tie to freeze-detector delta)
- Palette "pressure" — the current regime nudges palette hue toward a canonical color

---

## Render Modes

- **Curl field** — visualize the rotational component of the B gradient (spirals become vortex rings)
- **Phase portrait** — plot A vs B as a 2D scatter mapped to color (shows limit cycles clearly)
- **Laplacian** — second derivative of B (highlights curvature, not just edges)
- **Time-averaged** — blend last N frames together (ghosting effect, shows motion trails)

---

## Visual Effects

- **Trails / motion blur** — blend current frame with weighted previous frame (ghosting)
- **Distortion** — UV warp based on gradient field (ripple/heat-haze look)
- **Palette inversion** — flip the heat map midway through (inverts which chemical is "bright")
- **Dithering** — ordered or Floyd-Steinberg dither before display (crisp pixel-art feel)
- **Lens flare** — additive streaks from the brightest points

---

## Simulation

- **More GS regimes** — mine the full xmorphia / Munafo catalog for more named (f, k) pairs
- **Anisotropic diffusion** — vary Da/Db spatially or directionally (patterns gain directionality)
- **Noise field overlay** — add slow Perlin noise to f/k spatially (breaks global symmetry)
- **Wrap vs. clamp boundaries** — currently wraps; clamped edges create interesting border effects

---

## Phase 2 — NCA Physics Training (current focus)

Goal: teach the NCA Gray-Scott mechanics via one-step prediction, then let it run free.

### Architecture (decided)
- 16 channels: 0=A (food), 1=B (predator), 2-13=hidden, 14=f control, 15=k control
- 4 fixed perception kernels: Identity, Sobel X, Sobel Y, Laplacian → 64-dim input to net
- UpdateNet: Dense(128, tanh) → Dense(16, zero-init) — residual delta, no alive mask
- Fire rate 0.5 (stochastic async updates, per Growing NCA paper)

### Training (pool-based — key insight from Mordvintsev 2020)
- Pool of 512 live GS states (A, B, f, k snapshots from random regimes)
- Each step: sample 32 from pool → run NCA 1 step → compare to actual GS 1 step → backprop → write back
- Loss: MSE on channels 0 and 1 (A and B) only
- Also inject small noise during training and penalize deviation (persistence loss)
- Gradient normalization: per-variable L2 norm (prevents late-training spikes)
- Train until: loss plateau + visual shadow check + 10-min free run test passes

### Files to write (kktodo/)
- [x] 01_nca_model.md — full rewrite of nca/model.py
- [ ] 02_nca_train.md — new file nca/train.py (training loop)
- [ ] 03_main_integration.md — plug trained NCA into main.py

---

## Phase 3 — Dream Chamber (bolt-on after Phase 2 works)

Background process that runs inference-time novelty search:
- Fork 64 mini 64×64 grids with random control channel nudges
- Run 200 steps each, score with: entropy of B channel + temporal delta (3 lines of JAX)
- Blend top-4 winning control vectors back into main grid smoothly
- Save winning control vectors as .npy "genetic memory" files — reloadable later
- Connects to preference model: scoring biased toward what user has liked

---

## Phase 4 — Preference Learning

- Collect ~200 U/D ratings (currently at 0 — fresh start as of 2026-03-12)
- MLP (Flax) trained on ratings_log.json
- Input: (regime, palette, render_mode, effect) as one-hot
- Output: predicted like probability
- Biases NCA control channel nudges toward liked styles (not just GS regime picks)
- Retrain incrementally every N new ratings

---

## Phase 5 — Ecosystem Layer (downstream, after Phase 3)

Run 2 specialist NCAs on the same grid, sharing A/B channels:
- Specialist A: trained on GS spirals/waves
- Specialist B: trained on GS spots/mitosis
- Each updates only its own hidden channels (2-13), competing for shared A/B channels
- Add resource budget control channel — weak specialist patterns get recycled
- Start with 2 species, not 4

---

## Screensaver / System

- GNOME idle detection — auto-launch on idle, kill on any input
- Separate left/right screen rendering (different palette+mode per monitor)
- Config hot-reload — change config.py values without restarting

---

## Gallery (Phase 6)

- Auto-save PNG + JSON at visually interesting moments
- Lightweight review app (keyboard yes/no through saved pieces)
- Rated pieces feed back into preference model
