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

## Preference Learning (Phase 2)

- MLP (Flax) trained on ratings_log.json after ~200 ratings
- Input: (category, palette, regime, render_mode, effect) as one-hot or embeddings
- Output: predicted like probability
- Replaces _weighted_choice with model-scored sampling
- Retrain incrementally every N new ratings

---

## NCA Blending (Phase 3)

- Train NCA weights to produce stable self-maintaining patterns (not just random init)
- Blend GS output and NCA output in final render (alpha-mix, not replace)
- NCA rule set drifts independently of GS regime

---

## Screensaver / System

- GNOME idle detection — auto-launch on idle, kill on any input (Phase 2 daemon)
- Separate left/right screen rendering (different palette+mode per monitor, optional mode)
- Config hot-reload — change config.py values without restarting
- Web dashboard — localhost:port shows current state, ratings, live stats

---

## Gallery (Phase 4)

- Auto-save PNG + JSON at visually interesting moments (high reaction rate, novel pattern)
- Lightweight review app (keyboard yes/no through saved pieces)
- Rated pieces feed back into preference model
