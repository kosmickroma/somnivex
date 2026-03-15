# NCA Checkpoints

Each checkpoint is a saved set of Flax model parameters for the UpdateNet.
Loaded by `nca/run_free.py` at startup.

---

## params_050000.pkl

**Date:** 2026-03-14
**Training run:** 50,000 steps total
- Steps 0–20,000: single-step rollout (1 NCA step vs 1 GS step)
- Steps 20,000–50,000: multi-step rollout (8 NCA steps on own output vs 8 GS steps)
  Resumed from params_020000.pkl with fresh Adam optimizer state.

**Architecture:**
- 16 channels per cell (CH_A=0, CH_B=1, hidden=2–13, CH_F=14, CH_K=15)
- Perception: 4 fixed filters (identity, Sobel X, Sobel Y, Laplacian) × 16ch = 64 inputs
- UpdateNet: Linear(64→128, ReLU) → Linear(128→16)
- FIRE_RATE = 0.5 (stochastic updates — 50% of cells update each step)
- ~17,000 parameters

**Training data:**
- Pool of 512 live GS states (random regimes, random ages 0–1000 steps)
- 15 named GS regimes from Pearson 1993 catalog
- f range: 0.010–0.062, k range: 0.045–0.065 (training distribution)

**Final loss:** pred_loss ~0.0001–0.0005
**Training time:** ~154 minutes on NVIDIA GTX 1650

**Behavior in free run:**
- Stable indefinitely — no collapse observed in multi-hour runs
- Responds to f/k control channel changes mid-run without restarting
- Dominant attractor: worm/swirl morphologies (present in 7/15 training regimes)
- With spatial f/k fields: sustains co-existing spiral, blob, maze, and
  near-extinction regimes simultaneously
- Produces novel morphologies not present in any single GS regime

**Known characteristics:**
- Worm/swirl bias — these are the deepest attractors due to training distribution
- Spatial f/k variation (implemented in run_free.py) breaks this bias at runtime
- Beyond-training-range values (f<0.01, k>0.065) produce interesting extrapolation
  behavior — the NCA interpolates between what it knows

---

## params_020000.pkl

**Date:** 2026-03-13 (approx)
**Training run:** First 20,000 steps, single-step rollout only.
**Status:** Superseded by params_050000.pkl. Kept for reference.
**Behavior:** Collapses to solid color after ~30 seconds in free run.
  This is the distribution shift failure mode that motivated multi-step rollout.
  Kept as a historical reference — do not use for free run.

---

## Adding a new checkpoint

When retraining, document here:
- Date and reason for retraining
- What changed (architecture, training data, loss function, hyperparameters)
- Final loss and training time
- Observed behavioral differences from previous checkpoint
