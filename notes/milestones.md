# Somnivex — Milestones

A dated record of major breakthroughs. What changed, what was observed, why it matters.

---

## 2026-03-14 — Spatial f/k Parameter Fields

**What changed:**
Instead of broadcasting one f and k value to every cell, `run_free.py` now generates
a smooth 2D sine-wave field — each cell gets its own f/k drawn from a slowly drifting
landscape. Four independent phase clocks advance at slightly different speeds so the
field never becomes periodic. No retraining required.

**What was observed:**
For the first time the grid shows genuinely different behaviors in different regions
simultaneously:
- Dense intricate worms at two scales in the same frame
- Large organic blobs (cell/cloud-like forms) occupying most of the frame
- Ghost traces — near-invisible dark structures right at the edge of extinction,
  barely maintaining coherence against a black background
- Large blob outlines with glowing purple/violet edges, a color/structure character
  the NCA had never previously settled into

The system no longer locks into a single attractor. It rotates through radically
different visual characters and sustains genuinely diverse states indefinitely.

**Why it matters:**
Attractor lock-in was the core unsolved problem. Every previous mitigation
(perturbations, extreme bursts, reseeds) fought the attractor after the fact.
Spatial f/k variation removes the single-attractor condition structurally —
the grid is always a landscape of behaviors, not one behavior everywhere.

**Implementation:** `nca/run_free.py` — `make_fk_field()`, phase drift per frame,
spatial injection after every NCA step. ~50 lines of runtime geometry.

---

## 2026-03-14 — NCA Free Run Working (Phase 2 Complete)

**What changed:**
After diagnosing distribution shift (1-step training fails because the NCA never
sees its own outputs during training), `train.py` was rewritten with multi-step
rollout: run NCA 8 steps on its own output, compare to GS 8 steps, accumulate loss.
Trained for 50,000 steps (~2.5 hours on GTX 1650). Final pred_loss ~0.0001–0.0005.

Added to `run_free.py`: GS warmup seeding (50–800 random steps), random starting
regime, autonomous f/k drift, perturbation sequences, extreme bursts, timed reseeds,
saturation detection, autonomous palette crossfading, A+B channel rendering,
dual screen output, adjustable steps-per-frame.

**What was observed:**
NCA runs stably indefinitely. Produces spirals, diamonds, maze-like structures,
worm fields, blob patterns. Responds to f/k changes mid-run without restarting.
Produces morphologies that pure GS never generates — diamonds that evolve into ovals,
swirls that emerge from the collapse of structure.

**Why it matters:**
The trained NCA has internalized all 15 GS regimes simultaneously. It runs its own
version of those physics — not a simulation, a memory of physics — and finds paths
through the space that the original equations never would.

**Checkpoint:** `nca/checkpoints/params_050000.pkl`

---

## 2026-03-07 (approx) — Init: Gray-Scott Screensaver (Phase 1 Complete)

GPU-accelerated Gray-Scott reaction-diffusion on JAX. 15 named regimes from Pearson
1993, smooth morphing between them. 117 hand-crafted color palettes across 15
categories. 5 rendering modes, 6 post-processing effects. Dual screen output.
Screensaver mode. Preference rating system (U/D keys). ~900 GS steps/second on
GTX 1650.
