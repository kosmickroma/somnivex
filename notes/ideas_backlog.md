# Somnivex — Ideas Backlog
*A place to dump everything so nothing gets lost. Organized loosely by theme.*

---

## Physics Simulation as Living Art ⬅ BIG IDEA

### The Core Insight
The NCA architecture is a local update rule. Physics is mostly local update rules
(PDEs). What happens at a point depends on its neighborhood — that is literally
what the NCA computes every step. GS was just the first physics we taught it.
The architecture is a natural fit for any field equation.

### The Vision
Train an NCA on real physics data — not GS, but actual physical systems:
- Plasma dynamics / magnetohydrodynamics (quasar jets, solar flares)
- General relativistic field equations (black holes, wormholes, gravitational lensing)
- Fluid dynamics / Navier-Stokes (turbulence, vortex streets, convection)
- Quantum field fluctuations (vacuum energy, particle creation)
- Electromagnetic fields (lightning, aurora, plasma filaments)

Then run it free on its own output. The grid becomes a slice of that physics.
The patterns that emerge are not artistic interpretations — they ARE the physics,
computed by learned weights instead of explicit math. A quasar jet that never
repeats. A wormhole throat that breathes. Turbulence that organizes itself.

### The "Give It a Problem" Mode
Set initial conditions on the grid — a mass here, a field boundary there —
and watch it relax toward a solution. The NCA doesn't "know" it's solving a
problem. It just applies its learned local rules and the solution emerges from
millions of simultaneous local interactions, exactly as it does in nature.
This is what neural PDE solvers already do for fluid dynamics (FNO, DeepONet).
Nobody has made one beautiful.

### What Makes This New
Every physics simulator ever built shows you a graph, a color map, a diagram.
Clinical. Abstract. You understand it intellectually but you don't feel it.
Somnivex makes the visualization the computation. You're not watching a readout
of physics — you're watching physics itself emerge. There is no equivalent to
this that prioritizes beauty alongside correctness.

### Connection to Existing Somnivex Work
- The GS NCA already generalizes partially to GS-like systems in nature
  (animal coat patterns, coral growth, crystal formation) without retraining —
  because it learned the underlying math, not just the visual output
- Spatial f/k variation is already a form of "give it a problem" — different
  regions in different parameter territory, watching them interact at boundaries
- The blending of two models (model A + model B delta blend) is a path toward
  hybrid physics — something between GS and Lenia, or between GS and Navier-Stokes

### Next Steps Toward This
1. Pick one real physics system as a target (plasma/MHD is visually rich and
   has good simulation data available)
2. Generate training data from a classical simulator (or find existing datasets)
3. Train a second NCA on that data alongside or instead of GS
4. Blend it with the existing GS model using the delta-blend technique
5. The output is something between known physics — visually unprecedented

### Why This Matters Beyond Art
If it works, you have demonstrated that NCA architecture can learn and simulate
arbitrary local field equations from data. That has implications for:
- Fast approximate simulation of complex physical systems
- Learning physics from observational data where equations aren't fully known
- Visualizing high-dimensional physics in an intuitive, embodied way

*Noted: 2026-03-14*

### Ghost of Kepler — Real Observational Data as Training Input
*Source: Gemini, 2026-03-14 — cross-project idea with XenoScan*

Instead of generating synthetic MHD training data, feed the NCA real observational
data from protoplanetary disks or solar plasma observations. Since XenoScan already
works with Kepler lightcurve data, the pipeline for accessing that data exists.

The NCA learns the physics from actual cosmic observations rather than a simulator's
approximation of them. In free run it "hallucinates" the missing frames — continuations
of real cosmic history based on what it learned from the Kepler mission's neighborhood.

You aren't watching a simulation. You're watching the NCA reconstruct what it thinks
happened between two real observations of a star system. That framing alone is
extraordinary — a neural network dreaming the gaps in Kepler's data.

**Next step:** Identify which Kepler/observational datasets have the right spatial
resolution and physical variables to map to NCA channels. Solar Dynamics Observatory
(SDO) AIA imagery is a strong candidate — full-disk solar plasma imagery at multiple
wavelengths, high cadence, publicly available.

*Cross-reference: XenoScan project*

---

## Objectives / Steering

### Taste Learning (Personal RLHF)
Train a small preference model on like/dislike signals (U/D keys already exist in
the GS screensaver). Use it to steer f/k toward states that score higher over time.
The system learns what you personally find beautiful and drifts toward it.
- Already partially designed in original roadmap (Phase 4)
- MLP/Flax preference model, biased random sampling
- Could steer the f/k center, the spatial field amplitude, or the palette choice

### Novelty Drive
Measure how different the current state is from the last N frames — optical flow,
state entropy, or pixel-level difference. Steer the system away from states it has
visited recently. Rewards exploration, penalizes settling.
- Intrinsically Motivated Parameter Search (IMGEP) paper does this formally
  https://www.science.org/doi/10.1126/sciadv.adp0834

### Edge of Chaos Objective
Keep the system operating near criticality — the most interesting zone.
Too uniform → push toward more active territory.
Too chaotic → pull toward structure.
Measure: something like spatial frequency content or Lyapunov-adjacent entropy.
Could replace or supplement the saturation detector.

### Temporal Pacing
Don't change too fast, don't change too slow.
Penalize both frozen states and seizure-speed chaos.
A sense of rhythm — patterns should breathe, not flicker or freeze.

---

## Model Architecture / Training

### Dilated Perception Kernels for MHD Model
*Source: Gemini, 2026-03-14*

The current NCA uses a strict 3x3 neighborhood — each cell only sees 1 cell away.
For MHD physics where you want large-scale structures (galactic arms, jet filaments
that span the whole grid), that reach is too short. Local rules alone can't easily
produce structures that organize across hundreds of cells.

Dilated convolutions let some channels see further — 2, 4, or 8 cells away —
without increasing parameter count significantly. You add dilation as a parameter
to the perception kernel:

    # current: 3x3, dilation=1 — sees 1 cell away
    # dilated: 3x3, dilation=4 — sees 4 cells away, same parameter count

Implement as extra perception channels in the MHD model's kernel. Some channels
see close (local texture), some see far (global structure). The network learns
which scale matters for which aspect of the physics.

This is how you get sweeping spiral arms and jet filaments that span the full grid
emerging from local rules. Critical for MHD — implement from the start, not as
an afterthought.

### Train a Second Model, Blend the Deltas
Train model B on completely different physics — Lenia, Turing patterns, or a GS
distribution reweighted toward underrepresented regimes (diamonds, coral, bacteria).
At each step, blend both models' output deltas:

    delta = alpha * delta_A + (1 - alpha) * delta_B

Vary alpha over time (another drifting parameter). The output is neither model —
a hybrid physics that no equation describes. Model A stays untouched. This is the
right move before any retraining of the existing model.

### Rebalanced Training Distribution
The existing model has a worm/swirl bias because those regimes appear in 7/15
training regimes. A new training run with oversampling of underrepresented regimes
(diamonds, coral, bacteria, uskate) would produce a more balanced model.
Spatial f/k largely solves this at runtime, but a retrain would fix it at the
source.

### Genomic Signal Interpolation (requires retraining)
Train the NCA with a small per-cell conditioning signal encoding "what pattern type
should I be." At runtime, slowly rotate that signal. The NCA morphs between learned
pattern types continuously instead of locking into one attractor.
Paper: Multi-Texture Synthesis through Signal Responsive NCA (2024)
https://arxiv.org/abs/2407.05991

---

## Image / Semantic Input

### Feed It a Painting (Right Now, No Retraining)
Map a photograph or painting's pixel values to A/B channel concentrations and seed
the grid with that image instead of a GS warmup. The NCA won't recognize what the
image is — it only sees chemistry — but it will react to the structure of the values.
High-contrast edges become reaction fronts. The painting dissolves into chemistry
and reorganizes into the NCA's own patterns.
It doesn't draw the painting. It digests it and builds something new from the ruins.
Worth trying with a photograph — interesting as a concept on its own.

### Semantic Painting / CLIP Steering (requires retraining)
Encode a CLIP embedding of a text prompt ("a mountain at dusk") as a conditioning
channel. Train the NCA to respond to that signal — steer its hidden channels toward
patterns that match the semantic target. Full project, not trivial, but the
architecture is clear: extra input channels carrying the embedding, trained with
CLIP similarity loss.

---

## Runtime / Visual Improvements

### Multi-Scale Spatial Fields
Current field: one sine-wave landscape at one scale.
Add a second field at a different spatial frequency — coarse blobs of regime
territory with fine-grain noise inside each blob. Produces hierarchical structure:
large regions with their own internal variation. Could produce much richer
large-scale visual organization.

### Periodic Damage Circles (no retraining)
Drop random erasing circles into the live grid — wipe a region completely and let
it regrow from the edges. More organic than f/k pokes because regrowth happens
from the boundary inward. The NCA treats it as damage and self-repairs.
From: Growing NCA paper (Mordvintsev 2020).

### Model Blend Crossfade Over Time
Once a second model is trained — slowly crossfade the blend ratio alpha over
30+ minutes. The visual character of the system shifts from one physics to another
and back without any hard cut. Another dimension of the autonomous landscape.

---

## Long-Term / Big Picture

### Livestream (Phase 6)
24/7 autonomous generative art stream. System runs indefinitely, steers itself,
pipes output to OBS. No human needed. Probably needs the novelty drive and taste
model first so it doesn't get boring after an hour.

### Gallery + Preference Loop (Phase 4)
Save PNG+JSON of interesting frames. Yes/no review app. Train preference model.
Close the loop so the system is literally learning your taste over weeks/months.

### Color Intelligence (Phase 5)
Color as part of the NCA's state, not a post-processing lookup. The network learns
associations between chemical patterns and color during training. In free run it
chooses and evolves its own palette based on what it's doing.

---

## Audio

### Harmonic Audio Coupling — Physics You Can Hear
*Source: Gemini, 2026-03-14*

Map the average activation of specific physical channels to audio synthesis in
real time. The sound IS the physics — not a soundtrack layered on top, but the
actual field values driving the audio engine.

Mapping ideas for MHD model:
- Magnetic flux density → low-frequency oscillator (LFO) base frequency
- Vorticity / curl of velocity field → granular synthesis density
- Reconnection events (sharp local B field reversals) → percussive transients
- Density gradients → filter cutoff — thick plasma = dark/muffled, thin = bright
- Overall entropy → reverb tail length — chaotic state = long wash, ordered = dry

For the current GS model:
- B channel std → overall volume/presence
- Mean B → drone pitch center
- Perturbation events → triggered tonal hits

**Stack:** Python `sounddevice` or `pyaudio` for real-time output. Or route to a
synth via OSC. For the heavy/gritty aesthetic: granular synthesis with pitch-shifted
noise, sub-bass drone, distortion on reconnection transients.

When a vortex forms you hear it. When the wormhole breathes the room breathes with it.
When ghost traces appear the sound thins to almost nothing. Silence is a state,
not an absence.

**Priority:** Medium — implement after MHD model exists, so audio maps to richer
physics. Could prototype on current GS model first.

---

## Research / Paper

### Spatial Parameter Fields Paper
Already drafted in `notes/paper_draft.md`. Core contribution is novel.
Needs: systematic amplitude sweep, comparison with/without, possibly a second
example system beyond GS+NCA. Could submit to ALIFE, GECCO, or NeurIPS workshop.

### Document the Ghost Trace State
Observed 2026-03-14: near-extinction regions sustained locally by surrounding
active regions. This is a genuinely interesting emergent phenomenon — a state
that is globally unstable but locally stable because of boundary chemistry.
Worth characterizing formally. What f/k range produces it? How long does it persist?

---
*Last updated: 2026-03-14*
