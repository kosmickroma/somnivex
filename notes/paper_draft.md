# Spatial Parameter Fields for Long-Running Neural Cellular Automata
## A technique for preventing attractor lock-in in deployed generative systems

*Draft — Somnivex project notes — started 2026-03-14*

---

## Abstract

Neural Cellular Automata (NCA) trained on reaction-diffusion physics produce rich
emergent patterns but tend to collapse into stable attractors — a small set of
recurring behaviors that dominate indefinitely once the system settles. We describe
a runtime technique — spatial parameter fields — that prevents this collapse
structurally rather than reactively. Instead of broadcasting a single parameter
vector to every cell, we write a slowly drifting 2D noise field so that different
regions of the grid occupy different parameter regimes simultaneously. The field
is generated at runtime from overlapping sine waves and advances on independent
phase clocks. No retraining is required. Applied to a 17,000-parameter NCA trained
on Gray-Scott reaction-diffusion physics, the technique produces indefinitely
sustained visual diversity: multiple morphological regimes co-existing in the same
frame, continuously reorganizing as the landscape drifts. We observe behaviors that
do not occur in any single fixed-parameter run, including simultaneous spiral,
blob, maze, and near-extinction regimes within one grid.

---

## 1. Introduction

Reaction-diffusion systems like Gray-Scott (GS) are parameterized by a feed rate f
and kill rate k. Small changes in these values produce dramatically different
patterns — spirals, worms, coral-like branching, maze structures, and chaotic
regimes. This sensitivity makes GS a rich substrate for generative art.

Neural Cellular Automata (Mordvintsev et al., 2020) learn to approximate the
GS update rule from data. A trained NCA can then run freely on its own output,
producing dynamics that approximate but diverge from the original physics — its
hidden state channels develop their own interpretation of the rules, producing
behaviors that GS alone never generates.

However, long-running NCA deployments face a fundamental problem: **attractor
lock-in**. The NCA's learned dynamics have stable attractors — states the system
gravitates toward and cannot easily escape. In practice this means that after a
few minutes of free run, the grid settles into a small set of recurring patterns
(typically worm/swirl morphologies, which dominate because they appear in 7 of
the 15 GS training regimes) and stays there indefinitely.

Prior mitigation strategies are reactive: perturbation sequences that poke the
control channels mid-reaction, forcing temporary reorganization; saturation
detection that triggers full reseeds when the grid goes uniform; extreme parameter
bursts that push the NCA outside its training distribution. These approaches
fight the attractor after the fact. They can delay lock-in but not prevent it
structurally — the attractor reasserts itself between interventions.

We propose a different approach: remove the single-attractor condition entirely.
If every cell is in a different parameter state, there is no single global attractor
for the system to find.

---

## 2. Background

### 2.1 Gray-Scott Reaction-Diffusion

Gray-Scott models two chemicals A (food) and B (predator):

```
dA/dt = Da·∇²A − A·B² + f·(1−A)
dB/dt = Db·∇²B + A·B² − (f+k)·B
```

f (feed rate) and k (kill rate) are global scalars that determine the entire
behavioral regime. The Pearson (1993) catalog identifies 15 named regimes across
the (f, k) parameter space, each with qualitatively distinct morphology.

### 2.2 Neural Cellular Automata

An NCA (Mordvintsev et al., 2020) replaces the GS update equations with a small
neural network applied identically to every cell. Each cell maintains N_CHANNELS
state values. At each step:

1. A fixed perception kernel (identity, Sobel X/Y, Laplacian) convolves the
   neighborhood — 4 filters × N_CHANNELS = 64 inputs per cell
2. A two-layer MLP maps these inputs to a delta vector
3. The cell's state is updated: state += delta (with stochastic firing rate)

For GS emulation, channels 0 and 1 carry A and B concentrations, channels 2–13
are hidden, and channels 14–15 carry f and k as control inputs readable by every
cell at every step.

### 2.3 Training: Pool-Based Multi-Step Rollout

Naive single-step training (predict one GS step from a GS state) fails in free
run due to distribution shift: the NCA never sees its own outputs during training,
so compounding errors collapse the system within seconds.

The fix is multi-step rollout: run the NCA 8 steps on its own output and compare
to GS running 8 steps from the same start, accumulating loss across all steps.
This forces the NCA to remain coherent across a trajectory, not just match one
step from a clean starting point.

Additionally, the f/k control channels are re-injected after each NCA step during
training to prevent the network from drifting the control values.

---

## 3. The Attractor Problem in Deployed NCAs

After training, the NCA is released from the training loss and runs on its own
output indefinitely. The f and k channels steer its behavior, but the system
has strong dynamical preferences — certain (f, k) regions have deep attractor
basins that the NCA reliably finds and stays in.

In our system, worm and swirl morphologies dominate. This is structurally
expected: they appear in 7 of 15 training regimes, so the NCA has learned
strong attractor dynamics for those patterns. Any nearby initial condition
tends to flow toward them.

Observation (Somnivex free run, March 2026): regardless of starting conditions,
within 3–8 minutes of autonomous operation the grid settles into a worm/swirl
dominated state. Perturbation sequences temporarily disrupt this but the system
returns to swirl attractors within seconds of the perturbation ending.

This is not a failure of training — it is a fundamental property of dynamical
systems with strong attractors. The NCA has learned its physics well; those
physics have preferred states.

---

## 4. Spatial Parameter Fields

### 4.1 Core Idea

The attractor problem assumes a single global (f, k) — every cell is in the same
parameter state, so the whole grid eventually finds the same attractor.

The solution: give each cell its own (f, k). If different regions are in different
parameter territory simultaneously, they cannot all find the same attractor at the
same time. The grid becomes a landscape of behaviors rather than one behavior
everywhere.

### 4.2 Field Generation

We generate smooth 2D f and k fields using overlapping sine waves:

```python
f_noise = (
    sin(x * 1.3 + phase_fx) * cos(y * 0.9 + phase_fy) * 0.6 +
    cos(x * 0.7 + phase_fy * 0.5) * sin(y * 1.1 + phase_fx * 0.7) * 0.4
)

f_field = clip(f_center + f_noise * amplitude_f, F_MIN, F_MAX)
```

Two overlapping waves at different spatial frequencies and angles produce a
smooth but non-repetitive landscape. The clipping ensures all values remain
within the trained parameter range (or deliberately beyond it for extreme regions).

Parameters used in Somnivex:
- f amplitude: ±0.015 (on a total f range of 0.01–0.08)
- k amplitude: ±0.010 (on a total k range of 0.04–0.075)
- This produces meaningful regime variation: different regions span multiple
  named GS regimes simultaneously

### 4.3 Drift

The field is not static. Four independent phase clocks (one per spatial axis per
field) advance at slightly different rates each step:

```
phase_fx += vel_fx * steps_per_frame   # vel_fx ~ 0.0008 * U(0.7, 1.3)
```

The result: the landscape slowly scrolls across the grid. A region that was in
spiral territory drifts toward blob territory, which drifts toward maze territory,
which drifts toward near-extinction territory. The transitions are smooth — no
hard cuts. At 5 NCA steps/frame × 30fps, one full phase cycle takes ~26 minutes,
so the landscape moves slowly enough to feel geological.

### 4.4 Runtime Injection

After every NCA step, the spatial fields are written back to the f/k channels:

```python
for _ in range(steps_per_frame):
    grid, key = step_fn(grid, params, key)
    grid = grid.at[:, :, CH_F].set(jf_field)
    grid = grid.at[:, :, CH_K].set(jk_field)
```

This pins the control channels to the spatial field values — the NCA reads its
local parameter state each step and responds, but cannot modify those values.
The field is the truth; the NCA's update is conditioned on it.

### 4.5 No Retraining Required

This technique requires no changes to the NCA architecture or training procedure.
The NCA was trained with scalar f/k channels — it has learned to read those
channels and respond appropriately. Writing a 2D array instead of a scalar is
invisible to the network. It simply finds different cells giving it different
control inputs and responds locally. The spatial coherence in the output emerges
from the NCA's learned local rules interacting across the smooth parameter landscape.

---

## 5. Observed Results

Visual behaviors observed in Somnivex after implementing spatial parameter fields
(March 14, 2026):

**Co-existing regimes in a single frame:**
- Dense intricate worm networks at fine scale in one region
- Large organic blob forms (cell/cloud morphology) in another region
- Thin ghost traces at near-extinction in a third region
- Structured maze-like branching at region boundaries

**Temporal behavior:**
- Regions maintain their character for minutes, then slowly transform
- Boundaries between regimes are soft and move continuously
- The whole grid never collapses to one state — there is always some region
  doing something different from every other region
- Visual interest is sustained indefinitely without human intervention

**Previously unobserved morphologies:**
- Ghost trace state: barely-visible dark structures on near-black background,
  right at the edge of collapse. This state is never stable globally (the
  saturation detector would trigger a reseed) but is stable locally when
  surrounded by more active regions that maintain the boundary chemistry.
- Blob-to-worm boundary structures: at the edge between blob-territory and
  worm-territory cells, the NCA produces novel transition morphologies that
  do not appear in either pure regime.

---

## 6. Relationship to Prior Work

**Lenia** (Chan, 2019) achieves regime diversity through continuous state and
smooth spatial kernels rather than parameter variation. Flow-Lenia (Plantec et al.,
2022) localizes the rule parameters into the cell state itself. Our approach is
complementary: we use a trained NCA (not Lenia) and apply parameter localization
as a runtime technique without architectural changes.

**Growing NCA** (Mordvintsev et al., 2020) uses stochastic updates (50% cell
firing rate) to prevent synchronization. We preserve this and layer spatial
parameters on top — they address different aspects of the attractor problem.

**Multi-Texture NCA** (Palm et al., 2022) trains an NCA with a per-cell
conditioning signal to produce different textures in different regions. Our
approach uses the existing f/k control channels — the NCA was never trained
for spatial variation, yet responds coherently because it has learned local
parameter-conditioned dynamics.

To our knowledge, spatial parameter fields applied at runtime to a pre-trained
reaction-diffusion NCA, with drifting phase clocks producing continuous regime
landscape evolution, has not been previously described.

---

## 7. Open Questions

- **Optimal amplitude:** The current ±0.015/±0.010 amplitudes were set by
  observation. A systematic sweep across amplitudes would characterize the
  diversity-coherence tradeoff.

- **Field topology:** Sine waves produce smooth, periodic landscapes. Perlin
  noise or other aperiodic fields might produce more varied large-scale structure.

- **Interaction with training distribution:** Regions at the field extremes may
  be outside the training parameter range. The NCA extrapolates — sometimes
  producing the interesting "beyond-training" behaviors previously only accessible
  via extreme burst perturbations.

- **Phase velocity:** Slow drift (current: ~26 min/cycle) feels geological.
  Faster drift might produce more dynamic reshaping. The right rate is aesthetic
  and probably context-dependent.

- **Multi-scale fields:** A coarse field (large blobs of regime territory) +
  fine field (small-scale noise within each blob) might produce richer hierarchical
  structure — different large regions with their own internal variation.

---

## 8. Implementation

Full implementation in `nca/run_free.py`. Key function:

```python
def make_fk_field(H, W, f_center, k_center, phase_fx, phase_fy, phase_kx, phase_ky):
    xs = np.linspace(0, 2*np.pi, W, endpoint=False)
    ys = np.linspace(0, 2*np.pi, H, endpoint=False)
    xx, yy = np.meshgrid(xs, ys)
    f_noise = (
        np.sin(xx*1.3 + phase_fx) * np.cos(yy*0.9 + phase_fy) * 0.6 +
        np.cos(xx*0.7 + phase_fy*0.5) * np.sin(yy*1.1 + phase_fx*0.7) * 0.4
    )
    k_noise = (
        np.sin(xx*0.8 + phase_kx + 1.0) * np.cos(yy*1.2 + phase_ky) * 0.6 +
        np.cos(xx*1.1 + phase_ky*0.4) * np.sin(yy*0.7 + phase_kx*0.8) * 0.4
    )
    f_field = np.clip(f_center + f_noise * FK_SPATIAL_AMP_F, F_MIN, F_MAX)
    k_field = np.clip(k_center + k_noise * FK_SPATIAL_AMP_K, K_MIN, K_MAX)
    return f_field.astype(np.float32), k_field.astype(np.float32)
```

~50 lines of runtime geometry. No changes to model architecture, training
procedure, or checkpoint. Adds negligible compute overhead (numpy ops on
256×256 arrays, once per frame).

---

## References

- Mordvintsev et al. (2020). Growing Neural Cellular Automata. *Distill.*
  https://distill.pub/2020/growing-ca/
- Mordvintsev et al. (2021). Self-Organising Textures. *Distill.*
  https://distill.pub/selforg/2021/textures/
- Chan, B.W.C. (2019). Lenia: Biology of Artificial Life. *Complex Systems.*
  https://arxiv.org/abs/1812.05433
- Plantec et al. (2022). Flow-Lenia: Open-Ended Evolution through Mass
  Conservation. https://arxiv.org/abs/2212.07906
- Palm et al. (2022). Multi-Texture Synthesis through Signal Responsive NCA.
  https://arxiv.org/abs/2407.05991
- Pearson, J.E. (1993). Complex Patterns in a Simple System. *Science* 261.

---

*This document is a living draft. It will be updated as the system evolves.*
*Started: 2026-03-14*
