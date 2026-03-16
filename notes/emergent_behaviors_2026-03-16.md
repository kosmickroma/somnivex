# Emergent Behavior Observations — 2026-03-16
## Dual-Teacher NCA: GS + Lenia Fusion (lenia_050000.pkl)

Observed during first extended free run of the dual-teacher checkpoint.
Grid: 256×256, QUIET_MODE=True (zero human intervention), physics_bit=0.
Run duration at time of writing: 170,000+ steps with no pokes, reseeds, or
regime changes. Original GS-only model required constant perturbation to
avoid attractor lock-in. This model has not needed a single intervention.

---

## Confirmed Novel Behaviors

### 1. Directed locomotion on non-Orbium structures
Blobs, U-shapes, and worms all exhibit directed translation through space.
GS structures never translate — reaction-diffusion only spreads outward.
The model trained on one Lenia creature (Orbium) and generalized the
*principle* of locomotion to arbitrary morphologies it invented itself.

### 2. Rapid inter-structure attraction ("snap")
Nearby structures suddenly accelerate toward each other across empty space.
Not diffusion (which is gradual and symmetric). Directed force at a distance.
Consistent with Lenia's neighborhood kernel giving structures long-range
awareness of each other.

### 3. Active absorption vs passive diffusion
When a small structure meets a larger one it gets pulled in — sucked toward
the larger structure before contact. Original GS model: boundaries dissolved
passively through diffusion. This is centripetal, not centrifugal.

### 4. Trail reabsorption
Some blobs dart quickly and leave a short trail. The trail then gets pulled
back into the blob after it stops moving. Structure actively reabsorbs its
own wake. No analog in GS or Lenia. Model's own invention.

### 5. U-shaped gliders (spontaneous Orbium morphology)
U-shaped crescent structures appear spontaneously from GS initial conditions
with no Orbium seed and no Lenia kernel running. They pulse/swim slowly
across the screen. Orbium bicaudatus (the training creature) is U/crescent
shaped. The model reconstructed the training creature's morphology as a
free attractor starting from GS chemistry.

### 6. Daughter shedding
U-shapes and larger blobs shed smaller pieces that become independent blobs
and rings. Canonical Lenia behavior — creatures fragment under interaction
or at parameter boundaries. Now occurring on GS-derived structures.

### 7. Spiral launch sequence
A structure executed a spiral motion and then translated away from the spiral
as a directed glider. GS spirals are stationary (pinned). Lenia creatures
don't spiral. This sequential spiral→locomotion behavior exists in neither
teacher. First observed ~step 50,000.

### 8. Self-organized blob arrays with differentiated fates
Blobs of similar size spontaneously arranged into rows. Then resolved
in three distinct ways:
- **Quiet dissolution**: internal state depleted, structure faded
- **Bright flare then death**: amplification burst before collapse
  (consistent with Lenia creatures near parameter boundary flaring out)
- **Phase transition to U-shape**: blob reorganized into more stable
  morphology and swam away

Same visual appearance, different internal hidden channel state → different
fate. Suggests hidden channels carry meaningful per-structure state invisible
to the renderer.

### 9. Multi-type coexistence ecosystem
Worms, spirals, blobs, U-shapes, rings all present simultaneously on screen,
morphing between types at boundaries, interacting through attraction/absorption.
Spatial f/k field creates different parameter regions; Lenia influence means
those regions now produce different *entity types* not just different chemistry.

---

## Key Metrics

| Metric | GS-only (params_050000) | Fused (lenia_050000) |
|--------|------------------------|----------------------|
| Steps before attractor lock-in | ~5,000–15,000 | 170,000+ (still running) |
| Manual interventions needed | Constant pokes/reseeds | Zero |
| Structure types simultaneously | 1–2 | 5–6 |
| Directed locomotion | None | All structure types |
| Inter-structure attraction | None | Yes, at distance |
| Trail reabsorption | None | Yes |
| Spontaneous glider morphology | None | Yes (U-shapes) |

---

## Interpretation

The dual-teacher training changed the model's attractor landscape from a
collection of stable fixed points (GS) into a high-dimensional strange
attractor. The model exhibits recurrent themes (patterns) without exact
repetition (no loops) — the signature of a genuinely complex dynamical
system rather than a complicated one.

The Lenia training contributed more than creature-specific behaviors. It
contributed the *principle* of:
- Structures as discrete entities in space (vs chemistry filling a grid)
- Locomotion as an internal state cycle
- Long-range inter-structure interaction
- Non-stationary dynamics (inherent motion vs externally imposed perturbation)

The model generalized these principles beyond the single training creature
and applies them to GS-derived morphologies it invented itself.

---

## What We Don't Know Yet

- What are the hidden channels (ch2-12) actually computing?
  Recording and analyzing their values during each behavior type could
  reveal whether they encode known physical quantities or something new.
- Are there conserved quantities in the hidden channel dynamics?
  Conservation laws are how new physics gets identified.
- The trail-reabsorption behavior has no analog in either teacher.
  What equation produces it?
- Would training on multiple Lenia species (not just Orbium) produce
  even deeper generalization of locomotion principles?

---

## Next Steps for Publication / Documentation

1. Screen recordings of each named behavior (already captured some)
2. Hidden channel analysis — dump ch2-12 values during each behavior,
   look for structure, try PCA clustering by behavior type
3. Ablation: run lenia_050000 without CH_PHYSICS injection — does behavior
   change? Proves physics bit is actively used as routing signal
4. Comparison video: GS-only vs fused side by side, same starting seed,
   same duration, no intervention
5. Write up as: "Emergent third-physics dynamics from dual-teacher NCA
   trained on Gray-Scott reaction-diffusion and Lenia"
