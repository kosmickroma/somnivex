# What Next — Connecting Physarum to train_lenia.py

You've now built a Physarum field simulator that:
- Runs in Python/numpy
- Produces (H, W) float32 trail maps at each step
- Saves them as training data in the format the NCA already expects

Here's how it plugs into the existing training pipeline.

---

## The three teachers so far

| Teacher | Channel it uses | Physics bit (ch13) | What NCA learns |
|---------|-----------------|--------------------|-----------------|
| Gray-Scott | ch0=A, ch1=B | 0.0 | Reaction-diffusion blobs |
| Lenia | ch0=activation | 1.0 | Moving creatures |
| Physarum (new) | ch0=trail | 0.5 | Trail reinforcement |

ch13=0.5 puts Physarum exactly between GS (0.0) and Lenia (1.0).
The model sees ch13 and knows which physics to expect.

---

## What train_lenia.py needs to add Physarum

Two things:

**1. A Physarum pool** (like gs_pool and lenia_pool)
- Array of shape (POOL_SIZE, H, W, N_CHANNELS)
- Each state has: ch0=trail concentration, ch13=0.5, everything else=0
- Refreshed from physarum_training_data.npz

**2. A Physarum loss function** (like gs_loss_fn and lenia_loss_fn)
- Run the NCA forward on a Physarum state
- Compare ch0 of the NCA output to the actual next frame from the data
- Loss = mean squared error on ch0 only (same as Lenia loss)

The training ratio would be something like:
- 44% GS (14/32 batch slots)
- 44% Lenia (14/32 batch slots)
- 12% Physarum (4/32 batch slots)

---

## The experiment we're watching for

After adding Physarum as a teacher, run the model in research mode and
watch the hidden channel readings in the HUD.

**Hypothesis A (boring):** ch4_border_ratio increases slightly. No new channels activate.
The model absorbed Physarum into its existing GS vocabulary. Trail boundaries = blob
boundaries, same concept. Grammar doesn't expand.

**Hypothesis B (interesting):** ch5 (the slowly-growing one) jumps in activity.
The model found that ch5 is useful for "has flow been here recently" —
directed persistence, which GS doesn't need but Physarum does.
Grammar expands by one concept.

**Hypothesis C (very interesting):** A completely dormant channel (ch3, 6-12) activates
for the first time. The model discovered a concept that neither GS nor Lenia needed
but Physarum requires — something about gradient direction or trail age.

We'll know which one happened by looking at the feature logs after a run.

---

## How to know when to build this

The GS-only training run is still going.
When it finishes, run it in research mode and see what clusters it finds.

If GS-only WITHOUT Lenia still develops hidden channel activity and
finds 4+ distinct behavioral states → grammar is in GS physics alone.

If GS-only stays flat (2 states, no hidden channels) →
grammar requires the multi-physics tension, Physarum adds a new dimension.

Either way the Physarum teacher is worth adding.
The question is just whether it extends an existing grammar or starts a new one.

---

## Files to write when we get there

```
nca/physarum.py              — Physarum pool init + loss function
nca/train_lenia.py           — add --physarum flag, 12% ratio, third pool
```

That's it. Maybe 80 lines total.
The hard part is already done — you understand how the simulator works.
