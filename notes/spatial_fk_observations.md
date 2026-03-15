# Spatial f/k — Visual Observations
*Documenting multi-regime behavior as evidence for the paper*

This is the key result: **different regions of the grid simultaneously expressing
different structural behaviors**, because each cell has its own f/k drawn from the
drifting spatial field. These screenshots are evidence that the technique works.

Screenshot naming convention: `spatial_[description]_[palette].png`

---

## What to look for when taking screenshots

A good spatial f/k screenshot shows at least two of these in the same frame:

- **Spiral / concentric rings** — classic GS uskate/spiral regime
- **Dense worm networks** — tight interlocking filaments
- **Large blobs / cells** — smooth rounded forms, low-detail
- **Geometric angles / sharp corners** — non-GS structure, boundary territory
- **Ghost traces** — near-invisible dark lines on black, near-extinction
- **Maze branching** — dendritic, tree-like structure
- **Transition zones** — the boundary between two regimes, often the most novel

If you see something in one region that you've never seen in a uniform-parameter
run — screenshot it immediately. That's the NCA doing something genuinely new.

---

## Video Evidence

### spatial_fk_morphological_transition_2026-03-14.webm
*`media/spatial_fk_morphological_transition_2026-03-14.webm`*

1:52 recording of the system running autonomously. Contains the clearest evidence
of real-time morphological transition captured so far. The sequence from 00:39–00:48
(9 seconds) shows a complete regime transformation:

| Timestamp | Character |
|-----------|-----------|
| 00:39 | Tight comma-hooks, small, sparse, individual closed forms |
| 00:44 | Hooks grow and connect — worm/maze network forming |
| 00:45 | Network fills in, open clearing appears center-right |
| 00:46 | Worms dissolving, forms enlarging, amoeba-like transition |
| 00:47 | Full blob territory, mixed scales, reorganizing |
| 00:48 | Large continental flowing forms — completely different character |

This is the spatial field drift in action in real time. The parameter landscape
is physically moving across the grid and the entire visual character transforms
continuously with it — tight hooks to worms to blobs to continents in under 10
seconds. No cuts, no resets. Pure autonomous evolution.

**This is the demo sequence for the paper.**

### nca_first_dream_2026-03-14.webm
*`media/nca_first_dream_2026-03-14.webm`*
Earlier session — pre-spatial f/k or early spatial f/k run. Historical reference,
shows the contrast between the old behavior and what the system does now.

---

## Captured So Far

### spatial_spiral_geometric_sepia.png
*2026-03-15*
Large concentric spiral pulling into itself (right side) + sharp geometric angles
and straight lines breaking away (left side). Warm sepia palette.

**Why this matters:** The angular/geometric structure on the left does not appear
in any single-parameter GS run. It is a boundary morphology — the NCA's own
interpretation of the transition zone between two parameter regions. This is
genuinely novel structure that no fixed-parameter system produces.

**Regimes visible:** spiral (right), boundary/geometric (left)

---

### spatial_worms_swirl_blue.png
*2026-03-14*
Dense intricate worm network with embedded spiral nucleation points. Multiple
scales of structure visible simultaneously. Deep blue palette.

**Why this matters:** The worms vary in density and curvature across the frame —
evidence of the spatial gradient in f/k. Not one uniform worm field but a
landscape of worm characters.

**Regimes visible:** worm network, spiral nuclei

---

### spatial_dense_worms_blue.png
*2026-03-14*
Extremely dense worm network, tight interlocking filaments at fine scale.
Denser and more complex than any single-parameter worm run.

**Regimes visible:** dense worm network

---

### spatial_large_blobs_blue.png
*2026-03-14*
Large organic blob forms — smooth, rounded, cell-like or cloud-like. Scattered
across mostly dark background. Completely different scale and character from worms.

**Regimes visible:** large blob / low-k territory

---

### spatial_ghost_traces_dark.png
*2026-03-14*
Near-extinction state. Barely-visible dark structures on near-black background.
Thin lines and partial shapes maintaining coherence against collapse.

**Why this matters:** This state is globally unstable — the saturation detector
would normally kill it. But with spatial f/k, it persists because surrounding
regions in more active territory maintain the boundary chemistry that keeps it alive.
This is a locally stable / globally unstable state that cannot exist in a
uniform-parameter system.

**Regimes visible:** near-extinction / ghost traces

---

### spatial_blob_outlines_purple.png
*2026-03-14*
Large blob outlines with glowing purple/violet edges on deep dark background.
Different character from the blue blob screenshot — the palette crossfade
produces a completely different emotional reading of the same physics.

**Regimes visible:** large blob outlines

---

## How to Take Good Screenshots

1. When you see something interesting, hit `PrtSc` immediately — these states
   are transient. The landscape drifts and they won't come back exactly.

2. Note roughly what you're seeing in the two regions — even a quick mental
   note of "spiral left / worms right" is enough to name it meaningfully.

3. Copy from `~/Pictures/Screenshots/` to `screenshots/` with a descriptive name:
   `spatial_[what_left]_[what_right]_[palette].png`

4. Add an entry here with the date and what makes it interesting.

---

## States We Haven't Captured Yet (looking for these)

- **Spiral + blob** in the same frame — the two most visually distinct regimes
- **Maze + ghost traces** — high structure adjacent to near-extinction
- **Three distinct regimes** in one frame — this would be a strong paper figure
- **Transition zone close-up** — the actual boundary between two regimes in detail
- **Full grid diversity** — multiple small regime patches across the whole frame
