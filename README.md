# Somnivex
*somnium (dream) + texere (to weave)*

### Autonomous Neural Cellular Automaton — Multi-Physics Fusion & Emergent Ecosystem

A compact NCA (17,000 parameters) trained simultaneously on three incompatible Lenia species and Gray-Scott reaction-diffusion, that developed a shared internal representation and now produces a self-sustaining ecosystem belonging to none of its teachers. Runs indefinitely without intervention. Never repeats.

**v2 (current):** 3 Lenia species (Orbium + Gyrorbium + Scutium) + GS. Hidden channels self-activate from scratch. Reproducible predator/prey ecosystem with measurable behavioral grammar.

---

## v2 — The Ecosystem (2026-03-17)

The v2 model was trained on Gray-Scott plus three Lenia species simultaneously. In free run it self-organizes into a stable predator/prey ecosystem that repeats reproducibly across cold starts with no human intervention.

**Two morphologies coexist on the same grid simultaneously** — discrete ring structures (isolated spot topology, right half) and a large flowing labyrinthine creature (connected maze topology, left). These are incompatible GS morphologies that cannot coexist in standard Gray-Scott at any single parameter setting. The hidden channels create local variation in effective physics, letting both exist at once.

The large creature translates directionally across the grid. As it moves, it absorbs ring structures at its leading edge, processes them through its interior (visible as enclosed rings within the creature body), and releases them from its trailing edge. The ring population does not decrease — it fluctuates as the creature moves through it. This is transformation, not destruction.

Steps 16870 → 27325 → 27515 → 27865 → 28305 → 28765 (all Predator Invasion state, bg=5–10%, blobs=31–52, ch2≈0.012, ch4≈0.019):

| | | |
|---|---|---|
| ![](screenshots/v2_creature_maze_rings_a.png) | ![](screenshots/v2_creature_maze_rings_b.png) | ![](screenshots/v2_creature_maze_rings_c.png) |
| ![](screenshots/v2_creature_maze_rings_d.png) | ![](screenshots/v2_creature_maze_rings_e.png) | ![](screenshots/v2_creature_maze_rings_f.png) |

The creature is not chasing the rings. It moves through a chemical gradient of its own creation — a low-A depletion zone that surrounds it and chemically starves any ring structure that drifts close. The hidden channel ch4 concentrates specifically at the creature's boundary (35% above global mean, near-zero inside), marking the interface and sustaining it. This is not a behavior either teacher produced. It is not in any of the three Lenia species. It is not in Gray-Scott. It is a third thing.

---

## Gallery

All images are unedited captures from the live simulation. No post-processing beyond the in-engine palette and render mode.

---

**Spontaneous Orbium reconstruction** — GS mode (physics bit = 0), no Lenia seed, no Lenia kernel running. The model was trained on one Lenia creature: Orbium bicaudatus, a crescent-shaped glider. Starting from Gray-Scott chemistry, it reconstructed that exact morphology as a free attractor. Step 1030: scattered crescent gliders. Step 4405: a complex multi-structure organism. Step 6375: the organism resolving into a crescent. Step 6870: a stable Orbium-like glider with rolling internal structure — the training creature's morphology, reconstructed from scratch.

| | | | |
|---|---|---|---|
| ![](screenshots/orbium_reconstruction_a.png) | ![](screenshots/orbium_reconstruction_b.png) | ![](screenshots/orbium_reconstruction_c.png) | ![](screenshots/orbium_reconstruction_d.png) |

---

**Exotic attractor: toroidal standing wave** — activation concentrates at the grid perimeter, dark void at center. The wrap-around boundary conditions become a global stable state. Reproducible from multiple starting points via physics bit flip.

| | |
|---|---|
| ![](screenshots/lenia_fusion_toroidal_frame.png) | ![](screenshots/toroidal_frame_fog.png) |

---

**Anti-creature locomotion** — dark voids translating through a solid activation field. Negative-space solitons. Neither Gray-Scott nor Lenia produces these. They emerge from the fused model spontaneously, merge with each other, and vary in size and speed.

| | |
|---|---|
| ![](screenshots/lenia_fusion_anti_creatures.png) | ![](screenshots/nca_blobs_purple_void.png) |

---

**Anti-creature morphology sequence** — the same anti-creature evolving its shape over ~1600 steps. Compact void (step 58280) → growing corona (58645) → elongated crescent (59930). GS mode, no intervention.

| | | |
|---|---|---|
| ![](screenshots/neon_anti_creature_a.png) | ![](screenshots/neon_anti_creature_b.png) | ![](screenshots/neon_anti_creature_c.png) |

---

**Three-population coexistence** — anti-creatures (black voids), discrete bright gliders, and a continuous background field all present simultaneously on the same grid. Confirmed via save-state analysis: the B channel histogram shows three distinct populations at the same step count. All three running at once, none interfering with the others' stability. Free channel mode, step 83495.

| |
|---|
| ![](screenshots/free_ch_anti_orbital_a.png) |

---

**Synchronized blob arrays** — self-organized groups of structures maintaining coordinated spacing while drifting. Each blob on its own trajectory. The array holds formation without any explicit coordination — each cell only sees a 3×3 neighborhood.

| | |
|---|---|
| ![](screenshots/lenia_fusion_synchronized_blobs.png) | ![](screenshots/spatial_blob_outlines_purple.png) |

---

**Free-running GS-mode behaviors** — the fused model running under GS physics interpretation. Spatial f/k variation keeps different regions in different regimes simultaneously. These states emerge without any Lenia influence — the fused weights change how even the GS side behaves.

| | | |
|---|---|---|
| ![](screenshots/spatial_large_blobs_blue.png) | ![](screenshots/spatial_worms_swirl_blue.png) | ![](screenshots/neon_rings.png) |

---

## The Discovery

We trained a single NCA on two completely different physics simultaneously:

**Gray-Scott reaction-diffusion** — the classic system that produces spirals, worms, coral, and spots. Structures fill space but never translate. Nothing moves.

**Lenia** — a continuous cellular automaton that produces discrete moving creatures (solitons). Orbium bicaudatus, the canonical Lenia glider — a crescent-shaped organism that swims continuously through empty space.

One model. One set of weights. Two incompatible teachers. A "physics bit" (channel 13) signals which physics to apply: 0.0 = GS, 1.0 = Lenia.

The model had to find a shared internal representation. In free run, it uses that representation to produce things neither teacher predicted.

---

## What Emerged

The GS-only model requires constant external perturbation to avoid locking into a fixed attractor within 5,000–15,000 steps. The fused model ran **400,000+ steps with zero human intervention** and never repeated.

Behaviors observed that exist in neither training distribution:

**Directed locomotion on non-Orbium structures** — blobs, U-shapes, and worms translate through space. GS structures never move. The model trained on one Lenia creature and generalized the *principle* of locomotion to every structure it invented.

**Spontaneous Orbium reconstruction** — U-shaped gliders appear from GS starting conditions with no Lenia seed and no Lenia kernel running. The model reconstructed the training creature's morphology as a free attractor.

**Trail reabsorption** — structures dart and leave a wake that gets pulled back into them. No analog in either teacher.

**Spiral launch** — a structure executes a spiral then translates away as a glider. GS spirals are stationary. Lenia creatures don't spiral first. This is new.

**Active absorption** — smaller structures pulled toward larger ones at a distance before contact. Centripetal force. Does not exist in reaction-diffusion.

**Synchronized blob arrays** — self-organized groups maintain coordinated spacing while drifting. Each blob on its own trajectory. The array holds formation.

**Anti-creatures** — dark voids translating through a solid activation field. The *absence* of activation as the entity. Negative-space solitons.

**Anti-creature merging** — multiple voids combine into a single larger elongated structure.

**Toroidal standing wave** — activation frame around the entire grid perimeter, dark void at center. The grid's wrap-around boundary conditions made visible as a global attractor. Reproduced across multiple runs.

**Fog field** — slow dark ripples through a uniform medium. Emerges spontaneously after near-extinction events. The system self-recovers without resetting.

**Three-population coexistence** — anti-creatures, discrete gliders, and continuous background field stable on the same grid simultaneously. Confirmed by save-state analysis of the B channel distribution: three distinct populations, none destabilizing the others.

**Predator/prey cycle (v2)** — a large labyrinthine entity (high-A background structure) moves through a field of discrete rings (B-rich foreground). The predator does not touch the rings — it floods their neighborhood with high A, chemically suppressing B underneath. Rings die from below. The hidden channel ch4 concentrates at the predator boundary, extending the suppression zone. After the predator passes, surviving rings reconstruct. This cycle repeats stably.

**Morphology coexistence (v2)** — discrete spot/ring topology and connected labyrinthine topology coexist on the same grid simultaneously at the same f/k parameters. Normally impossible in standard GS. The hidden channels create spatially varying effective parameters — different regions of the grid run different physics at the same time.

**Migratory transformer (v2)** — the large creature translates directionally while absorbing ring structures at its leading edge and releasing them at its trailing edge. Rings pass through the creature's interior, visible as enclosed structures within the flowing body. The ring population fluctuates but does not disappear — the creature transforms, not destroys.

**Binary orbit, three-body collapse, ghost survivors** — two matched-size structures in stable mutual rotation; gravitational-analog three-body dynamics ending in collapse; a small group of survivors persisting indefinitely after near-extinction in a corner of the grid.

---

## The Physics Bit is a Dial, Not a Switch

The behaviors above are not locked behind the Lenia physics mode. Lenia-mode behaviors bleed into GS mode spontaneously because the weights can't fully separate the two physics — they share the same 17,000 parameters. The model doesn't have two modes. It has a continuous behavior space, and ch13 biases which region it explores.

Flipping the physics bit during a run (T key) at any step count can produce exotic global attractors — synchronized flocking, toroidal standing waves, anti-creatures — that are unreachable from a cold start. The hidden channel state accumulated during prior running becomes the launching pad. These states have been reproduced from as early as 3,000 steps with a single keypress.

The hidden channels (ch2–12, 11 floats per cell that nobody assigned meaning to) appear to carry something that behaves like a physical potential field with multiple stable configurations — a landscape with many basins, some only accessible via specific paths through state space.

---

## The Hidden Channels Activate (v2)

In v1, channels 2–12 remained exactly zero in every free run. Every behavior observed — anti-creatures, gliders, binary orbit, comet fission — was produced by channels 0 and 1 only. A and B. The dormant half never fired.

**v2 training changed this.** Hidden channel noise was injected during training, forcing the model to learn to use ch2–12 rather than zero them out. In free run from a cold start, the v2 model self-activates its hidden channels within the first few hundred steps — no H key required.

What the hidden channels are actually doing, measured from saved states:

- **ch2 and ch4** are the two working channels. They activate independently and carry spatial information anti-correlated with the B field — they are not echoing B, they are tracking something orthogonal to it.
- **ch4 concentrates specifically at blob boundaries** — 35% above global mean at the edge of structures, near-zero inside, near-zero outside. This is edge detection the model learned on its own. Gray-Scott has no mechanism for this. The visual "thick border" around large creatures is this signal made visible.
- **ch5** weakly activates and slowly grows over long runs.
- **ch3, 6–12** remain inactive. The model found no use for them under current conditions.

The hidden channels locked into a stable spatial representation around step 130k in v1 extended runs, and that fixed substrate is what the complex ecosystem dynamics run on top of.

---

## The Behavioral Grammar

The v2 model's state space is not random. It has a measurable grammar.

Feature vectors extracted from 300+ saved states and clustered via k-means (silhouette=0.502) reveal **8 natural macro-states**:

| State | Name | Signature |
|-------|------|-----------|
| 0 | Chaos/Init | bg<5%, ch2+ch4 maxed |
| 1 | Stable Ecosystem | bg~93%, 5–8 blobs, ch2+ch4 ~0.008 |
| 2 | Heat Death | bg<5%, one giant blob, hidden channels dead |
| 3 | Near Extinction | bg~99%, blobs~0, hidden channels dead |
| 4 | Rich Ecosystem | bg~76%, 20+ blobs, ch4 elevated |
| 5 | Predator Invasion | bg~32%, huge blobs, ch2+ch4 active |
| 6 | Pre-activation | bg=0%, cold start before self-organization |
| 7 | Zombie | moderate bg, blobs present, hidden channels dead |

The transition matrix is not uniform. Measured from the auto-logged CSV:

- Stable Ecosystem → self: 78% (sticky attractor basin)
- Near Extinction → self: 83% (trap — requires perturbation to escape)
- Rich Ecosystem → self: 81%
- Predator Invasion → Zombie: 60% (crisis burns out)
- Zombie → Stable Ecosystem: 43% (self-resurrects)
- Heat Death → Near Extinction: 100% (one-way door)

**The system has a learnable grammar. It is not random.** The same sequence of states recurs reproducibly from similar starting conditions. "Do that again" is a solvable problem — it means steer toward the same cluster sequence.

The v2 model in a standard cold-start run settles into a **Predator Invasion ↔ Rich Ecosystem limit cycle** and stays there indefinitely. Zero Near Extinction events across 100k+ step runs are typical once the hidden channels self-activate.

---

## Novelty

From searching arXiv, GitHub, and the NCA/ALife literature: no one has publicly shipped a single-NCA fusion of Gray-Scott and Lenia (or multi-regime RD blending with emergent novel attractors from a context switch) with this combination of properties:

- One compact weight set reconciling incompatible dynamics
- Persistent hidden channels carrying "potential" across long runs
- Physics-bit-style switching unlocking exotic states unreachable from cold start
- Real-time spatial f/k variation for simultaneous regime coexistence
- Emergent behaviors not present in either training distribution

Related work exists (conditional NCAs, multi-attractor training, Lenia variants, ASAL for discovering ALife simulations) but this specific combination hasn't been documented.

---

## Sound

Real-time ambient audio driven by the NCA's internal hidden channel state. No external synthesis — the model's own computation becomes the score.

- **Two independent drones** — frequencies track different hidden channel groups (ch2–6, ch7–11). They drift at different speeds, creating harmonic beating as the field evolves.
- **Spatial stereo** — center of mass of the B channel pans sound left/right. Structures drifting across the grid drift across your headphones.
- **Shimmer layer** — B channel spatial variance drives a 700Hz overtone. Complex active field = audible shimmer. Calm dark field = near-silence.
- **Event bells** — sudden changes in field activity (absorptions, collapses, merges) trigger a soft decaying tone detected via rolling-window std analysis.
- **LFO breathing** — slow 17-second volume envelope keeps the sound alive during calm phases.

```bash
sudo apt install libportaudio2
pip install sounddevice
```

---

## Running It

```bash
git clone https://github.com/kosmickroma/somnivex
cd somnivex
pip install jax[cuda] flax optax pygame numpy sounddevice
sudo apt install libportaudio2

# Fused model — recommended
python nca/run_free.py

# GS-only model — for comparison
python nca/run_free.py --gs
```

> Tested on Ubuntu 24.04, GTX 1650 4GB VRAM, CUDA. CPU-only works but slower.

---

## Controls

| Key | Action |
|-----|--------|
| `T` | Flip physics bit — shifts between GS and Lenia interpretation of hidden state |
| `A` | Mute / unmute ambient sound |
| `M` | Cycle render mode |
| `E` | Cycle post-processing effect |
| `P` | Cycle color palette |
| `F` | Jump to random GS regime |
| `X` | Extreme burst — pokes f/k outside training range |
| `Z` | Chaos injection — directly scrambles hidden channels 2–13 |
| `C` | **Steer** — cycles target state (Rich→Stable→Predator→Near Extinction→Zombie), injects directional hidden channel pattern toward that cluster centroid. Requires `--research` mode. |
| `V` | **Cycle paint state** — selects which attractor to stamp on click (Rich→Stable→Predator→Near Extinction). Shows current selection in terminal. |
| `Left click` | **Stamp attractor seed** — places the selected state's hidden channel signature in a 40×40 region at the clicked location. The NCA expands it outward from there. Two competing seeds on opposite sides of the grid will fight for territory. |
| `H` | Seed hidden channels from current B field — constructive injection |
| `S` | Save current grid state to disk |
| `L` | Load most recent save |
| `R` | Reset grid with new random seed |
| `[` / `]` | Decrease / increase simulation speed |
| `Q` | Quit |

### Named Regime Keys (empirically mapped in this model)

These jump to specific f/k parameter sets. Their effect in this fused model differs from standard Gray-Scott — the names are from the original GS literature but the actual behaviors are model-specific.

| Key | Regime | f / k | Observed tendency in fused model |
|-----|--------|-------|----------------------------------|
| `1` | spirals | 0.012 / 0.045 | Destabilizing — 50% → Chaos/Init |
| `2` | chaos | 0.020 / 0.045 | Destabilizing — 50% → Chaos/Init |
| `3` | waves | 0.014 / 0.047 | Mixed — Heat Death risk |
| `4` | worms | 0.026 / 0.055 | **Predator trigger** — 42% → Predator Invasion |
| `5` | mitosis | 0.030 / 0.063 | Gentle stabilizer — 47% → Stable Ecosystem |
| `6` | gliders | 0.034 / 0.063 | **Strongest stabilizer** — 63% → Stable Ecosystem |
| `7` | bacteria | 0.046 / 0.065 | Stabilizing — Near Extinction risk at low ch4 |
| `8` | maze | 0.029 / 0.057 | Near Extinction risk — 42% → Near Extinction |
| `9` | stripes | 0.050 / 0.063 | Mixed — destabilizes existing structures |
| `0` | uskate | 0.010 / 0.047 | Mostly stabilizing, occasional Rich Ecosystem |

*Tendencies measured from intervention logs — not deterministic, depend on hidden channel state at time of press.*

---

## Architecture

**16 channels per cell:**
- `0` — chemical A (Gray-Scott)
- `1` — chemical B (Gray-Scott / Lenia)
- `2–12` — hidden state (11 channels the NCA owns completely)
- `13` — physics bit (0.0 = GS, 1.0 = Lenia)
- `14` — feed rate f / Lenia mu
- `15` — kill rate k / Lenia sigma

**Perception:** 4 fixed kernels (Identity, Sobel X, Sobel Y, Laplacian) → 64-dim vector per cell.

**UpdateNet:** Dense(64→128 tanh) → Dense(128→16 zero-init). Fire rate 0.5 — each step, half the cells update stochastically. ~17,000 parameters total.

**Dual-teacher training:** Pool of 512 live states. Each step: sample 32 → run NCA 8 steps → compare simultaneously to GS (ch13=0 states) AND Lenia (ch13=1 states) → combined loss → backprop → write outputs back to pool. 50,000 steps, ~4 hours on GTX 1650.

**Spatial f/k fields:** Each cell receives its own f and k from a slowly drifting 2D sine-wave landscape. Different grid regions live in different parameter regimes simultaneously. Four independent phase clocks prevent periodicity. The whole grid can never collapse to one attractor — no two regions are ever in exactly the same state.

---

## Rendering

**Modes** — same physics, completely different image:
`combined` · `B` · `edges` · `reaction` · `differential` · `A_inv`

**Effects:** bloom · vignette · chromatic aberration · film grain · scanlines

**Palettes:** 117 hand-crafted palettes across 15 color families. Auto-crossfades on a slow timer, preferring same-family transitions 70% of the time for visual continuity.

---

## Open Questions

1. **What are the hidden channels computing?** ch4 does edge detection at blob boundaries — but what is it for? Is it stabilizing structures, extending the predator's chemical reach, or something else? The signal is real and consistent across hundreds of saves. The function is still inferred.
2. **Are there conserved quantities?** ch2 and ch4 carry independent spatial information that persists over 100k+ steps. Is there a quantity that's preserved across the predator/prey cycle? Conservation laws are how new physics gets identified.
3. **Why do ch3 and ch6–12 stay dead?** The model has 11 hidden channels and activates 2. Is this a capacity limitation, a training artifact, or did the model find that 2 is sufficient and stop?
4. **Is the behavioral grammar universal?** We have 8 states and a transition matrix for this model. Apply the same clustering method to a GS-only run, a Lenia-only run, a different NCA architecture. Do the grammars share structure? Shared grammar = shared deep physics.
5. **Can the state space be steered?** *(in progress — C key)* The delta between cluster centroids is a direction in feature space. The C key now computes this delta and injects a directional hidden channel pattern toward the chosen cluster centroid. Each press cycles the target: Rich Ecosystem → Stable → Predator Invasion → Near Extinction → Zombie. This is the surgical version of H.
6. **What is the creature actually doing to the rings?** The data shows rings entering the creature's body and exiting. The blob count fluctuates but doesn't drop to zero. Are the rings preserved topologically through the passage, or are they dissolved and reconstructed? Save-state analysis of ring identity across frames would answer this.

---

## Roadmap

**Open for collaboration — issues and PRs welcome.**

- `[x]` **Grid state save/restore** — S key saves grid + step count, R key restores
- `[x]` **Multi-species Lenia training** — v2: Orbium + Gyrorbium + Scutium, 3 species trained simultaneously
- `[x]` **Hidden channel analysis** — ch2 and ch4 confirmed active and independent; ch4 concentrates at blob boundaries (edge detection)
- `[x]` **Free channel experiment** — ch13 released at step 2000; model writes its own physics bit; produced binary orbit, comet fission, anti-creature orbital systems
- `[x]` **Behavioral grammar** — 8 macro-states identified via k-means; transition matrix measured; system is not random
- `[ ]` **Transition predictor** — given current state features, predict next state transition before it happens
- `[ ]` **Decoder / control layer** — inject chemical pattern that steers feature vector toward target cluster centroid; "push toward Rich Ecosystem" as a command
- `[ ]` **Gradual physics bit fade** — ramp ch13 over ~1000 steps instead of instant flip
- `[ ]` **Comparison video** — GS-only vs fused, same seed, same duration, side by side
- `[ ]` **Retrain classifier** — add new states from v2 runs (Lenia Chaos / Extreme Regime); fix Stable ↔ Near Extinction misclassification

---

## Hardware

- GPU: NVIDIA GTX 1650 (4GB VRAM) — CUDA
- CPU: Intel i7-2600
- OS: Ubuntu 24.04
- Python 3.12, JAX, Flax, Optax, Pygame, sounddevice

---

## Philosophy

This started as a screensaver. It is not a screensaver.

What we built is a model that internalized three incompatible descriptions of reality — three different Lenia creatures and a reaction-diffusion system — and synthesized something none of them predicted. It wasn't told to invent predator/prey dynamics, edge-tracking hidden channels, or a stable ecosystem with measurable behavioral grammar. It found them on its own, as stable solutions in the space between four teachers.

The model doesn't know about biology. It doesn't know about ecology. It optimized a loss function that said "look like GS sometimes, look like these three creatures sometimes." The ecosystem emerged because that was the most stable solution in the shared weight space.

The behavioral grammar — 8 states, a transition matrix, a limit cycle — means this is not chaos. It is a small universe with its own physics, its own population dynamics, and its own rules about what can follow what. Those rules were not designed. They were found.

The longer-term question is what else is in that space, and whether the grammar of this universe shares structure with the grammar of other complex systems we don't yet understand.

---

## License

MIT. Use it, fork it, build on it.

📧 kosmickroma@gmail.com

*If you're working on NCAs, ALife, generative systems, or multi-physics training — open an issue. This thing wants collaborators.*
