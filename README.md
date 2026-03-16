# Somnivex
*somnium (dream) + texere (to weave)*

### Autonomous Neural Cellular Automaton — Multi-Physics Fusion & Emergent Dynamics

A compact NCA (17,000 parameters) trained simultaneously on two incompatible physics systems — Gray-Scott reaction-diffusion and Lenia — that developed a shared internal representation and now produces attractor states belonging to neither teacher. Runs indefinitely without intervention. Never repeats.

---

| | | |
|---|---|---|
| ![](screenshots/lenia_fusion_anti_creatures.png) | ![](screenshots/lenia_fusion_synchronized_blobs.png) | ![](screenshots/lenia_fusion_toroidal_frame.png) |
| ![](screenshots/neon_anti_creature_a.png) | ![](screenshots/neon_anti_creature_b.png) | ![](screenshots/neon_anti_creature_c.png) |

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

---

## The Physics Bit is a Dial, Not a Switch

The behaviors above are not locked behind the Lenia physics mode. Lenia-mode behaviors bleed into GS mode spontaneously because the weights can't fully separate the two physics — they share the same 17,000 parameters. The model doesn't have two modes. It has a continuous behavior space, and ch13 biases which region it explores.

Flipping the physics bit during a run (T key) at any step count can produce exotic global attractors — synchronized flocking, toroidal standing waves, anti-creatures — that are unreachable from a cold start. The hidden channel state accumulated during prior running becomes the launching pad. These states have been reproduced from as early as 3,000 steps with a single keypress.

The hidden channels (ch2–12, 11 floats per cell that nobody assigned meaning to) appear to carry something that behaves like a physical potential field with multiple stable configurations — a landscape with many basins, some only accessible via specific paths through state space.

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
| `[` / `]` | Decrease / increase simulation speed |
| `R` | Reset grid with new random seed |
| `Q` | Quit |

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

1. What are the hidden channels actually computing? Is there new math in there or a lossy approximation of known physics?
2. Are there conserved quantities in the hidden channel dynamics? Conservation laws are how new physics gets identified.
3. The trail reabsorption behavior has no analog in either teacher. What rule produces it?
4. What happens if we stop re-injecting the control channels entirely and let the model write to all 16 channels freely?
5. Would training on 5–6 Lenia species generalize even deeper biological principles?

---

## Roadmap

**Open for collaboration — issues and PRs welcome.**

- `[ ]` **Grid state save/restore** — S key saves grid + step count to disk for reproducible experiments
- `[ ]` **Gradual physics bit fade** — ramp ch13 over ~1000 steps instead of instant flip
- `[ ]` **Hidden channel analysis** — PCA/t-SNE of ch2–12 during long runs; map behavior landscape; find emergent conservation laws
- `[ ]` **Free channel experiment** — stop re-injecting ch13/ch14/ch15; let model write all 16 channels freely
- `[ ]` **Multi-species Lenia training** — 5 creatures instead of 1; deeper generalization of biological principles
- `[ ]` **Comparison video** — GS-only vs fused, same seed, same duration, side by side proof

---

## Hardware

- GPU: NVIDIA GTX 1650 (4GB VRAM) — CUDA
- CPU: Intel i7-2600
- OS: Ubuntu 24.04
- Python 3.12, JAX, Flax, Optax, Pygame, sounddevice

---

## Philosophy

This started as a screensaver. It is not a screensaver.

What we built is a model that internalized two incompatible descriptions of reality and synthesized something neither of them predicted. It wasn't told to invent locomotion, anti-creatures, or toroidal standing waves. It found them on its own, as stable solutions in the space between two teachers.

The longer-term question is what else is in that space.

---

## License

MIT. Use it, fork it, build on it.

📧 kosmickroma@gmail.com

*If you're working on NCAs, ALife, generative systems, or multi-physics training — open an issue. This thing wants collaborators.*
