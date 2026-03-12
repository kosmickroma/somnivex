# Somnivex — Copilot Briefing Doc

> Paste this file into your Copilot chat at the start of each session, or reference it with @workspace.
> Updated by Claude between sessions to reflect progress and set up the next chunk.

---

## What Is Somnivex?

An autonomous generative art screensaver that learns the user's taste over time.
- No user prompts — ever. All parameters are randomly generated.
- Uses **Neural Cellular Automata (NCA)** on JAX + GPU to produce abstract, slowly-evolving visuals.
- Runs as a background process. Two pygame windows: one showing live generation, one showing the last completed piece.
- Completed pieces are saved to a gallery. The user rates them yes/no. A preference model learns their taste over time and biases future generation.

---

## Hardware & Stack

- GPU: GTX 1650, 4GB VRAM (~3.5GB free). Keep VRAM under 500MB.
- CPU: i7-2600 (old — keep CPU work light)
- RAM: 12GB
- Python 3.x, JAX (GPU), Flax, pygame, numpy
- Venv: `/home/kk/ai-env`
- Project root: `/home/kk/projects/ml/somnivex`
- Gallery storage: `/mnt/ml_storage/somnivex/gallery/`

---

## Project Structure

```
somnivex/
├── main.py              # entry point
├── config.py            # all tunable constants
├── nca/
│   ├── model.py         # JAX NCA step function + init
│   ├── rulesets.py      # different rule set flavors
│   └── params.py        # random parameter generation
├── display/
│   └── windows.py       # two pygame windows
└── gallery/
    └── storage.py       # save PNG + JSON metadata
```

---

## Three-Stage Generation

Each piece takes ~2 hours. Stages are time-based cutoffs:

| Stage | Duration | What happens |
|-------|----------|--------------|
| 1 | ~30 min | NCA rule set, seed pattern, color palette randomly chosen and locked in |
| 2 | ~45 min | Post-processing aesthetic style applied to Stage 1 state |
| 3 | ~45 min | Slow convergence — piece settles into final living form |

Random choices are frozen after Stage 1. Stage transitions are purely time-based (no stability detection).

---

## NCA Approach

- Compute grid: 256x256 or 512x512, upscaled to 1920x1080 for display
- Abstract visuals only — not literal imagery
- Rule set categories: fluid/organic, angular/geometric, crystalline, chaotic
- Aesthetics (chrome, watercolor, anime, etc.) applied as post-processing in Stage 2
- Each NCA cell has N channels (e.g. 16). The update rule is a learned or parametric function over local neighborhoods.

---

## Two-Window Display

- **Left window (GENERATING):** shows the NCA grid updating live across all 3 stages
- **Right window (LIVING):** shows the last completed piece, slowly animating
- Two separate pygame windows (not one spanning window — user needs to work while it runs)
- Both windows are 1920x1080, one per monitor

---

## Build Phases

- [x] Phase 0: Repo scaffold (done — empty stubs exist)
- [ ] **Phase 1: Visual pipeline** ← WE ARE HERE
  - NCA model (JAX step function)
  - Random parameter generation
  - Three-stage loop with time-based cutoffs
  - Two pygame windows rendering live
  - Runs in foreground (no daemon yet)
- [ ] Phase 2: Daemon + GNOME idle detection
- [ ] Phase 3: Gallery save + yes/no review app
- [ ] Phase 4: Preference model (MLP/Flax) + biased sampling

---

## Current Session Goal

**Build `nca/model.py`** — the core JAX NCA step function.

### What Copilot should produce:
A single file `nca/model.py` containing:

1. `init_grid(height, width, n_channels, seed)` — initializes the NCA grid as a JAX array. Start with a small random seed pattern in the center, zeros elsewhere.

2. `make_perception_kernel()` — returns the Sobel-x, Sobel-y, and identity filters used to perceive each cell's neighborhood.

3. `perceive(grid, kernel)` — applies the perception kernel to the grid via convolution. Each cell sees itself + its neighbors.

4. `nca_step(grid, weights, bias, update_rate)` — one NCA update step:
   - Perceive the grid
   - Apply a small MLP (two dense layers with relu) to get candidate updates
   - Stochastically apply updates (only `update_rate` fraction of cells update per step, ~0.5)
   - Return new grid, clipped to [0, 1]

5. A `@jax.jit` decorator on `nca_step` for GPU performance.

### How Copilot should teach:
- **Do NOT write the code for me.** Show me one function at a time in your chat window.
- For each function: explain what it does and why, show the code in chat, then wait for me to type it myself into the file.
- Only move to the next function after I confirm I've typed it and it makes sense.
- If I ask a question, explain it before moving on — don't just dump the next chunk.
- I am learning — treat this like a guided lesson, not code generation.

---

## Conventions

- JAX arrays everywhere (no numpy in hot paths)
- `jax.random` for all randomness (pass keys explicitly)
- Keep functions pure — no side effects, no global state
- `config.py` is the single source of truth for constants (grid size, channels, etc.)
- Don't optimize prematurely — clarity first, then jit

---

## Checkpoint Log

| Date | What was completed |
|------|--------------------|
| 2026-03-11 | COPILOT.md created, Phase 1 starting next session |

> Claude updates this table when checking in.
