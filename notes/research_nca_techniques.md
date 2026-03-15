# NCA Research — Techniques & Ideas for Somnivex
*Researched 2026-03-14*

Nobody has built a long-running autonomous NCA art system like Somnivex.
This is genuinely novel territory.

---

## Top 5 Most Actionable (priority order)

### 1. Spatial f/k Variation — NO RETRAINING
Instead of broadcasting one f and one k value to every cell in the grid,
write a spatial map — f and k vary smoothly across the grid using a
slow-drifting Perlin/noise field. Different regions behave differently
simultaneously. The whole grid becomes a landscape of behaviors instead
of one regime everywhere. The whole grid can NEVER collapse to one state
because different regions are in different parameter territory.

**This is the swirl-breaker. Build this next.**

---

### 2. Genomic Signal Interpolation — REQUIRES RETRAINING
Train the NCA with a small conditioning signal per cell (extra channels
encoding "what pattern type should I be"). At runtime, slowly rotate that
signal over time. The NCA morphs between learned pattern types continuously
instead of locking into one attractor.

Paper: Multi-Texture Synthesis through Signal Responsive NCA (2024)
https://arxiv.org/abs/2407.05991

---

### 3. Periodic Damage Circles — NO RETRAINING
Instead of f/k pokes, drop random erasing circles into the live grid.
Wipe a region completely and let it regrow from the edges. More natural
than pokes because regrowth happens organically from the boundary inward.
The NCA treats it as damage and self-repairs, producing fresh structure.

Source: Growing NCA paper (Mordvintsev 2020) — used during training,
works at runtime too.

---

### 4. Flow Lenia Localized Parameters — ARCHITECTURE CHANGE
Make the rule parameters (f, k) part of the cell state itself, so they
evolve spatially during the run. Adjacent cells mix their local rules when
they interact. No centralized parameter — each region develops its own
behavior and they blend at boundaries.

Paper: Flow-Lenia: Open-Ended Evolution through Mass Conservation (2022)
https://arxiv.org/abs/2212.07906
Site: https://sites.google.com/view/flowlenia/

---

### 5. Multi-Scale Hierarchical NCA — ARCHITECTURE CHANGE
Run a coarse 64x64 NCA to drive slow global intent.
Run a fine 256x256 NCA for local texture detail.
The coarse level provides slow-changing "what should be happening globally"
while the fine level handles moment-to-moment local dynamics.
Naturally produces the slow morphing + rich texture combination.

Paper: Evolving Hierarchical NCA (GECCO 2024)
https://dl.acm.org/doi/10.1145/3638529.3654150

---

## Other Interesting Techniques

### Stochastic Cell Updates
Apply updates to only ~50% of cells per step (random mask).
Prevents synchronization lock-in, keeps dynamics productively noisy,
prevents whole-grid freezing.
We already do this (FIRE_RATE=0.5 in model.py). Already implemented.

### Empowerment Signal
Give each cell a drive to maximize mutual information between its actions
and its future sensor states — intrinsic motivation to stay expressive.
Paper: Empowered Neural Cellular Automata (Grasso & Bongard, 2022)
https://arxiv.org/abs/2205.06771

### HyperNCA — Self-Modifying Weights
A HyperNetwork (itself an NCA) generates/modifies the weights of another
network. The "aesthetic layer" in Stage 2 becomes the NCA modifying its
own rules. Developmental metamorphosis of the network itself.
Paper: HyperNCA (Najarro et al., 2022)
https://arxiv.org/abs/2204.11674

### Steerable NCA
Cells carry an orientation state and can rotate it. Asymmetric updates
break global synchronization and produce more varied emergent structures.
Paper: Growing Steerable NCA (Randazzo, Mordvintsev et al., 2023)
https://arxiv.org/abs/2302.10197

### Intrinsically Motivated Parameter Search (IMGEP)
At each major state transition, run a lightweight diversity search over
parameter perturbations and pick the one that maximizes behavioral novelty
(measured by state entropy or optical flow). The system seeks interesting
territory automatically.
Paper: Discovering Sensorimotor Agency in CA (Inria/FlowerTeam, 2024)
https://www.science.org/doi/10.1126/sciadv.adp0834
Repo: https://github.com/flowersteam/sensorimotor-lenia-search

### Quality-Diversity for Open-Ended Evolution
Maintain a population of diverse parameter sets (MAP-Elites / AURORA).
When a piece gets visually stale (entropy drops), crossbreed two elites
from the population to produce something new. AURORA automatically learns
the diversity metric — you don't have to define what "different" means.
Paper: Toward Open-Ended Evolution in Lenia (ISAL 2024)
https://arxiv.org/abs/2406.04235

---

## Key Papers — Mordvintsev / Google Research lineage

| Year | Paper | Link |
|------|-------|------|
| 2020 | Growing Neural Cellular Automata | https://distill.pub/2020/growing-ca/ |
| 2021 | Self-Organising Textures | https://distill.pub/selforg/2021/textures/ |
| 2021 | μNCA: Ultra-Compact NCA (68 params) | https://arxiv.org/abs/2111.13545 |
| 2021 | Texture Generation with NCA | https://arxiv.org/abs/2105.07299 |
| 2022 | Particle Lenia | https://google-research.github.io/self-organising-systems/particle-lenia/ |
| 2023 | Growing Steerable NCA | https://arxiv.org/abs/2302.10197 |
| 2023 | Isotropic NCA | https://google-research.github.io/self-organising-systems/isonca/ |
| 2024 | Mesh NCA (NCA on arbitrary surfaces) | https://meshnca.github.io/ |

---

## Lenia — the other major lineage

Lenia (Bert Wang-Chak Chan, 2019) — continuous CA, 400+ species identified.
Avoids attractor lock-in via: continuous state + continuous time + smooth kernels.
https://arxiv.org/abs/1812.05433
https://github.com/Chakazul/Lenia

Key anti-attractor tricks from Lenia:
- Mass conservation (Flow Lenia) — patterns can't vanish, must reorganize
- Parameter localization — regions evolve independently
- Operating near criticality (edge of chaos) — most diverse behavior

---

## Useful Repos

- CAX — JAX library for CA, ICLR 2025 Oral
  https://github.com/maxencefaldor/cax

- Neural Automata Playground — WebGPU real-time NCA, live weight editing
  https://github.com/Stermere/Neural-Automata-Playground

- Lenia (official)
  https://github.com/Chakazul/Lenia

- Sensorimotor Lenia Search
  https://github.com/flowersteam/sensorimotor-lenia-search

- awesome-neural-cellular-automata
  https://github.com/dwoiwode/awesome-neural-cellular-automata

---

## What We've Already Built (for reference)

- FIRE_RATE=0.5 stochastic updates — already in model.py
- Pool-based training with damage — already in train.py
- Multi-step rollout loss — already in train.py
- Autonomous f/k drift — in run_free.py
- Perturbation sequences — in run_free.py
- Saturation detection + reseed — in run_free.py
- GS warmup seeding with random steps — in run_free.py
- Autonomous palette crossfading — in run_free.py
- Extreme burst (beyond training range) — in run_free.py
- Dual screen output — in run_free.py
