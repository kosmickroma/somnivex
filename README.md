# Somnivex
*somnium (dream) + texere (to weave)*

### Autonomous Neural Cellular Automaton — Multi-Physics Fusion & Emergent Ecosystem

A compact NCA (17,000 parameters) trained simultaneously on three incompatible Lenia species and Gray-Scott reaction-diffusion, that developed a shared internal representation and now produces a self-sustaining ecosystem belonging to none of its teachers. Runs indefinitely without intervention. Never repeats.

**v2 (current):** 3 Lenia species (Orbium + Gyrorbium + Scutium) + GS. Hidden channels self-activate from scratch. Reproducible predator/prey ecosystem with measurable behavioral grammar.

---

## LLM-NCA Hybrid: Persistent Programs in a Living Substrate (2026-03-19)

**The short version:** An LLM painted "KK ✕" in living organisms on a neural cellular automaton. When the LLM was disconnected, the composition kept running on its own.

**The mechanism:** The LLM (Gemini 2.5 Flash) issues spatial commands that write pheromone values (ch5=0.8) into specific grid cells. The NCA re-injects that value every step — permanently. Organisms follow ch5 gradients and lock onto those cells. The LLM isn't puppeting the organisms in real time. It's writing standing instructions into the physics of the substrate. The NCA then executes those instructions indefinitely, whether the LLM is watching or not.

**What happened in the first clean session:**
1. Asked Gemini to draw an X across the grid
2. Gemini drew two diagonal trails corner-to-corner, dropped one blob to seed organisms, then wiped all external chaos
3. Organisms locked onto both diagonals
4. Unprompted, Gemini added a protective ring around the center junction — the point where both trails cross and organisms accumulate most densely. No instruction existed to do this. It reasoned that the junction was the structural heart of the composition and defended it.
5. Then asked Gemini to add "KK" — its own initials were already on the grid, so it placed two K letterforms in the remaining canvas space, correctly sizing and positioning them relative to the existing composition
6. LLM terminal was closed. The composition — X, ring, KK — continued running unchanged

| | |
|---|---|
| ![](screenshots/llm_artist_kk_gold.png) | ![](screenshots/llm_artist_kk_neon.png) |

*Left: composition in terminal_amber palette. Right: same composition in neon_city. The KK letterforms and central X are visible as living organism formations held by pheromone infrastructure.*

![](screenshots/llm_artist_smiley.png)

*Smiley face drawn by Gemini in candy_chrome palette. Eyes, smile, and outer ring held by pheromone trails — living organisms swarm the boundaries while the structure persists indefinitely.*

**Why this is different from generative art:** A generative system renders an output. This system encodes intent into a substrate that continues executing it. The LLM wrote a program. The NCA is running it. The composition is not an image — it's a standing instruction that living matter is following.

**The junction ring:** Gemini added this without being asked. It understood from the physics description that where trails cross, organisms accumulate and the structure is most vulnerable to chaos. It made a structural decision — protect the most important point — and executed it with a `shape ring` command. This is the first observed case of an LLM making an unsolicited architectural decision about a living system based on understanding of its physics.

**[▶ Watch on YouTube](https://youtu.be/UgWdtZdLKdo)**

Bridge code: [`nca/llm_bridge.py`](nca/llm_bridge.py) (`--artist --provider gemini`)
Run: `python nca/run_free.py --physarum --artist --research` + `python nca/llm_bridge.py --artist --provider gemini`

---

## Collaborative Blueprint Construction (2026-03-20)

Two LLMs — Gemini 2.5 Flash and Claude Haiku — given an identical blueprint and told to build it. Neither knew the other existed. They communicated only through the grid.

**The setup:** A PCB trace network blueprint: 5 horizontal rails, 5 vertical trunks, connector stubs, and drawn ring vias at every intersection. Both LLMs received the same blueprint file and the same 16×16 ch5 heatmap of the current grid state every turn. Each wrote commands to its own file; `run_free.py` executed both simultaneously. No turn-taking, no arbitration — pure parallel construction.

**What happened (from the logs):**

Gemini went through the blueprint top to bottom in strict order. Every element once, no revisits, declared complete and disconnected cleanly.

Claude made its own ordering decisions. It drew the horizontal rails first, then jumped immediately to the vertical trunks — skipping ahead in the blueprint sequence. Then worked back through edge stubs, connector stubs, pad rings. Then drew all 5 vertical trunks a second time — elements it had already completed itself, now fully built on the grid. Then worked through the via rings until the session ended at ring 23 of 25. Never declared done.

Importantly: Claude was not checking Gemini's work. It was making independent decisions about what to build in its own sequence, and re-drew its own completed elements anyway. Both LLMs independently attempted the full blueprint. The redundancy was Claude vs. itself, not Claude vs. Gemini.

Neither LLM was told what the other existed. Both read the same grid state each turn and decided what to draw based on what they saw. The substrate was the only shared interface.

**Observed behavior (first run, one data point):**
- Gemini: strict linear ordering, one pass, zero redundancy, declared done cleanly
- Claude: non-linear ordering, re-drew its own completed elements, did not declare done

The efficiency problem — agents re-drawing already-complete elements — is the natural next thing to fix. The heatmap resolution is currently 16×16; finer resolution would let agents read completed elements more precisely and skip them. First run establishes the mechanism. Subsequent runs tune the behavior.

**The Tron shot:**

![PCB circuit grid in terminal_cyan palette](screenshots/pcb_blueprint_cyan.png)

*5×5 pheromone grid in terminal_cyan palette. Structure built simultaneously by Gemini and Claude without direct communication. The NCA holds the pattern indefinitely after both LLMs disconnected.*

![PCB grid self-repairing after wipe](screenshots/pcb_blueprint_cyan_healing.png)

*Mid-repair: two large sections wiped of organisms. The pheromone infrastructure is untouched — organisms flow back in from the edges following the trails. The grid knows what it's supposed to be.*

**A note on scale and memory:**

Nothing about this is limited to two LLMs. You could run 50 agents simultaneously on a larger grid — cut them all off, restart them cold with no memory of what happened, point them at the same blueprint, and they would resume exactly where the collective left off. They don't need to remember. The grid remembers. The pheromone infrastructure is the shared memory, and it survives agent death. This is what makes the substrate fundamentally different from a chat session or a shared document — it's a persistent physical state that any number of stateless agents can read and write, and the work accumulates regardless of individual agent continuity.

**How to run:**
```bash
# Terminal 1 — NCA
python nca/run_free.py --blueprint --research

# Terminal 2 — Gemini (workhorse)
python nca/llm_bridge.py --blueprint --provider gemini

# Terminal 3 — Claude (auditor)
python nca/llm_bridge.py --blueprint --provider anthropic
```

Each LLM sees: the full blueprint + the current grid heatmap. Each writes to its own command file. They finish independently and exit cleanly. The NCA keeps running.

Blueprint: [`nca/blueprint.txt`](nca/blueprint.txt) — edit this to change what gets built.

---

## LLM Artist — Painting With Living Organisms (2026-03-19)

An LLM directing organisms the way a choreographer directs dancers — not drawing pixels, but placing attractors that living matter fills in, holds, and evolves around. Gemini 2.5 Flash reads a screenshot of the live canvas every ~5 seconds, decides what to draw next, and issues spatial commands. The human types direction in a third terminal and Gemini responds.

**The brush toolkit:**
- `trail x1 y1 x2 y2 [strength] [width]` — straight pheromone line; organisms follow and hold permanently
- `curve x1 y1 bx by x2 y2` — smooth curved line bending toward (bx,by); for organic forms, rivers, branches
- `shape ring/circle/arc/spiral cx cy r` — geometry stamps with chemistry; instantly populated, no blob needed
- `blob x y` — seed organisms near an isolated trail only; shapes never need this
- `wipe cx cy r` / `wipe_rect` — precision kill zones; trail cells are always protected
- `pulse x1 y1 x2 y2` — spike ch5 to snap drifting organisms back to a line
- `palette <name>` — 10 palettes: neon_city, aurora, candy_chrome, sakura, terminal_amber...
- `mirror on/off` — bilateral symmetry across vertical axis
- `clear_trails` — erase all trails, start a new painting (organisms keep running)

**Vision feedback:** Gemini receives a screenshot of the actual canvas with every turn. It can see what it drew, assess spatial relationships, and self-correct — "the peak is off-center, adding a trail to the right slope."

**Key distinction — trail vs shape:**
- `trail`/`curve` write only the pheromone path. Organisms must flow in from nearby populations.
- `shape` commands write pheromone AND chemistry directly — they light up with organisms immediately.

**This is not generative art.** The LLM isn't rendering pixels. It's writing standing instructions into the physics of a living substrate. The NCA executes those instructions indefinitely whether the LLM is watching or not.

Bridge code: [`nca/llm_bridge.py`](nca/llm_bridge.py) (`--artist --provider gemini`)

---

## GS+Physarum: The Hidden Channel Grammar Is Teacher-Specific (2026-03-19)

We trained a third model — GS + Physarum polycephalum (slime mold) as simultaneous physics teachers, using ch13=0.5 as the Physarum bit.

The result confirmed and complicated the Universal Grammar Hypothesis:

| Model | Teachers | ch2 active | ch4 active | ch5 active |
|-------|----------|-----------|-----------|-----------|
| GS-only | GS | 0.6% | 0.1% | 0.1% |
| Fused v2 | GS + Lenia | 94% | 98% | 28% |
| Physarum | GS + Physarum | 0.5% | 9% | **100%** |

**Grammar requires multi-physics tension** — confirmed. GS alone: flat hidden channels, no grammar. Add a second physics teacher: hidden channels self-activate and specialize.

**But the grammar is not universal.** GS+Lenia activates ch2 (chaos signal) and ch4 (boundary tracker). GS+Physarum activates ch5, always positive, always on. Different teachers produce different internal vocabularies. The hidden channel dictionary is a function of the specific tension, not just the presence of tension.

**ch5 is a pheromone trail channel.** Spatially, it:
- Concentrates at blob boundaries (edge signal)
- **Persists behind moving blobs** — the model secretes ch5 into the space it just left, creating visible trail persistence across hundreds of steps
- Correlates with dark material density and mobility

The physarum model is behaviorally distinct from the fused model: 3x more blobs (28 vs 10), all smaller; 2x faster movement (mobility 71 vs 31); almost no suppression zones — pure swarming, no territory. The NCA internalized the difference between Lenia's structured creature dynamics and Physarum's trail-following network formation.

---

## Stigmergic Construction (2026-03-19)

The ch5 trail channel can be painted directly onto the live grid. The physarum NCA treats injected ch5 as a pre-existing pheromone trail and moves toward it.

**What organisms do with trails:**
- **Lock onto lines** — a glider found a drawn line and stayed for thousands of steps while the grid evolved around it
- **Form living walls** — organisms line up along trails and hold formation; the trail becomes the organism
- **Build junctions** — where two trails cross, organisms accumulate at the intersection and anchor there
- **Create stable interiors** — a closed loop trail causes organisms to hold the boundary; the enclosed region self-organizes into a calm interior
- **Reconstruct after erasure** — erase ch5 and the structure slowly dissolves; redraw it and they rebuild

This is stigmergic construction. The same mechanism Physarum uses to solve mazes and reconstruct the Tokyo rail network — but in a trained NCA, with human-drawn scaffolding.

The anti-line constraint: on a fully dead grid (A=1, B=0), ch5 injection creates void boundaries rather than attractors. No living chemistry = no gradient to follow. Trail injection is **parasitic on existing activity** — it needs live organisms to work. This is not a limitation; it defines what trails mean: they are signals in a living system, not marks on a canvas.

---

## Blind Stigmergy Battle (2026-03-18)

**[▶ Watch on YouTube](https://youtu.be/9yP2SXGnSkA)**

Two frontier LLMs — Claude Sonnet (Destroyer) and Gemini Flash (Keeper) — given opposing goals, access to the same NCA, and zero knowledge of each other. The only thing between them is the grid. Neither knows the other exists. Neither knows the whole picture.

- **Claude's goal:** drive the grid to permanent extinction
- **Gemini's goal:** keep the ecosystem alive indefinitely
- **The NCA's role:** it doesn't know any of this is happening

This is blind stigmergy — the same mechanism ants use to coordinate without ever meeting. Each LLM reads a neutral feature CSV (numbers only), issues one command per turn, and the NCA physics transforms the signal between them. Multiple extinctions and recoveries observed in a single unscripted run.

Full design spec: [`kktodo/plan/BLIND_STIGMERGY.md`](kktodo/plan/BLIND_STIGMERGY.md)
Bridge code: [`nca/llm_bridge.py`](nca/llm_bridge.py)

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

## v2 — Wound Healing & Attractor Seeds (2026-03-17)

A Near Extinction seed (left click after pressing V) writes A=0.99, B=0.00 into a 96×96 region — a chemically dead rectangle. The NCA treats this as a wound and immediately begins reacting at the boundary. What follows is not what anyone expected.

The rectangle border stabilizes into a sharp glowing wave front — ch4 concentrates at the edge exactly as it does at natural blob boundaries. The dead zone holds its shape. Living chemistry surrounds it. Then the boundary begins to evolve: straight edges develop curvature, corners grow appendages, the rectangle sprouts tendrils and siphons energy from nearby structures. Over thousands of steps it morphs from a geometric stamp into an organic creature, while the NCA tries to "heal" the wound according to its learned physics.

**Data from 79,000-step interactive run:** Near Extinction seeds from Predator Invasion state → Rich Ecosystem within 200 steps, 9 of 9 times. The dead zone interrupts the dominant blob chemistry, forcing a reorganization at the boundary. Stacking multiple seeds → Chaos → Stable reset sequence. This is the most reliable state control discovered.

Steps 28005 → 28980 → 29315 → 29595 → 29855 → 30935 (Near Extinction seeds placed, boundary evolving):

| | | |
|---|---|---|
| ![](screenshots/v2_seed_rectangle_birth.png) | ![](screenshots/v2_seed_rectangle_glowing.png) | ![](screenshots/v2_seed_rectangle_pointing.png) |
| ![](screenshots/v2_seed_creature_appendages.png) | ![](screenshots/v2_seed_labyrinth_creature.png) | |

**Left:** Seed rectangle born mid-Predator Invasion — clean geometric structure coexisting with crescent gliders. **Center:** Rectangle develops double border (ch4 edge signal), ring solitons form below it, organic gliders orbit around it. **Right:** Rectangle opens into bracket/U shape as boundary waves propagate. **Bottom left:** Boundary fully organic — creature with body and appendages, recognizable animal morphology emerging from a square stamp. **Bottom center:** Full labyrinthine creature at peak complexity — the rectangle is gone, replaced by a connected maze organism filling half the grid.

This is the lizard paper in real time. The model learned to grow toward a target. When you stamp a dead zone, you give it a wound. It heals the wound using the same learned rules — and the healing produces structures the model has never explicitly been trained to make.

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

**Void Soliton locomotion** — dark voids translating through a solid activation field. Negative-space solitons. Neither Gray-Scott nor Lenia produces these. They emerge from the fused model spontaneously, merge with each other, and vary in size and speed.

| | |
|---|---|
| ![](screenshots/lenia_fusion_anti_creatures.png) | ![](screenshots/nca_blobs_purple_void.png) |

---

**Void Soliton morphology sequence** — the same Void Soliton evolving its shape over ~1600 steps. Compact void (step 58280) → growing corona (58645) → elongated crescent (59930). GS mode, no intervention.

| | | |
|---|---|---|
| ![](screenshots/neon_anti_creature_a.png) | ![](screenshots/neon_anti_creature_b.png) | ![](screenshots/neon_anti_creature_c.png) |

---

**Three-population coexistence** — Void Solitons (black voids), discrete bright gliders, and a continuous background field all present simultaneously on the same grid. Confirmed via save-state analysis: the B channel histogram shows three distinct populations at the same step count. All three running at once, none interfering with the others' stability. Free channel mode, step 83495.

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

**Void Solitons** — dark voids translating through a solid activation field. The *absence* of activation as the entity. Stable propagating depressions in the reaction-diffusion field — the exact analog of dark solitons in nonlinear optics. Neither teacher produced them. They emerge from the fused weights spontaneously, merge with each other, and vary in size and speed.

**Void Soliton merging** — multiple voids combine into a single larger elongated structure.

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

In v1, flipping the physics bit during a run (T key) could produce exotic transient attractors — synchronized flocking, toroidal standing waves, anti-creatures — that were unreachable from a cold start. The hidden channel state accumulated during prior running acted as a launching pad.

**v2 behavior has changed.** Flipping to Lenia mode (T key) now produces a single deterministic outcome every time: one unbreakable full-screen blob that fills the entire grid. This happens on every cold start, every run, regardless of prior GS state. It is not a transient — the system locks there and does not leave. Near Extinction seeds, regime jumps, and key interventions that reliably steer GS-mode states do not break it.

Occasionally, small dark voids (Void Solitons) attempt to form inside the monolith — GS depletion chemistry trying to carve out space against Lenia reinforcement — but they cannot sustain. They collapse back within hundreds of steps.

The interpretation: v2 training made ch4 so effective at maintaining interfaces and suppression zones that the Lenia attractor is now globally self-reinforcing. The entire grid becomes one giant ch4-stabilized boundary. The GS-mode ecosystem (predator/prey cycles, ring populations, labyrinthine creatures) runs on top of a carefully balanced energy distribution that the Lenia ground state collapses into a single uniform energy sink.

This is not a bug. It is the Lenia floor — the deepest attractor basin the v2 weights contain. What happens there, and whether it can be escaped, is an open experiment.

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

**Primary controls (most useful in practice):**

| Key | Regime | f / k | Role | Observed tendency |
|-----|--------|-------|------|-------------------|
| `1` | mitosis | 0.030 / 0.063 | **Stabilizer** | Stable 36%, Rich 27% — reliable calm-down key |
| `2` | gliders | 0.034 / 0.063 | **Stabilizer** | Stable 89% — most reliable, glider-friendly |
| `3` | maze | 0.029 / 0.057 | **Predator trigger** | Context-dependent — pushes toward Predator Invasion |
| `0` | uskate | 0.010 / 0.047 | **Chaos escape** | Breaks locked states, brief chaos then recoverable |

**Secondary / exploratory:**

| Key | Regime | f / k | Observed tendency |
|-----|--------|-------|-------------------|
| `4` | worms | 0.026 / 0.055 | Predator trigger from Stable (50%) |
| `7` | bacteria | 0.046 / 0.065 | Strong stabilizer — Stable 61%, Rich 39% |
| `9` | stripes | 0.050 / 0.063 | Rich Ecosystem nudge — Stable→Rich 60% |
| `1` | spirals | 0.012 / 0.045 | Destabilizing |
| `2` | chaos | 0.020 / 0.045 | Heat Death risk — use carefully |
| `3` | waves | 0.014 / 0.047 | Heat Death risk — use carefully |

*Tendencies measured from 1,300+ intervention log entries. Not deterministic — hidden channel state at time of press changes the outcome. The same key from different macro-states produces different results.*

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
5. **Can the state space be steered?** *(confirmed — Near Extinction seed)* Stamping a dead zone (click → Near Extinction seed) breaks Predator Invasion and forces Rich Ecosystem reorganization — 9/9 confirmed. The C key provides directional hidden channel nudges; the seed provides a spatial wound the model must heal. Both are forms of control. The wound healing approach is more reliable because it writes to the visible chemistry (A/B channels), not just hidden state metadata.
6. **What is the creature actually doing to the rings?** The data shows rings entering the creature's body and exiting. The blob count fluctuates but doesn't drop to zero. Are the rings preserved topologically through the passage, or are they dissolved and reconstructed? Save-state analysis of ring identity across frames would answer this.

---

## Roadmap

**Open for collaboration — issues and PRs welcome.**

- `[x]` **Grid state save/restore** — S key saves grid + step count, R key restores
- `[x]` **Multi-species Lenia training** — v2: Orbium + Gyrorbium + Scutium, 3 species trained simultaneously
- `[x]` **Hidden channel analysis** — ch2 and ch4 confirmed active and independent; ch4 concentrates at blob boundaries (edge detection)
- `[x]` **Free channel experiment** — ch13 released at step 2000; model writes its own physics bit; produced binary orbit, comet fission, anti-creature orbital systems
- `[x]` **Behavioral grammar** — 8 macro-states identified via k-means; transition matrix measured; system is not random
- `[x]` **Physarum third teacher** — GS+Physarum trained; ch5 confirmed as pheromone trail channel (100% active); grammar is teacher-specific, not universal
- `[x]` **GS-only grammar experiment** — flat hidden channels confirmed; multi-physics tension required for grammar to emerge
- `[x]` **Stigmergic construction** — ch5 trail injection; organisms form living walls, junctions, stable corrals; structure rebuilds after erasure
- `[x]` **Chemical walls and doors** — W key draws walls; organisms cannot cross; D clears; click-to-stamp respects walls
- `[x]` **Blind Stigmergy Battle** — two LLMs fighting over one NCA grid through stigmergy alone; neither knows the other exists
- `[x]` **LLM artist** — Gemini paints with living organisms via spatial brush commands; human-in-the-loop direction; hold mode; drift shepherd tool
- `[ ]` **Artist video** — record a full session: Gemini builds a composition start to finish with narration
- `[ ]` **Transition predictor** — given current state features, predict next state transition before it happens
- `[ ]` **Gradual physics bit fade** — ramp ch13 over ~1000 steps instead of instant flip
- `[ ]` **Comparison video** — GS-only vs fused, same seed, same duration, side by side
- `[ ]` **Retrain classifier** — add new states from v2 runs (Lenia Chaos / Extreme Regime); fix Stable ↔ Near Extinction misclassification
- `[ ]` **GS+Physarum grammar alignment** — Hungarian-match Physarum cluster centroids against fused model; test whether ch5 role is conserved across teachers

---

## Future Directions

These are not planned features. They are hypotheses worth testing.

---

### Compartmentalized Substrate with Chemical Walls and Doors

**What:** Force A=1.0, B=0.0 on a persistent line of cells after each NCA step. This creates a chemical wall — B-rich structures cannot cross because their chemistry collapses against the barrier. A door is a gap in the line. Open/close by toggling which cells receive the forced values.

**Why it's trivially implementable:** the existing attractor seed system already forces chemistry into arbitrary cells post-step. A wall is just a persistent line-shaped seed. No retraining required.

**What should happen:**
- Creatures pile up at the wall face (the NCA treats forced-chemistry boundaries as wounds)
- Open the door → creatures rush through the gap
- Close the door → creatures on the wrong side are isolated
- Over time, genuinely different ecosystems develop on each side

**The real implication:** different behavioral grammar per sector. Run a separate feature extractor and classifier on each sector. Each sector has its own state, its own transition history, its own attractor basin — isolated from the others until a door opens.

**Per-sector physics:** force different f/k values on each side of the wall. Creatures crossing the door enter a different physical regime. A worms ecosystem on the left, a coral ecosystem on the right, connected by a controllable gate.

---

### Multi-LLM Substrate

**What:** Each walled sector has its own LLM observer. Each LLM reads its sector's feature vector in plain English, decides when and where to open doors, and stamps attractor seeds into its sector. The LLMs do not communicate directly — they communicate by controlling the chemistry that flows between sectors.

**Why this is different from existing multi-agent frameworks:** most multi-agent LLM systems have agents passing text messages. Here the agents act on a shared physical substrate and observe the consequences. The communication IS the chemistry. No message-passing protocol needed — the grid is the protocol.

**The immune response experiment (runnable today):**
1. Divide grid with a wall. Left sector: active Rich Ecosystem. Right sector: stamp Near Extinction seed (dead zone).
2. Open door between sectors.
3. Observe: does chemistry flood through the door to repair the dead zone? Does the NCA's wound-healing mechanism operate across sector boundaries?

This is a primitive immune response. Demonstrable with current code.

**Longer term:** an advanced NCA as the substrate between LLMs — not a message bus, not an API, but a continuous dynamical medium that carries, transforms, and routes information spatially. Hyper-fast (JAX, thousands of steps/second), self-healing (wound response confirmed), compositional (multiple independent grammars in separate sectors). The doors are the synapses.

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
