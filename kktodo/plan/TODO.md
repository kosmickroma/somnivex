# Somnivex — Master TODO
*Single source of truth. Keep this updated. Last updated: 2026-03-18*

---

## HACKER NEWS PREP — Do before posting to HN (est. 2 weeks out)

- [ ] **Getting Started section in README** — clean install steps, dependencies, how to run the battle from scratch in under 10 min
- [ ] **Host checkpoint on HuggingFace** — `lenia_100000.pkl` must be downloadable, people can't train from scratch
- [ ] **Document API key setup** — Anthropic + Gemini keys, .env or bashrc, clear instructions
- [ ] **Short technical writeup** — 500 words on blind stigmergy concept, link from README (can live in notes/)
- [ ] **End-to-end test: clone → run** — simulate a stranger cloning the repo cold, fix every wall they'd hit

*Concept is HN front page material. Repo just needs to be ready for strangers.*

---

## #1 PRIORITY — Blind Stigmergy Battle (see BATTLE_MODE.md + BLIND_STIGMERGY.md)

Two LLMs (Claude=Destroyer, Gemini=Keeper), opposing goals, NEITHER knows the other exists.
Only connection: the NCA grid. Pure stigmergy. Never been done. This is the video.
Full spec: `kktodo/plan/BLIND_STIGMERGY.md`
Full game design: `kktodo/plan/BATTLE_MODE.md`

**POC CONFIRMED WORKING 2026-03-18** — turn alternation works, agents fighting, screenshots saved.

### Video Polish TODO (do before recording on GPU)

- [ ] **Dramatic overlay text** — two layers per agent panel:
  - BIG BOLD line: dramatic phrase mapped from command+state
    - Destroyer examples: "PUSHING TOWARD ANNIHILATION" / "LOCKING IN CHAOS" / "PREPARING THE KILL SHOT" / "ACCELERATING COLLAPSE"
    - Keeper examples: "HOLDING THE LINE" / "EMERGENCY STABILIZATION" / "FIGHTING BACK" / "REBUILDING THE ECOSYSTEM"
  - Small line underneath: technical reason from LLM + key stats
  - Bigger font overall, readable at video resolution

- [ ] **Terminal transparency** — current terminals are too opaque, blocking grid
  - Target: ~60-70% opacity — readable text but grid visible through background
  - Not too transparent or text becomes unreadable
  - From screenshots: neon_city palette bleeds through nicely, just needs lighter terminal bg

- [ ] **Pygame overlay panels** — render directly on grid (no external terminals needed for video)
  - Bottom-left: DESTROYER panel
  - Bottom-right: KEEPER panel
  - Center: "TURN X — DESTROYER'S MOVE" banner
  - Extinction flash: red pulse + "EXTINCTION"
  - Recovery: green pulse

- [ ] **Fixed turn step count** — currently 2-second delay, needs to be fixed N steps (800)
  - Eliminates timing side-channel, cleaner science, better video pacing
  - GPU: 800 steps ≈ ~13 seconds at 60 steps/sec

- [ ] **"none" move optimization** — if state is already good, skip API call, just hold
  - Reduces API costs on long runs, especially when stable for many turns
  - Add threshold: if same state + same command would be "none" for 3 turns, auto-hold

- [ ] **GPU session** — run on lenia_100000.pkl (fused model, hidden channels active)
  - CPU confirmed working, GPU is where the visual drama happens
  - Space turns out more on GPU — grid moves faster, needs time to respond
  - Record with screen capture, voiceover: "These two AIs have never been introduced..."

---

## RIGHT NOW (in progress)

- [ ] **GS+Physarum training run** — `physarum_XXXXXX.pkl` — resumed from step 24000, ~129 min remaining
  - GS + Physarum (no Lenia), hidden channel noise, CH13_CONTINUOUS=False
  - Tests: does ch4 boundary-tracking grammar emerge with a different second teacher?

---

## TOMORROW MORNING — First thing

- [ ] **Grammar alignment test** (when GS-only finishes)
  1. Run `python nca/run_free.py --gs --research` (update CHECKPOINT to gs_only_100000.pkl first)
  2. Collect CSV — dwell in each state 5-10 rows before moving
  3. Run Hungarian matching on cluster centroids: GS-only vs lenia_100000
  4. Check if same hidden channels activate for same behaviors
  5. Document result in `notes/` — this is the core hypothesis test

---

## TOMORROW — GitHub Issues Session

Write up 6 public issues. Each needs: title, what it does, why it matters, implementation plan, label.
Full details in `kktodo/tomorrow_issues.md`.

- [ ] Issue 1: Compartmentalized substrate — chemical walls, doors, multi-sector LLM control
- [ ] Issue 2: Directional bias control — phase drift slider + per-sector flow fields
- [ ] Issue 3: Mouse-drawn pheromone trails (V1: deposit B; V2: post-Physarum trail channel)
- [ ] Issue 4: Physarum as third teacher
- [ ] Issue 5: Universal grammar hypothesis — GS-only alignment test
- [ ] Issue 6: LLM bridge (file-based asyncio, command injection)

- [ ] **Mouse-drawn pheromone trails — Version 1** (no training needed, test before Physarum)
  - Left drag → deposit B=0.8 overlay, decays 0.95/frame
  - Right drag → deposit high A (clear zone / corridor wall)
  - ~40 lines in `nca/run_free.py`
  - See if structures actually follow before committing to Physarum training for it
  - Full spec: `notes/ideas_backlog.md` → "Mouse-Drawn Pheromone Trails"

---

## THIS WEEK — Research & Data

- [ ] **Collect more research data** — run `--research` mode, dwell in each state
  - Need more Zombie / Near-Extinction rows especially
  - Goal: enough data to expand classifier from 4 → 8 states
- [ ] **Rebuild classifier at 8 states** (once data is collected)
  - Run `python nca/train_predictor.py` after new CSV is in place

---

## THIS WEEK — Code / Features

- [ ] **Directional bias control**
  - Add `direction_bias` float [-1, 1] to run_free.py
  - Left/right arrow keys adjust ±0.1, show in HUD
  - Full spec: `notes/ideas_backlog.md` → "Directional Control"

- [ ] **Chemical walls and doors — Version 1**
  - Force A=1.0, B=0.0 on a line of cells post-step
  - W key: wall placement mode, D key: toggle door
  - Full spec: `notes/ideas_backlog.md` → "Grid Walls and Doors"

---

## NEXT TRAINING RUN — Physarum as Third Teacher

Steps in order:
- [ ] Type out `kktodo/physarum_typing/` exercise (01 → 02 → 03 → read 04)
- [ ] Run `python kktodo/physarum_typing/03_generate_training_data.py` → `nca/physarum_training_data.npz`
- [ ] Add Physarum pool + loss to `nca/train_lenia.py` (`--physarum` flag, 12% ratio, ch13=0.5)
  - ~80 lines, spec in `kktodo/physarum_typing/04_what_next.md`
- [ ] Train v3 model, watch for ch5 / dormant channel activation (grammar expansion test)

---

## NEXT — LLM Bridge Demo Video (priority, low token cost)

LLM bridge is BUILT and working (Claude + Gemini both tested). Goal: get a clean video fast.

- [ ] **GPU session: find Rich Ecosystem conditions**
  - Run `--research` mode, get to Stable, hands off, log naturally
  - Does Rich emerge on its own from Stable? How long does it take?
  - What blob count / regime going in makes it more likely?

- [ ] **Record demo video: Human vs LLM**
  - Start stable, intervene manually (push to chaos/extinction), watch LLM respond
  - Terminal overlay on top of grid so both visible in one screen recording
  - Voiceover: "I am going to intervene now" — audience sees LLM react in real time
  - This is the POC video, no new code needed, just record it

- [ ] **Terminal overlay on pygame window** (optional but clean for video)
  - Render last N bridge log lines directly onto the grid surface
  - One screen, no separate terminal needed

## FUTURE — Battle Mode (Claude vs Gemini, adversarial, ~20 min video)

Design locked. Build after demo video. This is the rooftop moment.

**The concept:** Claude (Destroyer) vs Gemini (Keeper). Turn-based. Live NCA is the battlefield.
Audience doesn't need to understand NCAs — they follow the drama.

**Personas:**
- Claude = DESTROYER — wants extinction. Trash talks in REASON field.
- Gemini = KEEPER (Defender of the Universe) — wants Rich Ecosystem. Responds to threats.
- Opening title card: "DESTROYER vs DEFENDER OF THE UNIVERSE" with VS. graphic

**Video layout:**
- Two translucent terminal overlays, bottom corners, one per agent
- Each shows: agent name, last move, reason (the trash talk)
- "TURN X — DESTROYER'S MOVE" banner between turns
- Grid fills the screen behind both terminals

**Game mechanics:**
- Turn-based: alternating moves, NCA physics runs N steps between turns
- Keeper resources: 1 regime_1 move (recharges every X frames) + 1 mouse resurrection (rare)
- Destroyer resources: unlimited chaos/predator moves but chaos is slow — timing matters
- Destroyer optimal play: chaos → stall → let Keeper panic and waste regime_1 → global blob → kill
- Keeper optimal play: don't spend regime_1 on survivable chaos, save it for the window before global blob
- Key skill gap: Keeper has a window between chaos and global blob where regime_1 stabilizes — miss it and regime_1 triggers extinction instead

**Win conditions:**
- Destroyer: blank screen holds for N frames (extinction confirmed)
- Keeper: Rich Ecosystem holds for M consecutive frames

**Key finding from CPU testing:**
- Global Blob → any regime key = extinction (confirmed)
- Chaos → regime_1 before global blob = recovery (window exists, timing critical)
- Resurrection: mouse click only, recharges slowly, both agents know the timer

**Gaps to fill before building:**
- GPU session: find reliable path to Rich Ecosystem (Keeper's win condition)
- Tune turn length so ~20 min video has 15-20 meaningful turns
- Add extinction detection to classifier (b_active < 0.01 AND n_blobs == 0)

---

## LONG TERM / RESEARCH

- [ ] MHD / plasma physics as 4th teacher (dilated perception kernels needed)
  - Spec: `notes/ideas_backlog.md` → "Physics Simulation as Living Art"
  - Solar Dynamics Observatory data as training input (Ghost of Kepler idea)
- [ ] Taste learning — preference model on like/dislike keys, steers f/k
- [ ] Livestream mode — 24/7 autonomous, OBS output, novelty drive
- [ ] Paper: spatial parameter fields — draft in `notes/paper_draft.md`

---

## ARCHIVE — Old kktodo files (completed / superseded)

These files in `kktodo/` are from the original build-out phase. NCA is built and running.
They're kept for reference but nothing actionable remains in them.
- `00_overview.md` — original architecture plan (done)
- `01_nca_model.md` through `06_run_lenia.md` — original typing exercises (done)
