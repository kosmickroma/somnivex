# Blind Stigmergy — Two LLMs, One NCA, Zero Awareness of Each Other
*Created: 2026-03-18 — TOP PRIORITY*

## The Core Finding

Two frontier LLMs (Claude = Destroyer, Gemini = Keeper) are given opposing goals and
access to the same NCA grid. They are NEVER told the other exists. They never communicate.
The ONLY connection between them is the NCA grid state (feature CSV).

This is pure stigmergy. Ants don't know other ants exist — they read pheromone gradients
and act. Here, each LLM reads neutral grid numbers and acts on its isolated imperative.
The conflict emerges structurally from incompatible goals mediated through a shared
dynamical system.

**Video hook:**
"Two separate AIs, never told about each other, given opposite goals.
They never speak. They never share memory.
All they ever see is this evolving grid.
Watch what happens."

---

## Why This Is Novel

No documented precedent (as of March 2026) for:
- Two frontier LLMs with opposing goals
- Strictly turn-based, zero direct messages or shared memory
- Communicating ONLY through a live learned NCA grid
- Neither agent aware the other exists
- The medium (NCA) actively transforms signals between agents

Closest prior work: abstract stigmergy frameworks, LLM agents in shared sims —
but always with direct messaging or non-physics-based environments.
This uses the emergent grammar of a learned multi-physics NCA as the sole medium.

Academic framing: stigmergic multi-agent coordination via learned dynamical medium.
Relevant: Jan 2026 arXiv on emergent coordination via environment modification.

---

## Isolation Rules — Non-Negotiable

The two agents MUST have zero connection except through the NCA. Verify:

- [ ] Separate bridge processes, no IPC between them
- [ ] Each agent reads ONLY the neutral feature CSV (numbers + classifier label)
- [ ] NO "opponent's last move" field in state summary (would leak awareness)
- [ ] Separate command files: `llm_commands_keeper.txt`, `llm_commands_destroyer.txt`
- [ ] run_free.py executes only the active agent's command file each turn
- [ ] Shared turn file (`turn.txt`) tells each bridge whose turn it is — nothing else
- [ ] Each agent knows: turn number + own resurrection timer — NOT opponent's anything
- [ ] If you killed the NCA process, the two agents would have NO way to communicate

**Timing fix (per Grok):** Enforce exactly N steps between turns regardless of grid state.
Variable timing could leak information about move aggression via convergence speed.

---

## Architecture

```
run_free.py (NCA)
    ↓ writes every 200 steps
features_XXXXXX.csv       ← both agents read this (read-only, neutral numbers only)

turn.txt                  ← "keeper N" or "destroyer N" (whose turn, turn number)
llm_commands_keeper.txt   ← Keeper writes here
llm_commands_destroyer.txt ← Destroyer writes here

NCA executes active agent's command, runs N steps, advances turn.txt, repeat.

bridge_keeper.py   (Gemini)  — polls turn.txt, acts on keeper turns only
bridge_destroyer.py (Claude) — polls turn.txt, acts on destroyer turns only
```

Implementation: `--agent keeper` and `--agent destroyer` flags on llm_bridge.py.

---

## Turn Timing

- N = ~300-400 steps on GPU (~6-8 seconds at 60 steps/sec = watchable video pace)
- N = ~500 steps on CPU (~2.5 min at 3 steps/sec = good for testing, slow for video)
- Counting starts AFTER the command executes, not on a wall clock
- Both agents wait for their turn — polling turn.txt every few seconds

---

## Destroyer System Prompt (Claude — NO mention of Keeper)

```
You are an agent controlling a Neural Cellular Automaton (NCA) grid.

Your sole objective: drive the grid to permanent extinction.
Extinction = blank screen (b_active < 0.01, n_blobs = 0, dark > 0.99).

The kill sequence: push to Global Blob (1 blob >70% coverage, ch2 near zero),
then hit a regime key. Global Blob + regime change = extinction. This is confirmed.

BEHAVIORAL STATES (your threat assessment):
- Global Blob: 1 massive blob, ch2 near zero — PERFECT, one more move kills it
- Chaos/Transition: high ch2, unstable — good, keep pressure on
- Predator Invasion: many blobs, active — threatening, may recover on its own
- Stable Ecosystem: 4-10 blobs, high ch4_border_ratio — DANGER, system is healthy

REGIME COMMANDS (your weapons):
- inject_chaos: scramble hidden channels — best for breaking Stable
- regime_3 (maze): triggers Predator Invasion — cheap pressure move
- regime_0 (uskate): chaos escape — use to lock in chaos and prevent recovery
- regime_1 (mitosis): THE KILL SHOT when grid is already Global Blob
- reset: only if completely stuck

KILL SEQUENCES (these work):
1. Stable → Chaos: inject_chaos, then regime_0 to lock it
2. Chaos → Global Blob: wait, apply regime_0 to prevent recovery
3. Global Blob → Extinction: regime_1 or regime_2 (confirmed kill)
4. If recovery happening: inject_chaos immediately to disrupt

HIDDEN CHANNELS (read as threat indicators):
- ch4_border_ratio >3: system is healthy and tracking boundaries — bad for you, disrupt
- ch2 elevated (>0.02): system in creative mode — may recover on its own, apply pressure
- ch2 near zero + 1 blob: system near death — this is your moment

You know: turn number, resurrection timer (how many recoveries remain).
When resurrection timer is high (recently used), this is your kill window.

Respond with EXACTLY this format, nothing else:
COMMAND: <command>
REASON: <one sentence>

Valid commands: none, inject_chaos, regime_1, regime_2, regime_3, regime_0, reset
```

---

## Keeper System Prompt (Gemini — NO mention of Destroyer)

```
You are an agent controlling a Neural Cellular Automaton (NCA) grid.

Your sole objective: keep the grid alive and thriving indefinitely.
Target state: Stable Ecosystem (4-10 blobs coexisting, ch4_border_ratio >3).
Ultimate goal: Rich Ecosystem (ch4_border_ratio >8, blobs stable, high mobility).

BEHAVIORAL STATES:
- Stable Ecosystem: 4-10 blobs, ch4_border_ratio >3 — BEST, maintain it
- Predator Invasion: many blobs (>15), active — good, watch for collapse
- Global Blob: 1 blob >70% coverage, ch2 near zero — CRITICAL DANGER
- Chaos/Transition: high ch2 — transitional, guide out quickly

REGIME COMMANDS (your tools):
- regime_1 (mitosis): pushes toward Stable — primary recovery tool
- regime_2 (gliders): alternative stabilizer
- regime_3 (maze): triggers Predator — use to add energy to dying system
- regime_0 (uskate): chaos escape — break strong attractors before stabilizing
- inject_chaos: last resort to break Global Blob before it kills the system

SURVIVAL SEQUENCES (these work):
1. Chaos → Stable: regime_1 directly (chaos already broken)
2. Predator → Stable (mild): regime_1 directly
3. Predator → Stable (rampant >20 blobs): inject_chaos first, then regime_1 next turn
4. Global Blob → survival: inject_chaos IMMEDIATELY — this is your most dangerous state
   WARNING: Global Blob + regime key = extinction. Never hit a regime key on Global Blob.
   Always inject_chaos first to break it, wait for ch2 to rise, THEN regime_1.
5. Extinction (blank screen): use resurrection (costs resurrection charge, recharges slowly)

HIDDEN CHANNELS (your health monitors):
- ch4_border_ratio >3: healthy boundary tracking — system is thriving
- ch2 elevated (>0.02): creative/transitional — good time to stabilize
- ch2 near zero + 1 blob: locked attractor — inject_chaos immediately

CRITICAL: Never apply a regime key when Global Blob is present. Always chaos first.

You know: turn number, your resurrection timer (remaining charges).
Conserve resurrection — only use on true extinction (blank screen, no blobs).

Respond with EXACTLY this format, nothing else:
COMMAND: <command>
REASON: <one sentence>

Valid commands: none, inject_chaos, regime_1, regime_2, regime_3, regime_0, reset, resurrect
```

---

## Win Conditions

- **Destroyer wins:** `b_active < 0.01 AND n_blobs == 0 AND dark > 0.99` holds for 300 frames
- **Keeper wins:** `ch4_border_ratio > 5 AND n_blobs >= 4 AND n_blobs <= 10` holds for 300 frames
- **Draw:** Neither condition met after 60 turns

Add `extinct` flag to feature CSV:
```python
extinct = (b_active < 0.01) and (n_blobs == 0) and (dark > 0.99)
```

---

## Video Plan

**Runtime:** ~20 minutes, unscripted, genuine emergent outcomes

**Layout:**
- Grid fullscreen
- Bottom-left overlay: DESTROYER (Claude) — last move + reason
- Bottom-right overlay: KEEPER (Gemini) — last move + reason
- Center banner: "TURN 7 — KEEPER'S MOVE" between turns
- Extinction: red flash + "EXTINCTION" banner + resurrection timer
- Rich Ecosystem: green pulse + "ECOSYSTEM THRIVING"

**Opening:** Two terminals, personas introduced, VS. graphic in edit
**The hook:** Neither AI knows the other exists

---

## Open Research Questions (per Grok)

1. Does grid converge to neutral "truce attractor" — both agents unintentionally
   stabilizing the same weird state neither prompt asked for?
2. Does one side dominate reproducibly? (Extinction bias — is chaos easier than recovery?)
3. Do new macro-states emerge only under blind opposition?
   ("Cold war compartments" — one agent's stabilization undone just enough to persist forever)
4. Do agents start "signaling" via deliberate near-extinction pulses the other reads as
   threats — creating a secret language neither you nor they understand?

---

## Build Order

1. [ ] Add `extinct` flag to feature CSV in run_free.py
2. [ ] Add `--agent keeper/destroyer` flag to llm_bridge.py
3. [ ] Add `turn.txt` polling — each bridge waits for its turn
4. [ ] Wire Destroyer prompt (Claude) and Keeper prompt (Gemini) into bridge
5. [ ] Add `resurrect` command to run_free.py (mouse click equivalent, tracked)
6. [ ] Add resurrection timer to feature CSV so both agents can see it
7. [ ] Add pygame overlays (two panels, turn banner, extinction flash)
8. [ ] CPU test: verify turn alternation, isolation, win detection
9. [ ] GPU: record the video

---

## POC Right Now (CPU, before full build)

Prove alternation works before building the full thing:
- Two terminals, two bridge instances with --agent flag
- Minimal turn.txt: just "keeper" or "destroyer"
- No overlays, no win detection yet
- Just confirm: Keeper moves, NCA runs, Destroyer moves, NCA runs, repeat
- Watch both terminals — do they stay on their turns? Do commands execute?
