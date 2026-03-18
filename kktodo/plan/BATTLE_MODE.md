# Battle Mode — Claude vs Gemini over a shared NCA
*Created: 2026-03-18*

## The Core Concept

Two LLMs (Claude = Destroyer, Gemini = Keeper) fight over a live Neural Cellular Automaton.
They do NOT communicate with each other. The ONLY connection between them is the NCA grid.
One agent's action changes the physics. The NCA processes it. The other agent reads the result.
This is stigmergy — the same mechanism ants use to communicate via pheromone trails.

**The line:** "They're not talking to each other. The only thing between them is this."

---

## Isolation Rules — Critical

The two agents MUST have zero connection except through the NCA. Checklist:

- [ ] Two separate bridge processes, no shared memory, no IPC between them
- [ ] Each agent reads ONLY the feature CSV (NCA state) — no knowledge of other agent's process
- [ ] Each agent knows the OTHER's last move (from a shared turn log) — but that's it
- [ ] No shared API client, no shared history object, no shared anything except:
  - The feature CSV (NCA output)
  - The turn log (who moved last, what they did)
  - The command files (each agent has its own: llm_commands_keeper.txt, llm_commands_destroyer.txt)
- [ ] run_free.py executes only the active agent's command file each turn
- [ ] Verify: if you killed the NCA process, the two agents would have NO way to communicate

---

## Architecture

```
run_free.py (NCA)
    ↓ writes every 200 steps
features_XXXXXX.csv  ←── both agents read this (read-only, no conflict)

turn.txt             ←── shared turn file: "keeper" or "destroyer" + turn number
llm_commands_keeper.txt     ←── Keeper writes here, NCA executes on Keeper's turn
llm_commands_destroyer.txt  ←── Destroyer writes here, NCA executes on Destroyer's turn

bridge_keeper.py   (Gemini)  — reads CSV + turn.txt, writes llm_commands_keeper.txt
bridge_destroyer.py (Claude) — reads CSV + turn.txt, writes llm_commands_destroyer.txt
```

Simplest implementation: add `--agent keeper` and `--agent destroyer` flags to llm_bridge.py.
Turn file advances after NCA confirms command executed (or after N steps).

---

## Game Rules

**Personas:**
- DESTROYER (Claude): wants extinction. Trash talks.
- KEEPER / "Defender of the Universe" (Gemini): wants Rich Ecosystem or just survival.

**Turn structure:**
- Turns alternate: Keeper → Destroyer → Keeper → Destroyer
- After each move, NCA runs ~300 steps before next turn (enough for physics to respond)
- Each agent sees: current grid state + whose turn it is + opponent's last move + reason

**Resources:**
- Keeper: regime_1 move (unlimited but costs a turn), mouse resurrection (1 use, recharges every ~600 steps)
- Destroyer: all commands available, chaos is slow-acting so timing matters

**Win conditions:**
- Destroyer wins: blank screen (b_active < 0.01, n_blobs == 0, dark > 0.99) holds for 300 frames
- Keeper wins: Rich Ecosystem (ch4_border_ratio > 5, n_blobs 4-10, stable) holds for 300 frames
- Draw: neither condition met after 60 turns

**Extinction detection (add to feature CSV):**
```python
extinct = (b_active < 0.01) and (n_blobs == 0) and (dark > 0.99)
```

---

## Destroyer Strategy (Claude system prompt)
- Optimal path: regime_3 (cheap, tests Keeper) → if no response, chaos → stall at chaos →
  let Keeper panic and waste regime_1 → global blob forms → hit regime key → extinction
- Feint option: push toward predator, back off, repeat to drain Keeper's patience
- Kill window: Keeper's resurrection recharge timer is known — strike when timer is high
- Awareness: told Keeper's last move and resurrection timer status each turn

## Keeper Strategy (Gemini system prompt)
- Don't spend regime_1 on survivable chaos — read the trend, not just the snapshot
- The window: chaos → regime_1 before global blob = recovery. Miss it = extinction.
- Resurrection is nuclear option. Both agents know when it was last used.
- Goal: survive long enough for Rich Ecosystem to emerge naturally from Stable

---

## Video Plan

**Runtime target:** ~20 minutes, unscripted, genuine outcomes

**Layout:**
- Grid fullscreen
- Bottom-left overlay: DESTROYER (Claude) — last move + trash talk reason
- Bottom-right overlay: KEEPER (Gemini) — last move + reason
- Center banner between turns: "TURN 7 — DESTROYER'S MOVE"
- On extinction: red flash, "EXTINCTION" banner, resurrection timer visible
- On Rich Ecosystem: green pulse, "RICH ECOSYSTEM — KEEPER HOLDS"

**Opening:** Both terminals visible, personas announced, VS. graphic in video edit
**Hook:** "Two AIs. One living simulation. They can only talk through this."
**Drama moment:** Destroyer triggers extinction, Keeper uses resurrection click, crowd goes wild

---

## Build Order

1. [ ] Add `extinct` flag to feature CSV in run_free.py
2. [ ] Add `--agent` flag to llm_bridge.py (keeper/destroyer)
3. [ ] Add turn.txt polling — each agent waits for its turn before calling API
4. [ ] Write Destroyer system prompt (inverted goals, trash talk encouraged, knows Keeper's timer)
5. [ ] Write Keeper system prompt (survival focus, aware of Destroyer's last move)
6. [ ] Add resurrection command to run_free.py (single mouse-click equivalent, tracked)
7. [ ] Add pygame overlays (two panels, turn banner, extinction flash)
8. [ ] Test run on CPU first — verify isolation, verify turn alternation, verify win detection
9. [ ] Record on GPU — faster physics, more dramatic

---

## Open Questions

- How many steps between turns feels right for ~20 min video? (start with 300, tune)
- Does Destroyer need a "bluff" command — something that looks threatening but isn't?
- Should agents know the turn number (so Destroyer can play endgame differently)?
- Sound effects? Explosion on extinction, chime on Rich Ecosystem?
