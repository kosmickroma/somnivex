# AmI — Roadmap
*Build it first. Name it later.*

---

## Guiding Principle

Don't build the deep version first. Prove the plumbing works. See what it feels like when it runs. Then you'll know exactly what's missing.

---

## Phase 0 — Prove One Connection End To End ✓ DONE (2026-03-23)

**Goal:** One sensor → NCA signal → one action.

- [x] File watcher sees keyword → classifies intent → injects signal into NCA → LLM fires
- [x] Signal_id UUID chain enforces honest routing — LLM cannot fire before signal physically travels through NCA
- [x] Two signal types (politics → Claude, climate → Gemini) both working reliably end to end

**What was built:**
- `watcher.py` — watches input.txt, classifies intent, fires signal with UUID
- `experiment_nca.py` — trained NCA substrate, reads trigger, routes signal, writes zone state
- `responder.py` — watches zone state, verifies signal_id, calls correct API

**Proven:** "researching AI in politics" → Claude fires. "researching climate tech" → Gemini fires.

---

## Phase 1 — Routing Grammar Emerges ✓ DONE (2026-03-23)

**Goal:** Train a new NCA to route signals — routing logic in the physics, not bolted on.

- [x] Three-zone grid design (Zone A input, Zone B Claude, Zone C Gemini)
- [x] Two signal shapes (horizontal bar = politics, vertical bar = climate)
- [x] Training pipeline: pool-based rollout, routing loss, persistence loss
- [x] NCA trained from physarum_100000.pkl — 20,000 steps
- [x] Routing grammar analyzed — ch1 emerges as climate routing token, ch3 as politics marker
- [x] Both signals routing correctly above threshold (B=0.134 politics, C=0.151 climate avg across 16 test states)

**The research question was answered:** Yes, routing grammar emerges from signal tension. ch1 and ch3 developed as internal routing tokens — nobody designed them. They appeared because the training pressure demanded a way to distinguish signal types.

**Checkpoint:** `ami/routing_checkpoints/routing_020000.pkl`
**Analysis:** run `python ami/analyze_routing.py`
**Decisions log:** `ami/DECISIONS.md`

---

## Phase 2 — Richer Signals, More Zones (NEXT)

**Goal:** More than two intent types. More than two output zones.

- [ ] Add a third signal type and third output zone
- [ ] Test whether routing generalizes to a signal type not seen in training
- [ ] Richer watcher: detect more intent types from writing context, not just keyword lists
- [ ] Screen watcher: detect intent from what's visible on screen, not just what's typed in a file
- [ ] Log every routing event as future training data

**Open questions:**
- Does the routing NCA generalize to a third intent type it never saw in training?
- What's the minimum signal difference the NCA can reliably distinguish?
- Can we train more signal types and have the grammar stay coherent?

---

## Phase 3 — Substrate Persistence & Self-Repair

**Goal:** Grid stays alive indefinitely without hacks.

- [ ] Current hack: blending base trail every 50 steps to keep grid alive — needs a proper fix
- [ ] Train NCA to maintain its own substrate without external injection
- [ ] Test: kill signal mid-route. Does routing resume when signal returns?
- [ ] Self-repair: if zone connection breaks, does trail rebuild the same way Physarum reroutes?

---

## Phase 4 — LLM As Optional Layer

**Goal:** LLM only wakes up when something actually needs reasoning.

- [ ] Threshold detection: when does a zone activation warrant LLM involvement?
- [ ] Context assembly: when LLM is called, it gets rich context already assembled by the grid
- [ ] LLM acts, returns to sleep
- [ ] Prove the system works without the LLM for 90% of cases

---

## Phase 5 — It Feels Alive

**Goal:** Use it every day. It anticipates. It doesn't ask.

- [ ] Connect to actual calendar
- [ ] Connect to actual files and writing environment
- [ ] Run for a week. Note every moment it does something useful you didn't ask for.

---

## Open Questions (parking lot)

- How do you encode richer text intent as a spatial signal the NCA can learn from? (embedding → channel injection?)
- If routing grammar emerges for 3+ signal types — does the vocabulary stay interpretable?
- Multi-modal eventually? Screen + audio + calendar + files all injecting simultaneously?
- The substrate persistence problem: NCA collapses to zero in free-run. Is that a training problem or architecture?
