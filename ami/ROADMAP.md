# AmI — Roadmap
*Build it first. Name it later.*

---

## Guiding Principle

Don't build the deep version first. Prove the plumbing works. See what it feels like when it runs. Then you'll know exactly what's missing.

---

## Phase 0 — Prove One Connection End To End

**Goal:** One sensor → NCA signal → one action. Nothing more.

Pick ONE of these:
- [ ] **Writing trigger**: File watcher sees keyword in notepad → injects signal into NCA zone → scraper fires → articles returned
- [ ] **Calendar trigger**: Calendar watcher sees upcoming meeting → signal propagates → relevant file surfaces → notification fires

**Why this first:** If the plumbing doesn't work nothing else matters. Keep it stupid simple. A file watcher, a signal injection, a scraper call. 20 lines of Python per piece.

**Success criteria:** You type a header in a text file. Without touching anything else, relevant content appears somewhere you can see it.

**Questions to answer:**
- File watcher or screen OCR? (depends on where you type)
- Which zone of the current NCA grid becomes the "writing" zone?
- What does "signal injection" look like in practice — same ch5 mechanism or new channel?
- Does the current checkpoint work for this or do we need a routing head on top?

---

## Phase 1 — Multiple Zones, Multiple Sensors

**Goal:** At least 3 zones doing 3 different things simultaneously.

- [ ] Writing zone (notepad watcher)
- [ ] Calendar zone (calendar API or file)
- [ ] Research zone (scrapers / RSS)

**Questions to answer:**
- How do zones stay independent? (spatial separation in the grid? channel separation?)
- What happens when two zones activate at the same time — do they interfere?
- Does the NCA naturally keep them separate or do we need to enforce boundaries?

---

## Phase 2 — Emergent Routing Grammar

**Goal:** Train a new NCA specifically for routing where the routing logic IS the physics, not bolted on top.

- [ ] Define the two incompatible signal types to create tension
- [ ] Design training data: what does "intent signal" look like spatially? what does "data signal" look like?
- [ ] Train from scratch (or from existing checkpoint)
- [ ] k-means cluster the hidden channels after training
- [ ] Read the vocabulary that emerged — what tokens did it invent?

**This is the research question:** Does routing grammar emerge from signal tension the same way physics grammar emerged from incompatible physics teachers?

---

## Phase 3 — LLM As Optional Layer

**Goal:** LLM only wakes up when something actually needs reasoning.

- [ ] Threshold detection: when does a zone activation warrant LLM involvement?
- [ ] Context assembly: when LLM is called, it gets rich context already assembled by the grid
- [ ] LLM acts, returns to sleep
- [ ] Prove the system works without the LLM for 90% of cases

---

## Phase 4 — It Feels Alive

**Goal:** Use it every day. It anticipates. It doesn't ask.

- [ ] Connect to actual calendar
- [ ] Connect to actual files
- [ ] Connect to actual writing environment
- [ ] Run for a week. Note every moment it does something useful you didn't ask for.

---

## Open Questions (parking lot)

- How do you encode text as a spatial signal the NCA can learn from? (embedding → channel injection?)
- Is the current lenia_100000.pkl checkpoint sufficient for Phase 0 or do we need a routing head?
- What's the right way to define zone boundaries in the grid?
- If routing grammar emerges — who reads it? LLM? Another NCA? Does it route itself?
- How do you train on "intent signals" when intent is fuzzy and context-dependent?
- Multi-modal eventually? Screen + audio + calendar + files all injecting simultaneously?
