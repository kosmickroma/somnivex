# Somnivex — TODO (updated 2026-03-19)

## Immediate — Priority

- [ ] Add LICENSE file to repo (CC BY-NC 4.0 recommended)
- [ ] File ArXiv preprint — draft is in kktodo/preprint_draft.md, needs name filled in
- [ ] Decide: keep public repo or move to private for next phase

---

## LLM Artist — Next Steps

- [ ] Test vision feedback quality — does Gemini's speech show it's actually reading the canvas?
- [ ] Add `erase_trail cx cy r` — selective trail erasure (currently N key wipes everything)
- [ ] Tune ARTIST_STEPS — currently 300, may want to adjust based on vision feedback latency
- [ ] Session recording — clean 5-min video of full artist session with vision feedback active
- [ ] Try autonomous mode — "draw whatever you want" with vision, see if compositions improve

---

## LLM Artist — Known Issues

- Gemini still sometimes defaults to centered compositions despite off-center prompt fixes
- Blob spam partially fixed — `shape` commands no longer need blobs but Gemini still sometimes adds them
- Spatial placement of complex scenes (mountains, landscapes) still approximate without vision
- `wait` spam mostly fixed but still appears occasionally

---

## Blind Stigmergy / Multi-Agent

- [ ] Record a clean second Blind Stigmergy Battle video (first one is on YouTube)
- [ ] Try 3-agent battle (Claude + Gemini + third agent)
- [ ] Document the battle protocol more formally for the preprint

---

## Research / Publication

- [ ] Edit preprint_draft.md — fill in real name, verify references
- [ ] Create ArXiv account, get endorsement from CS/ALife researcher
- [ ] LinkedIn warm-up posts (battle video first, then artist video)
- [ ] Upload remaining videos unlisted before going public
- [ ] Reach out to Mordvintsev, Adamatzky after ArXiv is filed

---

## NCA System — Nice to Have

- [ ] Physarum grammar experiment — compare ch4 behavior vs lenia_100000.pkl
- [ ] Expand behavioral classifier from 4 states to 8 (need more diverse training data)
- [ ] GNOME idle detection for true screensaver mode
- [ ] Gallery auto-save — capture visually interesting moments to disk automatically

---

## Completed ✓

- [x] NCA v2 training — 3 Lenia species + GS, hidden channel noise, continuous ch13
- [x] LLM bridge — Gemini reading and writing to live NCA grid
- [x] Artist mode — full brush toolkit (trail, curve, arc, shape, wipe, blob, palette, mirror)
- [x] Vision feedback — screenshot sent to Gemini every turn
- [x] Blind Stigmergy Battle — Claude vs Gemini fighting over live NCA (YouTube: youtu.be/UgWdtZdLKdo)
- [x] Persistent compositions — LLM programs persist after disconnect (KK+X, bullseye+spiderweb)
- [x] speak.py — clean director input terminal for video recording
- [x] GS-only grammar experiment — flat hidden channels confirmed (multi-physics required)
- [x] Behavioral grammar — 8-state k-means, transition matrix, state_classifier.pkl
- [x] Walls/doors — interactive boundary drawing (W key)
- [x] kktodo/ removed from public repo
