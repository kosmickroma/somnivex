# AmI Architecture Decisions

---

## 2026-03-23 — Removed CH_SIGNAL from routing NCA training

**Decision:** Removed CH_SIGNAL (ch11) as an explicit signal type channel from train_routing.py.

**What it was:** A uniform scalar injected across the entire grid — 0.5 for politics, 1.0 for climate. Every cell knew what signal type it was carrying.

**Why we removed it:** If every cell already has the answer, there's no tension to resolve. The NCA would just conditionally route on a value we handed it — no hidden channel grammar needs to emerge. This is not what we're building.

**What we want instead:** The NCA gets only the trail shape in Zone A (horizontal bar = politics, vertical bar = climate) plus zone identity. The Sobel filters in perception already detect orientation gradients. The NCA must develop something in ch2-ch10 to carry "horizontal signal" vs "vertical signal" as the trail propagates across the grid. That internal encoding IS the routing token.

**Analogy to existing work:** ch5 emerged because GS and Physarum were incompatible — the visible channels couldn't carry both, so a hidden channel emerged to resolve the tension. Same principle here.

**Risk:** Might be too hard to train. If the NCA can't develop the routing grammar, routing loss won't drop and we'll get nothing useful.

**If it fails / to revert:**
- Add back `CH_SIGNAL = 11` constant
- In `generate_state`: add `grid[:, :, CH_SIGNAL] = sig_val` with sig_val = 0.0/0.5/1.0
- Restore ROLLOUT_STEPS to 6 (or keep at 24 — that change is independent and probably right regardless)
- The pre-flooding fix (inject Zone A shape only, not target zones) should be kept either way

**Other changes made same session:**
- ROLLOUT_STEPS: 6 → 24 (NCA needs enough steps to physically cross the gap, ~7 cells away)
- make_target: removed zone painting, pure trail diffusion only — let routing_loss be the sole routing teacher
- ROUTING_WEIGHT: 2.0 → 5.0 (more pressure since routing task is now genuinely hard)
