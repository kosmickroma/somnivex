# 00 — Overview: What We're Building and Why

Read this before anything else. It's the map.

---

## The Big Picture

Right now Somnivex runs Gray-Scott — a beautiful mathematical simulation, but it's not AI.
It doesn't learn. It doesn't create. It just computes a well-known equation forever.

What we're building next is the actual AI model — a Neural Cellular Automaton (NCA) that:
1. Learns HOW Gray-Scott works (the physics, not the pixels)
2. Runs on its own using what it learned
3. Eventually generates patterns GS can never produce

The end state: the screensaver runs the NCA. GS is retired or runs in the background.
The NCA is the artist. You just nudge it.

---

## The Teaching Strategy

We're not showing the NCA pretty GS images and saying "make this."
We're teaching it the UPDATE RULE — the moment-to-moment physics.

Every training step:
- Show NCA a GS grid state at time T (with f and k as control channels)
- Ask NCA: what does the next state look like?
- Show it what GS actually produced at time T+1
- Measure how wrong the NCA was (loss)
- Adjust the NCA's weights to be less wrong next time
- Repeat 20,000 times

After training: the NCA has internalized "if f is high and k is low, diffusion dominates,
do this kind of spreading." It learned the physics. Not from being told — from watching.

---

## The Pool (Key Idea from the Paper)

Don't just train start → finish → repeat. That teaches the NCA to grow patterns but not
to HOLD them. Screensavers need to run for hours. Patterns have to stay alive.

Instead: maintain a pool of 512 live GS states. Each training step:
- Grab 32 states from the pool
- Train NCA on them (compare NCA output to GS output)
- Write the NCA's outputs BACK to the pool

Next time those states get sampled, they're the NCA's own outputs — not fresh GS.
The NCA has to keep them looking like GS. Patterns become attractors, not trajectories.
This is what creates long-term stability. This is the trick.

---

## Files to Write (in order)

### 01 — nca/model.py (full rewrite)
The NCA cell brain. Already in kktodo. Write this first.
Changes: control channels (f/k), zero-init final layer, no alive mask, fire rate 0.5.

### 02 — nca/train.py (new file)
The training loop. This is where learning actually happens.
Pool-based, one-step prediction, persistence loss, checkpointing.
Run this once to train the model. Takes 30-60 mins on your GPU.

### 03 — nca/run_free.py (new file)
A standalone script to test the trained NCA on its own.
Loads a checkpoint, runs the NCA with no GS, shows output in a pygame window.
This is how you answer: "can it run on its own and does it look good?"

### 04 — main.py modifications
Plug the trained NCA into the screensaver. Run it alongside GS for comparison first,
then phase GS out. Kktodo file for this will be written after 01-03 are done.

---

## What "Success" Looks Like at Each Stage

After typing 01 (model rewrite):
→ The file exists. Nothing runs yet. It's just the brain sitting there.

After typing 02 (train.py) and running it:
→ You see loss numbers printing. They go down over time. Checkpoints appear in nca/checkpoints/.
→ Good: loss drops from ~0.01 to ~0.001 over 20k steps
→ Bad: loss explodes or stays flat (means something is wrong, come back and debug together)

After typing 03 (run_free.py) and running it:
→ A pygame window opens. You see patterns. They either:
   a) Look GS-like but run forever → success, training worked
   b) Die out, go blank, or explode → training needs more steps or something's wrong
   c) Look like something new / weird → also interesting, don't panic

After 04 (integration):
→ The screensaver runs the NCA. GS is still there for comparison.
→ You can toggle between them. See which looks better.

---

## Before You Run Training: Install optax

The training script uses optax for the optimizer. Check if you have it:

    source /home/kk/ai-env/bin/activate
    python -c "import optax; print(optax.__version__)"

If that fails:

    pip install optax

---

## Important Numbers

- Training grid: 64×64 (not 256×256 — pool memory would be 2GB, too much)
- Inference grid: 256×256 (works because all ops are local 3×3 convolutions)
- Pool size: 512 states
- Batch size: 32 per step
- Training steps: 20,000 (about 30-60 mins on GTX 1650)
- Checkpoint every: 1,000 steps
- Loss you want to see: dropping from ~0.01 toward ~0.001

---

## The Channel Layout (burned into your brain before you type anything)

    Channel 0  = A  (Gray-Scott food chemical)
    Channel 1  = B  (Gray-Scott predator — this is what makes the patterns)
    Channel 2  = hidden
    ...
    Channel 13 = hidden      (12 hidden channels total, NCA decides what to do with them)
    Channel 14 = f  (feed rate — written in from outside before every step)
    Channel 15 = k  (kill rate — written in from outside before every step)

The NCA reads f and k every step and learns to behave differently based on them.
One trained model covers all 15 GS regimes. Control channels are the knobs.

---

## After Training: The Fun Part

Once the NCA is trained, you can:
- Set f and k to values GS never visited — the NCA extrapolates
- Smoothly morph f and k in ways that produce novel transitions
- Freeze f and k mid-run and watch the NCA settle into attractors
- Eventually: have the preference model generate f/k nudges toward your taste

The NCA isn't just replaying GS. It built an internal model of diffusion-reaction physics.
What it does with that model when you poke it is what makes this interesting.
