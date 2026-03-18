# Physarum Field Simulator — Typing Exercise

Type out each file in order. Read the explanation before typing the code.
Don't copy-paste — typing it forces you to actually read every line.
When you finish a file, run it and see what it does before moving on.

## Files to type (in order)

1. `01_trail_map.py` — the core field: diffuse, decay, deposit
2. `02_visualize.py` — watch it run in real time with pygame
3. `03_generate_training_data.py` — save frames as numpy arrays for NCA training
4. `04_what_next.md` — how this connects to train_lenia.py

## What you're building

The Jeff Jones (2010) Physarum model has two layers:
- **Trail map** — a 2D grid of floats, the pheromone concentration field
- **Agent layer** — thousands of particles that sense the trail and deposit more

For training the NCA we only need the trail map dynamics, not the agents.
The trail map update is: diffuse → decay → deposit.
That's it. Three operations. The NCA already does all three with its own channels.

When you're done you'll have a standalone Physarum simulator that produces
(H, W) float arrays at each step — exactly the format the NCA training pipeline
already expects from GS and Lenia.

## Why this matters

GS teacher → NCA learns "blobs and waves"
Lenia teacher → NCA learns "locomotion"
Physarum teacher → NCA learns "trail reinforcement and directed flow"

The hidden channel that handles this (probably ch5 — it's been slowly growing)
will need to represent something like "has flow been here recently."
That's a concept neither GS nor Lenia needed. New vocabulary.
