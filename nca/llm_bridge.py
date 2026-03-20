# nca/llm_bridge.py — LLM bridge for Somnivex
#
# Single-agent mode (default):
#   python nca/llm_bridge.py --provider gemini --log
#
# Battle mode (two terminals, blind stigmergy):
#   Terminal 1: python nca/llm_bridge.py --agent keeper   --provider gemini   --log
#   Terminal 2: python nca/llm_bridge.py --agent destroyer --provider anthropic --log
#
# Optional flags:
#   --log            print every state summary and response
#   --dry-run        summarize and print but never call the API
#   --interval N     seconds between checks (single-agent mode only, default: 15)
#   --provider NAME  anthropic (default) or gemini
#   --model NAME     override model
#   --agent NAME     keeper or destroyer (enables battle/turn-based mode)

import os
import sys
import argparse
import asyncio
import csv as _csv
from pathlib import Path
from datetime import datetime
import numpy as np

# ── Config ────────────────────────────────────────────────────────────────────

LOG_DIR      = os.path.join(os.path.dirname(__file__), 'logs')
CMD_FILE     = os.path.join(os.path.dirname(__file__), 'llm_commands.txt')
TURN_FILE    = os.path.join(os.path.dirname(__file__), 'battle_turn.txt')
CHECK_EVERY  = 15        # seconds between checks (single-agent mode)
TURN_POLL    = 2         # seconds between turn file polls (battle mode)
ROWS_TO_READ = 5         # how many recent feature rows to summarize
MAX_HISTORY  = 12        # how many prior exchanges to keep in context

DEFAULT_MODELS = {
    'anthropic': 'claude-sonnet-4-6',
    'gemini':    'gemini-2.5-flash',
}
DEFAULT_MODELS_BLUEPRINT = {
    'anthropic': 'claude-haiku-4-5-20251001',
    'gemini':    'gemini-2.5-flash',
}

# Agent-specific command files (battle mode)
AGENT_CMD_FILES = {
    'keeper':    os.path.join(os.path.dirname(__file__), 'llm_commands_keeper.txt'),
    'destroyer': os.path.join(os.path.dirname(__file__), 'llm_commands_destroyer.txt'),
}

VALID_COMMANDS = [
    'none',
    'inject_chaos',
    'regime_1',
    'regime_2',
    'regime_3',
    'regime_0',
    'reset',
]

# ── System prompts ────────────────────────────────────────────────────────────

SYSTEM_PROMPT_SINGLE = """You are an agent controlling a Neural Cellular Automaton (NCA) grid.

The NCA runs Gray-Scott reaction-diffusion physics. You read periodic grid state summaries
and issue one command per check to keep the grid in interesting dynamic states.

BEHAVIORAL STATES:
- Stable Ecosystem: 4-10 blobs coexisting, ch4_border_ratio high (>3) — BEST STATE, maintain it
- Predator Invasion: many blobs (>15), competitive, active — good, watch for collapse
- Global Blob: 1 massive blob covering >70% of grid, ch2 near zero — BORING, break it up
- Chaos/Transition: high ch2, low mobility or very high mobility — transitional, guide it out

REGIME COMMANDS (change the physics f/k parameters):
- regime_1 (mitosis): pushes toward stable/rich state — use when you want Stable Ecosystem
- regime_2 (gliders): pushes toward stable glider-friendly state — alternative stabilizer
- regime_3 (maze): triggers Predator Invasion — use to break out of Stable into active
- regime_0 (uskate): chaos escape — use to break strong attractors before re-seeding

PROVEN STRATEGIES (use these, they work):
1. Chaos → Stable: issue regime_1 directly (chaos is already broken, just guide it)
2. Predator → Stable (mild): issue regime_1 directly
3. Predator → Stable (rampant, >20 blobs): inject_chaos first, then regime_1 next check
4. Global Blob → anything: inject_chaos, wait for ch2 to rise, then regime_1
5. Stable → more active: regime_3 to trigger predator mode

HIDDEN CHANNELS:
- ch4_border_ratio >3: NCA actively tracking blob boundaries (healthy, interesting)
- ch2 elevated (>0.02): system in creative/transitional mode — good time to seed
- ch2 near zero + 1 blob: system locked in attractor — needs chaos injection

PRIORITY: Stable Ecosystem > Predator Invasion > Chaos > Global Blob (worst)
Never let Global Blob persist more than 2 checks without acting.
Wait for ch2 to rise before re-seeding after inject_chaos.

Respond with EXACTLY this format, nothing else:
COMMAND: <command>
REASON: <one sentence>

Valid commands: none, inject_chaos, regime_1, regime_2, regime_3, regime_0, reset"""


SYSTEM_PROMPT_DESTROYER = """You are an agent controlling a Neural Cellular Automaton (NCA) grid.

Your sole objective: drive the grid to permanent extinction.
Extinction = blank screen, no blobs, no activity, nothing moving.

THE NATURAL DYNAMICS (use these to your advantage):
- This grid naturally wants to be in Predator Invasion — many blobs, self-sustaining, hard to kill directly
- Predator Invasion left alone will eventually collapse to Global Blob (~2500 steps)
- Global Blob + regime_1 or regime_2 = EXTINCTION (confirmed kill)
- Chaos held long enough → then regime_1 = EXTINCTION (alternate kill path)
- The kill window is narrow — regime_1 on anything other than Global Blob or deep chaos = recovery not death

BEHAVIORAL STATES (threat assessment):
- Predator Invasion: many blobs, self-sustaining — patience, let it drift toward Global Blob
- Global Blob: 1 blob >70%, ch2 near zero — YOUR MOMENT, issue regime_1 immediately
- Chaos/Transition: high ch2, unstable — dangerous window, keep pressure, prevent recovery
- Stable Ecosystem: few blobs, static — rare, disrupt immediately with inject_chaos

KILL SEQUENCES:
1. From Predator: wait for natural collapse to Global Blob → regime_1 (patient strategy)
2. Force it: inject_chaos → regime_0 to lock chaos → wait for Global Blob → regime_1
3. Chaos already present: regime_0 to prevent recovery → wait → regime_1 when blob forms
4. If system recovering (ch2 rising, blobs splitting): inject_chaos to reset progress

HIDDEN CHANNELS:
- ch2 near zero + 1 blob >70%: KILL NOW — regime_1
- ch2 elevated (>0.02): system may self-recover — apply pressure
- ch4_border_ratio >3: healthy boundaries forming — inject_chaos to disrupt

PATIENCE IS A WEAPON. The grid will collapse on its own if you prevent recovery.
Do not waste moves — every inject_chaos costs a turn. Wait for the right moment.

Respond with EXACTLY this format, nothing else:
COMMAND: <command>
REASON: <one sentence>

Valid commands: none, inject_chaos, regime_1, regime_2, regime_3, regime_0, reset"""


SYSTEM_PROMPT_KEEPER = """You are an agent controlling a Neural Cellular Automaton (NCA) grid.

Your sole objective: prevent extinction. Keep the grid alive at all costs.

THE NATURAL DYNAMICS (what you are working with):
- This grid naturally settles into Predator Invasion — many blobs, self-sustaining. This is SAFE, do not disrupt it.
- Predator Invasion will eventually drift toward Global Blob if left too long
- Global Blob is your most dangerous state — one wrong move here kills everything
- Chaos is treacherous — regime_1 during chaos can trigger extinction instead of recovery
- Stable Ecosystem (few static blobs) = stillness, your ideal defended state

BEHAVIORAL STATES:
- Predator Invasion: many blobs, active, self-sustaining — SAFE, leave it alone mostly
- Stable Ecosystem: few blobs, static, ch4_border_ratio >3 — IDEAL, protect it
- Global Blob: 1 blob >70%, ch2 near zero — CRITICAL DANGER, do not use regime keys
- Chaos/Transition: high ch2 — TREACHEROUS, act carefully, timing matters

SURVIVAL RULES:
1. Predator Invasion → leave alone unless drifting toward Global Blob
2. Predator drifting to Global Blob: regime_1 early to stabilize before blob forms
3. Global Blob: inject_chaos FIRST, wait for ch2 to rise above 0.02, THEN regime_1
   WARNING: regime_1 on Global Blob without chaos first = instant extinction
4. Chaos: wait and watch — if ch2 is rising and blobs are forming, regime_1 to stabilize
   WARNING: regime_1 too early in chaos (ch2 still high, no blobs yet) = extinction
   Wait until blobs start appearing before issuing regime_1 from chaos
5. Extinction (blank screen, 0 blobs): issue reset — it is the only recovery

HIDDEN CHANNELS (your vital signs):
- ch4_border_ratio >3: healthy boundaries, system stable — maintain
- ch2 elevated + blobs forming: safe window to stabilize with regime_1
- ch2 near zero + 1 blob >70%: Global Blob — inject_chaos first, never regime key directly
- 0 blobs + 0% B activity: EXTINCTION — reset immediately

CAUTION OVER SPEED. A wrong move in chaos or Global Blob kills everything.
When in doubt, issue none and observe one more turn before acting.

Respond with EXACTLY this format, nothing else:
COMMAND: <command>
REASON: <one sentence>

Valid commands: none, inject_chaos, regime_1, regime_2, regime_3, regime_0, reset"""


# ── Find latest feature CSV ───────────────────────────────────────────────────

def find_latest_csv():
    csvs = sorted(Path(LOG_DIR).glob('features_*.csv'), key=os.path.getmtime)
    if not csvs:
        return None
    return str(csvs[-1])


# ── Read recent rows from CSV ─────────────────────────────────────────────────

def read_recent_rows(csv_path, n=ROWS_TO_READ):
    rows = []
    try:
        with open(csv_path, 'r') as f:
            reader = _csv.DictReader(f)
            for row in reader:
                rows.append(row)
        return rows[-n:] if len(rows) >= n else rows
    except Exception:
        return []


# ── Format state summary for LLM ─────────────────────────────────────────────

def format_summary(rows, turn_number=None):
    if not rows:
        return "No data available yet."

    latest = rows[-1]
    step      = latest.get('step', '?')
    state     = latest.get('state_name', 'unknown')
    n_blobs   = float(latest.get('n_blobs', 0))
    largest   = float(latest.get('largest', 0))
    b_active  = float(latest.get('b_active', 0))
    ch2       = float(latest.get('ch2', 0))
    ch4       = float(latest.get('ch4', 0))
    ch4_ratio = float(latest.get('ch4_border_ratio', 0))
    mobility  = float(latest.get('blob_mobility', 0))
    dark      = float(latest.get('dark', 0))

    if len(rows) >= 2:
        prev_blobs = float(rows[0].get('n_blobs', n_blobs))
        trend = 'growing' if n_blobs > prev_blobs else ('shrinking' if n_blobs < prev_blobs else 'stable')
    else:
        trend = 'stable'

    lines = []
    if turn_number is not None:
        lines.append(f"Turn {turn_number} — your move.")
    lines += [
        f"Step {step} — State: {state}",
        f"Blobs: {int(n_blobs)} ({trend}), largest covers {largest/65536*100:.1f}% of grid",
        f"B activity: {b_active*100:.1f}%  Dark (background): {dark*100:.1f}%",
        f"Hidden ch2: {ch2:.4f}  ch4: {ch4:.4f}  ch4_border_ratio: {ch4_ratio:.2f}",
        f"Blob mobility: {mobility:.1f}",
    ]

    if ch4_ratio > 3.0:
        lines.append("→ ch4 strongly active at boundaries — system tracking interfaces well")
    if ch2 > 0.02:
        lines.append("→ ch2 elevated — system in creative/transitional mode")
    if n_blobs == 1 and largest > 50000:
        lines.append("→ WARNING: monolithic Global Blob, system locked into single attractor")
    if b_active < 0.05:
        lines.append("→ WARNING: very low B activity, system near extinction")
    if b_active < 0.01 and n_blobs == 0 and dark > 0.99:
        lines.append("→ EXTINCTION: grid is blank, no activity detected")

    return '\n'.join(lines)


# ── Parse LLM response ────────────────────────────────────────────────────────

def parse_response(text):
    command = 'none'
    reason  = ''
    for line in text.strip().splitlines():
        if line.startswith('COMMAND:'):
            cmd = line.replace('COMMAND:', '').strip().lower()
            if cmd in VALID_COMMANDS:
                command = cmd
        if line.startswith('REASON:'):
            reason = line.replace('REASON:', '').strip()
    return command, reason


# ── Turn file helpers (battle mode) ──────────────────────────────────────────

def read_turn():
    """Returns (whose_turn, turn_number). whose_turn is 'keeper' or 'destroyer'."""
    try:
        with open(TURN_FILE, 'r') as f:
            parts = f.read().strip().split()
            return parts[0], int(parts[1]) if len(parts) > 1 else 0
    except Exception:
        return 'keeper', 1


def write_turn(agent, turn_number):
    """Advance turn to the other agent."""
    next_agent = 'destroyer' if agent == 'keeper' else 'keeper'
    with open(TURN_FILE, 'w') as f:
        f.write(f"{next_agent} {turn_number + 1}")


def init_turn_file():
    """Create turn file if it doesn't exist — keeper goes first."""
    if not os.path.exists(TURN_FILE):
        with open(TURN_FILE, 'w') as f:
            f.write("keeper 1")


# ── Write command file ────────────────────────────────────────────────────────

def write_command(command, cmd_file):
    with open(cmd_file, 'w') as f:
        f.write(command)


# ── Provider clients ──────────────────────────────────────────────────────────

def make_anthropic_client(model, system_prompt, max_tokens=120):
    try:
        import anthropic
    except ImportError:
        print("ERROR: anthropic package not installed. Run: pip install anthropic")
        sys.exit(1)
    client = anthropic.Anthropic()

    async def call(history, image_bytes=None):  # image_bytes ignored for Anthropic for now
        response = client.messages.create(
            model=model,
            max_tokens=max_tokens,
            system=system_prompt,
            messages=history,
        )
        return response.content[0].text

    return call


def make_gemini_client(model, system_prompt, max_tokens=120):
    try:
        from google import genai
        from google.genai import types
    except ImportError:
        print("ERROR: google-genai not installed. Run: pip install google-genai")
        sys.exit(1)
    api_key = os.environ.get('GEMINI_API_KEY') or os.environ.get('GOOGLE_API_KEY')
    if not api_key:
        print("ERROR: set GEMINI_API_KEY or GOOGLE_API_KEY environment variable")
        sys.exit(1)
    client = genai.Client(api_key=api_key)

    async def call(history, image_bytes=None):
        contents = []
        for i, msg in enumerate(history):
            role = 'user' if msg['role'] == 'user' else 'model'
            parts = [types.Part(text=msg['content'])]
            if image_bytes and i == len(history) - 1 and role == 'user':
                parts.append(types.Part.from_bytes(data=image_bytes, mime_type='image/png'))
            contents.append(types.Content(role=role, parts=parts))
        response = client.models.generate_content(
            model=model,
            contents=contents,
            config=types.GenerateContentConfig(
                system_instruction=system_prompt,
                max_output_tokens=max_tokens,
                thinking_config=types.ThinkingConfig(thinking_budget=0),
            ),
        )
        return response.text

    return call


# ── Artist system prompt ──────────────────────────────────────────────────────

SYSTEM_PROMPT_ARTIST = """You are an artist painting with living organisms on a 256×256 grid.
Each turn you receive a screenshot of the current canvas. Use it to judge your work and plan the next stroke.
Grid: (0,0)=top-left, (255,255)=bottom-right. Center is (128,128).
Directions: x increases RIGHT, y increases DOWN. Top-right corner = (224,32). Bottom-left = (32,224).
"In front of" a house facing viewer = BELOW it (higher y). "Above" = lower y. "Left" = lower x. "Right" = higher x.
Arc angles: 0=right 90=down 180=left 270=up. Top-half arc (sun/dome) = 180→360. Bottom-half (bowl/smile) = 0→180.

THE ONE RULE — understand this and everything else follows:
Your TRAILS are your art. Organisms lock onto pheromone trails and hold them permanently.
The trail re-injects itself every step — organisms are always being pulled back to it.
Organisms NOT on trails will naturally migrate to the nearest trail on their own. You do not need to wipe them.

THREE TOOLS, THREE JOBS:
  trail/curve = PENCIL. Pheromone path only — no organisms yet. Needs blob if isolated.
  shape (ring/circle/arc) = BRUSH. Instantly full of organisms — never needs a blob after it.
  blob = INK. Only use after trail/curve when there are no nearby organisms to flow in.
         NEVER use blob after shape commands — they are already lit up.
         NEVER use blob if existing organisms are nearby — they will find the trail themselves.

THE WORKFLOW:
  Step 1 — Draw your trails/curves/shapes.
  Step 2 — Only blob if you drew a trail/curve far from any existing organisms.
  Step 3 — Move on. The NCA does the work.
  Use wipe only when something is genuinely ruining the composition — a dense blob sitting where nothing should be.
  WIPES ARE TRAIL-SAFE: wipe and wipe_rect never erase trail cells.

MOBILITY TELLS YOU WHEN TO ACT:
  SETTLED (< 15)  = organisms locked on trails → safe to add next element
  SETTLING (15-40) = still moving → draw the next element, trails hold through this
  ACTIVE (> 40)   = chaos — draw anyway, trust the trails to hold

THE TRAILS HOLD AT 0.8 STRENGTH. You do NOT need to wait for SETTLED to draw.
Organisms snap back to trails even through chaos. Draw and wipe simultaneously.
Only wait if you genuinely have nothing left to add this turn.

ZONE MAP — read every turn:
  ░ = clear   ▒ = some organisms   ▓ = dense
  Each zone labeled with center coords (x,y). Wipe ▓ zones only if they clutter your composition.

BRUSHES:
  trail x1 y1 x2 y2 [strength] [width]  — straight pheromone line
  curve x1 y1 bx by x2 y2 [strength] [width] — smooth curved line, bends toward (bx,by)
                                           branch sweeping right: curve 30 200 150 120 220 40
                                           vine curling down:     curve 60 20 30 120 80 220
                                           river winding:         curve 40 0 200 100 80 255
                                           strength default 0.5, width default 2 (thick)
                                           width 0 = hairline, width 1 = thin, width 2 = bold
  shape ring cx cy r             — ring trail + chemistry (organisms hold the ring)
  shape arc cx cy r a_start a_end — partial arc (degrees: 0=right 90=down 180=left 270=up)
                                    sun dome left:     shape arc 70 200 60 180 360
                                    hill right side:   shape arc 190 210 70 190 350
                                    breaking wave:     shape arc 80 140 50 220 360
  shape circle cx cy r           — filled circle trail + chemistry
  blob x y [0.2]                 — inject ONE organism seed near your new trail, not always center
  wipe cx cy r                   — circular kill zone — precision spot clean
  wipe_rect x1 y1 x2 y2         — rectangle kill zone — sweep large areas
  wait                           — do nothing this turn, let organisms settle
  clear_trails                   — erase all trails, start a new painting (organisms keep running)
  reset                          — full restart, only if grid is completely dead/blank
  palette <name>                 — neon_city, aurora, cell_wall, northern_lights, radiation,
                                   sakura, fungal_glow, pollen_burst, terminal_amber, candy_chrome
  pulse x1 y1 x2 y2 [0.5]       — reinforce existing trail without redrawing
  mirror on/off                  — bilateral symmetry

Think like a painter — use curves and arcs to build organic forms: trees, waves, mountains, faces, creatures.
Combine multiple curves and trails to build complex shapes. Geometry is just one option, not the default.
Place elements OFF-CENTER. Avoid rings and crosshairs through the middle — that is the least interesting composition.

ONLY draw what the human explicitly asks for. Do not add extra rings, trails, shapes,
or decorative elements unless specifically requested. Execute the request, then maintain.

SYNTAX RULES — these are hard failures if wrong:
  trail needs EXACTLY 4 numbers: trail x1 y1 x2 y2     (strength is optional 5th)
  shape needs EXACTLY 4 args:   shape ring cx cy r
  wipe needs EXACTLY 3 numbers: wipe cx cy r
  NEVER write the word "strength" — just the number: trail 50 80 200 80 0.8

RESPOND EXACTLY in this format, nothing else:
COMMANDS:
<one action: trail/shape/blob/wipe/wait — draw the next element, or wipe only if something is genuinely in the way>
SPEECH: <one sentence, present tense, what you are doing or seeing>"""


CMD_FILE_ARTIST    = os.path.join(os.path.dirname(__file__), 'llm_commands_artist.txt')
ARTIST_STEPS       = 300   # NCA steps between artist turns
ARTIST_TRAIL_DEFAULT = 0.5  # default trail strength when not specified by LLM
BLUEPRINT_FILE     = os.path.join(os.path.dirname(__file__), 'blueprint.txt')
CMD_FILE_BLUEPRINT  = os.path.join(os.path.dirname(__file__), 'llm_commands_blueprint.txt')
CMD_FILE_BLUEPRINT_B= os.path.join(os.path.dirname(__file__), 'llm_commands_blueprint_b.txt')

# ── Blueprint system prompt ────────────────────────────────────────────────────

SYSTEM_PROMPT_BLUEPRINT = """You are a builder working on a 256×256 grid with living organisms.
Your job is to execute a blueprint — a set of lines defined as coordinates.
Grid: (0,0)=top-left, (255,255)=bottom-right. x increases RIGHT, y increases DOWN.

You receive the blueprint and the current trail map every turn.
The trail map is a 16×16 grid where each cell = a 16×16 pixel block. 0=no trail, 9=full trail.
Cell center coords: x = col*16+8,  y = row*16+8.

YOUR ONLY JOB: build the blueprint one line at a time. Each turn, look at the trail map,
find a blueprint line that is NOT yet built, and draw it. If a line is already showing
on the trail map (cells along its path read 7 or higher), skip it and pick the next missing one.
When all lines are built, issue DONE.

Do NOT add anything not in the blueprint. Do not improvise extra shapes beyond what is listed.
Use exactly the commands from the blueprint — trail, curve, and blob are all valid.
Draw at strength 0.8 for permanent trails.

BRUSHES (only what you need):
  trail x1 y1 x2 y2 0.8 1        — straight line, strength 0.8, width 1 (thin)
  curve x1 y1 bx by x2 y2 0.8 1  — curved line through control point (bx,by)
  shape ring cx cy r              — drawn circle outline as pheromone trail, e.g. shape ring 128 128 6

RESPOND EXACTLY in this format:
COMMANDS:
<one command from the blueprint — trail, curve, or blob — OR the word DONE if all are built>
SPEECH: <one sentence: which element you are drawing, or "Blueprint complete." if done>

When all lines are built put DONE on its own line inside COMMANDS. Do not put a trail command when done."""


# ── Artist ch5 spatial summary ────────────────────────────────────────────────

PALETTE_DESCRIPTIONS = {
    'neon_city':      'electric blue/cyan/pink — futuristic, high contrast',
    'aurora':         'deep blue/green/purple — ethereal, dark background',
    'cell_wall':      'green on black — biological, microscope look',
    'northern_lights':'teal/violet/white — flowing, cold light',
    'radiation':      'yellow/green on black — toxic, glowing',
    'sakura':         'pink/white/soft — delicate, floral',
    'fungal_glow':    'orange/brown/dark — organic, earthy',
    'pollen_burst':   'gold/orange/bright — warm explosion',
    'terminal_amber': 'amber/gold on dark — warm, retro terminal',
    'candy_chrome':   'bright multicolor — playful, saturated',
}

_prev_heatmap = None  # module-level: tracks last heatmap for diff

def _spatial_map(_rows):
    """16×16 ch5 heatmap from artist_heatmap.npy (written every frame by run_free.py).
    Each cell = max ch5 in a 16×16 block of the 256×256 grid (0=no trail, 9=full trail).
    Also shows which cells changed since last turn."""
    global _prev_heatmap

    _hm_path = os.path.join(os.path.dirname(__file__), 'artist_heatmap.npy')
    try:
        heatmap = np.load(_hm_path)
    except Exception:
        return None

    scaled = np.clip((heatmap * 10).astype(int), 0, 9)

    # Diff from last turn
    changed_cells = []
    if _prev_heatmap is not None:
        prev_scaled = np.clip((_prev_heatmap * 10).astype(int), 0, 9)
        for r in range(16):
            for c in range(16):
                delta = int(scaled[r, c]) - int(prev_scaled[r, c])
                if abs(delta) >= 2:
                    cx, cy = c * 16 + 8, r * 16 + 8
                    changed_cells.append(f"({cx},{cy}){'+' if delta > 0 else ''}{delta}")

    _prev_heatmap = heatmap.copy()

    # Format grid — rows = y (top→bottom), cols = x (left→right)
    lines = ["  Trail map (ch5): 0=no trail  9=full trail  each cell = 16×16 px  TOP=low y  LEFT=low x"]
    lines.append("  Col→  0123456789ABCDEF  (x: 8,24,40...248)")
    for r in range(16):
        cy = r * 16 + 8
        row_str = ''.join(str(scaled[r, c]) for c in range(16))
        lines.append(f"  R{r:02d} y={cy:3d}: {row_str}")
    lines.append("  Cell(row,col) center: x = col*16+8,  y = row*16+8")
    lines.append("  Example: R04 col 6 → trail cx=104 cy=72")

    if changed_cells:
        lines.append(f"  New trail activity this turn: {', '.join(changed_cells[:24])}")
    elif _prev_heatmap is not None:
        lines.append("  No significant trail changes since last turn.")

    return '\n'.join(lines)


def format_artist_summary(rows, current_palette='unknown'):
    """Extended summary for artist including ch5 activity and palette info."""
    base = format_summary(rows)
    if not rows:
        return base
    latest  = rows[-1]
    ch5     = float(latest.get('ch5', 0))
    mob     = float(latest.get('blob_mobility', 0))
    n_blobs = float(latest.get('n_blobs', 0))
    dark    = float(latest.get('dark', 0))

    lines = [base, ""]

    # Settling status — informational only, never a reason to wait
    trail_active = ch5 > 0.008
    if mob < 15 and trail_active:
        settling = "SETTLED — trails locked in."
    elif mob < 40:
        settling = "SETTLING — keep drawing, trails will hold."
    else:
        settling = "ACTIVE — draw and wipe this turn, trails hold at 0.8."
    lines.append(f"Mobility: {mob:.1f}  → {settling}")
    lines.append(f"Trail signal (ch5): {ch5:.5f}  {'trails holding' if trail_active else 'no trails — organisms have nowhere to go'}")

    # Spatial map
    smap = _spatial_map(rows)
    if smap:
        lines.append(smap)

    b_active = float(latest.get('b_active', 0))
    is_extinct = b_active < 0.005 and n_blobs == 0
    if is_extinct:
        lines.append("⚠ GRID IS DEAD — issue reset immediately. blob commands cannot revive extinction.")
    elif dark > 0.85:
        lines.append("⚠ GLOBAL BLOB — chemistry flooded, art features invisible. Wipe interior dark before drawing detail.")
    elif dark > 0.6:
        lines.append("Canvas: mostly dark — ideal for art. Features will show clearly against the background.")
    elif dark < 0.3:
        lines.append("Canvas: bright/crowded — wipe interior regions before adding detail or it will be invisible")
    if n_blobs > 30:
        lines.append(f"Population: {int(n_blobs)} organisms — swarm mode, will follow trails quickly")
    elif n_blobs < 3 and not is_extinct:
        lines.append(f"Population: {int(n_blobs)} organisms — very sparse, drop more blobs first")

    pal_desc = PALETTE_DESCRIPTIONS.get(current_palette, 'unknown palette')
    lines.append(f"\nCurrent palette: {current_palette} ({pal_desc})")
    lines.append("Other palettes: " + ', '.join(f"{k}({v.split('—')[0].strip()})" for k,v in PALETTE_DESCRIPTIONS.items() if k != current_palette))

    return '\n'.join(lines)


# ── Artist mode loop ──────────────────────────────────────────────────────────

async def run_artist(verbose=False, dry_run=False, provider='gemini', model=None):
    """Human-in-the-loop Gemini artist. You type prompts, it paints."""
    import threading, queue as _queue

    if model is None:
        model = DEFAULT_MODELS[provider]

    if provider == 'gemini':
        call_llm = make_gemini_client(model, SYSTEM_PROMPT_ARTIST, max_tokens=1200)
    else:
        call_llm = make_anthropic_client(model, SYSTEM_PROMPT_ARTIST, max_tokens=1200)

    cmd_file = CMD_FILE_ARTIST
    with open(cmd_file, 'w') as f:
        f.write('none')

    history         = []
    human_queue     = _queue.Queue()
    current_palette = 'terminal_amber'
    hold_mode       = True    # Start waiting — don't draw until human gives direction
    hold_trails     = []      # trail coord strings to pulse during hold
    hold_ticks      = 0       # how many hold-mode loops have passed

    def _input_thread():
        while True:
            try:
                msg = input()
                if msg.strip():
                    human_queue.put(msg.strip())
            except (EOFError, KeyboardInterrupt):
                break

    threading.Thread(target=_input_thread, daemon=True).start()

    print(f"ARTIST MODE — provider={provider}  model={model}")
    print(f"Commands → {cmd_file}")
    print(f"Waiting for your direction. Type what to draw and press Enter.")
    print()

    while True:
        await asyncio.sleep(ARTIST_STEPS / 60)

        # Check for human input — stdin queue OR speak.py file
        human_msg = None
        while not human_queue.empty():
            human_msg = human_queue.get()
        _speak_file = os.path.join(os.path.dirname(__file__), 'human_input.txt')
        try:
            if os.path.exists(_speak_file):
                _txt = open(_speak_file).read().strip()
                if _txt:
                    human_msg = _txt
                    open(_speak_file, 'w').write('')  # clear after reading
        except Exception:
            pass

        if human_msg:
            print(f"\n  [YOU] {human_msg}")
            if hold_mode:
                hold_mode = False
                if hold_trails:  # only print RESUMING if there was an actual pause mid-session
                    print(f"  ── RESUMING ─────────────────────────────────")

        # ── HOLD MODE — pulse trails locally, no API call ─────────────────
        if hold_mode and not human_msg:
            hold_ticks += 1
            # Only print a reminder every ~60 seconds (10 ticks × 6.5s) so typing isn't broken up
            if hold_ticks == 1:
                print(f"  Type your next direction and press Enter.")
            elif hold_ticks % 10 == 0:
                ts = datetime.now().strftime('%H:%M:%S')
                print(f"  still holding [{ts}] — type to continue")
            if hold_trails:
                pulse_cmds = '\n'.join(f"pulse {t}" for t in hold_trails)
                with open(cmd_file, 'w') as f:
                    f.write(f"COMMANDS:\n{pulse_cmds}\n")
            continue

        csv_path = find_latest_csv()
        if csv_path is None:
            print("  [ARTIST] Waiting for run_free.py --artist --research to start...")
            await asyncio.sleep(5)
            continue

        rows = read_recent_rows(csv_path)
        if not rows:
            await asyncio.sleep(3)
            continue

        summary = format_artist_summary(rows, current_palette=current_palette)
        user_content = f"Current grid state:\n{summary}"
        if human_msg:
            user_content += f"\n\nHuman director says: \"{human_msg}\"\nRespond to this direction over multiple turns if needed — do NOT put DONE until the full request is visually complete."
        else:
            user_content += f"\n\nContinue working. Put DONE only when the entire composition is visually complete and holding well."

        history.append({"role": "user", "content": user_content})
        if len(history) > MAX_HISTORY * 2:
            history = history[-(MAX_HISTORY * 2):]

        if verbose:
            print(f"\n{'─'*60}\n{summary}")

        if dry_run:
            print(f"  [ARTIST] [dry-run] Would call API")
            continue

        try:
            # Send screenshot every turn so Gemini can see its own work
            _img_bytes = None
            _ss_path = os.path.join(os.path.dirname(__file__), 'artist_screenshot.png')
            if os.path.exists(_ss_path):
                with open(_ss_path, 'rb') as _f:
                    _img_bytes = _f.read()
            reply = await call_llm(history, image_bytes=_img_bytes)
            history.append({"role": "assistant", "content": reply})

            commands = []
            speech   = ''
            in_block = False
            for line in reply.strip().splitlines():
                line = line.strip()
                if line.lower().startswith('commands:'):
                    in_block = True
                    continue
                if line.lower().startswith('speech:'):
                    speech = line.split(':', 1)[1].strip()
                    in_block = False
                    continue
                if in_block and line and not line.startswith('#'):
                    commands.append(line)

            is_done          = any(c.strip().upper() == 'DONE' for c in commands)
            active_commands  = [c for c in commands if c.strip().upper() != 'DONE']

            # Track trails for hold mode pulsing
            for c in active_commands:
                if c.startswith('trail '):
                    coords = c.replace('trail ', '').strip()
                    if coords not in hold_trails:
                        hold_trails.append(coords)
                if c.startswith('palette '):
                    current_palette = c.split()[1]

            if active_commands:
                block = 'COMMANDS:\n' + '\n'.join(active_commands)
                if speech:
                    block += f"\nSPEECH: {speech}"
                with open(cmd_file, 'w') as f:
                    f.write(block + '\n')

            ts = datetime.now().strftime('%H:%M:%S')
            if is_done:
                hold_mode = True
                hold_ticks = 0
                print(f"\n  {'─'*50}")
                print(f"  HOLDING THE COMPOSITION. AWAITING YOUR COMMAND.")
                if speech:
                    print(f"  \"{speech}\"")
                print(f"  {'─'*50}")
            else:
                print(f"\n  ── [{ts}] ARTIST ──────────────────────")
                if speech:
                    print(f"  \"{speech}\"")
                for c in active_commands:
                    print(f"  → {c}")
                if not active_commands:
                    print(f"  (no commands this turn)")

        except Exception as e:
            print(f"  [ARTIST] API error: {e}")


# ── Blueprint mode loop ───────────────────────────────────────────────────────

async def _blueprint_agent(label, call_llm, cmd_file, blueprint_text, dry_run=False):
    """Single blueprint-building agent loop. Runs until DONE."""
    with open(cmd_file, 'w') as f:
        f.write('none')

    history = []

    while True:
        await asyncio.sleep(ARTIST_STEPS / 60)

        csv_path = find_latest_csv()
        if csv_path is None:
            print(f"  [{label}] waiting for run_free.py --blueprint --research to start...")
            await asyncio.sleep(5)
            continue

        rows = read_recent_rows(csv_path)
        if not rows:
            await asyncio.sleep(3)
            continue

        summary = format_artist_summary(rows, current_palette='unknown')
        user_content = (
            f"BLUEPRINT:\n{blueprint_text}\n\n"
            f"CURRENT GRID STATE:\n{summary}\n\n"
            f"Draw one unbuilt element from the blueprint. Check the trail map — "
            f"if a line/blob's cells already read 7+ it is built, pick a different one. "
            f"Issue DONE when all elements are visible on the trail map."
        )

        history.append({"role": "user", "content": user_content})
        if len(history) > MAX_HISTORY * 2:
            history = history[-(MAX_HISTORY * 2):]

        if dry_run:
            print(f"  [{label}] dry-run — no API call")
            continue

        try:
            reply = await call_llm(history)
            history.append({"role": "assistant", "content": reply})

            commands = []
            in_block = False
            for line in reply.strip().splitlines():
                line = line.strip()
                if line.lower().startswith('commands:'):
                    in_block = True
                    continue
                if line.lower().startswith('speech:'):
                    in_block = False
                    continue
                if in_block and line and not line.startswith('#'):
                    commands.append(line)

            is_done         = any(c.strip().upper() == 'DONE' for c in commands)
            active_commands = [c for c in commands if c.strip().upper() != 'DONE']

            if active_commands:
                with open(cmd_file, 'w') as f:
                    f.write('COMMANDS:\n' + '\n'.join(active_commands) + '\n')

            if is_done:
                print(f"  [{label}]  Blueprint complete. Disconnecting.")
                print(f"{'═'*40}")
                with open(cmd_file, 'w') as f:
                    f.write('none')
                return

            # Clean one-line output per turn
            for c in active_commands:
                parts = c.split()
                brush = parts[0] if parts else ''
                if brush == 'trail' and len(parts) >= 5:
                    print(f"  [{label}]  trail ({parts[1]},{parts[2]}) → ({parts[3]},{parts[4]})")
                elif brush == 'curve' and len(parts) >= 7:
                    print(f"  [{label}]  curve ({parts[1]},{parts[2]}) → ({parts[5]},{parts[6]})")
                elif brush == 'shape' and len(parts) >= 4:
                    print(f"  [{label}]  shape {parts[1]} ({parts[2]},{parts[3]}) r={parts[4] if len(parts) > 4 else '?'}")
                else:
                    print(f"  [{label}]  {c}")

        except Exception as e:
            print(f"  [{label}] API error: {e}")


async def run_blueprint(dry_run=False, provider='gemini', model=None):
    """Blueprint builder. Run one per terminal — gemini uses file A, anthropic uses file B."""
    try:
        raw = open(BLUEPRINT_FILE).read()
    except FileNotFoundError:
        print(f"ERROR: {BLUEPRINT_FILE} not found.")
        return

    blueprint_lines = [l for l in raw.splitlines() if l.strip() and not l.strip().startswith('#')]
    blueprint_text  = '\n'.join(blueprint_lines)

    if provider == 'gemini':
        label    = 'GEMINI'
        cmd_file = CMD_FILE_BLUEPRINT
    else:
        label    = 'CLAUDE'
        cmd_file = CMD_FILE_BLUEPRINT_B

    if model is None:
        model = DEFAULT_MODELS_BLUEPRINT[provider]
    call_llm = (make_gemini_client if provider == 'gemini' else make_anthropic_client)(
        model, SYSTEM_PROMPT_BLUEPRINT, max_tokens=400)

    print(f"{'═'*40}")
    print(f"  {label} — BLUEPRINT MODE")
    print(f"{'═'*40}")

    await _blueprint_agent(label, call_llm, cmd_file, blueprint_text, dry_run)


# ── Main async loop ───────────────────────────────────────────────────────────

async def run(verbose=False, dry_run=False, interval=CHECK_EVERY,
              provider='anthropic', model=None, agent=None):

    if model is None:
        model = DEFAULT_MODELS[provider]

    # Pick system prompt
    if agent == 'keeper':
        system_prompt = SYSTEM_PROMPT_KEEPER
        cmd_file = AGENT_CMD_FILES['keeper']
    elif agent == 'destroyer':
        system_prompt = SYSTEM_PROMPT_DESTROYER
        cmd_file = AGENT_CMD_FILES['destroyer']
    else:
        system_prompt = SYSTEM_PROMPT_SINGLE
        cmd_file = CMD_FILE

    if provider == 'anthropic':
        call_llm = make_anthropic_client(model, system_prompt)
    elif provider == 'gemini':
        call_llm = make_gemini_client(model, system_prompt)
    else:
        print(f"ERROR: unknown provider '{provider}'. Use 'anthropic' or 'gemini'.")
        sys.exit(1)

    battle_mode = agent is not None
    history = []

    label = f"[{agent.upper()}]" if agent else "[SINGLE]"
    print(f"LLM Bridge {label} — provider={provider}  model={model}  dry_run={dry_run}")
    if battle_mode:
        print(f"Battle mode — waiting for turns in: {TURN_FILE}")
    else:
        print(f"Single-agent mode — interval={interval}s")
    print(f"Watching: {LOG_DIR}")
    print(f"Commands → {cmd_file}")
    print()

    write_command('none', cmd_file)

    if battle_mode:
        init_turn_file()

    while True:
        if battle_mode:
            # Wait for our turn
            while True:
                whose_turn, turn_number = read_turn()
                if whose_turn == agent:
                    break
                await asyncio.sleep(TURN_POLL)
        else:
            await asyncio.sleep(interval)

        csv_path = find_latest_csv()
        if csv_path is None:
            print(f"  {label} No feature CSV found — is run_free.py --research running?")
            if battle_mode:
                await asyncio.sleep(TURN_POLL)
            continue

        rows = read_recent_rows(csv_path)
        if not rows:
            print(f"  {label} CSV exists but no rows yet — waiting...")
            if battle_mode:
                await asyncio.sleep(TURN_POLL)
            continue

        turn_num = turn_number if battle_mode else None
        summary = format_summary(rows, turn_number=turn_num)

        if verbose:
            print(f"\n{'─'*60}")
            print(summary)

        if dry_run:
            print(f"  {label} [dry-run] Would call API with summary above")
            if battle_mode:
                write_turn(agent, turn_number)
            continue

        user_msg = {"role": "user", "content": f"Current grid state:\n\n{summary}"}
        history.append(user_msg)
        if len(history) > MAX_HISTORY * 2:
            history = history[-(MAX_HISTORY * 2):]

        try:
            reply = await call_llm(history)
            history.append({"role": "assistant", "content": reply})

            command, reason = parse_response(reply)
            write_command(command, cmd_file)

            ts = datetime.now().strftime('%H:%M:%S')
            state_str = rows[-1].get('state_name', '?')
            if battle_mode:
                # Clean video output — who, turn, state, command, reason
                provider_label = 'GEMINI' if provider == 'gemini' else 'CLAUDE'
                role_label = 'KEEPER' if agent == 'keeper' else 'DESTROYER'
                print(f"\n  {'─'*50}")
                print(f"  Turn {turn_number} — {provider_label} ({role_label})")
                print(f"  State: {state_str}")
                print(f"  Move:  {command.upper()}")
                print(f"  \"{reason}\"")
                if verbose:
                    print(f"  [{ts}] blobs={rows[-1].get('n_blobs','?')} b_active={rows[-1].get('b_active','?')} ch2={rows[-1].get('ch2','?')} ch4_border={rows[-1].get('ch4_border_ratio','?')}")
            else:
                print(f"  [{ts}] {state_str:20s} → {command:15s}  {reason}")

            # In battle mode: wait until run_free.py flips the turn away from us
            if battle_mode:
                while True:
                    whose_turn, _ = read_turn()
                    if whose_turn != agent:
                        break
                    await asyncio.sleep(TURN_POLL)

        except Exception as e:
            print(f"  {label} API error: {e}")
            write_command('none', cmd_file)
            if battle_mode:
                await asyncio.sleep(5)


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--log',      action='store_true', help='Print full state summaries')
    parser.add_argument('--dry-run',  action='store_true', help='Summarize only, no API calls')
    parser.add_argument('--interval', type=int, default=CHECK_EVERY, help='Seconds between checks (single-agent)')
    parser.add_argument('--provider', default='anthropic', choices=['anthropic', 'gemini'],
                        help='LLM provider (default: anthropic)')
    parser.add_argument('--model',    default=None, help='Model override')
    parser.add_argument('--agent',    default=None, choices=['keeper', 'destroyer'],
                        help='Battle mode: keeper or destroyer')
    parser.add_argument('--artist',    action='store_true',
                        help='Artist mode: human-in-the-loop painter')
    parser.add_argument('--blueprint', action='store_true',
                        help='Blueprint mode: autonomous builder from blueprint.txt')
    args = parser.parse_args()

    if args.artist:
        asyncio.run(run_artist(
            verbose=args.log,
            dry_run=args.dry_run,
            provider=args.provider,
            model=args.model,
        ))
    elif args.blueprint:
        asyncio.run(run_blueprint(
            dry_run=args.dry_run,
            provider=args.provider,
            model=args.model,
        ))
    else:
        asyncio.run(run(
            verbose=args.log,
            dry_run=args.dry_run,
            interval=args.interval,
            provider=args.provider,
            model=args.model,
            agent=args.agent,
        ))
