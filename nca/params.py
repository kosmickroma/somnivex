# nca/params.py — parameter catalog + preference-weighted random sampling.
#
# HOW THE LEARNING WORKS:
#   Every U/D rating saves to ratings_log.json with the current era's category,
#   palette, etc. load_prefs() reads that file and tallies scores. sample_params()
#   uses those scores to weight the random choices — liked categories/palettes get
#   picked more often. Ships as a blank slate (equal weights), adapts over time.

import os
import json
import numpy as np
from dataclasses import dataclass
from jax import random
import jax.numpy as jnp

from nca.model import N_CHANNELS, N_FILTERS, HIDDEN_SIZE, UpdateNet
from config import GRID_H, GRID_W, MIN_RATINGS_TO_LEARN


# ── Catalogs ──────────────────────────────────────────────────────────────────

SUBJECTS = {
    "fluid":     ["jellyfish", "lava_flow", "aurora", "oil_slick", "smoke", "clouds"],
    "geometric": ["crystal", "circuit", "honeycomb", "lattice", "fracture", "shatter"],
    "organic":   ["coral", "lichen", "mycelium", "moss", "root_system"],
    "chaotic":   ["plasma", "static", "interference", "storm", "fire"],
}

AESTHETICS = [
    "raw", "chrome", "watercolor", "neon_glow", "ink_wash", "glitch",
    "pixel_art", "cyberpunk", "vaporwave", "steampunk", "bioluminescent", "prismatic",
]

PALETTES = {
    # Original 8
    "deep_ocean":      [(0,10,40),    (0,40,80),    (0,100,160),  (100,200,255)],
    "candy_chrome":    [(255,50,150), (200,0,255),  (50,200,255), (255,255,255)],
    "forest_floor":    [(20,40,10),   (60,90,20),   (120,160,40), (200,220,100)],
    "neon_city":       [(10,0,30),    (180,0,255),  (0,255,200),  (255,220,0)],
    "sunset_fire":     [(20,0,0),     (180,30,0),   (255,120,0),  (255,220,100)],
    "cosmic":          [(5,0,20),     (60,0,100),   (150,50,200), (220,180,255)],
    "monochrome":      [(10,10,10),   (60,60,60),   (150,150,150),(240,240,240)],
    "acid":            [(0,40,0),     (0,180,0),    (100,255,0),  (220,255,100)],
    # Previous additions
    "blood_moon":      [(10,0,0),     (80,0,0),     (200,20,0),   (255,80,20)],
    "void":            [(0,0,0),      (10,0,30),    (0,0,80),     (100,0,255)],
    "toxic":           [(10,20,0),    (40,100,0),   (150,220,0),  (220,255,80)],
    "rust_decay":      [(20,8,0),     (80,30,5),    (160,80,20),  (200,140,60)],
    "holographic":     [(200,180,255),(180,240,255),(255,200,240),(240,255,200)],
    "deep_bio":        [(0,0,10),     (0,30,40),    (0,180,120),  (100,255,220)],
    "inferno":         [(0,0,0),      (100,0,50),   (255,50,0),   (255,220,100)],
    "ice_cave":        [(0,10,30),    (20,80,120),  (100,180,220),(220,240,255)],
    # Warm / fire
    "amber_ember":     [(10,5,0),     (100,40,0),   (220,120,0),  (255,200,80)],
    "molten_gold":     [(20,10,0),    (120,60,0),   (220,160,20), (255,240,100)],
    "solar_flare":     [(10,2,0),     (200,50,0),   (255,180,0),  (255,255,200)],
    "magma_ocean":     [(5,0,0),      (60,10,0),    (180,60,0),   (255,160,50)],
    "deep_crimson":    [(5,0,0),      (60,0,10),    (160,0,30),   (255,50,80)],
    # Cool / ice / metal
    "northern_lights": [(0,10,10),    (0,80,60),    (0,200,120),  (180,255,200)],
    "ghost":           [(5,5,10),     (40,50,80),   (150,170,200),(240,245,255)],
    "abyssal":         [(0,0,5),      (0,10,20),    (0,60,80),    (0,200,180)],
    "titanium":        [(10,12,15),   (50,60,70),   (120,140,160),(210,220,230)],
    "storm_grey":      [(5,5,8),      (30,35,50),   (90,100,120), (180,190,210)],
    "bone_dust":       [(20,18,15),   (90,80,65),   (170,155,130),(240,235,220)],
    # Psychedelic / electric
    "radiation":       [(0,5,0),      (0,60,10),    (100,200,0),  (220,255,50)],
    "acid_wash":       [(5,0,20),     (80,0,100),   (200,50,200), (255,150,255)],
    "uv_rave":         [(5,0,10),     (60,0,150),   (0,200,255),  (200,0,255)],
    "oil_slick":       [(10,0,20),    (0,100,120),  (150,50,200), (255,200,100)],
    "void_bloom":      [(0,0,0),      (20,0,40),    (80,0,120),   (255,100,255)],
    # Nature / organic
    "deep_jungle":     [(0,10,5),     (10,60,15),   (40,130,30),  (120,200,60)],
    "neon_moss":       [(0,10,5),     (10,80,20),   (50,200,50),  (200,255,100)],
    "copper_verdigris":[(40,20,5),    (120,80,30),  (80,160,100), (180,220,160)],
    "sakura":          [(40,10,20),   (180,80,120), (255,160,180),(255,220,230)],
    "sunset_lavender": [(20,0,30),    (120,40,100), (200,100,180),(255,200,240)],
}

MOODS = [
    "calm", "tense", "mysterious", "playful", "melancholy", "energetic",
    "ominous", "whimsical", "serene", "chaotic", "dreamy", "eerie",
    "hopeful", "nostalgic", "futuristic", "ancient", "alien",
    "underwater", "celestial", "urban", "natural",
]


# ── Params container ──────────────────────────────────────────────────────────
@dataclass
class NCAParams:
    """
    Describes one era of the living world.
    Saved with each rating so we know exactly what the user liked/disliked.
    When the world drifts, a new NCAParams is created to label the new era.
    """
    subject:      str     # e.g. "jellyfish"
    category:     str     # e.g. "fluid"
    aesthetic:    str     # e.g. "chrome"
    palette_name: str     # e.g. "deep_ocean"
    mood:         str     # e.g. "eerie"
    palette:      list    # list of 4 RGB tuples — saved to JSON for reproducibility
    weight_scale: float   # controls how active/explosive the NCA is
    seed:         int     # random seed — lets you reproduce any era


# ── Preference loader ─────────────────────────────────────────────────────────
def load_prefs(ratings_file):
    """
    Read ratings_log.json and compute preference scores.

    Returns a dict like:
      {'categories': {'fluid': 3, 'chaotic': -1}, 'palettes': {'deep_ocean': 2}}
    or None if not enough ratings yet (uses equal weights below MIN_RATINGS_TO_LEARN).

    Score = sum of (+1 for like, -1 for dislike) per category/palette.
    Positive score = user tends to like this → picked more often.
    Negative score = user tends to dislike this → picked less often.
    """
    if not os.path.exists(ratings_file):
        return None
    try:
        with open(ratings_file) as f:
            ratings = json.load(f)
    except Exception:
        return None

    if len(ratings) < MIN_RATINGS_TO_LEARN:
        return None   # blank slate — not enough data yet

    cat_scores = {}
    pal_scores = {}
    for r in ratings:
        score = 1 if r['liked'] else -1
        c = r.get('category', '')
        p = r.get('palette', '')
        if c:
            cat_scores[c] = cat_scores.get(c, 0) + score
        if p:
            pal_scores[p] = pal_scores.get(p, 0) + score

    return {'categories': cat_scores, 'palettes': pal_scores}


def _weighted_choice(rng, options, scores):
    """
    Pick from options[] using scores dict as weights.
    Missing keys get 0. All weights shifted positive before normalizing.
    This is the core of the preference system — scored options get
    proportionally more probability mass.
    """
    vals = np.array([scores.get(o, 0) for o in options], dtype=float)
    vals = vals - vals.min() + 1.0   # shift so minimum weight is 1 (never zero)
    probs = vals / vals.sum()
    return rng.choice(options, p=probs)


# ── Random parameter draw ─────────────────────────────────────────────────────
def sample_params(prefs=None, seed=None):
    """
    Roll the slot machine. Returns one NCAParams describing the new era.

    prefs: output of load_prefs() — if provided, biases choices toward liked options.
           None = equal weights (blank slate).

    All randomness flows from a single integer seed → fully reproducible.
    """
    if seed is None:
        seed = int(np.random.randint(0, 2**31))

    rng = np.random.default_rng(seed)

    categories = list(SUBJECTS.keys())
    palettes   = list(PALETTES.keys())

    if prefs is not None:
        # Preference-biased: liked categories/palettes get higher probability
        category     = _weighted_choice(rng, categories, prefs['categories'])
        palette_name = _weighted_choice(rng, palettes,   prefs['palettes'])
    else:
        # Blank slate: pure random
        category     = rng.choice(categories)
        palette_name = rng.choice(palettes)

    subject   = rng.choice(SUBJECTS[category])
    aesthetic = rng.choice(AESTHETICS)
    palette   = PALETTES[palette_name]
    mood      = rng.choice(MOODS)

    # weight_scale: how strongly the net fires. Too low = nothing. Too high = explosion.
    # Chaotic category likes higher values — more conflict, more oscillation.
    scale_ranges = {
        "fluid":     (0.3, 0.8),
        "geometric": (0.4, 1.0),
        "organic":   (0.3, 0.7),
        "chaotic":   (0.6, 1.4),
    }
    lo, hi       = scale_ranges[category]
    weight_scale = float(rng.uniform(lo, hi))

    return NCAParams(
        subject=subject, category=category, aesthetic=aesthetic,
        palette_name=palette_name, mood=mood, palette=palette,
        weight_scale=weight_scale, seed=seed,
    )


# ── Weight initialization ─────────────────────────────────────────────────────
def init_net_params(nca_params, jax_key):
    """
    Initialize UpdateNet weights scaled by nca_params.weight_scale.
    Dense_0 (hidden layer) gets full scale. Dense_1 (output) gets 20% —
    big output weights cause instant explosion on the first few frames.
    """
    net   = UpdateNet()
    dummy = jnp.zeros((GRID_H, GRID_W, N_CHANNELS * N_FILTERS))

    jax_key, subkey = random.split(jax_key)
    params = net.init(subkey, dummy)

    k0 = params['params']['Dense_0']['kernel']
    k1 = params['params']['Dense_1']['kernel']
    params['params']['Dense_0']['kernel'] = k0 * nca_params.weight_scale
    params['params']['Dense_1']['kernel'] = k1 * (nca_params.weight_scale * 0.08)
    # tanh saturates differently than relu — output layer needs a much smaller
    # scale or the delta per step is too large and the grid locks into rectangles
    # immediately. 0.08 instead of 0.2 gives it room to evolve gradually.

    return params, jax_key
