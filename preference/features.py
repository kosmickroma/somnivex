# preference/features.py — encode an era into a feature vector.
#
# The model needs numbers, not strings. This file converts a rated era
# (regime name, palette name, category, f, k) into a float32 array
# the MLP can train on and predict from.
#
# We use one-hot encoding for categorical variables (regime, palette, category)
# and min-max normalized floats for continuous variables (f, k).
#
# One-hot means: for a list of N options, make a vector of N zeros,
# then put a 1 in the position of the actual choice.
# e.g. palette "acid" in a list of 16 palettes → [0,1,0,0,...,0]
# The model learns which positions (which choices) correlate with likes.

import numpy as np
from gs.engine  import GS_REGIMES
from nca.params import PALETTES

# Sorted so the encoding is stable — same order every run
REGIME_KEYS   = sorted(GS_REGIMES.keys())
PALETTE_KEYS  = sorted(PALETTES.keys())
CATEGORY_KEYS = sorted(['chaotic', 'fluid', 'geometric', 'organic'])

# F and K ranges across all regimes — used for normalization to [0, 1]
F_MIN, F_MAX = 0.010, 0.095
K_MIN, K_MAX = 0.040, 0.070

# Total feature vector size — used to build dummy input for model init
FEATURE_DIM = len(REGIME_KEYS) + len(PALETTE_KEYS) + len(CATEGORY_KEYS) + 2


def _one_hot(val, keys):
    """Return a list of floats: 1.0 at the position of val, 0.0 elsewhere."""
    return [1.0 if val == k else 0.0 for k in keys]


def encode_rating(r):
    """
    Encode a saved rating dict (from ratings_log.json) into a feature vector.
    Handles missing fields gracefully — old ratings without 'regime'/'f'/'k'
    get zeros for those dimensions (neutral, doesn't bias the model).
    """
    regime_oh   = _one_hot(r.get('regime',   ''), REGIME_KEYS)
    palette_oh  = _one_hot(r.get('palette',  ''), PALETTE_KEYS)
    category_oh = _one_hot(r.get('category', ''), CATEGORY_KEYS)

    f = r.get('f', (F_MIN + F_MAX) / 2)   # default to midpoint if missing
    k = r.get('k', (K_MIN + K_MAX) / 2)
    f_norm = (f - F_MIN) / (F_MAX - F_MIN)
    k_norm = (k - K_MIN) / (K_MAX - K_MIN)

    return np.array(regime_oh + palette_oh + category_oh + [f_norm, k_norm],
                    dtype=np.float32)


def encode_candidate(era_params, regime_name, f, k):
    """
    Encode a candidate era (not yet rated) for preference prediction.
    Same format as encode_rating so the model can score it.
    """
    return encode_rating({
        'regime':   regime_name,
        'palette':  era_params.palette_name,
        'category': era_params.category,
        'f': f,
        'k': k,
    })
