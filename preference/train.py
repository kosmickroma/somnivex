# preference/train.py — training loop + inference for the preference model.
#
# TRAINING:
#   Called after every PREF_RETRAIN_EVERY new ratings.
#   Reads ratings_log.json, encodes all entries, trains the MLP from scratch
#   (small enough that retraining from scratch takes ~50ms — no need for
#   incremental/online learning complexity).
#   Saves weights to preference_weights.pkl.
#
# INFERENCE:
#   At each drift event, we generate N random candidate eras and score each
#   through the trained model. The scores become sampling probabilities
#   (softmax with temperature) — higher predicted preference = more likely
#   to be picked, but NOT always the top pick. Temperature keeps variety.
#
# BLANK SLATE:
#   If no weights file exists (first run, or < MIN_RATINGS), predict() returns
#   0.5 for everything (neutral). Equal weights = random behavior.
#   The system gracefully degrades to pure random until enough data exists.

import os
import json
import pickle
import numpy as np
import jax
import jax.numpy as jnp
import optax
from flax.training import train_state

from preference.model    import PreferenceMLP
from preference.features import encode_rating, encode_candidate, FEATURE_DIM

WEIGHTS_FILE    = os.path.join(os.path.dirname(__file__), '..', 'preference_weights.pkl')
MIN_RATINGS     = 10     # don't train until we have at least this many
N_EPOCHS        = 400    # training epochs — fast at this scale (~50ms total)
LEARNING_RATE   = 0.005
TEMPERATURE     = 0.4    # softmax temperature for candidate selection
                         # lower = more deterministic (exploits known preferences)
                         # higher = more random (explores new combinations)
                         # 0.4 is a good balance — clearly influenced but not repetitive


def train(ratings_file):
    """
    Train the preference model from scratch on all accumulated ratings.
    Returns trained params dict, or None if not enough data yet.
    Saves weights to WEIGHTS_FILE as a side effect.
    """
    if not os.path.exists(ratings_file):
        return None

    try:
        with open(ratings_file) as f:
            ratings = json.load(f)
    except Exception:
        return None

    if len(ratings) < MIN_RATINGS:
        return None

    # Encode all ratings into feature matrix and label vector
    X = np.array([encode_rating(r) for r in ratings], dtype=np.float32)
    y = np.array([1.0 if r['liked'] else 0.0 for r in ratings], dtype=np.float32)

    # Initialize model
    model  = PreferenceMLP()
    key    = jax.random.PRNGKey(42)
    params = model.init(key, jnp.zeros((1, FEATURE_DIM)))

    tx    = optax.adam(LEARNING_RATE)
    state = train_state.TrainState.create(
        apply_fn=model.apply, params=params, tx=tx
    )

    X_j = jnp.array(X)
    y_j = jnp.array(y)

    # JIT the training step — fast even for first call since model is tiny
    @jax.jit
    def train_step(state, X, y):
        def loss_fn(params):
            logits = model.apply(params, X)
            # Binary cross-entropy: penalizes confident wrong predictions
            return optax.sigmoid_binary_cross_entropy(logits, y).mean()
        loss, grads = jax.value_and_grad(loss_fn)(state.params)
        return state.apply_gradients(grads=grads), loss

    for _ in range(N_EPOCHS):
        state, loss = train_step(state, X_j, y_j)

    # Save weights
    with open(WEIGHTS_FILE, 'wb') as f:
        pickle.dump(state.params, f)

    n_liked    = int(y.sum())
    n_disliked = len(y) - n_liked
    print(f"  Preference model trained: {len(ratings)} ratings "
          f"({n_liked} liked / {n_disliked} disliked), loss={float(loss):.4f}")
    return state.params


def load_weights():
    """Load saved preference model weights. Returns None if no file yet."""
    if not os.path.exists(WEIGHTS_FILE):
        return None
    try:
        with open(WEIGHTS_FILE, 'rb') as f:
            return pickle.load(f)
    except Exception:
        return None


def predict(params, features):
    """
    Predict probability [0-1] that the user will like an era with these features.
    Returns 0.5 (neutral) if no model available yet.
    """
    if params is None:
        return 0.5
    model  = PreferenceMLP()
    logit  = model.apply(params, jnp.array(features)[None])
    return float(jax.nn.sigmoid(logit)[0])


def pick_best_candidate(pref_params, candidates):
    """
    Given a list of (regime_name, f, k, pal_name, pal_arr, era_params) tuples,
    score each with the preference model and return one sampled proportionally
    to predicted preference (softmax with TEMPERATURE).

    WHY NOT ALWAYS PICK THE HIGHEST SCORE?
    Pure greedy exploitation collapses variety — the screensaver would keep
    showing the same few combos it thinks you like. Temperature sampling
    means liked combos are more likely but not guaranteed. Keeps it fresh
    while still trending toward your taste.
    """
    if pref_params is None or len(candidates) == 0:
        return candidates[np.random.randint(len(candidates))]

    scores = []
    for regime_name, f, k, pal_name, pal_arr, era in candidates:
        feat  = encode_candidate(era, regime_name, f, k)
        score = predict(pref_params, feat)
        scores.append(score)

    # Softmax with temperature
    scores = np.array(scores)
    exp_s  = np.exp((scores - scores.max()) / TEMPERATURE)
    probs  = exp_s / exp_s.sum()
    idx    = np.random.choice(len(candidates), p=probs)
    return candidates[idx]
