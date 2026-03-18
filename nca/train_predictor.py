#!/usr/bin/env python3
# nca/train_predictor.py — Transition predictor for NCA behavioral states
#
# Loads all feature CSVs + intervention CSVs from nca/logs/,
# joins interventions to the feature row ~800 steps later,
# trains a small MLP: (feature_vector + action_onehot) → P(next_state)
#
# Usage:
#   python nca/train_predictor.py              # train and evaluate
#   python nca/train_predictor.py --save       # also save the model
#
# Output: confusion matrix, per-class accuracy, ECE, Brier score

import os
import sys
import glob
import argparse
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report

# ── Constants ──────────────────────────────────────────────────────────────────
LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'logs')
PREDICTOR_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'transition_predictor.pkl')

N_STATES = 8
LOOKAHEAD = 800   # steps ahead to predict
LOOKAHEAD_TOL = 400  # ± tolerance when matching feature rows

FEATURE_COLS = [
    'bg', 'dark', 'b_active', 'a_std', 'b_max',
    'ch2', 'ch4', 'ch5', 'n_blobs', 'largest', 'second',
    'size_std', 'left_dark', 'right_dark', 'asym', 'corr_ch4_b',
    'ch4_border_ratio', 'ch2_border_ratio', 'blob_mobility', 'suppression_zone_frac'
]

# Map key strings to action index
ACTION_KEYS = ['H', 'F', 'T', 'X', 'Z', 'C', 'V_CLICK',
               'KEY_mitosis', 'KEY_gliders', 'KEY_predator', 'KEY_uskate',
               'KEY_orbium', 'KEY_worms', 'KEY_solitons', 'KEY_mazes', 'KEY_uskate2',
               'OTHER']
N_ACTIONS = len(ACTION_KEYS)


def _action_idx(key_str):
    for i, k in enumerate(ACTION_KEYS):
        if k in str(key_str):
            return i
    return ACTION_KEYS.index('OTHER')


def load_data():
    """Load all feature + intervention CSVs, join them, return X, y arrays."""
    feat_files = sorted(glob.glob(os.path.join(LOG_DIR, 'features_*.csv')))
    int_files  = sorted(glob.glob(os.path.join(LOG_DIR, 'interventions_*.csv')))

    if not feat_files:
        print(f"No feature CSVs found in {LOG_DIR}")
        sys.exit(1)

    print(f"Found {len(feat_files)} feature files, {len(int_files)} intervention files")

    # Load and concat feature files — handle old (16-col) and new (20-col) formats
    feat_dfs = []
    for f in feat_files:
        df = pd.read_csv(f)
        # Backfill missing new columns with 1.0 / 0.0 defaults
        for col, default in [('ch4_border_ratio', 1.0), ('ch2_border_ratio', 1.0),
                              ('blob_mobility', 0.0), ('suppression_zone_frac', 0.0)]:
            if col not in df.columns:
                df[col] = default
        feat_dfs.append(df)
    features = pd.concat(feat_dfs, ignore_index=True)
    features['step'] = features['step'].astype(int)
    features = features.sort_values('step').reset_index(drop=True)
    print(f"Total feature rows: {len(features)}")

    # If no intervention files, fall back to self-transitions (state_t → state_t+800)
    if not int_files:
        print("No intervention files found — building self-transition dataset")
        return _build_no_action_dataset(features)

    int_dfs = []
    for f in int_files:
        df = pd.read_csv(f)
        int_dfs.append(df)
    interventions = pd.concat(int_dfs, ignore_index=True)
    interventions['step'] = interventions['step'].astype(int)
    print(f"Total intervention rows: {len(interventions)}")

    return _build_action_dataset(features, interventions)


def _build_no_action_dataset(features):
    """No intervention data — predict next state from current features alone."""
    X_rows, y_rows = [], []
    for i, row in features.iterrows():
        target_step = row['step'] + LOOKAHEAD
        # Find nearest feature row at target_step
        diffs = np.abs(features['step'].values - target_step)
        j = np.argmin(diffs)
        if diffs[j] > LOOKAHEAD_TOL:
            continue
        feat_vec = [row[c] for c in FEATURE_COLS if c in features.columns]
        # Pad missing features
        while len(feat_vec) < len(FEATURE_COLS):
            feat_vec.append(0.0)
        # No action → zero vector
        action_vec = [0.0] * N_ACTIONS
        X_rows.append(feat_vec + action_vec)
        y_rows.append(int(features.iloc[j]['state_id']))

    X = np.array(X_rows, dtype=np.float32)
    y = np.array(y_rows, dtype=np.int32)
    print(f"Built {len(X)} no-action training pairs")
    return X, y


def _build_action_dataset(features, interventions):
    """Join each intervention to the feature row at t and at t+LOOKAHEAD."""
    X_rows, y_rows = [], []
    feat_steps = features['step'].values

    for _, iv in interventions.iterrows():
        iv_step = int(iv['step'])
        key_str = str(iv.get('key', 'OTHER'))

        # Feature row at time of intervention (nearest within 300 steps before)
        pre_diffs = feat_steps - iv_step
        valid_pre = np.where((pre_diffs >= -300) & (pre_diffs <= 0))[0]
        if len(valid_pre) == 0:
            continue
        pre_idx = valid_pre[np.argmax(pre_diffs[valid_pre])]  # closest before

        # Feature row at t + LOOKAHEAD
        target_step = iv_step + LOOKAHEAD
        post_diffs = np.abs(feat_steps - target_step)
        post_idx = np.argmin(post_diffs)
        if post_diffs[post_idx] > LOOKAHEAD_TOL:
            continue

        pre_row = features.iloc[pre_idx]
        post_row = features.iloc[post_idx]

        feat_vec = [float(pre_row.get(c, 0.0)) for c in FEATURE_COLS]
        action_vec = [0.0] * N_ACTIONS
        action_vec[_action_idx(key_str)] = 1.0

        X_rows.append(feat_vec + action_vec)
        y_rows.append(int(post_row['state_id']))

    # Also add null-action pairs (no intervention in window) to teach the model
    # what happens WITHOUT intervention
    null_pairs = _build_no_action_dataset(features)
    if null_pairs[0].shape[0] > 0:
        # Subsample to balance with action pairs (at most 2× action rows)
        n_action = len(X_rows)
        n_null = min(null_pairs[0].shape[0], max(n_action * 2, 200))
        idx = np.random.choice(null_pairs[0].shape[0], n_null, replace=False)
        X_rows.extend(null_pairs[0][idx].tolist())
        y_rows.extend(null_pairs[1][idx].tolist())

    X = np.array(X_rows, dtype=np.float32)
    y = np.array(y_rows, dtype=np.int32)
    print(f"Built {len(X)} training pairs ({n_action} with action, {n_null} null-action)")
    return X, y


# ── Small MLP in pure numpy (no torch/sklearn MLP needed) ─────────────────────
class MLP:
    """2-layer MLP with softmax output. Trained with SGD + label smoothing."""

    def __init__(self, in_dim, hidden=32, out_dim=N_STATES, lr=1e-3, smooth=0.05, l2=1e-4):
        rng = np.random.default_rng(42)
        self.W1 = rng.normal(0, np.sqrt(2.0/in_dim),  (in_dim,  hidden)).astype(np.float32)
        self.b1 = np.zeros(hidden, dtype=np.float32)
        self.W2 = rng.normal(0, np.sqrt(2.0/hidden), (hidden, out_dim)).astype(np.float32)
        self.b2 = np.zeros(out_dim, dtype=np.float32)
        self.lr = lr
        self.smooth = smooth
        self.l2 = l2

    def forward(self, X):
        h = np.maximum(0, X @ self.W1 + self.b1)   # ReLU
        logits = h @ self.W2 + self.b2
        logits -= logits.max(axis=1, keepdims=True)  # numerical stability
        exp = np.exp(logits)
        probs = exp / exp.sum(axis=1, keepdims=True)
        return probs, h

    def loss(self, X, y):
        probs, _ = self.forward(X)
        # Label smoothing
        n, K = probs.shape
        targets = np.full((n, K), self.smooth / K, dtype=np.float32)
        targets[np.arange(n), y] += 1.0 - self.smooth
        ce = -np.sum(targets * np.log(probs + 1e-12)) / n
        l2 = self.l2 * (np.sum(self.W1**2) + np.sum(self.W2**2))
        return ce + l2

    def step(self, X, y, class_weights=None):
        probs, h = self.forward(X)
        n, K = probs.shape
        targets = np.full((n, K), self.smooth / K, dtype=np.float32)
        targets[np.arange(n), y] += 1.0 - self.smooth
        dL_dlogits = (probs - targets) / n
        # Apply per-sample class weights if provided
        if class_weights is not None:
            w = class_weights[y].reshape(-1, 1)  # (n, 1)
            dL_dlogits = dL_dlogits * w
        # Layer 2 gradients
        dW2 = h.T @ dL_dlogits + self.l2 * 2 * self.W2
        db2 = dL_dlogits.sum(axis=0)
        # Layer 1 gradients
        dh = dL_dlogits @ self.W2.T
        dh[h <= 0] = 0.0   # ReLU derivative
        dW1 = X.T @ dh + self.l2 * 2 * self.W1
        db1 = dh.sum(axis=0)
        # SGD update
        self.W1 -= self.lr * dW1
        self.b1 -= self.lr * db1
        self.W2 -= self.lr * dW2
        self.b2 -= self.lr * db2

    def predict(self, X):
        probs, _ = self.forward(X)
        return np.argmax(probs, axis=1)

    def predict_proba(self, X):
        probs, _ = self.forward(X)
        return probs


def calibration_metrics(probs, y_true):
    """ECE and Brier score."""
    n = len(y_true)
    K = probs.shape[1]
    # Brier score (multiclass)
    onehot = np.zeros_like(probs)
    onehot[np.arange(n), y_true] = 1.0
    brier = np.mean(np.sum((probs - onehot)**2, axis=1))
    # ECE (confidence of top prediction vs accuracy)
    conf = probs.max(axis=1)
    pred = probs.argmax(axis=1)
    correct = (pred == y_true).astype(float)
    bins = np.linspace(0, 1, 11)
    ece = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (conf >= lo) & (conf < hi)
        if mask.sum() > 0:
            ece += (mask.sum() / n) * abs(correct[mask].mean() - conf[mask].mean())
    return brier, ece


def train(X, y, epochs=300, batch=64, balanced=True):
    in_dim = X.shape[1]
    model = MLP(in_dim=in_dim, hidden=64, out_dim=N_STATES, lr=5e-3, smooth=0.05, l2=1e-4)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X).astype(np.float32)

    X_tr, X_val, y_tr, y_val = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y if len(np.unique(y)) > 1 else None)

    # Class weights: inverse frequency, capped at 10× to avoid exploding on tiny classes
    if balanced:
        class_counts = np.bincount(y_tr, minlength=N_STATES).astype(np.float32)
        class_counts = np.maximum(class_counts, 1.0)
        class_weights = (len(y_tr) / (N_STATES * class_counts)).astype(np.float32)
        class_weights = np.minimum(class_weights, 10.0)
    else:
        class_weights = None

    rng = np.random.default_rng(0)
    for epoch in range(epochs):
        idx = rng.permutation(len(X_tr))
        for i in range(0, len(X_tr), batch):
            bi = idx[i:i+batch]
            model.step(X_tr[bi], y_tr[bi], class_weights)
        if (epoch + 1) % 50 == 0:
            tr_loss = model.loss(X_tr, y_tr)
            val_acc = (model.predict(X_val) == y_val).mean()
            print(f"  epoch {epoch+1:4d}  loss={tr_loss:.4f}  val_acc={val_acc:.3f}")

    return model, scaler, X_val, y_val


def evaluate(model, scaler, X_val, y_val):
    probs = model.predict_proba(X_val)
    preds = probs.argmax(axis=1)
    brier, ece = calibration_metrics(probs, y_val)

    print("\n── Evaluation ───────────────────────────────────────")
    print(f"  Val accuracy : {(preds == y_val).mean():.3f}")
    print(f"  Brier score  : {brier:.4f}  (lower = better, 0=perfect)")
    print(f"  ECE          : {ece:.4f}   (lower = better, 0=perfect)")

    STATE_NAMES = {
        0: 'Chaos', 1: 'Stable', 2: 'HeatDeath', 3: 'NearExt',
        4: 'Rich', 5: 'Predator', 6: 'PreAct', 7: 'Zombie'
    }
    labels = sorted(np.unique(np.concatenate([y_val, preds])))
    label_names = [STATE_NAMES.get(l, str(l)) for l in labels]

    print("\nConfusion matrix (rows=true, cols=pred):")
    cm = confusion_matrix(y_val, preds, labels=labels)
    header = "      " + "  ".join(f"{n:>8}" for n in label_names)
    print(header)
    for i, row in enumerate(cm):
        print(f"  {label_names[i]:>8}  " + "  ".join(f"{v:>8}" for v in row))

    print("\nPer-class report:")
    print(classification_report(y_val, preds, target_names=label_names, zero_division=0))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--save', action='store_true', help='Save trained predictor to disk')
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--no-balance', dest='balanced', action='store_false',
                        help='Disable class weighting (higher raw accuracy, poor minority recall)')
    parser.set_defaults(balanced=True)
    args = parser.parse_args()

    print("Loading data...")
    X, y = load_data()

    # Check class distribution
    unique, counts = np.unique(y, return_counts=True)
    print("\nClass distribution in training data:")
    STATE_NAMES = {0:'Chaos',1:'Stable',2:'HeatDeath',3:'NearExt',4:'Rich',5:'Predator',6:'PreAct',7:'Zombie'}
    for u, c in zip(unique, counts):
        print(f"  [{u}] {STATE_NAMES.get(u, '?'):12s}  {c:4d} rows")

    if len(unique) < 2:
        print("Need at least 2 classes to train. Collect more data first.")
        sys.exit(1)

    mode = "balanced (class-weighted)" if args.balanced else "unbalanced (raw frequency)"
    print(f"\nTraining MLP ({X.shape[1]} inputs → 64 hidden → {N_STATES} outputs) [{mode}]...")
    model, scaler, X_val, y_val = train(X, y, epochs=args.epochs, balanced=args.balanced)

    evaluate(model, scaler, X_val, y_val)

    if args.save:
        payload = {'model': model, 'scaler': scaler,
                   'feature_cols': FEATURE_COLS, 'action_keys': ACTION_KEYS,
                   'n_states': N_STATES, 'lookahead': LOOKAHEAD}
        with open(PREDICTOR_PATH, 'wb') as f:
            pickle.dump(payload, f)
        print(f"\nSaved predictor to: {PREDICTOR_PATH}")


if __name__ == '__main__':
    main()
