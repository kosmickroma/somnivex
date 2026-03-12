# preference/model.py — the preference MLP.
#
# A tiny neural net that learns to predict whether a user will like
# a given era of the screensaver based on its parameters.
#
# This is trained locally on your own ratings. It never leaves your machine.
# Ships as a blank slate (no pre-trained weights). Your taste trains it.
# Other people who download Somnivex start blank and train their own copy.
#
# Architecture: 3 Dense layers, ~2000 total parameters.
# Serialized to preference_weights.pkl — about 10KB on disk.

import flax.linen as nn
import jax.numpy as jnp


class PreferenceMLP(nn.Module):
    """
    Input:  feature vector encoding one era (~30 dims — see features.py)
    Output: single logit (sigmoid it to get probability of liking)

    Why so small?
    The dataset is small (tens to hundreds of ratings) and the input
    is low-dimensional (one-hot categoricals + 2 floats). A bigger model
    would just overfit. This size can generalize from ~50 ratings.

    Why relu here (not tanh like the NCA)?
    We want a fast converging classifier, not oscillating dynamics.
    relu is standard for classification MLPs.
    """
    @nn.compact
    def __call__(self, x):
        x = nn.Dense(64)(x)
        x = nn.relu(x)
        x = nn.Dense(32)(x)
        x = nn.relu(x)
        x = nn.Dense(1)(x)
        return x.squeeze(-1)    # shape () for single sample, (N,) for batch
