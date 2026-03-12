import jax.numpy as jnp
from jax import random

from nca.model import N_CHANNELS, N_FILTERS

IDX_IDENTITY  = slice(0,              N_CHANNELS)
IDX_SOBEL_X   = slice(N_CHANNELS,     N_CHANNELS * 2)
IDX_SOBEL_Y   = slice(N_CHANNELS * 2, N_CHANNELS * 3)
IDX_LAPLACIAN = slice(N_CHANNELS * 3, N_CHANNELS * 4)


def _apply_fluid(params, key):
    kernel = params['params']['Dense_0']['kernel']
    kernel = kernel.at[IDX_IDENTITY,  :].multiply(0.4)
    kernel = kernel.at[IDX_SOBEL_X,   :].multiply(3.0)
    kernel = kernel.at[IDX_SOBEL_Y,   :].multiply(3.0)
    kernel = kernel.at[IDX_LAPLACIAN, :].multiply(4.0)
    key, subkey = random.split(key)
    kernel = kernel + random.normal(subkey, kernel.shape) * 0.08
    params['params']['Dense_0']['kernel'] = kernel
    return params, key


def _apply_geometric(params, key):
    kernel = params['params']['Dense_0']['kernel']
    kernel = kernel.at[IDX_IDENTITY,  :].multiply(3.5)
    kernel = kernel.at[IDX_SOBEL_X,   :].multiply(0.3)
    kernel = kernel.at[IDX_SOBEL_Y,   :].multiply(0.3)
    kernel = kernel.at[IDX_LAPLACIAN, :].multiply(2.5)
    params['params']['Dense_0']['kernel'] = kernel
    return params, key


def _apply_organic(params, key):
    kernel = params['params']['Dense_0']['kernel']
    kernel = kernel.at[IDX_IDENTITY,  :].multiply(1.2)
    kernel = kernel.at[IDX_SOBEL_X,   :].multiply(1.5)
    kernel = kernel.at[IDX_SOBEL_Y,   :].multiply(1.5)
    kernel = kernel.at[IDX_LAPLACIAN, :].multiply(1.8)
    key, subkey = random.split(key)
    kernel = kernel + random.normal(subkey, kernel.shape) * 0.1
    params['params']['Dense_0']['kernel'] = kernel
    return params, key


def _apply_chaotic(params, key):
    kernel = params['params']['Dense_0']['kernel']
    kernel = kernel.at[IDX_IDENTITY,  :].multiply(2.5)
    kernel = kernel.at[IDX_SOBEL_X,   :].multiply(4.0)
    kernel = kernel.at[IDX_SOBEL_Y,   :].multiply(4.0)
    kernel = kernel.at[IDX_LAPLACIAN, :].multiply(6.0)
    # Flip 30% of weights negative — more conflict, more oscillation
    key, subkey = random.split(key)
    flip_mask = random.bernoulli(subkey, p=0.3, shape=kernel.shape)
    kernel    = jnp.where(flip_mask, -kernel, kernel)
    params['params']['Dense_0']['kernel'] = kernel
    return params, key


_RULESET_FNS = {
    "fluid":     _apply_fluid,
    "geometric": _apply_geometric,
    "organic":   _apply_organic,
    "chaotic":   _apply_chaotic,
}

def apply_ruleset(params, category: str, key):
    if category not in _RULESET_FNS:
        raise ValueError(f"Unknown category '{category}'. Valid: {list(_RULESET_FNS.keys())}")
    return _RULESET_FNS[category](params, key)
