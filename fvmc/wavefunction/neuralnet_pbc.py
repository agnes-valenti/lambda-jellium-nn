import dataclasses
from typing import Optional, Sequence, Tuple

import jax
import numpy as np
from flax import linen as nn
from jax import numpy as jnp

from ..utils import (Array, ElecConf, NuclConf, _t_real, build_mlp,
                     collect_elems, displace_matrix, ensure_no_spin, gen_kidx,
                     log_linear_exp, wrap_complex_linear)
from .base import FullWfn



def dist_features_pbc_generic(d_frac, latvec, n, frac_dist=False, keepdims=True):
    # minimum-image displacement for generic cell. 
    # Cassella, Gino, et al. "Discovering quantum phase transitions with fermionic neural networks." Physical Review Letters 130.3 (2023): 036401.
    d_sin = jnp.sin(2*jnp.pi * d_frac) + jnp.eye(n)[..., None] 
    d_cos = 1 - jnp.cos(2*jnp.pi * d_frac) + jnp.eye(n)[..., None] 
    S_ij = latvec @ latvec.T
    if frac_dist: S_ij = S_ij / jnp.linalg.det(latvec)
    comp1 = jnp.einsum('...i,ij,...j->...', d_sin, S_ij, d_sin) 
    comp2 = jnp.einsum('...i,ij,...j->...', d_cos, S_ij, d_cos) 
    dist = jnp.sqrt(comp1 + comp2)/(2*jnp.pi)
    if keepdims:
        dist = dist[..., None]  # [n, n, 1]
    return (d_sin,d_cos), dist

def dist_features_pbc_ortho(d_frac, latvec, n, frac_dist=False, keepdims=True):
    # legacy function for orthorhombic cell
    # orthorhombic minimum-image displacement
    d_hsin = jnp.sin(jnp.pi * d_frac)  # output \in [0, 1)
    if not frac_dist:
        d_hsin = d_hsin @ (latvec / jnp.pi)
    dist = jnp.linalg.norm(
        d_hsin + jnp.eye(n)[..., None],
        keepdims=keepdims, axis=-1)
    return d_hsin, dist
    

def dist_features_pbc(pos, latvec, frac_dist=False, keepdims=True, is_ortho=False):
    n = len(pos)
    invvec = jnp.linalg.inv(latvec)
    pos_frac = pos @ invvec
    d_frac = displace_matrix(pos_frac, pos_frac)
    d_frac = d_frac - jnp.rint(d_frac)
    dist_func = dist_features_pbc_ortho if is_ortho else dist_features_pbc_generic
    feuatures, dist = dist_func(d_frac, latvec, n, frac_dist, keepdims)
    if keepdims:
        dist = dist * (1.0 - jnp.eye(n)[..., None])
    else:
        dist = dist * (1.0 - jnp.eye(n)) 
    return d_frac, feuatures, dist


def raw_features_pbc(r, x, latvec, n_freq, frac_dist=False):
    n_elec = x.shape[0]
    n_nucl = r.shape[0]
    n_p = n_nucl + n_elec
    # use both nuclei and electron positions
    pos = jnp.concatenate([r, x], axis=0)
    # initial h1 is empty, trick to avoid error in kfac
    h1 = pos[:, :1] * 0
    # h2 for pbc handling
    d_frac, _, dist = dist_features_pbc(pos, latvec,
        frac_dist=frac_dist, keepdims=True)
    freqs = jnp.arange(1, n_freq+1).reshape(-1, 1)
    radfreqs = 2 * jnp.pi * freqs
    d_asin = jnp.sin(radfreqs * d_frac[:,:,None,:])
    d_acos = jnp.cos(radfreqs * d_frac[:,:,None,:])
    d_freq = jnp.concatenate([
        d_asin, d_acos
    ], axis=-2).reshape(n_p, n_p, -1)
    h2 = jnp.concatenate([
        d_freq, dist
    ], axis=-1)
    return h1, h2, dist

