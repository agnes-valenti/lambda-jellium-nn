from functools import partial
from typing import Callable, Optional, Sequence, Tuple, Union

import jax
from flax import linen as nn
from jax import numpy as jnp

from ..utils import (Array, ElecCoord, _t_real, build_mlp, parse_activation,
                     ensure_no_spin)
from .base import ElecWfn
from .heg import heg_rs
from .neuralnet_pbc import raw_features_pbc

class MessagePassingLayer(nn.Module):
    mlp_width: int
    attn_width: int
    activation: Union[Callable, str] = "gelu"
    kernel_init: Callable = nn.linear.default_kernel_init

    def setup(self):
        self.Dense = partial(
            nn.Dense,
            param_dtype=_t_real,
            kernel_init=self.kernel_init)
        self.actv_fn = parse_activation(self.activation)

    def omega(self, g_ij):
        # Constructs the attention weight matrix
        # Eq (8) in arXiv:2305.07240
        g_ij = nn.LayerNorm()(g_ij)
        # Construct Q, K & W, NOTE here we use mlp_width, not attn_width
        q_ij = self.Dense(self.mlp_width, use_bias=False)(g_ij)
        k_ij = self.Dense(self.mlp_width, use_bias=False)(g_ij)
        # rescale before apply gelu (divide by sqrt(n_elec))
        scale = jnp.sqrt(g_ij.shape[0]) # n_elec
        w_ij = self.actv_fn(
            q_ij.transpose((2,0,1)) @ k_ij.transpose((2,0,1)) / scale
        ).transpose((1,2,0))
        # Reduce from mlp width to attention width
        # NOTE: this is not the same as the original paper, and may be unnecessary
        w_ij = self.Dense(self.attn_width, use_bias=False)(w_ij)
        return w_ij

    def phi(self, g_ij):
        # Constructs the message to be passed from layer to layer
        # appear in arXiv:2305.07240 when define m_ij at the paragraph above Eq (8)
        return nn.Sequential([
            nn.LayerNorm(),
            self.Dense(self.mlp_width),
            self.actv_fn,
            self.Dense(self.attn_width, use_bias=False)
        ])(g_ij)

    def f1b(self, g_i, m_ij, out_size):
        # MLP transforming the message to the single-particle stream h_i
        # Eq (9) in arXiv:2305.07240
        m_sum = jnp.einsum('ijk->ik', m_ij) - jnp.einsum('iik->ik', m_ij)
        h1_in = jnp.concatenate([m_sum, g_i], axis=-1)
        return nn.Sequential([
            nn.LayerNorm(),
            self.Dense(self.mlp_width),
            self.actv_fn,
            self.Dense(out_size, use_bias=False)
        ])(h1_in)

    def f2b(self, g_ij, m_ij, out_size):
        # MLP transforming the message to the particle-particle stream h_ij
        # Eq (10) in arXiv:2305.07240
        h2_in = jnp.concatenate([m_ij, g_ij], axis=-1)
        return nn.Sequential([
            nn.LayerNorm(),
            self.Dense(self.mlp_width),
            self.actv_fn,
            self.Dense(out_size, use_bias=False)
        ])(h2_in)

    @nn.compact
    def __call__(self, h_i, h_ij, x0_i, x0_ij):
        n_elec = h_i.shape[0]
        assert h_ij.shape[:2] == (n_elec, n_elec)
        g_i = jnp.concatenate([h_i, x0_i], axis=-1)
        g_ij = jnp.concatenate([h_ij, x0_ij], axis=-1)
        # make attention matrix
        w_ij = self.omega(g_ij)
        # Update the message
        m_ij = w_ij * self.phi(g_ij)
        # Update the single-particle and particle-particle streams
        h_i = self.f1b(g_i, m_ij, out_size=h_i.shape[-1]) + h_i
        h_ij = self.f2b(g_ij, m_ij, out_size=h_ij.shape[-1]) + h_ij
        # Send to next layer
        return h_i, h_ij
