from functools import partial
from typing import Callable, Optional, Sequence, Tuple, Union

import jax
from flax import linen as nn
from jax import numpy as jnp

from ..utils import (Array, ElecCoord, _t_real, _t_cplx, build_mlp, parse_activation,
                     ensure_no_spin, split_spin, attach_spin)
from .base import ElecWfn
from .heg import heg_rs
from .neuralnet_pbc import raw_features_pbc
from .hegnn import MessagePassingLayer


class NeuralBackflowSpin(nn.Module):
    #spins: Sequence[int]
    cell: Array
    out_comps: int = 1 # number of components in the output
    single_size: int = 32 # size of h_i, before concat with x0_i
    pair_size: int = 26 # size of h_ij, before concat with x0_ij (size 6 for 2D)
    mlp_width: int = 32 # width of the mlp of the message passing layer
    attn_width: int = 32 # width of the message passing layer m_ij
    backflow_scale: float = 0.03 # hardcoded scale of the backflow shift (in rs)
    backflow_layers: int = 3 # number of message passing layers
    jastrow_width: int = 32 # width of the hidden layers in the jastrow mlp
    jastrow_layers: int = 3 # number of layers in the jastrow mlp
    jastrow_postlayers: int = 0 # number of layers in jastrow after sum over electrons
    activation: Union[Callable, str] = "gelu"
    kernel_init: Callable = nn.linear.default_kernel_init

    @nn.compact
    def __call__(self, x: ElecCoord) -> Tuple[ElecCoord, Tuple[float, float]]:
        # set up constants
        x, s = split_spin(x)
        n_elec, n_dim = x.shape
        #assert not self.spins or sum(self.spins) == n_elec
        actv_fn = parse_activation(self.activation)
        MyDense = partial(nn.DenseGeneral, param_dtype=_t_real, kernel_init=self.kernel_init) #jax.nn.initializers.zeros)
        # input independent arrays
        with jax.ensure_compile_time_eval():
            invvec = jnp.linalg.inv(self.cell)
            rs = heg_rs(self.cell, n_elec)
            # all same spin, no need to compute s_ij
            s_ij = jnp.zeros((n_elec, n_elec, 1))
        # make raw pbc feature (dummy r from empty array)
        d_i, d_ij, dist = raw_features_pbc(x[:0], x, self.cell, 1, frac_dist=True)
        x0_i = d_i[..., :0] # dummy x0_i, empty array
        if s is not None: #spin dependent BF
            s_idx=jnp.sign(s%1-.5)
            x0_i=s_idx[:,None]
            s_ij = s_idx[:,None,None]+s_idx[None,:,None]
        x0_ij = jnp.concatenate([d_ij, s_ij], axis=-1) # [sinx,siny,cosx,cosy,sind,s_ij]
        # Initialize (raw) h_i and h_ij
        h_i = self.param("h_i", nn.initializers.zeros, (self.single_size,))
        h_i = jnp.broadcast_to(h_i, (n_elec, h_i.shape[-1]))
        h_ij = self.param("h_ij", nn.initializers.zeros, (self.pair_size,))
        h_ij = jnp.broadcast_to(h_ij, (n_elec, n_elec, h_ij.shape[-1]))
        # Run the iterative backflow loop
        for _ in range(self.backflow_layers):
            h_i, h_ij = MessagePassingLayer(
                mlp_width=self.mlp_width, attn_width=self.attn_width,
                activation=actv_fn, kernel_init=self.kernel_init
            )(h_i, h_ij, x0_i, x0_ij)
        # the final displacement vector, shape: (n_elec, n_comp, n_dim)
        bfdense = MyDense((self.out_comps, n_dim), use_bias=False)
        x = x[:, None, :] + bfdense(h_i) * self.backflow_scale * rs
        # neural jastrow
        x_pbc = jnp.concatenate(
            [jnp.sin(2 * jnp.pi * x @ invvec), jnp.cos(2 * jnp.pi * x @ invvec)],
            axis=-1)
        x_expand = MyDense(self.single_size)(x_pbc.reshape(n_elec, -1))
        if s is not None:
            j_in = jnp.concatenate([h_i, x0_i, actv_fn(x_expand)], axis=-1)
        else:
            j_in = jnp.concatenate([h_i, actv_fn(x_expand)], axis=-1)
        # skip connection in the jastrow network
        J_out_size = 2 #if s is not None else 1
        presum_size = J_out_size if self.jastrow_postlayers <= 0 else self.jastrow_width
        jas_sizes = [self.jastrow_width] * self.jastrow_layers + [presum_size]
        jasmlp = build_mlp(jas_sizes,
                           activation=actv_fn, last_bias=False,
                           residual=True, rescale=True,
                           kernel_init=self.kernel_init,
                           param_dtype=_t_real)
        jastrow = jasmlp(j_in).mean(0)
        # deepset construction
        if self.jastrow_postlayers > 0:
            jastrow = nn.LayerNorm()(jastrow)
            postmlp = build_mlp([self.jastrow_width] * self.jastrow_postlayers + [J_out_size],
                                activation=actv_fn, last_bias=False,
                                residual=True, rescale=True,
                                kernel_init=self.kernel_init,
                                param_dtype=_t_real)
            jastrow = postmlp(jastrow)
        jastrow = n_elec * jastrow#.reshape(()) # scalar; scale by n_elec
        angle = jnp.exp(1j*jastrow[1]) if jastrow.shape[0]==2 else 1
        jastrow = (angle, jastrow[0])
        # return the final coordinates and the jastrow
        x = (jnp.squeeze(x, axis=-2) if self.out_comps == 1 else
             jnp.moveaxis(x, -2, 0)) # move the output components to the front
        return attach_spin(x,s) , jastrow
