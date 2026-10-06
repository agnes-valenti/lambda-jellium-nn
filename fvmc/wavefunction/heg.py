import functools
from typing import Optional, Sequence, Tuple, Union, Callable

import jax
import numpy as np
from flax import linen as nn
from jax import numpy as jnp
from jax.lax import dynamic_slice

from ..utils import (Array, DenseComplex, ElecCoord, ElecConf, _t_real,
                     fix_init, gen_ksphere_adaptive, log_linear_exp,
                     ensure_no_spin, split_spin, attach_spin,
                     displace_matrix, pdist)
from ..ewaldsum import gen_pbc_disp_fn
from .base import ElecWfn
from .neuralnet_pbc import dist_features_pbc


def heg_rs(cell: Array, nelec: int) -> float:
    """Calculate density parameter r_s of the homogeneous electron gas
    """
    ndim = jnp.asarray(cell).shape[-1]
    vol = jnp.abs(jnp.linalg.det(cell))
    vol_pp = vol / nelec
    rs = ((2 * (ndim-1) * jnp.pi) / (ndim * vol_pp)) ** (-1./ndim)
    return rs


def apply_backflow(backflow: Callable, x: ElecConf):
    _x, _s = split_spin(x) # backflow cannot depend on s
    out = backflow(_x)
    if isinstance(out, tuple):
        _x_bf, (sign, logf) = out
    else:
        _x_bf = out
        sign, logf = 1., 0.
    return attach_spin(_x_bf, _s), (sign, logf)


class ElecProductModel(ElecWfn):
    submodels: Sequence[nn.Module]
    backflow: Optional[nn.Module] = None
    apply_backflow: Union[bool, Sequence[bool]] = True

    def setup(self) -> None:
        self.bf_mask = ([self.apply_backflow] * len(self.submodels)
                        if isinstance(self.apply_backflow, bool)
                        else self.apply_backflow)

    def __call__(self, x: ElecConf) -> Tuple[Array, Array]:
        sign, logf = 1., 0.
        x_bf = x
        if self.backflow is not None:
            x_bf, (sign_bf, logf_bf) = apply_backflow(self.backflow, x)
            sign *= sign_bf
            logf += logf_bf
        for model, with_bf in zip(self.submodels, self.bf_mask):
            result = model(x_bf if with_bf else x)
            if isinstance(result, tuple):
                sign *= result[0]
                logf += result[1]
            else:
                logf += result
        return sign, logf

# for jastrow and backflow spin blocks
def block_diagonal_masks(spins: Sequence[int], n_elec: int, triu: bool) -> Tuple[Array]:
    assert sum(spins) == n_elec
    cum_idx = np.cumsum([0, *spins])
    offdiag_mask = jnp.ones((n_elec, n_elec), dtype=bool)
    for i, j in zip(cum_idx[:-1], cum_idx[1:]):
        offdiag_mask = offdiag_mask.at[i:j, i:j].set(False)
    if triu:
        blkdiag_mask = jnp.triu(~offdiag_mask, k=1)
        offdiag_mask = jnp.triu(offdiag_mask)
    else:
        blkdiag_mask = ~offdiag_mask
    return blkdiag_mask, offdiag_mask


class JastrowU2Bspline(nn.Module):
    n_params: int
    rcut: float
    init_scale: Optional[float] = 0.1

    @nn.compact
    def __call__(self, r: float, cusp: float) -> float:
        n_coeffs = self.n_params + 4
        n_knots = n_coeffs - 2
        delta_r = self.rcut / (n_knots - 1)
        params = self.param('coeffs', nn.initializers.normal(self.init_scale),
                            (self.n_params,))
        coeffs = jnp.concatenate([
            (params[0] - 2.0 * delta_r * cusp).reshape(1),
            params,
            jnp.zeros(3)
        ], axis=0)
        mask = r <= self.rcut
        r = jnp.clip(r, 0.0, self.rcut)
        d = bspline(r, delta_r, coeffs) * mask
        return d


def bspline(r: float, delta_r: float, coeffs: Array) -> float:
    a_mat = jnp.array([
        [-1.0, 3.0, -3.0, 1.0],
        [3.0, -6.0, 0.0, 4.0],
        [-3.0, 3.0, 3.0, 1.0],
        [1.0, 0.0, 0.0, 0.0],
    ]) / 6.0
    r = r / delta_r
    ival = jnp.floor(r).astype(int)
    r = jnp.expand_dims((r - ival), -1) # extra dim for broadcasting with a_mat
    d_comp = ((a_mat[:,0] * r + a_mat[:,1]) * r + a_mat[:,2]) * r + a_mat[:,3]
    d = (dynamic_slice(coeffs, (ival,), (4,)) * d_comp).sum(-1)
    return d


# below are backflow models
class BackflowEtaBessel(nn.Module):
    n_basis: int
    rcut: float
    npoly_smooth: Optional[int] = 6
    init_scale: Optional[float] = 0.1

    @nn.compact
    def __call__(self, r: float) -> float:
        coeffs = self.param('coeffs', nn.initializers.normal(self.init_scale),
                            (self.n_basis,))
        basis = bessel(r, self.n_basis, self.rcut)
        envelope = polynomial_envelope(r, self.npoly_smooth, self.rcut)
        return jnp.expand_dims(envelope, -1) * basis@coeffs


def polynomial_envelope(x: Array, p: int = 6, x_max: float=1.0) -> Array:
    #  smoothly cut off function defined from x \in [0, 1)
    #  arXiv: 2003.03123 eq. (8)
    p1 = p + 1
    p2 = p + 2
    a = - p1 * p2 / 2
    b = p * p2
    c = - p * p1 / 2
    d = x / x_max
    y = 1 + a * d**p + b * d**p1 + c * d**p2
    return jnp.where(d > 1, 0.0, y)


# from e3nn-jax with Apache 2.0
def bessel(x: Array, n: int, x_max: float = 1.0) -> Array:
    r"""Bessel basis functions.

    They obey the following normalization:

    .. math::

        \int_0^c r^2 B_n(r, c) B_m(r, c) dr = \delta_{nm}

    Args:
        x (jax.Array): input of shape ``[...]``
        n (int): number of basis functions
        x_max (float): maximum value of the input

    Returns:
        jax.Array: basis functions of shape ``[..., n]``

    Klicpera, J.; Groß, J.; Günnemann, S. Directional Message Passing for Molecular Graphs;
    ICLR 2020. Equation (7)
    """
    assert isinstance(n, int)
    x = jnp.expand_dims(x, -1)
    n = jnp.arange(1, n + 1, dtype=x.dtype)
    x_nonzero = jnp.where(x == 0.0, 1.0, x)
    return jnp.sqrt(2.0 / x_max) * jnp.where(
        x == 0,
        n * jnp.pi / x_max,
        jnp.sin(n * jnp.pi / x_max * x_nonzero) / x_nonzero,
    )


class BackflowEtaBspline(nn.Module):
    n_params: int
    rcut: float
    init_scale: Optional[float] = 0.1

    @nn.compact
    def __call__(self, r: float) -> float:
        n_coeffs = self.n_params + 3
        n_knots = n_coeffs - 2
        delta_r = self.rcut / (n_knots - 1)
        params = self.param('coeffs', nn.initializers.normal(self.init_scale),
                            (self.n_params,))
        coeffs = jnp.concatenate([params, jnp.zeros(3)], axis=0)
        mask = r <= self.rcut
        r = jnp.clip(r, 0.0, self.rcut)
        d = bspline(r, delta_r, coeffs) * mask
        return d
