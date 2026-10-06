import dataclasses
from functools import partial
from typing import Callable, Dict, NamedTuple, Tuple, Union, Optional

import jax
import numpy as np
from jax import lax
from jax import numpy as jnp
from jax import tree_util as jtu
from jax.flatten_util import ravel_pytree

from .utils import (PMAP_AXIS_NAME, Array, ArrayTree, PmapAxis, PyTree,
                    adaptive_split, build_moving_avg, clip_gradient,
                    parse_spin_num, ravel_shape, tree_where)
from .utils import log_cosh, sample_genlogistic # for relavistic hmc

KeyArray = Array
Params = ArrayTree
Sample = ArrayTree
State = Tuple[Sample, ...] # first element of state is always current sample
Data = Tuple[Sample, Array] # data is a tuple of sample and its log prob
Info = Dict[str, PyTree] # info is a dict containing stats of the sampling process

class MCSampler(NamedTuple):
    # to jit the sampler, use jtu.tree_map(jax.jit, sampler)
    sample: Callable[[KeyArray, Params, State], Tuple[State, Data, Info]]
    init: Callable[[KeyArray, Params], State]
    refresh: Callable[[State, Params], State]

    def __call__(self, key: KeyArray, params: Params, state: State):
        """Call the sample function. See `self.sample` for details."""
        return self.sample(key, params, state)

    def burn_in(self, key: KeyArray, params: Params, state: State, steps: int):
        """Burn in the state for given steps"""
        # inner = lambda s,k: (self.sample(k, params, s)[0], None)
        # return lax.scan(inner, state, jax.random.split(key, steps))[0]
        for _ in range(steps):
            key, subkey = adaptive_split(key, multi_device=key.ndim>1)
            state = self.sample(subkey, params, state)[0]
        return state


def choose_sampler_builder(name: str) -> Callable[..., MCSampler]:
    name = name.lower()
    if name in ("mixed_langevin", "mixed_mala"):
        return build_mixed_langevin
    raise NotImplementedError(f"unsupported sampler type: {name}")


def build_sampler(log_prob_fn: Callable[[Params, Sample], Array],
                  shape_or_init: Union[tuple, np.ndarray, callable],
                  name: str,
                  adaptive: Union[None, bool, dict] = None,
                  beta: float = 1,
                  **kwargs):   
    builder = choose_sampler_builder(name)
    logdens_fn = lambda p, x: beta * log_prob_fn(p, x)
    return builder(logdens_fn, shape_or_init, **kwargs)


def build_conf_init_fn(elems, nuclei, n_elec,
                       sigma_x=1., with_r=False, sigma_r=0.1):
    if nuclei.size == 0:
        nuclei = nuclei.sum(-2, keepdims=True)
        elems = [0]
    elems = jnp.asarray(elems, dtype=int)
    n_dim = nuclei.shape[-1]
    n_elec = int(sum(n_elec)) if not isinstance(n_elec, int) else n_elec
    if elems.sum() != n_elec: # put extra charge in first atoms
        elems = elems.at[0].add(n_elec - elems.sum())
    elec_list = [parse_spin_num(el, el%2 * (-1)**ii) for ii, el in enumerate(elems)]

    def init_fn(key):
        xa, xb = [], []
        for (na, nb), nuc in zip(elec_list, nuclei):
            key, ska, skb = jax.random.split(key, 3)
            xa.append(nuc + jax.random.normal(ska, (na, n_dim)) * sigma_x)
            xb.append(nuc + jax.random.normal(skb, (nb, n_dim)) * sigma_x)
        init_x = jnp.concatenate(xa + xb, axis=0)[:n_elec]
        if not with_r:
            return init_x
        else:
            init_r = nuclei + jax.random.normal(key, nuclei.shape) * sigma_r
            return [init_r, init_x]

    return init_fn


##### Below are sampler transformations #####

MC_BATCH_AXIS_NAME = "_mc_batch_axis"

def make_batched(sampler: MCSampler, n_batch: int, concat: bool = False,
                 vmap_axis_name: str = MC_BATCH_AXIS_NAME):
    sample_fn, init_fn, refresh_fn = sampler
    vaxis = PmapAxis(vmap_axis_name)
    def sample(key, params, state):
        vkey = jax.random.split(key, n_batch)
        new_state, *res = vaxis.vmap(sample_fn, (0, None, 0))(vkey, params, state)
        if concat:
            res = jtu.tree_map(jnp.concatenate, res)
        return new_state, *res
    def init(key, params):
        vkey = jax.random.split(key, n_batch)
        return vaxis.vmap(init_fn, (0, None))(vkey, params)
    refresh = vaxis.vmap(refresh_fn, (0, None))
    return MCSampler(sample, init, refresh)


def make_multistep(sampler: MCSampler, n_step: int, concat: bool = False):
    sample_fn, init_fn, refresh_fn = sampler
    multisample_fn = make_multistep_fn(sample_fn, n_step, concat)
    return MCSampler(multisample_fn, init_fn, refresh_fn)


def make_multistep_fn(sample_fn, n_step, concat=False):
    def _split_output(out): # to satisfy scan requirement
        return out[0], out[1:]
    def multi_sample(key, params, state):
        inner = lambda s,k: _split_output(sample_fn(k, params, s))
        keys = jax.random.split(key, n_step)
        new_state, res = lax.scan(inner, state, keys)
        if concat:
            res = jtu.tree_map(jnp.concatenate, res)
        return new_state, *res
    return multi_sample

def make_adaptive(
        sampler_factory: Callable[..., MCSampler],
        keyword: str, # the kwarg name to be tuned upon
        reference: str, # the info key to be used as reference
        target: Union[float, Tuple[float, float]], # the targeting interval for the reference
        transform: Optional[Callable[[float], float]] = None, # transform the reference
        increasing: bool = False, # if the reference is increasing w.r.t. the kwarg
        *, # above are required, below are optional user settings
        interval: int = 100, # the interval to update the kwarg
        scale_factor: float = 1.01, # the factor to scale the kwarg
        ema_decay: Optional[float] = None, # the decay of the exponential moving average
        vmap_axis_name: str = MC_BATCH_AXIS_NAME,
        pmap_axis_name: str = PMAP_AXIS_NAME,
) -> Callable[..., MCSampler]:

    # initialize pmap and vmap utilities
    vaxis = PmapAxis(vmap_axis_name)
    paxis = PmapAxis(pmap_axis_name)

    # exponential moving average of the reference
    _decay = 1 - 1. / interval if ema_decay is None else ema_decay
    moving_avg = build_moving_avg(_decay)

    # dynamic tuning function
    assert scale_factor > 1., "scale factor must be larger than 1"
    if not increasing:
        scale_factor = 1. / scale_factor
    scale_arr = jnp.array([scale_factor, 1., 1./scale_factor])
    tmin, tmax = (target, target) if isinstance(target, float) else target
    def tuning(value, reference):
        # 0: too small, 1: good, 2: too large
        idx = (reference > tmin).astype(int) + (reference > tmax).astype(int)
        return value * scale_arr[idx]

    # transformation of the reference
    transform = transform or (lambda x: x)

    # return a new sampler factory that takes the same arguments as the original
    def adaptive_builder(logdens_fn, shape_or_init, **kwargs):
        raw_sampler = sampler_factory(logdens_fn, shape_or_init, **kwargs)
        # build new sampler every time
        def sample(key, params, state):
            inner_sample, inner_aux, count, adapt_value, ema_ref = state
            inner_state = (inner_sample, *inner_aux)
            ada_kwargs = {**kwargs, keyword: adapt_value}
            ada_sampler = sampler_factory(logdens_fn, shape_or_init, **ada_kwargs)
            new_inner_state, data, info = ada_sampler.sample(key, params, inner_state)
            ref_update = transform(paxis.pmean(vaxis.all_mean(info[reference])))
            new_ema_ref = moving_avg(ema_ref, ref_update, count)
            adapt_value = jnp.where((count > 0) & (count % interval == 0),
                                    tuning(adapt_value, new_ema_ref),
                                    adapt_value)
            new_sample, *new_aux = new_inner_state
            new_state = (new_sample, new_aux, count+1, adapt_value, new_ema_ref)
            info["tuned_hparam"] = adapt_value
            return new_state, data, info
        # refresh the inner state
        def refresh(state, params):
            inner_sample, inner_aux, count, adapt_value, ema_ref = state
            new_sample, *new_aux = raw_sampler.refresh((inner_sample, *inner_aux), params)
            return new_sample, new_aux, count, adapt_value, ema_ref
        # init inner state and wrap it with adapting kwarg
        def init(key, params):
            count = 0
            inner_sample, *inner_aux = raw_sampler.init(key, params)
            init_value = kwargs.get(keyword, 0.1) # defaults to 0.1
            ema_ref = 0.
            return (inner_sample, inner_aux, count, init_value, ema_ref)
        # assemble the adaptive sampler
        return MCSampler(sample, init, refresh)

    return adaptive_builder


##### Below are generation functions for different samplers #####

def build_mixed_langevin(logdens_fn, shape_or_init, tau=0.1, steps=10,
                    grad_clipping=None, p_flip=0.25, discrete_sample=(False,True)):
    # shape and constants
    sample_shape = _extract_sample_shape(shape_or_init)
    xsize, unravel = ravel_shape(sample_shape)
    ravel_logd = lambda p, x: logdens_fn(p, unravel(_gclip(x, grad_clipping)))
    logd_and_grad = jax.value_and_grad(ravel_logd, 1)
    discrete_sample = _align_vector(discrete_sample, sample_shape)
    

    # log transition probability q(x2|x1)
    def log_q(x2, x1, g1):
        d = x2 - x1 - tau * jnp.where(discrete_sample,0, g1)
        norm2 = (d * d.conj()).real
        return (-1/(4*tau) * norm2).sum(-1)

    def step(key, params, state):
        x1, g1, ld1 = state
        g1key, g2key, ukey = jax.random.split(key,3)
        x2 = x1 + jnp.where(discrete_sample,0.5*jnp.heaviside(jax.random.uniform(g1key, shape=x1.shape)+p_flip-1,0),tau*g1 + jnp.sqrt(2*tau)*jax.random.normal(g2key, shape=x1.shape))
        x2 = x2.at[discrete_sample].set(jnp.mod(x2[discrete_sample],1.0))
        ld2, g2 = logd_and_grad(params, x2)
        g2 = g2.conj() # handle complex grads, no influence for real case
        ratio = ld2 + log_q(x1, x2, g2) - ld1 - log_q(x2, x1, g1)
        return (*mh_select(ukey, ratio, state, (x2, g2, ld2)), ratio)

    def sample(key, params, state):
        multi_step = make_multistep_fn(step, steps, concat=False)
        new_state, accepted, log_ratio = multi_step(key, params, state)
        new_sample, new_grads, new_logdens = new_state
        info = {"is_accepted": accepted.mean(),
                "recip_ratio": _recip_ratio(log_ratio).mean()}
        return new_state, (unravel(new_sample), new_logdens), info

    def refresh(state, params):
        sample = state[0]
        ld_new, grads_new = logd_and_grad(params, sample)
        return (sample, grads_new.conj(), ld_new)

    init = _gen_init_from_refresh(refresh, shape_or_init)

    return MCSampler(sample, init, refresh)

##### Below are helper functions for samplers #####

def logd_gaussian(x, mu=0., sigma=1.):
    """unnormalized log density of Gaussian distribution"""
    return -0.5 * jnp.square((x - mu) / sigma)


def mh_select(key, ratio, state1, state2):
    rnd = jnp.log(jax.random.uniform(key, shape=ratio.shape))
    cond = ratio > rnd
    new_state = tree_where(cond, state2, state1)
    return new_state, cond

def relavistic_ke(p, c):
    r"""relavistic kinetic energy, use formula $K(p) = c^2 \log(\cosh(p / c))$

    It is relativistic in the sense that the velocity $dK/dp = c \tanh(p / c)$
    is bounded between -c and c, and is roughly linear for small $p$.
    """
    return c**2 * log_cosh(p / c)


def sample_relavistic_momentum(key, c, shape=(), dtype=float):
    r"""Sample momentum from relativistic distribution

    The distribution is defined by the kinetic energy $K(p) = c^2 \log(\cosh(p / c))$.
    Hence the momentum is sampled from the distribution $p \sim \exp(-K(p))$, which
    is a generalized logistic distribution with scale $c/2$, and $a = b = c^2/2$.
    """
    return sample_genlogistic(key, c**2/2, c**2/2, shape, dtype) * c / 2


def _recip_ratio(log_ratio):
    return jnp.exp(-jnp.minimum(0, log_ratio))


def _gclip(x, bnd):
    if bnd is None:
        return x
    if isinstance(bnd, (int, float, Array)):
        bnd = (-abs(bnd), abs(bnd))
    return clip_gradient(x, *bnd)


def _extract_sample_shape(shape_or_init):
    # shape_or_init is either a pytree of shapes
    # or a init function that take a key and give an init x
    with jax.ensure_compile_time_eval():
        if not callable(shape_or_init): # is shape
            sample_shape = shape_or_init
        else: # is init function
            _dummy_key = jax.random.PRNGKey(0)
            init_sample = shape_or_init(_dummy_key)
            sample_shape = jtu.tree_map(lambda a: np.array(a.shape), init_sample)
    return sample_shape


def _gen_init_from_refresh(refresh_fn, shape_or_init):
    # shape_or_init is either a pytree of shapes
    # or a init function that take a key and give an init x
    with jax.ensure_compile_time_eval():
        if not callable(shape_or_init):
            size, unravel = ravel_shape(shape_or_init)
            mu, sigma = 0., 1.
            raw_init = lambda key: jax.random.normal(key, (size,)) * sigma + mu
        else:
            from jax.flatten_util import ravel_pytree
            raw_init = lambda key: ravel_pytree(shape_or_init(key))[0]

    def init(key, params):
        sample = raw_init(key)
        return refresh_fn((sample,), params)

    return init


def _align_vector(vec, sample_shape):
    xsize, unravel = ravel_shape(sample_shape)
    vec_ = ravel_pytree(vec)[0]
    vec = (vec_ if vec_.size in (1, xsize) else
           ravel_pytree(jtu.tree_map(jnp.broadcast_to, vec, sample_shape))[0])
    assert vec.shape in ((1,), (xsize,))
    return vec
