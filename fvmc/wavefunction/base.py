import abc
import functools
import dataclasses
from typing import Any, Callable, Sequence, Tuple

import jax
from flax import linen as nn
from jax import numpy as jnp
from jax import tree_util as jtu

from ..utils import ElecConf, NuclConf, Array


def model_wrapper(model: nn.Module,
                  wrap_out: Callable[[Array, Array], Any]):
    # switch between different model type
    if isinstance(model, FullWfn): # combine r, x into a tuple conf = (r, x)
        return lambda p, conf: wrap_out(*model.apply(p, *conf))
    elif isinstance(model, ElecWfn): # electron only case
        return lambda p, x: wrap_out(*model.apply(p, x))
    else: # fall back for any model
        # raise TypeError(f"unsupoorted model type {type(model)}")
        return lambda p, *a, **kw: wrap_out(*model.apply(p, *a, **kw))

def _wrap_sign(sign, logp):
    if jnp.iscomplexobj(sign):
        logp += jnp.log(sign)
    return logp

log_prob_from_model = functools.partial(model_wrapper, wrap_out=lambda s, l: 2*l)
log_psi_from_model = functools.partial(model_wrapper, wrap_out=_wrap_sign)


class FullWfn(nn.Module, abc.ABC):
    @abc.abstractmethod
    def __call__(self, r: Array, x: Array) -> Tuple[Array, Array]:
        """Take nuclei position r and electron position x, return sign and log|psi|"""
        raise NotImplementedError


class ElecWfn(nn.Module, abc.ABC):
    @abc.abstractmethod
    def __call__(self, x: Array) -> Tuple[Array, Array]:
        """Take only the electron position x, return sign and log|psi|"""
        raise NotImplementedError

@dataclasses.dataclass
class FrozenModel:
    model: nn.Module
    frozen_params: Any

    def init(self, rng, *args, **kwargs):
        raw_params = self.model.init(rng, *args, **kwargs)
        return jtu.tree_map(
            lambda x, y: y if x is None else None,
            self.frozen_params, raw_params, is_leaf=_is_none)

    def apply(self, params, *args, **kwargs):
        all_params = jtu.tree_map(
            lambda x, y: y if x is None else x,
            self.frozen_params, params, is_leaf=_is_none)
        return self.model.apply(all_params, *args, **kwargs)

    def __getattr__(self, name: str) -> Any:
        if hasattr(nn.Module, name):
            raise AttributeError(
                "only `init` and `apply` are allowed in FrozenModel")
        return getattr(self.model, name)


_is_none = lambda x: x is None
