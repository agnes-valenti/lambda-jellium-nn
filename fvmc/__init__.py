import jax
jax.config.update("jax_enable_x64", True)
# TODO: update pmap before the following flag is removed
if "jax_pmap_no_rank_reduction" in jax.config.values:
    jax.config.update("jax_pmap_no_rank_reduction", True)

import logging
logging.basicConfig(force=True, format='# [%(asctime)s] %(levelname)s: %(message)s')
LOGGER = logging.getLogger("fvmc")
del logging


__all__ = [
    "utils",
    "ewaldsum",
    "hamiltonian",
    "sampler",
    "estimator",
    "wavefunction",
    "config",
    "infer",
    "observable",
]

def __getattr__(name):
    from importlib import import_module
    if name in __all__:
        return import_module("." + name, __name__)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
