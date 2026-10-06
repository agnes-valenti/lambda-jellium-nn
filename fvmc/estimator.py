from functools import partial

import jax
from jax import lax
from jax import numpy as jnp
from jax import tree_util as jtu

from .ewaldsum import EwaldSum, EwaldSumScreened
from .hamiltonian import (calc_ke_elec, calc_ke_full, calc_pe,
                          calc_soc_base, calc_ke_and_soc, calc_spin_lapl)
from .utils import (PMAP_AXIS_NAME, ElecConf, PmapAxis,
                    split_spin, attach_spin, adaptive_grad)


def get_logpsi_fn(model_apply, params, stop_gradient=False):
    # make log of wavefunction from model.apply
    if stop_gradient:
        params = lax.stop_gradient(params)
    def log_psi(*args):
        sign, logd = model_apply(params, *args)
        if jnp.iscomplexobj(sign):
            logd += jnp.log(sign)
        return logd
    return log_psi


def get_logpsi_pauli_fn(model, params, stop_gradient=False):
    # make slog and pauli from model.apply
    if stop_gradient:
        params = lax.stop_gradient(params)
    def logpsi_n_pauli(*args):
        if hasattr(model, "slog_and_pauli"):
            (sign, logf), pauli = model.apply(params, *args,
                                              method="slog_and_pauli")
        else:
            sign, logf = model.apply(params, *args)
            pauli = model.apply(params, *args, method="pauli")
        if jnp.iscomplexobj(sign):
            logf += jnp.log(sign)
        return logf, pauli
    return logpsi_n_pauli


def build_eval_local_elec(model, elems, nuclei, cell=None, *,
                          ke_kwargs=None, pe_kwargs=None,
                          ext_pots=None, grad_pots=None,
                          spin_pots=None, soc_pots=None,
                          soc2_pots=None, stop_gradient=True):
    """create a function that evaluates local energy, sign and log abs of wavefunction.

    Args:
        model (nn.Module): a flax module that calculates the sign and log abs of wavefunction.
            `model.apply` should have signature (params, x) -> (sign(f(x)), log|f(x)|)
        elems (Array): the element indices (charges) of those nuclei.
        nuclei (Array): the position of nuclei.
        cell (Optional Array): if not None, using ewald summation for potential energy in PBC

    Returns:
        Callable with signature (params, x) -> (eloc, sign, logf) that evaluates
        local energy, sign and log abs of wavefunctions on given parameters and configurations.
    """

    ke_fn, ke_kdict = parse_ke_kwargs(ke_kwargs, quantum_nuclei=False, with_grads=grad_pots)
    pe_fn, pe_kdict = parse_pe_kwargs(pe_kwargs, elems=elems, nuclei=nuclei, cell=cell)
    extpot_fns = parse_ext_pots(ext_pots, elems=elems, nuclei=nuclei, cell=cell)
    gradpot_fns = parse_grad_pots(grad_pots, elems=elems, nuclei=nuclei, cell=cell)
    spinpot_fns = parse_spin_pots(spin_pots, elems=elems, nuclei=nuclei, cell=cell)
    socpot_fns = parse_soc_pots(soc_pots, elems=elems, nuclei=nuclei, cell=cell)
    soc2pot_fns = parse_soc_pots(soc2_pots, elems=elems, nuclei=nuclei, cell=cell)

    def eval_local(params, x: ElecConf):
        sign, logf = model.apply(params, x) # may be overwritten by slog_and_pauli
        logpsi_fn = get_logpsi_fn(model.apply, params,
                                    stop_gradient=stop_gradient)
        x_c, x_s = split_spin(x) # coords and spins, x_s can be None
        logpsi_partial = lambda x_c: logpsi_fn(attach_spin(x_c, x_s))
        ke_n_grad = ke_fn(logpsi_partial, x_c)
        ke = ke_n_grad[0] if isinstance(ke_n_grad, (tuple, list)) else ke_n_grad
        pe = pe_fn(elems, nuclei, x_c)
        ene_comps = {
            "e_kin": ke, "e_coul": pe,
            **{f"e_{name}": fn(x_c) for name, fn in extpot_fns.items()}
        }
        if gradpot_fns:
            grads = (ke_n_grad[1] if isinstance(ke_n_grad, (tuple, list))
                     else adaptive_grad(logpsi_partial)(x_c))
            ene_comps.update({f"e_{name}": (fn(x_c) * grads).sum()
                              for name, fn in gradpot_fns.items()})
        if spinpot_fns:
            if hasattr(model, "slog_and_pauli"): # fast path
                (sign, logf), pauli = model.apply(params, x, method="slog_and_pauli")
            else: # calculate separately, may be slower
                pauli = model.apply(params, x, method="pauli")
            ene_comps.update({f"e_{name}": (fn(x_c) * pauli).sum()
                              for name, fn in spinpot_fns.items()})
        if socpot_fns:
            logpsi_pauli_fn = get_logpsi_pauli_fn(model, params,
                                                  stop_gradient=stop_gradient)
            _partition = ke_kdict.get("partition_size", 1)
            soc_base = calc_soc_base(logpsi_pauli_fn, x, partition_size=_partition)
            ene_comps.update({f"e_{name}": (fn(x_c) * soc_base).sum()
                              for name, fn in socpot_fns.items()})
        eloc = jtu.tree_reduce(jnp.add, ene_comps, 0.0)
        if stop_gradient:
            ene_comps = jtu.tree_map(lax.stop_gradient, ene_comps)
            eloc = lax.stop_gradient(eloc)
        extras = {**ene_comps} # for now only log energy components
        return eloc, sign, logf, extras

    def eval_local_soc(params, x: ElecConf):
        sign, logf = model.apply(params, x)
        logpsi_pauli_fn = get_logpsi_pauli_fn(model, params,
                                              stop_gradient=stop_gradient)
        x_c, x_s = split_spin(x) # coords and spins, x_s can be None
        if soc2pot_fns:
            ke, grads, pauli, soc_base, spin_lapl = calc_spin_lapl(
                logpsi_pauli_fn, x, **ke_kdict)
            soc2_ene = {f"e_{name}": (fn(x_c) * spin_lapl).sum()
                        for name, fn in soc2pot_fns.items()}
        else:
            ke, grads, pauli, soc_base = calc_ke_and_soc(
                logpsi_pauli_fn, x, **ke_kdict)
            soc2_ene = {}
        pe = pe_fn(elems, nuclei, x_c)
        ext_ene = {f"e_{name}": fn(x_c)
                   for name, fn in extpot_fns.items()}
        grad_ene = {f"e_{name}": (fn(x_c) * grads).sum()
                    for name, fn in gradpot_fns.items()}
        spin_ene = {f"e_{name}": (fn(x_c) * pauli).sum()
                    for name, fn in spinpot_fns.items()}
        soc_ene = {f"e_{name}": (fn(x_c) * soc_base).sum()
                   for name, fn in socpot_fns.items()}
        ene_comps = dict(
            e_kin=ke, e_coul=pe,
            **ext_ene, **grad_ene, **spin_ene, **soc_ene, **soc2_ene)
        eloc = jtu.tree_reduce(jnp.add, ene_comps, 0.0)
        if stop_gradient:
            ene_comps = jtu.tree_map(lax.stop_gradient, ene_comps)
            eloc = lax.stop_gradient(eloc)
        extras = {**ene_comps}
        return eloc, sign, logf, extras

    fast_ke_soc = soc_pots and not callable(ke_kwargs) and ke_kdict.get("forward_mode", True)
    if (fast_ke_soc or soc2_pots):
        assert not callable(ke_kwargs), "soc2 cannot be used with custom ke function"
        return eval_local_soc
    return eval_local


def get_batched_local(eval_local_fn, mini_batch=None,
                      checkpoint=True, unroll_loop=False):
    if not mini_batch:
        return jax.vmap(eval_local_fn, in_axes=(None, 0), out_axes=0)
    def batch_local(params, data):
        partial_local = jax.vmap(partial(eval_local_fn, params), 0, 0)
        if checkpoint:
            partial_local = jax.checkpoint(partial_local, prevent_cse=False)
        stack_data = jtu.tree_map(lambda x: x.reshape(
            x.shape[0] // mini_batch, mini_batch, *x.shape[1:]), data)
        if not unroll_loop:
            stack_res = lax.map(partial_local, stack_data)
            res = jtu.tree_map(lambda x: x.reshape(-1, *x.shape[2:]), stack_res)
        else: # manually unroll the map to prevent a bug in compiling
            res_list = []
            n_iter, = set(jtu.tree_leaves(
                jtu.tree_map(lambda a:a.shape[0], stack_data)))
            for ii in range(n_iter):
                slice_data = jtu.tree_map(lambda a, ii=ii: a[ii], stack_data)
                res_list.append(partial_local(slice_data))
            res = jtu.tree_map(lambda *a: jnp.concatenate(a, axis=0), *res_list)
        return res
    return batch_local

def build_eval_energy(eval_local_fn, mini_batch=None, checkpoint=True,
                      unroll_loop=False, pmap_axis_name=PMAP_AXIS_NAME):
    paxis = PmapAxis(pmap_axis_name)
    batch_local = get_batched_local(eval_local_fn, mini_batch, checkpoint, unroll_loop)

    def eval_energy(params, data):
        conf, _ = data
        eloc, sign, logf, extras = batch_local(params, conf)
        etot = paxis.all_nanmean(eloc)
        var_e = paxis.all_nanmean(jnp.abs(eloc - etot)**2)
        aux = dict(e_tot=etot.real, var_e=var_e, std_e=jnp.sqrt(var_e),
                   nans=jnp.isnan(eloc).sum())
        for k, v in extras.items():
            aux[k] = paxis.all_nanmean(v).real
        return aux

    return eval_energy



################## below are parsers for hamiltonian terms ##################

def parse_ke_kwargs(ke_kwargs, quantum_nuclei=False, with_grads=False):
    """parse the ke_kwargs and build ke function

    the returned ke_fn takes log_psi and electron coords as input and
    return the kinetic energy. If `quantum_nuclei` is True, ke_fn takes
    log_psi, nuclei mass, nuclei coords and electron coords.
    """
    if callable(ke_kwargs):
        return ke_kwargs, {}
    ke_kwargs = dict(ke_kwargs or {})
    calc_ke = calc_ke_full if quantum_nuclei else calc_ke_elec
    ke_fn = partial(calc_ke, **ke_kwargs, with_grads=with_grads)
    return ke_fn, ke_kwargs


def parse_pe_kwargs(pe_kwargs, **sysinfo):
    """parse the pe_kwargs and build pe function

    the returned pe_fn takes element indices, nuclei coords and electron coords
    as input and return the potential energy.
    """
    if callable(pe_kwargs):
        return pe_kwargs, {}
    pe_kwargs = dict(pe_kwargs or {})
    scaling = pe_kwargs.pop("scaling", None)
    if scaling is not None:
        raw_pe_fn, pe_kwargs = parse_pe_kwargs(pe_kwargs, **sysinfo)
        pe_fn = lambda *args: scaling * raw_pe_fn(*args)
        return pe_fn, pe_kwargs
    cell = sysinfo.get("cell", None)
    screened = pe_kwargs.pop("screened", False)
    if cell is None:
        pe_fn = partial(calc_pe, **pe_kwargs)
    elif not screened:
        pe_fn = EwaldSum(cell, **pe_kwargs).calc_pe
    else:
        pe_fn = EwaldSumScreened(cell, **pe_kwargs).calc_pe
    return pe_fn, pe_kwargs


def parse_ext_pots(ext_pots, **sysinfo):
    """parse the ext_pots dict into a dict of functions

    external potentials (`ext_pots`) are functions that take electron
    configurations as input and return a scalar of the value of the potential.
    """
    ext_pots = ext_pots or {}
    parsed = {}
    for name, ext in ext_pots.items():
        if not callable(ext):
            raise ValueError(f'External potential {name!r} must be callable')
        parsed[name] = ext
    return parsed


def parse_grad_pots(grad_pots, **sysinfo):
    """parse the grad_pots dict into a dict of functions

    grad potentials (`grad_pots`) are functions that take electron
    coordinates as input and output a (n_elec, n_dim) array V_ia of the potential,
    corresponding to the potential on electron i and spatial dimention a.
    """
    grad_pots = grad_pots or {}
    parsed = {}
    for name, ext in grad_pots.items():
        if not callable(ext):
            raise ValueError(f'Gradient potential {name!r} must be callable')
        # 1-body potential, vmap over electrons
        if name.lower().endswith(("1b", "onebody")):
            ext = jax.vmap(ext, 0, 0)
        parsed[name] = ext
    return parsed


def parse_spin_pots(spin_pots, **sysinfo):
    """parse the spin_pots dict into a dict of functions

    spin potentials (`spin_pots`) are functions that take electron
    coordinates as input and output a (n_elec, 3) array V_is of the potential,
    corresponding to the potential on electron i and pauli component s.
    """
    spin_pots = spin_pots or {}
    parsed = {}
    for name, ext in spin_pots.items():
        if not callable(ext):
            raise ValueError(f'Spin potential {name!r} must be callable')
        # 1-body spin potential, vmap over electrons
        if name.lower().endswith(("1b", "onebody")):
            ext = jax.vmap(ext, 0, 0)
        parsed[name] = ext
    return parsed


def parse_soc_pots(soc_pots, **sysinfo):
    r"""parse the soc_pots dict into a dict of functions

    spin-orbit-coupling potentials (`soc_pots`) are functions that take electron
    coordinates as input and output a (n_elec, 3, n_dim) array V^s_{i,a},
    corresponding to the coeff on electron i and pauli component s and spatial direction a.
    The operators to be estimated is (V^s_{i,a} sigma^s_i \partial_{i,a} \psi) / \psi.
    """
    soc_pots = soc_pots or {}
    parsed = {}
    for name, ext in soc_pots.items():
        if not callable(ext):
            raise ValueError(f'SOC potential {name!r} must be callable')
        # 1-body soc potential, vmap over electrons
        if name.lower().endswith(("1b", "onebody")):
            ext = jax.vmap(ext, 0, 0)
        parsed[name] = ext
    return parsed
