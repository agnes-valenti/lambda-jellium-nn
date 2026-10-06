import os
from functools import partial
from typing import NamedTuple, Optional, Sequence, Tuple

import jax
import numpy as np
from flax import linen as nn
from jax import numpy as jnp
from jax import tree_util as jtu
from ml_collections import ConfigDict

from . import LOGGER
from .estimator import (build_eval_local_elec,
                         build_eval_energy)
from .sampler import (build_conf_init_fn, build_sampler, make_batched,
                      make_multistep)
from .utils import (PAXIS, Array, ArrayTree, Printer, PyTree, adaptive_split,
                    backup_if_exist, cfg_to_yaml, load_pickle,
                    multi_process_name, save_checkpoint)
from .wavefunction import (log_prob_from_model, log_psi_from_model)


class SysInfo(NamedTuple):
    spins: Sequence[int]
    elems: Sequence[int]
    nuclei: Array
    cell: Optional[Array] = None


class InferState(NamedTuple):
    key: Array
    params: ArrayTree
    mc_state: ArrayTree


def trim_infer_state(infer_state):
    key, params, mc_state = infer_state
    if key.ndim > 1: # pmapped
        params = jtu.tree_map(lambda x: x[0], params)
    return InferState(key, params, mc_state)


def match_loaded_state_to_device(infer_state):
    n_local_device = jax.local_device_count()
    key, params, mc_state = infer_state
    n_paxis = key.ndim - 1
    if n_paxis == 0: # from single device
        return infer_state
    else: # from multi device
        key = key[0]
        mc_state = jtu.tree_map(lambda x: x.reshape(-1, *x.shape[2:]), mc_state)
    return InferState(key, params, mc_state)


def parse_system_cfg(system_cfg):
    system_cfg = ConfigDict(system_cfg)
    # optional cell
    cell = system_cfg.get("cell", None)
    if cell is not None:
        cell = jnp.asarray(cell)
        if cell.ndim == 1:
            cell = jnp.diag(cell)
    # no nuclei case, HEG, has to be in pbc
    assert cell is not None
    space_dim = cell.shape[-1]
    nuclei = jnp.zeros((0, space_dim), dtype=float)
    elems = jnp.zeros((0,), dtype=int)
    
    tot_elec = int(sum(elems) - system_cfg.charge)
    spin = system_cfg.spin if system_cfg.spin is not None else tot_elec % 2
    assert (tot_elec - spin) % 2 == 0, \
        f"system with {tot_elec} electrons cannot have spin {spin}"
    n_elec = (tot_elec + spin) // 2, (tot_elec - spin) // 2
    system = SysInfo(n_elec, elems, nuclei, cell)
    return system


def parse_ansatz_cfg(ansatz_cfg, system):
    n_elec, elems, nuclei, cell = system
    assert all(callable(getattr(ansatz_cfg, _f, None)) for _f in ('init', 'apply')), \
        "cfg.ansatz must be a constructed flax module, not a hyperparameter dict"
    return ansatz_cfg

def load_ansatz_params(cfg_params):
    if isinstance(cfg_params, str):
        LOGGER.info("Loading parameters from saved file")
        param_path = multi_process_name(cfg_params)
        params = load_pickle(param_path)
    else:
        LOGGER.info("Restart from parameters in config")
        params = cfg_params
    if isinstance(params, tuple):
        params = params[1]
    if isinstance(params, ConfigDict): # happen to be converted
        params = params.to_dict()
    return params


def load_sampler_state(cfg_chains):
    if isinstance(cfg_chains, str):
        LOGGER.info("Loading MC states from saved file")
        chains_path = multi_process_name(cfg_chains)
        mc_state = load_pickle(chains_path)
    else:
        LOGGER.info("Restart from MC states in config")
        mc_state = cfg_chains
    if len(mc_state) in (3,4) and isinstance(mc_state[1], dict):
        mc_state = mc_state[2] # assuming TrainingState
    return mc_state


def prepare(system_cfg, ansatz_cfg, sample_cfg, loss_cfg, 
             key=None, restart_cfg=None):
    """prepare system, ansatz, sampler, optimizer and training state"""

    # handle multi device settings
    n_device = 1
    rng_split = partial(adaptive_split, multi_device=False)
    LOGGER.info("local device: %d, global devices: %d, process id: %d",
        jax.local_device_count(), n_device, jax.process_index())

    # make sure all cfg are ConfigDict
    system_cfg, sample_cfg, restart_cfg= \
        map(ConfigDict, (system_cfg, sample_cfg, restart_cfg))

    # parse system, may be changed (e.g. using pyscf mol)
    system = parse_system_cfg(system_cfg)
    n_elec, elems, nuclei, cell = system

    # make wavefunction
    ansatz = parse_ansatz_cfg(ansatz_cfg, system)
    log_prob_fn = log_prob_from_model(ansatz)
    log_psi_fn = log_psi_from_model(ansatz)

    # make estimators
    loss_cfg = dict(loss_cfg) # so that we can pop
    ke_kwargs = loss_cfg.pop("ke_kwargs", {})
    pe_kwargs = loss_cfg.pop("pe_kwargs", {})
    ext_pots = system_cfg.get("external_potentials", {})
    grad_pots = system_cfg.get("gradient_potentials", {})
    spin_pots = system_cfg.get("spin_potentials", {})
    soc_pots = system_cfg.get("soc_potentials", {})
    soc2_pots = system_cfg.get("soc2_potentials", {})
    lclargs = dict(ke_kwargs=ke_kwargs, pe_kwargs=pe_kwargs,
                   ext_pots=ext_pots, grad_pots=grad_pots,
                   spin_pots=spin_pots, soc_pots=soc_pots,
                   soc2_pots=soc2_pots, stop_gradient=True)
    local_fn = (build_eval_local_elec(ansatz, elems, nuclei, cell, **lclargs))
    #loss_fn = build_eval_total(local_fn,
    #    pmap_axis_name=PAXIS.name, **loss_cfg)
    
    eval_energy = jax.jit(build_eval_energy(local_fn, pmap_axis_name=PAXIS.name, **loss_cfg))

    # make sampler
    n_sample = sample_cfg.size
    n_chain = sample_cfg.chains or n_sample
    if n_sample % n_chain != 0:
        LOGGER.warning("Sample size not divisible by batch size, rounding up")
    n_multistep = -(-n_sample // n_chain)
    n_batch = n_chain // n_device
    conf_init_fn = sample_cfg.get("conf_init_fn", build_conf_init_fn(
        elems, nuclei, sum(n_elec), with_r=False))
    raw_sampler = build_sampler(
        log_prob_fn,
        conf_init_fn,
        name=sample_cfg.sampler,
        adaptive=sample_cfg.get("adaptive", None),
        **sample_cfg.get(sample_cfg.sampler, {}))
    sampler = make_multistep(raw_sampler, n_step=n_multistep, concat=False)
    sampler = make_batched(sampler, n_batch=n_batch, concat=True)
    sampler = jtu.tree_map(jax.jit, sampler)


    # make infer states
    if "states" in restart_cfg and restart_cfg.states:
        if isinstance(restart_cfg.states, str):
            LOGGER.info("Loading parameters and states from saved file")
            state_path = multi_process_name(restart_cfg.states)
            train_state = InferState(*load_pickle(state_path))
        else:
            LOGGER.info("Restart from parameters and states in config")
            train_state = InferState(*restart_cfg.states)
        train_state = match_loaded_state_to_device(train_state, multi_device)
    else:
        LOGGER.info("Initializing parameters and states")
        assert key is not None, \
            "key is required if not restarting from previous state"
        # initialize params
        if "params" in restart_cfg and restart_cfg.params:
            params = load_ansatz_params(restart_cfg.params)
        else:
            key, parkey, skey = jax.random.split(key, 3) # init use single key
            fake_input = conf_init_fn(skey)
            params = ansatz.init(parkey, fake_input)
        key, mckey, optkey, statekey = rng_split(key, 4)
        # initialize mc state
        if "chains" in restart_cfg and restart_cfg.chains:
            mc_state = load_sampler_state(restart_cfg.chains)
            mc_state = sampler.refresh(mc_state, params)
        else:
            mc_state = sampler.init(mckey, params)
        if "burn_in" in sample_cfg and sample_cfg.burn_in > 0:
            LOGGER.info(f"Burning in the sampler for {sample_cfg.burn_in} steps")
            key, subkey = rng_split(key)
            mc_state = sampler.burn_in(subkey, params, mc_state, sample_cfg.burn_in)
        key, subkey = rng_split(key)
        mc_state, init_data, _ = sampler.sample(subkey, params, mc_state)
        # initialize opt state
        # assemble training state
        infer_state = InferState(statekey, params, mc_state)

    return system, ansatz, sampler, eval_energy, infer_state


def build_infer_step(eval_energy, sampler):
    """generate a training loop step function from sampler and optimizer"""

    rng_split = partial(adaptive_split, multi_device=False)

    def infer_step(infer_state):
        key, params, mc_state = infer_state
        key, mckey= rng_split(key, 2)
        mc_state, data, mc_info = sampler.sample(mckey, params, mc_state)
        aux = eval_energy(params, data)
        mc_state = sampler.refresh(mc_state, params)
        return InferState(key, params, mc_state), (mc_info, aux), data

    return infer_step


def run(step_fn, infer_state, iterations, log_cfg):
    """run the optimization loop (sample + update)"""

    log_cfg = ConfigDict(log_cfg)

    use_tensorboard = log_cfg.get("use_tensorboard", False)
    if use_tensorboard:
        try:
            from tensorboardX import SummaryWriter
        except ImportError:
            LOGGER.warning("no tensorboardx, set use_tensorboard = False")
            use_tensorboard = False

    # writer and tensorboard tracker
    if jax.process_index() == 0:
        backup_if_exist(log_cfg.stat_path, prefix="bck",
                        max_keep=log_cfg.get("stat_keep"))
        writer = open(log_cfg.stat_path, "w", buffering=1)
        if use_tensorboard:
            tracker = SummaryWriter(log_cfg.tracker_path)
    # training log printer
    print_fields = {"step": "",
                    "e_tot": ".4f", "std_e": ".3e",
                    "acc": ".2f", "hacc": ".2f"}
    printer = Printer(print_fields, time_format=".2f")
    # sample trajectory dumper
    dumper = None
    if log_cfg.dump_every > 0 and log_cfg.dump_path:
        from npy_append_array import NpyAppendArray
        dump_path = multi_process_name(log_cfg.dump_path)
        backup_if_exist(dump_path, prefix="bck",
                        max_keep=log_cfg.get("dump_keep"))
        assert not os.path.exists(dump_path)
        dumper = NpyAppendArray(dump_path, delete_if_exists=True)
    # determine chekcpoint path for potential multi device
    ckpt_path = multi_process_name(log_cfg.ckpt_path)

    # mysterious step to prevent kfac memory error
    #infer_state = jtu.tree_map(jnp.copy, infer_state)

    LOGGER.info("Start training")
    if jax.process_index() == 0:
        printer.print_header("# ")

    for ii in range(iterations):
        printer.reset_timer()

        # main training loop
        infer_state, (mc_info, aux), sample_data = step_fn(infer_state)

        # nan check
        if not jtu.tree_all(
          jtu.tree_map(lambda a: jnp.all(~jnp.isnan(a)), infer_state.params)):
            raise ValueError(f"NaN found in params at step {ii}")
        if jnp.any(aux["nans"] > 0):
            LOGGER.warning("%d NaN(s) found in local energy at step %d",
                           aux["nans"].sum(), ii)

        # log stats
        if ((ii % log_cfg.stat_every == 0 or ii == iterations-1)
          and jax.process_index() == 0): # only print for process 0 (all same)
            acc_rate = PAXIS.all_mean(mc_info["is_accepted"])
            hacc_rate = 1. / PAXIS.all_mean(mc_info["recip_ratio"])
            stat_dict = {"step": ii, **aux,
                         "acc": acc_rate, "hacc": hacc_rate}
            if "tuned_hparam" in mc_info:
                stat_dict["mc_hp"] = PAXIS.all_mean(mc_info["tuned_hparam"])
            stat_dict = {k: v[0] if jnp.ndim(v) > 0 else v # collect from potential pmap
                         for k, v in stat_dict.items()}
            printer.print_fields(stat_dict)
            np.savetxt(writer, np.array(list(stat_dict.values())).reshape(1,-1),
                        header=" ".join(stat_dict.keys()) if ii == 0 else "")
            if use_tensorboard:
                for k,v in stat_dict.items():
                    if k != "step":
                        tracker.add_scalar(k, v, stat_dict["step"])

        # dump traj
        if dumper is not None and ii % log_cfg.dump_every == 0:
            sconf, slogw = sample_data
            flat_conf = np.concatenate([
                np.asarray(sc).reshape(slogw.size, -1)
                for sc in jtu.tree_leaves(sconf)
            ], axis=-1)[None]
            dumper.append(flat_conf) # [n_step, n_batch, n_coord]

        # checkpoint
        if ii % log_cfg.ckpt_every == 0 or ii == iterations-1:
            save_checkpoint(
                ckpt_path,
                tuple(trim_infer_state(infer_state)),
                max_keep=log_cfg.get("ckpt_keep"))

    if jax.process_index() == 0:
        writer.close()
        if use_tensorboard:
            tracker.close()
    if dumper is not None:
        dumper.close()

    return infer_state


def save_cfg_to_yaml(cfg, path):
    # add git info
    import subprocess as sp
    try:
        cfg._git_hash = sp.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=os.path.dirname(os.path.abspath(__file__))
        ).decode().strip()
    except sp.CalledProcessError:
        cfg._git_hash = None
    # save to yaml
    if path:
        with open(path, "w") as hpfile:
            print(cfg_to_yaml(cfg), file=hpfile)
    else:
        LOGGER.warning("No path to save hyperparameters")


def main(cfg):
    cfg = ConfigDict(cfg)

    import logging
    verbosity = getattr(logging, cfg.verbosity.upper())
    LOGGER.setLevel(verbosity)

    if jax.process_index() == 0:
        save_cfg_to_yaml(cfg, cfg.log.get("hpar_path", None))

    key = jax.random.PRNGKey(cfg.seed) if 'seed' in cfg else None
    system, ansatz, sampler, eval_energy_fn, infer_state \
        = prepare(cfg.system, cfg.ansatz, cfg.sample, cfg.loss,
                   key, cfg.restart)


    infer_step = build_infer_step(eval_energy_fn, sampler)
    infer_state = run(infer_step, infer_state, cfg.iterations, cfg.log)

    return infer_state
