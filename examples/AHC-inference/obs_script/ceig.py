#!/usr/bin/env python3
import argparse
import os
import jax.numpy as jnp
import numpy as np
from importlib import import_module
from typing import Callable
from fvmc import observable as obs
from fvmc.wavefunction.heg import heg_rs
from fvmc.utils import load_pickle_tolerant, attach_spin
from vis import bin_centers

import sys
#import runconfig
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../')))


def import_function(path: str) -> Callable:
    """import function at run time

    Args:
      path (str): path to function, e.g., "numpy.linalg.norm"
    Return:
      Callable: imported function, e.g., norm
    """
    mod_path, fn_name = path.rsplit('.', maxsplit=1)
    mod = import_module(mod_path)
    fn = getattr(mod, fn_name)
    return fn

def main():
    parser = obs.default_argparse(6400)
    parser.add_argument('--get_config', type=str, default='runconfig.get_runconfig',
        help="extract ansatz from config; get_config(nelec: int, rs: float)")
    parser.add_argument('--fchk', type=str, default='checkpoint.pkl')
    parser.add_argument('--n_fold', '-N', type=int, default=3)
    parser.add_argument('--AHC', action=argparse.BooleanOptionalAction, default=True,
                        help='shift phases for AHC (use --no-AHC to disable)')
    args = parser.parse_args()
    fchk = args.fchk

    offset = np.array([0.0, 0.0])

    # set up folder to cache results
    tname = os.path.basename(args.ftraj)
    cache_dir = os.path.join(os.path.dirname(args.ftraj),
                             'cache-' + tname[:tname.rfind('.')] + '-obs')

    # read metadata
    meta_sys = obs.read_meta(args.fyml)
    cell = jnp.asarray(meta_sys['system']['cell'])
    nelec = meta_sys['system']['nelec']
    print('nelec = ', nelec)
    meta_sys = dict(cell=cell.tolist())

    # derived metadata
    ndim = len(cell)
    rs = heg_rs(cell, nelec)

    # read trajectories
    traj, straj = obs.read_traj(args.ftraj, ndim, nelec)
    traj = obs.reshape_traj(traj[args.iter:args.jter],
        args.batch_size, max_batch=args.max_batch)
    straj = obs.reshape_traj(straj[args.iter:args.jter],
        args.batch_size, max_batch=args.max_batch)
    if straj.size == 0:
        straj = None

    invvec = jnp.linalg.inv(cell)
    frac_traj = (traj @ invvec) % (1/jnp.sqrt(nelec))
    offset = jnp.mean(jnp.exp(frac_traj*jnp.sqrt(nelec)*2*jnp.pi*1j), axis=tuple(range(frac_traj.ndim - 1))) 
    offset = jnp.angle(offset)/(jnp.sqrt(nelec)*2*jnp.pi) @ cell
    print("Origin of C3 symmetry shifted by: ", offset)

    get_config = import_function(args.get_config)
    cfg = get_config(nelec, rs)#, twist)
    ansatz = cfg.ansatz
    # read ansatz parameters
    _, params, _ = load_pickle_tolerant(fchk)

    # calculate observables
    calc_ceig = obs.gen_calc_ceig(cell, args.n_fold, ansatz, params, offset, shift_phases=args.AHC)

    meta_ceig, ceig_mean, ceig_err = obs.calc_obs(attach_spin(traj, straj), calc_ceig)
    meta = dict(aname='ceig', n_fold=int(meta_ceig['n_fold']))
    meta.update(meta_sys)
    obs.save_obs('%s/ceig' % cache_dir, meta, ceig_mean, ceig_err)
    np.save('%s/ceig.npy' % cache_dir, np.array([jnp.abs(ceig_mean[0]), jnp.angle(ceig_mean[0]) * 180/jnp.pi]))
    print('estimated ceig: ', ceig_mean[0])
    print('abs and angle:', jnp.abs(ceig_mean[0]), jnp.angle(ceig_mean[0]) * 180/jnp.pi)
    print('std between samples: ', ceig_mean[1])
    print('estimatation error: ', ceig_err)

if __name__ == '__main__':
     main()  # set no global variable
