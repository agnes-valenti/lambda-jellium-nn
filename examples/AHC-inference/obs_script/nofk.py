#!/usr/bin/env python3
import os
import numpy as np
import jax
import jax.numpy as jnp
from importlib import import_module
from typing import Callable
from fvmc import observable as obs
from fvmc.wavefunction.heg import heg_rs
from fvmc.utils import load_pickle_tolerant, attach_spin

## Agnes: added for convenience to import runconfig from parent dir
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../')))
##


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
    parser = obs.default_argparse()
    parser.add_argument('--get_config', type=str, default='runconfig.get_runconfig',
        help="extract ansatz from config; get_config(nelec: int, rs: float)")
    parser.add_argument('--fchk', type=str, default='checkpoint.pkl')
    parser.add_argument('--tx', type=float, default=0.0)
    parser.add_argument('--ty', type=float, default=0.0)
    parser.add_argument('--kcut', type=float)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    fchk = args.fchk
    kcut = args.kcut
    twist = jnp.array([args.tx, args.ty])
    key = jax.random.PRNGKey(args.seed)

    # set up folder to cache results
    tname = os.path.basename(args.ftraj)
    cache_dir = os.path.join(os.path.dirname(args.ftraj),
                             'cache-' + tname[:tname.rfind('.')] + '-obs')

    # read metadata
    meta_sys = obs.read_meta(args.fyml)
    cell = jnp.asarray(meta_sys['system']['cell'])
    nelec = meta_sys['system']['nelec']
    spins = meta_sys['system']['spins']
    print('spins = ', spins)
    meta_sys = dict(spins=list(spins), cell=cell.tolist())

    # derived metadata
    ndim = len(cell)
    rs = heg_rs(cell, nelec)
    if kcut is None:
        nspin = len(spins)
        kf = 2./rs/nspin**0.5
        kcut = 4*kf

    # read trajectories
    traj, straj = obs.read_traj(args.ftraj, ndim, nelec)
    traj = obs.reshape_traj(traj[args.iter:args.jter],
        args.batch_size, max_batch=args.max_batch)
    if straj.size == 0:
        straj = None
    else:
        straj = obs.reshape_traj(straj[args.iter:args.jter],
            args.batch_size, max_batch=args.max_batch)
    

    get_config = import_function(args.get_config)
    cfg = get_config(nelec, rs)#, twist)
    ansatz = cfg.ansatz
    # read ansatz parameters 
    _, params, _ = load_pickle_tolerant(fchk)

    # calculate observables
    calc_nofk = obs.gen_calc_nofk(cell, kcut, twist, nelec, ansatz, params, key)
    meta_nofk, nkm, nke = obs.calc_obs(attach_spin(traj, straj), calc_nofk)
    meta = dict(aname='nofk', kvecs=meta_nofk['kvecs'].tolist())
    meta.update(meta_sys)
    obs.save_obs('%s/nofk' % cache_dir, meta, nkm, nke)

if __name__ == '__main__':
     main()  # set no global variable
