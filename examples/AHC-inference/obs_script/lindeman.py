import os
import jax.numpy as jnp
import numpy as np
from importlib import import_module
from typing import Callable
from fvmc import observable as obs
from fvmc.wavefunction.heg import heg_rs
from fvmc.utils import load_pickle, attach_spin

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
    args = parser.parse_args()

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

    #find folded COM and shift to origin
    invvec = jnp.linalg.inv(cell)
    L = jnp.sqrt(nelec) # Number of electrons per cell length. Works only for LxL lattice
    frac_traj = (traj @ invvec) % (1/L)
    offset = jnp.mean(jnp.exp(frac_traj*L*2*jnp.pi*1j), axis=tuple(range(frac_traj.ndim - 1))) 
    offset = jnp.angle(offset)/(L*2*jnp.pi) @ cell
    print("offset: ", offset)
    # Shift origin to COM
    offset_ = jnp.broadcast_to(offset, traj.shape)
    traj = traj - offset_

    disp = (traj @ invvec + 1/(2*L)) % (1/L) - 1/(2*L) #displacement vector relative to closest crystal position
    disp = disp @ cell
    disp2 = jnp.sum(disp**2, axis=-1) #displacement squared relative to closest crystal position

    mean_disp2 = jnp.mean(disp2)
    err_disp2 = jnp.sqrt( jnp.mean((disp2-mean_disp2)**2) / (disp2.size - 1))
    
    rms_disp = mean_disp2**0.5
    rms_disp_err = err_disp2 / (2*rms_disp)

    rms_disp_normalized = rms_disp / (jnp.linalg.norm(cell[0])/L)
    rms_disp_normalized_err = rms_disp_err / (jnp.linalg.norm(cell[0])/L)


    meta = dict(aname='rms_disp')
    meta.update(meta_sys)
    obs.save_obs('%s/rms_disp' % cache_dir, meta, rms_disp, rms_disp_err)
    meta = dict(aname='lind_ratio')
    meta.update(meta_sys)
    obs.save_obs('%s/lind_ratio' % cache_dir, meta, rms_disp_normalized, rms_disp_normalized_err)
    # np.save('%s/ceig.npy' % cache_dir, np.array([jnp.abs(ceig_mean[0]), jnp.angle(ceig_mean[0]) * 180/jnp.pi]))
    print(f'estimated disp: {rms_disp} +/- {rms_disp_err}')
    print(f'estimated ratio: {rms_disp_normalized} +/- {rms_disp_normalized_err}')

if __name__ == '__main__':
    main()  # set no global variable