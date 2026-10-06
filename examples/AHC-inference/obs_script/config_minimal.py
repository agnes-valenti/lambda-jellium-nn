import jax
from jax import numpy as jnp
import numpy as np

import fvmc
from fvmc.utils import gen_ksphere_adaptive
from fvmc.wavefunction.base import FrozenModel
from fvmc.wavefunction.heg import JastrowU2Bspline, BackflowEtaBessel
from fvmc.wavefunction.hegspin import PlanewaveSlaterSpin, ElecProductModelSpin, PairJastrowSpin, PairSpinDependentJastrow, JastrowU2BsplineSpinfull, PairBackflowSpin #, NeuralJastrowSpin
from fvmc.wavefunction.hegnnspin import NeuralBackflowSpin

import deserialize
import os
import sys


# The published runs were produced by two generations of the training config,
# which differ in how the orbital k-set of the translation-invariant (liq)
# state is built.  `liquid_kset` selects between them:
#   'kcut'  -> gen_ksphere_adaptive(cell, k_cut=3*kf)   e.g. 163 kvecs at Ne=36
#   'shell' -> gen_ksphere_adaptive(cell, n_k=9*nelec)  e.g. 145 kvecs at Ne=16
# The crystal states (ahc/twc/hwc) use the Gmn set from the HF file, which is
# the same for every electron number, so they are unaffected.
CONVENTIONS = {
    'ne36': {'liquid_kset': 'kcut'},
    'ne16': {'liquid_kset': 'shell'},
}


def get_jastrow_coeffs_from_checkpoint(filename):
  fchk = filename
  if os.path.isfile(fchk):
    
    from fvmc.utils import load_pickle_tolerant
    _, params, _ = load_pickle_tolerant(fchk)
    params = params['params']
    if 'submodels_1' in params:
      params_submodels1=params['submodels_1']
      return params_submodels1
    else:
      print("No jastrow coefficients found in checkpoint")
      return None
  else:
    print('no checkpoint "%s"' % fchk)
    return None

def get_orbital_coeffs_from_checkpoint(filename):
  fchk = filename
  if os.path.isfile(fchk):
    from fvmc.utils import load_pickle_tolerant
    _, params, _ = load_pickle_tolerant(fchk)
    params = params['params']
    if 'submodels_0' in params:
      params_submodels0=params['submodels_0']['orbital_fn']['VmapDenseComplex_0']['kernel']
      return params_submodels0
    else:
      print("No orbital coefficients found in checkpoint")
      return None
  else:
    print('no checkpoint "%s"' % fchk)
    return None

def get_backflow_coeffs_from_checkpoint(filename):
  fchk = filename
  if os.path.isfile(fchk):
    from fvmc.utils import load_pickle_tolerant
    _, params, _ = load_pickle_tolerant(fchk)
    params = params['params']
    if 'backflow' in params:
      params_backflow=params['backflow']
      return params_backflow
    else:
      print("No backflow coefficients found in checkpoint")
      return None
  else:
    print('no checkpoint "%s"' % fchk)
    return None

# from qharv.inspect import axes_pos
# from qharv.cross import pwscf, pwscf_xml, pwscf_hdf
def rwsc(axes, dn=1):
  """ radius of the inscribed sphere inside the real-space
  Wigner-Seitz cell of the given cell

  Args:
    axes (np.array): lattice vectors in row-major
    dn (int,optional): number of image cells to search in each
     dimension, default dn=1 searches 26 images in 3D.
  Returns:
    float: Wigner-Seitz cell radius
  """
  ndim = len(axes)
  from itertools import product
  r2imgl  = []  # keep a list of distance^2 to all neighboring images
  images = product(range(-dn, dn+1), repeat=ndim)
  for ushift in images:
    if sum(ushift) == 0:
      continue  # ignore self
    shift = np.dot(ushift, axes)
    r2imgl.append(np.dot(shift, shift))
  rimg = np.sqrt(min(r2imgl))
  return rimg/2.

def get_kvecs(file, init_state):
  # fwd_file = "TwoBandMinimal_SCHF_rs_20_lambda_0.6666666666666666_Nb_13_Nk1_4_relqcut_2.5_distance_Inf_Delta_0.1_arrayID_20_0000.h5"#TwoBandMinimal_SCHF_rs_1.0_lambda_1_Nb_13_Nk1_3_relqcut_2.5.h5" #"TwoBandMinimal_SCHF_rs_30_lambda_0_Nb_13_Nk1_4_relqcut_3_distance_100.00000.h5"  #"TwoBandMinimal_SCHF_rs_1.0_lambda_1_Nb_13_Nk1_3_relqcut_2.5.h5" #"twobandrs2.0_3.h5" #"small_example.h5"
  schcf = deserialize.SCHFData(file, init_state)
   
  rs = schcf.rs #/(2*np.sqrt(2)) #*np.sqrt(2.0)
  #print("rs: ", rs)
  
  Delta = schcf.delta/2.0
  Lambda_s = schcf.lambda_s*rs
  dgate = None #schcf.gate_distance*rs

  k_band_vals = schcf.Ks_in_BZ/rs
  k_band_vals = np.reshape(k_band_vals, (k_band_vals.shape[0]*k_band_vals.shape[1], 2))
  kvecs = schcf.Gmn/rs
  g1 = schcf.G1/rs
  g2 = schcf.G2/rs

  L = 2*np.pi/g1[0]
  W = 2*np.pi/g2[0]
  assert np.isclose(L, W)

  nx = schcf.Nk[0]
  ny = schcf.Nk[1]
  nelec = nx*ny

  a1 = L * np.array([0.5, -np.sqrt(3)/2]) #*nx
  a2 = L * np.array([0.5, np.sqrt(3)/2])#*ny

  wcpos = np.zeros((nx*ny, 2))
  for ix in range(nx):
    for iy in range(ny):
      wcpos[ix*ny+iy] = ix*a1 + iy*a2

  a1full = a1*nx
  a2full = a2*ny

  cell = np.zeros((2,2))
  cell[0,:] = a1full
  cell[1,:] = a2full
  rho = schcf.rho_SCHF_planewavebasis
  nkx = rho.shape[0]
  nky = rho.shape[1]
  ng = rho.shape[2]
  nspins = rho.shape[3]
  rho = np.reshape(rho, (nkx*nky, ng, nspins, ng, nspins))
  rhosquare = np.reshape(rho, (nkx*nky, ng*nspins, ng*nspins))

  coeffs = np.zeros((nkx*nky, ng, nspins)) + 0j
  k_band_vals_orbs = np.zeros((nkx*nky, 2))
  norb_it = 0
  for kval in range(nkx*nky):
    vals, vecs = np.linalg.eigh(rhosquare[kval,:,:])
    vals = np.round(vals).astype(int)
    indices = np.where(vals > 0.5)[0]
    for ind in indices:
        coeffs[norb_it, :, :] = np.reshape(vecs[:,ind], (ng, nspins))
        k_band_vals_orbs[norb_it,:]=k_band_vals[kval,:]
        norb_it += 1

  coeffs = np.transpose(coeffs, (1,0,2))

  gvecs_test = np.zeros((nkx*nky*ng, 2))
  coeffs_test = np.zeros((nkx*nky*ng, nkx*nky, nspins)) + 0j
  for ind in range(nkx*nky):
    coeffs_test[ind*ng:(ind+1)*ng,ind,:] = coeffs[:,ind,:]
    gvecs_test[ind*ng:(ind+1)*ng,:] = kvecs + k_band_vals_orbs[ind]
  return rs, jnp.array(k_band_vals_orbs), jnp.array(kvecs), jnp.array(cell), jnp.array(wcpos), nelec, jnp.array(coeffs), Delta, Lambda_s, gvecs_test, coeffs_test, dgate

def get_config(file_name, mix_init=False, discrete_spin=False, orbital_params = None, jastrow_params=None, backflow_params=None,
                 Delta_factorization=None,  save_ckpt_path = 'checkpoint', stat_path = 'data.txt', mc_chain_checkpoint=None, lambda_s_loaded=None, twist_load=None,
               out_dir='.', convention='ne36', with_spin_bf=False,
               **nncfg):
  key = jax.random.PRNGKey(0)

  if convention not in CONVENTIONS:
    raise ValueError(f'unknown convention {convention!r}; '
                     f'expected one of {sorted(CONVENTIONS)}')
  conv = CONVENTIONS[convention]

  if with_spin_bf and not discrete_spin:
    raise ValueError('a spin dependent backflow requires discrete_spin=True; '
                     'slow_slog_and_pauli asserts not overcomplete_spin')

  checkpoint_backflow = None
  
  if backflow_params is not None:
    if 'checkpoint' in backflow_params:
      checkpoint_backflow = backflow_params['checkpoint']

  wrap_dist_jastrow = False
  checkpoint_jastrow = None
  j_num_knots = 8
  apply_bf_jastrow = False
  assert jastrow_params is not None

  if 'wrap_dist' in jastrow_params:
    wrap_dist_jastrow = jastrow_params['wrap_dist']
  if 'checkpoint' in jastrow_params:
    checkpoint_jastrow = jastrow_params['checkpoint']
  if 'num_knots' in jastrow_params:
    j_num_knots = jastrow_params['num_knots']
  if 'apply_bf' in jastrow_params:
    apply_bf_jastrow = jastrow_params['apply_bf']

  full_translation = False
  checkpoint_orbs = None
  init_state = 7

  assert 'checkpoint' in orbital_params
  checkpoint_orbs = orbital_params['checkpoint']
  if 'full_translation' in orbital_params:
    full_translation = orbital_params['full_translation']
  if 'init_state' in orbital_params:
    init_state = orbital_params['init_state']
    
  coeff_orbs = None 
  coeff_jastrow = None 
  coeff_backflow = None 

  assert checkpoint_orbs is not None, "Checkpoint for orbitals must be provided"
  coeff_orbs = get_orbital_coeffs_from_checkpoint(checkpoint_orbs)
  
  assert checkpoint_jastrow is not None, "Checkpoint for jastrow must be provided"
  coeff_jastrow = get_jastrow_coeffs_from_checkpoint(checkpoint_jastrow)

  assert checkpoint_backflow is not None, "Checkpoint for backflow must be provided"
  coeff_backflow = get_backflow_coeffs_from_checkpoint(checkpoint_backflow)
    
    
  if twist_load is None:
    twist_load = jnp.zeros(2)
  else:
    twist_load = jnp.array(twist_load)
  
  rs, kband_vals, kvecs, cell, wcpos, nelec, coeff, Delta, lambda_s, gvecs_test, coeffs_test, dgate = get_kvecs(file_name, init_state)

  overall_factor = (0.5 + Delta*lambda_s**2/2)*2.0
  np.savetxt(os.path.join(out_dir, "overall_factor.txt"), [overall_factor])

  if lambda_s_loaded is not None:
    lambda_s = lambda_s_loaded*rs
    print("lambda_s loaded: ", lambda_s)

  kf = 2**0.5/rs
  tx = twist_load[0]
  ty = twist_load[1]

  if full_translation:
    # translation-invariant liquid: plane waves on a k-sphere, no band labels
    if conv['liquid_kset'] == 'kcut':
      kvecs = gen_ksphere_adaptive(cell, k_cut=3*kf, twist=(tx, ty))
    else:
      kvecs = gen_ksphere_adaptive(cell, n_k=nelec*9, twist=(tx, ty))
  else:
    recvec = 2*jnp.pi*jnp.linalg.inv(cell).T
    twist = twist_load @ recvec
    kvecs = twist[None, :] + kvecs

  assert len(kvecs) == coeff_orbs.shape[0], (
      f"convention {convention!r} builds {len(kvecs)} orbital k-vectors but the "
      f"checkpoint holds {coeff_orbs.shape[0]}; the checkpoint was made with a "
      f"different convention (see CONVENTIONS above)")

  det = PlanewaveSlaterSpin(kvecs, complex_coeff=True, k_band = (kband_vals if not full_translation else None), overcomplete_spin= not discrete_spin)

  wf = [det]
  apply_bf = [True] #False] #True]
  cutoff = rwsc(cell) #min(cell[0,0], cell[1,1]) #/2.0 #/2.0 #*np.pi #/rs
  if wrap_dist_jastrow:
     cutoff=.7*np.sqrt(np.linalg.det(cell))

  
  mass_anisotropy = - Delta * lambda_s**2 * 0.5
  jmod = JastrowU2BsplineSpinfull(j_num_knots,cutoff)
  jas = PairSpinDependentJastrow(cell, jmod, optimize_cusp=True, mass_anisotropy=mass_anisotropy, wrap_dist=False)
  if wrap_dist_jastrow:
    jas = PairSpinDependentJastrow(cell, jmod, optimize_cusp=True, mass_anisotropy=mass_anisotropy, wrap_dist=True)
    #jas = NeuralJastrowSpin(cell)

  wf.append(jas)
  apply_bf.append(apply_bf_jastrow) #True)

  bf = NeuralBackflowSpin(cell)  #PairBackflowSpin((nelec, ),cell,SpinDepBackflowEtaBessel(20,cutoff), spinBF=True)

  ansatz = ElecProductModelSpin(wf, backflow = bf, overcomplete_spin = not discrete_spin, apply_backflow=apply_bf, spinBF=with_spin_bf)
  
  Sz_avg = np.sum(np.abs(coeff_orbs[:,:,0])**2 - np.abs(coeff_orbs[:,:,1])**2)/np.sum(np.abs(coeff_orbs)**2)
  sig = 2.6*rs**(3./4)
  def conf_init_fn(key):
    key1, key2, key3 = jax.random.split(key, 3)
    coords = ((wcpos if not mix_init else jax.random.permutation(key3, wcpos, axis=0))
             + 0.1 * sig * jax.random.normal(key2, wcpos.shape))
    minval = .25*(1+Sz_avg) if discrete_spin else 0
    maxval = .25*(3+Sz_avg) if discrete_spin else 2 * jnp.pi 
    spins = jax.random.uniform(key1, (coords.shape[0],), minval=minval, maxval=maxval) 
   
    return coords, spins

  cfg = fvmc.config.default()
  # <wf>
  cfg.ansatz = ansatz
  # <hamiltonian>
  #if ext_pot is not None:
  #  cfg.system.external_potentials = ext_pot
  # <project>
  cfg.seed = 42
  cfg.verbosity = "INFO"
  cfg.multi_device = False
  # <system>
  cfg.system.charge = -nelec
  cfg.system.elems = None
  cfg.system.nuclei = None
  cfg.system.cell = cell
  
  ###############
  #lambda_s = 0.1
  vmat = np.zeros((3,2)) + 0j
  vmat[0,0] = 1.0
  vmat[1,1] = 1.0
  vmat = jnp.array(vmat)
  #print("ediag: ",(testdiag * Delta * lambda_s**2 * 0.5 + ekin * Delta * lambda_s**2 + testdiag1 * Delta + testdiag12 * Delta)/9.0*2.0)

  Delta = Delta/overall_factor
  v = 1j * lambda_s * Delta * vmat # in 2D, but pauli matrices is always 3 components
  cfg.system.soc_potentials = {"offdiag1b": lambda x: v}
  cfg.system.soc2_potentials = {"diag1b": lambda x: - Delta * lambda_s**2 * 0.5 * jnp.array([0.0, 0.0, 1.0])}
  #print("ediag rescaled: ",((testdiag * Delta * lambda_s**2 * 0.5 + ekin + testdiag1 * Delta + testdiag12 * Delta)/9.0*2.0)*overall_factor - ekin/9.0*2.0)

  #print("testdiag: ", testdiag * Delta * lambda_s**2 * 0.5)
  
  # (0 .. 1) = (0.5, ... 0.5) - (0.5, ... -0.5) = 0.5*eye - 0.5*sigma_z
  cfg.system.spin_potentials = {"sz1b": lambda x: - 0.5 * Delta * jnp.array([0.0, 0.0, 1.0])} 
  cfg.system.external_potentials = {"const": lambda x: 0.5 * Delta * nelec}

  # <vmc>
  cfg.sample.conf_init_fn = conf_init_fn
  cfg.sample.size = 512
  cfg.sample.burn_in = 1000
  
 
  cfg.sample.sampler = 'mixed_mala'
  cfg.sample.mixed_mala.tau = 0.15 * rs
  cfg.sample.mixed_mala.steps = 20 #at N=9, lambda=1/3, rs=30 the correlation time is 9
  cfg.sample.mixed_mala.grad_clipping = 1.
  cfg.sample.mixed_mala.p_flip = (1-np.abs(Sz_avg)) / nelec

  
  # <optimize>
  cfg.loss.pe_kwargs = {"scaling": 1.0/overall_factor}  #{"screened": True, "dgate": dgate} #{"scaling": (1+Delta*lambda_s**2/2.0)}  #{"screened": True, "dgate": dgate}
  mini_batch = 64
  cfg.loss.mini_batch = mini_batch  #64 if with_bf!!
  cfg.loss.ke_kwargs.forward_mode = True
  cfg.loss.ke_kwargs.partition_size = None
  # None->nelec: fastest & most memory, 1: slowest & least memory

  # <output>
  cfg.log.stat_every = 1
  cfg.log.dump_every = 5
  cfg.log.ckpt_keep = None
  cfg.log.ckpt_every = 300
  cfg.log.use_tensorboard = False
  cfg.log.ckpt_path = save_ckpt_path #+ '.pkl'
  cfg.log.stat_path = stat_path

  
   
  key = jax.random.PRNGKey(0)
  s = jnp.zeros(nelec)


  params = ansatz.init(key, (wcpos,s))

  # a spin dependent backflow widens backflow/Dense_0 and the message passing
  # layer inputs by one (the spin channel), so a plain/spin-BF mix-up shows up
  # here rather than as an opaque shape error deep inside the first apply
  for _name, _ckpt in (('submodels_1', coeff_jastrow), ('backflow', coeff_backflow)):
    _want = jax.tree_util.tree_map(jnp.shape, params['params'][_name])
    _got = jax.tree_util.tree_map(jnp.shape, _ckpt)
    if _want != _got:
      raise ValueError(
          f"{_name} in the checkpoint does not match the ansatz built with "
          f"with_spin_bf={with_spin_bf}; the checkpoint was most likely trained "
          f"with the other setting")

  params['params']['submodels_0']['orbital_fn']['VmapDenseComplex_0']['kernel'] = coeff_orbs
  params['params']['submodels_1'] = coeff_jastrow 
  params['params']['backflow'] = coeff_backflow 
  #insert here
  #params['params']['submodels_0']['orbital_fn']['VmapDenseComplex_0']['kernel'] = mycoeff
  cfg.restart.params = params
  if mc_chain_checkpoint is not None:
    cfg.restart.chains = mc_chain_checkpoint 
    print('load mc sampling checkpoint "%s"' % cfg.restart.chains)
    
  #if freeze_orb:
  #  if with_Jastrow: params['params']['submodels_1'] = None
  #  if with_bf: params['params']['backflow'] = None
  #  ansatz = FrozenModel(ansatz, params)
  #  cfg.ansatz = ansatz
  #  params = ansatz.init(key, (wcpos,s))
  
  return cfg

if __name__ == '__main__':
  fvmc.train.main(cfg)
