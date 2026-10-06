import functools
from typing import Optional, Sequence, Tuple, Callable

import jax
from flax import linen as nn
from jax import numpy as jnp

from ..utils import (Array, DenseComplex, ElecConf, _t_real,
                     ensure_spin, split_spin, attach_spin,
                     fix_init, gen_ksphere_adaptive, log_linear_exp, pdist)
from ..ewaldsum import gen_pbc_disp_fn
from .base import ElecWfn
from .neuralnet_pbc import dist_features_pbc, raw_features_pbc
from .heg import ElecProductModel, block_diagonal_masks, heg_rs, JastrowU2Bspline, BackflowEtaBessel, bessel, polynomial_envelope

def apply_backflow(backflow: Callable, x: ElecConf, spinBF = False):
    _x, _s = split_spin(x)
    if not spinBF:
        x = _x # backflow cannot depend on s
    out = backflow(x)
    if isinstance(out, tuple):
        x_bf, (sign, logf) = out
    else:
        x_bf = out
        sign, logf = 1., 0.
    if not isinstance(x_bf, tuple):
        x_bf = attach_spin(x_bf, _s)
    return x_bf, (sign, logf)
        

class ElecProductModelSpin(ElecProductModel):
    """Product wavefunction that pauli matrices can act on"""
    overcomplete_spin: bool = True # Whether to use overcomplete (continuous) basis for spin. If True, only a single spin dependent model is allowed.
    spinBF: bool = False # Whether to apply spin dependent backflow

    def __call__(self, x: ElecConf) -> Tuple[Array, Array]:
        sign, logf = 1., 0.
        x_bf = x
        if self.backflow is not None:
            x_bf, (sign_bf, logf_bf) = apply_backflow(self.backflow, x, self.spinBF)
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
    
    def call_with_phases(self, x: ElecConf, phases: Array) -> Tuple[Array, Array]:
        sign, logf = 1., 0.
        x_bf = x
        if self.backflow is not None:
            x_bf, (sign_bf, logf_bf) = apply_backflow(self.backflow, x, self.spinBF)
            sign *= sign_bf
            logf += logf_bf
        for model, with_bf in zip(self.submodels, self.bf_mask):
            if hasattr(model, "call_with_phases"):
               result = model.call_with_phases(x_bf if with_bf else x, phases)
            else:
               result = model(x_bf if with_bf else x)
            if isinstance(result, tuple):
                sign *= result[0]
                logf += result[1]
            else:
                logf += result
        return sign, logf
    

    def pauli(self, x: ElecConf) -> Array:
        """apply Pauli matrices (for each electron) to the wavefunction

        Return an Array of shape (n_elec, 3) corresponding to
        sigma^i_{x|y|z} psi(r,s) / psi(r,s) for each electron i.
        To use this method, one and only one of the submodels should have
        the method `pauli` implemented.
        """
        if self.spinBF: #slow pauli calculation for spin dependent backflow
            assert not self.overcomplete_spin, "Overcomplete spin basis is not supported with spin dependent backflow"
            _, pauli = self.slow_slog_and_pauli(x)
        else:
            spin_models = self.get_spin_submodels()
            pauli = jnp.ones((x[0].shape[0], 3))
            if self.backflow is not None:
                x_bf, _ = apply_backflow(self.backflow, x)
            else:
                x_bf = x
            for model, with_bf in zip(self.submodels, self.bf_mask):
                if model not in spin_models:
                    continue
                x_m = x_bf if with_bf else x
                model_pauli = model.pauli(x_m)
                pauli *= model_pauli
        return pauli
        

    def slog_and_pauli(self, x: ElecConf) -> Tuple[Tuple[Array, Array], Array]:
        """return the sign and log of the wavefunction, and the Pauli matrices

        See `pauli` method for details. This method ensures the backflow is
        called only once (although jax complier should have figured it out).
        """
        if self.spinBF: 
            assert not self.overcomplete_spin, "Overcomplete spin basis is not supported with spin dependent backflow"
            return self.slow_slog_and_pauli(x)
        else: 
            return self.fast_slog_and_pauli(x)

    def slow_slog_and_pauli(self, x: ElecConf) -> Tuple[Tuple[Array, Array], Array]:
        x, s = split_spin(x)
        flipped_s =jnp.repeat(s[None,:],s.shape[0],axis=0)
        flipped_s = flipped_s + 0.5*jnp.eye(s.shape[0])
        flipped_x = jnp.repeat(x[None,:,:],x.shape[0],axis=0)
        sign, logf = self.__call__((x,s))
        sign_flipped, logf_flipped = jax.vmap(self.__call__)((flipped_x,flipped_s))

        pauli_z =  jnp.sign(s%1-.5)
        pauli_x =  jnp.exp(logf_flipped-logf) * sign_flipped/sign
        pauli_y = -1j*pauli_z*pauli_x
        return (sign, logf), jnp.stack((pauli_x,pauli_y,pauli_z),axis=-1)
    
    def fast_slog_and_pauli(self, x: ElecConf) -> Tuple[Tuple[Array, Array], Array]:
        sign, logf = 1., 0.
        pauli = jnp.ones((x[0].shape[0], 3))
        x_bf = x
        # backflow, cannot depend on s
        if self.backflow is not None:
            x_bf, (sign_bf, logf_bf) = apply_backflow(self.backflow, x)
            sign *= sign_bf
            logf += logf_bf
        # iter over models
        spin_models = self.get_spin_submodels() # one and only one
        for model, with_bf in zip(self.submodels, self.bf_mask):
            x_m = x_bf if with_bf else x
            if model in spin_models: # calculate Pauli with sign and logf
                if hasattr(model, "slog_and_pauli"): # potential fast path
                    result, model_pauli = model.slog_and_pauli(x_m)
                else: # slow path, but make sure pauli is calculated
                    result = model(x_m)
                    model_pauli = model.pauli(x_m)
                pauli *= model_pauli
            else: # normal model without pauli
                result = model(x_m)
            if isinstance(result, tuple):
                sign *= result[0]
                logf += result[1]
            else:
                logf += result
        pauli = self.pauli(x)
        return (sign, logf), pauli

    @nn.nowrap
    def get_spin_submodels(self) -> Optional[nn.Module]:
        """return the index and the submodel that has the method `pauli`"""
        spin_models = [m for m in self.submodels if hasattr(m, "pauli")]
        if self.overcomplete_spin and len(spin_models) != 1:
            raise ValueError("Exactly one submodel should have the method `pauli`")
        # return spin_models[0] if self.overcomplete_spin else spin_models
        return spin_models
    


class PlanewaveSlaterSpin(ElecWfn):
    kvecs: Array
    multi_det: Optional[int] = None
    init_scale: float = 1e-2
    complex_coeff: bool = False
    k_band: Optional[Array] = None
    overcomplete_spin: bool = True # Whether to use overcomplete (continuous) basis for spin. 
                                   # If False, uses discrete (up, down) basis and allows for multiple spin modules.
    spinor_trans: Optional[nn.Module] = None # A many-body dependent transformation of the spinors. 

    @nn.nowrap
    @classmethod
    def from_cell(
        cls, cell: Array,
        n_k: Optional[int] = None, close_shell: bool = True,
        k_cut: Optional[float] = None, margin: float = 0.2,
        twist: Optional[Array] = None, **kwargs
    ) -> 'PlanewaveSlaterSpin':
        kvecs = gen_ksphere_adaptive(cell, n_k, close_shell, k_cut, margin, twist)
        return cls(kvecs=kvecs, **kwargs)

    def setup(self) -> None:
        kwargs = dict(
            kvecs=self.kvecs,
            init_scale=self.init_scale, complex_coeff=self.complex_coeff, k_band=self.k_band, spinor_trans=self.spinor_trans)
        self._is_multi_det = self.multi_det and self.multi_det > 1
        if not self._is_multi_det:
            self.orbital_fn = PlanewaveOrbitalSpin(**kwargs)
        else:
            self.orbital_fn = nn.vmap(
                PlanewaveOrbitalSpin,
                in_axes=None, out_axes=0,
                axis_size=self.multi_det,
                variable_axes={"params": 0},
                split_rngs={"params": True},
            )(**kwargs)
            self.det_weights = self.param(
                "det_weights", nn.initializers.ones, (self.multi_det,))

    @nn.nowrap
    def _sbasis(self, s):
        # return a spin basis that has shape [n_elec, n_comp]
        # for the current case, use e^{+-is} for 2 components (spin up/down) in the overcomplete case
        # and (1,0) or (0,1) for [int-0.5,int] or [int,int+0.5] correspondingly in the discrete case
        if self.overcomplete_spin:
            return jnp.exp(1j * jnp.array([1, -1]) * s[:, None])
        else:
            s=s%1-.5
            return (jnp.sign(s)[:, None]* jnp.array([1, -1]) + jnp.array([1, 1]))/2
        
    @nn.nowrap
    def _apply_phases(self, phases):
        return jnp.exp(1j * jnp.array([0, 1]) * phases[:, None])
            
    def __call__(self, x: ElecConf) -> Tuple[Array, Array]:
        x, s = ensure_spin(x)
        spin_orbs = self.orbital_fn(x) # not a list, since no spin up/dn separation
        return self._slog_from_so(spin_orbs, s)
    
    def call_with_phases(self, x: ElecConf, phases: Array) -> Tuple[Array, Array]:
        x, s = ensure_spin(x)
        spin_orbs = self.orbital_fn(x)
        return self._slog_from_so_phases(spin_orbs, s, phases)

    def _slog_from_so(self, spin_orbs: Array, s: Array) -> Tuple[Array, Array]:
        # contract spin basis
        orbs = jnp.einsum("...ijs,is->...ij", spin_orbs, self._sbasis(s))
        # calculate determinants
        sign, logf = jnp.linalg.slogdet(orbs)
        # handle potential multi-determinant
        if self._is_multi_det:
            sign, logf = log_linear_exp(sign, logf, self.det_weights, axis=0)
        return sign, logf
    
    def _slog_from_so_phases(self, spin_orbs: Array, s: Array, phases: Array) -> Tuple[Array, Array]:
        # contract spin basis
        orbs = jnp.einsum("...ijs,is,is->...ij", spin_orbs, self._sbasis(s), self._apply_phases(phases))
        # calculate determinants
        sign, logf = jnp.linalg.slogdet(orbs)
        # handle potential multi-determinant
        if self._is_multi_det:
            sign, logf = log_linear_exp(sign, logf, self.det_weights, axis=0)
        return sign, logf

    def pauli(self, x: ElecConf) -> Array:
        """apply Pauli matrices (for each electron) to the wavefunction

        Return an Array of shape (n_elec, 3) corresponding to
        sigma^i_{x|y|z} psi(r,s) / psi(r,s) for each electron i,
        with last axis being the components, 0: x, 1: y, 2: z.
        """
        x, s = ensure_spin(x)
        spin_orbs = self.orbital_fn(x)
        return self._pauli_from_so(spin_orbs, s)

    def _pauli_from_so(self, spin_orbs: Array, s: Array) -> Array:
        sb0 = self._sbasis(s)
        sbx = jnp.flip(sb0, axis=-1) # (e^-is, e^is)
        sby = jnp.array([1j, -1j]) * sbx # (ie^-is, -ie^is)
        sbz = jnp.array([1, -1]) * sb0 # (e^is, -e^-is)
        psbas = jnp.stack((sbx, sby, sbz), axis=-1) # (n_elec, 2, 3)
        mat0 = jnp.einsum("...ijs,is->...ij", spin_orbs, sb0)
        pmats = jnp.einsum("...ijs,isp->...ijp", spin_orbs, psbas)
        row_diffs = pmats - mat0[..., None]
        # calculate Pauli matrices
        pauli_fn = (det_row_update if not self._is_multi_det else
                    functools.partial(multi_det_row_update,
                                      det_weights=self.det_weights))
        batch_pauli_fn = jax.vmap(pauli_fn, in_axes=(None, -1), out_axes=-1)
        return batch_pauli_fn(mat0, row_diffs)

    def slog_and_pauli(self, x: ElecConf) -> Tuple[Tuple[Array, Array], Array]:
        x, s = ensure_spin(x)
        spin_orbs = self.orbital_fn(x)
        return self._slog_from_so(spin_orbs, s), self._pauli_from_so(spin_orbs, s)


class PlanewaveOrbitalSpin(nn.Module):
    kvecs: Array
    init_scale: float = 1e-2
    complex_coeff: bool = False
    k_band: Optional[Array] = None
    spinor_trans: Optional[nn.Module] = None

    @nn.compact
    def __call__(self, x: Array) -> Sequence[Array]:
        # x is electron coords, spin is not handled here
        assert x.ndim in (2, 3), "x should be [(n_comp,) n_elec, n_dim]"
        n_elec, n_spacedim = x.shape[-2:]
        n_orb = n_elec
        n_comp = 2 if x.ndim == 2 else x.shape[0]

        if self.k_band is not None:
           assert self.kvecs.shape[0] >= n_orb, "not enough basis!"
        # plane wave
        eikx = jnp.exp(1j * x @ self.kvecs.T) # [n_elec, n_k]
        # dense for coefficients
        PWDense = functools.partial(
            DenseComplex if self.complex_coeff else nn.Dense,
            use_bias=False, param_dtype=_t_real,
            kernel_init=nn.initializers.normal(self.init_scale))
        # full spinor determinant [n_elec, n_orb, n_comp]
        spin_orbs = nn.vmap(
            PWDense,
            in_axes=None if x.ndim == 2 else 0,
            out_axes=-1,
            axis_size=n_comp,
            variable_axes={"params": -1},
            split_rngs={"params": True}
        )(n_orb)(eikx)

        if self.k_band is not None:
            # apply k_band to the coefficients
            assert self.k_band.shape == (n_orb, self.kvecs.shape[1]), "k_band shape mismatch"
            eikbx = jnp.exp(1j * x @ self.k_band.T) # [n_elec, n_orb]
            spin_orbs = jnp.einsum("ij,ijk->ijk", eikbx, spin_orbs) if x.ndim == 2 else jnp.einsum("kij,ijk->ijk", eikbx, spin_orbs)
        if self.spinor_trans is not None:
            spin_orbs = self.spinor_trans(x,spin_orbs)
        
        return spin_orbs  # spin orbital, multi component at last axis

        


class PairJastrowSpin(ElecWfn):
    cell: Array
    jastrow_u2: nn.Module
    wrap_dist: bool = True
    optimize_cusp: bool = False

    @nn.compact
    def __call__(self, x: ElecConf) -> Tuple[Array, Array]:
        x, _ = split_spin(x)
        if x.ndim == 3:
            x = x.mean(0) # average over spin components for backflowed coords
        # preparation constants
        n_elec, ndim = x.shape
        with jax.ensure_compile_time_eval():
            blkdiag_mask, _ = block_diagonal_masks([n_elec], n_elec, True)
            disp_fn = gen_pbc_disp_fn(self.cell)
        # minimal image distance
        if self.wrap_dist: # sin wrap
            _, _, dist = dist_features_pbc(x, self.cell, frac_dist=False, keepdims=False)
        else: # no wrap
            dist = pdist(x, disp_fn)
        # Jastrow
        cusp = 1 / (ndim - 1) # cusp for anti parallel spins
        if self.optimize_cusp:
            cusp = self.param("cusp", fix_init, cusp, _t_real)
        logpsi = jax.vmap(self.jastrow_u2, (0, None))(dist[blkdiag_mask], cusp).sum()
        return 1, logpsi


class PairSpinDependentJastrow(ElecWfn):
    cell: Array
    jastrow_u2_spin: nn.Module #return (2,2) matrix corresponding to the spin dependent 2 body jastrow (see _sidx for index convention)
    wrap_dist: bool = True
    optimize_cusp: bool = False
    mass_anisotropy: Optional[float] = 0 # spin mass anisotropy - the coefficeint of the term \sigma_z*\grad^2

    
    def __call__(self, x: ElecConf) -> Tuple[Array, Array]:
        x, s = split_spin(x)
        n_elec, ndim = x.shape
        logJ2 = self.pair_jastrows(x) 
        return self._slog_from_pair_matrices(logJ2, s, n_elec)

    def _slog_from_pair_matrices(self, logJ2: Array, s: Array, n_elec: int) -> Tuple[Array, Array]:
        sidx = self._sidx(s)
        s_pairs=sidx[jnp.indices((n_elec,n_elec)),0].T# [n_elec, n_elec, 2]
        with jax.ensure_compile_time_eval():
            blkdiag_mask, _ = block_diagonal_masks([n_elec], n_elec, True)
        # s_pairs =s_pairs[blkdiag_mask,:]
        return 1, logJ2[jnp.arange(logJ2.shape[0]),s_pairs[blkdiag_mask,0],s_pairs[blkdiag_mask,1]].sum()

    def pauli(self, x: ElecConf) -> Array:
        """apply Pauli matrices (for each electron) to the wavefunction

        Return an Array of shape (n_elec, 3) corresponding to
        exp[\sum_j sigma^i_{x|y|z} J(s_i,s_j) / J(s_i,s_j)] for each electron i,
        with last axis being the components, 0: x, 1: y, 2: z.
        """
        x, s = split_spin(x)
        n_elec, ndim = x.shape
        logJ2 = self.pair_jastrows(x)

        return self.pauli_from_j2(logJ2, s, n_elec)

    def slog_and_pauli(self, x: ElecConf) -> Tuple[Tuple[Array, Array], Array]:
        x, s = split_spin(x)
        n_elec, ndim = x.shape
        logJ2 = self.pair_jastrows(x) 
        
        return self._slog_from_pair_matrices(logJ2, s, n_elec), self.pauli_from_j2(logJ2, s, n_elec)


    def pauli_from_j2(self, logJ2: Array, s: Array, n_elec: int) -> Array:
        """ 
        sigma^i_z acts trivially on the spin dependent Jastrow, while sigma^i_{x|y} both act as a flip of the i spin index. 
        The way to see it is by considering the more general form of spin dependent Jastrow - a 4x4 matrix acting on the bi-spinor J(s_i,s_j;s'_i,s'_j) on which the pauli matrices act as conjugation (\sigma * J * \sigma^-1). 
        In this formulation the Jastrow we consider is in fact diagonal and therefore the sigma^i_z acts trivially.
        """
        # sidx = self._sidx(s)[:,0]
        # sidx_flipped = 1-sidx
        sb0 = self._sbasis(s)
        sbx = jnp.flip(sb0, axis=-1)
        with jax.ensure_compile_time_eval():
            blkdiag_mask, _ = block_diagonal_masks([n_elec], n_elec, True)
            
        logJ2 = jnp.zeros([n_elec,n_elec,2,2]).at[blkdiag_mask,:,:].set(logJ2) #reshape logJ2 to [n_elec,n_elec,2,2] lower triangular matrix in the first two indices
        logJ2 = logJ2 + jnp.moveaxis(logJ2,0,1) #symmetrize logJ2
        # logJ2 = logJ2[:,jnp.arange(n_elec),:,sidx] # fix the second spin index[n_elec, n_elec, 2]
        logJ20 = jnp.einsum('...ijsq,is,jq->...ij',logJ2,sb0,sb0)
        logJ2X = jnp.einsum('...ijsq,is,jq->...ij',logJ2,sbx,sb0)
        Jastrow_diff=jnp.exp(jnp.sum(logJ2X-logJ20,axis=-1))
        # Jastrow_diff=jnp.exp((logJ2[jnp.arange(n_elec),:,sidx_flipped]-logJ2[jnp.arange(n_elec),:,sidx]).sum(axis=1))
        return jnp.stack((Jastrow_diff, Jastrow_diff, jnp.ones(n_elec)), axis=-1)
    
    @nn.compact
    def pair_jastrows(self, x: ElecConf) -> Array:
        """computes the pair-wise Jastrow matrices in the computational spin basis J(s_i,s_j)
        
        Returns: an Array of shape (n_elec, n_elec, 2, 2) corresponding to the pair-wise Jastrow matrices"""
        # preparation constants
        n_elec, ndim = x.shape
        with jax.ensure_compile_time_eval():
            blkdiag_mask, _ = block_diagonal_masks([n_elec], n_elec, True)
            disp_fn = gen_pbc_disp_fn(self.cell)
        # minimal image distance
        if self.wrap_dist: # sin wrap
            _, _, dist = dist_features_pbc(x, self.cell, frac_dist=False, keepdims=False)
        else: # no wrap
            dist = pdist(x, disp_fn)
        # Jastrow
        mean_mass = 1/(1-self.mass_anisotropy**2)
        cusp = mean_mass / 3 # cusp for parallel spins
        cusp_z = self.mass_anisotropy*mean_mass/ 3 # cusp for parallel spins
        cusp_anti = 1 # cusp for anti parallel spins (given by harmonic mean of the two masses)
        if self.optimize_cusp:
            cusp = self.param("cusp", fix_init, cusp, _t_real)
            cusp_z = self.param("cusp_z", fix_init, cusp_z, _t_real)
            cusp_anti = self.param("cusp_anti", fix_init, cusp_anti, _t_real)
        return jax.vmap(self.jastrow_u2_spin, (0, None, None, None))(dist[blkdiag_mask], cusp, cusp_z, cusp_anti)

    @nn.nowrap
    def _sidx(self, s):
        # return a spin basis index that has shape [n_elec] (0 for [1,0]] and 1 for [0,1])
        s=s%1-.5
        return jnp.int16(jnp.heaviside(-s[:, None], 0))
        
    
    @nn.nowrap
    def _sbasis(self, s):
        # return a spin basis that has shape [n_elec, n_comp]
        sidx = self._sidx(s)
        return jnp.array([[1, 0], [0, 1]])[sidx[:,0]]
        # s=s%1-.5
        # return (jnp.sign(s)[:, None]* jnp.array([1, -1]) + jnp.array([1, 1]))/2
        


class JastrowU2BsplineSpinfull(nn.Module):
    n_params: int
    rcut: float
    init_scale: Optional[float] = 0.1
    nonhomogenous_init_scale: Optional[float] = 1e-2
    
    @nn.compact
    def __call__(self, r: float, cusp: float, cusp_z: float, cusp_anti: float) -> Array:
        J_homo=JastrowU2Bspline(self.n_params, self.rcut, self.init_scale)(r, cusp)
        J_z=JastrowU2Bspline(self.n_params, self.rcut, self.nonhomogenous_init_scale)(r, cusp_z)
        J_offdiag=JastrowU2Bspline(self.n_params, self.rcut, self.nonhomogenous_init_scale)(r, cusp_anti-cusp)
        
        return J_homo*jnp.ones((2,2)) + J_offdiag*jnp.array([[0,1],[1,0]])+ J_z*jnp.array([[1,0],[0,-1]])
        
        

class PairBackflowSpin(nn.Module):
    spins: Sequence[int]
    cell: Array
    backflow_eta: nn.Module
    backflow_eta_a: Optional[nn.Module] = None
    spinBF: Optional[bool] = False

    @nn.compact
    def __call__(self, x: ElecConf) -> ElecConf:
        x, s = split_spin(x)
        n = x.shape[-2]
        with jax.ensure_compile_time_eval():
            latvec = self.cell
            rs = heg_rs(latvec, n)
            spins = self.spins
            blkdiag_mask, offdiag_mask = block_diagonal_masks(spins, n, False)
            eta_para = self.backflow_eta  # parallel spin
            if len(spins) > 1:  # antiparallel spin, default to same as parallel
              eta_anti = self.backflow_eta_a or eta_para
        # displacements
        _, d_hsin, dist = dist_features_pbc(x, latvec, frac_dist=False, keepdims=False)
        if isinstance(d_hsin, tuple):
            d_hsin = d_hsin[0]
        x_by_rs = dist / rs
        # !!!! inefficient construction of eta matrix
        if self.spinBF:
            eta = eta_para(x_by_rs, s) * blkdiag_mask * (1 - jnp.eye(n))
        else:
            eta = eta_para(x_by_rs) * blkdiag_mask * (1 - jnp.eye(n))
        if len(self.spins) > 1:
            if self.spinBF:
                eta = eta + eta_anti(x_by_rs, s) * offdiag_mask
            else:
                eta = eta + eta_anti(x_by_rs) * offdiag_mask
        dr = jnp.einsum('...ij,ijl->...jl', eta, -d_hsin)
        return x + dr #attach_spin(x + dr, s)

def det_row_update(mat, row_diffs):
    """batched version for updating a determinant by a row

    Args:
        mat (Array): a square matrix with shape (n, n)
        row_diffs (Array): same shape as mat, each row is the update at that row

    Returns:
        det(A') / det(A), the scale factors from updating those n rows, shape (n,)
    """
    # det(A + u @ v^T) = (1 + v^T @ A^-1 @ u) * det(A)
    # u: one hot on updated row index, v^T: updated row
    # v^T @ A^-1 @ u is the n-th col of v^T @ A^-1 for electron n
    inv = jnp.linalg.inv(mat) # square matrix, "ij" in einsum
    # u is just \delta_jn and is pre-contracted to inv0, ij,jn->in
    updates = 1 + jnp.einsum("ni,in->n", row_diffs, inv)
    return updates


def multi_det_row_update(mat, row_diffs, det_weights):
    """batched version updating sum of multiple determinant by same single row

    Args:
        mat (Array): multiple square matrix with shape (m, n, n)
        row_diffs (Array): same shape as mat, each row is the update at that row

    Returns:
        scale factors from updating sum of determinants by those n rows, shape (n,)
    """
    # multi-determinant, do the same thing for each determinant
    updates = jax.vmap(det_row_update, in_axes=0, out_axes=0)(mat, row_diffs)
    msign, mlogf = jnp.linalg.slogdet(mat) # both shape (m,)
    sign0, logf0 = log_linear_exp(msign, mlogf, det_weights, axis=0) # scalar
    uweights = det_weights[:, None] * updates # shape (m, n)
    signu, logfu= log_linear_exp(msign, mlogf, uweights, axis=0) # (n,)
    return signu / sign0 * jnp.exp(logfu - logf0)
