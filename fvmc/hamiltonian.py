import jax
from jax import lax
from jax import numpy as jnp
from jax.flatten_util import ravel_pytree

from .utils import Array, pdist, cdist, adaptive_grad, split_spin, attach_spin


def calc_coulomb(charge, pos):
    n = pos.shape[-2]
    dist = pdist(pos)
    charge =jnp.atleast_1d(charge)
    cmat = jnp.expand_dims(charge, -1) * jnp.expand_dims(charge, -2)
    coulomb = jnp.triu(cmat / (dist + jnp.eye(n)), k=1)
    return coulomb.sum((-1,-2))


def calc_coulomb_2(charge_a, pos_a, charge_b, pos_b):
    dist = cdist(pos_a, pos_b)
    charge_a, charge_b = map(jnp.atleast_1d, [charge_a, charge_b])
    cmat = jnp.expand_dims(charge_a, -1) * jnp.expand_dims(charge_b, -2)
    coulomb = cmat / dist
    return coulomb.sum((-1,-2))


def calc_pe(elems, r, x):
    # r is nuclei position
    # x is electron positions
    el_el = calc_coulomb(-1, x)
    el_ion = calc_coulomb_2(-1, x, elems, r)
    ion_ion = calc_coulomb(elems, r)
    return el_el + el_ion + ion_ion


def laplacian_over_f(log_f, scale=None, forward_mode=True,
                     partition_size=1, with_grads=False):
    # (\nable^2 f) / f = \nabla^2 log|f| + (\nabla log|f|)^2
    # handle potential forward mode
    from . import LOGGER
    if forward_mode:
        try:
            import fwdlap
        except ImportError:
            LOGGER.warning("fwdlap is not installed, use normal backward mode instead.")
            forward_mode = False

    if with_grads and scale is not None:
        LOGGER.warning("`with_grads` will return scaled grads since `scale` is not None.")

    # handle scale (can be used to set mass of particles)
    scale = ravel_pytree(scale)[0] if scale is not None else jnp.array([1.0])

    # normal bwd-fwd hessian version
    def _lapl_backward(*args):
        flat_in, unravel = ravel_pytree(args)
        ncoord = flat_in.size
        seye = jnp.eye(ncoord) * jnp.diag(scale)
        f = lambda flat_in: log_f(*unravel(flat_in)) # take flattened x
        grad_f = adaptive_grad(f)
        grad_value, dgrad_f = jax.linearize(grad_f, flat_in)
        laplacian = (scale * grad_value**2).sum()
        # different partition size need different implementation
        if partition_size is None:
            hess = jax.vmap(dgrad_f)(seye)
            laplacian += hess.trace()
        elif partition_size == 1:
            loop_fn = lambda i, val: val + dgrad_f(seye[i])[i]
            laplacian += lax.fori_loop(0, ncoord, loop_fn, 0.0)
        else:
            seye = seye.reshape(-1, partition_size, ncoord)
            hess = lax.map(jax.vmap(dgrad_f), seye)
            laplacian += hess.reshape(ncoord, ncoord).trace()
        if not with_grads:
            return laplacian
        else:
            return laplacian, unravel(grad_value * jnp.sqrt(scale))

    # fwd laplacian version
    def _lapl_forward(*args):
        flat_in, unravel = ravel_pytree(args)
        ncoord = flat_in.size
        seye = jnp.eye(ncoord) * jnp.diag(jnp.sqrt(scale))
        zero = fwdlap.zero_tangent_from_primal(flat_in)
        f = lambda flat_in: log_f(*unravel(flat_in)) # take flattened x
        if partition_size is None:
            primals, grads, laps = fwdlap.lap(f, (flat_in,), (seye,), (zero,))
            laplacian = (grads**2).sum() + laps
        else:
            seye = seye.reshape(-1, partition_size, ncoord)
            primals, f_lap_pe = fwdlap.lap_partial(f, (flat_in,), (seye[0],), (zero,))
            inner_fn = lambda d_in: f_lap_pe((d_in,), (zero,))
            raw_grads, raw_laps = lax.map(inner_fn, seye)
            grads = raw_grads.reshape(ncoord)
            laplacian = (grads**2).sum() + raw_laps.sum()
        if not with_grads:
            return laplacian
        else:
            return laplacian, unravel(grads)

    return _lapl_forward if forward_mode else _lapl_backward


def calc_ke_elec(log_psi, x, *, forward_mode=True, partition_size=1, with_grads=False):
    # calc -0.5 * (\nable^2 \psi) / \psi
    # handle batch of x automatically
    lapl_fn = laplacian_over_f(
        log_psi,
        scale=None,
        forward_mode=forward_mode,
        partition_size=partition_size,
        with_grads=with_grads)
    if not with_grads:
        return -0.5 * lapl_fn(x)
    else:
        lap, [grads] = lapl_fn(x)
        return -0.5 * lap, grads


def calc_ke_full(log_psi, mass, r, x, *,
                 forward_mode=True, partition_size=1, with_grads=False):
    # calc -0.5 * (\nable^2 \psi) / \psi
    # handle batch of r, x automatically
    mass = jnp.reshape(mass, -1)
    minv = (jnp.repeat(1/mass, r.shape[-1]), jnp.ones(x.shape[-1] * x.shape[-2]))
    lapl_fn = laplacian_over_f(
        log_psi,
        scale=minv,
        forward_mode=forward_mode,
        partition_size=partition_size,
        with_grads=with_grads)
    if not with_grads:
        return -0.5 * lapl_fn(r, x)
    else:
        lap, grads = lapl_fn(r, x)
        return -0.5 * lap, grads

def calc_soc_base(logp_n_pauli, x, *, partition_size=1):
    # given a function that takes electron confs with spin (a tuple (x, s))
    # and returns the log of the wavefunction psi (can be complex)
    # and the spin term pauli = sigma^i psi / psi (shape: nelec, 3)
    # calculate the base of spin-orbit coupling term
    #  (\nabla_i sigma^i psi) / psi (shape: nelec, 3, ndim)
    # = (\nabla_i pauli) + (\nabla_i log(psi)) * pauli
    x, s = split_spin(x)
    assert x.ndim == 2, "electron coords should be 2D, (nelec, ndim)"
    (logf, pauli), jvp_fn = jax.linearize(
        lambda x_in: logp_n_pauli(attach_spin(x_in, s)), x)
    def inner_fn(d_in):
        d_logf, d_pauli = jvp_fn(d_in) # shape: (1,), (nelec, 3)
        return d_pauli + d_logf * pauli
    vinner_fn = jax.vmap(inner_fn)
    partition_size = partition_size or x.size
    eye = jnp.eye(x.size).reshape(-1, partition_size, *x.shape)
    jacobian = lax.map(vinner_fn, eye) # shape: (-1, partition_size, nelec, 3)
    diag = jacobian.reshape(*x.shape, *pauli.shape).diagonal(axis1=0, axis2=2)
    return diag.transpose(2, 1, 0) # (ndim, 3, nelec) -> (nelec, 3, ndim)


def calc_ke_and_soc(logp_n_pauli, x, *, forward_mode=True, partition_size=1):
    # calculate kinetic energy, pauli term and soc base all together
    assert forward_mode, "forward laplacian is required for fast SOC calculation."
    import fwdlap
    x, s = split_spin(x)
    assert x.ndim == 2, "electron coords should be 2D, (nelec, ndim)"
    wrapped_fn = lambda x_in: logp_n_pauli(attach_spin(x_in, s))
    partition_size = partition_size or x.size
    eye = jnp.eye(x.size).reshape(-1, partition_size, *x.shape)
    zero = fwdlap.zero_tangent_from_primal(x)
    (logf, pauli), lap_fn = fwdlap.lap_partial(wrapped_fn, (x,), (eye[0],), (zero,))
    def inner_fn(d_in):
        (j_log, j_pauli), (l_log, l_pauli) = lap_fn((d_in,), (zero,))
        lapl = (j_log**2).sum() + l_log
        jacp = j_pauli + j_log[:, None, None] * pauli
        return lapl, j_log, jacp
    lapl, jlog, jacp = lax.map(inner_fn, eye)
    ke = -0.5 * lapl.sum()
    grads = jlog.reshape(*x.shape)
    diagp = jacp.reshape(*x.shape, *pauli.shape).diagonal(axis1=0, axis2=2)
    soc_base = diagp.transpose(2, 1, 0) # (ndim, 3, nelec) -> (nelec, 3, ndim)
    return ke, grads, pauli, soc_base


def calc_spin_lapl(logp_n_pauli, x, *, forward_mode=True, partition_size=1):
    # calculate kinetic energy, pauli term, soc_base and spin_laplacian all together
    # spin lapl is defined as (\nabla^2 sigma^i psi) / psi
    # = pauli (\nabla^2 log(psi)) + \nabla^2 pauli
    # + ((\nabla sigma^i psi) / psi + \nabla pauli) \dot (\nabla log(psi))
    assert forward_mode, "forward laplacian is required for fast SOC calculation."
    import fwdlap
    x, s = split_spin(x)
    assert x.ndim == 2, "electron coords should be 2D, (nelec, ndim)"
    wrapped_fn = lambda x_in: logp_n_pauli(attach_spin(x_in, s))
    partition_size = partition_size or x.shape[0] # partition over electrons
    eye = jnp.eye(x.size).reshape(*x.shape, *x.shape)
    zero = fwdlap.zero_tangent_from_primal(x)
    (logf, pauli), lap_fn = fwdlap.lap_partial(wrapped_fn, (x,), (eye[0],), (zero,))
    def per_elec_fn(i_elec):
        # jacobian: j_log: (ndim,), j_pauli: (ndim, nelec, 3)
        # laplacian: l_log: (), l_pauli: (nelec, 3)
        (j_log, j_pauli), (l_log, l_pauli) = lap_fn((eye[i_elec],), (zero,))
        # lapl: (\nabla^2 psi) / psi, scalar
        lapl = (j_log**2).sum() + l_log
        # jo_pauli: (\nabla sigma^i psi) / psi, (ndim, 3)
        jo_pauli = j_pauli[:, i_elec] + j_log[:, None] * pauli[i_elec]
        # lo_pauli: (\nabla^2 sigma^i psi) / psi, (3,)
        lo_pauli = (l_pauli[i_elec] + pauli[i_elec] * l_log
                    + j_log @ (j_pauli[:, i_elec] + jo_pauli))
        return j_log, lapl, jo_pauli, lo_pauli
    # (iter, batch), iterr x batch = nelec
    elec_indices = jnp.arange(x.shape[0]).reshape(-1, partition_size)
    jpsi, lpsi, jpauli, lpauli = lax.map(jax.vmap(per_elec_fn), elec_indices)
    # (iter, batch) -> 1
    ke = -0.5 * lpsi.sum()
    # (iter, batch, ndim) -> (nelec, ndim)
    grads = jpsi.reshape(*x.shape)
    # (iter, batch, ndim, 3) -> (nelec, 3, ndim)
    soc_base = jpauli.reshape(*x.shape, pauli.shape[-1]).transpose(0, 2, 1)
    # (iter, batch, 3) -> (nelec, 3)
    spin_lapl = lpauli.reshape(*pauli.shape)
    return ke, grads, pauli, soc_base, spin_lapl
