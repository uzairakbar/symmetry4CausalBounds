"""
JAX value+gradient for the last-`l`-layer NLP hot path.

SLSQP finite-differences everything otherwise: at 257+ parameters that is one
constraint sweep per gradient, and the constraint dominates the solve. Analytic
gradients remove it, and the l=2 head (~74k parameters) is unsolvable without them.

Mirrors `partial_r2_net.py::head_index` / `r2_value` / `_extra_value` EXACTLY --
same features, same link, same reductions. The feasibility checks run on the numpy
side and gate a27 compares the two, so a drift surfaces rather than hides.

CPU only: the features are cached and the solves are scipy loops -- a GPU launch
per SLSQP iteration costs more than the evaluation it replaces.
"""

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")  # must precede `import jax`
os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=1")

import jax  # noqa: E402
import numpy as np  # noqa: E402

jax.config.update("jax_enable_x64", True)  # MUST: float32 would cost ~7 digits

import jax.numpy as jnp  # noqa: E402
from jax.scipy.stats import norm as jnorm  # noqa: E402

LINKS = {"probit": jnorm.cdf, "logistic": jax.nn.sigmoid}


def _const(a):
    return jnp.asarray(np.asarray(a, dtype=np.float64))


def _index(theta, phi, shapes):
    """`head_index` mirror: pre-link head output on (n, d) features -> (n,)."""
    a = phi
    offset = 0
    for i, (dout, din) in enumerate(shapes):
        W = theta[offset : offset + dout * din].reshape(dout, din)
        offset += dout * din
        b = theta[offset : offset + dout]
        offset += dout
        a = a @ W.T + b
        if i < len(shapes) - 1:
            a = jax.nn.relu(a)
    return a[:, 0]


def build_terms(model):
    """(objective, r2, extra, gram, band) jitted terms for one fitted model.

    objective(theta, phi_q) is the INDEX at one query -- phi_q is traced, not
    static, so one compile serves every query. r2/extra close over the cached
    constraint features; extra is None for an unconstrained model. gram is the
    constraint-Gram matvec for the deep-head directed starts.

    Args:
        model: a fitted PartialR2Net (or subclass)

    Returns:
        (objective, r2, extra, gram, band): jitted callables; the first three and
        the last are value_and_grad wrt theta, gram maps (theta, v) -> S v.
        `band` is the SIGNED level defect m(theta), None unless the model
        matches means (Lem. 2).
    """
    shapes = tuple(model.head_shapes_)
    link = LINKS[model.link_name]
    phi = _const(model.phi_)
    mu = _const(model.mu_)

    def g(theta, phi_q):
        return _index(theta, phi_q[None, :], shapes)[0]

    objective = jax.jit(jax.value_and_grad(g, argnums=0))

    def r2(theta):
        h = link(_index(theta, phi, shapes))
        return jnp.mean((h - mu) ** 2)

    r2_vg = jax.jit(jax.value_and_grad(r2))

    def h_rows(theta):
        return link(_index(theta, phi, shapes))

    def gram_mv(theta, v):
        """S v = J'(J v)/n for J = dh(rows)/dtheta: the constraint Gram as a
        matvec, for the deep-head directed starts -- the dense S of the
        single-Linear case is unaffordable at ~74k parameters."""
        _, jv = jax.jvp(h_rows, (theta,), (v,))
        _, vjp_fn = jax.vjp(h_rows, theta)
        return vjp_fn(jv)[0] / phi.shape[0]

    gram = jax.jit(gram_mv)

    extra_vg = None
    if getattr(model, "phi_gx_", None) is not None:  # invariance-constrained
        phi_gx = _const(model.phi_gx_)

        def inv(theta):
            d = link(_index(theta, phi, shapes)) - link(_index(theta, phi_gx, shapes))
            return jnp.mean(d**2)

        extra_vg = jax.jit(jax.value_and_grad(inv))
    elif getattr(model, "Qz_", None) is not None:  # IV-constrained
        y, Qz = _const(model.y_), _const(model.Qz_)

        def iv(theta):
            residual = y - link(_index(theta, phi, shapes))
            projected = Qz.T @ residual
            return projected @ projected / len(y)

        extra_vg = jax.jit(jax.value_and_grad(iv))

    # Lem. 2's slice, SIGNED: m(theta) = mean_n h_theta - ybar. The two-sided band
    # |m| <= tau then enters as the PAIR (m <= tau, -m <= tau), both LINEAR in m,
    # rather than the one inequality m^2 <= tau^2. Squaring is what made the
    # encoding ill-conditioned: SLSQP meets a constraint to an absolute tolerance
    # d on the constraint VALUE, and on m^2 that d reappears in |m| as d/(2 tau) --
    # unbounded as the slab thins. Linear in m, d stays d. (`_Negated` supplies the
    # second entry; the augmented-Lagrangian path takes inequalities and so needs
    # no new case either way.)
    band_vg = None
    if getattr(model, "mean_match", False):
        ybar = float(model.ybar_)

        def band(theta):
            return jnp.mean(link(_index(theta, phi, shapes))) - ybar

        band_vg = jax.jit(jax.value_and_grad(band))

    return objective, r2_vg, extra_vg, gram, band_vg
