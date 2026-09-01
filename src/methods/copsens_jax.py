"""
JAX value+gradient for the CopSens NLP hot path.

SLSQP finite-differences everything otherwise: at m = 32 that is 33 objective calls
per gradient, and the constraint dominates the solve. Analytic gradients remove it.

Mirrors `copsens.py::_h_v` / `_constraint` EXACTLY -- same anchors (`anchors_` for the
objective, the coarse `anchors_c_` for the constraint), same clipping. Gate A8 checks
the two against each other.

CPU only: at m = 32 a GPU launch costs more than the whole evaluation.
"""

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")  # must precede `import jax`
os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=1")

import numpy as np  # noqa: E402
import jax  # noqa: E402

jax.config.update("jax_enable_x64", True)  # MUST: float32 would cost ~7 digits

import jax.numpy as jnp  # noqa: E402
from jax.scipy.special import ndtri  # noqa: E402
from jax.scipy.stats import norm as jnorm  # noqa: E402

CLIP = 1e-6


def _kernels(link: str):
    """(g_inv, h_scalar, h_array) for one link. Closures, so `link` stays static."""
    gaussian = link == "gaussian"

    def g_inv(mu, s):
        if gaussian:
            return mu
        return jnp.sqrt(1.0 + s**2) * ndtri(jnp.clip(mu, CLIP, 1.0 - CLIP))

    def h_scalar(d, s):
        """`_h_v` tail: reduce (J,) deviations to a scalar."""
        if gaussian:
            return d.mean()
        return jnorm.cdf(d / jnp.sqrt(1.0 + s * s)).mean()

    def h_array(base, s):
        """`_h` tail: reduce (n, J, c) to (n, c) by averaging over anchors."""
        if gaussian:
            return base.mean(axis=1)
        return jnorm.cdf(base / jnp.sqrt(1.0 + s * s)).mean(axis=1)

    return g_inv, h_scalar, h_array


def _const(a):
    return jnp.asarray(np.asarray(a, dtype=np.float64))


def build_objective(model, radius):
    """value_and_grad of `_h_v` wrt v. (v, mu_q_i, mu_y_i) -> (h, dh/dv).

    mu_q_i / mu_y_i are traced, not static, so ONE compile serves every query.
    """
    g_inv, h_scalar, _ = _kernels(model.link_name)
    halfinv, anchors = _const(model.latent_.halfinv_), _const(model.anchors_)
    r = float(radius)

    def h(v, mu_q, mu_y):
        g = r * (v @ halfinv)
        s = r * jnp.sqrt(v @ v)
        d = g_inv(mu_y, s) + anchors @ g - (mu_q @ g)
        return h_scalar(d, s)

    return jax.jit(jax.value_and_grad(h, argnums=0))


def build_constraint(model, radius):
    """value_and_grad of `_con_v` wrt v, or None if the model is unconstrained.

    Takes the SAME radius the objective got. SOURCE deliberately read `self.gamma`
    here instead, which silently evaluates the constraint on a different ball than
    the objective whenever predict(gamma=...) overrides the budget -- exactly what a
    gamma sweep does.
    """
    m = model.latent_.Sigma_.shape[0]
    if model._constraint(np.zeros((1, m)), 0.0) is None:
        return None

    g_inv, _, h_array = _kernels(model.link_name)
    halfinv, anchors_c = _const(model.latent_.halfinv_), _const(model.anchors_c_)
    r = float(radius)

    def _h(mu_q, Gam, s, mu_y):
        P = anchors_c @ Gam.T  # (J, 1)
        Q = mu_q @ Gam.T  # (n, 1)
        base = g_inv(mu_y, s)[:, None, None] + P[None, :, :] - Q[:, None, :]
        return h_array(base, s)  # (n, 1)

    if hasattr(model, "cGX_"):  # invariance-constrained
        cX, cGX = _const(model.cX_), _const(model.cGX_)
        cmuX, cmuGX = _const(model.cmuX_), _const(model.cmuGX_)

        def con(v):
            Gam = (r * (v @ halfinv))[None, :]
            s = r * jnp.sqrt(v @ v)
            d = _h(cX, Gam, s, cmuX) - _h(cGX, Gam, s, cmuGX)
            return ((d**2).mean(axis=0))[0]
    else:  # IV-constrained
        cX, cmuX = _const(model.cX_), _const(model.cmuX_)
        cy, Qz = _const(model.cy_), _const(model.Qz_)

        def con(v):
            Gam = (r * (v @ halfinv))[None, :]
            s = r * jnp.sqrt(v @ v)
            residual = cy[:, None] - _h(cX, Gam, s, cmuX)  # (n, 1)
            fitted = Qz @ (Qz.T @ residual)
            return ((fitted**2).mean(axis=0) - fitted.mean(axis=0) ** 2)[0]

    return jax.jit(jax.value_and_grad(con))


def build_terms(model, radius):
    """(objective, constraint) jitted value_and_grad pair. The constraint may be None."""
    return build_objective(model, radius), build_constraint(model, radius)
