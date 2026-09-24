import cvxpy as cp
import numpy as np
from loguru import logger

from src.methods.abstract import pointEstimator


def _solve_conic(prob, backend=None):
    """One conic solve: the pinned `(name, opts)` backend when given (the perf
    sweep's seed_var knob, no warm start), else the CLARABEL-then-ECOS chain."""
    if backend is not None:
        name, opts = backend
        prob.solve(solver=name, warm_start=False, **opts)
        return
    try:
        prob.solve(solver=cp.CLARABEL)
    except cp.SolverError:
        logger.warning("CLARABLE solver failed, falling back to ECOS.")
        prob.solve(solver=cp.ECOS)


class LeastSquaresClosedForm(pointEstimator):
    """Closed-form least squares regression.

    `fit_intercept` is what makes this the ERM of Lem. 2's hypothesis class: the
    lemma centres the identified set on argmin over a class closed under constant
    shifts, so under `mean_match` the plotted ERM must carry an intercept too, or
    it is no longer the centre of the ball drawn around it. Default False keeps
    the internal callers (sigma-hat, the PI centre, the floor) on the plain fit
    they already centre themselves.
    """

    def __init__(self, fit_intercept: bool = False):
        self.fit_intercept = fit_intercept
        super().__init__()

    def _fit(self, X, y, **kwargs):
        """Fit using closed-form solution. Ignores extra kwargs."""
        self._mu = X.mean(axis=0) if self.fit_intercept else np.zeros(X.shape[1])
        self._offset = float(np.mean(y)) if self.fit_intercept else 0.0
        self._W = np.linalg.pinv(X - self._mu) @ (np.asarray(y) - self._offset)
        return self

    def _predict(self, X, **kwargs):
        """Predict outcomes. Ignores extra kwargs."""
        return (X - self._mu) @ self._W + self._offset


class LeastSquaresIterative(pointEstimator):
    def __init__(self, backend: tuple[str, dict] | None = None):
        self.backend = backend
        super().__init__()

    def _fit(self, X, y, **kwargs):
        h0 = np.linalg.pinv(X) @ y
        h = cp.Variable(h0.shape)
        cost = cp.norm(y - X @ h)
        prob = cp.Problem(cp.Minimize(cost))
        _solve_conic(prob, self.backend)
        self._W = h.value
        return self

    def _predict(self, X, **kwargs):
        return X @ self._W


class TwoStageLeastSquaresIV(pointEstimator):
    def __init__(self, fit_intercept: bool = False, backend: tuple[str, dict] | None = None, **kwargs):
        self.fit_intercept = fit_intercept
        self.backend = backend  # the second stage's conic solve honours it
        super().__init__(**kwargs)

    def _fit(self, X, y, Z, **kwargs):
        # both stages centre together, so the intercept is eliminated once and
        # restored at predict time (same convention as LeastSquaresClosedForm)
        self._mu = X.mean(axis=0) if self.fit_intercept else np.zeros(X.shape[1])
        offset = float(np.mean(y)) if self.fit_intercept else 0.0
        Zc = Z - Z.mean(axis=0) if self.fit_intercept else Z

        S1 = LeastSquaresClosedForm().fit(Zc, X - self._mu).solution
        Xhat = Zc @ S1

        S2 = LeastSquaresIterative(backend=self.backend).fit(Xhat, np.asarray(y) - offset).solution
        self._W = S2
        self._offset = offset

        return self

    def _predict(self, X, **kwargs):
        return (X - self._mu) @ self._W + self._offset


class MomentConstrainedLeastSquares(pointEstimator):
    """ERM subject to the IV moment pinned at its attainable floor: (P2) at gamma_z = 0.

    The feasible set is the affine set `A h == P b` with `A = Q' Xc`, `b = Q' yc`
    and `Q` a rank-revealing basis of the (centred) instrument. It is exactly the
    ball `|| b - A h || <= m*` at the attainable floor `m* = ||(I - P) b||`, needs
    no tolerance constant, and always contains `pinv(A) b`.
    """

    def __init__(self, fit_intercept: bool = False, backend: tuple[str, dict] | None = None, **kwargs):
        self.fit_intercept = fit_intercept
        self.backend = backend  # the conic stage honours it
        super().__init__(**kwargs)

    def _fit(self, X, y, Z, **kwargs):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        Z = np.asarray(Z, dtype=float).reshape(len(X), -1)

        # both the design and the instrument centre together, so the intercept is
        # eliminated once and restored at predict time (LeastSquaresClosedForm's
        # convention). The constraint is then the DEMEANED moment Zc'(yc - Xc h),
        # which is the right object once an intercept is free: the raw moment's
        # constant component is absorbed by the intercept. (P2)'s mean-match row
        # then holds identically under `fit_intercept`, sum_i [(x_i - mu)'h + ybar]
        # = n ybar, for any h; without one it does not, and is not meant to
        self._mu = X.mean(axis=0) if self.fit_intercept else np.zeros(X.shape[1])
        offset = float(np.mean(y)) if self.fit_intercept else 0.0
        Xc, yc = X - self._mu, y - offset
        Zc = Z - Z.mean(axis=0) if self.fit_intercept else Z

        # an orthonormal basis of the SPAN of the instrument, truncated at the same
        # rank cut the DA-side allowance uses: a plain QR of a rank deficient block
        # completes the basis arbitrarily, and a hard equality would then force the
        # moment to zero along directions the instrument does not span. A constant
        # instrument column centres to exactly zero and is dropped here, which is
        # why the empty branch below has to come after the cut, not before it
        left, singular, _ = np.linalg.svd(Zc, full_matrices=False)
        keep = singular > max(float(singular[0]), 1.0) * 1e-12 if singular.size else singular.astype(bool)
        Q = left[:, keep]

        h_erm = np.linalg.pinv(Xc) @ yc
        if Q.shape[1] == 0:
            # no instrument: the constraint is vacuous and this IS plain ERM.
            # Computed into self, never delegated: every caller keeps the object it
            # built and discards the return, so handing back another estimator would
            # leave _W / _mu / _offset unset and predict would raise
            self._W = h_erm
            self._offset = offset
            return self

        A, b = Q.T @ Xc, Q.T @ yc
        rhs = A @ (np.linalg.pinv(A) @ b)  # P b, the projection onto col(A)
        m_star = float(np.linalg.norm(b - rhs))

        # the variable takes h_erm's shape, so an (n, 1) y keeps an (n, 1)
        # prediction: a bare (n,) one would be read as an interval downstream
        h = cp.Variable(h_erm.shape)
        prob = cp.Problem(cp.Minimize(cp.norm(yc - Xc @ h)), [A @ h == rhs])
        _solve_conic(prob, self.backend)
        self._W = h.value
        self._offset = offset

        # gamma_min: the smallest ERM budget of (P2) whose ellipsoid reaches this
        # set, || Xc (h - h_erm) ||^2 / (n sigma-hat^2) at the returned point
        sigma_sq = float(np.mean((yc - Xc @ h_erm) ** 2))
        gamma_min = float(np.sum((Xc @ (self._W - h_erm)) ** 2) / (len(Xc) * sigma_sq)) if sigma_sq > 0 else 0.0
        logger.info(
            f"ERM+IV: d_z={Zc.shape[1]}, d_h={Xc.shape[1]}, moment floor m*={m_star:.3e}, gamma_min={gamma_min:.6g}"
        )
        # a floor above the numerical zero means the exact-IV set is empty: the
        # instrument over-determines the moment system and this is a RELAXATION,
        # not (P2)'s gamma_z = 0 row. The test is relative; an under- or exactly
        # identified fit leaves m*/||b|| at 1e-16 and an over-identified one at 1e-2
        if m_star > 1e-9 * max(float(np.linalg.norm(b)), 1.0):
            logger.warning(
                f"ERM+IV: moment floor m*={m_star:.3e} with d_z={Zc.shape[1]}, d_h={Xc.shape[1]}: the "
                "exact-IV set is empty and the fit solves the relaxation at that floor."
            )
        return self

    def _predict(self, X, **kwargs):
        return (X - self._mu) @ self._W + self._offset


class GradientDescentERM(pointEstimator):
    """Torch ERM for image treatments. Returns the conditional MEAN, not a label.

    Doubles as the PI outcome model: `predict_mean` + `prefit_` let a net trained
    once on the full draw be handed to every PI variant instead of refitted per method.
    """

    def __init__(self, model: str = "domnist-fast"):
        self.model = model
        super().__init__()

    def _setup(self, X, lr, weight_decay, betas, init_seed):
        """Seed, build the net and its Adam; returns (device, optimizer)."""
        import torch

        from src.methods.nets import NETS, device

        dev = device()
        # Same init_seed => same weight init AND same batch order, so nets differing
        # only in labels are a matched pair. cudnn determinism is REQUIRED, not
        # optional: the nondeterministic drift between two same-seed same-data fits
        # measures 60% of the signal the coupling exists to isolate.
        if init_seed is not None:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            torch.manual_seed(int(init_seed))
            torch.cuda.manual_seed_all(int(init_seed))

        self._input_dim = int(X.shape[1])
        self.f = NETS[self.model](self._input_dim).float().to(dev)
        self.f.train()
        opt = torch.optim.Adam(self.f.parameters(), lr=lr, weight_decay=weight_decay, betas=tuple(betas))
        return dev, opt

    def _criterion(self, loss, label_smoothing):
        """The per-batch fit loss on the net's output."""
        import torch
        import torch.nn.functional as F

        sig = isinstance(self.f[-1], torch.nn.Sigmoid)

        def _loss(p, target):
            t = target * (1 - 2 * label_smoothing) + label_smoothing if sig else target
            # Brier: proper for the MEAN, which is what the PI backend consumes.
            # BCE drives logits to +-inf on noisy labels.
            if not sig or loss == "mse":
                return F.mse_loss(p, t)
            return F.binary_cross_entropy(p, t)

        return _loss

    @staticmethod
    def _scheduler(opt, lr, epochs, n_batches, onecycle):
        import torch

        if not onecycle:
            return None
        return torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=int(epochs) * n_batches)

    def _finish(self):
        self.f.eval()
        self._W = np.concatenate([w.detach().cpu().numpy().ravel() for w in self.f.parameters()])[:, None]
        self.prefit_ = True  # PartialR2Net: reuse instead of refitting
        return self

    def _fit(
        self,
        X,
        y,
        lr=0.01,
        batch=256,
        epochs=1,
        weight_decay=0.0,
        label_smoothing=0.0,
        optimizer="adam",
        loss="mse",
        betas=(0.7, 0.9),
        onecycle=True,
        init_seed=None,
        **kwargs,
    ):
        import torch

        dev, opt = self._setup(X, lr, weight_decay, betas, init_seed)
        Xt = torch.tensor(np.asarray(X), dtype=torch.float, device=dev)
        yt = torch.tensor(np.asarray(y).reshape(-1, 1), dtype=torch.float, device=dev)
        _loss = self._criterion(loss, label_smoothing)

        # data already lives on the device; a DataLoader here is pure overhead
        n_train, n_batches = len(Xt), max(1, int(np.ceil(len(Xt) / batch)))
        sched = self._scheduler(opt, lr, epochs, n_batches, onecycle)

        for _ in range(int(epochs)):
            perm = torch.randperm(n_train, device=dev)
            for b in range(n_batches):
                idx = perm[b * batch : (b + 1) * batch]
                opt.zero_grad(set_to_none=True)
                _loss(self.f(Xt[idx]), yt[idx]).backward()
                opt.step()
                if sched is not None:
                    sched.step()

        del Xt, yt
        return self._finish()

    def _predict(self, X, **kwargs):
        import torch

        X = np.asarray(X)
        dev = next(self.f.parameters()).device
        self.f.eval()
        out = []
        with torch.no_grad():
            for i in range(0, len(X), 8192):  # CNN activations are big
                xb = torch.tensor(X[i : i + 8192], dtype=torch.float, device=dev)
                out.append(self.f(xb).cpu().numpy().ravel())
        # (n, 1), like every other pointEstimator: metrics._as_interval reads a bare
        # (n,) as an interval and silently mis-scores the method
        return np.concatenate(out)[:, None]

    def predict_mean(self, X):
        """Outcome-model protocol; mu_y is a flat (n,)."""
        return self._predict(np.asarray(X).reshape(len(X), -1)).ravel()

    def state_sha1(self) -> str:
        """sha1 over the float32 parameters in `state_dict` order: the net's identity."""
        import hashlib

        digest = hashlib.sha1()  # noqa: S324 (provenance digest, not security)
        for value in self.f.state_dict().values():
            digest.update(np.ascontiguousarray(value.detach().cpu().numpy(), dtype=np.float32).tobytes())
        return digest.hexdigest()

    # -- state round-trip: `self.f` only exists after _fit, so caching needs this --

    def state_blob(self) -> dict:
        """The CPU copy `save_state` writes and `from_blob` reads back (about 1 MB)."""
        return {
            "model": self.model,
            "input_dim": self._input_dim,
            "state": {k: v.detach().cpu() for k, v in self.f.state_dict().items()},
        }

    def save_state(self, path):
        import torch

        torch.save(self.state_blob(), path)
        return path

    @classmethod
    def from_blob(cls, blob):
        from src.methods.nets import NETS, device

        self = cls(blob["model"])
        self._input_dim = int(blob["input_dim"])
        self.f = NETS[self.model](self._input_dim).float().to(device())
        self.f.load_state_dict(blob["state"])
        self.f.eval()
        self._W = np.concatenate([w.detach().cpu().numpy().ravel() for w in self.f.parameters()])[:, None]
        self.prefit_ = True
        return self

    @classmethod
    def load_state(cls, path):
        import torch

        return cls.from_blob(torch.load(path, map_location="cpu", weights_only=True))


class InvariantGradientDescentERM(GradientDescentERM):
    """ERM+INV: the ERM risk minimised subject to invariance on the DA pairs,

        min E[(y - h(X))^2]  s.t.  c = E[(h(X) - h(GX))^2] <= tau,

    by an augmented Lagrangian on `c / tau - 1 <= 0`. Same net, Adam, OneCycle,
    batch order and `init_seed` reseeding as `GradientDescentERM`, so it is a
    matched third member of the ERM / DA+ERM pair; X and GX are row-aligned and
    share one permutation. `c` is read on the net's raw output.

    The multiplier schedule is in epoch units: each epoch's batches are split into
    `al_updates_per_epoch` near-equal windows (`np.array_split`; fewer when an epoch
    has fewer batches), so a fit makes exactly that many updates per epoch at 60k and
    at 1.2M draws.
    At each window end `lam = max(0, lam + mu v)` on the window mean `v`, and `mu`
    grows by `al_growth` (capped at `al_mu_max`) unless the violation fell below a
    quarter of the previous window's. `al_trace_` keeps `(step, c_bar, lam, mu)`
    per window. At tau = 0 the constraint has no interior and the scheme is an
    increasing penalty; tau > 0 is a regular inequality.
    """

    @staticmethod
    def al_window_ends(n_batches: int, al_updates_per_epoch: int) -> list[int]:
        """The 0-based batch indices within an epoch after which the multipliers
        update: the last batch of each of the near-equal windows."""
        windows = np.array_split(np.arange(int(n_batches)), min(max(1, int(al_updates_per_epoch)), int(n_batches)))
        return [int(w[-1]) for w in windows]

    def _fit(
        self,
        X,
        y,
        GX=None,
        lr=0.01,
        batch=256,
        epochs=1,
        weight_decay=0.0,
        label_smoothing=0.0,
        optimizer="adam",
        loss="mse",
        betas=(0.7, 0.9),
        onecycle=True,
        init_seed=None,
        al_tau=4e-4,
        al_mu0=1e-4,
        al_growth=2.0,
        al_updates_per_epoch=20,
        al_mu_max=1e4,
        **kwargs,
    ):
        import torch

        if GX is None:
            raise ValueError("InvariantGradientDescentERM needs the DA pairs GX, row-aligned with X.")
        GX = np.asarray(GX).reshape(len(GX), -1)
        if GX.shape != np.asarray(X).shape:
            raise ValueError(f"GX {GX.shape} is not row-aligned with X {np.asarray(X).shape}.")
        if not al_tau > 0:
            raise ValueError(f"al_tau must be positive; got {al_tau!r}.")
        tau = float(al_tau)

        dev, opt = self._setup(X, lr, weight_decay, betas, init_seed)
        Xt = torch.tensor(np.asarray(X), dtype=torch.float, device=dev)
        GXt = torch.tensor(GX, dtype=torch.float, device=dev)
        yt = torch.tensor(np.asarray(y).reshape(-1, 1), dtype=torch.float, device=dev)
        _loss = self._criterion(loss, label_smoothing)

        n_train, n_batches = len(Xt), max(1, int(np.ceil(len(Xt) / batch)))
        sched = self._scheduler(opt, lr, epochs, n_batches, onecycle)
        ends = set(self.al_window_ends(n_batches, al_updates_per_epoch))

        lam, mu, previous = 0.0, float(al_mu0), np.inf
        running, count, step = torch.zeros((), device=dev), 0, 0
        self.al_trace_ = []
        for _ in range(int(epochs)):
            perm = torch.randperm(n_train, device=dev)
            for b in range(n_batches):
                idx = perm[b * batch : (b + 1) * batch]
                opt.zero_grad(set_to_none=True)
                out = self.f(torch.cat([Xt[idx], GXt[idx]]))
                p, pg = out[: len(idx)], out[len(idx) :]
                c = ((p - pg) ** 2).mean()
                t = torch.clamp(lam + mu * (c / tau - 1.0), min=0.0)
                (_loss(p, yt[idx]) + (t**2 - lam**2) / (2.0 * mu)).backward()
                opt.step()
                if sched is not None:
                    sched.step()
                running += c.detach()
                count += 1
                step += 1
                if b in ends:
                    # one host sync per window, not per step
                    c_bar = float(running) / count
                    violation = c_bar / tau - 1.0
                    lam = max(0.0, lam + mu * violation)
                    if max(violation, 0.0) > 0.25 * previous:
                        mu = min(mu * float(al_growth), float(al_mu_max))
                    previous = max(violation, 0.0)
                    self.al_trace_.append((step, c_bar, lam, mu))
                    running.zero_()
                    count = 0

        del Xt, GXt, yt
        self.al_tau_ = tau
        return self._finish()


class GeneralizedMomentMethodIV(pointEstimator):
    """Unused. NOT the live Pi_Z implementation -- `iv_constraint_terms` is."""

    def __init__(self, backend: tuple[str, dict] | None = None, **kwargs):
        self.backend = backend
        super().__init__(**kwargs)

    def _fit(self, X, y, Z, **kwargs):
        h0 = np.linalg.pinv(X) @ y
        h = cp.Variable(h0.shape)
        Pi_Z = Z @ np.linalg.pinv(Z)
        moment_vector = cp.Constant(y) - cp.Constant(X) @ h
        cost = cp.quad_form(moment_vector, cp.psd_wrap(cp.Constant(Pi_Z)))
        prob = cp.Problem(cp.Minimize(cost))
        _solve_conic(prob, self.backend)
        self._W = h.value
        return self

    def _predict(self, X, **kwargs):
        return X @ self._W
