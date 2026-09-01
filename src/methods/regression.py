import cvxpy as cp
import numpy as np
from loguru import logger

from src.methods.abstract import pointEstimator


class LeastSquaresClosedForm(pointEstimator):
    """Closed-form least squares regression."""

    def _fit(self, X, y, **kwargs):
        """Fit using closed-form solution. Ignores extra kwargs."""
        self._W = np.linalg.pinv(X) @ y
        return self

    def _predict(self, X, **kwargs):
        """Predict outcomes. Ignores extra kwargs."""
        return X @ self._W


class LeastSquaresIterative(pointEstimator):
    def _fit(self, X, y, **kwargs):
        h0 = np.linalg.pinv(X) @ y
        h = cp.Variable(h0.shape)
        cost = cp.norm(y - X @ h)
        prob = cp.Problem(cp.Minimize(cost))
        try:
            result = prob.solve(solver=cp.CLARABEL)
        except:
            logger.warning("CLARABLE solver failed, falling back to ECOS.")
            result = prob.solve(solver=cp.ECOS)
        self._W = h.value
        return self

    def _predict(self, X, **kwargs):
        return X @ self._W


class TwoStageLeastSquaresIV(pointEstimator):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _fit(self, X, y, Z, **kwargs):

        S1 = LeastSquaresClosedForm().fit(Z, X).solution
        Xhat = Z @ S1

        S2 = LeastSquaresIterative().fit(Xhat, y).solution
        self._W = S2

        return self

    def _predict(self, X, **kwargs):
        return X @ self._W


class GradientDescentERM(pointEstimator):
    """Torch ERM for image treatments. Returns the conditional MEAN, not a label.

    Doubles as CopSens's outcome model: `predict_mean` + `prefit_` let a net trained
    once on the full draw be handed to every PI variant instead of refitted per method.
    """

    def __init__(self, model: str = "domnist-fast"):
        self.model = model
        super().__init__()

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
        import torch.nn.functional as F

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

        Xt = torch.tensor(np.asarray(X), dtype=torch.float, device=dev)
        yt = torch.tensor(np.asarray(y).reshape(-1, 1), dtype=torch.float, device=dev)
        sig = isinstance(self.f[-1], torch.nn.Sigmoid)

        # data already lives on the device; a DataLoader here is pure overhead
        n_train, n_batches = len(Xt), max(1, int(np.ceil(len(Xt) / batch)))
        sched = (
            torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=int(epochs) * n_batches)
            if onecycle
            else None
        )

        def _loss(p, target):
            t = target * (1 - 2 * label_smoothing) + label_smoothing if sig else target
            # Brier: proper for the MEAN, which is what CopSens consumes. BCE drives
            # logits to +-inf on noisy labels.
            if not sig or loss == "mse":
                return F.mse_loss(p, t)
            return F.binary_cross_entropy(p, t)

        for _ in range(int(epochs)):
            perm = torch.randperm(n_train, device=dev)
            for b in range(n_batches):
                idx = perm[b * batch : (b + 1) * batch]
                opt.zero_grad(set_to_none=True)
                _loss(self.f(Xt[idx]), yt[idx]).backward()
                opt.step()
                if sched is not None:
                    sched.step()

        self.f.eval()
        del Xt, yt
        self._W = np.concatenate([w.detach().cpu().numpy().ravel() for w in self.f.parameters()])[:, None]
        self.prefit_ = True  # CopSensPI: reuse instead of refitting
        return self

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
        """CopSens outcome-model protocol; mu_y is a flat (n,)."""
        return self._predict(np.asarray(X).reshape(len(X), -1)).ravel()

    # -- state round-trip: `self.f` only exists after _fit, so caching needs this --

    def save_state(self, path):
        import torch

        torch.save(
            {
                "model": self.model,
                "input_dim": self._input_dim,
                "state": {k: v.detach().cpu() for k, v in self.f.state_dict().items()},
            },
            path,
        )
        return path

    @classmethod
    def load_state(cls, path):
        import torch

        from src.methods.nets import NETS, device

        blob = torch.load(path, map_location="cpu", weights_only=True)
        self = cls(blob["model"])
        self._input_dim = int(blob["input_dim"])
        self.f = NETS[self.model](self._input_dim).float().to(device())
        self.f.load_state_dict(blob["state"])
        self.f.eval()
        self._W = np.concatenate([w.detach().cpu().numpy().ravel() for w in self.f.parameters()])[:, None]
        self.prefit_ = True
        return self


class GeneralizedMomentMethodIV(pointEstimator):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _fit(self, X, y, Z, **kwargs):
        h0 = np.linalg.pinv(X) @ y
        h = cp.Variable(h0.shape)
        Pi_Z = Z @ np.linalg.pinv(Z)
        moment_vector = cp.Constant(y) - cp.Constant(X) @ h
        cost = cp.quad_form(moment_vector, cp.psd_wrap(cp.Constant(Pi_Z)))
        prob = cp.Problem(cp.Minimize(cost))
        try:
            result = prob.solve(solver=cp.CLARABEL)
        except:
            logger.warning("CLARABLE solver failed, falling back to ECOS.")
            result = prob.solve(solver=cp.ECOS)
        self._W = h.value
        return self

    def _predict(self, X, **kwargs):
        return X @ self._W
