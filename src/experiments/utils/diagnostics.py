"""do-MNIST diagnostics: the DA pre-screen (rho, tr(S)/k, tr(M*M)), the centre-error
report against the analytic target, and the ERM probe report.

These keep the do-MNIST run json's own definitions, which differ from the sweep
metrics of the linear datasets on purpose:

- `prescreen`'s `rho` is the ratio of the prefit nets' held-out squared errors
  (`net_noise`), not `metrics.rho_hat`'s OLS ratio; the pixel-logistic ratio it
  replaced is kept as `rho_linear`;
- `shift_operators`' `tr_S_over_k` CARRIES the mean-shift term tr(M*M) and is NOT
  `metrics.trace_S_over_k`, which centres both designs. The do-MNIST json has to
  reproduce the mixin's 0.6321 and the omega axis of the other datasets must not
  move, so both stay.

Not star-exported from `utils/__init__.py`: do-MNIST is the only caller.
"""

import numpy as np
from loguru import logger
from numpy.typing import NDArray


def _flat(A) -> NDArray:
    return np.asarray(A).reshape(len(A), -1)


def mmse(X: NDArray, y: NDArray, link: str = "probit") -> float:
    """The best LINEAR predictor's error on the pixels, the provenance `rho_linear`.
    Non-gaussian links all use the logistic fit: `rho_linear` is a RATIO of two such
    fits, so the link only has to be consistent across the pair."""
    from sklearn.linear_model import LinearRegression, LogisticRegression

    X, y = _flat(X), np.asarray(y).ravel()
    if link == "gaussian":
        r = y - LinearRegression().fit(X, y).predict(X)
        return float(np.mean(r**2))
    p = LogisticRegression(max_iter=2000, C=1e4).fit(X, (y > 0.5).astype(int)).predict_proba(X)[:, 1]
    return float(np.mean(p * (1 - p)))


def shift_operators(X: NDArray, GX: NDArray, keep: float = 0.999) -> dict[str, float]:
    """Lambda, M*M, S on the linear shape space H/R1 (Lem. 1).

    `keep` = fraction of X~-variance retained before whitening. Needed: the near-null
    eigendirections of Sg are noise, and 1/sqrt(w) blows them up. Untruncated, two
    IDENTICALLY distributed samples give tr(S)/k ~ 10 instead of 1 at d=588, n=60k.
    """
    X, GX = _flat(X), _flat(GX)
    Xc, Gc = X - X.mean(0), GX - GX.mean(0)
    Sx, Sg = Xc.T @ Xc / len(X), Gc.T @ Gc / len(GX)

    w, V = np.linalg.eigh(Sg)
    o = np.argsort(w)[::-1]
    w, V = w[o], V[:, o]
    pos = w > max(w.max(), 1e-30) * 1e-10
    w, V = w[pos], V[:, pos]
    if keep < 1.0:
        r = int(np.searchsorted(np.cumsum(w) / w.sum(), keep) + 1)
        w, V = w[:r], V[:, :r]
    k = len(w)

    Hi = V / np.sqrt(w)  # whitens the X~-geometry
    Lam = Hi.T @ Sx @ Hi  # <f, Lam g>_X~ = <f, g>_X

    dm = X.mean(0) - GX.mean(0)  # mean shift, rank 1
    mu = Hi.T @ dm
    tr_MM = float(mu @ mu)

    return {
        "k": int(k),
        "tr_Lambda_over_k": float(np.trace(Lam) / max(k, 1)),
        "tr_MM": tr_MM,
        "tr_S_over_k": float((np.trace(Lam) + tr_MM) / max(k, 1)),
    }


def net_noise(nets, X: NDArray, GX: NDArray, y: NDArray) -> dict[str, float]:
    """sigma^2, sigma~^2 and rho = sigma~^2 / sigma^2 off the prefit nets.

    The paper's sigma^2 := min_h R_erm(h) and sigma~^2 := min_h R_erm~(h) are the
    best-achievable squared errors before and after DA (Prop. 1, Lem. 1, Prop. 3).
    The estimate:
      - sigma^2 is the ERM net's mean squared error against y on the observed B rows
        X, and sigma~^2 the DA+ERM net's on the same rows' augmentation GX, the mixed
        measure it was trained on (and DA+PI's);
      - squared error against y, the risk the paper minimises and the nets' own
        loss, NOT mean mu(1 - mu): that is E[Var(Y|X)] only for a calibrated net,
        and its miscalibration differs between the two nets;
      - the B rows are held out from the nets' A draw, so each is an honest
        estimate of that net's risk, which sits above the minimum by the net's
        excess risk only; the pixel-linear fit (`mmse`) sits far above it once
        the DA moves the pixels, which is why it overstates rho;
      - mu unclipped: the clip is a device of the PI ball, not of the risk.
    rho falls back to 1 when the X net fits y exactly or the GX error is not finite.
    """
    y = np.asarray(y).ravel().astype(float)
    s2 = float(np.mean((y - np.asarray(nets["X"].predict_mean(X)).ravel()) ** 2))
    s2t = float(np.mean((y - np.asarray(nets["GX"].predict_mean(GX)).ravel()) ** 2))
    ok = s2 > 0.0 and np.isfinite(s2t)
    return {"sigma2": s2, "sigma2_tilde": s2t, "rho": s2t / s2 if ok else 1.0, "rho_ok": bool(ok)}


def prescreen(
    X: NDArray, y: NDArray, GX: NDArray, link: str = "probit", keep: float = 0.999, nets=None
) -> dict[str, float]:
    """Selection criterion: tr(S)/k <= 1 (Prop. 2 with an absolute budget).

    The radius is absolute, so rho does not enter the sufficient condition for
    sharpening; it enters Omega-hat = rho tr(S)/k and the calibrated criterion.
    `rho` (with `sigma2`, `sigma2_tilde`) is `net_noise` on the prefit `nets` when
    given, else the pixel-logistic ratio; that ratio is always kept as
    `rho_linear` (`sigma2_linear`, `sigma2_tilde_linear`).
    """
    X, GX = _flat(X), _flat(GX)
    s2, s2t = mmse(X, y, link), mmse(GX, y, link)
    rho_linear = s2t / max(s2, 1e-12)
    out: dict[str, float] = {"sigma2_linear": s2, "sigma2_tilde_linear": s2t, "rho_linear": rho_linear}
    if nets is None:
        out.update(sigma2=s2, sigma2_tilde=s2t, rho=rho_linear)
    else:
        noise = net_noise(nets, X, GX, y)
        noise.pop("rho_ok")
        out.update(noise)
    rho = out["rho"]
    out.update(shift_operators(X, GX, keep=keep))
    out["contracts"] = bool(out["tr_S_over_k"] <= 1.0)  # selection criterion
    out["slack"] = float(1.0 - out["tr_S_over_k"])  # bigger => sharper
    out["contracts_calibrated"] = bool(out["tr_S_over_k"] <= 1.0 / max(rho, 1e-12))  # if calibrate_sigma
    out["da_inert"] = bool(abs(rho - 1.0) < 1e-3)
    return out


def centre_error_report(mu, h_star, h_erm) -> dict[str, float]:
    """The quantity that sets the PI width: d = |mu_erm - h*|, against the analytic target.

    Coverage is a QUANTILE requirement, so p95(d), not the median and not |bias|, sets
    the narrowest valid interval. floor95 = 2 p95(d) is the best any method can do at
    95% coverage with this centre, so it is the fair reference for the achieved width.

    The centre error splits exactly as e = mu - h* = e_est + b, with e_est = mu - h_erm
    the net's own estimation error and b = h_erm - h* = (2C-1) beta(1/2-eta) the
    confounding bias (both h's analytic). Gamma is there for b; e_est is what it
    silently absorbs.
      err_corr    = corr(e_est, b), i.e. corr(e_est, C) up to sign: does the net's own
                    error line up with colour?
      shared_frac = mean(b^2) / (mean(b^2) + mean(e_est^2)), in [0, 1]: the share of
                    the centre error's independent-sum second moment that is
                    irreducible confounding (1 => all bias; -> 0 => estimation error
                    dominates). Not mean(b^2)/mean(e^2), which is unbounded when
                    e_est cancels b.
    On the probe mean(b^2) = (beta(1/2-eta))^2 and mean(e_est^2) = rmse_mu_h_erm^2, so
    shared_frac = bias^2 / (bias^2 + rmse_mu_h_erm^2).
    """
    mu, h_star, h_erm = (np.asarray(a, dtype=float).ravel() for a in (mu, h_star, h_erm))
    d = np.abs(mu - h_star)
    out = {f"d_p{int(q * 100)}": float(np.quantile(d, q)) for q in (0.5, 0.9, 0.95, 0.99)}
    out["d_max"] = float(d.max())
    out["floor95"] = 2.0 * out["d_p95"]
    e_est = mu - h_erm  # ERM's own estimation error
    b = h_erm - h_star  # confounding bias, analytic
    out["err_corr"] = float(np.corrcoef(e_est, b)[0, 1])
    out["err_est_sd"] = float(np.std(e_est))
    bb = float(np.mean(b**2))
    out["shared_frac"] = bb / max(bb + float(np.mean(e_est**2)), 1e-12)
    return out


def probe(sem_test, n: int = 10_000, seed: int = 7) -> dict[str, dict]:
    """Held-out obs/do draws from the TEST SEM, PAIRED (one draw, both label lines),
    plus every analytic target. Keys 'obs' and 'do', each with X (flat float32), y,
    f, h_erm, h_star."""
    Xt, y_obs, y_do = sem_test.sample_paired(n, seed=seed)
    t = dict(sem_test.last_)
    X = _flat(Xt).astype(np.float32)
    base = dict(X=X, f=t["f"], h_erm=sem_test.h_erm(t["f"], t["C"]), h_star=sem_test.h_star(t["f"]))
    return {"obs": dict(base, y=y_obs.ravel()), "do": dict(base, y=y_do.ravel())}


def erm_report(erm, probe: dict[str, dict]) -> dict[str, float]:
    """Bayes rule and ERM accuracy on obs AND do, f recovery, RMSE(mu, h_erm), the
    bias norm and the centre-error report. `erm` needs `predict_mean`."""
    d: dict[str, float] = {}
    for mode, q in probe.items():
        p = np.asarray(erm.predict_mean(q["X"]), dtype=float).ravel()
        hard = (p > 0.5).astype(float)
        d[f"acc_{mode}"] = float((hard == q["y"]).mean())
        d[f"acc_bayes_{mode}"] = float((q["f"] == q["y"]).mean())
        if mode == "obs":
            d["rmse_mu_h_erm"] = float(np.sqrt(np.mean((p - q["h_erm"]) ** 2)))
            d["f_recovery"] = float((hard == q["f"]).mean())
            d["bias_norm"] = float(np.sqrt(np.mean((p - q["h_star"]) ** 2)))
            d.update(centre_error_report(p, q["h_star"], q["h_erm"]))
    logger.info(
        f"ERM: acc obs {d['acc_obs']:.4f} do {d['acc_do']:.4f} | f rec {d['f_recovery']:.4f} | "
        f"RMSE(mu,h_erm) {d['rmse_mu_h_erm']:.4f} | d p50/p95 {d['d_p50']:.4f}/{d['d_p95']:.4f} "
        f"floor95 {d['floor95']:.4f} | err corr {d['err_corr']:+.3f} shared {d['shared_frac']:.1%}"
    )
    return d
