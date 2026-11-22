"""Estimators of the crop-share vector theta_T of a target domain T.

All estimators take the map's predicted shares on the *whole* target,
``q_T = mean_{i in T} onehot(yhat_i)``, which is observed exactly. What
differs is where the correction for map error comes from:

==========================  =====================================  ==========================
estimator                   correction learned from                 transport assumption
==========================  =====================================  ==========================
plug-in                     nothing                                 the map is unbiased in T
transported PPI             labels in other domains                 E[Y - f] is the same in T
transported stratified      labels in other domains                 P(Y | Yhat) is the same in T
label shift (BBSE)          labels in other domains                 P(Yhat | Y) is the same in T
IPW-PPI                     labels in other domains, reweighted     P(Y | X) same AND overlap
in-target PPI / PPI++       a probability sample *inside* T         none (design-based)
classical                   a probability sample inside T           none
==========================  =====================================  ==========================

The first five are what one can do when the labelling probability in T is
exactly zero. The last two restore a positive labelling probability by design.

Intervals are two-sided at level 1 - alpha. Variances treat the labelled
units as i.i.d. draws; the q_T term contributes q(1-q)/N_T (a super-population
view), which is negligible here.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import nnls
from scipy.stats import norm

from .data import K


@dataclass
class Interval:
    est: np.ndarray
    lo: np.ndarray
    hi: np.ndarray

    @classmethod
    def wald(cls, est: np.ndarray, se: np.ndarray, alpha: float) -> Interval:
        z = norm.ppf(1 - alpha / 2)
        return cls(est=est, lo=est - z * se, hi=est + z * se)


def onehot(y: np.ndarray, k: int = K) -> np.ndarray:
    out = np.zeros((len(y), k))
    out[np.arange(len(y)), y] = 1.0
    return out


def predicted_shares(yhat: np.ndarray) -> np.ndarray:
    return np.bincount(yhat, minlength=K) / len(yhat)


# --------------------------------------------------------------------------
# zero labelling probability in T: transport a correction from elsewhere
# --------------------------------------------------------------------------

def plugin(q_T: np.ndarray) -> Interval:
    """Pixel counting. No interval: the map's shares are reported as the answer."""
    return Interval(est=q_T.copy(), lo=q_T.copy(), hi=q_T.copy())


def transported_ppi(q_T: np.ndarray, N_T: int, y_L: np.ndarray, yhat_L: np.ndarray,
                    alpha: float) -> Interval:
    """PPI mean estimator (Angelopoulos et al., 2023) with the rectifier from L."""
    R = onehot(y_L) - onehot(yhat_L)
    est = q_T + R.mean(0)
    se = np.sqrt(R.var(0, ddof=1) / len(y_L) + q_T * (1 - q_T) / N_T)
    return Interval.wald(est, se, alpha)


def transported_stratified(q_T: np.ndarray, y_L: np.ndarray, yhat_L: np.ndarray,
                           alpha: float) -> Interval:
    """Error-matrix (post-stratified) estimator: theta_c = sum_k q_k P(Y=c | Yhat=k).

    This is the standard area estimator of Olofsson et al. (2014), here with the
    error matrix borrowed from other domains. Strata with no labelled unit are
    taken at face value (P(Y=k | Yhat=k) = 1) and contribute no variance.
    """
    M = np.zeros((K, K))  # rows: predicted k, cols: true c
    np.add.at(M, (yhat_L, y_L), 1.0)
    n_k = M.sum(1)
    P = np.eye(K)
    has = n_k > 0
    P[has] = M[has] / n_k[has, None]
    est = q_T @ P
    with np.errstate(divide="ignore", invalid="ignore"):
        v = np.where(has[:, None], P * (1 - P) / np.maximum(n_k - 1, 1)[:, None], 0.0)
    se = np.sqrt((q_T[:, None] ** 2 * v).sum(0))
    return Interval.wald(est, se, alpha)


def error_profile(y: np.ndarray, yhat: np.ndarray):
    """Row-stochastic C with C[c, k] = P(Yhat = k | Y = c), and class counts n."""
    M = np.zeros((K, K))
    np.add.at(M, (y, yhat), 1.0)
    n = M.sum(1)
    C = np.full((K, K), np.nan)
    C[n > 0] = M[n > 0] / n[n > 0, None]
    return C, n


def _project_simplex(v: np.ndarray) -> np.ndarray:
    """Euclidean projection onto the probability simplex (Duchi et al., 2008)."""
    u = np.sort(v)[::-1]
    css = np.cumsum(u) - 1.0
    k = np.nonzero(u - css / np.arange(1, len(v) + 1) > 0)[0][-1]
    return np.maximum(v - css[k] / (k + 1), 0.0)


def _simplex_ls_pg(A: np.ndarray, b: np.ndarray, iters: int = 2000, tol: float = 1e-12) -> np.ndarray:
    """Accelerated projected gradient (FISTA) for min ||Ax - b||^2 on the simplex."""
    m = A.shape[1]
    step = 1.0 / max(np.linalg.norm(A, 2) ** 2, 1e-12)
    x = y = np.full(m, 1.0 / m)
    t = 1.0
    for _ in range(iters):
        x_new = _project_simplex(y - step * (A.T @ (A @ y - b)))
        t_new = 0.5 * (1 + np.sqrt(1 + 4 * t * t))
        y = x_new + ((t - 1) / t_new) * (x_new - x)
        if np.abs(x_new - x).max() < tol:
            x = x_new
            break
        x, t = x_new, t_new
    return x


def _simplex_ls(A: np.ndarray, b: np.ndarray, w: float = 10.0) -> np.ndarray:
    """argmin ||A x - b||_2 over the probability simplex.

    Fast path: NNLS with a weighted sum-to-one row. SciPy's active-set NNLS can
    stall or hit a singular system on degenerate bootstrap resamples; then we
    fall back to projected gradient, which always returns a simplex point.
    """
    m = A.shape[1]
    A2 = np.vstack([A, w * np.ones((1, m))])
    b2 = np.r_[b, w]
    try:
        x, _ = nnls(A2, b2, maxiter=50 * m)
        if not np.all(np.isfinite(x)):
            raise RuntimeError
    except Exception:
        return _simplex_ls_pg(A, b)
    s = x.sum()
    return x / s if s > 0 else _simplex_ls_pg(A, b)


def _bbse_point(q_T: np.ndarray, y_L: np.ndarray, yhat_L: np.ndarray) -> np.ndarray:
    C, n = error_profile(y_L, yhat_L)
    keep = n > 0
    theta = np.zeros(K)
    theta[keep] = _simplex_ls(C[keep].T, q_T)
    return theta


def label_shift(q_T: np.ndarray, y_L: np.ndarray, yhat_L: np.ndarray, alpha: float,
                B: int, rng: np.random.Generator) -> Interval:
    """Black-box shift estimation (Lipton et al., 2018), simplex-constrained.

    Solves C^T theta = q_T, where C is the error profile learned from L.
    Point-identifies theta_T if P(Yhat | Y) is the same in T and C is invertible.
    Percentile-bootstrap interval over L.
    """
    est = _bbse_point(q_T, y_L, yhat_L)
    n = len(y_L)
    boots = np.empty((B, K))
    for b in range(B):
        idx = rng.integers(0, n, n)
        boots[b] = _bbse_point(q_T, y_L[idx], yhat_L[idx])
    lo, hi = np.quantile(boots, [alpha / 2, 1 - alpha / 2], axis=0)
    return Interval(est=est, lo=lo, hi=hi)


def ipw_ppi(q_T: np.ndarray, N_T: int, y_L: np.ndarray, yhat_L: np.ndarray, w: np.ndarray,
            alpha: float) -> Interval:
    """Covariate-shift PPI: rectifier reweighted by an estimated density ratio.

    This is the inverse-probability-weighted route of Kluger et al. (2025) with
    the labelling propensity *estimated* by a domain classifier, as one would do
    when it is unknown. Valid only if P(Y | X) is shared and the target lies in
    the support of the labelled domains. Self-normalised; linearised variance.
    """
    R = onehot(y_L) - onehot(yhat_L)
    wn = w / w.sum()
    rbar = wn @ R
    est = q_T + rbar
    se = np.sqrt((wn[:, None] ** 2 * (R - rbar) ** 2).sum(0) + q_T * (1 - q_T) / N_T)
    return Interval.wald(est, se, alpha)


# --------------------------------------------------------------------------
# positive labelling probability in T, by design
# --------------------------------------------------------------------------

def classical(y_T: np.ndarray, alpha: float) -> Interval:
    Y = onehot(y_T)
    return Interval.wald(Y.mean(0), Y.std(0, ddof=1) / np.sqrt(len(y_T)), alpha)


def classical_wilson(y_T: np.ndarray, alpha: float) -> Interval:
    """Wilson score interval per crop (the standard small-sample fix for Wald)."""
    n = len(y_T)
    p = np.bincount(y_T, minlength=K) / n
    z = norm.ppf(1 - alpha / 2)
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return Interval(est=p, lo=centre - half, hi=centre + half)


def target_ppi_adjusted(q_T: np.ndarray, N_T: int, y_T: np.ndarray, yhat_T: np.ndarray,
                        alpha: float) -> Interval:
    """PPI++ with an Agresti-Min small-sample adjustment.

    For each crop, (Y_c, F_c) on the sampled units is a matched pair of
    indicators. Following Agresti & Min (2005) for matched proportions, half a
    pseudo-unit is added to each cell of the 2x2 table before computing the
    moments, lambda and the variance. This keeps the variance away from zero
    when the map makes few errors in the sample, which is where the
    unadjusted Wald interval undercovers.
    """
    n = len(y_T)
    Y, F = onehot(y_T), onehot(yhat_T)
    n11 = (Y * F).sum(0)
    nY, nF = Y.sum(0), F.sum(0)
    m = n + 2.0
    EY, EF = (nY + 1.0) / m, (nF + 1.0) / m
    EYF = (n11 + 0.5) / m
    vY, vFs, cov = EY * (1 - EY), EF * (1 - EF), EYF - EY * EF
    varF = q_T * (1 - q_T)
    with np.errstate(divide="ignore", invalid="ignore"):
        lam = np.where(varF > 0, cov / ((1 + n / N_T) * varF), 0.0)
    lam = np.clip(lam, 0.0, 1.0)
    est = lam * q_T + EY - lam * EF
    vD = np.maximum(vY + lam ** 2 * vFs - 2 * lam * cov, 0.0)
    se = np.sqrt(vD / m + lam ** 2 * varF / N_T)
    return Interval.wald(est, se, alpha)


def target_ppi(q_T: np.ndarray, N_T: int, y_T: np.ndarray, yhat_T: np.ndarray, alpha: float,
               tuned: bool = True) -> Interval:
    """PPI (lambda = 1) or PPI++ (power-tuned lambda; Angelopoulos, Duchi & Zrnic, 2023)
    from a simple random sample of the target itself.

    Uses the map's predictions on all N_T target units. The variance of the
    predicted indicator over the target is known exactly, q(1-q). With
    ``tuned=True``, lambda_c = Cov(Y_c, F_c) / ((1 + n/N) Var(F_c)), clipped to
    [0, 1]; lambda = 0 recovers the classical estimator.
    """
    n = len(y_T)
    Y, F = onehot(y_T), onehot(yhat_T)
    varF = q_T * (1 - q_T)
    if tuned:
        cov = ((Y - Y.mean(0)) * (F - F.mean(0))).sum(0) / (n - 1)
        with np.errstate(divide="ignore", invalid="ignore"):
            lam = np.where(varF > 0, cov / ((1 + n / N_T) * varF), 0.0)
        lam = np.clip(lam, 0.0, 1.0)
    else:
        lam = np.ones(K)
    D = Y - lam * F
    est = lam * q_T + D.mean(0)
    se = np.sqrt(D.var(0, ddof=1) / n + lam ** 2 * varF / N_T)
    return Interval.wald(est, se, alpha)
