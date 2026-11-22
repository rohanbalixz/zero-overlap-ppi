"""Partial identification of target crop shares when no target unit is labelled.

Setup
-----
Let J_T[c, k] = P_T(Y = c, Yhat = k) be the target's joint distribution of
truth and map prediction. Its column sums are the predicted shares q_T, which
are observed exactly; its row sums are the estimand theta_T. Row c, divided by
theta_c, is the target's *error profile* for crop c, rho_c = P_T(Yhat | Y = c).

Without assumptions, any theta in the simplex is consistent with q_T, so the
identified set is the whole simplex. We assume instead that each target error
profile lies within L1 distance delta_c of the convex hull of the error
profiles measured, for the same crop, in labelled domains:

    rho_c  in  conv{ C_j[c] : j labelled }  (+)  B_1(delta_c).

Writing lambda_{cj} = theta_c * (mixing weight), the set of (theta, J) that
satisfy this is a polytope, so the sharp bounds on every theta_c (and on the
target map accuracy, trace J) are linear programs.

Special cases: one reference and delta = 0 is label shift (BBSE); delta >= 2
is the trivial bound. The set grows monotonically in delta.

Calibrating delta
-----------------
``benchmark_delta`` computes, for every labelled domain j and crop c, the L1
distance from j's profile to the hull of the *other* labelled domains'
profiles, and returns the maximum. If the m labelled domains and the target
are exchangeable, this covers the target's own distance with probability at
least m / (m + 1). See docs/NOTE.md, Proposition 3.

Sampling error
--------------
Reference profiles are estimated. ``l1_slack`` returns, per crop, a
(Bonferroni) upper quantile of the L1 estimation error of each reference row
via a parametric bootstrap; adding it to delta keeps the true target profile
inside the (estimated) constraint set with the stated probability.
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from scipy import sparse
from scipy.optimize import linprog

from .data import K


@dataclass
class BoundResult:
    lo: np.ndarray      # (K,) lower bounds on theta_c
    hi: np.ndarray      # (K,) upper bounds on theta_c
    acc_lo: float       # bounds on target map accuracy, sum_c J[c, c]
    acc_hi: float
    feasible: bool


class _Rows:
    """Accumulates sparse constraint rows  sum_i coef_i * x[col_i]  (op)  rhs."""

    def __init__(self) -> None:
        self.rows: list[int] = []
        self.cols: list[int] = []
        self.vals: list[float] = []
        self.rhs: list[float] = []

    def add(self, terms: dict[int, float], rhs: float) -> None:
        r = len(self.rhs)
        for col, val in terms.items():
            self.rows.append(r)
            self.cols.append(col)
            self.vals.append(val)
        self.rhs.append(rhs)

    def matrix(self, n_vars: int):
        if not self.rhs:
            return None, None
        A = sparse.csr_matrix((self.vals, (self.rows, self.cols)), shape=(len(self.rhs), n_vars))
        return A, np.array(self.rhs)


@dataclass
class _Layout:
    """Column indices of the LP variables x = [J (K*K), s (K*K), lambda (sum_c m_c)]."""

    m: list[int]  # number of reference rows per crop

    def __post_init__(self) -> None:
        self.offset = np.r_[0, np.cumsum(self.m)].astype(int)
        self.n_vars = 2 * K * K + int(self.offset[-1])

    def J(self, c: int, k: int) -> int:  # P_T(Y = c, Yhat = k)
        return c * K + k

    def s(self, c: int, k: int) -> int:  # slack |J_ck - (Lambda V)_ck|
        return K * K + c * K + k

    def lam(self, c: int, j: int) -> int:  # theta_c x (weight of reference j for crop c)
        return 2 * K * K + int(self.offset[c]) + j

    def row(self, c: int) -> dict[int, float]:  # theta_c = sum_k J_ck
        return {self.J(c, k): 1.0 for k in range(K)}


def _build(q: np.ndarray, verts: Sequence[np.ndarray | None], delta: np.ndarray,
           zero: Sequence[int] = (), box: tuple[np.ndarray, np.ndarray] | None = None):
    """Equality/inequality systems and variable bounds of the polytope in Proposition 1."""
    lay = _Layout([0 if v is None else len(v) for v in verts])
    eq, ub = _Rows(), _Rows()

    for k in range(K):  # column sums of J are the observed predicted shares
        eq.add({lay.J(c, k): 1.0 for c in range(K)}, q[k])

    for c in range(K):
        if lay.m[c] == 0:
            continue  # no reference for crop c: its error profile is unrestricted
        V = verts[c]
        # sum_j Lambda_cj = theta_c
        eq.add({**{lay.lam(c, j): 1.0 for j in range(lay.m[c])}, **{lay.J(c, k): -1.0 for k in range(K)}}, 0.0)
        # |J_ck - sum_j Lambda_cj V_jk| <= s_ck, as two linear inequalities
        for k in range(K):
            for sign in (1.0, -1.0):
                terms = {lay.J(c, k): sign, lay.s(c, k): -1.0}
                terms.update({lay.lam(c, j): -sign * V[j, k] for j in range(lay.m[c])})
                ub.add(terms, 0.0)
        # sum_k s_ck <= delta_c * theta_c
        ub.add({**{lay.s(c, k): 1.0 for k in range(K)}, **{lay.J(c, k): -delta[c] for k in range(K)}}, 0.0)

    if box is not None:  # aggregate statistics: lo_c <= theta_c <= hi_c
        lo, hi = box
        for c in range(K):
            ub.add(lay.row(c), hi[c])
            ub.add({col: -1.0 for col in lay.row(c)}, -lo[c])

    bounds = [(0.0, None)] * lay.n_vars
    for c in range(K):
        for k in range(K):
            if c in set(zero):  # crop known to be absent from the target
                bounds[lay.J(c, k)] = (0.0, 0.0)
            if lay.m[c] == 0:
                bounds[lay.s(c, k)] = (0.0, 0.0)
    A_eq, b_eq = eq.matrix(lay.n_vars)
    A_ub, b_ub = ub.matrix(lay.n_vars)
    return A_eq, b_eq, A_ub, b_ub, bounds, lay


def _solve(cvec, A_eq, b_eq, A_ub, b_ub, bounds):
    res = linprog(cvec, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method="highs")
    return res.fun if res.status == 0 else None


def identified_set(q: np.ndarray, verts: Sequence[np.ndarray], delta, zero: Sequence[int] = (),
                   box: tuple[np.ndarray, np.ndarray] | None = None,
                   accuracy: bool = True) -> BoundResult:
    """Sharp bounds on each theta_c (and on accuracy) under the hull model."""
    delta = np.broadcast_to(np.asarray(delta, float), (K,)).copy()
    A_eq, b_eq, A_ub, b_ub, bnds, lay = _build(q, verts, delta, zero, box)
    lo, hi = np.full(K, np.nan), np.full(K, np.nan)
    feasible = True
    for c in range(K):
        cvec = np.zeros(lay.n_vars)
        cvec[list(lay.row(c))] = 1.0
        a = _solve(cvec, A_eq, b_eq, A_ub, b_ub, bnds)
        if a is None:
            feasible = False
            break
        b = _solve(-cvec, A_eq, b_eq, A_ub, b_ub, bnds)
        lo[c], hi[c] = a, -b
    acc_lo = acc_hi = np.nan
    if feasible and accuracy:
        cvec = np.zeros(lay.n_vars)
        cvec[[lay.J(c, c) for c in range(K)]] = 1.0
        acc_lo = _solve(cvec, A_eq, b_eq, A_ub, b_ub, bnds)
        acc_hi = -_solve(-cvec, A_eq, b_eq, A_ub, b_ub, bnds)
    return BoundResult(lo=np.clip(lo, 0, 1), hi=np.clip(hi, 0, 1), acc_lo=acc_lo, acc_hi=acc_hi,
                       feasible=feasible)


def hull_distance(r: np.ndarray, V: np.ndarray) -> float:
    """min over the simplex of || r - lambda^T V ||_1."""
    m = V.shape[0]
    c = np.r_[np.zeros(m), np.ones(K)]
    A_ub = np.vstack([np.c_[-V.T, -np.eye(K)], np.c_[V.T, -np.eye(K)]])
    b_ub = np.r_[-r, r]
    A_eq = np.r_[np.ones(m), np.zeros(K)][None]
    res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=[1.0], bounds=(0, None), method="highs")
    return float(res.fun)


Profiles = dict[str, tuple[np.ndarray, np.ndarray]]  # domain -> (C, n)


def references(profiles: Profiles, domains: Sequence[str], min_n: int) -> list[np.ndarray | None]:
    """Per crop, the stacked error profiles of the labelled domains that have it."""
    out = []
    for c in range(K):
        rows = [profiles[d][0][c] for d in domains if profiles[d][1][c] >= min_n]
        out.append(np.array(rows) if rows else None)
    return out


def leave_one_out_distances(profiles: Profiles, domains: Sequence[str], min_n: int) -> np.ndarray:
    """d[j, c]: distance of domain j's crop-c profile from the hull of the others' (nan if undefined)."""
    d = np.full((len(domains), K), np.nan)
    for c in range(K):
        have = [i for i, dom in enumerate(domains) if profiles[dom][1][c] >= min_n]
        for i in have:
            others = [profiles[domains[o]][0][c] for o in have if o != i]
            if others:
                d[i, c] = hull_distance(profiles[domains[i]][0][c], np.array(others))
    return d


def benchmark_delta(profiles: Profiles, domains: Sequence[str], min_n: int) -> float:
    d = leave_one_out_distances(profiles, domains, min_n)
    return float(np.nanmax(d)) if np.isfinite(d).any() else 2.0


def target_distances(C_T: np.ndarray, n_T: np.ndarray, verts: Sequence[np.ndarray | None]) -> np.ndarray:
    """Oracle: the target's own distance from the reference hull, per crop present in T."""
    out = np.zeros(K)
    for c in range(K):
        if n_T[c] > 0 and verts[c] is not None:
            out[c] = hull_distance(C_T[c], verts[c])
    return out


def l1_slack(profiles: Profiles, domains: Sequence[str], min_n: int, alpha: float, B: int,
             rng: np.random.Generator) -> np.ndarray:
    """Per crop, max over reference rows of the (1 - alpha/R) quantile of ||C_hat - C||_1.

    R is the number of reference rows (Bonferroni). Parametric bootstrap at the
    estimated row. By the triangle inequality the true hull is within this
    distance of the estimated hull, simultaneously for all crops, w.p. >= 1 - alpha
    (up to the bootstrap approximation).
    """
    rows = [(c, profiles[d][0][c], int(profiles[d][1][c])) for d in domains for c in range(K)
            if profiles[d][1][c] >= min_n]
    R = max(len(rows), 1)
    out = np.zeros(K)
    for c, p, n in rows:
        draws = rng.multinomial(n, p / p.sum(), size=B) / n
        e = np.quantile(np.abs(draws - p).sum(1), 1 - alpha / R)
        out[c] = max(out[c], e)
    return out


def contains(q: np.ndarray, verts, delta, theta: np.ndarray, zero: Sequence[int] = (),
             tol: float = 1e-7) -> bool:
    """Is the whole vector theta in the identified set? (feasibility LP with theta fixed)"""
    delta = np.broadcast_to(np.asarray(delta, float), (K,)).copy()
    box = (np.maximum(theta - tol, 0.0), theta + tol)
    A_eq, b_eq, A_ub, b_ub, bnds, lay = _build(q, verts, delta, zero, box)
    return _solve(np.zeros(lay.n_vars), A_eq, b_eq, A_ub, b_ub, bnds) is not None


def breakdown_delta(q: np.ndarray, verts, theta_true: np.ndarray, zero: Sequence[int] = (),
                    hi: float = 2.0, iters: int = 18) -> float:
    """Smallest common delta whose identified set contains theta_true (bisection; sets are nested)."""
    def covers(dl: float) -> bool:
        return contains(q, verts, dl, theta_true, zero=zero)

    if not covers(hi):
        return np.inf
    lo = 0.0
    if covers(lo):
        return 0.0
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        lo, hi = (lo, mid) if covers(mid) else (mid, hi)
    return hi
