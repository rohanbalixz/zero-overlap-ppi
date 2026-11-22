import numpy as np

from zop.data import K
from zop.estimators import (
    classical,
    error_profile,
    ipw_ppi,
    label_shift,
    predicted_shares,
    target_ppi,
    transported_ppi,
    transported_stratified,
)

ALPHA = 0.1


def _population(rng, N, theta, C):
    y = rng.choice(K, size=N, p=theta)
    u = rng.random(N)
    cum = np.cumsum(C[y], axis=1)
    yhat = (u[:, None] > cum).sum(1)
    return y, np.minimum(yhat, K - 1)


def _C(rng, acc=0.7):
    M = rng.dirichlet(np.ones(K), size=K)
    return (1 - acc) * M + acc * np.eye(K)


def test_transported_ppi_is_valid_without_shift():
    """Same population for labels and target: PPI should have nominal coverage."""
    rng = np.random.default_rng(0)
    theta, C = rng.dirichlet(np.full(K, 2.0)), _C(rng)
    yT, pT = _population(rng, 200_000, theta, C)
    truth = np.bincount(yT, minlength=K) / len(yT)
    q = predicted_shares(pT)
    cover = []
    for _ in range(400):
        yL, pL = _population(rng, 500, theta, C)
        r = transported_ppi(q, len(yT), yL, pL, ALPHA)
        cover.append((r.lo <= truth) & (truth <= r.hi))
    cov = np.mean(cover)
    assert 0.86 <= cov <= 0.94, cov


def test_transported_ppi_fails_under_shift():
    rng = np.random.default_rng(1)
    theta, C = rng.dirichlet(np.full(K, 2.0)), _C(rng)
    yT, pT = _population(rng, 100_000, theta, _C(rng, acc=0.4))  # different error profile in T
    truth = np.bincount(yT, minlength=K) / len(yT)
    q = predicted_shares(pT)
    yL, pL = _population(rng, 20_000, theta, C)
    r = transported_ppi(q, len(yT), yL, pL, ALPHA)
    assert np.mean((r.lo <= truth) & (truth <= r.hi)) < 0.5


def test_label_shift_recovers_theta_under_label_shift():
    rng = np.random.default_rng(2)
    C = _C(rng, acc=0.75)
    thS, thT = rng.dirichlet(np.ones(K) * 2), rng.dirichlet(np.ones(K) * 2)
    yT, pT = _population(rng, 200_000, thT, C)
    yL, pL = _population(rng, 50_000, thS, C)
    r = label_shift(predicted_shares(pT), yL, pL, ALPHA, B=50, rng=rng)
    assert np.abs(r.est - np.bincount(yT, minlength=K) / len(yT)).max() < 0.02


def test_stratified_exact_when_predictive_values_transfer():
    """If L and T share P(Y | Yhat), the stratified estimator is consistent."""
    rng = np.random.default_rng(3)
    theta, C = rng.dirichlet(np.ones(K) * 2), _C(rng)
    y, p = _population(rng, 300_000, theta, C)
    T = np.arange(len(y)) < 150_000
    r = transported_stratified(predicted_shares(p[T]), y[~T], p[~T], ALPHA)
    assert np.abs(r.est - np.bincount(y[T], minlength=K) / T.sum()).max() < 0.01


def test_ipw_with_unit_weights_equals_ppi():
    rng = np.random.default_rng(4)
    theta, C = rng.dirichlet(np.ones(K)), _C(rng)
    yL, pL = _population(rng, 1000, theta, C)
    q = rng.dirichlet(np.ones(K))
    a = transported_ppi(q, 10_000, yL, pL, ALPHA)
    b = ipw_ppi(q, 10_000, yL, pL, np.ones(len(yL)), ALPHA)
    assert np.allclose(a.est, b.est)
    assert np.allclose(a.hi - a.lo, b.hi - b.lo, rtol=0.01)


def test_in_target_ppi_valid_and_no_wider_than_classical():
    rng = np.random.default_rng(5)
    theta, C = rng.dirichlet(np.full(K, 2.0)), _C(rng, acc=0.8)
    yT, pT = _population(rng, 50_000, theta, C)
    truth = np.bincount(yT, minlength=K) / len(yT)
    q = predicted_shares(pT)
    cov_ppi, cov_cl, w_ppi, w_cl = [], [], [], []
    for _ in range(400):
        idx = rng.choice(len(yT), 200, replace=False)
        a = target_ppi(q, len(yT), yT[idx], pT[idx], ALPHA, tuned=True)
        b = classical(yT[idx], ALPHA)
        cov_ppi.append((a.lo <= truth) & (truth <= a.hi))
        cov_cl.append((b.lo <= truth) & (truth <= b.hi))
        w_ppi.append(a.hi - a.lo)
        w_cl.append(b.hi - b.lo)
    assert np.mean(cov_ppi) > 0.86
    assert np.mean(w_ppi) < np.mean(w_cl)


def test_error_profile_rows_are_stochastic():
    rng = np.random.default_rng(6)
    y, p = rng.integers(0, K, 500), rng.integers(0, K, 500)
    C, n = error_profile(y, p)
    assert np.allclose(np.nansum(C, 1)[n > 0], 1)


def test_simplex_ls_survives_ill_conditioned_input():
    """Regression: a bootstrap resample once made NNLS stall; the solver must still return a simplex point."""
    from zop.estimators import _simplex_ls
    rng = np.random.default_rng(7)
    A = np.eye(K)
    A[:, 1] = A[:, 0] + 1e-12  # two nearly identical columns
    x = _simplex_ls(A, rng.dirichlet(np.ones(K)))
    assert np.all(x >= 0) and abs(x.sum() - 1) < 1e-9


def test_projected_gradient_matches_nnls_path():
    from zop.estimators import _simplex_ls, _simplex_ls_pg
    rng = np.random.default_rng(8)
    for _ in range(20):
        A = rng.random((K, K)) + np.eye(K)
        b = A @ rng.dirichlet(np.ones(K))
        assert np.allclose(_simplex_ls(A, b), _simplex_ls_pg(A, b), atol=1e-4)


def test_singular_design_falls_back():
    from zop.estimators import _simplex_ls
    A = np.zeros((K, K))
    A[:, 0] = 1.0  # rank one
    x = _simplex_ls(A, np.ones(K))
    assert np.all(x >= 0) and abs(x.sum() - 1) < 1e-9


def test_adjusted_ppi_fixes_small_sample_undercoverage():
    """Accurate map + small n: the raw Wald PPI++ interval often has zero width; the adjusted one should not."""
    from zop.estimators import classical_wilson, target_ppi_adjusted
    rng = np.random.default_rng(9)
    theta, C = rng.dirichlet(np.full(K, 2.0)), _C(rng, acc=0.97)
    yT, pT = _population(rng, 40_000, theta, C)
    truth = np.bincount(yT, minlength=K) / len(yT)
    q = predicted_shares(pT)
    raw, adj, wil = [], [], []
    for _ in range(400):
        idx = rng.choice(len(yT), 50, replace=False)
        for store, iv in ((raw, target_ppi(q, len(yT), yT[idx], pT[idx], ALPHA)),
                          (adj, target_ppi_adjusted(q, len(yT), yT[idx], pT[idx], ALPHA)),
                          (wil, classical_wilson(yT[idx], ALPHA))):
            store.append((iv.lo <= truth) & (truth <= iv.hi))
    assert np.mean(adj) >= 0.88 > np.mean(raw)
    assert np.mean(wil) >= 0.85
