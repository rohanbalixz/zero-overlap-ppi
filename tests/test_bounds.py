import numpy as np
import pytest

from zop.bounds import (
    benchmark_delta,
    breakdown_delta,
    contains,
    hull_distance,
    identified_set,
    l1_slack,
    leave_one_out_distances,
    references,
)
from zop.data import K


def _stochastic(rng, n, conc=1.0, diag=0.0):
    M = rng.dirichlet(np.full(K, conc), size=n)
    if diag:
        M = (1 - diag) * M + diag * np.eye(K)[:n]
    return M


def _instance(rng, m=3, perturb=0.2):
    """References per crop, a target error profile within known L1 distance of their hull."""
    verts = [_stochastic(rng, m, conc=0.5) for _ in range(K)]
    rho, dist = np.zeros((K, K)), np.zeros(K)
    for c in range(K):
        mix = rng.dirichlet(np.ones(m)) @ verts[c]
        noise = rng.dirichlet(np.ones(K))
        t = rng.uniform(0, perturb)
        rho[c] = (1 - t) * mix + t * noise
        dist[c] = hull_distance(rho[c], verts[c])
    theta = rng.dirichlet(np.ones(K))
    q = theta @ rho
    return verts, rho, theta, q, dist


@pytest.mark.parametrize("seed", range(25))
def test_truth_is_inside_when_assumption_holds(seed):
    rng = np.random.default_rng(seed)
    verts, rho, theta, q, dist = _instance(rng)
    delta = dist.max() + 1e-6
    assert contains(q, verts, delta, theta)
    r = identified_set(q, verts, delta)
    assert r.feasible
    assert np.all(r.lo <= theta + 1e-6) and np.all(theta <= r.hi + 1e-6)
    acc = float(np.sum(theta * np.diag(rho)))
    assert r.acc_lo - 1e-6 <= acc <= r.acc_hi + 1e-6


def test_per_crop_delta_also_valid():
    rng = np.random.default_rng(99)
    verts, rho, theta, q, dist = _instance(rng)
    r = identified_set(q, verts, dist + 1e-6)
    assert np.all(r.lo <= theta + 1e-6) and np.all(theta <= r.hi + 1e-6)


def test_sets_are_nested_in_delta():
    rng = np.random.default_rng(1)
    verts, _, _, q, _ = _instance(rng)
    widths = []
    for d in [0.05, 0.1, 0.2, 0.4, 0.8, 1.6]:
        r = identified_set(q, verts, d)
        if r.feasible:
            widths.append(r.hi - r.lo)
    widths = np.array(widths)
    assert np.all(np.diff(widths, axis=0) >= -1e-7)


def test_delta_two_is_trivial():
    rng = np.random.default_rng(2)
    verts, _, _, q, _ = _instance(rng)
    r = identified_set(q, verts, 2.0)
    assert np.allclose(r.lo, 0, atol=1e-7) and np.allclose(r.hi, 1, atol=1e-7)


def test_single_reference_delta_zero_is_label_shift():
    rng = np.random.default_rng(3)
    C = _stochastic(rng, K, conc=1.0, diag=0.7)  # invertible, diagonally dominant
    theta = rng.dirichlet(np.full(K, 3.0))
    q = theta @ C
    verts = [C[c][None] for c in range(K)]
    r = identified_set(q, verts, 0.0)
    assert r.feasible
    assert np.allclose(r.lo, theta, atol=1e-6) and np.allclose(r.hi, theta, atol=1e-6)


def test_zero_classes_are_respected():
    rng = np.random.default_rng(4)
    verts, _, _, q, _ = _instance(rng)
    r = identified_set(q, verts, 2.0, zero=[0, 1])
    assert np.allclose(r.hi[[0, 1]], 0)


def test_hull_distance():
    rng = np.random.default_rng(5)
    V = _stochastic(rng, 4)
    lam = rng.dirichlet(np.ones(4))
    assert hull_distance(lam @ V, V) == pytest.approx(0, abs=1e-8)
    r = rng.dirichlet(np.ones(K))
    assert hull_distance(r, V[:1]) == pytest.approx(np.abs(r - V[0]).sum(), abs=1e-8)


def test_breakdown_delta_bounded_by_oracle_distance():
    rng = np.random.default_rng(6)
    verts, rho, theta, q, dist = _instance(rng)
    b = breakdown_delta(q, verts, theta)
    assert b <= dist.max() + 1e-3


def test_benchmark_and_slack():
    rng = np.random.default_rng(7)
    profiles = {}
    for d in "abcd":
        C = _stochastic(rng, K, conc=1.0, diag=0.6)
        profiles[d] = (C, np.full(K, 200))
    doms = list("abcd")
    D = leave_one_out_distances(profiles, doms, min_n=20)
    assert D.shape == (4, K) and np.all(D >= -1e-9)
    assert benchmark_delta(profiles, doms, 20) == pytest.approx(np.nanmax(D))
    s = l1_slack(profiles, doms, 20, alpha=0.1, B=500, rng=rng)
    assert np.all(s > 0) and np.all(s < 1)
    assert all(v is not None for v in references(profiles, doms, 20))


def test_slack_covers_true_rows():
    """The bootstrap slack covers the true row error at roughly the nominal rate."""
    rng = np.random.default_rng(8)
    p = rng.dirichlet(np.ones(K))
    n, hits = 150, 0
    for _ in range(300):
        phat = rng.multinomial(n, p) / n
        prof = {"a": (np.tile(phat, (K, 1)), np.full(K, n))}
        e = l1_slack(prof, ["a"], 1, alpha=0.1 * K, B=400, rng=rng)[0]  # alpha per row = 0.1
        hits += np.abs(phat - p).sum() <= e
    assert hits / 300 > 0.8
