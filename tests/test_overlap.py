import numpy as np

from zop.overlap import diagnose, kish_ess


def test_identical_domains_overlap():
    rng = np.random.default_rng(0)
    A, B = rng.normal(size=(2000, 5)), rng.normal(size=(2000, 5))
    r = diagnose(A, B, seed=0, n_max=2000, folds=3)
    assert r.auc < 0.6
    assert r.overlap_share > 0.95


def test_disjoint_domains_no_overlap():
    rng = np.random.default_rng(1)
    A, B = rng.normal(size=(2000, 5)), rng.normal(loc=6.0, size=(2000, 5))
    r = diagnose(A, B, seed=0, n_max=2000, folds=3)
    assert r.auc > 0.99
    assert r.overlap_share < 0.02


def test_kish_ess():
    assert kish_ess(np.ones(10)) == 10
    assert kish_ess(np.r_[1.0, np.zeros(9)]) == 1
