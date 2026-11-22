"""Feature-space positivity diagnostic.

A domain classifier is trained to tell labelled-domain units (class 0) from
target units (class 1) on a balanced sample, with cross-fitting. Its
propensity e(x) = P(target | x) (at a 50/50 prior) gives

* AUC: 0.5 means the domains are indistinguishable, 1.0 means disjoint;
* overlap share: the fraction of target units with e(x) <= 1 - eps, i.e. the
  share of the target that *could* have been labelled under a design that
  samples from the labelled domains (Crump et al., 2009 use eps = 0.1);
* the Kish effective sample size of the density-ratio weights
  w = e / (1 - e) on labelled units, which is what IPW-PPI would use.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

CLIP = 0.01  # propensities are clipped to [CLIP, 1 - CLIP] before forming weights


def _clf(seed: int) -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(max_iter=200, learning_rate=0.1, random_state=seed)


@dataclass
class OverlapResult:
    auc: float
    overlap_share: float  # P_target(e <= 1 - eps)
    overlap_share_crump: float  # P_target(eps <= e <= 1 - eps) with eps = 0.1
    model: HistGradientBoostingClassifier

    def weights(self, X: np.ndarray) -> np.ndarray:
        e = np.clip(self.model.predict_proba(X)[:, 1], CLIP, 1 - CLIP)
        return e / (1 - e)


def kish_ess(w: np.ndarray) -> float:
    return float(w.sum() ** 2 / np.sum(w ** 2))


def diagnose(X_lab: np.ndarray, X_tgt: np.ndarray, seed: int, n_max: int = 4000,
             eps: float = 0.05, folds: int = 5) -> OverlapResult:
    """Cross-fitted domain classifier on a balanced subsample of the two pools."""
    rng = np.random.default_rng(seed)
    n = min(n_max, len(X_lab), len(X_tgt))
    A = X_lab[rng.choice(len(X_lab), n, replace=False)]
    B = X_tgt[rng.choice(len(X_tgt), n, replace=False)]
    Z = np.vstack([A, B])
    t = np.r_[np.zeros(n), np.ones(n)]
    e = np.empty(len(t))
    for tr, te in StratifiedKFold(folds, shuffle=True, random_state=seed).split(Z, t):
        e[te] = _clf(seed).fit(Z[tr], t[tr]).predict_proba(Z[te])[:, 1]
    eT = e[t == 1]
    final = _clf(seed).fit(Z, t)
    return OverlapResult(
        auc=float(roc_auc_score(t, e)),
        overlap_share=float(np.mean(eT <= 1 - eps)),
        overlap_share_crump=float(np.mean((eT >= 0.1) & (eT <= 0.9))),
        model=final,
    )
