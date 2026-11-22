"""The crop map: a classifier trained in one country and applied everywhere.

We use gradient-boosted trees for speed. A random forest with 500 trees
(the authors' setting) is used only in ``experiments/reproduce.py`` to check the
pipeline against published accuracies.
"""
from __future__ import annotations

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier


def fit_map(X: np.ndarray, y: np.ndarray, seed: int) -> HistGradientBoostingClassifier:
    model = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.1, random_state=seed)
    return model.fit(X, y)


def fit_rf(X: np.ndarray, y: np.ndarray, seed: int, n_estimators: int = 500) -> RandomForestClassifier:
    model = RandomForestClassifier(n_estimators=n_estimators, n_jobs=-1, random_state=seed,
                                   class_weight="balanced")
    return model.fit(X, y)


def predict(model, X: np.ndarray, batch: int = 50_000) -> np.ndarray:
    out = [model.predict(X[i:i + batch]) for i in range(0, len(X), batch)]
    return np.concatenate(out).astype(np.int64)
