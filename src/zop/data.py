"""CropGlobe loading and the domain registry.

CropGlobe (Tong & Wang, arXiv:2509.03497) ships one ``.npz`` per country and
representation with two arrays, ``fea`` (N x d) and ``lbl`` (N,). There are no
coordinates, so every analysis here is at the level of a *domain*: a country in
2023, or the USA in 2017 / 2020.

Rows are de-duplicated exactly as in the authors' loader (``np.unique`` on the
feature matrix), so sample counts match their tables.
"""
from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np

CROPS = ("corn", "soybeans", "rice", "wheat", "sugarcane", "cotton", "other")
K = len(CROPS)

# The map is trained on part of USA 2023 ("USA"); every other domain is a
# candidate target. USA_17 / USA_20 are the same country in other years.
SOURCE = "USA"
DOMAINS = ("USA", "FRA", "BEL", "NLD", "GBR", "CHN", "AUS", "ARG", "USA_17", "USA_20")
TARGETS = tuple(d for d in DOMAINS if d != SOURCE)
LABEL = {
    "USA": "USA '23", "FRA": "France", "BEL": "Belgium", "NLD": "Netherlands",
    "GBR": "UK", "CHN": "China", "AUS": "Australia", "ARG": "Argentina",
    "USA_17": "USA '17", "USA_20": "USA '20",
}
TEMPORAL = ("USA_17", "USA_20")

REPRESENTATIONS = {
    "alphaearth": "AlphaEarth/alphaearth_embedding_{d}.npz",
    "harmonic": "Harmonic/harmonic_{d}_month5-11.npz",
}

DEFAULT_ROOT = os.path.join(os.path.dirname(__file__), "..", "..", "data", "raw", "cropglobe")


def data_root() -> str:
    return os.path.abspath(os.environ.get("CROPGLOBE_ROOT", DEFAULT_ROOT))


def load_domain(rep: str, dom: str, root: str | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Features and labels for one domain, de-duplicated like the CropNet loader."""
    path = os.path.join(root or data_root(), REPRESENTATIONS[rep].format(d=dom))
    with np.load(path, allow_pickle=False) as z:
        X, y = z["fea"].astype(np.float32), z["lbl"].astype(np.int64)
    X, idx = np.unique(X, axis=0, return_index=True)
    return X, y[idx]


def load_all(rep: str, root: str | None = None) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    return {d: load_domain(rep, d, root) for d in DOMAINS}


@dataclass
class Split:
    """Index sets for one seed.

    map_train : USA 2023 units used to fit the map (never used for inference)
    calib     : per labelled domain, units that may be drawn as calibration labels
    nuisance  : per domain, units reserved for fitting the domain classifier
    """

    map_train: np.ndarray
    calib: dict[str, np.ndarray]
    nuisance: dict[str, np.ndarray]


def make_split(sizes: dict[str, int], seed: int, map_frac: float = 0.5) -> Split:
    rng = np.random.default_rng(seed)
    calib, nuisance = {}, {}
    perm = rng.permutation(sizes[SOURCE])
    n_map = int(map_frac * sizes[SOURCE])
    map_train = np.sort(perm[:n_map])
    rest = perm[n_map:]
    calib[SOURCE] = np.sort(rest[: len(rest) // 2])
    nuisance[SOURCE] = np.sort(rest[len(rest) // 2:])
    for d in TARGETS:
        p = rng.permutation(sizes[d])
        calib[d] = np.sort(p[: sizes[d] // 2])
        nuisance[d] = np.sort(p[sizes[d] // 2:])
    return Split(map_train=map_train, calib=calib, nuisance=nuisance)


def shares(y: np.ndarray) -> np.ndarray:
    return np.bincount(y, minlength=K) / len(y)
