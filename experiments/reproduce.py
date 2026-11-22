"""Sanity check: does this pipeline reproduce published CropGlobe accuracies?

Follows the authors' RF baseline (500 trees, balanced class weights, an 80/20
train/validation split of the source, AUS/CHN category alignment as in their
Section 4.4) on AlphaEarth embeddings, and compares with Table 7 of
Tong & Wang (arXiv:2509.03497v3): AlphaEarth_RF, USA -> AUS 47.57 +/- 0.03,
USA -> ARG 24.82 +/- 0.14; and Table 11 (Appendix F): USA 2023 -> 2017
95.52 +/- 0.04, USA 2023 -> 2020 96.37 +/- 0.03 (overall accuracy, %, 10 runs).

Writes ``results/reproduce.csv``.
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
import config as cfg

PUBLISHED = {"AUS": 47.57, "ARG": 24.82, "USA_17": 95.52, "USA_20": 96.37}  # Table 7; Table 11
ALIGN = {"AUS": (4, 5, 6), "CHN": (0, 1, 2, 6)}  # classes defined in the target's map legend


def main(runs: int = 3):
    from zop.data import load_all
    from zop.maps import fit_rf, predict

    data = load_all("alphaearth")
    X, y = data["USA"]
    rows = []
    for seed in range(runs):
        rng = np.random.default_rng(81 + seed)
        tr = rng.permutation(len(y))[: int(0.8 * len(y))]
        model = fit_rf(X[tr], y[tr], seed=81 + seed)
        for T in ("AUS", "ARG", "CHN", "USA_17", "USA_20"):
            p = predict(model, data[T][0])
            if T in ALIGN:
                p = np.where(np.isin(p, ALIGN[T]), p, 6)
            rows.append(dict(seed=seed, target=T, oa=100 * float(np.mean(p == data[T][1])),
                             published=PUBLISHED.get(T, np.nan)))
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(cfg.RESULTS, "reproduce.csv"), index=False)
    print(df.groupby("target")[["oa", "published"]].agg(["mean", "std"]).round(2))


if __name__ == "__main__":
    main()
