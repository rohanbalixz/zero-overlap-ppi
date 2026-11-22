"""Stage 3: make the labelling probability positive by design.

Draw a simple random sample of n_T units inside the target, label them, and
estimate the target's crop shares from labels alone (Wald, Wilson) or with
the map: PPI (lambda = 1), PPI++ (power-tuned lambda), and PPI++ with an
Agresti-Min small-sample adjustment. This is
Assumption 1 of Kluger et al. restored by design: known, positive labelling
probability n_T / N_T for every target unit.

Writes ``results/design.csv`` aggregated over draws: coverage and mean width
per (representation, seed, target, n_T, method, crop).
"""
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
import config as cfg

METHODS = ("classical (Wald)", "classical (Wilson)", "PPI (lambda=1)", "PPI++ (Wald)", "PPI++ (adjusted)")


def job(args):
    rep, seed = args
    from zop import estimators as E
    from zop.data import TARGETS, K

    z = np.load(cfg.cache_path(rep, seed), allow_pickle=False)
    rng = np.random.default_rng([cfg.REPS.index(rep), seed, 3])
    rows = []
    for T in TARGETS:
        yT, pT = z[f"y|{T}"], z[f"pred|{T}"]
        N = len(yT)
        truth = np.bincount(yT, minlength=K) / N
        q = E.predicted_shares(pT)
        for n in cfg.N_T_GRID:
            acc = {m: ([], []) for m in METHODS}
            for _ in range(cfg.DESIGN_DRAWS):
                idx = rng.choice(N, n, replace=False)
                ivs = {"classical (Wald)": E.classical(yT[idx], cfg.ALPHA),
                       "classical (Wilson)": E.classical_wilson(yT[idx], cfg.ALPHA),
                       "PPI (lambda=1)": E.target_ppi(q, N, yT[idx], pT[idx], cfg.ALPHA, tuned=False),
                       "PPI++ (Wald)": E.target_ppi(q, N, yT[idx], pT[idx], cfg.ALPHA, tuned=True),
                       "PPI++ (adjusted)": E.target_ppi_adjusted(q, N, yT[idx], pT[idx], cfg.ALPHA)}
                for m, iv in ivs.items():
                    acc[m][0].append((iv.lo - 1e-12 <= truth) & (truth <= iv.hi + 1e-12))
                    acc[m][1].append(iv.hi - iv.lo)
            for m, (cov, wid) in acc.items():
                cov, wid = np.mean(cov, 0), np.mean(wid, 0)
                for c in range(K):
                    rows.append(dict(rep=rep, seed=seed, target=T, n_T=n, method=m, crop=c,
                                     coverage=cov[c], width=wid[c], truth=truth[c]))
    return rows


def main():
    jobs = [(r, s) for r in cfg.REPS for s in cfg.SEEDS]
    with ProcessPoolExecutor(max_workers=10) as ex:
        rows = [row for part in ex.map(job, jobs) for row in part]
    pd.DataFrame(rows).to_csv(os.path.join(cfg.RESULTS, "design.csv"), index=False)


if __name__ == "__main__":
    main()
