"""Stage 1: fit the map and the domain classifiers once per (representation, seed).

Outputs ``results/cache/{rep}_seed{s}.npz`` (map predictions for every unit,
labels, splits, and IPW weights for every calibration unit) and
``results/overlap.csv`` (the positivity diagnostic for every scenario/target).
``ess_frac`` is the Kish effective sample size of the clipped weights over n;
when the domains are separable every weight sits at the clip floor, so a high
value there means "constant weights", not "good overlap". Use AUC and the
overlap share instead.
"""
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
import config as cfg


def job(args):
    rep, seed = args
    from threadpoolctl import threadpool_limits

    from zop.data import DOMAINS, SOURCE, load_all, make_split
    from zop.maps import fit_map, predict
    from zop.overlap import diagnose, kish_ess

    t0 = time.time()
    with threadpool_limits(limits=2):
        data = load_all(rep)
        split = make_split({d: len(v[1]) for d, v in data.items()}, seed)
        X0, y0 = data[SOURCE]
        model = fit_map(X0[split.map_train], y0[split.map_train], seed)
        out = {"map_train": split.map_train}
        for d in DOMAINS:
            out[f"pred|{d}"] = predict(model, data[d][0])
            out[f"y|{d}"] = data[d][1]
            out[f"calib|{d}"] = split.calib[d]
            out[f"nuis|{d}"] = split.nuisance[d]
        rows = []
        rng = np.random.default_rng(1000 + seed)
        for scen, spec in cfg.SCENARIOS.items():
            for T, S in spec.items():
                per = cfg.OVERLAP_NMAX // len(S)
                lab = []
                for j in S:
                    pool = split.nuisance[j]
                    lab.append(data[j][0][rng.choice(pool, min(per, len(pool)), replace=False)])
                res = diagnose(np.vstack(lab), data[T][0], seed=seed, n_max=cfg.OVERLAP_NMAX,
                               folds=cfg.OVERLAP_FOLDS)
                wall = []
                for j in S:
                    w = res.weights(data[j][0][split.calib[j]])
                    out[f"w|{scen}|{T}|{j}"] = w.astype(np.float32)
                    wall.append(w)
                wall = np.concatenate(wall)
                rows.append(dict(rep=rep, seed=seed, scenario=scen, target=T, auc=res.auc,
                                 overlap=res.overlap_share, overlap_crump=res.overlap_share_crump,
                                 ess_frac=kish_ess(wall) / len(wall)))
    os.makedirs(cfg.CACHE, exist_ok=True)
    np.savez_compressed(cfg.cache_path(rep, seed), **out)
    print(f"[prepare] {rep} seed {seed} done in {time.time() - t0:.0f}s", flush=True)
    return rows


def main():
    jobs = [(r, s) for r in cfg.REPS for s in cfg.SEEDS]
    with ProcessPoolExecutor(max_workers=5) as ex:
        rows = [row for part in ex.map(job, jobs) for row in part]
    pd.DataFrame(rows).to_csv(os.path.join(cfg.RESULTS, "overlap.csv"), index=False)


if __name__ == "__main__":
    main()
