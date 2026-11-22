"""Stage 4: population-level identification analysis (no subsampling noise).

Uses every calibration-pool label of the labelled domains (about half of
each domain) to estimate error profiles, then reports per target:

* map accuracy and the plug-in's worst crop-share error;
* delta_T: the target's own distance from the reference hull (an oracle,
  needs the hidden target labels), and the breakdown delta* (smallest common
  delta whose identified set contains the true share vector);
* delta_hat: the leave-one-domain-out benchmark that is actually usable;
* widths of the bound at delta_hat (+ slack) and at the oracle delta*;
* label-free bounds on the target map's accuracy from (a) the crop list
  alone and (b) exact national crop shares, the kind of aggregate statistic
  used by Kluger, Wang & Lobell (2021).

Writes ``results/population.csv``.
"""
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
import config as cfg
from transport import gated


def job(args):
    rep, seed = args
    from zop import bounds as B
    from zop import estimators as E
    from zop.data import K

    z = np.load(cfg.cache_path(rep, seed), allow_pickle=False)
    po = pd.read_csv(os.path.join(cfg.RESULTS, "pairwise_overlap.csv"))
    rng = np.random.default_rng([cfg.REPS.index(rep), seed, 4])
    rows = []
    for scen, spec in cfg.SCENARIOS.items():
        for T, S in spec.items():
            yT, pT = z[f"y|{T}"], z[f"pred|{T}"]
            truth = np.bincount(yT, minlength=K) / len(yT)
            present = truth > 0
            absent = [c for c in range(K) if not present[c]]
            q = E.predicted_shares(pT)
            C_T, n_T = E.error_profile(yT, pT)
            prof = {}
            rect = []
            for j in S:
                idx = z[f"calib|{j}"]
                yj, pj = z[f"y|{j}"][idx], z[f"pred|{j}"][idx]
                prof[j] = E.error_profile(yj, pj)
                rect.append(np.bincount(yj, minlength=K) / len(yj) - E.predicted_shares(pj))
            near = gated(po, rep, T, S, cfg.TAU)
            row = dict(rep=rep, seed=seed, scenario=scen, target=T, n_refs_gated=len(near),
                       refs_gated=" ".join(near), acc=float(np.mean(yT == pT)),
                       plugin_err=float(np.abs(q - truth).max()),
                       transported_err=float(np.abs(q + np.mean(rect, 0) - truth).max()))
            for tag, refs in (("all", S), ("gated", near)):
                if len(refs) < 1:
                    row.update({f"delta_T_{tag}": np.nan, f"delta_star_{tag}": np.nan,
                                f"delta_hat_{tag}": np.nan, f"width_hat_{tag}": 1.0,
                                f"width_star_{tag}": 1.0, f"cover_hat_{tag}": True})
                    continue
                verts = B.references(prof, refs, cfg.MIN_N)
                dT = B.target_distances(C_T, n_T, verts)
                d_star = B.breakdown_delta(q, verts, truth)
                r_star = B.identified_set(q, verts, d_star + 1e-6, accuracy=False)
                if len(refs) >= 2:
                    d_hat = B.benchmark_delta(prof, refs, cfg.MIN_N)
                    slack = B.l1_slack(prof, refs, cfg.MIN_N, cfg.ALPHA, cfg.SLACK_B, rng)
                    r_hat = B.identified_set(q, verts, d_hat + slack, accuracy=False)
                    w_hat = float(np.mean((r_hat.hi - r_hat.lo)[present]))
                    cov = bool(np.all((r_hat.lo - 1e-7 <= truth) & (truth <= r_hat.hi + 1e-7)))
                else:
                    d_hat, w_hat, cov = np.nan, 1.0, True
                row.update({f"delta_T_{tag}": float(dT[present].max()),
                            f"delta_star_{tag}": d_star, f"delta_hat_{tag}": d_hat,
                            f"width_hat_{tag}": w_hat,
                            f"width_star_{tag}": float(np.mean((r_star.hi - r_star.lo)[present])),
                            f"cover_hat_{tag}": cov})
            # label-free accuracy bounds (no error-profile assumption: delta = 2)
            any_refs = B.references(prof, S, cfg.MIN_N)
            r_list = B.identified_set(q, any_refs, 2.0, zero=absent)
            r_stats = B.identified_set(q, any_refs, 2.0, box=(truth, truth))
            row.update(acc_hi_croplist=r_list.acc_hi, acc_lo_stats=r_stats.acc_lo,
                       acc_hi_stats=r_stats.acc_hi,
                       frechet_hi=float(np.minimum(truth, q).sum()))
            rows.append(row)
    return rows


def main():
    jobs = [(r, s) for r in cfg.REPS for s in cfg.SEEDS]
    with ProcessPoolExecutor(max_workers=5) as ex:
        rows = [row for part in ex.map(job, jobs) for row in part]
    pd.DataFrame(rows).to_csv(os.path.join(cfg.RESULTS, "population.csv"), index=False)


if __name__ == "__main__":
    main()
