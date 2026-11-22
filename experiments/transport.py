"""Stage 2: labels exist only in the labelled domains; the target has none.

For every (representation, seed, labels-per-domain n, draw) and every
scenario/target, draw n calibration labels from each labelled domain and
estimate the target's crop shares with each transported method, plus the
hull bound with its benchmarked delta and sampling slack.

Two families of reference sets are compared:
  * all labelled domains (what one does by default), and
  * overlap-gated: only labelled domains whose feature-space support covers at
    least TAU of the target (``results/pairwise_overlap.csv``; features only).

Writes ``results/transport.csv.gz`` (one row per method x crop x replication;
crop = -1 rows hold the bounds on the target map's accuracy).
"""
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
import config as cfg


def gated(po: pd.DataFrame, rep: str, T: str, S, tau: float):
    sub = po[(po.rep == rep) & (po.target == T)].set_index("labelled").overlap
    return tuple(j for j in S if sub.get(j, 0.0) >= tau)


def pooled(lab, doms):
    """Concatenate (labels, predictions, weights) of the calibration draws of several domains."""
    return [np.concatenate([lab[j][i] for j in doms]) for i in range(3)]


def job(args):
    rep, seed, n_per = args
    from zop import bounds as B
    from zop import estimators as E
    from zop.data import K

    z = np.load(cfg.cache_path(rep, seed), allow_pickle=False)
    po = pd.read_csv(os.path.join(cfg.RESULTS, "pairwise_overlap.csv"))
    rng = np.random.default_rng([cfg.REPS.index(rep), seed, n_per, 2])  # deterministic
    rows = []

    def record(**kw):
        rows.append(dict(rep=rep, seed=seed, n_per=n_per, **kw))

    for draw in range(cfg.DRAWS):
        for scen, spec in cfg.SCENARIOS.items():
            for T, S in spec.items():
                yT, pT = z[f"y|{T}"], z[f"pred|{T}"]
                truth = np.bincount(yT, minlength=K) / len(yT)
                q = E.predicted_shares(pT)
                N_T = len(yT)
                acc_true = float(np.mean(yT == pT))
                absent = [c for c in range(K) if truth[c] == 0]
                lab = {}
                for j in S:
                    pool = z[f"calib|{j}"]
                    pick = rng.choice(len(pool), min(n_per, len(pool)), replace=False)
                    idx = pool[pick]
                    lab[j] = (z[f"y|{j}"][idx], z[f"pred|{j}"][idx], z[f"w|{scen}|{T}|{j}"][pick])
                prof = {j: E.error_profile(v[0], v[1]) for j, v in lab.items()}

                yL, pL, wL = pooled(lab, S)
                out = {
                    "plug-in": E.plugin(q),
                    "transported PPI": E.transported_ppi(q, N_T, yL, pL, cfg.ALPHA),
                    "transported stratified": E.transported_stratified(q, yL, pL, cfg.ALPHA),
                    "label shift (BBSE)": E.label_shift(q, yL, pL, cfg.ALPHA, cfg.BBSE_B, rng),
                    "IPW-PPI": E.ipw_ppi(q, N_T, yL, pL, wL, cfg.ALPHA),
                }
                near = gated(po, rep, T, S, cfg.TAU)
                if near:
                    yG, pG, _ = pooled(lab, near)
                    out["transported PPI (gated)"] = E.transported_ppi(q, N_T, yG, pG, cfg.ALPHA)

                bound_specs = [("bound (all labelled)", S, ()),
                               ("bound (gated)", near, ()),
                               ("bound (gated) + crop list", near, absent),
                               ("bound (gated, tau=0.3)", gated(po, rep, T, S, cfg.TAU_ALT), ())]
                for name, refs, zero in bound_specs:
                    if len(refs) >= 2:
                        verts = B.references(prof, refs, cfg.MIN_N)
                        d_hat = B.benchmark_delta(prof, refs, cfg.MIN_N)
                        slack = B.l1_slack(prof, refs, cfg.MIN_N, cfg.ALPHA, cfg.SLACK_B, rng)
                        r = B.identified_set(q, verts, d_hat + slack, zero=zero)
                        lo, hi, alo, ahi, feas = r.lo, r.hi, r.acc_lo, r.acc_hi, r.feasible
                        sl = float(slack.mean())
                    else:  # not identified: report the trivial set (crop list still applies)
                        hi = np.ones(K)
                        hi[list(zero)] = 0.0
                        lo = np.zeros(K)
                        alo = 0.0
                        ahi = 1.0 - float(q[list(zero)].sum()) if zero else 1.0
                        d_hat, sl, feas = np.nan, np.nan, True
                    out[name] = E.Interval(est=0.5 * (lo + hi), lo=lo, hi=hi)
                    record(draw=draw, scenario=scen, target=T, method=name, crop=-1, est=np.nan,
                           lo=alo, hi=ahi, truth=acc_true, delta=d_hat, slack=sl, n_refs=len(refs),
                           feasible=feas)
                for name, iv in out.items():
                    for c in range(K):
                        record(draw=draw, scenario=scen, target=T, method=name, crop=c,
                               est=iv.est[c], lo=iv.lo[c], hi=iv.hi[c], truth=truth[c],
                               delta=np.nan, slack=np.nan, n_refs=np.nan, feasible=True)
    return rows


def main():
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    jobs = [(r, s, n) for n in cfg.N_PER_GRID for r in cfg.REPS for s in cfg.SEEDS]
    t0 = time.time()
    rows = []
    with ProcessPoolExecutor(max_workers=10) as ex:
        for i, part in enumerate(ex.map(job, jobs)):
            rows.extend(part)
            print(f"[transport] {i + 1}/{len(jobs)} jobs ({time.time() - t0:.0f}s)", flush=True)
    pd.DataFrame(rows).to_csv(os.path.join(cfg.RESULTS, "transport.csv.gz"), index=False)


if __name__ == "__main__":
    main()
