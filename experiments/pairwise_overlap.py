"""Pairwise positivity diagnostic between all ten domains (features only, no labels).

For every unordered pair, a cross-fitted domain classifier gives the AUC and,
in each direction, the share of one domain's units that lie inside the other's
support (propensity <= 0.95). Writes ``results/pairwise_overlap.csv``.
"""
import itertools
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
import config as cfg


def job(args):
    rep, a, b = args
    from threadpoolctl import threadpool_limits

    from zop.data import load_domain
    from zop.overlap import diagnose

    with threadpool_limits(limits=1):
        Xa, Xb = load_domain(rep, a)[0], load_domain(rep, b)[0]
        r_ab = diagnose(Xa, Xb, seed=0, n_max=3000, folds=3)   # share of b inside a's support
        r_ba = diagnose(Xb, Xa, seed=0, n_max=3000, folds=3)   # share of a inside b's support
    return [dict(rep=rep, target=b, labelled=a, auc=r_ab.auc, overlap=r_ab.overlap_share),
            dict(rep=rep, target=a, labelled=b, auc=r_ba.auc, overlap=r_ba.overlap_share)]


def main():
    from zop.data import DOMAINS

    jobs = [(r, a, b) for r in cfg.REPS for a, b in itertools.combinations(DOMAINS, 2)]
    with ProcessPoolExecutor(max_workers=10) as ex:
        rows = [row for part in ex.map(job, jobs) for row in part]
    pd.DataFrame(rows).to_csv(os.path.join(cfg.RESULTS, "pairwise_overlap.csv"), index=False)


if __name__ == "__main__":
    main()
