"""Single source of truth for every experimental setting."""
import os

from zop.data import DOMAINS, TARGETS

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
CACHE = os.path.join(ROOT, "results", "cache")
RESULTS = os.path.join(ROOT, "results")
FIGURES = os.path.join(ROOT, "figures")

REPS = ("alphaearth", "harmonic")
SEEDS = (0, 1, 2, 3, 4)          # map seeds (USA training split + model seed)
ALPHA = 0.10                     # all intervals are nominal 90%

# Transport study: labels exist only in the labelled domains of a scenario.
N_PER_GRID = (50, 100, 200, 400, 800, 1600)   # calibration labels per labelled domain
N_PER_MAIN = 800
DRAWS = 20                       # calibration draws per (seed, n_per) -> 100 replications
BBSE_B = 200                     # bootstrap replicates for the label-shift interval
SLACK_B = 2000                   # parametric bootstrap draws for reference-row slack
MIN_N = 20                       # a domain's crop row is a reference only with >= MIN_N labels

# Design study: a simple random sample drawn inside the target.
N_T_GRID = (25, 50, 100, 200, 400)
DESIGN_DRAWS = 200               # per seed -> 1000 replications

# Label geographies. "loco": every other domain is labelled (most favourable).
# "north": labels only where 2023 maps are published in the USA and four
# European countries; the targets are other continents and other years.
NORTH = ("USA", "FRA", "BEL", "NLD", "GBR")
SCENARIOS = {
    "loco": {T: tuple(d for d in DOMAINS if d != T) for T in TARGETS},
    "north": {T: NORTH for T in ("CHN", "AUS", "ARG", "USA_17", "USA_20")},
}

OVERLAP_NMAX = 4000
OVERLAP_FOLDS = 5


def cache_path(rep: str, seed: int) -> str:
    return os.path.join(CACHE, f"{rep}_seed{seed}.npz")

# Overlap gate: a labelled domain is a reference for target T only if at least
# TAU of T's units lie inside its feature-space support (pairwise diagnostic,
# features only). The bound needs >= 2 such domains to calibrate delta;
# otherwise it is reported as trivial ([0, 1]), i.e. "not identified here".
TAU = 0.10
TAU_ALT = 0.30
