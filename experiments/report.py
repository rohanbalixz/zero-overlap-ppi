"""Stage 5: figures and ``results/summary.md`` (every number quoted in the README)."""
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

sys.path.insert(0, os.path.dirname(__file__))
import config as cfg
from zop.data import CROPS, DOMAINS, LABEL, TARGETS

# ---- palette (reference instance of the dataviz skill; light surface) ----
SURF, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]
RAMP = ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7", "#3987e5", "#2a78d6",
        "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b"]
REPNAME = {"alphaearth": "AlphaEarth embeddings", "harmonic": "Harmonic features"}

plt.rcParams["svg.hashsalt"] = "zero-overlap-ppi"
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 10, "axes.titlesize": 11, "axes.titleweight": "semibold",
    "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "axes.linewidth": 0.8,
    "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelcolor": INK2, "ytick.labelcolor": INK2,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "grid.linestyle": "-",
    "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
    "legend.frameon": False, "text.color": INK, "lines.linewidth": 2,
    "lines.solid_capstyle": "round", "lines.solid_joinstyle": "round",
})


def _clean(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(length=0)


def end_labels(ax, ends, fmt, min_gap):
    """Direct end labels, skipping any that would collide with one already placed."""
    placed = []
    for x, y in ends:
        if all(abs(y - p) >= min_gap for p in placed):
            ax.annotate(fmt(y), (x, y), xytext=(6, 0), textcoords="offset points", va="center",
                        fontsize=8.5, color=INK2)
            placed.append(y)


def table_header(cols):
    return ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]


def covered(df):
    return (df.lo - 1e-9 <= df.truth) & (df.truth <= df.hi + 1e-9)


def save(fig, name):
    os.makedirs(cfg.FIGURES, exist_ok=True)
    # no timestamps and fixed SVG ids, so a re-run reproduces the files byte for byte
    fig.savefig(os.path.join(cfg.FIGURES, name + ".png"), dpi=200, bbox_inches="tight",
                metadata={"Software": None})
    fig.savefig(os.path.join(cfg.FIGURES, name + ".svg"), bbox_inches="tight", metadata={"Date": None})
    plt.close(fig)


# ---------------------------------------------------------------------------
def fig_overlap(po):
    cmap = LinearSegmentedColormap.from_list("seq", ["#f3f7fd"] + RAMP)
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 5.0))
    order = list(DOMAINS)
    for ax, rep in zip(axes, cfg.REPS):
        M = po[po.rep == rep].pivot(index="target", columns="labelled", values="overlap").reindex(
            index=order, columns=order).values
        ax.imshow(np.where(np.isnan(M), np.nan, M), cmap=cmap, vmin=0, vmax=1)
        ax.grid(False)
        for i in range(len(order)):
            for j in range(len(order)):
                v = M[i, j]
                if i == j:
                    ax.add_patch(plt.Rectangle((j - .5, i - .5), 1, 1, color="#f0efec", lw=0))
                elif v >= 0.10:
                    ax.text(j, i, f"{v:.2f}".lstrip("0"), ha="center", va="center", fontsize=8,
                            color="#ffffff" if v > 0.45 else INK)
        ticks = [LABEL[d] for d in order]
        ax.set_xticks(range(len(order)), ticks, rotation=45, ha="right")
        ax.set_yticks(range(len(order)), ticks)
        ax.set_xlabel("labelled domain (support)")
        if rep == cfg.REPS[0]:
            ax.set_ylabel("target domain")
        ax.set_title(REPNAME[rep], loc="left")
        for s in ax.spines.values():
            s.set_visible(False)
        ax.tick_params(length=0)
    fig.suptitle("Share of each target's pixels that lie inside a labelled domain's feature support",
                 x=0.06, ha="left", fontsize=12, fontweight="semibold")
    fig.text(0.06, -0.13, "Cross-fitted domain classifier on features only (propensity ≤ 0.95). "
             "Values below .10 are left blank.", color=INK2, fontsize=9)
    save(fig, "fig1_overlap")


METHODS_T = [("bound (gated)", "overlap-gated bound"), ("transported PPI", "transported PPI"),
             ("transported stratified", "transported stratified"),
             ("label shift (BBSE)", "label shift (BBSE)"), ("IPW-PPI", "IPW-PPI")]


def fig_coverage(tr):
    d = tr[(tr.scenario == "loco") & (tr.crop >= 0) & (tr.truth > 0)].copy()  # crops present in the target
    d["hit"] = covered(d)
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.3), sharey=True)
    for ax, rep in zip(axes, cfg.REPS):
        g = d[d.rep == rep].groupby(["method", "n_per"]).hit.mean()
        ends = []
        for (m, lab), col in zip(METHODS_T, SERIES):
            y = g.loc[m]
            ax.plot(y.index, y.values, color=col, marker="o", ms=4.5, mec=SURF, mew=1.5, label=lab)
            ends.append((y.index[-1], y.values[-1]))
        end_labels(ax, ends, lambda v: f"{v:.0%}", 0.04)
        ax.axhline(1 - cfg.ALPHA, color=MUTED, lw=1)
        ax.text(cfg.N_PER_GRID[0], 1 - cfg.ALPHA + 0.02, "nominal 90%", color=INK2, fontsize=8.5)
        ax.set_xscale("log", base=2)
        ax.set_xticks(cfg.N_PER_GRID, [str(n) for n in cfg.N_PER_GRID])
        ax.minorticks_off()
        ax.set_ylim(-0.02, 1.05)
        ax.set_xlabel("calibration labels per labelled domain")
        ax.set_title(REPNAME[rep], loc="left")
        ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
        _clean(ax)
    axes[0].set_ylabel("coverage of 90% intervals")
    axes[1].legend(loc="center left", bbox_to_anchor=(1.08, 0.5), fontsize=9)
    fig.suptitle("More labels from covered countries do not fix transport", x=0.06, ha="left",
                 fontsize=12, fontweight="semibold")
    fig.text(0.06, -0.04, "Leave-one-domain-out: every domain but the target is labelled. Coverage of the share of "
             "every crop present in the target, pooled over 9 targets and 100 replications.", color=INK2, fontsize=9)
    fig.text(0.06, -0.08, "The bound covers because it is honest, not because it is sharp: where fewer than two "
             "labelled domains overlap the target it is the whole interval [0, 1] (see Figure 3 and "
             "results/summary.md).",
             color=INK2, fontsize=9)
    save(fig, "fig2_coverage_vs_labels")


def _design_interval(rep, T, n_T, seed=0):
    from zop import estimators as E
    z = np.load(cfg.cache_path(rep, seed), allow_pickle=False)
    yT, pT = z[f"y|{T}"], z[f"pred|{T}"]
    rng = np.random.default_rng([7, cfg.REPS.index(rep), TARGETS.index(T)])
    idx = rng.choice(len(yT), n_T, replace=False)
    return E.target_ppi_adjusted(E.predicted_shares(pT), len(yT), yT[idx], pT[idx], cfg.ALPHA)


def fig_forest(tr, crop=0, rep="alphaearth", n_per=cfg.N_PER_MAIN, n_T=100):
    d = tr[(tr.rep == rep) & (tr.scenario == "loco") & (tr.crop == crop) & (tr.n_per == n_per)
           & (tr.seed == 0) & (tr.draw == 0)]
    fig, ax = plt.subplots(figsize=(8.6, 5.2))
    rows = list(TARGETS)[::-1]
    off = {"bound (gated)": 0.22, "transported PPI": 0.0, "design": -0.22}
    cols = {"bound (gated)": SERIES[0], "transported PPI": SERIES[1], "design": SERIES[2]}
    for i, T in enumerate(rows):
        r = d[d.target == T].set_index("method")
        truth = r.truth.iloc[0]
        for m in ("bound (gated)", "transported PPI"):
            lo, hi = r.loc[m, "lo"], r.loc[m, "hi"]
            ax.plot([lo, hi], [i + off[m]] * 2, color=cols[m], lw=2.2)
            ax.plot(r.loc[m, "est"] if m != "bound (gated)" else np.nan, i + off[m], "o",
                    color=cols[m], ms=4)
        iv = _design_interval(rep, T, n_T)
        ax.plot([max(iv.lo[crop], 0), min(iv.hi[crop], 1)], [i + off["design"]] * 2, color=cols["design"], lw=2.2)
        if r.loc["bound (gated)", "lo"] <= 1e-9 and r.loc["bound (gated)", "hi"] >= 1 - 1e-9:
            ax.text(1.0, i + off["bound (gated)"] + 0.07, "not identified: fewer than two overlapping labelled domains",
                    ha="right", va="bottom", fontsize=7.5, color=INK2)
        ax.plot(iv.est[crop], i + off["design"], "o", color=cols["design"], ms=4)
        ax.plot(truth, i, marker="D", color=INK, ms=6.5, mec=SURF, mew=1.5, zorder=5)
    ax.set_yticks(range(len(rows)), [LABEL[t] for t in rows])
    ax.set_xlim(-0.02, 1.02)
    ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.set_xlabel(f"{CROPS[crop]} share of the target")
    ax.grid(axis="y", visible=False)
    _clean(ax)
    from matplotlib.lines import Line2D
    handles = [Line2D([], [], color=INK, marker="D", ls="", ms=6, label="truth (hidden labels)"),
               Line2D([], [], color=cols["transported PPI"], lw=2.2,
                      label=f"transported PPI, {n_per} labels per other domain"),
               Line2D([], [], color=cols["bound (gated)"], lw=2.2, label="overlap-gated bound, same labels"),
               Line2D([], [], color=cols["design"], lw=2.2, label=f"PPI++ with {n_T} labels drawn inside the target")]
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=2, fontsize=9)
    ax.set_title(f"What each approach reports for the {CROPS[crop]} share (90% intervals)", loc="left",
                 pad=26)
    ax.text(0, 1.02, f"{REPNAME[rep]}; map trained in USA 2023; one replication (seed 0). Transported "
            "intervals are narrow and wrong,\nthe bound is honest, and a small sample inside the target "
            "is both.", transform=ax.transAxes, color=INK2, fontsize=8.5, va="bottom")
    save(fig, f"fig3_{CROPS[crop]}_{rep}")


def fig_design(de, tr):
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.3), sharey=True)
    for ax, rep in zip(axes, cfg.REPS):
        g = de[(de.rep == rep) & (de.truth > 0)].groupby(["method", "n_T"])
        w, c = g.width.mean(), g.coverage.mean()
        ends = []
        for (m, lab), col in zip([("PPI++ (adjusted)", "PPI++, small-sample adjusted"),
                                  ("PPI++ (Wald)", "PPI++, Wald (undercovers at small n_T)"),
                                  ("classical (Wilson)", "labels only (Wilson)")], SERIES):
            y = w.loc[m] / 2
            ax.plot(y.index, y.values, color=col, marker="o", ms=4.5, mec=SURF, mew=1.5, label=lab)
            if m != "PPI++ (Wald)":
                ends.append((y.index[-1], y.values[-1]))
        end_labels(ax, ends, lambda v: f"±{v:.3f}", 0.002)
        cov = c.loc["PPI++ (adjusted)"]
        ax.set_title(REPNAME[rep], loc="left")
        ax.text(0.98, 0.97, "adjusted PPI++ coverage\n" + ", ".join(f"{v:.0%}" for v in cov.values) +
                f"\nat n_T = {', '.join(map(str, cov.index))}", transform=ax.transAxes, fontsize=8.5,
                color=INK2, ha="right", va="top")
        ax.set_xscale("log", base=2)
        ax.set_xticks(cfg.N_T_GRID, [str(n) for n in cfg.N_T_GRID])
        ax.minorticks_off()
        ax.set_xlabel("labels drawn inside the target, n_T")
        _clean(ax)
    axes[0].set_ylabel("mean half-width of 90% interval")
    axes[1].legend(loc="center left", bbox_to_anchor=(1.12, 0.5), fontsize=9)
    fig.suptitle("Making the labelling probability positive by design", x=0.06, ha="left", fontsize=12,
                 fontweight="semibold")
    fig.text(0.06, -0.04, "Simple random sample inside each target; crops present in the target; "
             "9 targets × 1,000 replications.", color=INK2, fontsize=9)
    save(fig, "fig4_design")


def fig_accuracy(pop, val_acc):
    p = pop[pop.scenario == "loco"].groupby(["rep", "target"]).mean(numeric_only=True)
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6), sharey=True)
    rows = list(TARGETS)[::-1]
    for ax, rep in zip(axes, cfg.REPS):
        for i, T in enumerate(rows):
            r = p.loc[(rep, T)]
            ax.plot([0, r.acc_hi_croplist], [i + 0.17] * 2, color=SERIES[3], lw=2.2)
            ax.plot([r.acc_lo_stats, r.acc_hi_stats], [i - 0.17] * 2, color=SERIES[0], lw=2.2)
            ax.plot(r.acc, i, marker="D", color=INK, ms=6, mec=SURF, mew=1.5, zorder=5)
        ax.axvline(val_acc[rep], color=MUTED, lw=1)
        ax.text(val_acc[rep], len(rows) - 0.4, f"in-domain validation {val_acc[rep]:.0%}", ha="right",
                fontsize=8.5, color=INK2)
        ax.set_yticks(range(len(rows)), [LABEL[t] for t in rows])
        ax.set_xlim(0, 1.02)
        ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
        ax.set_xlabel("map accuracy in the target")
        ax.set_title(REPNAME[rep], loc="left")
        ax.grid(axis="y", visible=False)
        _clean(ax)
    from matplotlib.lines import Line2D
    handles = [Line2D([], [], color=INK, marker="D", ls="", ms=6, label="true accuracy (hidden labels)"),
               Line2D([], [], color=SERIES[3], lw=2.2, label="bound from the crop list alone"),
               Line2D([], [], color=SERIES[0], lw=2.2, label="bound from national crop shares alone")]
    axes[1].legend(handles=handles, loc="center left", bbox_to_anchor=(1.03, 0.5), fontsize=9)
    fig.suptitle("Aggregate statistics bound a map's accuracy where no pixel is labelled", x=0.06,
                 ha="left", fontsize=12, fontweight="semibold")
    fig.text(0.06, -0.03, "Bounds use the map's predictions in the target plus the aggregate only; "
             "no labels from the target and no transport assumption.", color=INK2, fontsize=9)
    save(fig, "fig5_accuracy_bounds")


# ---------------------------------------------------------------------------
def main():
    ov = pd.read_csv(os.path.join(cfg.RESULTS, "overlap.csv"))
    po = pd.read_csv(os.path.join(cfg.RESULTS, "pairwise_overlap.csv"))
    tr = pd.read_csv(os.path.join(cfg.RESULTS, "transport.csv.gz"))
    de = pd.read_csv(os.path.join(cfg.RESULTS, "design.csv"))
    pop = pd.read_csv(os.path.join(cfg.RESULTS, "population.csv"))
    rp = os.path.join(cfg.RESULTS, "reproduce.csv")
    rep_df = pd.read_csv(rp) if os.path.exists(rp) else None

    # in-domain validation accuracy (USA 2023 calibration half), averaged over seeds
    val_acc = {}
    for rep in cfg.REPS:
        a = []
        for s in cfg.SEEDS:
            z = np.load(cfg.cache_path(rep, s), allow_pickle=False)
            idx = z["calib|USA"]
            a.append(np.mean(z["y|USA"][idx] == z["pred|USA"][idx]))
        val_acc[rep] = float(np.mean(a))

    fig_overlap(po)
    fig_coverage(tr)
    fig_forest(tr)
    fig_design(de, tr)
    fig_accuracy(pop, val_acc)

    # ----------------------------- summary tables -----------------------------
    L = ["# Results summary", "", "Generated by `experiments/report.py`. All intervals are nominal 90%.", ""]
    cr = tr[tr.crop >= 0].copy()
    cr["hit"] = covered(cr)
    cr["width"] = cr.hi - cr.lo
    present = cr.truth > 0
    cr.groupby(["scenario", "rep", "n_per", "method", "target", "crop"]).agg(
        coverage=("hit", "mean"), mean_width=("width", "mean"), truth=("truth", "first"),
        mean_est=("est", "mean")).reset_index().to_csv(
        os.path.join(cfg.RESULTS, "transport_summary.csv"), index=False, float_format="%.5g")

    cp = cr[present]  # headline metric: crops present in the target (true share > 0)
    m0 = cp[(cp.scenario == "loco") & (cp.n_per == cfg.N_PER_MAIN)]
    h = m0.groupby(["method", "rep"]).agg(cov=("hit", "mean"), wid=("width", "median")).unstack("rep")
    # Monte Carlo s.e. of a pooled coverage: replications are the independent units, so pool
    # within each (seed, draw) replication and take the s.e. of the 100 replication means
    per_rep = m0.groupby(["method", "rep", "seed", "draw"]).hit.mean()
    mcse = (per_rep.groupby(["method", "rep"]).std(ddof=1)
            / np.sqrt(per_rep.groupby(["method", "rep"]).size())).unstack("rep")
    d100 = de[(de.truth > 0) & (de.n_T == 100) & (de.method == "PPI++ (adjusted)")]
    dd0 = d100.groupby("rep").agg(cov=("coverage", "mean"), wid=("width", "median"))
    # design.csv keeps per-seed means over 200 draws; the 5 seeds are the independent units
    per_seed = d100.groupby(["rep", "seed"]).coverage.mean()
    dd0["mcse"] = per_seed.groupby("rep").std(ddof=1) / np.sqrt(per_seed.groupby("rep").size())
    order = [("plug-in", "the map is unbiased in the target"),
             ("transported PPI", "the map's bias is the same everywhere"),
             ("transported stratified", "P(truth given prediction) is the same everywhere"),
             ("label shift (BBSE)", "P(prediction given truth) is the same everywhere"),
             ("IPW-PPI", "P(truth given features) is the same, and features overlap"),
             ("transported PPI (gated)", "as transported PPI, overlapping domains only"),
             ("bound (gated)", "error rates within the spread seen among overlapping domains")]
    L += ["## 0. Headline", "",
          f"Leave-one-domain-out, {cfg.N_PER_MAIN} labels per labelled domain, 100 replications; coverage of "
          "nominal 90% intervals for the share of every crop present in the target, with its Monte Carlo "
          "standard error in parentheses (across the 100 replications; for the design row, across the 5 "
          "map seeds of 200 draws each); median interval width.", "",
          "| method | assumption | AlphaEarth coverage | AlphaEarth width | harmonic coverage | harmonic width |",
          "|---|---|---|---|---|---|"]
    for meth, assum in order:
        r = h.loc[meth]
        se = mcse.loc[meth]
        L.append(f"| {meth} | {assum} | {r[('cov', 'alphaearth')]:.3f} ({se['alphaearth']:.3f}) | "
                 f"{r[('wid', 'alphaearth')]:.3f} | {r[('cov', 'harmonic')]:.3f} ({se['harmonic']:.3f}) | "
                 f"{r[('wid', 'harmonic')]:.3f} |")
    L.append(f"| PPI++ (adjusted), 100 labels inside the target | no transport assumption (design-based) | "
             f"{dd0.loc['alphaearth', 'cov']:.3f} ({dd0.loc['alphaearth', 'mcse']:.3f}) | "
             f"{dd0.loc['alphaearth', 'wid']:.3f} | "
             f"{dd0.loc['harmonic', 'cov']:.3f} ({dd0.loc['harmonic', 'mcse']:.3f}) | "
             f"{dd0.loc['harmonic', 'wid']:.3f} |")
    L += ["", "Gated PPI is computed only for targets with an overlapping labelled domain. The gated bound is "
          "[0, 1] wherever fewer than two labelled domains overlap the target (see section 2).", ""]

    L += ["## 1. Positivity diagnostic (labelled pool vs target)", ""]
    o = ov[ov.scenario == "loco"].groupby(["rep", "target"])[["auc", "overlap"]].mean()
    L += ["| representation | target | AUC | share of target in support |", "|---|---|---|---|"]
    for (rep, T), r in o.iterrows():
        L.append(f"| {rep} | {LABEL[T]} | {r.auc:.3f} | {r.overlap:.3f} |")
    L.append("")

    L += ["## 2. Transported corrections (no target labels)", "",
          f"Coverage for crops present in the target (primary) and over all 7 crops (absent crops have true "
          f"share 0), and median width for present crops, at {cfg.N_PER_MAIN} labels per labelled domain; "
          "100 replications.", ""]
    m = cr[cr.n_per == cfg.N_PER_MAIN]
    tab = m.groupby(["scenario", "rep", "method"]).agg(coverage=("hit", "mean")).join(
        m[present.loc[m.index]].groupby(["scenario", "rep", "method"]).agg(
            width=("width", "median"), cov_present=("hit", "mean")))
    L += ["| scenario | representation | method | coverage (present crops) | coverage (all crops) | median width |",
          "|---|---|---|---|---|---|"]
    for (sc, rep, meth), r in tab.iterrows():
        L.append(f"| {sc} | {rep} | {meth} | {r.cov_present:.3f} | {r.coverage:.3f} | {r.width:.3f} |")
    L.append("")

    ident = tr[(tr.crop < 0) & (tr.n_per == cfg.N_PER_MAIN) & (tr.method == "bound (gated)") & (tr.n_refs >= 2)]
    keys = set(zip(ident.scenario, ident.rep, ident.target))
    mi = m[(m.method == "bound (gated)") & present.loc[m.index]]
    mi = mi[[k in keys for k in zip(mi.scenario, mi.rep, mi.target)]]
    L += ["Overlap-gated bound where it is identified (>= 2 overlapping labelled domains): "
          + "; ".join(f"{sc}/{rp}: {', '.join(LABEL[t] for t in sorted(set(g.target)))} — coverage "
                      f"{g.hit.mean():.3f}, median width {g.width.median():.3f}"
                      for (sc, rp), g in mi.groupby(["scenario", "rep"])), ""]
    pn = pd.read_csv(os.path.join(cfg.RESULTS, "population.csv"))
    pn = pn[pn.scenario == "north"].groupby(["rep", "target"]).mean(numeric_only=True)
    fails = pn[pn.delta_T_all > pn.delta_hat_all]
    L += [f"North scenario: the target's own error-profile distance exceeds the leave-one-out benchmark in "
          f"{len(fails)} of {len(pn)} (representation, target) cells: "
          + ", ".join(f"{r}/{LABEL[t]}" for r, t in fails.index) + ".", ""]

    L += ["### Coverage of transported PPI as labels grow (loco, present crops)", ""]
    g = cp[(cp.scenario == "loco") & (cp.method == "transported PPI")].groupby(["rep", "n_per"]).hit.mean()
    L += ["| representation | " + " | ".join(map(str, cfg.N_PER_GRID)) + " |",
          "|---|" + "---|" * len(cfg.N_PER_GRID)]
    for rep in cfg.REPS:
        L.append(f"| {rep} | " + " | ".join(f"{g.loc[(rep, n)]:.3f}" for n in cfg.N_PER_GRID) + " |")
    L.append("")

    L += [f"### Overlap-gated bound, per target (loco, {cfg.N_PER_MAIN} labels per domain; present crops)", ""]
    b = m[(m.scenario == "loco") & (m.method == "bound (gated)")]
    bw = b[b.truth > 0].groupby(["rep", "target"]).width.median()
    bc = b[b.truth > 0].groupby(["rep", "target"]).hit.mean()
    acc_rows = tr[(tr.crop < 0) & (tr.n_per == cfg.N_PER_MAIN) & (tr.scenario == "loco")
                  & (tr.method == "bound (gated)")]
    nref = acc_rows.groupby(["rep", "target"]).n_refs.first()
    dh = acc_rows.groupby(["rep", "target"]).delta.median()
    mp = m[m.truth > 0]
    pp = mp[(mp.scenario == "loco") & (mp.method == "transported PPI")].groupby(["rep", "target"]).hit.mean()
    L += table_header(["representation", "target", "gated refs", "median delta_hat", "bound coverage",
                       "bound median width", "transported PPI coverage"])
    for (rep, T) in bc.index:
        dtxt = "–" if pd.isna(dh.loc[(rep, T)]) else f"{dh.loc[(rep, T)]:.2f}"
        L.append(f"| {rep} | {LABEL[T]} | {int(nref.loc[(rep, T)])} | {dtxt} | "
                 f"{bc.loc[(rep, T)]:.3f} | {bw.loc[(rep, T)]:.3f} | {pp.loc[(rep, T)]:.3f} |")
    L.append("")

    gp = mp[(mp.method == "transported PPI (gated)")]
    if len(gp):
        L += ["### Transported PPI restricted to overlapping domains (where any exist; present crops)", ""]
        t2 = gp.groupby(["scenario", "rep", "target"]).hit.mean()
        L += ["| scenario | representation | target | coverage |", "|---|---|---|---|"]
        for (sc, rep, T), v in t2.items():
            L.append(f"| {sc} | {rep} | {LABEL[T]} | {v:.3f} |")
        L.append("")

    L += ["## 3. Design: a probability sample inside the target", ""]
    dd = de[de.truth > 0].groupby(["rep", "method", "n_T"]).agg(coverage=("coverage", "mean"),
                                                                 half_width=("width", lambda x: x.mean() / 2))
    dall = de.groupby(["rep", "method", "n_T"]).coverage.mean()
    L += ["| representation | method | n_T | coverage (present) | coverage (all crops) | mean half-width |",
          "|---|---|---|---|---|---|"]
    for (rep, meth, n), r in dd.iterrows():
        L.append(f"| {rep} | {meth} | {n} | {r.coverage:.3f} | {dall.loc[(rep, meth, n)]:.3f} | {r.half_width:.4f} |")
    L.append("")
    w = de[de.truth > 0].groupby(["rep", "method", "n_T"]).width.mean()
    L += ["Effective-sample-size gain of adjusted PPI++ over labels-only Wilson, (width_Wilson / width_PPI++)^2:", ""]
    for rep in cfg.REPS:
        L.append(f"* {rep}: " + ", ".join(
            f"n_T={n}: {(w.loc[(rep, 'classical (Wilson)', n)] / w.loc[(rep, 'PPI++ (adjusted)', n)]) ** 2:.2f}x"
            for n in cfg.N_T_GRID))
    L.append("")

    L += ["## 4. Population-level identification (all calibration labels)", ""]
    pl = pop.groupby(["scenario", "rep", "target"]).mean(numeric_only=True)
    L += table_header(["scenario", "rep", "target", "map acc", "plug-in max err", "transported max err",
                       "delta_T (all)", "delta_hat (all)", "delta* (gated)", "delta_hat (gated)",
                       "width at delta_hat (gated)", "width at delta* (gated)", "acc cap (crop list)",
                       "acc bounds (national shares)"])
    for (sc, rep, T), r in pl.iterrows():
        f = lambda v: "–" if pd.isna(v) else f"{v:.2f}"  # noqa: E731
        L.append(f"| {sc} | {rep} | {LABEL[T]} | {r.acc:.3f} | {r.plugin_err:.3f} | {r.transported_err:.3f} | "
                 f"{f(r.delta_T_all)} | {f(r.delta_hat_all)} | {f(r.delta_star_gated)} | {f(r.delta_hat_gated)} | "
                 f"{f(r.width_hat_gated)} | {f(r.width_star_gated)} | {r.acc_hi_croplist:.3f} | "
                 f"[{r.acc_lo_stats:.3f}, {r.acc_hi_stats:.3f}] |")
    L.append("")
    L.append("In-domain validation accuracy (USA 2023 held-out half): "
             + ", ".join(f"{k} {v:.3f}" for k, v in val_acc.items()))
    L.append("")
    if rep_df is not None:
        L += ["## 5. Reproduction check (AlphaEarth, RF 500 trees, as in Tong & Wang)", ""]
        rr = rep_df.groupby("target").agg(oa=("oa", "mean"), sd=("oa", "std"), published=("published", "first"))
        L += ["| target | this repo OA % (3 runs) | published OA % (Tables 7 and 11) |", "|---|---|---|"]
        for T, r in rr.iterrows():
            pub = "–" if pd.isna(r.published) else f"{r.published:.2f}"
            L.append(f"| {LABEL[T]} | {r.oa:.2f} ± {r.sd:.2f} | {pub} |")
        L.append("")
    with open(os.path.join(cfg.RESULTS, "summary.md"), "w") as fh:
        fh.write("\n".join(L))
    print("\n".join(L))


if __name__ == "__main__":
    main()
