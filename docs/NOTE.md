# Technical note: prediction-powered inference when the labelling probability is zero

Rohan Bali · November 2026

This note states the problem, the four transport assumptions that the
baselines rely on, the bound used here and its guarantees, and the
limitations of the study. The numbers it refers to are produced by
`make all` and collected in [`results/summary.md`](../results/summary.md).

## 1. Setup

There are ten **domains**: eight countries in 2023 (USA, FRA, BEL, NLD, GBR,
CHN, AUS, ARG) and the USA in 2017 and 2020, all from CropGlobe
(Tong & Wang, arXiv:2509.03497). A **map** $f$ is a gradient-boosted
classifier trained on half of USA 2023, as in the paper's single-source
protocol. It assigns each pixel one of $K = 7$ classes (corn, soybeans, rice,
wheat, sugarcane, cotton, other).

Fix a **target** domain $T$ with $N_T$ units. We observe every feature
vector $X_i$ in $T$, and therefore every prediction $f(X_i)$. We observe no
label. The estimand is the target's crop-share vector

$$\theta_{T,c} = \frac{1}{N_T}\sum_{i \in T} \mathbf 1 \lbrace Y_i = c \rbrace, \qquad c = 1,\dots,K.$$

Write $q_T$ for the predicted shares (observed exactly). Then

$$\theta_T = q_T + \Delta_T, \qquad \Delta_T = \mathbb E_T [ \mathbf 1_Y - \mathbf 1_{f(X)} ].$$

Prediction-powered inference (PPI) estimates the rectifier $\Delta_T$ from
labelled units. Kluger et al. (2025) allow nonuniform labelling, but their
Assumption 1 requires the labelling probability
$\pi(\tilde X) \equiv P(I = 1 \mid \tilde X)$ to be known (part iii) and to
satisfy the overlap condition $a \le \pi(\tilde X) \le b$ almost surely, for
some $a, b \in (0, 1)$ (part iv). Here $\tilde X$ collects what is observed for
every unit; in this study that is the features, the map's prediction and the
unit's domain. Every unit of $T$ has $\pi(\tilde X) = 0$, because the labels
sit in other domains $S(T)$, so (iv) fails.

**Observation 1 (nothing is identified without a link).** Let
$J_T[c,k] = P_T(Y = c, f = k)$. Its column sums are $q_T$ and its row sums
are $\theta_T$. For any $\theta$ in the simplex, the independent coupling
$J = \theta q_T^{\top}$ has column sums $q_T$ and row sums $\theta$. So the
data from $T$ are consistent with every share vector. Labels from other
domains enter only through an assumption that links $J_T$ to them.

## 2. Transport assumptions behind the baselines

| | assumption linking $T$ to labelled domains $S$ | estimator | checkable without target labels? |
|---|---|---|---|
| A1 | $\Delta_T = \Delta_S$ (rectifier invariance) | transported PPI | no |
| A2 | $P_T(Y \mid f) = P_S(Y \mid f)$ (predictive values) | transported stratified (Olofsson et al., 2014) | no |
| A3 | $P_T(f \mid Y) = P_S(f \mid Y)$ (label shift) | BBSE (Lipton et al., 2018) | partly (see §3, $\delta = 0$) |
| A4 | $P_T(Y \mid X) = P_S(Y \mid X)$ **and** bounded $dP_T/dP_S$ | IPW-PPI with an estimated propensity | the overlap half, yes |

A4 is what weighting needs. Its overlap half can be tested from features
alone, and §5 does this. In CropGlobe's AlphaEarth embeddings it fails
outright. A domain classifier separates every pair of countries with
AUC ≥ 0.99, except Belgium and the Netherlands (0.95). It also places
essentially none of China, Australia or Argentina inside the labelled
support.

## 3. A bound that says how far the labelled countries reach

**Error profiles.** For domain $j$ and crop $c$, let
$C_j[c] = P_j(f = \cdot \mid Y = c)$, a probability vector over predicted
classes. For the target, $\rho_{T,c} = P_T(f = \cdot \mid Y = c)$, which is
unobserved.

**Assumption $\mathcal A(\delta)$ (hull model).** Fix reference domains
$R \subseteq S(T)$. For every crop $c$ present in $T$,

$$\mathrm{dist}_1 \big( \rho_{T,c},\ \mathrm{conv} \lbrace C_j[c] : j \in R \rbrace \big) \le \delta_c .$$

In words, for each crop the target's map errors are a mixture of those seen
in labelled domains, up to an $L_1$ slack of $\delta_c$ (which lies in $[0, 2]$).

**Proposition 1 (sharp bounds are linear programs).** Let $V_c$ stack the
reference rows for crop $c$. Define $\Theta(\delta)$ as the set of
$\theta$ for which there exist $J \ge 0$, $\Lambda \ge 0$, $s \ge 0$ such that

$$J^{\top}\mathbf 1 = q_T,\quad J\mathbf 1 = \theta,\quad \Lambda\mathbf 1 = \theta,\quad |J_{c\cdot} - \Lambda_{c\cdot}V_c| \le s_{c\cdot},\quad s_{c\cdot}\mathbf 1 \le \delta_c \theta_c\ \ \forall c .$$

Then $\theta_T \in \Theta(\delta)$ under $\mathcal A(\delta)$. Conversely, every
$\theta \in \Theta(\delta)$ is the share vector of some target population
that has predicted shares $q_T$ and satisfies $\mathcal A(\delta)$. Hence
$[\min_{\Theta(\delta)} \theta_c,\ \max_{\Theta(\delta)} \theta_c]$ are sharp
bounds, and so is the corresponding interval for the map's target accuracy
$\mathrm{tr}  J$.

*Proof.* ($\Rightarrow$) Take $J = J_T$. Let $\lambda_c^\star$ attain the
distance in $\mathcal A(\delta)$, set $\Lambda_{c\cdot} = \theta_{T,c}\lambda_c^\star$
and $s = |J - \Lambda V|$. Then

$$\sum_k s_{ck} = \theta_{T,c} \cdot \mathrm{dist}_1 \le \delta_c \theta_{T,c} .$$

($\Leftarrow$) Given a feasible $(J, \Lambda, s)$, take $c$ with $\theta_c \gt 0$.
Then $\rho_c = J_{c\cdot}/\theta_c$ and $\lambda_c = \Lambda_{c\cdot}/\theta_c$
are probability vectors, and $\Vert \rho_c - \lambda_c V_c \Vert_1 \le \delta_c$. If
$\theta_c = 0$, the constraints force $J_{c\cdot} = 0$. The population with
joint distribution $J$ realises $\theta$. ∎

Three properties follow. $\Theta(\delta)$ is convex and nested in $\delta$.
At $\delta \ge 2$ it is the whole simplex (Observation 1). With one reference
and $\delta = 0$ it becomes $\lbrace \theta : C^{\top}\theta = q_T \rbrace$, the label-shift
(BBSE) solution, which is a single point when $C$ is invertible. Crops with no
reference row are left unrestricted. Known-absent crops (a *crop list*) enter
as $\theta_c = 0$, and aggregate statistics enter as box constraints on
$\theta$. All three are linear.

**Proposition 2 (estimated references).** Suppose that, for every crop $c$,
$\Vert \hat C_j[c] - C_j[c] \Vert_1 \le \varepsilon_c$ for all $j \in R$; call this event $E$. On
$E \cap \mathcal A(\delta)$, $\theta_T \in \hat\Theta(\delta + \varepsilon)$, the
set computed from the estimated rows. *Proof:* any point of
$\mathrm{conv} \lbrace C_j[c] \rbrace$ is within $\max_j \Vert \hat C_j[c] - C_j[c] \Vert_1$ of
the same mixture of the $\hat C_j[c]$. Then use the triangle inequality. ∎
We set $\varepsilon_c$ to the $(1-\alpha/r)$ quantile of a parametric
bootstrap of each row's $L_1$ error, with a Bonferroni correction over the
$r$ reference rows. That makes $P(E)$ at least $1 - \alpha$, up to the bootstrap approximation.

**Proposition 3 (calibrating $\delta$ from the labelled domains).** Let
$R = \lbrace 1,\dots,m \rbrace$ and define

$$d_j = \max_c \mathrm{dist}_1 \big( C_j[c],\ \mathrm{conv} \lbrace C_i[c] : i \in R \setminus \lbrace j \rbrace \rbrace \big),\qquad \hat\delta = \max_j d_j,$$

and let $d_T$ be the target's own distance to the hull of all $m$ references,
maximised over crops. If the $m+1$ error-profile arrays
$(C_1,\dots,C_m,\rho_T)$ are exchangeable, then
$P(d_T \gt \hat\delta) \le 1/(m+1)$.

*Proof.* Let $e_i$ be domain $i$'s distance to the hull of the other $m$
domains among all $m+1$. By exchangeability, $P(e_T \gt \max_{j\in R} e_j) \le 1/(m+1)$.
Now $e_T = d_T$, and $e_j \le d_j$ because adding $\rho_T$ can only enlarge
the hull. Hence $\lbrace d_T \gt \hat\delta \rbrace \subseteq \lbrace e_T \gt \max_j e_j \rbrace$. ∎

This is the same rank argument as split conformal prediction, with **regions
as the exchangeable units**, and it is only as good as that assumption. When
labels exist only where national statistics are strong, the target is
exactly the domain that is *not* exchangeable with the references; the
`north` scenario tests this case. The proposition also assumes every crop is
present in every domain, which CropGlobe does not satisfy, so with unequal
crop lists treat it as a heuristic. Finally, in the code $\hat\delta$ is
computed from the *estimated* reference profiles (a plug-in): Proposition 2's
slack covers the estimation error of the vertices, but not that of
$\hat\delta$ itself.

**Overlap gate.** References are restricted to labelled domains $j$ whose
feature-space support covers at least $\tau$ of the target. The coverage is
measured by the pairwise domain-classifier diagnostic, which uses only
features, so the gate uses no target labels. With fewer than two such
domains, $\delta$ cannot be calibrated and the bound is reported as the
simplex, meaning "not identified from these labels". We use $\tau = 0.1$
and report $\tau = 0.3$. The gate makes the exchangeability in Proposition 3
more plausible, but it does not make it true.

## 4. Restoring positivity by design

Drawing a simple random sample of $n_T$ target units makes
$\pi = n_T/N_T \gt 0$ and known, so Assumption 1 of Kluger et al. holds by
construction. This is the simplest two-phase design; Kluger & Bates (2026)
study adaptive multiwave versions. We use PPI ($\lambda = 1$) and PPI++
(power-tuned $\lambda$; Angelopoulos, Duchi & Zrnic, 2023) with the same map. PPI++ is asymptotically
no worse than the classical estimator. This works however biased the map is
in $T$: PPI needs the map to be *correlated* with the truth, not unbiased.

**Small samples.** With hard predictions, each crop's $(Y_c, F_c)$ on the
sampled units is a matched pair of indicators. Where the map makes few
errors, the residual $Y_c - \lambda F_c$ is often identically zero in a
sample of 25–100 units. The Wald variance is then zero and the interval
misses. In our runs this drove pooled unadjusted PPI++ coverage down to
78–88% at $n_T \le 100$. Following Agresti & Min (2005) for matched proportions, we add
half a pseudo-unit to each cell of the $2 \times 2$ table before computing
the moments, $\lambda$ and the variance (`estimators.target_ppi_adjusted`).
This brings pooled coverage to 89.8–92.4% at every $n_T$ (88–97% per
target at $n_T = 100$) and widens the interval only slightly. Labels-only intervals use
the Wilson score interval.

## 5. Protocol

* **Map:** `HistGradientBoostingClassifier` (up to 300 iterations, otherwise
  scikit-learn defaults, including early stopping on 10% of the training data)
  on a random half of USA 2023. There are 5 seeds; each seed redraws the split
  and the model seed.
  A 500-tree random forest reproduces the paper's published accuracies
  (`experiments/reproduce.py`).
* **Splits:** each labelled domain is split in half. Calibration labels are drawn
  from one half, and the domain classifier is fit on the other. For USA 2023,
  the half not used to train the map is split this way.
* **Transported estimators** pool the calibration draws of all labelled
  domains, with equal numbers per domain. IPW-PPI clips propensities to
  $[0.01, 0.99]$ before forming weights.
* **Scenarios:** `loco` labels every domain except the target (nine targets).
  `north` labels only USA 2023, FRA, BEL, NLD and GBR, with targets CHN, AUS,
  ARG, USA 2017 and USA 2020.
* **Transport study:** $n \in \lbrace 50,\dots,1600 \rbrace$ labels per labelled domain,
  20 draws per seed (100 replications), 90% intervals.
* **Design study:** $n_T \in \lbrace 25,\dots,400 \rbrace$, 200 draws per seed
  (1,000 replications).
* **Coverage metric:** the share of nominal-90% intervals that contain the true
  share, computed per crop **present in the target** (true share $\gt 0$) and
  pooled over targets and replications. Coverage over all seven crops is
  also reported. It is higher for some baselines, because a crop absent from
  the target is easy to estimate at zero.
* **Monte Carlo error.** Coverage is estimated by simulation. For the
  transport study, the standard error is computed across the 100
  replications, each pooled over targets and crops. For the design study it
  is computed across the 5 map seeds, each averaging 200 draws. For every
  headline coverage it is at most 0.008 (`results/summary.md`, section 0), so
  the gaps the conclusions rest on are hundreds of standard errors wide. At
  $n_T = 100$ the design-based coverage is 0.921 (s.e. 0.001) for AlphaEarth
  and 0.898 (0.001) for harmonic features, the latter 0.2 points below
  nominal.
* **Positivity diagnostic:** a cross-fitted (5-fold) gradient-boosted domain
  classifier on a balanced sample. The reported quantities are the AUC and the
  share of target units with propensity $\le 0.95$.

## 6. Limitations

1. **The shares are benchmark shares.** CropGlobe samples are class-balanced
   draws from national maps, so $\theta_T$ is the composition of the
   benchmark sample, not crop area. The identification question does not
   depend on this; the magnitudes do.
2. **"Truth" is a reference map.** CropGlobe labels come from national crop
   maps, which have their own errors.
3. **Single-source map.** With a map trained on all labelled domains, every
   reference profile would be in-distribution. The benchmark would then need
   nested leave-one-domain-out maps; the code does not do this.
4. **No spatial structure.** CropGlobe has no coordinates. Units are treated
   as independent within a domain, and the intervals ignore spatial
   autocorrelation, as Lu et al. also note for their intervals.
5. **Hard predictions.** Assumptions A2 and A3, and the hull model, are
   stated for hard labels. Soft scores would change the error-profile
   geometry.
6. **The gate and $\tau$ are choices.** They are reported, not tuned on
   target labels. The $\tau = 0.3$ variant is in the results.
7. **The crop list comes from the target's map legend**, which is the same
   side information Tong & Wang use for category alignment (their §4.4). In
   CropGlobe, for example, Australian wheat is labelled "other".

## References

* Agresti, Min (2005). Simple improved confidence intervals for comparing matched proportions. *Statistics in Medicine* 24.
* Angelopoulos, Bates, Fannjiang, Jordan, Zrnic (2023). Prediction-powered inference. *Science*.
* Angelopoulos, Duchi, Zrnic (2023). PPI++: Efficient prediction-powered inference. arXiv:2311.01453.
* Crump, Hotz, Imbens, Mitnik (2009). Dealing with limited overlap in estimation of average treatment effects. *Biometrika*.
* Emmenegger, Stahler, Podimata (2026). Prediction-powered inference across many tasks for AI evaluation & social science research. arXiv:2605.29249.
* Kluger, Lu, Zrnic, Wang, Bates (2025). Prediction-powered inference with imputed covariates and nonuniform sampling. arXiv:2501.18577 (v3).
* Kluger, Bates (2026). M-estimation under two-phase multiwave sampling with applications to prediction-powered inference. arXiv:2602.16933.
* Kluger, Wang, Lobell (2021). Two shifts for crop mapping: leveraging aggregate crop statistics to improve satellite-based maps in new regions. *Remote Sensing of Environment* 262.
* Lipton, Wang, Smola (2018). Detecting and correcting for label shift with black box predictors. *ICML*.
* Lu, Kluger, Bates, Wang. Regression coefficient estimation from remote sensing maps. arXiv.
* Manski (2003). *Partial Identification of Probability Distributions*. Springer.
* Olofsson, Foody, Herold, Stehman, Woodcock, Wulder (2014). Good practices for estimating area and assessing accuracy of land change. *Remote Sensing of Environment* 148.
* Testa, Xu, Lei, Roeder (2025). Semiparametric semi-supervised learning under distribution shift and decaying overlap. arXiv.
* Tong, Wang (2026). Invariant features for global crop type classification. arXiv:2509.03497v3.
