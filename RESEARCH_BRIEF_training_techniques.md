# Research Brief: Training Techniques That Actually Move a 50%-Accuracy Crypto Directional Model

**Scope:** what has real out-of-sample evidence vs. what is folklore, with formulas, hyperparameters, and URLs.
**System under diagnosis:** Binance USDT-M perps, 5-min bars, 12-bar (1h) horizon, 30 scale-free features, LightGBM+CatBoost, ~1.26M rows, 12 symbols, ~366 days. Measured: 49.2–51.7% accuracy, −7.6 to −18.6 bps net edge at 12 bps cost, bootstrap P(edge>0)=0.08–0.27, OOS Sharpe −3.0 to −3.4, active_share 0.02–0.35%, ECE 0.007 → 0.49 after isotonic calibration.

---

## 0. Bottom line up front

Five findings, in descending order of importance for this specific system.

**0.1. Your measurement infrastructure is probably not the problem — your target/horizon is.** The most directly comparable public artifact is the `purgedcv` project's crypto selection-regret study: **daily BTC/USDT 2021–2023 with ordinary technical features, where no honest model has predictive power.** Even after switching to leakage-free purged CV, the *honest* selection was Ridge α=100 with **deployment Sharpe −0.26** (naive shuffled KFold picked RF and deployed at Sharpe **−0.77**). Both lost money; honest CV just lost less. ([github.com/eslazarev/purged-cross-validation](https://github.com/eslazarev/purged-cross-validation)) This is the single most relevant published data point: for crypto with technical/derived features, honest CV *changes which model you pick* but **does not manufacture edge**.

**0.2. Your ECE going 0.007 → 0.49 is a diagnostic, not a side issue.** An ECE of 0.49 after isotonic calibration means the calibration map is effectively anti-correlated with outcomes — the classic signature of fitting the calibrator on a set that is not exchangeable with the evaluation set (overlapping labels, or a calibrator fitted across a purged boundary). Tree ensembles are *already* reasonably calibrated out of the box; isotonic regression has unlimited flexibility and will happily destroy calibration when fit on highly autocorrelated overlapping samples. **Remove isotonic calibration entirely as a first action.** It cannot create edge and here it is provably destroying the probability scale you need for bet sizing.

**0.3. Meta-labeling is the highest-expected-value structural change**, and it is the one technique where the direction of the evidence is unambiguous: you stop asking the model to predict *side* and start asking it to predict *whether the primary signal is right*. Reported OOS effect from Hudson & Thames on S&P 500 E-mini with event-based sampling + triple barrier: **precision 0.48 → 0.54, accuracy 48% → 55%**.

**0.4. GBDT vs linear is a red herring at your signal strength.** Gu, Kelly & Xiu (2020) report monthly stock-level R²_oos of **0.34% for GBRT vs 0.16% for OLS-3** — GBDT wins, but *both are tiny*. The honest reading is that the choice of learner is a second-order effect; the first-order effects are the label, the feature's economic content, and the horizon. At daily/monthly equity horizon deep learning does not help (NN3 peaks at 0.40%, NN4/NN5 get *worse*: "the benefits of 'deep' learning are limited").

**0.5. At a 12 bps round-trip cost and a 1-hour horizon you need a per-trade gross edge of >12 bps.** The only documented microstructure result that clears that bar is *short-horizon* (seconds): Sokolovsky et al. find order-flow-imbalance features explain crypto short-horizon returns with tradable signals — but the prediction target there is a **3-second** mid-price return, and even then BTC's taker-strategy t-stat is **−0.67 (not significant)**. A 1-hour horizon with 5-min bars is in an evidence desert.

---

## 1. Label engineering

### 1.1 Triple-barrier (López de Prado, AFML ch.3) — event-based, and it is genuinely better than fixed-horizon

The method sets two horizontal price barriers (profit-take, stop-loss at ±volatility-scaled width) and one vertical barrier (max holding time). The label is *which barrier was touched first*. Reference implementations: [mlfinpy Labelling](https://mlfinpy.readthedocs.io/en/latest/Labelling.html), [mlfinlab](https://hudsonthames.org/mlfinlab/) (now commercial; the free fork is [purgedcv](https://github.com/eslazarev/purged-cross-validation)).

**Key claim that is NOT folklore:** fixed-horizon labeling produces massively overlapping, near-duplicate labels, because consecutive bars' 12-bar forward returns share 11 of 12 bars of information. Triple-barrier makes the label's *end time* data-dependent (`t1`), which (a) reduces overlap, and (b) makes volatility-scaling natural.

**Evidence for TB over plain direction labels.** A 2025 study using **5-minute bars on BTC/USD, EUR/USD and S&P 500** with supervised autoencoders directly asks the question ("Does triple barrier labeling improve classifier performance over simple direction classification?") and answers: *"Triple barrier labeling generally outperformed simple labeling due to its ability to handle market noise better."* It also reports that **every approach that rejected the null hypothesis involved either SAE or triple-barrier labeling.** Notably, in that same paper **BTC/USD had a −25.29% cumulative return and information ratio −0.50** in one configuration — TB is not a magic fix. [doi:10.1186/s40537-025-01267-7](https://doi.org/10.1186/s40537-025-01267-7)

**Concrete configuration notes.**
- Barrier width: scale to volatility, e.g. `barrier = k * sigma_t` with `sigma_t` an EWM of 5-min return std; k ∈ [1, 2] typical. Do **not** use a fixed bps width — it makes labels non-comparable across regimes, and you have 12 symbols with very different vol.
- Vertical barrier: your 12 bars is the vertical barrier. Reasonable. But it interacts with cost: with a 12 bps hurdle you need horizontal barriers *wider* than 12 bps or the label is dominated by cost noise.
- Add a **neutral/0 class** for the vertical-barrier outcome (as mlfinpy's 3-class form does: `{-1, 0, 1}`). The 2025 paper notes that on vertical-barrier touch "to minimize noise in results, we ideally stay out of the market in this case." Your `active_share_pct` of 0.02–0.35% suggests the model already learned most rows are untradeable — giving that a *label* instead of forcing a binary decision is a real change.

### 1.2 Meta-labeling — what the second model predicts and why it works

**Definition (from AFML ch.3, quoted verbatim in [Hudson & Thames](https://hudsonthames.org/meta-labeling-a-toy-example/)):**

> "I call this problem meta labeling because we want to build a secondary ML model that learns how to use a primary exogenous model... The ML algorithm will be trained to decide whether to take the bet or pass, a purely binary prediction. When the predicted label is 1, we can use the probability of this secondary prediction to derive the size of the bet, **where the side (sign) of the position has been set by the primary model**."

**What the second model predicts:** `y_meta = 1` if the primary model's side would have been *correct/profitable*, else 0. It predicts **P(primary is right)**, not direction.

**Why it works — four mechanisms, three of which are real:**
1. **Precision/recall decomposition.** Build a high-recall primary signal (accept many trades), then let the meta-model filter false positives to restore precision. This is a genuine and well-understood statistical decomposition. H&T: meta-labeling "will increase your F1-score by filtering out the false positives."
2. **Overfitting limitation.** "The effects of overfitting are limited when you apply meta labeling, because ML will not decide the side of your bet, only the size." The meta-model has a much lower-variance job.
3. **Asymmetric feature routing.** "The features driving a rally may differ from the features driving a sell-off" — you can train separate long-side and short-side meta-models.
4. **Position sizing** — the probability maps directly to bet size, which is what you want instead of a binary threshold.

**Documented OOS effect** ([Singh & Joubert, "Does Meta-Labeling Add to Signal Efficacy?"](https://hudsonthames.org/wp-content/uploads/2022/04/Does-Meta-Labeling-Add-to-Signal-Efficacy.pdf), S&P 500 E-mini, event-based (dollar/volume/tick) bars + triple barrier + random forest): on the **out-of-sample** set, **precision 0.48 → 0.54 and accuracy 48% → 55%.**

**Honest caveats.** This is one paper, two strategies, one asset, one decade. The authors explicitly write: *"If the algorithm is bad then meta-labeling would likely only reduce the downside."* Meta-labeling **reallocates** accuracy from recall to precision; it does not add information. **For your system this is the crucial point:** meta-labeling will *not* fix a negative edge in the primary signal. If your primary signal has no edge, meta-labeling converts "lose on every trade" into "lose on fewer trades" — which does improve your risk-adjusted numbers and is worth doing, but it is not the fix.

**Concrete recipe for your system:** primary = simple rule-based or high-recall signal (e.g. EMA20/50 gap sign, or a logistic regression on 3 features). Secondary = your GBDT ensemble. Target = `1` if realized 12-bar forward return signed by the primary side exceeds +12 bps (make the meta-label **cost-aware** — this is not in the book but is essential for you). Bet size = `2*P - 1` clipped, or `size ∝ max(0, P - 0.5)`.

### 1.3 Volatility-scaled barriers — real, and it is the standard

Barrier width should be `sigma_t * k`, with `sigma_t` a rolling/EWM realized vol. This is not really contested in the literature; it is how triple-barrier is specified in AFML and every implementation. The **practical** importance for you: it makes the label *stationary across regimes*, which your 30 scale-free features already attempt on the feature side. If your labels are not vol-scaled but your features are, you have a mismatch.

### 1.4 Trend-scanning labels — the best option if you want a *duration* model

From [mlfinlab docs](https://random-docs.readthedocs.io/en/latest/implementations/labeling_trend_scanning.html): fit OLS regressions from `t` to `t+L` for a range of `L` (e.g. L ∈ [5, 20]), and **select the L that maximizes the t-value of the slope coefficient.** Outputs: `t1` (timestamp of the farthest observation), `t-value`, trend return, bin.

**Why it is interesting and underrated:**
- Classification: sign of t-value → `{-1, +1}`; with a minimum |t| threshold → `{-1, 0, +1}` (a natural "no-trend" regime).
- **Regression: the t-value itself can be used as the regression target** — you are predicting trend *strength*, not just sign.
- **The t-values can be used directly as sample weights in classification.** This is the cleanest answer to your sample-weighting problem, because trend-scanning already emits a per-sample confidence.

**Flag:** trend-scanning has essentially **no independent OOS performance evidence** in the public literature. It is documented and implemented, its construction is principled, but nobody has published "trend-scanning beats triple-barrier by X bps." Treat it as *implementation-documented, unevidenced*.

### 1.5 Fixed-time vs event-based — the event-based claim has real but limited evidence

The H&T paper claims: *"by using tick data and converting into event based sampling methods such as volume, dollar or tick leads to better statistical properties of the data and that in turn helps machine learning algorithms learn and predict."* The underlying claim (Fama & Blume 1966: daily returns are more long-tailed than normal; Easley, López de Prado & O'Hara 2011 on volume clocks) is legitimate.

**For your system this is arguably the single biggest available change.** You are on 5-min *time* bars. Crypto volume is wildly non-uniform across the day and across symbols. Dollar bars / volume bars / **imbalance bars** would give you a roughly constant-information sampling. This is a data-construction change, not a model change, and it is cheap to try.

### 1.6 What actually helps — ranked

| Technique | Evidence grade | Verdict for you |
|---|---|---|
| Triple-barrier with vol-scaled barriers | **Real OOS evidence** (5-min bars, incl. BTC) | Do it. You may already have it — verify vol scaling and add a neutral class. |
| Meta-labeling on top of a high-recall primary | **Real but thin OOS evidence** (1 paper, precision .48→.54, acc 48%→55%) | Do it. Highest structural ROI. Won't fix negative edge but will fix the *shape* of the loss. |
| Event-based (dollar/volume/imbalance) bars | Real but indirect evidence (statistical properties, not P&L) | High-value, cheap to test. |
| Trend-scanning labels | **Folklore / implementation-documented only** | Try as an auxiliary target; t-values are excellent sample weights. |
| Vol-scaled barriers | Standard practice, uncontested | Already assumed; verify. |

---

## 2. Sample weighting — concrete math

### 2.1 Concurrency and uniqueness (AFML ch.4)

**Concurrency.** Labels `y_i` and `y_j` are concurrent at `t` if they are a function of at least one common return `r_{t-1,t}`. Define the concurrency count:

```
c_t = sum_i  1{ t ∈ [ t_{i,0}, t_{i,1} ] }        # how many labels span bar t
```

**Uniqueness of label i at bar t:**

```
u_{i,t} = 1 / c_t                                 # zero if t outside label i's span
```

**Average uniqueness of label i:**

```
u_bar_i = ( sum_{t : t ∈ [t_{i,0}, t_{i,1}]} u_{i,t} ) / ( t_{i,1} - t_{i,0} + 1 )
```

So a label that overlaps many others has low `u_bar_i`; an isolated label approaches 1. Source: [mlfinpy Data Sampling](https://mlfinpy.readthedocs.io/en/latest/Sampling.html) (functions `get_ind_matrix`, `get_ind_mat_average_uniqueness`, `get_ind_mat_label_uniqueness`), and [H&T sequential bootstrapping](https://hudsonthames.org/bagging-in-financial-machine-learning-sequential-bootstrapping-python/).

**Scale-free sanity check from the docs:** in their worked example, the first sample's average uniqueness is **0.2388** (computed two ways, agreeing to ~4 decimals) — i.e. in a typical overlapping TB dataset, the average label carries under a quarter of a bar's worth of unique information. **Compute your own average uniqueness first.** If it's ~0.2, your effective sample size is ~250k, not 1.26M — that reframes your whole confidence interval, and it is the most likely reason bootstrap P(edge>0) is 0.08–0.27.

**Turning uniqueness into weights.** The AFML scheme is a product of three components:

```
w_i  ∝  u_bar_i  *  |return_attribution_i|  *  time_decay_i
```

- **Return attribution weight:** `|sum of log returns over label i's span|`. Larger moves matter more.
- **Time-decay weight:** `w_i = d^{|i - T|} / Z`, `T` = index of the last observation, `Z` = normalizer. Per [mlfinpy docs](https://mlfinpy.readthedocs.io/en/latest/Sampling.html), the `decay` parameter semantics are: `1` = no decay; `0 < decay < 1` = linear decay with strictly positive weights; `0` = weights converge linearly to zero; `decay < 0` = the oldest portion receives exactly zero weight (erased from memory). Their example uses **`decay = 0.4`**.

**Flag:** the *uniqueness* weighting has solid theoretical grounding and is universally implemented. The **time-decay component is folklore in the sense that nobody has published a controlled OOS A/B showing decay=0.4 beats decay=1.0.** It is a reasonable prior (regimes drift), not an evidenced result. Tune it, don't assume it.

### 2.2 Sequential bootstrap — full algorithm

Goal: sample *with replacement* such that each bootstrap draw **maximizes average uniqueness**, rather than drawing IID.

```
phi = []                                  # chosen sample indices
while len(phi) < n_samples_to_bootstrap:
    for each candidate sample i:
        ind_mat_reduced = ind_mat[:, phi + [i]]
        avg_uniqueness[i] = get_ind_mat_average_uniqueness(ind_mat_reduced)
    # take the LAST column's mean over nonzero values = uniqueness of i given phi
    p = avg_uniqueness / sum(avg_uniqueness)      # normalize to probabilities
    draw next index ~ Categorical(p)
    phi.append(drawn)
```

Details confirmed from [mlfinpy](https://mlfinpy.readthedocs.io/en/latest/Sampling.html):
- **1st iteration:** all labels have equal probability, since the average uniqueness of a 1-column indicator matrix is 1.
- Worked example: after choosing sample 1, the probability array becomes `[0.357, 0.214, 0.429]` — sample 2 is favored.
- API: `seq_bootstrap(ind_mat, sample_length=None, warmup_samples=None, compare=False, verbose=False, random_state=...)`. `compare=True` prints standard-bootstrap uniqueness vs sequential-bootstrap uniqueness so you can see the gain.

**This is directly applicable to your LightGBM + CatBoost.** Both support per-row sample weights and both have stochastic subsampling. Feed `u_bar_i` (× attribution × decay) as `sample_weight`. For CatBoost specifically, you can additionally use the bootstrap-type/subsample options, but CatBoost does not expose sequential bootstrap natively — you would need a custom `get_bootstrap`-style hook or precompute weights.

**Evidence grade:** sequential bootstrap is a **well-specified algorithm with a clear theoretical objective**, implemented in three independent libraries. There is **no published controlled OOS experiment showing it improves trading P&L**. It is best described as *principled and unimplemented-by-you*, not *proven*.

### 2.3 What this means for your numbers

Your dataset is ~1.26M rows of 5-min bars across 12 symbols over ~366 days. With a **12-bar (1h) label horizon**, each label concurrency count `c_t` is roughly 12 × (number of concurrently-labeled symbols) ≈ 12 × 12 = **144**, if you label all symbols at every bar. That implies **average uniqueness on the order of 1/144 ≈ 0.007** if labels are aligned across symbols. Even accounting for unaligned end times, you are almost certainly in the 0.01–0.10 range.

**Your effective sample size is likely 2–4 orders of magnitude smaller than 1.26M.** This, not model choice, explains a bootstrap P(edge>0) of 0.08–0.27 and it explains why calibration catastrophically failed (calibration curves fitted on ~1.26M pseudo-independent points are fitted on noise). **Compute average uniqueness before anything else.** This is the single highest-value diagnostic available to you.

---

## 3. Feature engineering that genuinely predicts

### 3.1 Fractional differentiation — the strongest theoretical case, weak empirical case

**Why integer differencing is destructive.** From [H&T Fractional Differentiation](https://hudsonthames.org/fractional-differentiation/), quoting AFML ch.5:

> "Virtually all finance papers attempt to recover stationarity by applying an integer differentiation d = 1, which means that most studies have over-differentiated the series, that is, they have removed much more memory than was necessary to satisfy standard econometric assumptions."

**The math (FFD — fixed-width window fracdiff):**

Weights are generated by the binomial expansion:
```
w_0 = 1
w_k = -w_{k-1} * (d - k + 1) / k
X_tilde_t = sum_{k=0}^{inf} w_k * X_{t-k}
```

Infinite memory is a practical problem, so a fixed-width window truncates. With tolerance `tau ∈ [0,1]`, find `l*` such that `|w_{l*}| <= tau` and `|w_{l*+1}| > tau`:
```
w_tilde_k = w_k  if k <= l*,   else 0
X_tilde_t = sum_{k=0}^{l*} w_tilde_k * X_{t-k}
```
Note the window is *fixed-width*: it is always `l*+1` weights, not expanding. Reference implementation: `frac_diff_ffd(series, d)` and `plot_min_ffd(series)` in [mlfinpy](https://mlfinpy.readthedocs.io/en/latest/FractionalDifferentiated.html); `thresh` is the minimum-weight cut-off.

**How to choose d:** sweep d and plot (right axis) the ADF statistic on the downsampled series and (left axis) the correlation between the original and differenced series. **Pick the minimum d where ADF crosses the 95% critical value.** H&T's e-Mini S&P 500 figure shows ADF reaching the 95% critical value at **d < 0.2**, with **correlation > 90%** to the original series — versus d=1 where correlation is ~0.

**Honest empirical grade.** The *diagnostic* is reproducible and the reasoning is sound. But: **there is no published controlled study showing a fracdiff-augmented feature set produces higher OOS trading P&L than simple returns + the standard lag/rolling feature block.** The 2025 SAE paper does use fractional differencing with walk-forward validation and calls it "a robust framework... maximizing both predictive accuracy and statistical validity," but it does not isolate fracdiff's contribution. **Grade: theoretically strong, empirically unevidenced for P&L.**

**Practical note for you:** your 30 features already include EMA gaps, RSI, ATR%, 10-bar return, etc. — these are *exactly* the "recover memory via lagged returns and rolling statistics" workaround that fracdiff is proposed to replace. Adding `fracdiff(log(close), d≈0.3–0.5)` per symbol is cheap and worth testing, but do not expect a step change.

### 3.2 Microstructure / order-flow imbalance — the one area with hard evidence, but at the wrong horizon for you

**The foundational result:** Cont, Kukanov & Stoikov, *The Price Impact of Order Book Events* ([arXiv:1011.6402](https://arxiv.org/abs/1011.6402)) — order flow imbalance (OFI) is approximately linear in price changes, with the coefficient inversely proportional to market depth. This is among the most robust empirical regularities in market microstructure.

**Crypto-specific, with actual backtests.** *Explainable Patterns in Cryptocurrency Microstructure* ([arXiv:2602.00776](https://arxiv.org/html/2602.00776v1)) builds a unified feature library over tick orderbook+trade data for five crypto assets (BTC, ENJ, ETC, LTC, ROSE). Key results:

- **Prediction target: a 3-second mid-price log return** — `r_{t→t+3s} = log(mid_{t+3s}/mid_t)`. Not 1 hour.
- **SHAP-consistent across assets:** "order flow imbalance, bid–ask spreads, and VWAP-to-mid deviations dominate mean absolute SHAP."
- **Shape of the effect (economically meaningful):** "the effect of order flow imbalance on returns is predominantly monotone with concavity at extremes (diminishing incremental impact as pressure accumulates); wider spreads associate with attenuated predictive effects." — i.e. OFI saturates, and wide spreads kill predictability. Both are actionable priors.
- **Tradability, top-of-book taker backtest, marked pessimistically (buy at ask, sell at bid):**
  - BTC: ARC 0.13, IR* **0.25**, t-stat **−0.6671**, p **0.7474** — *not significant*.
  - ETC: IR* 8.97, t-stat 1.7208, p **0.0431** — significant.
  - ENJ: IR* 6.58, t-stat 1.7942, p **0.0368** — significant.
  - ROSE: IR* 7.00, t-stat 2.0777, p **0.0192** — significant (but "largely due to the extreme event on 2025-10-10").
  - LTC: IR* 0.07, t-stat −0.1736 — not significant.
  - **Maker backtest: all p > 0.05 for every asset** — no significance at all.
- The authors' own summary: "only the taker strategies on ETC, ENJ, and ROSE demonstrate statistically significant outperformance at the 5% level."

**Read this carefully.** Even at a 3-second horizon, with genuine L2 orderbook data, the prediction is *statistically* strong but the **economic** result on BTC is indistinguishable from zero, and ROSE's is driven by one flash crash. The pattern is: **microstructure edge concentrates in less-liquid alts and in tail events.**

**Implication for your system:** you are on 5-min bars, ~1h horizon, 12 majors-mostly symbols, and paying 12 bps. The microstructure literature does not support an edge there. If you want to use OFI, you must move the horizon down by 2–3 orders of magnitude (seconds), and you will then be competing with HFT firms on latency — and your 12 bps cost assumption becomes 12 bps *per second* of churn, which is fatal.

**Also relevant:** Stoikov's **micro-price**, a bid-ask-imbalance-adjusted mid, is a directly related and cheap feature.

### 3.3 Realized volatility / semivariance — strong OOS evidence, but for *volatility*, not direction

This is where the evidence is genuinely strong, and it is a hint about what you should be predicting (see §4).

- **HAR-RV** (Corsi, *A Simple Approximate Long-Memory Model of Realized Volatility*): forecast RV with daily, weekly, monthly realized-vol components. This is the workhorse and it has decades of OOS evidence.
- **Realized semivariance / "good" vs "bad" volatility** (Barndorff-Nielsen, Kinnebrock & Shephard; Amaya et al.; Bollerslev, Li & Zhao):
  - *Good Volatility, Bad Volatility, and the Cross Section of Stock Returns* ([doi:10.1017/S0022109019000097](https://doi.org/10.1017/s0022109019000097)): sorting stocks on **normalized good-minus-bad volatility** produces "economically large and highly statistically significant differences in subsequent portfolio returns," surviving controls for firm characteristics. → **Realized semivariance has documented cross-sectional return predictability.**
  - *Good and Bad Variance Premia and Expected Returns* ([doi:10.1287/mnsc.2017.2890](https://doi.org/10.1287/mnsc.2017.2890)): "The two variance premium components jointly predict excess returns over the next one and two years... **The R²s reach about 10% for aggregate equity and portfolio returns and 20% for corporate bond returns.**" → **R² of 10–20% from volatility decomposition is enormous by financial-ML standards** (compare GKX's 0.40% monthly).
- Realized skewness and realized kurtosis improve volatility forecasting OOS at daily/weekly/monthly horizons ([doi:10.1002/for.2813](https://doi.org/10.1002/for.2813)).
- Crypto-specific: *Good vs. bad volatility in major cryptocurrencies* ([doi:10.1016/j.intfin.2024.102062](https://doi.org/10.1016/j.intfin.2024.102062)) extends the good/bad decomposition to crypto.

**The actionable insight:** the *direction* of returns is nearly unpredictable, but the **magnitude** of returns is highly predictable (R² 10–20% at long horizons, and RV models do well at short horizons). Your model is fighting for 50% on the hard problem and ignoring the easy one. See §4.

### 3.4 Amihud illiquidity, Kyle lambda, Roll spread

Amihud illiquidity (|return| / dollar volume) is a well-established cross-sectional return predictor — and in the **Chinese** market, *Machine learning in the Chinese stock market* ([doi:10.1016/j.jfineco.2021.08.017](https://doi.org/10.1016/j.jfineco.2021.08.017)) finds **"liquidity emerges as the most important predictor,"** which then leads the authors to "closely examine the impact of transaction costs." That last part is the honest catch: liquidity predicts returns *precisely because* illiquid assets have high costs. **Grade: real, well-replicated equity evidence; but it is a cross-sectional, low-frequency, cost-arbitrage effect, and it does not transfer to 5-min perp direction.**

### 3.5 Cross-sectional ranking and feature neutralization

- **Cross-sectional ranking** (rank features within a timestamp across the 12 symbols, map to [-1,1] or z-score) is standard practice in equity ML because it removes time-varying level effects and is robust to outliers. In crypto perpetuals with 12 symbols, your cross-section is *small* — 12 names is thin but workable, and it would make your "z" features genuinely comparable across symbols.
- **Feature neutralization** (regress the feature/target against factor exposures, keep the residual) is standard at equity stat-arb shops.
- **Grade: widely used, essentially unevidenced in the public literature.** There is no clean OOS ablation published. Treat as *plausible practice*, and note that cross-sectional demeaning at a 12-symbol cross-section has much higher estimation noise than at a 3000-stock cross-section.

### 3.6 Feature ranking summary

| Feature family | Real OOS evidence? | Applicable to your 5-min/1h setup? |
|---|---|---|
| Order flow imbalance / trade imbalance | **Yes** — but at 1–3 second horizons | Only if you shorten horizon drastically |
| Realized volatility / HAR-RV | **Yes, strong** | Yes — for a *volatility* target |
| Realized semivariance (good/bad vol) | **Yes** — cross-sectional return prediction, R² 10–20% | Yes for vol; unproven for crypto direction |
| Realized skew / kurtosis | Yes (vol forecasting) | Cheap to add |
| Amihud / Kyle lambda / Roll | Yes (equities, cross-sectional, low-freq) | Weak transfer to perp direction |
| Fractional differentiation | **No P&L evidence**, strong theory | Try; low expectation |
| Cross-sectional rank/neutralize | No public ablation | Reasonable hygiene |
| Funding rate / basis carry | Plausible; perp pricing theory is solid | See §4.4 |

---

## 4. Targets better than return sign

This is where I would bet your largest expected gain, for a reason that is entirely evidence-based rather than speculative.

### 4.1 Direction vs value prediction — direction *is* better, and this is a real paper

*Direction is More Important than Speed: A Comparison of Direction and Value Prediction of Stock Excess Returns* (Cheng, Shang & Zhao, [SSRN 5176925](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5176925)), summarized by [Quantpedia](https://quantpedia.com/is-machine-learning-better-in-prediction-of-direction-or-value/). Specific numbers from the paper's Table 4:

| Model | Direction accuracy | Value accuracy |
|---|---|---|
| Logistic Regression | **54.91** | 52.08 (linear reg) |
| Random Forest | **61.31** | 58.78 |

F-scores diverge much more: **Lasso classification F-score 37.80 vs 1.45 for its value counterpart.**

The authors' mechanism: *"volatility itself carries predictive power: during high-volatility periods, markets tend to experience more negative returns, and vice versa"* — plus the Campbell-Shiller identity. And: *"models achieve higher accuracy and yield greater economic gains, mainly because of their stronger ability to predict market downturns."*

**Caveat:** these are equity/macro predictors at monthly frequency, not 5-min crypto. The accuracy levels (54–61%) are far above your 49–52%. But it does establish that **direction is a legitimate and tractable target** — your 50% is a statement about crypto 5-min/1h, not about directional prediction in general.

### 4.2 Volatility / magnitude prediction — the target where R² is genuinely large

Given §3.3 (harvesting R² of 10–20% from variance decomposition, vs 0.40% from return prediction), **volatility is 25–50× more predictable than returns.** A viable redesign:

1. Predict **realized volatility** over the next 12 bars (target: log RV, or RV quantiles). Expect real R².
2. Use that prediction to (a) **size positions** (vol targeting — a genuine, well-evidenced risk management technique), and (b) **set your triple-barrier widths and your entry threshold**, since a 12 bps cost hurdle is trivial when `sigma_12bar` is 200 bps and impossible when it is 20 bps.
3. Only take directional trades when *predicted sigma* >> *cost*, i.e. when `E[|move|] > k * cost`.

**This directly addresses your `active_share_pct = 0.02–0.35%` problem.** Right now your model is presumably filtering on direction confidence. It should be filtering on **expected move size relative to cost**. That's a different, and much more predictable, quantity.

### 4.3 Quantile regression

Quantile/distributional forecasting is well established for volatility and for tail risk, and there is practitioner evidence that quantile forecasts improve distributional accuracy "particularly in the tails" (Norbert Gehrke / Wilkens, [LinkedIn summary of "Beyond the Mean"](https://www.linkedin.com/posts/norbertgehrke_wilkens-beyond-the-mean-activity-7450543570064760833-xIpO)). LightGBM supports quantile objectives natively (`objective='quantile', alpha=q`). **Practically: train LightGBM quantile models at q ∈ {0.1, 0.5, 0.9} on the 12-bar forward return, and trade only when the interquartile range clears cost.** Grade: plausible, moderate evidence, cheap to implement.

### 4.4 Classification on the triple-barrier outcome + cost-aware meta-labels

Covered in §1. The key non-standard addition for you: **make the label cost-aware.** Standard triple-barrier labels a +5 bps touch as a "win." At 12 bps round-trip cost that is a loss. Set the profit barrier to at least `cost + margin`, e.g. `+25 bps`, and the stop to `-25 bps`. Otherwise you are training the model to detect moves that are *guaranteed unprofitable*. **This alone may explain a meaningful part of the −7.6 to −18.6 bps net edge:** if your labels reward sub-cost moves, the model learns to predict sub-cost moves, and the sign of the P&L is then determined entirely by the cost assumption.

### 4.5 Direction-aware loss functions (GMADL)

*Generalized Mean Absolute Directional Loss for Machine Learning Trading Models* ([SSRN 7236118](https://doi.org/10.2139/ssrn.7236118)). GMADL is a loss function designed to reward directional correctness directly rather than through squared error. Reported: *"the GMADL objective demonstrated robust improvement in both predictive accuracy and trading outcomes, especially on finer time intervals, underscoring the importance of direction-aware loss functions for short-horizon forecasting in liquid markets."* This is cited approvingly in the crypto microstructure paper, which uses GMADL-optimized models for its backtests.

Related: *Improving Forecasting Accuracy of Stock Market Indices Utilizing Attention-Based LSTM Networks with a Novel Asymmetric Loss Function* ([doi:10.3390/ai6100268](https://doi.org/10.3390/ai6100268)) reports **"test-time directional accuracy, with gains of 3.4–6.1 percentage points over MSE/MAE."**

**Grade: real, replicated across at least two independent papers, and directly targeted at exactly your failure mode (optimizing a metric that isn't the one you care about).** LightGBM/CatBoost do not have GMADL built in — you would need custom objectives. This is a real implementation cost, but the evidence is among the better in this brief.

### 4.6 Sharpe-ratio targets — no evidence, skip

I found **no published evidence** for predicting a Sharpe-ratio-like target directly as a supervised label. Sharpe is a *strategy-level* statistic, not a per-sample quantity; making it a label requires defining a local return/vol ratio, which is just a vol-normalized return (i.e. return minus volatility prediction, already covered). Treat "Sharpe-ratio targets" as folklore and spend the effort on the cost-aware barrier instead.

---

## 5. Model evidence

### 5.1 Do GBDTs beat Ridge/linear on noisy financial panels? Yes — by a little, which is the point

**Gu, Kelly & Xiu (2020), *Empirical Asset Pricing via Machine Learning*** ([doi:10.1093/rfs/hhaa009](https://doi.org/10.1093/rfs/hhaa009), NBER [w25398](https://www.nber.org/system/files/working_papers/w25398/w25398.pdf)). Verbatim Table 1, **monthly percentage R²_oos**, pooled panel, ~30,000 stocks:

| Model | R²_oos (%) |
|---|---|
| OLS (all 920 predictors) | **−3.46** |
| OLS-3 (size, B/M, momentum) | 0.16 |
| PLS | 0.27 |
| PCR | 0.26 |
| ENet+H | 0.11 |
| GLM+H | 0.19 |
| **RF+H** | **0.33** |
| **GBRT+H** | **0.34** |
| NN1 | 0.33 |
| **NN3** | **0.40** |
| NN4 | 0.39 |
| NN5 | 0.36 |

Annual-horizon version (Table 2): OLS **−34.86**, OLS-3 2.50, GBRT+H 3.09, **NN3+H 3.40**.

**Economic translation** (their Table 6 / Appendix A.7, using `SR* = sqrt((SR²+R²)/(1−R²))`):
- Timing the S&P 500 with NN3 forecasts: **Sharpe 0.77 vs 0.51 buy-and-hold** (+26pp).
- Long-short decile spread on NN4: **Sharpe 1.35 value-weighted, 2.45 equal-weighted.**
- Same strategy on **OLS benchmark: Sharpe 0.61 (VW) and 0.83 (EW).**

**Five lessons that bear directly on your system:**
1. **GBDT ≈ NN > linear, but the gap is ~0.18pp of R² per month.** Your model choice is not the bottleneck.
2. **OLS with 920 predictors gives R²_oos = −3.46%** — catastrophic overfit. Your 30 features with GBDT is a sane regime; more features is not obviously better.
3. **"Shallow learning outperforms deeper learning":** NN3 beats NN4/NN5. "The benefits of 'deep' learning are limited."
4. **Ensembling/regularization is what made trees work** — they report GBRT with **Huber loss** as the robust version ("perform better than the version without") and GBRT partitions on ~30 features early, rising to ~50 later. Note the Huber loss detail; it's a free win for heavy-tailed financial returns.
5. **Portfolio aggregation is where the Sharpe comes from.** They explicitly note: *"Aggregating into portfolios averages out much of the unpredictable stock-level noise and boosts the signal strength."* A stock-level R² of 0.40% becomes a Sharpe of 1.35 **only after cross-sectional aggregation across ~3000 names.** You have **12 symbols**. You do not have this averaging mechanism. **This is a structural disadvantage and it is worth stating plainly: the famous ML-in-finance results are cross-sectional, large-N-panel results, and a 12-symbol time-series directional model is a fundamentally different and harder problem.**

### 5.2 Tree-based vs deep learning on tabular data — a settled question

**Grinsztajn, Oyallon & Varoquaux (NeurIPS 2022), *Why do tree-based models still outperform deep learning on typical tabular data?*** ([arXiv:2207.08815](https://arxiv.org/abs/2207.08815), [PDF](https://papers.neurips.cc/paper_files/paper/2022/file/0378c7692da36807bdec87ab043cdadc-Paper-Datasets_and_Benchmarks.pdf)):

> "Results show that tree-based models remain state-of-the-art on medium-sized data (~10K samples) even without accounting for their superior speed."

They identify three inductive-bias reasons NNs lose on tabular data: (1) **must be robust to uninformative features**, (2) **must preserve the orientation of the data**, (3) **must easily learn irregular functions.** They ran a **20,000 compute-hour** hyperparameter search on 45 datasets.

**For you:** 1.26M rows sounds large, but after the uniqueness correction (§2.3) your *effective* N may be ~10K–100K, squarely in the regime where trees win. **Stay with GBDT.**

### 5.3 Deep learning for crypto — evidence is weak and frequently cost-free

**Crypto LSTM vs GBDT:** *Short-term bitcoin market prediction via machine learning* ([doi:10.1016/j.jfds.2021.03.001](https://doi.org/10.1016/j.jfds.2021.03.001), 161 citations) — horizons 1–60 min: *"while all models outperform a random classifier, recurrent neural networks and gradient boosting classifiers are especially well-suited."* Also: *"predictability increases for longer prediction horizons"* — relevant, since your 12-bar horizon is at the longer end.

**A cautionary example — read the fine print.** *Technical Analysis Meets Machine Learning: Bitcoin Evidence* ([arXiv:2511.00665](https://arxiv.org/abs/2511.00665)) is widely cited as "LSTM beats LightGBM on Bitcoin":

| Strategy | Cumulative Return | Accuracy |
|---|---|---|
| LSTM | 65.23% | **0.5611** |
| LightGBM | 53.38% | **0.5840** |
| Buy & Hold | 42.51% | — |

**Note that LightGBM had HIGHER accuracy (0.5840 vs 0.5611) but LOWER return.** And critically, the paper states: *"It is important to mention that these trading strategies do not include transaction costs (brokerage fees, slippage, etc.), which significantly impact real-world performance."* With a 0.1% fee the LSTM drops 65.23% → 53.23% (120 trades) and LightGBM 53.38% → 39.78% (136 trades). **This is a one-year, single-asset, cost-free comparison, and the headline result is an artifact of leverage/timing rather than accuracy.** Do not use it to justify LSTM.

**PatchTST / TimesNet / TFT:** PatchTST ([arXiv:2211.14730](https://arxiv.org/abs/2211.14730)) and TFT ([arXiv:1912.09363](https://arxiv.org/abs/1912.09363)) are strong general long-horizon forecasting architectures, but the benchmarks they are validated on (ETT, weather, electricity, traffic) have **fundamentally higher signal-to-noise than 5-min crypto returns**. I found **no credible OOS study showing PatchTST or TimesNet producing net-of-cost edge in crypto directional trading.** Grade: **transplanted from a different domain; no finance-specific evidence.**

**One architecture with genuine crypto-relevant evidence:** *Supervised autoencoder MLP* ([doi:10.1186/s40537-025-01267-7](https://doi.org/10.1186/s40537-025-01267-7)) tests **SAE + noise augmentation + triple-barrier** on S&P 500, EUR/USD and **BTC/USD at 5-minute bars** — the closest published analogue to your setup. Findings: balanced noise augmentation (Gaussian noise scaled to a fraction of historical feature volatility, e.g. **0.1 noise ratio**) and moderate bottleneck size significantly boost risk-adjusted returns, but *"excessive noise and large bottleneck sizes can impair performance."* **But BTC's IRR in their Approach-1 configuration was negative (−25.29% cumulative, IR −0.50).**

### 5.4 Ensembling and stacking

- GKX is itself evidence that *regularized* ensembles (GBRT with many shallow trees + Huber loss) work where a single OLS fails catastrophically (R²_oos −3.46% → +0.34%).
- Your LightGBM+CatBoost ensemble is already the right idea. **Two caveats specific to you:** (1) CatBoost's default ordered boosting is genuinely useful on small effective samples, so it should carry real weight; (2) **do not stack with a meta-learner trained on OOF predictions unless those OOF predictions come from purged folds with embargo** — otherwise the stacker learns leakage.
- **Grade: ensembling = well-evidenced in general; the specific LightGBM+CatBoost stack for crypto direction = no public controlled evidence.**

---

## 6. Overfitting controls — exact procedures

### 6.1 Purged K-fold + embargo

**Problem:** with overlapping labels, a training row whose label span overlaps the test set's label span leaks the test answer. And rows immediately *after* the test fold are serially correlated with it even when their labels do not overlap.

**Two rules:**
1. **Purge:** remove from the training set any observation whose label span `[t_{i,0}, t_{i,1}]` overlaps the test set's span.
2. **Embargo:** additionally drop a fixed buffer of training rows immediately *after* each test fold. The standard default is **`pctEmbargo = 0.01`** (1% of total observations) in AFML and mlfinlab.

For your data: with a 12-bar label horizon on 5-min bars, purge width ≈ 12 bars, and embargo = 1% of ~1.26M rows ≈ **12,600 rows ≈ 43 days** at 5-min bars × 12 symbols. That is a *large* embargo; check whether it is proportionate, and note that your "~366 days" of data is short enough that a 43-day embargo per fold destroys a lot of training data. **This is a genuine tension in your setup: 366 days with a 1h horizon gives very few independent blocks.**

Implementation: [purgedcv](https://github.com/eslazarev/purged-cross-validation) — `PurgedKFold(n_splits=4, prediction_times=pred, evaluation_times=evalu)`, `apply_embargo(embargo="3D")`, `PurgedGroupKFold`. Purge and embargo are applied automatically per fold. mlfinlab has `PurgedKFold` and `CombinatorialPurgedKFold` (commercial tier).

### 6.2 Combinatorial Purged Cross-Validation (CPCV)

From mlfinlab/AFML ch.12 and [purgedcv](https://github.com/eslazarev/purged-cross-validation):

- Partition the data into **N groups**.
- Form all `C(N, k)` combinations of `k` groups as the **test** set (the rest is train, after purging/embargo).
- Each combination produces predictions for `k` groups. Since each group appears in many combinations, you can **reassemble the predictions into `phi` complete backtest paths**, where `phi = k * C(N,k) / N = C(N-1, k-1)`.

**Worked example (from the purgedcv README):** C(6,2) = **15 splits** over 6 blocks tile into **5 backtest paths**. Check: `C(6-1, 2-1) = C(5,1) = 5`. ✓

**Why it matters:** instead of one chronological path you get 5 (or hundreds) of paths, so you can look at the **distribution** of OOS Sharpe and compute PBO directly. Typical parameters in practice: N=6–10, k=2. N=6,k=2 → 15 splits, 5 paths. N=10,k=2 → 45 splits, 9 paths.

**Important warning found in the purgedcv docs:** *"The Deflated Sharpe Ratio deflates by the number of strategy *configurations* searched (not the number of CPCV paths)."* Do not conflate the two — a common and costly error.

**Evidence that CPCV is worth it** — *Backtest overfitting in the machine learning era: A comparison of out-of-sample testing methods in a synthetic controlled environment* ([doi:10.1016/j.knosys.2024.112477](https://doi.org/10.1016/j.knosys.2024.112477), [SSRN 4778909](https://doi.org/10.2139/ssrn.4778909)). A **synthetic controlled environment** is the right experimental design for this question because you know the ground truth. This is the best available head-to-head comparison of OOS testing methods.

**Also worth knowing:** the purgedcv synthetic leakage proof — *"Naive CV reports R² ≈ 0.83–0.91 on a target nothing can predict"* while PurgedKFold collapses below zero. If you are currently using any shuffled split, **your entire measured performance is fabricated.** Verify this first.

### 6.3 Deflated Sharpe Ratio (DSR)

Bailey & López de Prado, *The Deflated Sharpe Ratio: Correcting for Selection Bias, Backtest Overfitting, and Non-Normality* ([SSRN 2460551](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2460551), [PDF](https://www.davidhbailey.com/dhbpapers/deflated-sharpe.pdf)). Full text extracted.

**Step 1 — expected maximum Sharpe under N independent trials:**

```
E[max SR_hat]  =  mu_SR + sigma_SR * ( (1 - gamma) * Z^-1(1 - 1/N)
                                     + gamma   * Z^-1(1 - 1/(N*e)) )
```
where **gamma ≈ 0.5772** is the Euler–Mascheroni constant, `Z^-1` is the inverse standard normal CDF, `e` is Euler's number, `mu_SR` = mean SR across trials (`E[{SR_hat}]`), `sigma_SR` = std of SR across trials (`sqrt(V[{SR_hat}])`).

Reference implementation from the paper itself:
```python
import numpy as np, scipy.stats as ss
def getExpMaxSR(mu, sigma, numTrials):
    emc = 0.5772156649                     # Euler-Mascheroni constant
    maxZ = (1-emc)*ss.norm.ppf(1-1./numTrials) + emc*ss.norm.ppf(1-1./(numTrials*np.e))
    return mu + sigma*maxZ
```

**Step 2 — the DSR statistic** (a Probabilistic Sharpe Ratio with the threshold replaced by the expected max):

```
DSR = Z( ( (SR_hat - SR_0_hat) * sqrt(T - 1) ) /
         sqrt( 1 - skew*SR_hat + ((kurt - 1)/4) * SR_hat^2 ) )
```
where `SR_0_hat = E[max SR_hat]` from Step 1, **T = sample length (number of returns)**, `skew` = skewness of the selected strategy's returns, `kurt` = kurtosis (**note: not excess kurtosis** — the `(kurt-1)/4` term assumes raw kurtosis with 3.0 for normal). DSR = P(true SR > 0) after accounting for multiplicity.

**Step 3 — correlated trials (`N` when trials are not independent):** the paper's Appendix derives, from the average correlation `rho_bar` across trials:
```
N_hat = rho_bar + (1 - rho_bar) * M
```
i.e. interpolate between `N = M` (rho=0, fully independent) and `N = 1` (rho=1, fully redundant). The paper warns: *"In practice M almost always exceeds the sample length, T. Then the estimate of average correlation may itself be overfit... it is not guaranteed that [the matrix is positive definite]."* Caveats: correlation measures only linear dependence; for short samples use dimension reduction (clustering) or information-theoretic redundancy estimates instead.

**Applied to you:** your stated bar is bootstrap P(edge>0) ≥ 0.95, and you measured 0.08–0.27. Rather than debugging the bootstrap, **compute DSR**, which is the correct multiplicity-adjusted statistic. You must supply `N` = number of configurations you actually searched (feature sets × hyperparameters × horizons × label params — be honest, this is usually hundreds) and `V[SR_hat]` across those trials.

**Reference implementation:** `deflated_sharpe_ratio`, `deflated_sharpe_ratio_full`, `DSRDiagnostics`, `effective_n_trials`, `minimum_backtest_length`, `min_track_record_length` in [purgedcv](https://github.com/eslazarev/purged-cross-validation).

### 6.4 Probability of Backtest Overfitting (PBO) via CSCV

Bailey, Borwein, López de Prado & Zhu, *The Probability of Backtest Overfitting* ([SSRN 2326253](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2326253), [PDF](https://www.davidhbailey.com/dhbpapers/backtest-prob.pdf)). Full text extracted.

**Procedure (verbatim structure from the paper):**
1. Form a matrix **M** of T observations × N strategy configurations (N = number of trials in the search).
2. Partition the T rows into **S disjoint submatrices** of equal length (S even).
3. Form all `C(S, S/2)` combinations of `S/2` submatrices as the in-sample (IS) set; the complement is OOS. Each combination is reused symmetrically (IS↔OOS), hence "combinatorially symmetric."
4. For each combination `c`: rank configurations on IS, pick **n*** = the IS-best configuration; note its relative rank `omega_bar_c = r_bar_{n*} / (N+1) ∈ (0,1)` on the OOS set.
5. Compute the **logit** `lambda_c = ln( omega_bar_c / (1 - omega_bar_c) )`. **High logit = IS/OOS consistency; low logit = overfitting.**
6. **PBO = relative frequency of `lambda_c <= 0`** across all combinations, i.e. the probability that the IS-selected configuration underperforms the median OOS.

The paper derives 4 complementary statistics: **PBO**, performance degradation, probability of loss, and stochastic dominance.

**Typical interpretation:** PBO > 0.5 means your selection procedure is worse than random. PBO in the 0.2–0.5 range is common even for legitimate strategies. Implementation: `CombinatoriallySymmetricCV` and `probability_of_backtest_overfitting` in [purgedcv](https://github.com/eslazarev/purged-cross-validation).

### 6.5 Crypto-specific PBO evidence

*Deep Reinforcement Learning for Cryptocurrency Trading: Practical Approach to Address Backtest Overfitting* ([arXiv:2209.05559](https://arxiv.org/abs/2209.05559)). Formulates backtest-overfitting detection as a **hypothesis test**, trains DRL agents, estimates PBO, and **rejects overfitted agents.** Tested on 10 cryptocurrencies during the May–June 2022 crash. Their framing is worth adopting: *"Existing works applied deep reinforcement learning methods and optimistically reported increased profits in backtesting, which may suffer from the false positive issue due to overfitting."* (The PBO paper's Monte Carlo/EVT validation study is at SSRN 2568435.)

---

## 7. Open-source projects to reuse — ranked by fit

| Project | URL | What you get | Fit |
|---|---|---|---|
| **purgedcv** | [github.com/eslazarev/purged-cross-validation](https://github.com/eslazarev/purged-cross-validation), `pip install purgedcv` | PurgedKFold, PurgedGroupKFold, CombinatorialPurgedCV, CombinatoriallySymmetricCV, `reconstruct_paths`, PSR, **DSR (full)**, **PBO via CSCV**, MinTRL, MinBTL, `effective_n_trials`, Optuna integration, **leakage audit diagnostics** | **Best single fit.** sklearn-compatible, MIT, actively maintained, conda-forge. Replaces the paywalled mlfinlab. |
| **mlfinpy** | [mlfinpy.readthedocs.io](https://mlfinpy.readthedocs.io/en/latest/) | FFD fracdiff, CUSUM filter, triple-barrier, trend-scanning, raw/fixed-horizon labels, sample uniqueness, **sequential bootstrap**, sample weights (return-attribution + time-decay) | **Best fit for labeling/sampling.** Free docs; actively documented. |
| **mlfinlab** (H&T) | [hudsonthames.org/mlfinlab](https://hudsonthames.org/mlfinlab/) | Triple-barrier, meta-labeling, **trend-scanning, tail sets, matrix flags**, microstructure features, fracdiff, sequentially-bootstrapped ensembles, **deflated/haircut Sharpe, profit hurdles, MinTRL**, ONC clustering, CORGAN, networks | **Comprehensive but now commercial** (£100/mo). Docs are public and worth reading even without a license. Features are the best catalogue of what to build. |
| **research notebooks (H&T)** | [github.com/hudson-and-thames/research](https://github.com/hudson-and-thames/research) | Chapter-by-chapter AFML notebooks (sequential bootstrap, fracdiff, etc.) | Free reference implementations. |
| **backtest-overfitting-lab** | [github.com/KinSushi/backtest-overfitting-lab](https://github.com/KinSushi/backtest-overfitting-lab) | Pure-Python PBO + DSR pipeline | Zero-dependency alternative. |

**Note on mlfinlab:** it went closed-source/commercial. The purgedcv README states the rationale: *"People have asked scikit-learn, auto-sklearn, and mlpack for purging and embargo support and been turned down or left waiting for years. The one mature implementation, mlfinlab, went closed-source and paid. The free alternative has been unmaintained since 2018. That gap is the reason this exists."*

---

## 8. Priority order for this system

**Tier 0 — diagnostics that may invalidate everything (do first, cost ~1 day)**
1. **Compute average label uniqueness.** If it's ≲0.05, your 1.26M rows are ~50K effective, which explains the bootstrap P(edge>0) and the calibration blow-up. Everything downstream changes.
2. **Confirm no shuffled splits anywhere.** Leakage *inflates* results; you have negative results, so leakage is unlikely — but the calibrator may still be leaking.
3. **Remove isotonic calibration.** ECE 0.007→0.49 is beyond salvage; GBDT outputs are already near-calibrated.
4. **Recompute with DSR and PBO** (not a bootstrap) to get the honest multiplicity-adjusted P(edge>0).

**Tier 1 — label changes with the best evidence-to-effort ratio**
5. **Make the barriers cost-aware:** profit/stop at ≥ ±25 bps, not ±5 bps. Train on moves that can pay for themselves.
6. **Meta-labeling**, with the secondary target = "primary side's realized return exceeds cost." Expect precision/recall redistribution, not new edge.
7. **Add a neutral class** for the vertical barrier instead of forcing binary.

**Tier 2 — data construction**
8. **Move from time bars to dollar/volume/imbalance bars.** Highest-leverage data change; crypto volume is wildly non-uniform.
9. **Add realized vol / semivariance / realized skew** as features, and consider a **volatility target** alongside direction.

**Tier 3 — model and validation**
10. **Feed uniqueness × attribution × time-decay weights** into LightGBM and CatBoost `sample_weight`.
11. **Apply GMADL-style direction-aware objective** (custom loss — real implementation cost, real evidence).
12. **Switch to CPCV (N=6, k=2 → 15 splits, 5 paths)** for validation and to generate a Sharpe distribution.
13. **Keep GBDT.** Do not invest in LSTM/PatchTST/TimesNet — no finance-specific evidence, and Grinsztajn et al. explain why trees win on this data shape.

**Tier 4 — the honest option**
14. **Consider that 5-min/1-hour crypto direction at 12 bps cost may not be a tradeable problem.** The closest published analogue (purgedcv's daily BTC/USDT study) found no edge even with honest CV, and the strongest microstructure evidence (3-second horizon, full L2 book) produced a **non-significant negative t-stat on BTC**. The documented edges in this space are: (a) seconds-horizon microstructure in *less liquid* alts, (b) volatility/magnitude prediction, (c) cross-sectional aggregation across large N. A 12-symbol, 1-hour, directional, cost-bearing model has no documented edge in the public literature — and that absence is itself the most important finding in this brief.

---

## 9. Evidence-grading summary

**Real, replicated OOS evidence**
- Triple-barrier > plain direction labels (5-min bars, BTC included) — [doi:10.1186/s40537-025-01267-7](https://doi.org/10.1186/s40537-025-01267-7)
- Meta-labeling improves precision/accuracy OOS (precision .48→.54, acc 48%→55%) — [H&T PDF](https://hudsonthames.org/wp-content/uploads/2022/04/Does-Meta-Labeling-Add-to-Signal-Efficacy.pdf) *(single paper, single asset — thin)*
- Direction > value prediction — [SSRN 5176925](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5176925) *(equity/monthly, not crypto/5min)*
- GBDT/NN > linear on financial panels; R²_oos 0.34% vs 0.16% — [GKX](https://www.nber.org/system/files/working_papers/w25398/w25398.pdf); NN3 best, deeper is worse
- Trees > deep learning on tabular data, 45 datasets, 20k GPU-hours — [arXiv:2207.08815](https://arxiv.org/abs/2207.08815)
- Realized semivariance predicts cross-sectional returns; variance premia R² 10–20% — [doi:10.1017/S0022109019000097](https://doi.org/10.1017/s0022109019000097), [doi:10.1287/mnsc.2017.2890](https://doi.org/10.1287/mnsc.2017.2890)
- Order flow imbalance dominates crypto short-horizon SHAP; but BTC taker t-stat −0.67 and all maker p>0.05 — [arXiv:2602.00776](https://arxiv.org/html/2602.00776v1)
- Purging/embargo prevents fabricated R² of 0.83–0.91 on unpredictable targets — [purgedcv](https://github.com/eslazarev/purged-cross-validation)
- DSR and PBO formulas and procedures — [DSR PDF](https://www.davidhbailey.com/dhbpapers/deflated-sharpe.pdf), [PBO PDF](https://www.davidhbailey.com/dhbpapers/backtest-prob.pdf)

**Implementation-documented but no controlled OOS P&L evidence**
- Sample uniqueness weights, sequential bootstrap, time-decay weights
- Fractional differentiation (FFD with ADF-selected d)
- Trend-scanning labels (though the t-values are excellent sample weights)
- Cross-sectional ranking / feature neutralization
- CPCV vs walk-forward head-to-head (a synthetic comparison exists: [doi:10.1016/j.knosys.2024.112477](https://doi.org/10.1016/j.knosys.2024.112477))

**Folklore — no evidence found**
- Sharpe-ratio as a per-sample supervised target
- "Sharpe threshold 0.95 on a bootstrap" as a discovery bar (DSR/PBO are the correct tools)
- Deep learning (LSTM/TFT/PatchTST/TimesNet) beating GBDT for crypto direction net of costs
- Any claim from a paper that excludes transaction costs (explicitly flagged in [arXiv:2511.00665](https://arxiv.org/abs/2511.00665))

---

## 10. On "better training data"

The brief asks for better data. Concretely, and in order of expected value:

1. **Higher-frequency L2 orderbook + trade data** — but only if you also move the horizon to seconds. Otherwise the information is thrown away by 5-min aggregation. The microstructure evidence shows the edge lives there ([arXiv:2602.00776](https://arxiv.org/html/2602.00776v1)).
2. **Event-based bars** (dollar/volume/imbalance) instead of 5-min time bars.
3. **More symbols, not more time.** GKX's Sharpe of 1.35 comes from cross-sectional aggregation over ~3000 names. Twelve symbols gives you almost no averaging. Expanding the cross-section to 50–150 perps and going cross-sectional is a structurally better use of data budget than extending the history.
4. **Longer history.** 366 days is short for 5-min bars with a 12-bar horizon; you need enough independent blocks for CPCV and for DSR's `T` term. Note the tension in §6.1 (a 43-day embargo × many folds eats a 366-day sample).
5. **Point-in-time funding/basis/OI data.** *Perpetual Futures Pricing* ([NBER w32936](https://doi.org/10.3386/w32936)) gives the no-arbitrage structure of the funding mechanism — the cleanest economic prior you have in this asset class. *The Two-Tiered Structure of Cryptocurrency Funding Rate Markets* ([doi:10.3390/math14020346](https://doi.org/10.3390/math14020346), 35.7M 1-minute observations, 26 exchanges, 749 symbols) documents that CEX dominate price discovery with **zero reverse causality from DEX** — a concrete, exploitable cross-venue feature direction.
6. **Survivorship-bias audit.** 12 symbols over 366 days for USDT-M perps: verify you have delisted/dead perps included. If you selected the 12 largest *current* perps, that is a look-ahead bias in the universe.
