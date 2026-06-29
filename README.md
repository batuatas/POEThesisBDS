# POE Tactical Thesis — Research Materials

Minimal research materials illustrating the core methodology of a tactical predict–optimize
portfolio thesis: return prediction, portfolio optimization, a plausibility prior over macro states,
gradient-based and gradient-free scenario samplers with convergence diagnostics, decision probes,
and post-hoc regime/analogue diagnostics. This is **not** a turnkey reproduction pipeline — data,
fitted models, and results are excluded by design, and the market data the thesis uses (CRSP, via
WRDS) cannot be redistributed under licence.

## Data provenance (not scripted here)

Acquisition is documented, not scripted. A researcher with their own WRDS licence can obtain the
inputs and apply this code. Full detail in [`src/poe_thesis/data_provenance.md`](src/poe_thesis/data_provenance.md).

- **CRSP** monthly stock file (returns) — proprietary, WRDS subscription, **license-restricted; not included**.
- **OSAP** firm characteristics — Chen & Zimmermann, Open Source Asset Pricing, release `202510`, public.
- **Welch–Goyal** macro predictors — public.
- **NFCI** national financial conditions index — Chicago Fed (via FRED), public.

## What the code covers

- **`predictors.py`** — the return predictors: feed-forward neural network (`AssetPricingFNN`, with an ensemble wrapper) and the gradient-boosted tree ensemble, plus numpy-predictor factories.
- **`pipelines.py`** — the training pipelines: predict-then-optimize (PTO) backtest/grid and predict-and-optimize (PAO) decision-focused training. (Data-loading code is excluded; see data provenance.)
- **`optimizer.py`** — the robust mean–variance optimizer (SOCP) and covariance conditioning (EWMA and Ledoit–Wolf nonlinear shrinkage), plus the differentiable `CvxpyLayer` used by PAO.
- **`plausibility_prior.py`** — the VAR(1) stationary-law prior: Goyal–Welch VAR(1) fit and the stationary moments (μ\*, Σ\*).
- **`samplers.py`** — the scenario samplers: preconditioned MALA and the gradient-free affine-invariant (Goodman–Weare) ensemble, with the energy / log-target and VAR(1) log-prior gradient.
- **`diagnostics.py`** — the convergence diagnostics: Gelman–Rubin R-hat, effective sample size, and the convergence gates.
- **`probes.py`** — the probing functions: benchmark-return, concentration, and divergence (decision-layer event-loss rewards, event membership, thresholds), plus the probe registry.
- **`regime_and_analogues.py`** — the post-hoc diagnostics: the regime classifier and the nearest-historical-analogue (NFCI-style) scoring.

`macro_features.py` is shared infrastructure (macro standardization and the firm × macro
interaction-feature builder) used by several of the stages above.

## Install

```bash
pip install -e .
```

(Python ≥ 3.10; dependencies pinned in `requirements.txt`.)
