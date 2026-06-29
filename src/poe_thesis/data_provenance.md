# Data provenance (documented, not scripted)

These materials ship **no data, no fitted models, and no results**. The code illustrates the
methodology; it does not retrieve or build the inputs. This document records where the inputs come
from so that a researcher with the appropriate access can reconstruct them and apply this code.

## Sources

| Source | Variables | Access | Notes |
|---|---|---|---|
| **CRSP** monthly stock file | returns, prices, shares, delisting, SIC | **Proprietary — WRDS subscription. Not included and cannot be redistributed.** | Delisting-adjusted returns (`ret` → `dlret` → `−0.30` for involuntary delistings). |
| **OSAP** firm characteristics | open-source cross-sectional predictors | Public | Chen & Zimmermann, Open Source Asset Pricing, release `202510` (via the `openassetpricing` package). |
| **Welch–Goyal** macro predictors | `dp ep bm ntis tbl tms dfy svar infl` (9) | Public | The standardized macro state the scenarios sample over. |
| **NFCI** financial-conditions index | national financial conditions | Public | Chicago Fed (also distributed via FRED); used by the regime / nearest-analogue layer. |
| **NBER** recession dating | recession indicator | Public | Used for regime labelling. |

## Keys, target, features

- Firm and macro panels join on **`(permno, yyyymm)`** (`yyyymm` an integer `YYYYMM`).
- Target: next-month return `ret_tplus1`; models use the excess return `excess_ret = ret_tplus1 − Rfree`.
- Features: cleaned firm characteristics × the 9 macro predictors, interaction columns `{firm}_x_{macro}`
  (built by `build_interaction_feature_matrix` in `macro_features.py`).

## Splits & standardization

- Train `198001–200512`, validation `200601–201512`, test `201601–202412`.
- The macro scaler (`macro_features.MacroScaler`) and any thresholds are **fit on train only**.

## What is intentionally absent

Data-acquisition and data-loading code (CRSP/parquet readers, the monthly-shard dataset iterator,
the anchor-decision-context builder) is **excluded by design**. A few illustrative functions in
`pipelines.py` (e.g. `precompute_monthly_inputs`, `train_pao`) and `pipelines.precompute_monthly_inputs`
reference an external loader (`iter_shards_monthly`) and a CRSP reader that are **not shipped**; those
functions are illustrative of the training procedure and are not runnable without the user's own data
pipeline.
