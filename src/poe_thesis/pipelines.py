# Training pipelines: predict-then-optimize (PTO) backtest + grid and predict-and-optimize (PAO) decision-focused training. Data loaders excluded — see data_provenance.md.
from __future__ import annotations


"""Predict-Then-Optimize (PTO) backtest for the tactical POE track.

Pipeline per month t:
  1.  FNN forward pass → μ̂ for all firms in the cross-section
  2.  Universe selection: top-K by μ̂ (default) or fixed top-N largest-cap (universe_rule="size")
  3.  Covariance Σ from the selected firms' 60-month CRSP history (nonlinear shrinkage, LW2020)
  4.  Unified robust MVO (optimization/robust_mvo.solve_robust_mvo, shared with the E2E layer):
        max μ̂'w − κ·√(w'Ω w) − (λ/2)·w'Σ w   s.t. sum(w)=1, w≥0,  Ω = diag(Σ) by default
  5.  Realised excess return: w' r_{t+1}

All MVO parameters (λ, κ, Ω mode, Σ estimator, universe) live in PtoConfig — Batuhan sets them.
"""


import dataclasses
import json
import logging
from pathlib import Path
from typing import Dict, Iterator, Optional

import numpy as np
import torch

log = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
#  Config
# ──────────────────────────────────────────────────────────────────────────────

@dataclasses.dataclass
class PtoConfig:
    """Hyper-parameters for the PTO backtest. All are Batuhan's design decisions."""
    lambd: float = 10.0       # quadratic risk-aversion parameter (λ)
    kappa: float = 0.5        # robust uncertainty penalty (κ)
    omega_mode: str = "diag_sigma"   # uncertainty matrix Ω: diag_sigma (YPS) / identity / sigma_over_T
    cov_estimator: str = "nl_shrinkage"  # Σ estimator: nl_shrinkage (LW2020) / lw_linear / sample
    universe_rule: str = "top_mu"    # "top_mu" (top_k by μ̂) / "size" (top-N largest-cap) / "size_then_mu"
    universe_size: int = 100         # N for universe_rule="size" (E2E-comparable, tractable)
    top_k: int = 1000         # firms selected per month by descending μ̂ ("top_mu" / Stage-2 of "size_then_mu")
    prescreen_size: int = 1000  # Stage-1 liquid pre-screen (top-M largest-cap) for "size_then_mu"
    lookback: int = 60        # rolling-window months for covariance estimation
    min_obs: int = 24         # minimum monthly return observations per firm
    diagonal_shrinkage: float = 0.0   # (deprecated; covariance regularisation handled by cov_estimator)
    rfree_annualise: bool = False      # True → use annual risk-free; False → monthly tbl/12


@dataclasses.dataclass
class MonthlyResult:
    yyyymm: int
    n_universe: int          # firms in that month's cross-section
    n_selected: int          # firms after top-K cut
    weights: np.ndarray      # (n_selected,) float32 portfolio weights
    permnos: np.ndarray      # (n_selected,) int32 selected CRSP identifiers
    mu_hat: np.ndarray       # (n_selected,) float32 FNN return predictions
    realized_return: float   # w' r_{t+1}  (nan if returns unavailable)


@dataclasses.dataclass
class BacktestResult:
    split: str
    config: PtoConfig
    monthly: list[MonthlyResult]

    @property
    def returns(self) -> np.ndarray:
        return np.array([m.realized_return for m in self.monthly], dtype=np.float64)

    @property
    def yyyymms(self) -> np.ndarray:
        return np.array([m.yyyymm for m in self.monthly], dtype=np.int32)

    def r2_oos(self) -> float:
        """Strategy-level diagnostic on the realized *portfolio* return series:
        1 - Σr² / Σ(r - r̄)²  (zero-benchmark numerator, mean-benchmark denominator).

        NOTE: this is a portfolio-return diagnostic and is distinct from GKX's
        cross-sectional return-*prediction* R²_oos (the consistent zero benchmark
        1 - Σ(r-r̂)²/Σr²), which is computed in scripts/train_fnn.py and stored in
        the ensemble manifest (ensemble_val / ensemble_test).
        """
        r = self.returns
        r = r[np.isfinite(r)]
        if len(r) == 0:
            return float("nan")
        ss_res = float(np.sum(r ** 2))
        ss_tot = float(np.sum((r - r.mean()) ** 2))
        if ss_tot < 1e-12:
            return float("nan")
        return 1.0 - ss_res / ss_tot

    def sharpe(self, periods_per_year: int = 12) -> float:
        r = self.returns
        r = r[np.isfinite(r)]
        if len(r) < 2:
            return float("nan")
        return float(r.mean() / (r.std() + 1e-12) * np.sqrt(periods_per_year))

    def portfolio_metrics(self, periods_per_year: int = 12) -> dict:
        """Decision-quality metrics for the PTO-vs-E2E comparison.

        Returns annualized Sharpe/return/vol plus average turnover, Herfindahl,
        max weight, and effective N (= 1/Herfindahl) across months.
        """
        r = self.returns
        r = r[np.isfinite(r)]
        herf, maxw, eff_n = [], [], []
        for m in self.monthly:
            w = np.asarray(m.weights, dtype=np.float64)
            h = float(np.sum(w ** 2))
            herf.append(h)
            maxw.append(float(w.max()) if w.size else float("nan"))
            eff_n.append(1.0 / h if h > 0 else float("nan"))
        # turnover = Σ|w_t − w_{t−1}| with weights aligned by permno
        turn = []
        prev = None
        for m in self.monthly:
            cur = dict(zip(m.permnos.tolist(), np.asarray(m.weights, dtype=np.float64).tolist()))
            if prev is not None:
                keys = set(cur) | set(prev)
                turn.append(sum(abs(cur.get(k, 0.0) - prev.get(k, 0.0)) for k in keys))
            prev = cur
        return {
            "sharpe": self.sharpe(periods_per_year),
            "ann_return": float(r.mean() * periods_per_year) if len(r) else float("nan"),
            "ann_vol": float(r.std() * np.sqrt(periods_per_year)) if len(r) else float("nan"),
            "mean_monthly_return": float(r.mean()) if len(r) else float("nan"),
            "avg_turnover": float(np.mean(turn)) if turn else float("nan"),
            "avg_herfindahl": float(np.mean(herf)) if herf else float("nan"),
            "avg_max_weight": float(np.mean(maxw)) if maxw else float("nan"),
            "avg_effective_n": float(np.mean(eff_n)) if eff_n else float("nan"),
            "n_months": len(r),
        }

    def to_json(self) -> dict:
        return {
            "split": self.split,
            "n_months": len(self.monthly),
            "sharpe": self.sharpe(),
            "r2_oos": self.r2_oos(),
            "mean_monthly_return": float(np.nanmean(self.returns)),
            "config": dataclasses.asdict(self.config),
        }


# ──────────────────────────────────────────────────────────────────────────────
#  Covariance builder
# ──────────────────────────────────────────────────────────────────────────────



def build_monthly_covariance(
    crsp_ret: np.ndarray,     # (n_months_history, n_all_firms) float32; np.nan for missing
    crsp_permnos: np.ndarray,  # (n_all_firms,) int32 — column index of crsp_ret
    crsp_yyyymms: np.ndarray,  # (n_months_history,) int32 — row index of crsp_ret
    selected_permnos: np.ndarray,  # (K,) int32 — firms to include
    yyyymm: int,              # current month (lookback ends here, exclusive)
    lookback: int,            # number of months of history to use
    min_obs: int,             # minimum non-missing months per firm
    cov_estimator: str = "nl_shrinkage",
) -> tuple[np.ndarray, np.ndarray]:
    """Extract the rolling-window covariance Σ for the selected firms.

    Returns:
        valid_permnos: (M,) int32 — firms with enough history
        sigma: (M, M) float64 — PD covariance from `cov_estimator` (default LW2020 nonlinear)
    """
    from poe_thesis.optimizer import estimate_covariance
    # Find row indices for lookback window [t-lookback, t)
    # crsp_yyyymms are sorted ascending
    mask_before = crsp_yyyymms < yyyymm
    row_indices = np.where(mask_before)[0]
    if len(row_indices) == 0:
        return selected_permnos[:0], np.empty((0, 0))
    row_indices = row_indices[-lookback:]  # last `lookback` months

    # Find column indices for selected firms
    permno_to_col = {p: i for i, p in enumerate(crsp_permnos)}
    col_indices = np.array(
        [permno_to_col[p] for p in selected_permnos if p in permno_to_col],
        dtype=np.int64,
    )
    valid_permnos = np.array(
        [p for p in selected_permnos if p in permno_to_col],
        dtype=np.int32,
    )
    if len(col_indices) == 0:
        return valid_permnos[:0], np.empty((0, 0))

    ret_sub = crsp_ret[np.ix_(row_indices, col_indices)].astype(np.float64)

    # Drop firms with too few observations
    obs_count = np.sum(np.isfinite(ret_sub), axis=0)
    enough = obs_count >= min_obs
    ret_sub = ret_sub[:, enough]
    valid_permnos = valid_permnos[enough]

    if ret_sub.shape[1] < 2:
        return valid_permnos[:0], np.empty((0, 0))

    # Replace NaN with column mean (simple imputation for missing months)
    col_means = np.nanmean(ret_sub, axis=0)
    nan_mask = np.isnan(ret_sub)
    ret_sub[nan_mask] = np.take(col_means, np.where(nan_mask)[1])

    sigma = estimate_covariance(ret_sub, method=cov_estimator)
    return valid_permnos, sigma


# The robust MVO solve lives in optimization/robust_mvo.solve_robust_mvo (the single
# source of truth shared with the E2E differentiable layer). The backtest loop below
# calls it directly.


# ──────────────────────────────────────────────────────────────────────────────
#  Main backtest runner
# ──────────────────────────────────────────────────────────────────────────────

def run_pto_backtest(
    model_dir: str | Path,
    features_dir: str | Path,
    crsp_path: str | Path,
    split: str,
    config: PtoConfig | None = None,
    gw_path: str | Path | None = None,
    device: str = "cpu",
) -> BacktestResult:
    """Run PTO backtest for one split (val or test).

    Args:
        model_dir:    Directory with `state_dict.pt` + `model_config.json`.
        features_dir: Root features dir (contains train/val/test subdirs).
        crsp_path:    Raw CRSP parquet (permno, date, ret_adj).
        split:        "val" or "test".
        config:       PtoConfig; defaults applied if None.
        gw_path:      Goyal-Welch parquet for rfree (optional; uses tbl column).
        device:       Torch device for FNN inference.

    Returns:
        BacktestResult with monthly weights + realised returns.
    """
    if config is None:
        config = PtoConfig()
    log.info(
        "Running PTO backtest split='%s' (universe=%s, omega=%s, cov=%s) …",
        split, config.universe_rule, config.omega_mode, config.cov_estimator,
    )
    precomp = precompute_monthly_inputs(
        model_dir, features_dir, crsp_path, split, config=config, gw_path=gw_path, device=device
    )
    return _backtest_from_precomputed(precomp, config, split)


def _select_universe(
    universe_rule: str,
    mu_hat: np.ndarray,
    size_sig: "np.ndarray | None",
    n_universe: int,
    *,
    universe_size: int,
    top_k: int,
    prescreen_size: int,
) -> np.ndarray:
    """Pick the month's investable cross-section. Pure (numpy) so it is unit-testable.

    Modes:
      "size"          top-N largest cap (smallest Size signal = −log mktcap). Exogenous, ignores μ̂.
      "top_mu"        top-K by predicted return μ̂ over the whole cross-section (no liquidity floor).
      "size_then_mu"  two-stage (POE-faithful): pre-screen to the top-M largest caps (liquid set),
                      then take the top-K by μ̂ within it — lets the FNN pick names while staying
                      investable, avoiding the micro-cap contamination of raw "top_mu".
    Returns row indices into the month's cross-section (ordered as the original per-mode logic).
    """
    if universe_rule == "size":
        if size_sig is None:
            raise ValueError("universe_rule='size' requires a 'Size' feature column")
        k = min(universe_size, n_universe)
        return np.argsort(size_sig)[:k]
    if universe_rule == "size_then_mu":
        if size_sig is None:
            raise ValueError("universe_rule='size_then_mu' requires a 'Size' feature column")
        m = min(prescreen_size, n_universe)
        prescreen_idx = np.argsort(size_sig)[:m]                  # Stage 1: largest caps
        k = min(top_k, m)
        mu_pre = mu_hat[prescreen_idx]
        top_local = np.argpartition(mu_pre, -k)[-k:]              # Stage 2: top-K by μ̂ within liquid set
        top_local = top_local[np.argsort(mu_pre[top_local])[::-1]]
        return prescreen_idx[top_local]
    # "top_mu"
    k = min(top_k, n_universe)
    sel = np.argpartition(mu_hat, -k)[-k:]
    return sel[np.argsort(mu_hat[sel])[::-1]]


def precompute_monthly_inputs(
    model_dir: str | Path,
    features_dir: str | Path,
    crsp_path: str | Path,
    split: str,
    config: "PtoConfig | None" = None,
    gw_path: str | Path | None = None,
    device: str = "cpu",
) -> dict:
    """Load model + CRSP, select universe, estimate Σ, compute μ̂ per month — everything that
    does NOT depend on (κ, λ, Ω). A κ×λ grid (solve_pto_grid) reuses these per-month inputs
    instead of recomputing predictions/covariance for every (κ, λ) cell."""
    import pyarrow.parquet as pq

    from poe_thesis.tactical.data.month_shard_dataset import iter_shards_monthly
    from poe_thesis.predictors import (
        load_fnn_from_dir,
        load_fnn_ensemble_from_dir,
        is_ensemble_dir,
    )

    if config is None:
        config = PtoConfig()
    features_dir = Path(features_dir)
    model_dir = Path(model_dir)

    # ── Load predictor: FNN (torch) or tree (numpy) — both expose predict_fn(X_month)->(N,) ──
    from poe_thesis.predictors import is_tree_dir, make_tree_numpy_predictor

    if is_tree_dir(model_dir):
        log.info("Loading tree ensemble from %s", model_dir)
        tree_predict = make_tree_numpy_predictor(model_dir)
        feature_cols = json.loads((model_dir / "feature_columns.json").read_text())
        model_cfg = json.loads((model_dir / "tree_manifest.json").read_text())

        def predict_fn(X_month):
            return np.asarray(tree_predict(X_month.detach().cpu().numpy()), dtype=np.float64)
    else:
        if is_ensemble_dir(model_dir):
            log.info("Loading FNN ensemble from %s", model_dir)
            model, feature_cols, model_cfg = load_fnn_ensemble_from_dir(model_dir, map_location=device)
        else:
            log.info("Loading FNN from %s", model_dir)
            model, feature_cols, model_cfg = load_fnn_from_dir(model_dir, map_location=device)
        model = model.to(device)
        # Inference mode: frozen BatchNorm running stats — matches the scenario-explanation path
        # (make_fnn_numpy_predictor → predictor.eval()) so the back-tested μ̂ equals the explained μ̂.
        # (The headline consensus-universe backtest, run_4way_topmu_backtest.py, already uses that eval
        #  path; this precompute path feeds the now-abandoned matched-β/size experiments. Fixed for
        #  consistency.) On MPS the eval-mode running_var can underflow, so fall back to batch stats there.
        model.eval()
        if str(device) == "mps":
            model.train()
            for m in model.modules():
                if isinstance(m, torch.nn.Dropout):
                    m.eval()

        def predict_fn(X_month):
            with torch.no_grad():
                return model(X_month).detach().cpu().numpy()

    # ── Load CRSP returns (month × firm matrix) ───────────────────────────────
    log.info("Loading CRSP returns for covariance estimation …")
    crsp_ret_matrix, crsp_yyyymms, crsp_permnos_all = load_crsp_return_matrix(crsp_path)

    # ── Load rfree if available ───────────────────────────────────────────────
    rfree_map: dict[int, float] = {}
    if gw_path is not None:
        gw_table = pq.read_table(str(gw_path), columns=["yyyymm", "tbl"])
        gw_df = gw_table.to_pandas()
        rfree_map = dict(zip(gw_df["yyyymm"], gw_df["tbl"] / 12.0))

    split_dir = features_dir / split
    size_col = feature_cols.index("Size") if "Size" in feature_cols else None
    log.info("Precomputing monthly inputs (split='%s', universe=%s, cov=%s) …",
             split, config.universe_rule, config.cov_estimator)

    months: list[dict] = []
    for X_month, y_month, permno_month, yyyymm_arr in iter_shards_monthly(
        split_dir, n_features=len(feature_cols), device=device
    ):
        yyyymm = int(yyyymm_arr[0])
        n_universe = X_month.shape[0]

        mu_hat = predict_fn(X_month)  # (N,) — FNN (torch) or tree (numpy), uniform interface

        # Universe selection (independent of κ/λ)
        size_sig = X_month[:, size_col].cpu().numpy() if size_col is not None else None
        sel_idx = _select_universe(
            config.universe_rule, mu_hat, size_sig, n_universe,
            universe_size=config.universe_size, top_k=config.top_k,
            prescreen_size=config.prescreen_size,
        )
        k = len(sel_idx)

        selected_permnos = permno_month[sel_idx]
        selected_mu = mu_hat[sel_idx].astype(np.float32)

        valid_permnos, sigma = build_monthly_covariance(
            crsp_ret_matrix, crsp_permnos_all, crsp_yyyymms, selected_permnos,
            yyyymm, config.lookback, config.min_obs, config.cov_estimator,
        )

        if len(valid_permnos) < 2:
            months.append({
                "yyyymm": yyyymm, "n_universe": n_universe, "k": k,
                "permnos": selected_permnos, "mu": None, "sigma": None,
                "fallback_mu": selected_mu[:k], "rfree": rfree_map.get(yyyymm, 0.0),
            })
        else:
            perm_to_mu = dict(zip(selected_permnos.tolist(), selected_mu.tolist()))
            aligned_mu = np.array([perm_to_mu.get(p, 0.0) for p in valid_permnos], dtype=np.float64)
            months.append({
                "yyyymm": yyyymm, "n_universe": n_universe, "k": k,
                "permnos": valid_permnos, "mu": aligned_mu, "sigma": sigma,
                "fallback_mu": None, "rfree": rfree_map.get(yyyymm, 0.0),
            })
        if yyyymm % 100 == 1 or len(months) % 24 == 0:
            log.info("  precompute %d  n_universe=%d  n_selected=%d", yyyymm, n_universe, len(valid_permnos))

    return {
        "crsp_ret_matrix": crsp_ret_matrix, "crsp_yyyymms": crsp_yyyymms,
        "crsp_permnos_all": crsp_permnos_all, "months": months,
    }


def _backtest_from_precomputed(precomp: dict, config: PtoConfig, split: str) -> BacktestResult:
    """Solve the robust MVO for each precomputed month under `config` (κ, λ, Ω). The solve
    is the ONLY κ/λ-dependent step, so solve_pto_grid reuses `precomp` across the whole grid."""
    from poe_thesis.optimizer import solve_robust_mvo

    crsp_ret_matrix = precomp["crsp_ret_matrix"]
    crsp_yyyymms = precomp["crsp_yyyymms"]
    crsp_permnos_all = precomp["crsp_permnos_all"]
    results: list[MonthlyResult] = []
    for mo in precomp["months"]:
        yyyymm = mo["yyyymm"]
        if mo["sigma"] is None:
            k = mo["k"]
            weights = np.ones(k, dtype=np.float32) / k
            permnos, mu_out = mo["permnos"], mo["fallback_mu"]
        else:
            weights = solve_robust_mvo(
                mo["mu"], mo["sigma"], kappa=config.kappa, lambd=config.lambd,
                omega_mode=config.omega_mode, n_obs=config.lookback,
            ).astype(np.float32)
            permnos, mu_out = mo["permnos"], mo["mu"].astype(np.float32)
        realized = _compute_realized(
            permnos, weights, _next_month(yyyymm),
            crsp_ret_matrix, crsp_yyyymms, crsp_permnos_all, mo["rfree"],
        )
        results.append(MonthlyResult(
            yyyymm=yyyymm, n_universe=mo["n_universe"], n_selected=len(permnos),
            weights=weights, permnos=permnos, mu_hat=mu_out, realized_return=realized,
        ))
    return BacktestResult(split=split, config=config, monthly=results)


def solve_pto_grid(
    precomp: dict,
    kappa_grid: list,
    lambda_grid: list,
    omega_mode: str = "diag_sigma",
    split: str = "val",
    base_config: PtoConfig | None = None,
) -> list[dict]:
    """Sweep κ×λ over precomputed monthly inputs (μ̂/Σ computed once). Returns one
    `portfolio_metrics` dict per (κ, λ) cell (plus the κ/λ keys)."""
    base = base_config or PtoConfig()
    out: list[dict] = []
    for lam in lambda_grid:
        for kap in kappa_grid:
            cfg = dataclasses.replace(base, kappa=float(kap), lambd=float(lam), omega_mode=omega_mode)
            res = _backtest_from_precomputed(precomp, cfg, split)
            m = res.portfolio_metrics()
            m.update({"kappa": float(kap), "lambda": float(lam)})
            out.append(m)
    return out


# ──────────────────────────────────────────────────────────────────────────────
#  Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _next_month(yyyymm: int) -> int:
    y, m = divmod(yyyymm, 100)
    m += 1
    if m > 12:
        m = 1
        y += 1
    return y * 100 + m


def _compute_realized(
    permnos: np.ndarray,
    weights: np.ndarray,
    next_yyyymm: int,
    crsp_ret_matrix: np.ndarray,
    crsp_yyyymms: np.ndarray,
    crsp_permnos_all: np.ndarray,
    rfree: float,
) -> float:
    """Compute w' r_{t+1} − rfree for the given portfolio."""
    row_idx_arr = np.where(crsp_yyyymms == next_yyyymm)[0]
    if len(row_idx_arr) == 0:
        return float("nan")
    row_idx = int(row_idx_arr[0])

    permno_to_col = {p: i for i, p in enumerate(crsp_permnos_all)}
    portfolio_ret = 0.0
    weight_used = 0.0
    for p, w in zip(permnos, weights):
        col = permno_to_col.get(int(p))
        if col is None:
            continue
        ret = crsp_ret_matrix[row_idx, col]
        if np.isfinite(ret):
            portfolio_ret += float(w) * float(ret)
            weight_used += float(w)

    if weight_used < 1e-6:
        return float("nan")
    # Rescale by weight actually used
    return portfolio_ret / weight_used - rfree


import copy


from poe_thesis.optimizer import DifferentiableRobustMVOLayer
from poe_thesis.predictors import AssetPricingFNN, load_fnn_from_dir
from poe_thesis.optimizer import cholesky_psd
from poe_thesis.optimizer import build_omega_half

log = logging.getLogger(__name__)

DECISION_LOSSES = ("robust_utility", "mv_utility", "neg_return", "neg_sharpe")


@dataclasses.dataclass
class PaoConfig:
    """Decision-focused (E2E) training hyper-parameters. All are Batuhan's choices."""
    decision_loss: str = "robust_utility"   # one of DECISION_LOSSES
    kappa: float = 0.5
    lambd: float = 10.0
    omega_mode: str = "diag_sigma"
    cov_estimator: str = "nl_shrinkage"
    universe_size: int = 100               # final top-N per month (largest-cap, or top-K by μ̂ for size_then_mu)
    universe_rule: str = "size"            # "size" (top-N cap) or "size_then_mu" (top-N by μ̂ within liquid screen)
    prescreen_size: int = 1000             # Stage-1 liquid pre-screen (top-M cap) for "size_then_mu"
    lookback: int = 60
    min_obs: int = 24
    months_per_batch: int = 12             # months per optimizer step
    lr: float = 3e-4
    weight_decay: float = 1e-5
    grad_clip: float = 1.0
    epochs: int = 15
    patience: int = 4
    seed: int = 42
    warm_start_member: str = "seed_42"     # GKX ensemble member to warm-start from


# ──────────────────────────────────────────────────────────────────────────────
#  Decision losses (operate on a batch of per-month portfolio outcomes)
# ──────────────────────────────────────────────────────────────────────────────

def _batch_decision_loss(
    decision_loss: str,
    port_rets: torch.Tensor,   # (B,) realized portfolio excess returns
    risks: torch.Tensor,       # (B,) w'Σw per month
    robpens: torch.Tensor,     # (B,) ‖A w‖₂ per month
    kappa: float,
    lambd: float,
) -> torch.Tensor:
    """Negative realized objective over a batch of B months (to minimize)."""
    if decision_loss == "neg_return":
        return -port_rets.mean()
    if decision_loss == "mv_utility":
        return -(port_rets - (lambd / 2.0) * risks).mean()
    if decision_loss == "robust_utility":
        return -(port_rets - kappa * robpens - (lambd / 2.0) * risks).mean()
    if decision_loss == "neg_sharpe":
        mean = port_rets.mean()
        std = port_rets.std(unbiased=False) + 1e-8
        return -(mean / std)
    raise ValueError(f"unknown decision_loss: {decision_loss!r} (expected {DECISION_LOSSES})")


# ──────────────────────────────────────────────────────────────────────────────
#  Per-month decision step (differentiable)
# ──────────────────────────────────────────────────────────────────────────────

def _month_decision(
    model: AssetPricingFNN,
    layer: DifferentiableRobustMVOLayer,
    X_month: torch.Tensor,
    y_month: torch.Tensor,
    permno_month: np.ndarray,
    yyyymm: int,
    *,
    cfg: PaoConfig,
    crsp_ret: np.ndarray,
    crsp_permnos: np.ndarray,
    crsp_yyyymms: np.ndarray,
    size_col: int,
    layer_cache: dict,
):
    """Return (port_ret, risk, robpen) torch scalars for one month, or None if infeasible.

    `layer_cache` maps an asset count → a DifferentiableRobustMVOLayer of that size
    (the cvxpy problem is fixed-shape, so one layer per realized universe size).
    """
    n_all = X_month.shape[0]
    size_sig = X_month[:, size_col].detach().cpu().numpy()
    # Stage-1 liquid candidate set by market cap (smallest Size signal = largest cap), with headroom
    # so the CRSP-history filter still leaves enough names to trim to the target K.
    if cfg.universe_rule == "size_then_mu":
        cand_k = int(min(max(cfg.prescreen_size, cfg.universe_size + 50), n_all))
    else:
        cand_k = int(min(max(2 * cfg.universe_size, cfg.universe_size + 50), n_all))
    cand = np.argsort(size_sig)[:cand_k]                  # X_month row indices (largest caps)
    cand_permnos = permno_month[cand]

    valid_permnos, sigma = build_monthly_covariance(
        crsp_ret, crsp_permnos, crsp_yyyymms, cand_permnos,
        yyyymm, cfg.lookback, cfg.min_obs, cfg.cov_estimator,
    )
    if len(valid_permnos) < 2:
        return None
    cand_pos = {int(p): i for i, p in enumerate(cand_permnos.tolist())}     # permno → cand-local idx
    valid_local = np.array([cand_pos[int(p)] for p in valid_permnos.tolist()], dtype=np.int64)

    # Trim to a FIXED universe size K among the history-valid firms so the cvxpy layer shape is CONSTANT
    # across months (canonicalized once / cached — the N-perf fix). `keep` indexes into valid_permnos.
    k = min(cfg.universe_size, len(valid_permnos))
    if cfg.universe_rule == "size_then_mu":
        # Stage-2: top-K by PREDICTED return μ̂ within the liquid set (prediction-driven; mirrors the
        # PTO size_then_mu rule). Forward μ̂ once over the candidate set — differentiable, also feeds the layer.
        mu_cand = model(X_month[torch.as_tensor(cand, dtype=torch.long)])   # (cand_k,) differentiable
        mu_valid = mu_cand[torch.as_tensor(valid_local, dtype=torch.long)]  # aligned to valid_permnos
        keep = np.sort(np.argpartition(mu_valid.detach().cpu().numpy(), -k)[-k:])
        mu_hat = mu_valid[torch.as_tensor(keep, dtype=torch.long)]          # (K,) differentiable
    else:
        # legacy: top-K largest cap among valid
        size_of = {int(p): float(s) for p, s in zip(permno_month.tolist(), size_sig.tolist())}
        valid_sizes = np.array([size_of[int(p)] for p in valid_permnos.tolist()])
        keep = np.sort(np.argsort(valid_sizes)[:k])
        mu_hat = model(X_month[torch.as_tensor(cand[valid_local[keep]], dtype=torch.long)])  # (K,) differentiable
    kept_permnos = valid_permnos[keep]
    sigma = sigma[np.ix_(keep, keep)]                     # principal submatrix of a PD matrix → PD
    r = y_month[torch.as_tensor(cand[valid_local[keep]], dtype=torch.long)].double()  # realized excess

    n = len(kept_permnos)
    layer = layer_cache.get(n)
    if layer is None:
        layer = DifferentiableRobustMVOLayer(n, lambd=cfg.lambd, kappa=cfg.kappa)
        layer_cache[n] = layer

    U = torch.as_tensor(cholesky_psd(sigma), dtype=torch.float64)
    A = torch.as_tensor(
        build_omega_half(sigma, cfg.omega_mode, n_obs=cfg.lookback), dtype=torch.float64
    )
    w = layer(mu_hat, U, A).double()

    sigma_t = torch.as_tensor(sigma, dtype=torch.float64)
    port_ret = (w * r).sum()
    risk = w @ (sigma_t @ w)
    robpen = torch.linalg.norm(A @ w)
    return port_ret, risk, robpen


# ──────────────────────────────────────────────────────────────────────────────
#  Training
# ──────────────────────────────────────────────────────────────────────────────

def _warm_start_model(gkx_dir: Path, member: str, input_dim: int, dropout: float) -> AssetPricingFNN:
    member_dir = gkx_dir / "members" / member
    if not (member_dir / "state_dict.pt").exists():
        # single-checkpoint ensemble (ensemble_size==1) writes at the top level
        member_dir = gkx_dir
    model, _, _ = load_fnn_from_dir(member_dir, map_location="cpu")
    fresh = AssetPricingFNN(input_dim=input_dim, dropout_rate=dropout)
    fresh.load_state_dict(model.state_dict())
    return fresh


def _eval_val_sharpe(model, val_dir, n_features, device, cfg, crsp, crsp_p, crsp_y, size_col):
    """Annualized Sharpe of the decision portfolio on the val split (no grad)."""
    model.eval()
    rets = []
    cache: dict = {}
    with torch.no_grad():
        for X_m, y_m, permno_m, yyyymm_arr in iter_shards_monthly(val_dir, n_features, device):
            out = _month_decision(
                model, None, X_m, y_m, permno_m, int(yyyymm_arr[0]),
                cfg=cfg, crsp_ret=crsp, crsp_permnos=crsp_p, crsp_yyyymms=crsp_y,
                size_col=size_col, layer_cache=cache,
            )
            if out is not None:
                rets.append(float(out[0].item()))
    model.train()
    if len(rets) < 2:
        return float("nan")
    r = np.array(rets)
    return float(r.mean() / (r.std() + 1e-12) * np.sqrt(12))


def train_pao(
    *,
    gkx_dir: Path,
    features_dir: Path,
    crsp_path: Path,
    feature_cols: list,
    input_dim: int,
    dropout: float,
    cfg: PaoConfig,
    device: str = "cpu",
) -> dict:
    """Run decision-focused E2E training; returns a result dict + writes nothing.

    The caller persists the model (state_dict) and manifest.
    """
    if cfg.decision_loss not in DECISION_LOSSES:
        raise ValueError(f"decision_loss must be one of {DECISION_LOSSES}")
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    size_col = feature_cols.index("Size")
    train_dir = features_dir / "train"
    val_dir = features_dir / "val"

    log.info("Loading CRSP return matrix for covariance …")
    crsp, crsp_y, crsp_p = load_crsp_return_matrix(crsp_path)

    model = _warm_start_model(gkx_dir, cfg.warm_start_member, input_dim, dropout).to(device)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    layer_cache: dict = {}

    best_val = -float("inf")
    best_state = copy.deepcopy(model.state_dict())
    best_epoch = -1
    no_improve = 0
    history = []

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        batch: list = []
        step_losses = []
        n_months = 0
        for X_m, y_m, permno_m, yyyymm_arr in iter_shards_monthly(train_dir, input_dim, device):
            out = _month_decision(
                model, None, X_m, y_m, permno_m, int(yyyymm_arr[0]),
                cfg=cfg, crsp_ret=crsp, crsp_permnos=crsp_p, crsp_yyyymms=crsp_y,
                size_col=size_col, layer_cache=layer_cache,
            )
            if out is None:
                continue
            batch.append(out)
            n_months += 1
            if len(batch) >= cfg.months_per_batch:
                step_losses.append(_optimizer_step(optimizer, model, batch, cfg))
                batch = []
        if batch:
            step_losses.append(_optimizer_step(optimizer, model, batch, cfg))

        val_sharpe = _eval_val_sharpe(
            model, val_dir, input_dim, device, cfg, crsp, crsp_p, crsp_y, size_col
        )
        train_loss = float(np.mean(step_losses)) if step_losses else float("nan")
        log.info("[%s] epoch %2d/%d | train_loss=%.5f | val_sharpe=%.4f | months=%d",
                 cfg.decision_loss, epoch, cfg.epochs, train_loss, val_sharpe, n_months)
        history.append({"epoch": epoch, "train_loss": train_loss, "val_sharpe": val_sharpe})

        if np.isfinite(val_sharpe) and val_sharpe > best_val:
            best_val = val_sharpe
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= cfg.patience:
                log.info("[%s] early stop at epoch %d (best epoch=%d val_sharpe=%.4f)",
                         cfg.decision_loss, epoch, best_epoch, best_val)
                break

    model.load_state_dict(best_state)
    return {
        "decision_loss": cfg.decision_loss,
        "best_epoch": best_epoch,
        "best_val_sharpe": best_val,
        "history": history,
        "state_dict": best_state,
        "config": dataclasses.asdict(cfg),
    }


def _optimizer_step(optimizer, model, batch, cfg) -> float:
    port_rets = torch.stack([b[0] for b in batch])
    risks = torch.stack([b[1] for b in batch])
    robpens = torch.stack([b[2] for b in batch])
    loss = _batch_decision_loss(cfg.decision_loss, port_rets, risks, robpens, cfg.kappa, cfg.lambd)
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
    optimizer.step()
    return float(loss.item())
