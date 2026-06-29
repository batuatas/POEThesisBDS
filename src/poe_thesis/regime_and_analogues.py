# Regime classifier + nearest-historical-analogue (NFCI-style) diagnostics over the standardized macro state.
from __future__ import annotations


"""Historical standardized macro support diagnostics with no sampling behavior."""


from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

from poe_thesis.macro_features import (
    EXPECTED_MACRO_VARIABLES,
    MacroScaler,
    transform_macro_panel,
)


@dataclass(frozen=True)
class MacroSupportConfig:
    """Configuration for descriptive historical macro support diagnostics."""

    anchor_yyyymm: int = 202004
    macro_order: tuple[str, ...] = EXPECTED_MACRO_VARIABLES
    quantile_levels: tuple[float, ...] = (0.01, 0.05, 0.50, 0.95, 0.99)
    lower_quantile_bound: float = 0.01
    upper_quantile_bound: float = 0.99
    nearest_k: int = 5
    mahalanobis_regularization: float = 1e-6
    maximum_condition_number: float = 1e12
    reference_cutoff_policy: str = "strictly_pre_anchor"
    diagnostic_scope: str = "historical_macro_support_no_sampling"

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["macro_order"] = list(self.macro_order)
        payload["quantile_levels"] = list(self.quantile_levels)
        return payload


@dataclass(frozen=True)
class MacroSupportDiagnostics:
    """Reference-panel summaries shared by evaluated macro states."""

    reference_start_yyyymm: int
    reference_end_yyyymm: int
    reference_row_count: int
    raw_history_shape: tuple[int, int]
    standardized_history_shape: tuple[int, int]
    standardized_history_finite: bool
    covariance_condition_number: float
    covariance_stable: bool
    per_variable_summary: dict[str, dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class StateSupportDiagnostics:
    """Descriptive support diagnostics for one standardized macro state."""

    label: str
    raw_state: tuple[float, ...]
    standardized_state: tuple[float, ...]
    percentile_ranks: dict[str, float]
    historical_range_violations: dict[str, bool]
    quantile_bound_violations: dict[str, bool]
    euclidean_distance_to_historical_mean: float
    nearest_euclidean_analogs: tuple[dict[str, Any], ...]
    mahalanobis_distance_to_historical_mean: float | None
    nearest_mahalanobis_analogs: tuple[dict[str, Any], ...]
    covariance_stable: bool
    descriptive_only: bool = True
    plausible_or_implausible_classification_made: bool = False
    scenario_claim_made: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def validate_macro_support_config(config: MacroSupportConfig) -> None:
    """Raise when the historical-support configuration is invalid."""

    if not isinstance(config, MacroSupportConfig):
        raise TypeError("config must be a MacroSupportConfig")
    if tuple(config.macro_order) != EXPECTED_MACRO_VARIABLES:
        raise ValueError("macro_order must match the canonical ordered nine-variable list")
    if not isinstance(config.anchor_yyyymm, int) or config.anchor_yyyymm <= 0:
        raise ValueError("anchor_yyyymm must be a positive integer")
    levels = np.asarray(config.quantile_levels, dtype=np.float64)
    if levels.ndim != 1 or levels.size == 0 or not np.isfinite(levels).all():
        raise ValueError("quantile_levels must be a nonempty finite sequence")
    if np.any(levels <= 0) or np.any(levels >= 1) or np.any(np.diff(levels) <= 0):
        raise ValueError("quantile_levels must be strictly increasing values between 0 and 1")
    if (
        config.lower_quantile_bound not in config.quantile_levels
        or config.upper_quantile_bound not in config.quantile_levels
        or config.lower_quantile_bound >= config.upper_quantile_bound
    ):
        raise ValueError("quantile bounds must be ordered members of quantile_levels")
    if not isinstance(config.nearest_k, int) or config.nearest_k <= 0:
        raise ValueError("nearest_k must be a positive integer")
    if (
        not np.isfinite(config.mahalanobis_regularization)
        or config.mahalanobis_regularization <= 0
    ):
        raise ValueError("mahalanobis_regularization must be finite and positive")
    if not np.isfinite(config.maximum_condition_number) or config.maximum_condition_number <= 1:
        raise ValueError("maximum_condition_number must be finite and greater than one")
    if config.reference_cutoff_policy != "strictly_pre_anchor":
        raise ValueError("reference_cutoff_policy must be strictly_pre_anchor")
    if config.diagnostic_scope != "historical_macro_support_no_sampling":
        raise ValueError("diagnostic_scope must be historical_macro_support_no_sampling")






def _regularized_inverse_covariance(
    standardized_history: np.ndarray, config: MacroSupportConfig
) -> tuple[np.ndarray | None, float, bool]:
    covariance = np.cov(standardized_history, rowvar=False, ddof=1)
    regularized = covariance + config.mahalanobis_regularization * np.eye(covariance.shape[0])
    condition_number = float(np.linalg.cond(regularized))
    stable = bool(np.isfinite(condition_number) and condition_number <= config.maximum_condition_number)
    if not stable:
        return None, condition_number, False
    try:
        inverse = np.linalg.inv(regularized)
    except np.linalg.LinAlgError:
        return None, condition_number, False
    return inverse, condition_number, bool(np.isfinite(inverse).all())


def regularized_mahalanobis_distance(
    state: Sequence[float] | np.ndarray,
    center: Sequence[float] | np.ndarray,
    inverse_covariance: np.ndarray,
) -> float:
    """Return a finite regularized Mahalanobis distance."""

    state_array = np.asarray(state, dtype=np.float64)
    center_array = np.asarray(center, dtype=np.float64)
    inverse = np.asarray(inverse_covariance, dtype=np.float64)
    if state_array.shape != (9,) or center_array.shape != (9,) or inverse.shape != (9, 9):
        raise ValueError("state, center, and inverse_covariance must have shapes (9,), (9,), (9, 9)")
    if not np.isfinite(state_array).all() or not np.isfinite(center_array).all() or not np.isfinite(inverse).all():
        raise ValueError("Mahalanobis inputs must be finite")
    delta = state_array - center_array
    squared = float(delta @ inverse @ delta)
    if not np.isfinite(squared) or squared < -1e-10:
        raise ValueError("Mahalanobis squared distance is invalid")
    return float(np.sqrt(max(0.0, squared)))


def nearest_historical_analogs(
    state: Sequence[float] | np.ndarray,
    standardized_history: np.ndarray,
    history_dates: Sequence[int] | np.ndarray,
    *,
    nearest_k: int,
    inverse_covariance: np.ndarray | None = None,
) -> tuple[dict[str, Any], ...]:
    """Return nearest historical dates and distances under Euclidean or Mahalanobis distance."""

    state_array = np.asarray(state, dtype=np.float64)
    history = np.asarray(standardized_history, dtype=np.float64)
    dates = np.asarray(history_dates)
    if state_array.shape != (9,) or history.ndim != 2 or history.shape[1] != 9:
        raise ValueError("state and standardized_history must have shapes (9,) and (n, 9)")
    if dates.shape != (history.shape[0],) or nearest_k <= 0:
        raise ValueError("history_dates and nearest_k are invalid")
    deltas = history - state_array
    if inverse_covariance is None:
        distances = np.linalg.norm(deltas, axis=1)
        metric = "standardized_euclidean"
    else:
        inverse = np.asarray(inverse_covariance, dtype=np.float64)
        if inverse.shape != (9, 9) or not np.isfinite(inverse).all():
            raise ValueError("inverse_covariance must be finite with shape (9, 9)")
        squared = np.einsum("ij,jk,ik->i", deltas, inverse, deltas)
        distances = np.sqrt(np.maximum(0.0, squared))
        metric = "regularized_mahalanobis"
    if not np.isfinite(distances).all():
        raise ValueError("nearest-analog distances must be finite")
    order = np.argsort(distances, kind="stable")[: min(nearest_k, history.shape[0])]
    return tuple(
        {"yyyymm": int(dates[index]), "distance": float(distances[index]), "metric": metric}
        for index in order
    )


def compute_macro_support_summary(
    reference_panel: pd.DataFrame,
    standardized_history: np.ndarray,
    config: MacroSupportConfig,
) -> MacroSupportDiagnostics:
    """Compute descriptive reference-panel support summaries."""

    validate_macro_support_config(config)
    history = np.asarray(standardized_history, dtype=np.float64)
    if history.shape != (len(reference_panel), 9) or not np.isfinite(history).all():
        raise ValueError("standardized_history does not match the reference panel")
    _, condition_number, stable = _regularized_inverse_covariance(history, config)
    summaries: dict[str, dict[str, Any]] = {}
    for index, name in enumerate(config.macro_order):
        values = history[:, index]
        summaries[name] = {
            "minimum": float(np.min(values)),
            "maximum": float(np.max(values)),
            "quantiles": {
                f"{level:.2f}": float(np.quantile(values, level))
                for level in config.quantile_levels
            },
        }
    return MacroSupportDiagnostics(
        reference_start_yyyymm=int(reference_panel["yyyymm"].iloc[0]),
        reference_end_yyyymm=int(reference_panel["yyyymm"].iloc[-1]),
        reference_row_count=len(reference_panel),
        raw_history_shape=(len(reference_panel), 9),
        standardized_history_shape=history.shape,
        standardized_history_finite=True,
        covariance_condition_number=condition_number,
        covariance_stable=stable,
        per_variable_summary=summaries,
    )


def evaluate_macro_state_support(
    *,
    label: str,
    raw_state: Sequence[float] | np.ndarray,
    standardized_state: Sequence[float] | np.ndarray,
    reference_panel: pd.DataFrame,
    standardized_history: np.ndarray,
    config: MacroSupportConfig,
) -> StateSupportDiagnostics:
    """Evaluate one state descriptively against strict pre-anchor history."""

    raw = np.asarray(raw_state, dtype=np.float64)
    state = np.asarray(standardized_state, dtype=np.float64)
    history = np.asarray(standardized_history, dtype=np.float64)
    if raw.shape != (9,) or state.shape != (9,) or not np.isfinite(raw).all() or not np.isfinite(state).all():
        raise ValueError("raw_state and standardized_state must be finite with shape (9,)")
    if history.shape != (len(reference_panel), 9) or not np.isfinite(history).all():
        raise ValueError("standardized_history does not match the reference panel")
    inverse, _, stable = _regularized_inverse_covariance(history, config)
    mean = np.mean(history, axis=0)
    percentiles: dict[str, float] = {}
    range_violations: dict[str, bool] = {}
    quantile_violations: dict[str, bool] = {}
    for index, name in enumerate(config.macro_order):
        values = history[:, index]
        percentiles[name] = float(np.mean(values <= state[index]))
        range_violations[name] = bool(state[index] < np.min(values) or state[index] > np.max(values))
        lower = np.quantile(values, config.lower_quantile_bound)
        upper = np.quantile(values, config.upper_quantile_bound)
        quantile_violations[name] = bool(state[index] < lower or state[index] > upper)
    dates = reference_panel["yyyymm"].to_numpy()
    euclidean = nearest_historical_analogs(
        state, history, dates, nearest_k=config.nearest_k
    )
    mahalanobis_to_mean = None
    mahalanobis_analogs: tuple[dict[str, Any], ...] = ()
    if stable and inverse is not None:
        mahalanobis_to_mean = regularized_mahalanobis_distance(state, mean, inverse)
        mahalanobis_analogs = nearest_historical_analogs(
            state, history, dates, nearest_k=config.nearest_k, inverse_covariance=inverse
        )
    return StateSupportDiagnostics(
        label=label,
        raw_state=tuple(raw),
        standardized_state=tuple(state),
        percentile_ranks=percentiles,
        historical_range_violations=range_violations,
        quantile_bound_violations=quantile_violations,
        euclidean_distance_to_historical_mean=float(np.linalg.norm(state - mean)),
        nearest_euclidean_analogs=euclidean,
        mahalanobis_distance_to_historical_mean=mahalanobis_to_mean,
        nearest_mahalanobis_analogs=mahalanobis_analogs,
        covariance_stable=stable,
    )


from collections import Counter
from dataclasses import dataclass, field


from poe_thesis.macro_features import EXPECTED_MACRO_VARIABLES, MacroScaler

_IDX = {name: i for i, name in enumerate(EXPECTED_MACRO_VARIABLES)}  # dp,ep,bm,ntis,tbl,tms,dfy,svar,infl


@dataclass(frozen=True)
class LandingZoneConfig:
    """Weights + percentile thresholds for the rule-based NFCI-analog / recession stance."""
    nearest_k: int = 5
    w_credit: float = 0.4    # dfy (BAA-AAA default spread)
    w_vol: float = 0.4       # svar (realized stock variance ≈ VIX)
    w_curve: float = 0.2     # tms (term slope; LOW slope = tight/inverted = stress)
    stress_high: float = 0.75
    stress_moderate: float = 0.60
    risk_on: float = 0.40
    hi: float = 0.75         # "high" percentile gate (rates/inflation)
    lo: float = 0.25         # "low" percentile gate (curve/inflation)


def _pctl_ranks(z: np.ndarray, history: np.ndarray) -> np.ndarray:
    """Percentile rank (∈[0,1]) of each coord of z within the standardized history (n,9)."""
    return (np.asarray(history) < np.asarray(z)).mean(axis=0)


def score_state(z: Sequence[float], history: np.ndarray, config: LandingZoneConfig) -> dict[str, Any]:
    """NFCI-analog stress index + financial-conditions level + recession stance for one standardized state."""
    p = _pctl_ranks(np.asarray(z, dtype=np.float64), history)
    dfy_p, svar_p, tms_p = p[_IDX["dfy"]], p[_IDX["svar"]], p[_IDX["tms"]]
    tbl_p, infl_p = p[_IDX["tbl"]], p[_IDX["infl"]]
    nfci = config.w_credit * dfy_p + config.w_vol * svar_p + config.w_curve * (1.0 - tms_p)  # high = stress
    level = ("high_stress" if nfci >= config.stress_high
             else "moderate_stress" if nfci >= config.stress_moderate
             else "risk_on" if nfci <= config.risk_on else "neutral")
    if tms_p < config.lo and tbl_p > config.hi:
        stance = "tightening_inversion"
    elif tms_p < 0.5 and tbl_p > config.hi:
        stance = "flat_curve_tightening"
    elif tbl_p > config.hi:
        stance = "high_rates"
    elif infl_p > config.hi:
        stance = "inflation"
    elif infl_p < config.lo:
        stance = "disinflation_slowdown"
    else:
        stance = "neutral"
    return {"nfci": float(nfci), "stress_level": level, "recession_stance": stance,
            "dfy_pctl": float(dfy_p), "svar_pctl": float(svar_p), "tms_pctl": float(tms_p)}




def land_scenario_cloud(
    samples: np.ndarray, history: np.ndarray, history_dates: Sequence[int], *,
    config: LandingZoneConfig | None = None, inverse_covariance: np.ndarray | None = None, max_eval: int = 500,
) -> dict[str, Any]:
    """Economic landing-zone summary of a posterior scenario cloud (n,9 of standardized z).

    Returns the nearest historical month(s) of the cloud mean, plus the DISTRIBUTION over financial-conditions
    stress levels and recession stances across the cloud (the regime the scenarios land in)."""
    config = config or LandingZoneConfig()
    flat = np.asarray(samples, dtype=np.float64).reshape(-1, 9)
    zbar = flat.mean(axis=0)
    analogs = nearest_historical_analogs(zbar, history, history_dates, nearest_k=config.nearest_k,
                                         inverse_covariance=inverse_covariance)
    sub = flat[np.linspace(0, len(flat) - 1, min(max_eval, len(flat))).astype(int)]
    scores = [score_state(z, history, config) for z in sub]
    nfcis = np.array([s["nfci"] for s in scores])
    stance_dist = {k: round(v / len(scores), 3) for k, v in Counter(s["recession_stance"] for s in scores).items()}
    level_dist = {k: round(v / len(scores), 3) for k, v in Counter(s["stress_level"] for s in scores).items()}
    nfci_mean = float(nfcis.mean())
    dom_level = max(level_dist, key=level_dist.get)
    dom_stance = max(stance_dist, key=stance_dist.get)

    # Anchor baseline (the model's conditioning state = last strictly-pre-anchor month). The cloud's regime is
    # dominated by this baseline (plausibility keeps scenarios near it); report the SHIFT, not just the label.
    anchor = score_state(np.asarray(history, dtype=np.float64)[-1], history, config)
    delta_nfci = nfci_mean - anchor["nfci"]
    return {
        "nearest_months": [int(a["yyyymm"]) for a in analogs],
        "nearest_distances": [round(float(a["distance"]), 3) for a in analogs],
        "analog_metric": analogs[0]["metric"] if analogs else None,
        "nfci_mean": round(nfci_mean, 3),
        "nfci_std": round(float(nfcis.std()), 3),
        "stress_level_distribution": dict(sorted(level_dist.items(), key=lambda kv: -kv[1])),
        "recession_stance_distribution": dict(sorted(stance_dist.items(), key=lambda kv: -kv[1])),
        "dominant_stress_level": dom_level,
        "dominant_recession_stance": dom_stance,
        # anchor baseline + shift-vs-anchor (the honest per-probe signal: same-anchor probes inherit the regime)
        "anchor_nfci": round(float(anchor["nfci"]), 3),
        "anchor_stress_level": anchor["stress_level"],
        "anchor_recession_stance": anchor["recession_stance"],
        "delta_nfci": round(float(delta_nfci), 3),
        "stress_vs_anchor": "same" if dom_level == anchor["stress_level"] else f"{anchor['stress_level']}→{dom_level}",
        "stance_vs_anchor": "same" if dom_stance == anchor["recession_stance"] else f"{anchor['recession_stance']}→{dom_stance}",
        "n_eval": int(len(sub)),
    }
