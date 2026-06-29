# Plausibility prior: Goyal-Welch VAR(1) fit + stationary-law moments (mu*, Sigma*) and innovation scoring.
from __future__ import annotations


"""Clean VAR(1)-style macro innovation diagnostics with no sampling behavior."""


from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
from scipy.stats import chi2

from poe_thesis.macro_features import EXPECTED_MACRO_VARIABLES, MacroScaler


@dataclass(frozen=True)
class VAR1PlausibilityConfig:
    """Configuration for a bounded clean VAR(1) innovation diagnostic."""

    anchor_yyyymm: int = 202004
    fit_end_yyyymm: int = 202003
    lag_order: int = 1
    include_intercept: bool = True
    macro_order: tuple[str, ...] = EXPECTED_MACRO_VARIABLES
    residual_covariance_regularization: float = 1e-6
    maximum_condition_number: float = 1e12
    top_innovation_count: int = 3
    diagnostic_scope: str = "var1_plausibility_no_sampling"

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["macro_order"] = list(self.macro_order)
        return payload


@dataclass(frozen=True)
class VAR1FitDiagnostics:
    """Fit and stability diagnostics for one clean standardized VAR(1)."""

    fit_start_yyyymm: int
    fit_end_yyyymm: int
    history_row_count: int
    fit_observation_count: int
    coefficient_shape: tuple[int, int]
    intercept_shape: tuple[int, ...]
    residual_shape: tuple[int, int]
    residual_covariance_shape: tuple[int, int]
    regularized_covariance_shape: tuple[int, int]
    regularized_covariance_condition_number: float
    residual_covariance_stable: bool
    maximum_absolute_eigenvalue: float
    var1_dynamics_stable: bool
    residual_summary_by_macro: dict[str, dict[str, float]]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class VAR1StateDiagnostics:
    """One descriptive one-step VAR(1) innovation score."""

    label: str
    target_yyyymm: int
    lag_yyyymm: int
    target_z: tuple[float, ...]
    lag_z: tuple[float, ...]
    predicted_z: tuple[float, ...]
    innovation: tuple[float, ...]
    innovation_l2_norm: float
    innovation_max_absolute: float
    innovation_mahalanobis_distance: float | None
    innovation_chi_square_tail_probability: float | None
    top_innovation_macros: tuple[dict[str, Any], ...]
    in_sample_fit_target: bool
    descriptive_only: bool = True
    forecast_validation_claim_made: bool = False
    plausible_or_implausible_classification_made: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class VAR1PlausibilityDiagnostics:
    """Combined fit, anchor, and optional historical analog diagnostics."""

    fit: VAR1FitDiagnostics
    anchor: VAR1StateDiagnostics
    analog_states: tuple[VAR1StateDiagnostics, ...]
    analog_skip_reasons: tuple[dict[str, Any], ...]
    descriptive_only: bool = True
    scenario_claim_made: bool = False
    forecast_validation_claim_made: bool = False
    causal_claim_made: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class VAR1Fit:
    """In-memory fitted parameters required for scoring."""

    intercept: np.ndarray
    coefficient: np.ndarray
    residuals: np.ndarray
    residual_covariance: np.ndarray
    regularized_residual_covariance: np.ndarray
    inverse_regularized_residual_covariance: np.ndarray | None
    fit_diagnostics: VAR1FitDiagnostics


def validate_var1_config(config: VAR1PlausibilityConfig) -> None:
    """Raise when the bounded clean VAR(1) configuration is invalid."""

    if not isinstance(config, VAR1PlausibilityConfig):
        raise TypeError("config must be a VAR1PlausibilityConfig")
    if tuple(config.macro_order) != EXPECTED_MACRO_VARIABLES:
        raise ValueError("macro_order must match the canonical ordered nine-variable list")
    if config.lag_order != 1:
        raise ValueError("only lag_order=1 is supported")
    if config.include_intercept is not True:
        raise ValueError("include_intercept must remain true")
    if config.fit_end_yyyymm >= config.anchor_yyyymm:
        raise ValueError("fit_end_yyyymm must be strictly before anchor_yyyymm")
    if (
        not np.isfinite(config.residual_covariance_regularization)
        or config.residual_covariance_regularization <= 0
    ):
        raise ValueError("residual_covariance_regularization must be finite and positive")
    if not np.isfinite(config.maximum_condition_number) or config.maximum_condition_number <= 1:
        raise ValueError("maximum_condition_number must be finite and greater than one")
    if (
        not isinstance(config.top_innovation_count, int)
        or config.top_innovation_count <= 0
        or config.top_innovation_count > 9
    ):
        raise ValueError("top_innovation_count must be between one and nine")
    if config.diagnostic_scope != "var1_plausibility_no_sampling":
        raise ValueError("unsupported diagnostic_scope")




def regularize_residual_covariance(
    residual_covariance: np.ndarray,
    *,
    ridge: float,
    maximum_condition_number: float,
) -> tuple[np.ndarray, np.ndarray | None, float, bool]:
    """Apply ridge regularization and return inverse only when stable."""

    covariance = np.asarray(residual_covariance, dtype=np.float64)
    if covariance.shape != (9, 9) or not np.isfinite(covariance).all():
        raise ValueError("residual_covariance must be finite with shape (9, 9)")
    if not np.isfinite(ridge) or ridge <= 0:
        raise ValueError("ridge must be finite and positive")
    regularized = covariance + ridge * np.eye(9)
    condition_number = float(np.linalg.cond(regularized))
    stable = bool(
        np.isfinite(condition_number) and condition_number <= maximum_condition_number
    )
    if not stable:
        return regularized, None, condition_number, False
    try:
        inverse = np.linalg.inv(regularized)
    except np.linalg.LinAlgError:
        return regularized, None, condition_number, False
    return regularized, inverse, condition_number, bool(np.isfinite(inverse).all())


def compute_var1_stability_diagnostics(coefficient: np.ndarray) -> tuple[float, bool]:
    """Return spectral radius and the standard VAR(1) dynamics stability flag."""

    matrix = np.asarray(coefficient, dtype=np.float64)
    if matrix.shape != (9, 9) or not np.isfinite(matrix).all():
        raise ValueError("coefficient must be finite with shape (9, 9)")
    maximum = float(np.max(np.abs(np.linalg.eigvals(matrix))))
    return maximum, bool(np.isfinite(maximum) and maximum < 1.0)


@dataclass(frozen=True)
class VAR1StationaryMoments:
    """Unconditional (stationary) moments of a fitted VAR(1). `stable` is False (None moments) when
    ρ(Φ)≥1 or the covariance solve is ill-conditioned — the caller then falls back to one-step."""

    mean: np.ndarray | None
    covariance: np.ndarray | None
    inverse_covariance: np.ndarray | None
    spectral_radius: float
    stable: bool


def var1_stationary_moments(
    coefficient: np.ndarray,
    intercept: np.ndarray,
    residual_covariance: np.ndarray,
    *,
    ridge: float = 1e-6,
    maximum_condition_number: float = 1e10,
) -> VAR1StationaryMoments:
    """Stationary moments of z_t = c + Φ z_{t-1} + ε, ε~N(0,Q):
        μ* = (I − Φ)^{-1} c,   Σ* solves the discrete Lyapunov eq  Σ* = Φ Σ* Φᵀ + Q.
    Requires ρ(Φ) < 1; returns stable=False otherwise (no stationary distribution exists)."""
    from scipy.linalg import solve_discrete_lyapunov

    phi = np.asarray(coefficient, dtype=np.float64)
    c = np.asarray(intercept, dtype=np.float64)
    q = np.asarray(residual_covariance, dtype=np.float64)
    if phi.shape != (9, 9) or c.shape != (9,) or q.shape != (9, 9):
        raise ValueError("coefficient (9,9), intercept (9,), residual_covariance (9,9) required")
    rho, stable = compute_var1_stability_diagnostics(phi)
    if not stable:
        return VAR1StationaryMoments(None, None, None, rho, False)
    try:
        mean = np.linalg.solve(np.eye(9) - phi, c)
        sigma = solve_discrete_lyapunov(phi, q)
        sigma = 0.5 * (sigma + sigma.T) + ridge * np.eye(9)        # symmetrize + ridge
        condition_number = float(np.linalg.cond(sigma))
        if not np.isfinite(condition_number) or condition_number > maximum_condition_number:
            return VAR1StationaryMoments(mean, sigma, None, rho, False)
        inverse = np.linalg.inv(sigma)
    except (np.linalg.LinAlgError, ValueError):
        return VAR1StationaryMoments(None, None, None, rho, False)
    if not (np.isfinite(mean).all() and np.isfinite(sigma).all() and np.isfinite(inverse).all()):
        return VAR1StationaryMoments(None, None, None, rho, False)
    return VAR1StationaryMoments(mean, sigma, inverse, rho, True)


def fit_var1(
    standardized_history: np.ndarray,
    history_dates: Sequence[int] | np.ndarray,
    config: VAR1PlausibilityConfig,
) -> VAR1Fit:
    """Fit z_t = c + A z_(t-1) + eps_t using transparent OLS."""

    validate_var1_config(config)
    history = np.asarray(standardized_history, dtype=np.float64)
    dates = np.asarray(history_dates)
    if history.ndim != 2 or history.shape[1] != 9 or dates.shape != (history.shape[0],):
        raise ValueError("history and dates must have shapes (n, 9) and (n,)")
    if history.shape[0] < 12 or not np.isfinite(history).all():
        raise ValueError("history must contain sufficient finite observations")
    if int(dates[-1]) != config.fit_end_yyyymm or np.any(dates >= config.anchor_yyyymm):
        raise ValueError("fit history must end at fit_end_yyyymm and remain pre-anchor")
    lagged = history[:-1]
    targets = history[1:]
    design = np.column_stack([np.ones(lagged.shape[0]), lagged])
    parameters, _, rank, _ = np.linalg.lstsq(design, targets, rcond=None)
    if rank != design.shape[1]:
        raise ValueError("VAR(1) design matrix is rank deficient")
    intercept = parameters[0]
    coefficient = parameters[1:].T
    residuals = targets - design @ parameters
    residual_covariance = np.cov(residuals, rowvar=False, ddof=1)
    regularized, inverse, condition_number, covariance_stable = regularize_residual_covariance(
        residual_covariance,
        ridge=config.residual_covariance_regularization,
        maximum_condition_number=config.maximum_condition_number,
    )
    eigenvalue_max, dynamics_stable = compute_var1_stability_diagnostics(coefficient)
    residual_summary = {
        name: {
            "mean": float(np.mean(residuals[:, index])),
            "standard_deviation": float(np.std(residuals[:, index], ddof=1)),
            "minimum": float(np.min(residuals[:, index])),
            "maximum": float(np.max(residuals[:, index])),
        }
        for index, name in enumerate(config.macro_order)
    }
    diagnostics = VAR1FitDiagnostics(
        fit_start_yyyymm=int(dates[0]),
        fit_end_yyyymm=int(dates[-1]),
        history_row_count=history.shape[0],
        fit_observation_count=targets.shape[0],
        coefficient_shape=coefficient.shape,
        intercept_shape=intercept.shape,
        residual_shape=residuals.shape,
        residual_covariance_shape=residual_covariance.shape,
        regularized_covariance_shape=regularized.shape,
        regularized_covariance_condition_number=condition_number,
        residual_covariance_stable=covariance_stable,
        maximum_absolute_eigenvalue=eigenvalue_max,
        var1_dynamics_stable=dynamics_stable,
        residual_summary_by_macro=residual_summary,
    )
    return VAR1Fit(
        intercept=intercept,
        coefficient=coefficient,
        residuals=residuals,
        residual_covariance=residual_covariance,
        regularized_residual_covariance=regularized,
        inverse_regularized_residual_covariance=inverse,
        fit_diagnostics=diagnostics,
    )


def score_var1_innovation(
    *,
    label: str,
    target_yyyymm: int,
    lag_yyyymm: int,
    target_z: Sequence[float] | np.ndarray,
    lag_z: Sequence[float] | np.ndarray,
    fit: VAR1Fit,
    config: VAR1PlausibilityConfig,
    in_sample_fit_target: bool,
) -> VAR1StateDiagnostics:
    """Score one one-step innovation descriptively under a fitted VAR(1)."""

    target = np.asarray(target_z, dtype=np.float64)
    lag = np.asarray(lag_z, dtype=np.float64)
    if target.shape != (9,) or lag.shape != (9,) or not np.isfinite(target).all() or not np.isfinite(lag).all():
        raise ValueError("target_z and lag_z must be finite with shape (9,)")
    predicted = fit.intercept + fit.coefficient @ lag
    innovation = target - predicted
    absolute = np.abs(innovation)
    order = np.argsort(-absolute, kind="stable")[: config.top_innovation_count]
    top = tuple(
        {
            "macro": config.macro_order[index],
            "innovation": float(innovation[index]),
            "absolute_innovation": float(absolute[index]),
        }
        for index in order
    )
    distance: float | None = None
    tail: float | None = None
    inverse = fit.inverse_regularized_residual_covariance
    if fit.fit_diagnostics.residual_covariance_stable and inverse is not None:
        squared = float(innovation @ inverse @ innovation)
        if squared < -1e-10 or not np.isfinite(squared):
            raise ValueError("innovation Mahalanobis squared distance is invalid")
        squared = max(0.0, squared)
        distance = float(np.sqrt(squared))
        tail = float(chi2.sf(squared, df=9))
    return VAR1StateDiagnostics(
        label=label,
        target_yyyymm=int(target_yyyymm),
        lag_yyyymm=int(lag_yyyymm),
        target_z=tuple(target),
        lag_z=tuple(lag),
        predicted_z=tuple(predicted),
        innovation=tuple(innovation),
        innovation_l2_norm=float(np.linalg.norm(innovation)),
        innovation_max_absolute=float(np.max(absolute)),
        innovation_mahalanobis_distance=distance,
        innovation_chi_square_tail_probability=tail,
        top_innovation_macros=top,
        in_sample_fit_target=in_sample_fit_target,
    )
