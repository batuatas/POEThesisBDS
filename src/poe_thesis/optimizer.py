# Optimizer: covariance conditioning (EWMA / Ledoit-Wolf nonlinear shrinkage) + robust mean-variance SOCP + differentiable CvxpyLayer (PAO layer).
from __future__ import annotations


"""Covariance estimation for the tactical robust MVO (single source of truth).

The portfolio universe always has N ≥ T (e.g. N=1000 firms vs a 60-month lookback),
so the *sample* covariance is singular and ill-conditioned and MVO error-maximizes
(Michaud 1989). This module provides shrinkage estimators that return a positive-
definite, well-conditioned Σ.

Default: **analytical nonlinear shrinkage** (Ledoit & Wolf 2020, *Analytical nonlinear
shrinkage of large-dimensional covariance matrices*, Annals of Statistics 48(5)). It
shrinks eigenvalue-by-eigenvalue via a kernel estimate of the sample spectral density
and its Hilbert transform, handles the p>n (N>T) case (lifts the zero eigenvalues to
positive values), and gives the best conditioning of the available estimators.

Fallbacks (selectable / used when nonlinear shrinkage is not applicable): linear
Ledoit-Wolf (2004) and the raw sample covariance.
"""


import logging

import numpy as np

log = logging.getLogger(__name__)

_EPS = 1e-8
_MIN_EFFECTIVE_N = 12  # nonlinear shrinkage needs a non-trivial effective sample size


def nonlinear_shrinkage(returns: np.ndarray, demean: bool = True) -> np.ndarray:
    """Analytical nonlinear shrinkage covariance (Ledoit & Wolf 2020).

    Ported from Michael Wolf's reference implementation (econ.uzh.ch wp264 / AoS 2020).

    Args:
        returns: (T, N) array — T observations (months) in rows, N variables (firms) in cols.
        demean: subtract the column means (and reduce the effective sample size by 1).

    Returns:
        (N, N) symmetric positive-definite shrunk covariance.

    Raises:
        ValueError: if the effective sample size is too small or the kept spectrum is
            singular (the caller `estimate_covariance` catches this and falls back).
    """
    data = np.asarray(returns, dtype=np.float64)
    if data.ndim != 2:
        raise ValueError(f"returns must be 2D (T, N); got {data.shape}")
    T, p = data.shape
    k = 0
    if demean:
        data = data - data.mean(axis=0)
        k = 1
    n = T - k  # effective sample size
    if n < _MIN_EFFECTIVE_N:
        raise ValueError(f"effective sample size {n} < {_MIN_EFFECTIVE_N}")

    sample_cov = (data.T @ data) / n
    # ascending eigenvalues / eigenvectors of the (symmetric) sample covariance
    lam, u = np.linalg.eigh(sample_cov)
    lam = lam[max(0, p - n):]  # drop the p-n structural zeros when p > n
    if np.any(lam / lam.sum() < _EPS):
        raise ValueError("kept spectrum is numerically singular")

    L = np.tile(lam, (min(p, n), 1)).T
    h = n ** (-1.0 / 3.0)  # bandwidth (Eq. 4.9)
    H = h * L.T
    x = (L - L.T) / H
    ftilde = (3.0 / 4.0 / np.sqrt(5)) * np.mean(np.maximum(1.0 - x ** 2 / 5.0, 0.0) / H, axis=1)
    # Hilbert transform of the spectral density (Eq. 4.7 / 4.8)
    with np.errstate(divide="ignore", invalid="ignore"):
        Hftemp = (-3.0 / 10.0 / np.pi) * x + (3.0 / 4.0 / np.sqrt(5) / np.pi) * (
            1.0 - x ** 2 / 5.0
        ) * np.log(np.abs((np.sqrt(5) - x) / (np.sqrt(5) + x)))
    edge = np.abs(x) == np.sqrt(5)
    Hftemp[edge] = (-3.0 / 10.0 / np.pi) * x[edge]
    Hftilde = np.mean(Hftemp / H, axis=1)

    if p <= n:
        dtilde = lam / (
            (np.pi * (p / n) * lam * ftilde) ** 2
            + (1.0 - (p / n) - np.pi * (p / n) * lam * Hftilde) ** 2
        )
    else:
        Hftilde0 = (1.0 / np.pi) * (
            3.0 / 10.0 / h ** 2
            + 3.0 / 4.0 / np.sqrt(5) / h * (1.0 - 1.0 / 5.0 / h ** 2)
            * np.log((1.0 + np.sqrt(5) * h) / (1.0 - np.sqrt(5) * h))
        ) * np.mean(1.0 / lam)
        dtilde0 = 1.0 / (np.pi * (p - n) / n * Hftilde0)
        dtilde1 = lam / (np.pi ** 2 * lam ** 2 * (ftilde ** 2 + Hftilde ** 2))
        dtilde = np.concatenate([dtilde0 * np.ones(p - n), dtilde1])

    sigma = (u * dtilde) @ u.T
    return 0.5 * (sigma + sigma.T)  # symmetrize


def _linear_lw(returns: np.ndarray) -> np.ndarray:
    from sklearn.covariance import ledoit_wolf

    cov, _ = ledoit_wolf(np.asarray(returns, dtype=np.float64), assume_centered=False)
    return cov


def _sample(returns: np.ndarray) -> np.ndarray:
    return np.cov(np.asarray(returns, dtype=np.float64), rowvar=False)


# POE4Nisan EWMA recipe defaults (configs/pto_config.py: LAM=0.94, SHRINK=0.10, RIDGE=1e-6)
EWMA_LAMBDA = 0.94
EWMA_SHRINK = 0.10
EWMA_RIDGE = 1e-6


def ewma_covariance(
    returns: np.ndarray,
    *,
    lam: float = EWMA_LAMBDA,
    shrink: float = EWMA_SHRINK,
    ridge: float = EWMA_RIDGE,
    demean: bool = True,
) -> np.ndarray:
    """Exponentially-weighted-moving-average covariance (POE4Nisan recipe).

    Weights decay geometrically with `lam` (most recent month weight 1, normalized), then the
    estimate is shrunk toward its diagonal by `shrink`, ridged by `ridge`, and eigenvalue-floored
    to guarantee positive-definiteness. Mirrors POE's `src/optimization/risk.py`.
    """
    data = np.asarray(returns, dtype=np.float64)
    if data.ndim != 2:
        raise ValueError(f"returns must be 2D (T, N); got {data.shape}")
    t, n = data.shape
    if t < 2 or n < 1:
        raise ValueError(f"need at least 2 observations and 1 asset; got {data.shape}")

    weights = lam ** np.arange(t - 1, -1, -1)  # oldest→smallest, newest→1
    weights = weights / weights.sum()
    if demean:
        mu = weights @ data
        data = data - mu
    weighted = data * np.sqrt(weights)[:, None]
    sigma = weighted.T @ weighted
    sigma = (1.0 - shrink) * sigma + shrink * np.diag(np.diag(sigma))
    sigma = 0.5 * (sigma + sigma.T) + ridge * np.eye(n)

    # eigenvalue floor → PD (POE PSD-projects; ridge usually suffices, this makes it certain)
    evals, evecs = np.linalg.eigh(sigma)
    floor = max(ridge, 1e-12)
    if evals.min() < floor:
        evals = np.clip(evals, floor, None)
        sigma = (evecs * evals) @ evecs.T
        sigma = 0.5 * (sigma + sigma.T)
    return sigma


def estimate_covariance(
    returns: np.ndarray,
    method: str = "nl_shrinkage",
    *,
    demean: bool = True,
) -> np.ndarray:
    """Estimate a PD covariance from a (T, N) return matrix.

    method ∈ {"nl_shrinkage" (default, Ledoit-Wolf 2020), "lw_linear" (Ledoit-Wolf 2004),
    "ewma" (POE4Nisan EWMA(0.94)+diag-shrink), "sample"}. nl_shrinkage falls back to linear LW
    when its effective-sample-size / non-singularity preconditions are not met.
    """
    returns = np.asarray(returns, dtype=np.float64)
    if method == "nl_shrinkage":
        try:
            return nonlinear_shrinkage(returns, demean=demean)
        except (ValueError, np.linalg.LinAlgError) as exc:
            log.warning("nonlinear shrinkage unavailable (%s); falling back to linear LW", exc)
            return _linear_lw(returns)
    if method == "lw_linear":
        return _linear_lw(returns)
    if method == "ewma":
        return ewma_covariance(returns, demean=demean)
    if method == "sample":
        return _sample(returns)
    raise ValueError(f"unknown covariance method: {method!r}")


def cholesky_psd(sigma: np.ndarray, jitter: float = 1e-10) -> np.ndarray:
    """Upper-triangular Cholesky factor U with Σ = Uᵀ U (matches the legacy convention).

    Adds increasing diagonal jitter on failure; final fallback is the diagonal sqrt.
    """
    sigma = np.asarray(sigma, dtype=np.float64)
    n = sigma.shape[0]
    j = jitter
    for _ in range(6):
        try:
            return np.linalg.cholesky(sigma + j * np.eye(n)).T
        except np.linalg.LinAlgError:
            j = max(j * 10.0, 1e-12)
    diag_var = np.clip(np.diag(sigma), 1e-12, None)
    return np.diag(np.sqrt(diag_var))


import dataclasses
from typing import Optional

import cvxpy as cp


log = logging.getLogger(__name__)


@dataclasses.dataclass
class MeanVarianceResult:
    """Diagnostic outputs from mean-variance optimization."""
    weights: np.ndarray
    expected_return: float
    portfolio_variance: float
    solver_status: str
    solver_time: Optional[float] = None


def equal_weight_allocation(n_assets: int) -> np.ndarray:
    """Compute equal-weight allocation for N assets.

    Verifies that the allocation is long-only (nonnegative, finite, and sums to one).

    Args:
        n_assets: Number of assets.

    Returns:
        A 1D float64 numpy array of length n_assets.
    """
    if not isinstance(n_assets, int) or n_assets <= 0:
        raise ValueError(f"Number of assets must be a positive integer, got {n_assets}")

    weights = np.ones(n_assets, dtype=np.float64) / n_assets

    # Validation
    if not np.isfinite(weights).all():
        raise ValueError("Weights contain non-finite or missing values.")
    if (weights < 0.0).any():
        raise ValueError("Weights contain negative allocations (long-only violation).")
    if not np.isclose(np.sum(weights), 1.0, atol=1e-12):
        raise ValueError("Weights do not sum to one.")

    return weights


def validate_expected_returns(mu: np.ndarray) -> np.ndarray:
    """Validate expected returns vector."""
    if not isinstance(mu, np.ndarray):
        mu = np.array(mu, dtype=np.float64)
    if mu.ndim != 1:
        raise ValueError(f"Expected returns must be 1D, got shape {mu.shape}")
    if not np.isfinite(mu).all():
        raise ValueError("Expected returns contain non-finite values.")
    return mu.astype(np.float64)


def validate_covariance_matrix(sigma: np.ndarray, n_assets: int) -> np.ndarray:
    """Validate covariance matrix."""
    if not isinstance(sigma, np.ndarray):
        sigma = np.array(sigma, dtype=np.float64)
    if sigma.shape != (n_assets, n_assets):
        raise ValueError(f"Covariance matrix shape {sigma.shape} does not match ({n_assets}, {n_assets})")
    if not np.isfinite(sigma).all():
        raise ValueError("Covariance matrix contains non-finite values.")
    if not np.allclose(sigma, sigma.T, atol=1e-8):
        raise ValueError("Covariance matrix must be symmetric.")
    return sigma.astype(np.float64)


def identity_covariance(n_assets: int) -> np.ndarray:
    """Return an identity covariance matrix for structural smoke tests."""
    if not isinstance(n_assets, int) or n_assets <= 0:
        raise ValueError("n_assets must be positive integer.")
    return np.eye(n_assets, dtype=np.float64)


def diagonal_covariance(variances: np.ndarray) -> np.ndarray:
    """Return a diagonal covariance matrix from a variance vector."""
    if not isinstance(variances, np.ndarray):
        variances = np.array(variances, dtype=np.float64)
    if variances.ndim != 1:
        raise ValueError("variances must be a 1D array.")
    if not np.isfinite(variances).all() or (variances < 0).any():
        raise ValueError("variances must be finite and non-negative.")
    return np.diag(variances).astype(np.float64)


def solve_long_only_mean_variance(
    expected_returns: np.ndarray,
    covariance_matrix: np.ndarray,
    gamma: float = 1.0
) -> MeanVarianceResult:
    """Solve long-only mean-variance optimization using cvxpy.
    
    maximize:  mu^T w - gamma * w^T Sigma w
    subject to: sum(w) == 1
                w >= 0
    """
    mu = validate_expected_returns(expected_returns)
    n_assets = len(mu)
    sigma = validate_covariance_matrix(covariance_matrix, n_assets)
    
    if not np.isfinite(gamma) or gamma < 0:
        raise ValueError("Gamma must be non-negative and finite.")

    w = cp.Variable(n_assets)
    
    # Objective
    ret = mu @ w
    risk = cp.quad_form(w, sigma)
    prob = cp.Problem(cp.Maximize(ret - gamma * risk), [cp.sum(w) == 1, w >= 0])
    
    # Solve
    prob.solve(solver=cp.OSQP)
    
    if w.value is None:
        raise RuntimeError(f"Solver failed to find a solution. Status: {prob.status}")

    w_val = w.value
    # Cleanup small numerical noise
    w_val[w_val < 1e-8] = 0.0
    w_val = w_val / np.sum(w_val)
    
    return MeanVarianceResult(
        weights=w_val,
        expected_return=float(ret.value),
        portfolio_variance=float(risk.value),
        solver_status=prob.status,
        solver_time=prob.solver_stats.solve_time if hasattr(prob, "solver_stats") else None
    )


# ──────────────────────────────────────────────────────────────────────────────
#  Unified robust MVO — single source of truth for PTO and E2E
#
#  Objective:  max μ'w − κ·√(w'Ω w) − (λ/2)·w'Σ w     s.t.  Σw = 1, w ≥ 0
#  written as: min −μ'w + κ·‖A w‖₂ + (λ/2)·‖U w‖₂²     with A = Ω^½, U = chol(Σ).
#  This is the Ceria & Stubbs (2006) / Yin-Perchet-Soupé (2021) robust MVO; the
#  numpy solver here and the cvxpylayers layer in models/e2e.py share this exact
#  objective by construction (build_omega_half feeds both).
# ──────────────────────────────────────────────────────────────────────────────

OMEGA_MODES = ("diag_sigma", "identity", "sigma_over_T")


def build_omega_half(
    sigma: np.ndarray,
    mode: str = "diag_sigma",
    n_obs: Optional[int] = None,
) -> np.ndarray:
    """Return the matrix A with ‖A w‖₂ = √(w'Ω w) for the robust term.

    Modes (Ω = the expected-return estimation-error uncertainty matrix):
      - "diag_sigma": Ω = diag(Σ)  → A = diag(√σ_ii)   [Yin-Perchet-Soupé, default]
      - "identity":   Ω = I         → A = I
      - "sigma_over_T": Ω = Σ/T     → A = chol(Σ)/√T    [Ceria-Stubbs sample-mean error]
    """
    sigma = np.asarray(sigma, dtype=np.float64)
    n = sigma.shape[0]
    if mode == "diag_sigma":
        return np.diag(np.sqrt(np.clip(np.diag(sigma), 0.0, None)))
    if mode == "identity":
        return np.eye(n, dtype=np.float64)
    if mode == "sigma_over_T":
        if not n_obs or n_obs <= 0:
            raise ValueError("omega_mode='sigma_over_T' requires n_obs > 0")
        return cholesky_psd(sigma) / np.sqrt(float(n_obs))
    raise ValueError(f"unknown omega_mode: {mode!r} (expected one of {OMEGA_MODES})")


def solve_robust_mvo(
    mu: np.ndarray,
    sigma: np.ndarray,
    *,
    kappa: float,
    lambd: float,
    omega_mode: str = "diag_sigma",
    n_obs: Optional[int] = None,
) -> np.ndarray:
    """Solve the unified robust MVO (SOCP), long-only, sum-to-one.

    Returns a float64 weight vector. ECOS → SCS fallback; equal-weight on failure.
    κ=0 reduces this to the nominal long-only MVO.
    """
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    n = len(mu)
    U = cholesky_psd(sigma)
    A = build_omega_half(sigma, omega_mode, n_obs)

    w = cp.Variable(n, nonneg=True)
    risk = cp.sum_squares(U @ w)            # = w'Σw
    robust = cp.norm(A @ w, 2)              # = √(w'Ωw)
    obj = cp.Minimize(-mu @ w + (float(lambd) / 2.0) * risk + float(kappa) * robust)
    prob = cp.Problem(obj, [cp.sum(w) == 1])

    attempts = []
    installed = cp.installed_solvers()
    if "ECOS" in installed:
        attempts.append({"solver": cp.ECOS})
    if "SCS" in installed:
        attempts.append({"solver": cp.SCS, "max_iters": 8000, "eps": 1e-4, "verbose": False})
    for sargs in attempts:
        try:
            prob.solve(**sargs)
            if prob.status in ("optimal", "optimal_inaccurate") and w.value is not None:
                weights = np.clip(w.value, 0.0, None)
                total = weights.sum()
                if total > 1e-10:
                    return (weights / total).astype(np.float64)
        except Exception as exc:  # noqa: BLE001
            log.debug("robust MVO solver %s failed: %s", sargs.get("solver", "?"), exc)
    log.warning("all robust MVO solvers failed; equal-weight for %d assets", n)
    return np.ones(n, dtype=np.float64) / n


class RobustMVOSolver:
    """Pre-canonicalized robust MVO for a FIXED covariance; only μ varies (DPP `cp.Parameter`).

    The scenario reward re-solves the SAME SOCP at thousands of candidate macro states z that share one
    anchor Σ (macro-independent). Rebuilding the cvxpy problem each call re-canonicalizes the SOCP and
    re-queries ``cp.installed_solvers()`` — the dominant per-eval cost. This builds Σ^½, chol(Σ) and the
    canonical form ONCE; each :meth:`solve` only updates μ and re-solves (warm-started). With ECOS the
    weights match :func:`solve_robust_mvo` exactly (Δw = 0); the SCS fallback and equal-weight degeneracy
    handling mirror the function so behavior is identical.
    """

    def __init__(
        self,
        sigma: np.ndarray,
        *,
        kappa: float,
        lambd: float,
        omega_mode: str = "diag_sigma",
        n_obs: Optional[int] = None,
    ) -> None:
        sigma = np.asarray(sigma, dtype=np.float64)
        self.n = int(sigma.shape[0])
        U = cholesky_psd(sigma)
        A = build_omega_half(sigma, omega_mode, n_obs)
        self._w = cp.Variable(self.n, nonneg=True)
        self._mu = cp.Parameter(self.n)
        risk = cp.sum_squares(U @ self._w)          # = w'Σw
        robust = cp.norm(A @ self._w, 2)            # = √(w'Ωw)
        obj = cp.Minimize(-self._mu @ self._w + (float(lambd) / 2.0) * risk + float(kappa) * robust)
        self._prob = cp.Problem(obj, [cp.sum(self._w) == 1])
        installed = cp.installed_solvers()
        self._attempts = []
        if "ECOS" in installed:
            self._attempts.append({"solver": cp.ECOS, "warm_start": True})
        if "SCS" in installed:
            self._attempts.append(
                {"solver": cp.SCS, "warm_start": True, "max_iters": 8000, "eps": 1e-4, "verbose": False}
            )

    def solve(self, mu: np.ndarray) -> np.ndarray:
        """Return long-only sum-to-one weights for expected returns ``mu`` (length n)."""
        self._mu.value = np.asarray(mu, dtype=np.float64)
        for sargs in self._attempts:
            try:
                self._prob.solve(**sargs)
                if self._prob.status in ("optimal", "optimal_inaccurate") and self._w.value is not None:
                    weights = np.clip(self._w.value, 0.0, None)
                    total = weights.sum()
                    if total > 1e-10:
                        return (weights / total).astype(np.float64)
            except Exception as exc:  # noqa: BLE001
                log.debug("RobustMVOSolver %s failed: %s", sargs.get("solver", "?"), exc)
        log.warning("RobustMVOSolver: all solvers failed; equal-weight for %d assets", self.n)
        return np.ones(self.n, dtype=np.float64) / self.n


import torch
import torch.nn as nn

try:
    import cvxpy as cp
    from cvxpylayers.torch import CvxpyLayer
    _HAVE_CVXPYLAYERS = True
    _CVXPY_IMPORT_ERROR: Exception | None = None
except Exception as _e:
    cp = None  # type: ignore[assignment]
    CvxpyLayer = None  # type: ignore[assignment]
    _HAVE_CVXPYLAYERS = False
    _CVXPY_IMPORT_ERROR = _e


class DifferentiableRobustMVOLayer(nn.Module):
    """Differentiable robust MVO via cvxpylayers.

    Objective: max mu'w - kappa||Aw||_2 - (lambd/2) w'Sigma w
    subject to: sum(w) = 1, w >= 0

    Args:
        n_assets: number of assets (portfolio dimension)
        lambd: quadratic risk penalty weight
        kappa: robust uncertainty penalty weight (0 → plain MVO)
    """

    def __init__(self, n_assets: int, lambd: float, kappa: float):
        super().__init__()
        if not _HAVE_CVXPYLAYERS:
            raise ImportError(
                f"cvxpy/cvxpylayers required for DifferentiableRobustMVOLayer: {_CVXPY_IMPORT_ERROR}"
            )

        n = int(n_assets)
        w = cp.Variable(n, nonneg=True)
        mu = cp.Parameter(n)
        U = cp.Parameter((n, n))
        A = cp.Parameter((n, n))

        risk = cp.sum_squares(U @ w)
        obj = cp.Minimize(
            -mu @ w + float(kappa) * cp.norm(A @ w, 2) + (float(lambd) / 2.0) * risk
        )
        cons = [cp.sum(w) == 1]
        prob = cp.Problem(obj, cons)
        self.layer = CvxpyLayer(prob, parameters=[mu, U, A], variables=[w])
        self.n_assets = n

    def forward(self, mu: torch.Tensor, U: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
        # cvxpylayers selects the cone solver via the `solve_method` key (NOT `solver`).
        # ECOS gives accurate forward + clean implicit gradients for this SOCP; SCS is
        # the diffcp default fallback.
        solver_try = [
            {"solve_method": "ECOS"},
            {"solve_method": "SCS", "eps": 1e-6, "max_iters": 10000},
        ]

        last_err: Exception | None = None
        for sargs in solver_try:
            try:
                (w,) = self.layer(mu.double(), U.double(), A.double(), solver_args=sargs)
                w = w.float()
                w = torch.clamp(w, min=0.0)
                return w / (w.sum() + 1e-12)
            except Exception as exc:  # noqa: BLE001
                last_err = exc

        print(f"[DifferentiableRobustMVOLayer solver fail] {last_err}")
        return torch.ones(self.n_assets, device=mu.device) / self.n_assets
