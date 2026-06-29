# Convergence diagnostics: Gelman-Rubin R-hat, effective sample size, convergence gates, anchor chi-square plausibility.
from __future__ import annotations


"""Multi-chain MALA driver + convergence diagnostics (Gelman–Rubin R̂, ESS).

Promotes the synthetic one-step kernel (`mala_kernel.evaluate_mala_kernel_step`) and the capped
synthetic chain runner into a real multi-chain sampler with burn-in and convergence diagnostics. The
sampler reuses the *validated* kernel primitives (`mala_proposal_mean`, `gaussian_log_density`) so the
proposal/acceptance math is exactly the one-step kernel's, just iterated over chains.

Parameterization (reconciliation with thesis skeleton Eq 3.4): the kernel uses a single `step_size` s
with proposal `z' ~ N(z + ½ s² ∇log π(z), s² I)` — textbook MALA on the target π. The thesis's explicit
(τ, η) form `z' ~ N(z − η∇G̃, 2τη I)` with the target `π = exp(−G̃/τ)` is the SAME chain under
`s² = 2τη` and `β = 1/τ` folded into the log-target (the energy uses `log π = β·objective + log p₀`).
`eq34_step_size(tau, eta)` makes that mapping explicit.

Validation: `tests/test_mala_gaussian_closed_form.py` runs a Gaussian target and requires the empirical
chain mean/cov to match the analytic moments, R̂≈1; a wrong-coefficient variant must NOT match.
"""


from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np
from scipy import stats





# ── convergence diagnostics ─────────────────────────────────────────────────────

def gelman_rubin_rhat(chains: np.ndarray) -> np.ndarray:
    """Per-coordinate Gelman–Rubin R̂. `chains`: (n_chains, n_draws, d). Needs ≥2 chains, ≥2 draws."""
    chains = np.asarray(chains, dtype=np.float64)
    if chains.ndim != 3:
        raise ValueError("chains must be (n_chains, n_draws, d)")
    m, n, d = chains.shape
    if m < 2 or n < 2:
        return np.full(d, np.nan)
    chain_means = chains.mean(axis=1)                 # (m, d)
    chain_vars = chains.var(axis=1, ddof=1)           # (m, d)
    grand_mean = chain_means.mean(axis=0)             # (d,)
    between = n * ((chain_means - grand_mean) ** 2).sum(axis=0) / (m - 1)
    within = chain_vars.mean(axis=0)
    var_hat = (n - 1) / n * within + between / n
    return np.sqrt(np.where(within > 0, var_hat / within, np.nan))


def _autocorr(x: np.ndarray) -> np.ndarray:
    """Normalized autocorrelation of a 1-D series via FFT."""
    x = np.asarray(x, dtype=np.float64)
    x = x - x.mean()
    n = x.size
    f = np.fft.rfft(x, n=2 * n)
    acf = np.fft.irfft(f * np.conjugate(f))[:n].real
    return acf / acf[0] if acf[0] > 0 else np.zeros(n)


def effective_sample_size(chains: np.ndarray) -> np.ndarray:
    """Per-coordinate ESS using the chain-averaged autocorrelation with Geyer initial-positive
    truncation. `chains`: (n_chains, n_draws, d)."""
    chains = np.asarray(chains, dtype=np.float64)
    m, n, d = chains.shape
    ess = np.zeros(d)
    for j in range(d):
        rho = np.mean([_autocorr(chains[k, :, j]) for k in range(m)], axis=0)
        s = 0.0
        for lag in range(1, n):
            if rho[lag] <= 0:
                break
            s += rho[lag]
        ess[j] = m * n / (1.0 + 2.0 * s)
    return ess


def grouped_rhat_ess(post: np.ndarray, n_groups: int = 4) -> tuple[np.ndarray, np.ndarray]:
    """Pool an ensemble's walkers into `n_groups` chains, then classic R̂ + Geyer ESS.

    The Goodman–Weare ensemble runs many short, correlated walkers; reading each walker as its own
    "chain" makes R̂/ESS noisy. Here the (k, n, d) post-burn-in array is split into `n_groups` blocks
    of floor(k/n_groups) walkers and each block is CONCATENATED along the draw axis into one pooled
    chain of length block·n, so R̂ is a genuine between-group statistic on long chains (e.g. 40 walkers
    → 4 groups of 10 → 4 chains of 10·n draws). Returns (rhat, ess) per coordinate, identical in shape
    to the un-grouped diagnostics. Diagnostic-only: it never touches the sampler or its invariant law.
    """
    post = np.asarray(post, dtype=np.float64)
    if post.ndim != 3:
        raise ValueError("post must be (n_walkers, n_draws, d)")
    k, n, d = post.shape
    if n_groups < 2:
        raise ValueError("n_groups must be >= 2 for a between-group R̂")
    block = k // n_groups
    if block < 1:
        raise ValueError(f"need >= n_groups walkers (got {k} walkers, {n_groups} groups)")
    grouped = np.empty((n_groups, block * n, d), dtype=np.float64)
    for g in range(n_groups):
        chunk = post[g * block:(g + 1) * block]          # (block, n, d)
        grouped[g] = chunk.reshape(block * n, d)         # concatenate walkers along the draw axis
    return gelman_rubin_rhat(grouped), effective_sample_size(grouped)


# ── conditional (clamped) Gaussian moments — for clamped probing (skeleton §3.11) ──

def gaussian_conditional_moments(
    mean: np.ndarray,
    covariance: np.ndarray,
    clamp_idx: np.ndarray,
    clamp_values: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Schur-complement conditional of N(mean, covariance) given the clamped coords x_C = clamp_values.

    Returns (free_idx, μ_{F|C}, Σ_{F|C}) with
        μ_{F|C} = μ_F + Σ_FC Σ_CC⁻¹ (x_C − μ_C),   Σ_{F|C} = Σ_FF − Σ_FC Σ_CC⁻¹ Σ_CF.
    Under the VAR(1) stationary prior N(μ*, Σ*) this is the closed-form prior for a clamped probe.
    """
    mean = np.asarray(mean, dtype=np.float64)
    cov = np.asarray(covariance, dtype=np.float64)
    clamp_idx = np.asarray(clamp_idx, dtype=int)
    clamp_values = np.asarray(clamp_values, dtype=np.float64)
    d = mean.size
    free_idx = np.array([i for i in range(d) if i not in set(clamp_idx.tolist())], dtype=int)
    s_ff = cov[np.ix_(free_idx, free_idx)]
    s_fc = cov[np.ix_(free_idx, clamp_idx)]
    s_cc_inv = np.linalg.inv(cov[np.ix_(clamp_idx, clamp_idx)])
    cond_mean = mean[free_idx] + s_fc @ s_cc_inv @ (clamp_values - mean[clamp_idx])
    cond_cov = s_ff - s_fc @ s_cc_inv @ s_fc.T
    return free_idx, cond_mean, cond_cov


# ── multi-chain MALA driver ─────────────────────────────────────────────────────





# ── gradient-free affine-invariant ensemble sampler (Goodman–Weare) ──────────



# ── overdispersed chain starts ───────────────────────────────────────────────



# ── enforced convergence gates ───────────────────────────────────────────────

@dataclass(frozen=True)
class ConvergenceGates:
    """Publication-grade convergence thresholds (POE recipe). A run that fails any gate is not
    promoted to a thesis result."""

    rhat_max: float = 1.05
    ess_min: float = 100.0
    accept_low: float = 0.40
    accept_high: float = 0.70


# Relaxed acceptance band for the affine-invariant ensemble sampler: the Goodman–Weare stretch
# move's efficient regime is NOT the MALA 0.574 optimum, so the 0.40–0.70 MALA band does not apply.
# R̂/ESS gates are unchanged (convergence is convergence regardless of sampler).
ENSEMBLE_GATES = ConvergenceGates(accept_low=0.20, accept_high=0.50)


@dataclass(frozen=True)
class ConvergenceReport:
    passed: bool
    rhat_max: float
    ess_min: float
    accept_rate: float
    rhat_ok: bool
    ess_ok: bool
    accept_ok: bool
    failures: tuple[str, ...]


def check_convergence(
    result: MalaChainsResult, gates: ConvergenceGates = ConvergenceGates()
) -> ConvergenceReport:
    """Evaluate the convergence gates against a multi-chain run; report per-gate pass/fail."""
    rhat_max = float(np.nanmax(result.rhat))
    ess_min = float(np.nanmin(result.ess))
    accept = float(result.accept_rate)
    rhat_ok = rhat_max <= gates.rhat_max
    ess_ok = ess_min >= gates.ess_min
    accept_ok = gates.accept_low <= accept <= gates.accept_high
    failures: list[str] = []
    if not rhat_ok:
        failures.append(f"rhat_max {rhat_max:.3f} > {gates.rhat_max}")
    if not ess_ok:
        failures.append(f"ess_min {ess_min:.1f} < {gates.ess_min}")
    if not accept_ok:
        failures.append(
            f"accept_rate {accept:.3f} outside [{gates.accept_low}, {gates.accept_high}]"
        )
    return ConvergenceReport(
        passed=rhat_ok and ess_ok and accept_ok,
        rhat_max=rhat_max,
        ess_min=ess_min,
        accept_rate=accept,
        rhat_ok=rhat_ok,
        ess_ok=ess_ok,
        accept_ok=accept_ok,
        failures=tuple(failures),
    )


# ── η / β tuning protocol (POE recipe) ───────────────────────────────────────







# ── anchor plausibility under the prior (χ² tail) ────────────────────────────

@dataclass(frozen=True)
class AnchorPlausibilityReport:
    mahalanobis_sq: float
    dof: int
    tail_prob: float
    flagged: bool


def anchor_plausibility_chi2(
    point: np.ndarray,
    mean: np.ndarray,
    covariance: np.ndarray,
    flag_threshold: float = 1e-8,
) -> AnchorPlausibilityReport:
    """χ² upper-tail plausibility of `point` under the Gaussian prior N(mean, covariance).

    Mahalanobis² = (x−μ)ᵀ Σ⁻¹ (x−μ) is χ²_d under the prior; the upper-tail probability is a soft
    plausibility score (no hard gate). Flag if tail < `flag_threshold` (the old SC-WW pathology where
    the anchor sat in the prior's extreme tail). Reported alongside the convergence gates per run.
    """
    point = np.asarray(point, dtype=np.float64).ravel()
    mean = np.asarray(mean, dtype=np.float64).ravel()
    cov = np.asarray(covariance, dtype=np.float64)
    diff = point - mean
    maha_sq = float(diff @ np.linalg.solve(cov, diff))
    dof = point.size
    tail = float(stats.chi2.sf(maha_sq, dof))
    return AnchorPlausibilityReport(
        mahalanobis_sq=maha_sq,
        dof=dof,
        tail_prob=tail,
        flagged=tail < flag_threshold,
    )
