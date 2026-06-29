# Samplers: preconditioned MALA + gradient-free affine-invariant ensemble, plus the energy/log-target and VAR(1) log-prior gradient.
from __future__ import annotations
from poe_thesis.diagnostics import effective_sample_size, gelman_rubin_rhat


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

LogTarget = Callable[[np.ndarray], float]
GradLogTarget = Callable[[np.ndarray], np.ndarray]


def eq34_step_size(tau: float, eta: float) -> float:
    """Skeleton Eq 3.4 ↔ code mapping: the kernel `step_size` s satisfies s² = 2τη."""
    if tau <= 0 or eta <= 0:
        raise ValueError("tau and eta must be positive")
    return float(np.sqrt(2.0 * tau * eta))


# ── convergence diagnostics ─────────────────────────────────────────────────────









# ── conditional (clamped) Gaussian moments — for clamped probing (skeleton §3.11) ──



# ── multi-chain MALA driver ─────────────────────────────────────────────────────

@dataclass
class MalaChainsResult:
    samples: np.ndarray        # (n_chains, n_draws_post_burnin, d)
    rhat: np.ndarray           # (d,)
    ess: np.ndarray            # (d,)
    accept_rate: float
    step_size: float
    n_chains: int
    n_steps: int
    burn_in: int
    trajectory: Optional[np.ndarray] = None  # (n_chains, n_steps, d) incl. warmup, if keep_warmup
    plateau_fraction: Optional[float] = None  # ensemble only: share of in-box proposals with Δlog_target≈0


def run_mala_chains(
    log_target: LogTarget,
    grad_log_target: GradLogTarget,
    init: np.ndarray,
    *,
    n_steps: int,
    burn_in: int,
    step_size: float,
    seed: int,
    lower: Optional[np.ndarray] = None,
    upper: Optional[np.ndarray] = None,
    density_step_scale: float = 1.0,
    clamp_idx: Optional[np.ndarray] = None,
    clamp_values: Optional[np.ndarray] = None,
    precond: Optional[np.ndarray] = None,
    keep_warmup: bool = False,
) -> MalaChainsResult:
    """Run one MALA chain per row of `init` on a target given by `log_target`/`grad_log_target`.

    Dimension-agnostic: the state dimension `d` is read from `init` (9 for the tactical macro state,
    19 for the strategical one) — no fixed-dimension assumption.

    `precond` is an optional PD mass matrix M (default: identity ⇒ isotropic MALA). With a preconditioner
    the proposal is `z' ~ N(z + ½ s² M ∇log π(z), s² M)` (draw `z + ½ s² M g + s·chol(M)·ξ`), and the MH
    correction uses the M⁻¹-weighted proposal density — detailed balance holds for any PD M. Setting
    `M = Σ*` (the VAR(1) stationary covariance) is the POE-proven mixing fix on anisotropic targets.
    The proposal-density NORMALIZERS (−d·log s − ½·log|M| − …) cancel in the forward/reverse ratio, so
    only the M⁻¹-weighted quadratic is evaluated here.

    Out-of-box proposals (outside [lower, upper], if given) are REJECTED — never reflected/projected —
    so the invariant law is the target restricted to the box (skeleton §3.7 boundary convention).

    `density_step_scale` MUST be 1.0 for a correct sampler; it exists only so the closed-form test can
    inject a WRONG proposal-density coefficient (the MH correction then uses the wrong Gaussian width)
    and verify the chain no longer matches the target — the "wrong-coefficient must fail" guard.
    """
    init = np.atleast_2d(np.asarray(init, dtype=np.float64))
    m, d = init.shape
    if n_steps <= burn_in:
        raise ValueError("n_steps must exceed burn_in")
    if step_size <= 0:
        raise ValueError("step_size must be positive")
    rng = np.random.default_rng(seed)
    s_corr = step_size * float(density_step_scale)
    draws = np.empty((m, n_steps, d), dtype=np.float64)
    n_accept = 0

    # Preconditioner setup: M = mass matrix (None ⇒ isotropic identity). chol(M) shapes the noise;
    # M⁻¹ weights the proposal density.
    clamp = clamp_idx is not None
    if precond is not None:
        precond_m = np.asarray(precond, dtype=np.float64)
        if precond_m.shape != (d, d):
            raise ValueError(f"precond must be ({d}, {d}), got {precond_m.shape}")
        if clamp:
            raise NotImplementedError(
                "clamped probing with a preconditioner needs the Schur conditional metric; build the "
                "conditional Σ* with gaussian_conditional_moments and pass that as `precond` instead"
            )
        chol_m = np.linalg.cholesky(precond_m)   # raises LinAlgError if not PD
        m_inv = np.linalg.inv(precond_m)
    else:
        precond_m = None
        chol_m = None
        m_inv = None

    # Clamped probing (skeleton §3.11): the clamped coords are pinned at clamp_values and excluded from
    # the drift/noise, so the chain samples the free coords from the conditional p₀(m_F | m_C) restricted
    # to the slice — no explicit conditional needed in the energy (the pinned coords cancel in the MH ratio).
    if clamp:
        clamp_idx = np.asarray(clamp_idx, dtype=int)
        clamp_values = np.asarray(clamp_values, dtype=np.float64)
        init[:, clamp_idx] = clamp_values

    def _in_box(z: np.ndarray) -> bool:
        if lower is not None and np.any(z < lower):
            return False
        if upper is not None and np.any(z > upper):
            return False
        return True

    def _drift(z: np.ndarray, g: np.ndarray) -> np.ndarray:
        if precond_m is None:
            return z + 0.5 * step_size**2 * g
        return z + 0.5 * step_size**2 * (precond_m @ g)

    def _log_q(z_to: np.ndarray, mean_: np.ndarray) -> float:
        # M⁻¹-weighted proposal-density quadratic (normalizers cancel in the MH ratio).
        diff = z_to - mean_
        if m_inv is None:
            return -0.5 * float(diff @ diff) / s_corr**2
        return -0.5 * float(diff @ m_inv @ diff) / s_corr**2

    for c in range(m):
        z = init[c].copy()
        lt = float(log_target(z))
        g = np.asarray(grad_log_target(z), dtype=np.float64)
        if clamp:
            g[clamp_idx] = 0.0
        for t in range(n_steps):
            mean = _drift(z, g)
            if chol_m is None:
                prop = mean + step_size * rng.standard_normal(d)
            else:
                prop = mean + step_size * (chol_m @ rng.standard_normal(d))
            if clamp:
                prop[clamp_idx] = clamp_values
            if _in_box(prop):
                lt_p = float(log_target(prop))
                g_p = np.asarray(grad_log_target(prop), dtype=np.float64)
                if clamp:
                    g_p[clamp_idx] = 0.0
                fwd = _log_q(prop, mean)
                rev = _log_q(z, _drift(prop, g_p))
                log_alpha = lt_p - lt + rev - fwd
                if np.log(rng.uniform()) < min(0.0, log_alpha):
                    z, lt, g = prop, lt_p, g_p
                    n_accept += 1
            draws[c, t] = z

    post = draws[:, burn_in:, :]
    return MalaChainsResult(
        samples=post,
        rhat=gelman_rubin_rhat(post),
        ess=effective_sample_size(post),
        accept_rate=n_accept / (m * n_steps),
        step_size=step_size,
        n_chains=m,
        n_steps=n_steps,
        burn_in=burn_in,
        trajectory=draws if keep_warmup else None,
    )


# ── gradient-free affine-invariant ensemble sampler (Goodman–Weare) ──────────

def run_ensemble_chains(
    log_target: LogTarget,
    init: np.ndarray,
    *,
    n_steps: int,
    burn_in: int,
    seed: int,
    a: float = 2.0,
    lower: Optional[np.ndarray] = None,
    upper: Optional[np.ndarray] = None,
    keep_warmup: bool = False,
    step_size: float = float("nan"),
    jacobian_scale: float = 1.0,
) -> MalaChainsResult:
    """Affine-invariant ensemble sampler (Goodman & Weare 2010) — **gradient-free**.

    The model-agnostic sampler backend: it needs only `log_target(z)` (no ∇), so it can scenario-
    generate a NON-differentiable predictor (e.g. the gradient-boosted strategical model) that exact
    MALA cannot touch. Adequate at the 10–20-dim macro state (the lit review refuted the "fails in
    high-dim" claim for n<50); reuses the same R̂/ESS/box machinery so it drops into `run_scenario`
    interchangeably with `run_mala_chains` and returns the same `MalaChainsResult`.

    Walkers ↔ chains: each row of `init` is one walker, mapped to a "chain" so `gelman_rubin_rhat` /
    `effective_sample_size` apply unchanged. The complementary ensemble must span the state, so use
    plenty of walkers (rule of thumb K ≳ 2d; the d=21 closed-form test uses K=60). The ensemble is
    split into two halves (Goodman–Weare split-walker update): half A is updated using only the
    (fixed) walkers of half B, then B is updated using the now-updated A — this keeps the move a valid
    Markov transition that preserves the target even with the ensemble's shared state.

    Stretch move: for walker `X_i`, pick a complement walker `X_j`, draw the stretch factor
    `Z ~ g(z) ∝ 1/√z` on `[1/a, a]` (`Z = ((a−1)U + 1)² / a`, `U~Uniform(0,1)`), propose
    `Y = X_j + Z·(X_i − X_j)`, accept with `log α = (d−1)·log Z + log π(Y) − log π(X_i)`. The
    `(d−1)·log Z` Jacobian is what makes the affine-invariant move correct — it accounts for the
    volume change of the stretch (cf. the MALA proposal-density normalizer).

    Out-of-box proposals (outside [lower, upper]) are REJECTED (never reflected), matching
    `run_mala_chains` so the invariant law is the target restricted to the box.

    `step_size` is accepted and IGNORED (there is no step size in a stretch move); it exists only so
    the `run_scenario` switch can pass sampler args uniformly. `jacobian_scale` MUST be 1.0 for a
    correct sampler — it exists solely so the closed-form test can drop the `(d−1)·log Z` Jacobian
    (`jacobian_scale=0.0`) and verify moment recovery breaks (the "wrong-target must fail" guard).
    """
    init = np.atleast_2d(np.asarray(init, dtype=np.float64))
    k, d = init.shape
    if n_steps <= burn_in:
        raise ValueError("n_steps must exceed burn_in")
    if k < 4:
        raise ValueError("ensemble sampler needs >= 4 walkers (split into two non-trivial halves)")
    if a <= 1.0:
        raise ValueError("stretch parameter a must be > 1")
    rng = np.random.default_rng(seed)

    half = k // 2
    halves = (np.arange(0, half), np.arange(half, k))

    def _in_box(z: np.ndarray) -> bool:
        if lower is not None and np.any(z < lower):
            return False
        if upper is not None and np.any(z > upper):
            return False
        return True

    walkers = init.copy()
    lt = np.array([float(log_target(walkers[i])) for i in range(k)], dtype=np.float64)
    draws = np.empty((k, n_steps, d), dtype=np.float64)
    n_accept = 0
    n_inbox = 0
    n_plateau = 0  # in-box proposals whose log_target equals the walker's (a degenerate level-set move)

    for t in range(n_steps):
        for s in (0, 1):
            update_idx = halves[s]
            comp_idx = halves[1 - s]
            for i in update_idx:
                j = comp_idx[rng.integers(comp_idx.size)]
                u = rng.uniform()
                z_stretch = ((a - 1.0) * u + 1.0) ** 2 / a
                prop = walkers[j] + z_stretch * (walkers[i] - walkers[j])
                if _in_box(prop):
                    lt_p = float(log_target(prop))
                    n_inbox += 1
                    if abs(lt_p - lt[i]) <= 1e-12 * (1.0 + abs(lt[i])):
                        n_plateau += 1
                    log_alpha = (
                        jacobian_scale * (d - 1) * np.log(z_stretch) + lt_p - lt[i]
                    )
                    if np.log(rng.uniform()) < min(0.0, log_alpha):
                        walkers[i] = prop
                        lt[i] = lt_p
                        n_accept += 1
        draws[:, t] = walkers

    post = draws[:, burn_in:, :]
    return MalaChainsResult(
        samples=post,
        rhat=gelman_rubin_rhat(post),
        ess=effective_sample_size(post),
        accept_rate=n_accept / (k * n_steps),
        step_size=float(step_size),
        n_chains=k,
        n_steps=n_steps,
        burn_in=burn_in,
        trajectory=draws if keep_warmup else None,
        plateau_fraction=(n_plateau / n_inbox) if n_inbox else None,
    )


# ── overdispersed chain starts ───────────────────────────────────────────────

def make_overdispersed_starts(
    anchor: np.ndarray, n_chains: int, scale: float, seed: int
) -> np.ndarray:
    """`n_chains` symmetric overdispersed start states `anchor + scale·(2U−1)` per coordinate.

    Returns shape (n_chains, d). The draw is symmetric about the anchor (each coordinate uniform on
    [anchor−scale, anchor+scale]) so the chains genuinely bracket the anchor — the precondition for R̂
    to be a valid between-chain diagnostic. The POE failure mode was near-identical one-sided starts
    (`random_start_scale=0.005`, half-line draw), which made R̂ optimistic; this fixes it.
    Units are the sampler's working units (σ-units in the standardized macro space).
    """
    anchor = np.asarray(anchor, dtype=np.float64).ravel()
    if n_chains < 1:
        raise ValueError("n_chains must be >= 1")
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("scale must be finite and positive")
    rng = np.random.default_rng(seed)
    u = rng.uniform(size=(n_chains, anchor.size))
    return anchor[None, :] + scale * (2.0 * u - 1.0)


# ── enforced convergence gates ───────────────────────────────────────────────



# Relaxed acceptance band for the affine-invariant ensemble sampler: the Goodman–Weare stretch
# move's efficient regime is NOT the MALA 0.574 optimum, so the 0.40–0.70 MALA band does not apply.
# R̂/ESS gates are unchanged (convergence is convergence regardless of sampler).






# ── η / β tuning protocol (POE recipe) ───────────────────────────────────────

def adapt_step_size(step_size: float, accept_rate: float, target: float = 0.574) -> float:
    """One step-size adaptation: `η ← η·exp(2·(acc − target))`.

    Roberts–Rosenthal optimal MALA acceptance is ≈0.574; the multiplicative update grows the step when
    acceptance is too high (proposals too timid) and shrinks it when too low. Used by `tune_step_size`.
    """
    if not np.isfinite(step_size) or step_size <= 0:
        raise ValueError("step_size must be finite and positive")
    return float(step_size * np.exp(2.0 * (accept_rate - target)))


def tune_step_size(
    pilot_fn: Callable[[float], float],
    step_size0: float,
    *,
    target: float = 0.574,
    band: tuple[float, float] = (0.45, 0.65),
    max_iters: int = 3,
) -> tuple[float, float, list[tuple[float, float]]]:
    """Iterate the step size via `adapt_step_size` until a pilot's acceptance lands in `band`.

    `pilot_fn(step_size) -> accept_rate` runs a short pilot chain and returns its pooled acceptance.
    Returns `(step_size, accept_rate, history)` where history is the list of `(η, acc)` probed. Stops
    as soon as acceptance ∈ band, or after `max_iters` (POE saw ≤3 iterations in practice).
    """
    eta = float(step_size0)
    history: list[tuple[float, float]] = []
    acc = float("nan")
    for _ in range(max_iters):
        acc = float(pilot_fn(eta))
        history.append((eta, acc))
        if band[0] <= acc <= band[1]:
            break
        eta = adapt_step_size(eta, acc, target)
    return eta, acc, history


def smallest_beta_that_flips(
    betas, decision_fn: Callable[[float], bool]
) -> Optional[float]:
    """Smallest β (ascending) for which `decision_fn(β)` is True, else None.

    `decision_fn(beta)` should encode the per-scenario flip rule (e.g. the contrast probe's gap is
    negative in ALL chains, or entropy/effN exceed the anchor). The β-ladder is climbed from the
    smallest candidate so the chosen explanation is the *least* tilted that still flips the decision.
    """
    for b in sorted(betas):
        if decision_fn(float(b)):
            return float(b)
    return None


# ── anchor plausibility under the prior (χ² tail) ────────────────────────────


from dataclasses import asdict, dataclass
from typing import Any, Callable, Mapping, Sequence


from poe_thesis.plausibility_prior import VAR1Fit
from poe_thesis.macro_features import (
    EXPECTED_MACRO_VARIABLES,
    MacroScaler,
    inverse_transform_macro_panel,
    transform_macro_panel,
)

ObjectiveCallable = Callable[[np.ndarray], float]
DEFAULT_ANCHOR_Z = (
    -0.5220630556899118,
    -0.9185092872654348,
    -0.6230752864567398,
    -1.220284678234281,
    -1.7874746399425967,
    -1.1248309562218408,
    1.3644093385965768,
    2.7551597871293607,
    -3.0902105949272127,
)


@dataclass(frozen=True)
class MALAEnergyConfig:
    """Frozen configuration for objective, prior, support, and gradient mechanics."""

    objective_id: str = "model_output_contrast_summer_vs_winter"
    anchor_yyyymm: int = 202004
    macro_order: tuple[str, ...] = EXPECTED_MACRO_VARIABLES
    anchor_z: tuple[float, ...] = DEFAULT_ANCHOR_Z
    objective_anchor_value: float = 0.023063493406
    objective_scale: float = 0.0009308417953261617
    objective_scale_floor: float = 1e-6
    beta: float = 0.5
    prior_scale: float = 1.0
    var1_prior_weight: float = 0.0
    var1_prior_mode: str = "one_step"     # "one_step" (legacy default) | "stationary" (decided headline)
    anchor_prior_active: bool = True      # isotropic anchor prior; set False with the stationary prior
    support_radius_from_anchor: float = 3.0
    absolute_z_bound: float = 8.0
    finite_difference_step_z: float = 0.01

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["macro_order"] = list(self.macro_order)
        payload["anchor_z"] = list(self.anchor_z)
        return payload

    @property
    def var1_active(self) -> bool:
        return self.var1_prior_weight > 0.0


@dataclass(frozen=True)
class MALAEnergyEvaluation:
    """One deterministic log-target evaluation with no sampler behavior."""

    z_standardized: tuple[float, ...]
    within_support: bool
    objective_value: float | None
    standardized_objective: float | None
    log_prior: float
    var1_log_prior: float
    log_target: float
    energy: float
    valid: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class MALAGradientEvaluation:
    """Central finite-difference gradient of the deterministic log target."""

    z_standardized: tuple[float, ...]
    gradient: tuple[float, ...]
    finite_difference_step_z: float
    objective_evaluation_count: int
    valid: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _coerce_state(state: Sequence[float] | np.ndarray, name: str) -> np.ndarray:
    values = np.asarray(state, dtype=np.float64)
    if values.shape != (len(EXPECTED_MACRO_VARIABLES),):
        raise ValueError(f"{name} must have shape (9,), got {values.shape}")
    if not np.isfinite(values).all():
        raise ValueError(f"{name} must contain only finite values")
    return values


def validate_mala_energy_config(config: MALAEnergyConfig) -> None:
    """Raise ValueError when a mechanics configuration is invalid."""

    if not isinstance(config, MALAEnergyConfig):
        raise TypeError("config must be a MALAEnergyConfig")
    from poe_thesis.probes import PROBE_REGISTRY
    if config.objective_id not in PROBE_REGISTRY:
        raise ValueError(
            f"unsupported objective_id {config.objective_id!r}; "
            f"valid IDs: {list(PROBE_REGISTRY)}"
        )
    if not isinstance(config.anchor_yyyymm, int) or config.anchor_yyyymm <= 0:
        raise ValueError("anchor_yyyymm must be a positive integer")
    if tuple(config.macro_order) != EXPECTED_MACRO_VARIABLES:
        raise ValueError("macro_order must match the canonical nine-variable order")
    _coerce_state(config.anchor_z, "anchor_z")
    finite_fields = (
        config.objective_anchor_value,
        config.objective_scale,
        config.objective_scale_floor,
        config.beta,
        config.prior_scale,
        config.support_radius_from_anchor,
        config.absolute_z_bound,
        config.finite_difference_step_z,
    )
    if not np.isfinite(finite_fields).all():
        raise ValueError("numeric config fields must be finite")
    if config.objective_scale <= 0 or config.objective_scale_floor <= 0:
        raise ValueError("objective scales must be positive")
    if config.beta < 0:
        raise ValueError("beta must be nonnegative")
    if config.var1_prior_weight < 0:
        raise ValueError("var1_prior_weight must be nonnegative")
    if config.var1_prior_mode not in ("one_step", "stationary"):
        raise ValueError("var1_prior_mode must be 'one_step' or 'stationary'")
    if config.prior_scale <= 0:
        raise ValueError("prior_scale must be positive")
    if config.support_radius_from_anchor <= 0 or config.absolute_z_bound <= 0:
        raise ValueError("support bounds must be positive")
    if config.finite_difference_step_z <= 0:
        raise ValueError("finite_difference_step_z must be positive")
    if not is_within_hard_support(config.anchor_z, config):
        raise ValueError("anchor_z must be within hard support")


def standardized_to_raw_macro(z_standardized: Sequence[float], scaler: MacroScaler) -> np.ndarray:
    """Inverse-transform one canonical standardized macro state."""

    z = _coerce_state(z_standardized, "z_standardized")
    if tuple(scaler.macro_predictors) != EXPECTED_MACRO_VARIABLES:
        raise ValueError("scaler macro order must match the canonical order")
    raw = np.asarray(inverse_transform_macro_panel(z, scaler), dtype=np.float64)
    return _coerce_state(raw, "raw_macro_state")


def raw_to_standardized_macro(raw_macro_state: Sequence[float], scaler: MacroScaler) -> np.ndarray:
    """Transform one canonical raw macro state."""

    raw = _coerce_state(raw_macro_state, "raw_macro_state")
    if tuple(scaler.macro_predictors) != EXPECTED_MACRO_VARIABLES:
        raise ValueError("scaler macro order must match the canonical order")
    z = np.asarray(transform_macro_panel(raw, scaler), dtype=np.float64)
    return _coerce_state(z, "z_standardized")


def is_within_hard_support(
    z_standardized: Sequence[float], config: MALAEnergyConfig
) -> bool:
    """Return whether a state lies in the anchor-relative and absolute boxes."""

    z = _coerce_state(z_standardized, "z_standardized")
    anchor = _coerce_state(config.anchor_z, "anchor_z")
    return bool(
        np.all(np.abs(z - anchor) <= config.support_radius_from_anchor)
        and np.all(np.abs(z) <= config.absolute_z_bound)
    )


def gaussian_anchor_log_prior(
    z_standardized: Sequence[float], config: MALAEnergyConfig
) -> float:
    """Evaluate the unnormalized isotropic Gaussian anchor log prior."""

    z = _coerce_state(z_standardized, "z_standardized")
    if not config.anchor_prior_active:
        return 0.0
    anchor = _coerce_state(config.anchor_z, "anchor_z")
    return float(-0.5 * np.sum(((z - anchor) / config.prior_scale) ** 2))


def _var1_prior_mean_metric(
    config: MALAEnergyConfig, var1_fit: VAR1Fit
) -> tuple[np.ndarray, np.ndarray] | None:
    """Return (μ, M) for the VAR(1) Mahalanobis log-prior −½·w·(z−μ)ᵀM(z−μ) under the selected mode.

    "anchor_stationary" (HEADLINE, Batuhan 2026-06-14): μ=anchor m̄, M=Σ*⁻¹ — plausibility = Mahalanobis
    deviation of the scenario FROM THE ANCHOR (where we are) under the historical VAR(1) stationary
    covariance Σ* (captures macro variances + correlations). This is the counterfactual explanation cost.
    "stationary" (alt, robustness exhibit): μ*=(I−Φ)⁻¹c, M=Σ*⁻¹ (centered at the long-run mean). Both fall
    back to one-step when ρ(Φ)≥1. "one_step" (alt): μ=c+Φ·anchor, M=Q⁻¹. Returns None when unavailable.
    """
    from poe_thesis.plausibility_prior import var1_stationary_moments

    if config.var1_prior_mode in ("stationary", "anchor_stationary"):
        moments = var1_stationary_moments(
            var1_fit.coefficient, var1_fit.intercept, var1_fit.regularized_residual_covariance
        )
        if moments.stable and moments.inverse_covariance is not None:
            # anchor_stationary keeps Σ*'s metric but recentres on the anchor (deviation from where we are)
            center = (np.asarray(config.anchor_z, dtype=np.float64)
                      if config.var1_prior_mode == "anchor_stationary" else moments.mean)
            return center, moments.inverse_covariance
        # ρ(Φ)≥1 → fall through to one-step (caller/readiness check logs the spectral radius)
    q_inv = var1_fit.inverse_regularized_residual_covariance
    if q_inv is None:
        return None
    anchor = np.asarray(config.anchor_z, dtype=np.float64)
    mu = var1_fit.intercept + var1_fit.coefficient @ anchor
    return mu, q_inv


def _var1_log_prior(
    z: np.ndarray,
    config: MALAEnergyConfig,
    var1_fit: VAR1Fit | None,
) -> float:
    """Additive VAR(1) log-prior contribution; zero when inactive or unavailable."""
    if not config.var1_active or var1_fit is None:
        return 0.0
    mean_metric = _var1_prior_mean_metric(config, var1_fit)
    if mean_metric is None:
        return 0.0
    mu, metric = mean_metric
    diff = z - mu
    return float(-0.5 * config.var1_prior_weight * (diff @ metric @ diff))


def var1_log_prior_gradient(
    z_standardized: Sequence[float] | np.ndarray,
    config: MALAEnergyConfig,
    var1_fit: VAR1Fit | None,
) -> np.ndarray:
    """Closed-form gradient of the VAR(1) Mahalanobis log-prior: ∇ = −w·M·(z−μ)."""
    z = _coerce_state(z_standardized, "z_standardized")
    if not config.var1_active or var1_fit is None:
        return np.zeros(9)
    mean_metric = _var1_prior_mean_metric(config, var1_fit)
    if mean_metric is None:
        return np.zeros(9)
    mu, metric = mean_metric
    return -config.var1_prior_weight * (metric @ (z - mu))


def evaluate_log_target(
    z_standardized: Sequence[float] | np.ndarray,
    config: MALAEnergyConfig,
    objective_callable: ObjectiveCallable,
    var1_fit: VAR1Fit | None = None,
) -> MALAEnergyEvaluation:
    """Evaluate objective reward, prior, log target, and energy without sampling.

    When var1_fit is provided and config.var1_prior_weight > 0, adds a VAR(1)
    Mahalanobis log-prior term: the candidate z is penalized by how implausible
    it is as a one-step forward state from anchor_z under the fitted VAR(1).
    """

    validate_mala_energy_config(config)
    if not callable(objective_callable):
        raise TypeError("objective_callable must be callable")
    z = _coerce_state(z_standardized, "z_standardized")
    log_prior = gaussian_anchor_log_prior(z, config)
    var1_lp = _var1_log_prior(z, config, var1_fit)
    if not is_within_hard_support(z, config):
        return MALAEnergyEvaluation(
            z_standardized=tuple(z),
            within_support=False,
            objective_value=None,
            standardized_objective=None,
            log_prior=log_prior,
            var1_log_prior=var1_lp,
            log_target=float("-inf"),
            energy=float("inf"),
            valid=False,
        )

    objective_output = np.asarray(objective_callable(z.copy()), dtype=np.float64)
    if objective_output.shape != ():
        raise ValueError("objective_callable must return one finite scalar")
    objective_value = float(objective_output)
    if not np.isfinite(objective_value):
        raise ValueError("objective_callable must return one finite scalar")
    scale = max(config.objective_scale, config.objective_scale_floor)
    standardized_objective = (objective_value - config.objective_anchor_value) / scale
    log_target = config.beta * standardized_objective + log_prior + var1_lp
    return MALAEnergyEvaluation(
        z_standardized=tuple(z),
        within_support=True,
        objective_value=objective_value,
        standardized_objective=float(standardized_objective),
        log_prior=log_prior,
        var1_log_prior=var1_lp,
        log_target=float(log_target),
        energy=float(-log_target),
        valid=True,
    )


def finite_difference_log_target_gradient(
    z_standardized: Sequence[float],
    config: MALAEnergyConfig,
    objective_callable: ObjectiveCallable,
    var1_fit: VAR1Fit | None = None,
) -> MALAGradientEvaluation:
    """Compute a central finite-difference gradient; fail closed at boundaries."""

    validate_mala_energy_config(config)
    z = _coerce_state(z_standardized, "z_standardized")
    if not is_within_hard_support(z, config):
        raise ValueError("gradient state must be within hard support")
    step = config.finite_difference_step_z
    gradient = np.empty(z.shape, dtype=np.float64)
    for index in range(z.size):
        lower = z.copy()
        upper = z.copy()
        lower[index] -= step
        upper[index] += step
        if not is_within_hard_support(lower, config) or not is_within_hard_support(upper, config):
            raise ValueError("central finite difference crosses hard support")
        lower_value = evaluate_log_target(lower, config, objective_callable, var1_fit).log_target
        upper_value = evaluate_log_target(upper, config, objective_callable, var1_fit).log_target
        gradient[index] = (upper_value - lower_value) / (2.0 * step)
    if not np.isfinite(gradient).all():
        raise ValueError("finite-difference gradient is non-finite")
    return MALAGradientEvaluation(
        z_standardized=tuple(z),
        gradient=tuple(gradient),
        finite_difference_step_z=step,
        objective_evaluation_count=2 * z.size,
        valid=True,
    )
