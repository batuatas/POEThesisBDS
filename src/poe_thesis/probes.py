# Probing functions: probe registry + benchmark-return, concentration, divergence (decision-layer event-loss rewards Y(z), event membership, thresholds).
from __future__ import annotations
from poe_thesis.macro_features import EXPECTED_MACRO_VARIABLES, MacroScaler, build_interaction_feature_matrix
from poe_thesis.samplers import standardized_to_raw_macro


from dataclasses import dataclass


@dataclass(frozen=True)
class ProbeSpec:
    probe_id: str
    family: str
    description: str
    direction: str
    requires_e2e: bool


PROBE_REGISTRY: dict[str, ProbeSpec] = {
    "model_output_contrast_summer_vs_winter": ProbeSpec(
        probe_id="model_output_contrast_summer_vs_winter",
        family="F-A",
        description=(
            "Prediction-contrast: mean absolute per-firm return-forecast disagreement "
            "between SummerChild and WinterWolf FNNs."
        ),
        direction="maximize",
        requires_e2e=False,
    ),
    "mean_return_level": ProbeSpec(
        probe_id="mean_return_level",
        family="F-C",
        description=(
            "Return-level direction: signed cross-sectional mean of a single FNN's "
            "return predictions.  sign=+1 → bullish probe; sign=-1 → bearish probe."
        ),
        direction="maximize",
        requires_e2e=False,
    ),
    "allocation_contrast_summer_vs_winter": ProbeSpec(
        probe_id="allocation_contrast_summer_vs_winter",
        family="F-B",
        description=(
            "Allocation-contrast: mean absolute per-firm portfolio-weight disagreement "
            "between the two E2E models (SummerChild and WinterWolf)."
        ),
        direction="maximize",
        requires_e2e=True,
    ),
    "portfolio_concentration_herfindahl": ProbeSpec(
        probe_id="portfolio_concentration_herfindahl",
        family="F-D",
        description=(
            "Portfolio concentration: Herfindahl index (sum of squared weights) for a "
            "single E2E model.  Higher value = more concentrated allocation."
        ),
        direction="maximize",
        requires_e2e=True,
    ),
}


def get_probe(probe_id: str) -> ProbeSpec:
    """Return the ProbeSpec for *probe_id*, or raise KeyError listing valid IDs."""
    if probe_id not in PROBE_REGISTRY:
        valid = list(PROBE_REGISTRY)
        raise KeyError(f"Unknown probe_id {probe_id!r}; valid IDs: {valid}")
    return PROBE_REGISTRY[probe_id]


import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np

from poe_thesis.macro_features import (
    EXPECTED_MACRO_VARIABLES,
    MacroScaler,
    load_clean_macro_scaler,
)

# ── default repo locations (overridable from the CLI) ────────────────────────
_DEF_SCALER = "configs/scaler/macro_scaler_thesis.json"
# single-net headline (POE-recipe re-run): seed_42 FNN member, NOT the ensemble
_DEF_FNN = "runtime/models/fnn_tactical/members/seed_42"
_DEF_E2E = "runtime/models/pao_tactical/robust_utility"
_DEF_TEST_SHARDS = "runtime/data/tactical/features/test"
_DEF_MACRO_FINAL = "runtime/data/macro_final.parquet"
_DEF_MACRO_PREDICTORS = "runtime/data/macro_predictors.npy"
_DEF_CRSP = "runtime/data/tactical/raw/crsp_monthly.parquet"
N_FIRM_CHARS = 146
N_MACRO = len(EXPECTED_MACRO_VARIABLES)  # 9

# Headline single-net construction = the ported POE recipe (run_poe_fair_compare.py): EWMA(0.94)
# covariance, λ=5, top-200-by-μ̂. Matched κ=0.25 (light robustness) is the corrected headline — the
# fair shared-universe test gives PTO@0.25 0.685 ≈ E2E@0.25 0.650 > EW 0.558 (κ=10 was a val-select
# artifact, worse on test and not matched to E2E). Both pipelines use κ=0.25 → clean PTO-vs-E2E.
PTO_KAPPA = 0.25
E2E_KAPPA = 0.25
PTO_LAMBDA = 5.0
PTO_OMEGA_MODE = "diag_sigma"
PTO_UNIVERSE_K = 200          # top-K names by μ̂(z_anchor); keeps the SOCP solve fast + book investable
PTO_COV_LOOKBACK = 60
PTO_COV_MIN_OBS = 24
PTO_COV_ESTIMATOR = "ewma"

# probes that need the PTO decision layer (Σ + realized returns) vs predictions only
_PTO_PROBES = {
    "benchmark_beating_fragility", "defensive_tilt",
    "benchmark_return",                               # Session-27: G=(r_π−b)²; r_π=w·realized toward benchmark b
    # Session-25 single-pipeline behavioral menu (each cast as an event loss G=dist(Y,A)²):
    "decision_comfort", "decision_struggle",          # #1: U* (robust-MVO utility) high / low
    "book_concentration", "book_diversification",     # #2: HHI / entropy of the book
    "book_concentration_e2e",                         # #3: HHI of the E2E book (predictor=E2E net, κ=E2E_KAPPA)
    "modelclass_utility_edge",                        # #5: U*_FNN − U*_tree (needs second_predictor=tree)
}
# probes that additionally need the E2E decision pipeline (a second predictor)
_E2E_PROBES = {
    "pto_vs_e2e_divergence", "same_return_diff_sharpe",
    "pto_vs_e2e_consensus",                           # #4: ½‖w_PTO−w_E2E‖₁ ≤ δ (consensus; divergence's twin)
}

# POE event thresholds δ for the canonical event loss G=dist(Y,A)² (data-grounded by the achievability
# scan, Batuhan 2026-06-14; reports/tactical_scenario_events.md). Featured probes: P1 (fragility @ COVID),
# P3 (divergence @ calm). P4/P2 dropped (vol/Sharpe barely macro-responsive). The chain runs min G via
# reward = −G, so log_target = −β·G + log p₀ = exp(−G/τ)·p₀ with τ=1/β (the POE-canonical Gibbs target).
EVENT_THRESHOLDS = {
    "benchmark_beating_fragility": 0.01,   # δ₁: PTO trails EW by ≥ 1%/mo  →  A={ (PTO−EW) ≤ −δ₁ }
    "pto_vs_e2e_divergence": 0.5,          # δ₃: ½‖w_PTO−w_E2E‖₁ ≥ 0.5      →  A={ Y ≥ δ₃ }
    # Session-25 menu — DATA-GROUNDED δ from the 600-draw achievability scan on the consensus k=200 book, each
    # at its SCAN-WINNING anchor. δ = the SAMPLABLE MEDIAN of the outcome Y under the prior (prior P(Y∈A)≈0.5):
    # the comfort lesson — a gradient sampler can't plant a flag inside a flat-topped event when the prior reach
    # is only 0.25 (it piles at the edge); at the median it samples cleanly (cf. #4 divergence, prior 0.60 → R̂
    # 1.05, achieve 0.76). The driver attribution is robust to δ. DO NOT guess. See [[thesis-session25-probe-menu]].
    "decision_comfort": -0.00517,          # @COVID 202004: A={ U*_FNN ≥ δ } ("more comfortable than typical")
    "decision_struggle": -0.00517,         # @COVID 202004: A={ U*_FNN ≤ δ } ("more stressed than typical")
    "book_concentration": 0.15860,         # @calm 201801:  A={ HHI(w_tree) ≥ δ }
    "book_diversification": 2.16516,       # @calm 201801:  A={ entropy(w_tree) ≥ δ }
    "book_concentration_e2e": 0.03362,     # @calm 201801:  A={ HHI(w_E2E) ≥ δ }
    "pto_vs_e2e_consensus": 0.08682,       # @COVID 202004: A={ ½‖w_PTO−w_E2E‖₁ ≤ δ } (agree in the crash)
    "modelclass_utility_edge": -0.12019,   # @COVID 202004: A={ U*_FNN − U*_tree ≥ δ }
}
# G = max(0, …)² ≤ ACHIEVE_EPS  ⇔  the event A is achieved (P(Y∈A) faithfulness; cf. scripts/tau_ladder.py)
ACHIEVE_EPS = 1e-12

# Achievement-first menu (Session-27): the THINNED posterior median of the decision quantity must land in A.
# β is walked warm→cold to the smallest β whose thinned median ∈ A and the convergence gates pass; KL / weight-
# ESS / σ-shifts are REPORTED (not gates). CEILING only if A is physically unreachable or gates fail first.
ACHIEVE_TARGET = 0.80     # reported target true-event P(Y∈A) on the thinned cloud (not a gate)
BENCH_EPS_TOL = 0.005     # benchmark event A = {|median realized − b| ≤ ε_tol} (50 bps/mo); reported.
# (25 bps was structurally near-CEILING — equaling a scalar within 25 bps on the pipelines' wide return
#  clouds forced β≈50k / ESS collapse, and the most-reachable b was within 25 bps of a pipeline's anchor
#  → trivial. 50 bps makes "median effectively matches the S&P" achievable and a non-trivial anchor exists.)


# ── data + model wiring ──────────────────────────────────────────────────────




@dataclass
class AnchorDecisionContext:
    """A fixed, macro-independent decision universe at the anchor for PTO-based probes.

    The K names are selected once (top-K by μ̂ at the anchor z); the covariance Σ is rolling-window
    and macro-independent, so it is built ONCE. Only μ̂ — and hence the PTO weights — varies with a
    candidate macro state z, which is what keeps the probe's reward smooth and the finals tractable.
    """

    firm_matrix: np.ndarray   # (K, 146) macro-independent firm characteristics
    realized: np.ndarray      # (K,) realized next-month excess returns
    sigma: np.ndarray         # (K, K) PD covariance (EWMA(0.94) rolling-60m; PTO_COV_ESTIMATOR="ewma")
    permnos: np.ndarray       # (K,) int32


# 'Size' (rank-normalized log market cap) is firm char index 4 in firm_feature_names; larger = bigger cap.
_SIZE_COL = 4




def _robust_utility(
    w: np.ndarray, mu: np.ndarray, sigma: np.ndarray, A: np.ndarray, kappa: float, lambd: float
) -> float:
    """U* = the robust-MVO objective value at the solved book w = the pipeline's "decision comfort".

    The solver Minimizes ``−μᵀw + (λ/2)·wᵀΣw + κ·‖Ω^½w‖`` (robust_mvo.py), so the achieved
    certainty-equivalent utility is U* = μᵀw − (λ/2)·wᵀΣw − κ·‖A w‖ with A = Ω^½ (= −objective). HIGH U*
    ⇒ the pipeline sees clear, trustworthy opportunity (comfortable); LOW ⇒ flat/ambiguous (struggling)."""
    w = np.asarray(w, dtype=np.float64)
    mu = np.asarray(mu, dtype=np.float64)
    quad = float(w @ sigma @ w)
    robust = float(np.linalg.norm(A @ w))
    return float(mu @ w) - 0.5 * float(lambd) * quad - float(kappa) * robust


def _hhi(w: np.ndarray) -> float:
    """Herfindahl concentration Σ wᵢ² ∈ [1/K, 1] (long-only book; 1 = single name, 1/K = equal weight)."""
    w = np.clip(np.asarray(w, dtype=np.float64), 0.0, None)
    return float(w @ w)


def _entropy(w: np.ndarray) -> float:
    """Diversification entropy −Σ wᵢ·log wᵢ ∈ [0, log K] (max at equal weights). w·log w → 0 as w→0, so
    masking near-zeros (w>1e-12) is exact and avoids log(0); weights are NOT renormalized (solver sums≈1)."""
    w = np.clip(np.asarray(w, dtype=np.float64), 0.0, None)
    nz = w[w > 1e-12]
    return float(-(nz * np.log(nz)).sum())


def build_pto_reward(
    probe: str,
    *,
    context: AnchorDecisionContext,
    scaler: MacroScaler,
    predictor: Callable[[np.ndarray], np.ndarray],
    anchor_yyyymm: int,
    kappa: float = PTO_KAPPA,
    lambd: float = PTO_LAMBDA,
    omega_mode: str = PTO_OMEGA_MODE,
    e2e_predictor: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    e2e_kappa: float = E2E_KAPPA,
    second_predictor: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    return_gap_scale: float = 0.01,
    event_threshold: Optional[float] = None,
    benchmark_value: Optional[float] = None,
) -> Callable[[np.ndarray], float]:
    """Return a raw reward r(z) for a decision-layer probe over a fixed anchor universe.

    All books are the ported POE recipe (EWMA Σ, λ, Ω=diagΣ); the μ̂ feeding the robust MVO is the
    only z-dependent input. The reward functionals (Claude's implementation; Batuhan's modeling
    sign-off):
      * ``benchmark_beating_fragility`` (P1): r = −(PTO realized − equal-weight realized). HIGH =
        macro state where the PTO book most UNDER-performs equal-weight → where the headline edge is fragile.
      * ``defensive_tilt`` (P4): r = −√(wᵀΣw), negative portfolio volatility. HIGH = most defensive book.
      * ``pto_vs_e2e_divergence`` (P3): r = ½‖w_PTO − w_E2E‖₁ ∈ [0,1]. HIGH = macro state of maximal
        allocation disagreement between the two-stage and end-to-end pipelines (the portability core).
      * ``same_return_diff_sharpe`` (P2): r = |Sharpe_PTO − Sharpe_E2E|·exp(−(Δreturn/scale)²), the
        ex-ante Sharpe gap GATED on matched realized return. HIGH = same return, different risk-adjusted.
    """
    from poe_thesis.optimizer import RobustMVOSolver, build_omega_half
    from poe_thesis.macro_features import build_interaction_feature_matrix

    sigma = context.sigma
    realized = context.realized
    firm_matrix = context.firm_matrix
    ew_return = float(np.mean(realized))
    # Ω^½ for the robust-MVO utility U* (= comfort signal); matches the solver's internal A exactly so U*
    # reconstructed in numpy from the solved w equals −(solver objective). Σ fixed → compute once.
    A_omega = build_omega_half(sigma, omega_mode)

    # Σ is macro-independent (fixed at the anchor) → pre-canonicalize the SOCP once; only μ varies with z.
    # The reward is re-evaluated thousands of times per chain, so this DPP-parameterized reuse is the main
    # per-eval speedup (the SOCP solve, not the predict, dominates the decision-layer probes).
    _pto_solver = RobustMVOSolver(sigma, kappa=kappa, lambd=lambd, omega_mode=omega_mode)
    _e2e_solver = (
        RobustMVOSolver(sigma, kappa=e2e_kappa, lambd=lambd, omega_mode=omega_mode)
        if e2e_predictor is not None else None
    )
    _solver_for = {kappa: _pto_solver}
    if _e2e_solver is not None:
        _solver_for[e2e_kappa] = _e2e_solver

    def _mu(pred: Callable[[np.ndarray], np.ndarray], z: np.ndarray) -> np.ndarray:
        raw_macro = standardized_to_raw_macro(z, scaler)
        feats = build_interaction_feature_matrix(firm_matrix, raw_macro, scaler)
        return np.asarray(pred(feats), dtype=np.float64)

    def _weights(mu_hat: np.ndarray, kap: float) -> np.ndarray:
        solver = _solver_for.get(kap) or RobustMVOSolver(sigma, kappa=kap, lambd=lambd, omega_mode=omega_mode)
        return solver.solve(mu_hat)

    if probe == "benchmark_return":
        # Push the realized return r_π = w·realized toward a market benchmark b: G = (r_π − b)², reward = −G.
        # Unlike the hinge probes, achievement is judged on the THINNED MEDIAN (|median − b| ≤ ε_tol), so there
        # is no δ here — only b. The same w (robust MVO at κ on μ̂(z)) as every other PTO book.
        if benchmark_value is None:
            raise ValueError("benchmark_return requires benchmark_value (the benchmark b)")
        _b = float(benchmark_value)

        def reward(z: np.ndarray) -> float:
            w = _weights(_mu(predictor, z), kappa)
            return -((float(w @ realized) - _b) ** 2)
        return reward

    if probe == "benchmark_beating_fragility":
        # POE event loss: event A = {Y ≤ −δ}, Y = PTO − EW return; G = dist(Y,A)² = max(0, Y+δ)²;
        # reward = −G (chain runs min G). G=0 ⇔ PTO trails EW by ≥ δ (the fragility event achieved).
        delta = EVENT_THRESHOLDS["benchmark_beating_fragility"] if event_threshold is None else event_threshold

        def reward(z: np.ndarray) -> float:
            w = _weights(_mu(predictor, z), kappa)
            y = float(w @ realized) - ew_return
            return -(max(0.0, y + delta) ** 2)
        return reward

    if probe == "defensive_tilt":
        def reward(z: np.ndarray) -> float:
            w = _weights(_mu(predictor, z), kappa)
            return -float(np.sqrt(max(float(w @ sigma @ w), 0.0)))  # high → low-vol (defensive)
        return reward

    if probe == "decision_comfort":
        # POE event loss: event A = {U* ≥ δ_hi}, U* = robust-MVO utility (the pipeline's decision comfort);
        # G = max(0, δ_hi − U*)²; reward = −G. G=0 ⇔ the pipeline achieves "comfortable" utility ≥ δ_hi.
        delta = EVENT_THRESHOLDS["decision_comfort"] if event_threshold is None else event_threshold

        def reward(z: np.ndarray) -> float:
            mu = _mu(predictor, z)
            u = _robust_utility(_weights(mu, kappa), mu, sigma, A_omega, kappa, lambd)
            return -(max(0.0, delta - u) ** 2)
        return reward

    if probe == "decision_struggle":
        # event A = {U* ≤ δ_lo} (the pipeline is "struggling" — low achievable utility); G = max(0, U* − δ_lo)².
        delta = EVENT_THRESHOLDS["decision_struggle"] if event_threshold is None else event_threshold

        def reward(z: np.ndarray) -> float:
            mu = _mu(predictor, z)
            u = _robust_utility(_weights(mu, kappa), mu, sigma, A_omega, kappa, lambd)
            return -(max(0.0, u - delta) ** 2)
        return reward

    if probe in ("book_concentration", "book_concentration_e2e"):
        # event A={HHI≥δ}; G=max(0,δ−HHI)². `predictor` is the book's OWN model (tree for #2, the E2E net for
        # #3 — set at the call site with κ=E2E_KAPPA=PTO_KAPPA). HIGH HHI = the book piles into a few names.
        delta = EVENT_THRESHOLDS[probe] if event_threshold is None else event_threshold

        def reward(z: np.ndarray) -> float:
            return -(max(0.0, delta - _hhi(_weights(_mu(predictor, z), kappa))) ** 2)
        return reward

    if probe == "book_diversification":
        # event A={entropy≥δ}; G=max(0,δ−entropy)². HIGH entropy = the (tree) book spreads out.
        delta = EVENT_THRESHOLDS["book_diversification"] if event_threshold is None else event_threshold

        def reward(z: np.ndarray) -> float:
            return -(max(0.0, delta - _entropy(_weights(_mu(predictor, z), kappa))) ** 2)
        return reward

    if probe == "modelclass_utility_edge":
        # event A={U*_a − U*_b ≥ δ}; G=max(0,δ−(U*_a−U*_b))². a=`predictor` (FNN), b=`second_predictor`
        # (tree) — both solved at κ on the SAME consensus book → macro state that favors one model class.
        if second_predictor is None:
            raise ValueError("modelclass_utility_edge requires second_predictor (the tree)")
        delta = EVENT_THRESHOLDS["modelclass_utility_edge"] if event_threshold is None else event_threshold

        def reward(z: np.ndarray) -> float:
            mu_a, mu_b = _mu(predictor, z), _mu(second_predictor, z)
            u_a = _robust_utility(_weights(mu_a, kappa), mu_a, sigma, A_omega, kappa, lambd)
            u_b = _robust_utility(_weights(mu_b, kappa), mu_b, sigma, A_omega, kappa, lambd)
            return -(max(0.0, delta - (u_a - u_b)) ** 2)
        return reward

    if probe in _E2E_PROBES:
        if e2e_predictor is None:
            raise ValueError(f"probe {probe!r} requires an e2e_predictor")

        def _exante_sharpe(w: np.ndarray, mu_hat: np.ndarray) -> float:
            vol = np.sqrt(max(float(w @ sigma @ w), 1e-12))
            return float(w @ mu_hat) / vol

        if probe == "pto_vs_e2e_divergence":
            # POE event loss: event A = {Y ≥ δ}, Y = ½‖w_PTO−w_E2E‖₁; G = max(0, δ−Y)²; reward = −G.
            # G=0 ⇔ the two pipelines disagree on ≥ δ of the book (the divergence event achieved).
            delta = EVENT_THRESHOLDS["pto_vs_e2e_divergence"] if event_threshold is None else event_threshold

            def reward(z: np.ndarray) -> float:
                w_pto = _weights(_mu(predictor, z), kappa)
                w_e2e = _weights(_mu(e2e_predictor, z), e2e_kappa)
                y = 0.5 * float(np.abs(w_pto - w_e2e).sum())
                return -(max(0.0, delta - y) ** 2)
            return reward

        if probe == "pto_vs_e2e_consensus":
            # POE event loss: event A = {Y ≤ δ}, Y = ½‖w_PTO−w_E2E‖₁ (the CONSENSUS twin of divergence);
            # G = max(0, Y − δ)²; reward = −G. G=0 ⇔ the two pipelines agree to within δ of the book.
            delta = EVENT_THRESHOLDS["pto_vs_e2e_consensus"] if event_threshold is None else event_threshold

            def reward(z: np.ndarray) -> float:
                w_pto = _weights(_mu(predictor, z), kappa)
                w_e2e = _weights(_mu(e2e_predictor, z), e2e_kappa)
                y = 0.5 * float(np.abs(w_pto - w_e2e).sum())
                return -(max(0.0, y - delta) ** 2)
            return reward

        if probe == "same_return_diff_sharpe":
            def reward(z: np.ndarray) -> float:
                mu_pto, mu_e2e = _mu(predictor, z), _mu(e2e_predictor, z)
                w_pto, w_e2e = _weights(mu_pto, kappa), _weights(mu_e2e, e2e_kappa)
                sharpe_gap = abs(_exante_sharpe(w_pto, mu_pto) - _exante_sharpe(w_e2e, mu_e2e))
                ret_gap = abs(float(w_pto @ realized) - float(w_e2e @ realized))
                gate = float(np.exp(-((ret_gap / return_gap_scale) ** 2)))
                return sharpe_gap * gate  # high → matched return, divergent Sharpe
            return reward

    raise NotImplementedError(f"decision-layer probe {probe!r} is not implemented")


def event_contains(
    probe: str,
    y: float,
    delta: Optional[float] = None,
    *,
    benchmark_value: Optional[float] = None,
    event_tol: Optional[float] = None,
) -> bool:
    """Is a scalar decision value ``y`` inside the probe's target event A? (the achievement-first test).

    Encodes each probe's event DIRECTION so the thinned-median achievement check is unambiguous. ``delta`` is
    the probe's δ (``EVENT_THRESHOLDS``); ``benchmark_value``/``event_tol`` are used only by ``benchmark_return``
    (A = {|y − b| ≤ ε_tol})."""
    if probe == "benchmark_return":
        if benchmark_value is None or event_tol is None:
            raise ValueError("benchmark_return event needs benchmark_value and event_tol")
        return abs(float(y) - float(benchmark_value)) <= float(event_tol)
    d = float(EVENT_THRESHOLDS[probe] if delta is None else delta)
    if probe == "benchmark_beating_fragility":
        return float(y) <= -d         # A = {Y ≤ −δ}  (PTO trails EW by ≥ δ)
    if probe in ("decision_struggle", "pto_vs_e2e_consensus"):
        return float(y) <= d          # A = {Y ≤ δ}
    # A = {Y ≥ δ}: divergence, comfort, concentration(_e2e), diversification, modelclass edge
    return float(y) >= d


def build_decision_quantity(
    probe: str,
    *,
    context: AnchorDecisionContext,
    scaler: MacroScaler,
    predictor: Callable[[np.ndarray], np.ndarray],
    anchor_yyyymm: int,
    kappa: float = PTO_KAPPA,
    lambd: float = PTO_LAMBDA,
    omega_mode: str = PTO_OMEGA_MODE,
    e2e_predictor: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    e2e_kappa: float = E2E_KAPPA,
    second_predictor: Optional[Callable[[np.ndarray], np.ndarray]] = None,
) -> Callable[[np.ndarray], float]:
    """Return Y(z): the per-draw HEADLINE decision quantity for ``probe`` (the inner Y of G=dist(Y,A)²).

    Shares the SAME robust-MVO wiring as :func:`build_pto_reward` so Y is byte-identical to the Y inside each
    reward branch — the thinned median + range + true-event achieve are then all derived from one Y array
    (used by the achievement-first β-walk and the write-up replay). Y per probe:
      benchmark_return → w·realized;  benchmark_beating_fragility → w·realized − EW;
      decision_comfort/struggle → U*;  book_concentration(_e2e) → HHI(w);  book_diversification → entropy(w);
      modelclass_utility_edge → U*_a − U*_b;  pto_vs_e2e_divergence/consensus → ½‖w_PTO − w_E2E‖₁."""
    from poe_thesis.optimizer import RobustMVOSolver, build_omega_half
    from poe_thesis.macro_features import build_interaction_feature_matrix

    sigma, realized, firm_matrix = context.sigma, context.realized, context.firm_matrix
    ew_return = float(np.mean(realized))
    A_omega = build_omega_half(sigma, omega_mode)
    _pto_solver = RobustMVOSolver(sigma, kappa=kappa, lambd=lambd, omega_mode=omega_mode)
    _e2e_solver = (RobustMVOSolver(sigma, kappa=e2e_kappa, lambd=lambd, omega_mode=omega_mode)
                   if e2e_predictor is not None else None)
    _solver_for = {kappa: _pto_solver}
    if _e2e_solver is not None:
        _solver_for[e2e_kappa] = _e2e_solver

    def _mu(pred, z):
        raw_macro = standardized_to_raw_macro(z, scaler)
        feats = build_interaction_feature_matrix(firm_matrix, raw_macro, scaler)
        return np.asarray(pred(feats), dtype=np.float64)

    def _weights(mu_hat, kap):
        solver = _solver_for.get(kap) or RobustMVOSolver(sigma, kappa=kap, lambd=lambd, omega_mode=omega_mode)
        return solver.solve(mu_hat)

    if probe == "benchmark_return":
        return lambda z: float(_weights(_mu(predictor, z), kappa) @ realized)
    if probe == "benchmark_beating_fragility":
        return lambda z: float(_weights(_mu(predictor, z), kappa) @ realized) - ew_return
    if probe in ("decision_comfort", "decision_struggle"):
        def yq(z):
            mu = _mu(predictor, z)
            return _robust_utility(_weights(mu, kappa), mu, sigma, A_omega, kappa, lambd)
        return yq
    if probe in ("book_concentration", "book_concentration_e2e"):
        return lambda z: _hhi(_weights(_mu(predictor, z), kappa))
    if probe == "book_diversification":
        return lambda z: _entropy(_weights(_mu(predictor, z), kappa))
    if probe == "modelclass_utility_edge":
        if second_predictor is None:
            raise ValueError("modelclass_utility_edge decision quantity requires second_predictor")
        def yq(z):
            mu_a, mu_b = _mu(predictor, z), _mu(second_predictor, z)
            u_a = _robust_utility(_weights(mu_a, kappa), mu_a, sigma, A_omega, kappa, lambd)
            u_b = _robust_utility(_weights(mu_b, kappa), mu_b, sigma, A_omega, kappa, lambd)
            return u_a - u_b
        return yq
    if probe in ("pto_vs_e2e_divergence", "pto_vs_e2e_consensus"):
        if e2e_predictor is None:
            raise ValueError(f"{probe} decision quantity requires e2e_predictor")
        def yq(z):
            w_pto = _weights(_mu(predictor, z), kappa)
            w_e2e = _weights(_mu(e2e_predictor, z), e2e_kappa)
            return 0.5 * float(np.abs(w_pto - w_e2e).sum())
        return yq
    raise NotImplementedError(f"decision quantity for probe {probe!r} is not implemented")








# ── probe → reward dispatch ──────────────────────────────────────────────────






# ── the run ──────────────────────────────────────────────────────────────────




# ── CLI ──────────────────────────────────────────────────────────────────────




if __name__ == "__main__":
    main()
