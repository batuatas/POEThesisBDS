# Return predictors: feed-forward neural network (AssetPricingFNN, ensemble) + gradient-boosted tree ensemble, with numpy-predictor factories.
from __future__ import annotations


"""AssetPricingFNN — firm-level return predictor for the tactical POE track.

Ported from POE4Nisan/Predict-Optimize-Explain/src/modules/pao_model_defs.py.
Architecture: input → 32 → 16 → 8 → 1 with BN, ReLU, Dropout at each hidden layer.
"""


import json
from pathlib import Path
from typing import Any, Dict, List, Tuple, Union

import torch
import torch.nn as nn


class AssetPricingFNN(nn.Module):
    def __init__(self, input_dim: int, dropout_rate: float = 0.5):
        super().__init__()
        self.layer1 = nn.Sequential(
            nn.Linear(int(input_dim), 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(float(dropout_rate)),
        )
        self.layer2 = nn.Sequential(
            nn.Linear(32, 16),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Dropout(float(dropout_rate)),
        )
        self.layer3 = nn.Sequential(
            nn.Linear(16, 8),
            nn.BatchNorm1d(8),
            nn.ReLU(),
            nn.Dropout(float(dropout_rate)),
        )
        self.output = nn.Linear(8, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return self.output(x).squeeze(-1)


def load_fnn_from_dir(
    load_dir: Union[str, Path],
    map_location: str = "cpu",
) -> Tuple[AssetPricingFNN, List[str], Dict[str, Any]]:
    """Load a trained AssetPricingFNN from a directory containing
    model_config.json, feature_columns.json, and state_dict.pt."""
    load_dir = Path(load_dir)
    cfg_path = load_dir / "model_config.json"
    cols_path = load_dir / "feature_columns.json"
    state_path = load_dir / "state_dict.pt"

    for p in (cfg_path, cols_path, state_path):
        if not p.exists():
            raise FileNotFoundError(f"Missing required file: {p}")

    model_cfg: Dict[str, Any] = json.loads(cfg_path.read_text())
    feature_cols: List[str] = json.loads(cols_path.read_text())

    state = torch.load(state_path, map_location=map_location)
    model = AssetPricingFNN(
        input_dim=int(model_cfg["input_dim"]),
        dropout_rate=float(model_cfg.get("dropout_rate", 0.5)),
    )
    model.load_state_dict(state, strict=True)
    model.eval()
    return model, feature_cols, model_cfg


class EnsemblePredictor:
    """Average-of-members predictor matching the AssetPricingFNN call interface.

    GKX (2020) average the predictions of multiple independently-seeded networks.
    This wrapper holds the loaded member models and exposes ``__call__(x) -> (N,)``
    returning the mean prediction across members, so downstream consumers
    (pto.py, the MALA objectives) can treat it like a single model.
    """

    def __init__(self, members: List[AssetPricingFNN]) -> None:
        if not members:
            raise ValueError("EnsemblePredictor requires at least one member model.")
        self.members = members

    def to(self, device: str | torch.device) -> "EnsemblePredictor":
        self.members = [m.to(device) for m in self.members]
        return self

    def eval(self) -> "EnsemblePredictor":
        for m in self.members:
            m.eval()
        return self

    def train(self, mode: bool = True) -> "EnsemblePredictor":
        for m in self.members:
            m.train(mode)
        return self

    def modules(self):
        for m in self.members:
            yield from m.modules()

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        preds = torch.stack([m(x) for m in self.members], dim=0)
        return preds.mean(dim=0)


def load_fnn_ensemble_from_dir(
    load_dir: Union[str, Path],
    map_location: str = "cpu",
) -> Tuple[EnsemblePredictor, List[str], Dict[str, Any]]:
    """Load an FNN ensemble written by scripts/train_fnn.py.

    Expects ``ensemble_manifest.json`` + ``feature_columns.json`` at ``load_dir``
    and one member subdirectory per seed (each a valid load_fnn_from_dir dir).
    Returns (EnsemblePredictor, feature_cols, manifest).
    """
    load_dir = Path(load_dir)
    manifest_path = load_dir / "ensemble_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing ensemble manifest: {manifest_path}")
    manifest: Dict[str, Any] = json.loads(manifest_path.read_text())

    cols_path = load_dir / "feature_columns.json"
    feature_cols: List[str] = json.loads(cols_path.read_text())

    members: List[AssetPricingFNN] = []
    for entry in manifest["members"]:
        member_dir = Path(entry["member_dir"])
        if not member_dir.is_absolute():
            member_dir = load_dir.parent / member_dir
        model, _, _ = load_fnn_from_dir(member_dir, map_location=map_location)
        members.append(model)

    return EnsemblePredictor(members), feature_cols, manifest


def is_ensemble_dir(load_dir: Union[str, Path]) -> bool:
    """True if ``load_dir`` holds an ensemble manifest (vs a single checkpoint)."""
    return (Path(load_dir) / "ensemble_manifest.json").exists()


from typing import Any, Callable, Dict, List, Tuple, Union

import numpy as np

TreeReg = Any  # sklearn HistGradientBoostingRegressor | RandomForestRegressor


class TacticalTreeEnsemble:
    """Average-of-members tree predictor matching the FNN `EnsemblePredictor` call interface.

    Ensembling tree members (different seeds) is required, not cosmetic: a single GBRT's feature
    attribution is unstable ("first-mover bias" — importance lands on arbitrary members of correlated
    feature groups), so we average independently-seeded members before any explanation comparison.
    """

    def __init__(self, members: List[TreeReg]) -> None:
        if not members:
            raise ValueError("TacticalTreeEnsemble requires at least one member model.")
        self.members = members

    def __call__(self, features: np.ndarray) -> np.ndarray:
        features = np.asarray(features, dtype=np.float32)
        preds = np.stack([m.predict(features) for m in self.members], axis=0)
        return preds.mean(axis=0)


def save_tree_to_dir(model: TreeReg, save_dir: Union[str, Path], feature_cols: List[str],
                     model_config: Dict[str, Any]) -> None:
    """Persist one tree member: model.joblib + model_config.json + feature_columns.json."""
    import joblib

    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, str(save_dir / "model.joblib"), compress=3)
    (save_dir / "model_config.json").write_text(json.dumps(model_config, indent=2))
    (save_dir / "feature_columns.json").write_text(json.dumps(feature_cols, indent=2))


def load_tree_from_dir(load_dir: Union[str, Path]) -> Tuple[TreeReg, List[str], Dict[str, Any]]:
    """Load one tree member written by `save_tree_to_dir`."""
    import joblib

    load_dir = Path(load_dir)
    cfg_path, cols_path, model_path = (load_dir / "model_config.json",
                                       load_dir / "feature_columns.json",
                                       load_dir / "model.joblib")
    for p in (cfg_path, cols_path, model_path):
        if not p.exists():
            raise FileNotFoundError(f"Missing required file: {p}")
    cfg = json.loads(cfg_path.read_text())
    feature_cols = json.loads(cols_path.read_text())
    model = joblib.load(str(model_path))
    return model, feature_cols, cfg


def is_tree_dir(load_dir: Union[str, Path]) -> bool:
    """True if `load_dir` holds a tree ensemble (`tree_manifest.json`) — distinct from an FNN dir."""
    return (Path(load_dir) / "tree_manifest.json").exists()


def load_tree_ensemble_from_dir(
    load_dir: Union[str, Path],
) -> Tuple[TacticalTreeEnsemble, List[str], Dict[str, Any]]:
    """Load a tree ensemble (`tree_manifest.json` + one member subdir per seed)."""
    load_dir = Path(load_dir)
    manifest = json.loads((load_dir / "tree_manifest.json").read_text())
    feature_cols = json.loads((load_dir / "feature_columns.json").read_text())
    members: List[TreeReg] = []
    for entry in manifest["members"]:
        member_dir = Path(entry["member_dir"])
        if not member_dir.is_absolute():
            member_dir = load_dir / member_dir
        model, _, _ = load_tree_from_dir(member_dir)
        members.append(model)
    return TacticalTreeEnsemble(members), feature_cols, manifest


def make_tree_numpy_predictor(load_dir: Union[str, Path]) -> Callable[[np.ndarray], np.ndarray]:
    """Load a tree (ensemble or single) → numpy `(N,1460) -> (N,)` closure (mirrors the FNN closure)."""
    if is_tree_dir(load_dir):
        ensemble, _cols, _m = load_tree_ensemble_from_dir(load_dir)
    else:
        model, _cols, _cfg = load_tree_from_dir(load_dir)
        ensemble = TacticalTreeEnsemble([model])

    def predict(features: np.ndarray) -> np.ndarray:
        return np.asarray(ensemble(features), dtype=np.float64)

    return predict


# ── default repo locations (overridable from the CLI) ────────────────────────
# single-net headline (POE-recipe re-run): seed_42 FNN member, NOT the ensemble

# Headline single-net construction = the ported POE recipe (run_poe_fair_compare.py): EWMA(0.94)
# covariance, λ=5, top-200-by-μ̂. Matched κ=0.25 (light robustness) is the corrected headline — the
# fair shared-universe test gives PTO@0.25 0.685 ≈ E2E@0.25 0.650 > EW 0.558 (κ=10 was a val-select
# artifact, worse on test and not matched to E2E). Both pipelines use κ=0.25 → clean PTO-vs-E2E.

# probes that need the PTO decision layer (Σ + realized returns) vs predictions only
# probes that additionally need the E2E decision pipeline (a second predictor)

# POE event thresholds δ for the canonical event loss G=dist(Y,A)² (data-grounded by the achievability
# scan, Batuhan 2026-06-14; reports/tactical_scenario_events.md). Featured probes: P1 (fragility @ COVID),
# P3 (divergence @ calm). P4/P2 dropped (vol/Sharpe barely macro-responsive). The chain runs min G via
# reward = −G, so log_target = −β·G + log p₀ = exp(−G/τ)·p₀ with τ=1/β (the POE-canonical Gibbs target).
# G = max(0, …)² ≤ ACHIEVE_EPS  ⇔  the event A is achieved (P(Y∈A) faithfulness; cf. scripts/tau_ladder.py)

# Achievement-first menu (Session-27): the THINNED posterior median of the decision quantity must land in A.
# β is walked warm→cold to the smallest β whose thinned median ∈ A and the convergence gates pass; KL / weight-
# ESS / σ-shifts are REPORTED (not gates). CEILING only if A is physically unreachable or gates fail first.
# (25 bps was structurally near-CEILING — equaling a scalar within 25 bps on the pipelines' wide return
#  clouds forced β≈50k / ESS collapse, and the most-reachable b was within 25 bps of a pipeline's anchor
#  → trivial. 50 bps makes "median effectively matches the S&P" achievable and a non-trivial anchor exists.)


# ── data + model wiring ──────────────────────────────────────────────────────






# 'Size' (rank-normalized log market cap) is firm char index 4 in firm_feature_names; larger = bigger cap.
















def make_fnn_numpy_predictor(ensemble_dir: str | Path) -> Callable[[np.ndarray], np.ndarray]:
    """Load the GKX FNN ensemble and expose a numpy (N, 1460) → (N,) predictor closure."""
    import torch

    from poe_thesis.predictors import (
        is_ensemble_dir,
        load_fnn_ensemble_from_dir,
        load_fnn_from_dir,
    )

    if is_ensemble_dir(ensemble_dir):
        predictor, _cols, _manifest = load_fnn_ensemble_from_dir(ensemble_dir)
    else:
        predictor, _cols, _cfg = load_fnn_from_dir(ensemble_dir)
    predictor.eval()

    def predict(features: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            tensor = torch.from_numpy(np.ascontiguousarray(features)).float()
            out = predictor(tensor)
        return out.detach().cpu().numpy().astype(np.float64)

    return predict






# ── probe → reward dispatch ──────────────────────────────────────────────────






# ── the run ──────────────────────────────────────────────────────────────────




# ── CLI ──────────────────────────────────────────────────────────────────────




if __name__ == "__main__":
    main()
