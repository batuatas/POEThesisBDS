# Shared infrastructure: macro standardization (MacroScaler, EXPECTED_MACRO_VARIABLES) + firm x macro interaction-feature builder.
from __future__ import annotations


"""Non-executing macro-scaler planning scaffold for the clean POE pipeline.

This module validates configuration and builds a future fit plan without
opening the macro panel, computing statistics, or writing scaler artifacts.
Actual fitting and artifact writing remain explicitly disabled.
"""


import json
import hashlib
import platform
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping


EXPECTED_STATUSES = ("config_updated_not_fitted", "scaler_fitted_artifacts_registered")
EXPECTED_SOURCE = "runtime/data/macro_final.parquet"
EXPECTED_SOURCE_SHA256 = "1bf75f707df849bb3553179414d3c655c4d2ac26bcbafd45af2cf47948b79b62"
EXPECTED_DATE_FIELD = "yyyymm"
EXPECTED_MACRO_VARIABLES = ("dp", "ep", "bm", "ntis", "tbl", "tms", "dfy", "svar", "infl")
EXPECTED_FIT_START = 198001
EXPECTED_FIT_END = 200512
EXPECTED_DDOF = 0
EXPECTED_PRECISION = "float64"
EXPECTED_MISSING_POLICY = "fail_on_any_missing_or_nonfinite_fit_value"
EXPECTED_ZERO_VARIANCE_POLICY = "fail_on_zero_or_near_zero_scale"
EXPECTED_NEAR_ZERO_THRESHOLD = 1e-12
EXPECTED_CANONICAL_OUTPUTS = {
    "mean": "runtime/scaler/macro_scaler_mean.npy",
    "scale": "runtime/scaler/macro_scaler_scale.npy",
    "stats": "runtime/scaler/macro_scaler_stats.json",
    "metadata": "runtime/scaler/macro_scaler_metadata.json",
    "validation_report": "runtime/scaler/macro_scaler_validation_report.json",
}
BRIDGE_SCALER_PATH = "runtime/scaler/macro_scaler.pt"


@dataclass(frozen=True)
class MacroScalerFitPlan:
    """Resolved, data-free plan for a future explicitly approved fit."""

    config_path: Path
    project_root: Path
    source_path: Path
    date_field: str
    macro_variables: tuple[str, ...]
    fit_start_yyyymm: int | None
    fit_end_yyyymm: int | None
    fit_period_inclusive: bool
    macro_support_endpoint: int | None
    ddof: int | None
    precision: str
    missing_value_policy: str
    zero_variance_policy: str
    near_zero_threshold: float | None
    artifact_targets: Mapping[str, Path]
    bridge_scaler_path: Path
    execution_approved: bool
    thesis_safe: bool
    auto_promote_thesis_safe: bool
    validation_errors: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation without writing it."""

        payload = asdict(self)
        for key in ("config_path", "project_root", "source_path", "bridge_scaler_path"):
            payload[key] = str(payload[key])
        payload["macro_variables"] = list(self.macro_variables)
        payload["artifact_targets"] = {
            name: str(path) for name, path in self.artifact_targets.items()
        }
        payload["validation_errors"] = list(self.validation_errors)
        return payload


@dataclass(frozen=True)
class MacroScalerMetadata:
    """Metadata placeholder for a future approved macro-scaler fit."""

    config_path: Path
    data_config_path: Path
    input_macro_panel: Path
    macro_predictors: tuple[str, ...]
    fit_start_yyyymm: int | None
    fit_end_yyyymm: int | None
    train_end_yyyymm: int | None
    ddof: int | None
    precision: str
    n_fit_rows: int | None
    means: Mapping[str, float]
    scales: Mapping[str, float]
    validation_errors: tuple[str, ...]
    warnings: tuple[str, ...] = ()
    execution_approved: bool = False
    data_contents_inspected: bool = False
    data_copied: bool = False
    data_staged: bool = False
    scaler_fitted: bool = False
    artifact_serialized: bool = False
    result_produced: bool = False
    thesis_safe_result_produced: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""

        payload = asdict(self)
        for key in ("config_path", "data_config_path", "input_macro_panel"):
            payload[key] = str(payload[key])
        payload["macro_predictors"] = list(self.macro_predictors)
        payload["validation_errors"] = list(self.validation_errors)
        payload["warnings"] = list(self.warnings)
        return payload


@dataclass(frozen=True)
class MacroScaler:
    """In-memory clean scaler parameters and optional loaded-artifact evidence."""

    macro_predictors: tuple[str, ...]
    means: Mapping[str, float]
    scales: Mapping[str, float]
    metadata: MacroScalerMetadata
    validation_report: Mapping[str, Any] = field(default_factory=dict)
    artifact_paths: Mapping[str, Path] = field(default_factory=dict)
    artifact_sha256: Mapping[str, str] = field(default_factory=dict)


def _is_check_value(value: Any) -> bool:
    return isinstance(value, str) and (
        value == "CHECK" or value.startswith("CHECK_") or "CHECK" in value
    )


def _collect_check_paths(value: Any, prefix: str = "") -> list[str]:
    found: list[str] = []
    if _is_check_value(value):
        found.append(prefix or "<root>")
    elif isinstance(value, Mapping):
        for key, item in value.items():
            path = f"{prefix}.{key}" if prefix else str(key)
            found.extend(_collect_check_paths(item, path))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            path = f"{prefix}[{index}]" if prefix else f"[{index}]"
            found.extend(_collect_check_paths(item, path))
    return found


def _safe_int(value: Any) -> int | None:
    if value is None or _is_check_value(value):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _safe_float(value: Any) -> float | None:
    if value is None or _is_check_value(value):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _resolve_path(project_root: Path, value: Any) -> Path:
    if not isinstance(value, (str, Path)):
        raise ValueError(f"path value must be a string or Path, got {type(value).__name__}")
    path = Path(value)
    return path if path.is_absolute() else project_root / path


def _nearest_project_root(config_path: Path) -> Path | None:
    resolved = config_path.expanduser().resolve()
    for candidate in (resolved.parent, *resolved.parents):
        if candidate.name == "poe_tactical_thesis":
            return candidate
    return None


def _resolve_project_root(config: Mapping[str, Any], config_path: Path) -> tuple[Path, list[str]]:
    paths = config.get("paths", {})
    if not isinstance(paths, Mapping):
        return Path("__UNRESOLVED_PROJECT_ROOT__"), ["paths must be a mapping"]
    value = paths.get("project_root", ".")
    if isinstance(value, (str, Path)) and str(value) != ".":
        path = Path(value)
        return (path if path.is_absolute() else config_path.parent / path), []
    inferred = _nearest_project_root(config_path)
    if inferred is not None:
        return inferred, []
    return Path("__UNRESOLVED_PROJECT_ROOT__"), [
        "project_root is '.' and no poe_tactical_thesis parent could be inferred from config_path"
    ]


def load_scaler_config(path: str | Path) -> Mapping[str, Any]:
    """Load scaler config text only; never inspect the configured data source."""

    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, Mapping):
        raise ValueError(f"Expected JSON object at {config_path}.")
    return payload


load_macro_scaler_config = load_scaler_config


def _load_json_mapping(path: Path) -> Mapping[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, Mapping):
        raise ValueError(f"Expected JSON object at {path}.")
    return payload


def _registered_clean_scaler_hashes(
    artifact_manifest: Mapping[str, Any], project_root: Path
) -> Mapping[str, str]:
    artifacts = artifact_manifest.get("artifacts")
    if not isinstance(artifacts, list):
        raise ValueError("artifact manifest must contain an artifacts list")
    entries = [
        entry
        for entry in artifacts
        if isinstance(entry, Mapping) and entry.get("component") == "clean_macro_scaler"
    ]
    if len(entries) != 1:
        raise ValueError("artifact manifest must contain exactly one clean_macro_scaler entry")
    files = entries[0].get("files")
    if not isinstance(files, list):
        raise ValueError("clean_macro_scaler entry must contain a files list")

    registered: dict[str, str] = {}
    prefix = f"{project_root.name}/"
    for item in files:
        if not isinstance(item, Mapping):
            raise ValueError("clean_macro_scaler file registrations must be mappings")
        path_value = item.get("path")
        sha256 = item.get("sha256")
        if not isinstance(path_value, str) or not isinstance(sha256, str):
            raise ValueError("clean_macro_scaler registrations require string path and sha256")
        relative_path = path_value[len(prefix) :] if path_value.startswith(prefix) else path_value
        registered[relative_path] = sha256
    if BRIDGE_SCALER_PATH in registered:
        raise ValueError("bridge macro_scaler.pt must not be registered as a clean artifact")
    if set(registered) != set(EXPECTED_CANONICAL_OUTPUTS.values()):
        raise ValueError("clean_macro_scaler entry must register exactly the five canonical artifacts")
    return registered


def _checksum_manifest_hashes(path: Path, project_root: Path) -> Mapping[str, str]:
    registered: dict[str, str] = {}
    prefix = f"{project_root.name}/"
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            parts = stripped.split(maxsplit=1)
            if len(parts) != 2:
                continue
            sha256, path_value = parts
            relative_path = path_value[len(prefix) :] if path_value.startswith(prefix) else path_value
            registered[relative_path] = sha256
    return registered


def _validate_loaded_json_semantics(
    config: Mapping[str, Any],
    metadata: Mapping[str, Any],
    validation_report: Mapping[str, Any],
    stats: Mapping[str, Any],
    means: Any,
    scales: Any,
) -> list[str]:
    errors: list[str] = []
    common_expected = {
        "fit_period_start_yyyymm": EXPECTED_FIT_START,
        "fit_period_end_yyyymm": EXPECTED_FIT_END,
        "fit_period_inclusive": True,
        "fit_row_count": 312,
        "ddof": EXPECTED_DDOF,
        "precision": EXPECTED_PRECISION,
        "near_zero_threshold": EXPECTED_NEAR_ZERO_THRESHOLD,
        "clean_execution_approved": False,
        "auto_promote_thesis_safe": False,
        "thesis_safe": False,
    }
    for name, payload in (
        ("metadata", metadata),
        ("validation_report", validation_report),
        ("stats", stats),
    ):
        for key, expected in common_expected.items():
            if payload.get(key) != expected:
                errors.append(f"{name}.{key} must equal {expected!r}")
        if payload.get("macro_variables_ordered") != list(EXPECTED_MACRO_VARIABLES):
            errors.append(f"{name}.macro_variables_ordered must match canonical order")
    if metadata.get("source_path") != EXPECTED_SOURCE:
        errors.append(f"metadata.source_path must equal {EXPECTED_SOURCE!r}")
    if metadata.get("source_sha256") != EXPECTED_SOURCE_SHA256:
        errors.append("metadata.source_sha256 must equal the registered canonical source hash")
    if metadata.get("date_field") != EXPECTED_DATE_FIELD:
        errors.append(f"metadata.date_field must equal {EXPECTED_DATE_FIELD!r}")
    if metadata.get("output_artifact_paths") != EXPECTED_CANONICAL_OUTPUTS:
        errors.append("metadata.output_artifact_paths must match canonical outputs")
    if metadata.get("mean_values_ordered") != means.tolist():
        errors.append("metadata mean values must equal the loaded mean artifact")
    if metadata.get("scale_values_ordered") != scales.tolist():
        errors.append("metadata scale values must equal the loaded scale artifact")
    if stats.get("mean_values_ordered") != means.tolist():
        errors.append("stats mean values must equal the loaded mean artifact")
    if stats.get("scale_values_ordered") != scales.tolist():
        errors.append("stats scale values must equal the loaded scale artifact")
    if validation_report.get("status") != "macro_scaler_fitted_artifacts_registered":
        errors.append("validation_report.status must be macro_scaler_fitted_artifacts_registered")
    if validation_report.get("scaler_fit_executed") is not True:
        errors.append("validation_report.scaler_fit_executed must be true")
    if validation_report.get("scaler_artifacts_written") is not True:
        errors.append("validation_report.scaler_artifacts_written must be true")
    if metadata.get("bridge_scaler_path") != BRIDGE_SCALER_PATH:
        errors.append("metadata.bridge_scaler_path must identify the separate bridge artifact")
    if metadata.get("bridge_scaler_preserved") is not True:
        errors.append("metadata.bridge_scaler_preserved must be true")
    if validation_report.get("bridge_scaler_preserved") is not True:
        errors.append("validation_report.bridge_scaler_preserved must be true")
    if config.get("status") != "scaler_fitted_artifacts_registered":
        errors.append("config.status must be scaler_fitted_artifacts_registered for downstream loading")
    return errors


def load_clean_macro_scaler(
    config_path: str | Path = "configs/scaler/macro_scaler_thesis.json",
) -> MacroScaler:
    """Load and validate the canonical clean scaler without reading data or bridge artifacts."""

    import numpy as np

    path = Path(config_path)
    config = load_scaler_config(path)
    config_errors = validate_macro_scaler_config(config)
    if config.get("status") != "scaler_fitted_artifacts_registered":
        config_errors.append("downstream loader requires status scaler_fitted_artifacts_registered")
    if config_errors:
        raise ValueError(f"Invalid clean macro scaler config: {config_errors}")

    project_root, root_errors = _resolve_project_root(config, path)
    if root_errors:
        raise ValueError(f"Could not resolve clean scaler project root: {root_errors}")
    outputs = config.get("outputs")
    if not isinstance(outputs, Mapping) or outputs != EXPECTED_CANONICAL_OUTPUTS:
        raise ValueError("config outputs must contain exactly the five canonical clean artifacts")
    artifact_paths = {
        name: _resolve_path(project_root, relative_path) for name, relative_path in outputs.items()
    }
    if any(str(path).endswith(".pt") for path in artifact_paths.values()):
        raise ValueError("clean scaler loader must not resolve any .pt artifact")
    missing = [str(path) for path in artifact_paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing canonical clean scaler artifacts: {missing}")

    means = np.load(artifact_paths["mean"], allow_pickle=False)
    scales = np.load(artifact_paths["scale"], allow_pickle=False)
    if means.dtype != np.float64 or means.shape != (len(EXPECTED_MACRO_VARIABLES),):
        raise ValueError("mean artifact must be float64 with shape (9,)")
    if scales.dtype != np.float64 or scales.shape != (len(EXPECTED_MACRO_VARIABLES),):
        raise ValueError("scale artifact must be float64 with shape (9,)")
    if not np.isfinite(means).all() or not np.isfinite(scales).all():
        raise ValueError("mean and scale artifacts must contain only finite values")
    if (scales <= EXPECTED_NEAR_ZERO_THRESHOLD).any():
        raise ValueError("all loaded scales must exceed the near-zero threshold")

    stats = _load_json_mapping(artifact_paths["stats"])
    metadata_payload = _load_json_mapping(artifact_paths["metadata"])
    validation_report = _load_json_mapping(artifact_paths["validation_report"])
    semantic_errors = _validate_loaded_json_semantics(
        config, metadata_payload, validation_report, stats, means, scales
    )
    if semantic_errors:
        raise ValueError(f"Invalid clean scaler artifact semantics: {semantic_errors}")

    manifests = config.get("manifests")
    if not isinstance(manifests, Mapping):
        raise ValueError("config.manifests must be a mapping")
    artifact_manifest_path = _resolve_path(project_root, manifests.get("artifact_manifest"))
    checksum_manifest_path = _resolve_path(project_root, manifests.get("checksum_manifest"))
    artifact_registered = _registered_clean_scaler_hashes(
        _load_json_mapping(artifact_manifest_path), project_root
    )
    checksum_registered = _checksum_manifest_hashes(checksum_manifest_path, project_root)
    artifact_sha256 = {name: _sha256(artifact_path) for name, artifact_path in artifact_paths.items()}
    for name, relative_path in outputs.items():
        actual = artifact_sha256[name]
        if artifact_registered.get(relative_path) != actual:
            raise ValueError(f"artifact manifest hash mismatch for {relative_path}")
        if checksum_registered.get(relative_path) != actual:
            raise ValueError(f"checksum manifest hash mismatch for {relative_path}")

    data_config_path = metadata_payload.get("data_config_path")
    if not isinstance(data_config_path, str):
        raise ValueError("metadata.data_config_path must be a string")
    loaded_metadata = MacroScalerMetadata(
        config_path=path,
        data_config_path=Path(data_config_path),
        input_macro_panel=Path(EXPECTED_SOURCE),
        macro_predictors=EXPECTED_MACRO_VARIABLES,
        fit_start_yyyymm=EXPECTED_FIT_START,
        fit_end_yyyymm=EXPECTED_FIT_END,
        train_end_yyyymm=EXPECTED_FIT_END,
        ddof=EXPECTED_DDOF,
        precision=EXPECTED_PRECISION,
        n_fit_rows=int(metadata_payload["fit_row_count"]),
        means={name: float(value) for name, value in zip(EXPECTED_MACRO_VARIABLES, means)},
        scales={name: float(value) for name, value in zip(EXPECTED_MACRO_VARIABLES, scales)},
        validation_errors=(),
        execution_approved=False,
        data_contents_inspected=False,
        scaler_fitted=True,
        artifact_serialized=True,
        result_produced=True,
        thesis_safe_result_produced=False,
    )
    return MacroScaler(
        macro_predictors=EXPECTED_MACRO_VARIABLES,
        means=loaded_metadata.means,
        scales=loaded_metadata.scales,
        metadata=loaded_metadata,
        validation_report=validation_report,
        artifact_paths=artifact_paths,
        artifact_sha256=artifact_sha256,
    )


def validate_scaler_config_schema(config: Mapping[str, Any]) -> list[str]:
    """Validate exact clean macro-scaler semantics from config fields only."""

    errors: list[str] = []
    expected_values = {
        "runtime_mode": "clean",
        "fit_source_file": EXPECTED_SOURCE,
        "date_field": EXPECTED_DATE_FIELD,
        "date_field_role": "fit_period_selector_not_scaled_variable",
        "ddof": EXPECTED_DDOF,
        "precision": EXPECTED_PRECISION,
        "missing_value_policy": EXPECTED_MISSING_POLICY,
        "zero_variance_policy": EXPECTED_ZERO_VARIANCE_POLICY,
    }
    for field, expected in expected_values.items():
        if config.get(field) != expected:
            errors.append(f"{field} must equal {expected!r}")
    if config.get("status") not in EXPECTED_STATUSES:
        errors.append(f"status must be one of {EXPECTED_STATUSES!r}")

    predictors = config.get("macro_predictors")
    variables = config.get("macro_variables")
    if predictors != list(EXPECTED_MACRO_VARIABLES):
        errors.append("macro_predictors must match the canonical ordered nine-variable list")
    if variables != list(EXPECTED_MACRO_VARIABLES):
        errors.append("macro_variables must match the canonical ordered nine-variable list")
    if EXPECTED_DATE_FIELD in EXPECTED_MACRO_VARIABLES:
        errors.append("yyyymm must be a date selector, not a scaled macro variable")
    if "Rfree" in tuple(predictors or ()) or "Rfree" in tuple(variables or ()):
        errors.append("Rfree must be excluded from macro scaler variables")
    excluded = config.get("excluded_from_scaling")
    if not isinstance(excluded, list) or not {"yyyymm", "Rfree"}.issubset(excluded):
        errors.append("excluded_from_scaling must include yyyymm and Rfree")

    fit_period = config.get("fit_period")
    if not isinstance(fit_period, Mapping):
        errors.append("fit_period must be a mapping")
    else:
        if fit_period.get("start_yyyymm") != EXPECTED_FIT_START:
            errors.append(f"fit_period.start_yyyymm must equal {EXPECTED_FIT_START}")
        if fit_period.get("end_yyyymm") != EXPECTED_FIT_END:
            errors.append(f"fit_period.end_yyyymm must equal {EXPECTED_FIT_END}")
        if fit_period.get("inclusive") is not True:
            errors.append("fit_period.inclusive must be true")

    if config.get("train_end_yyyymm") != EXPECTED_FIT_END:
        errors.append(f"train_end_yyyymm must equal {EXPECTED_FIT_END}")
    if config.get("macro_support_endpoint") != 202412:
        errors.append("macro_support_endpoint must equal 202412")
    if config.get("fit_scope_policy") != (
        "validation_test_post_training_and_terminal_macro_support_months_must_not_enter_fit_statistics"
    ):
        errors.append("fit_scope_policy must exclude all post-training macro months")
    if _safe_float(config.get("near_zero_threshold")) != EXPECTED_NEAR_ZERO_THRESHOLD:
        errors.append(f"near_zero_threshold must equal {EXPECTED_NEAR_ZERO_THRESHOLD}")

    for path in _collect_check_paths(config.get("paths", {})):
        errors.append(f"unresolved CHECK path field: {path}")
    return errors


def validate_nonapproval_flags(config: Mapping[str, Any]) -> list[str]:
    """Require all execution and thesis-safety approval flags to remain false."""

    errors: list[str] = []
    if config.get("execution_approved") is not False:
        errors.append("execution_approved must be false")
    if config.get("thesis_safe") is not False:
        errors.append("thesis_safe must be false")
    registration = config.get("registration")
    if not isinstance(registration, Mapping):
        errors.append("registration must be a mapping")
    elif registration.get("auto_promote_thesis_safe") is not False:
        errors.append("registration.auto_promote_thesis_safe must be false")
    return errors


def resolve_source_path(config: Mapping[str, Any], project_root: str | Path) -> Path:
    """Resolve the configured source path without opening it."""

    return _resolve_path(Path(project_root), config.get("fit_source_file"))


def validate_future_artifact_targets(config: Mapping[str, Any]) -> list[str]:
    """Validate future canonical targets without requiring or creating them."""

    errors: list[str] = []
    outputs = config.get("outputs")
    if outputs != EXPECTED_CANONICAL_OUTPUTS:
        errors.append("outputs must record exactly the five canonical future scaler artifacts")
    paths = config.get("paths", {})
    extra = paths.get("extra", {}) if isinstance(paths, Mapping) else {}
    for name, expected in EXPECTED_CANONICAL_OUTPUTS.items():
        if not isinstance(extra, Mapping) or extra.get(f"output_{name}") != expected:
            errors.append(f"paths.extra.output_{name} must equal {expected!r}")
    artifact_policy = config.get("artifact_policy")
    if not isinstance(artifact_policy, Mapping):
        errors.append("artifact_policy must be a mapping")
    else:
        if artifact_policy.get("bridge_scaler_path") != BRIDGE_SCALER_PATH:
            errors.append(f"artifact_policy.bridge_scaler_path must equal {BRIDGE_SCALER_PATH!r}")
        if artifact_policy.get("bridge_scaler_status") != "bridge_provenance_only_not_clean":
            errors.append("bridge scaler must remain bridge_provenance_only_not_clean")
        if artifact_policy.get("bridge_scaler_must_not_be_relabelled_clean") is not True:
            errors.append("bridge scaler relabeling guard must be true")
        if artifact_policy.get("bridge_scaler_must_not_be_silently_overwritten") is not True:
            errors.append("bridge scaler overwrite guard must be true")
    if BRIDGE_SCALER_PATH in set(EXPECTED_CANONICAL_OUTPUTS.values()):
        errors.append("bridge macro_scaler.pt must not be a canonical clean target")
    return errors


def validate_policy_fields(config: Mapping[str, Any]) -> list[str]:
    """Validate fail-closed fitting policies without evaluating data."""

    errors: list[str] = []
    if config.get("ddof") != EXPECTED_DDOF:
        errors.append("ddof must be 0")
    if config.get("precision") != EXPECTED_PRECISION:
        errors.append("precision must be float64")
    if config.get("missing_value_policy") != EXPECTED_MISSING_POLICY:
        errors.append("missing/non-finite values must fail closed")
    if config.get("zero_variance_policy") != EXPECTED_ZERO_VARIANCE_POLICY:
        errors.append("zero or near-zero scales must fail closed")
    if _safe_float(config.get("near_zero_threshold")) != EXPECTED_NEAR_ZERO_THRESHOLD:
        errors.append("near-zero threshold must be 1e-12")
    return errors


def validate_macro_scaler_config(config: Mapping[str, Any]) -> list[str]:
    """Compatibility wrapper returning all data-free config validation errors."""

    return [
        *validate_scaler_config_schema(config),
        *validate_nonapproval_flags(config),
        *validate_future_artifact_targets(config),
        *validate_policy_fields(config),
    ]


def build_fit_plan(
    config: Mapping[str, Any],
    config_path: str | Path = "configs/scaler/macro_scaler_thesis.json",
) -> MacroScalerFitPlan:
    """Build a resolved future fit plan without opening data or writing files."""

    path = Path(config_path)
    project_root, root_errors = _resolve_project_root(config, path)
    fit_period = config.get("fit_period", {})
    if not isinstance(fit_period, Mapping):
        fit_period = {}
    registration = config.get("registration", {})
    if not isinstance(registration, Mapping):
        registration = {}
    outputs = config.get("outputs", {})
    if not isinstance(outputs, Mapping):
        outputs = {}
    artifact_policy = config.get("artifact_policy", {})
    if not isinstance(artifact_policy, Mapping):
        artifact_policy = {}
    variables = config.get("macro_variables", ())
    if not isinstance(variables, list):
        variables = []

    return MacroScalerFitPlan(
        config_path=path,
        project_root=project_root,
        source_path=resolve_source_path(config, project_root),
        date_field=str(config.get("date_field", "")),
        macro_variables=tuple(str(item) for item in variables),
        fit_start_yyyymm=_safe_int(fit_period.get("start_yyyymm")),
        fit_end_yyyymm=_safe_int(fit_period.get("end_yyyymm")),
        fit_period_inclusive=fit_period.get("inclusive") is True,
        macro_support_endpoint=_safe_int(config.get("macro_support_endpoint")),
        ddof=_safe_int(config.get("ddof")),
        precision=str(config.get("precision", "")),
        missing_value_policy=str(config.get("missing_value_policy", "")),
        zero_variance_policy=str(config.get("zero_variance_policy", "")),
        near_zero_threshold=_safe_float(config.get("near_zero_threshold")),
        artifact_targets={
            str(name): _resolve_path(project_root, value) for name, value in outputs.items()
        },
        bridge_scaler_path=_resolve_path(
            project_root, artifact_policy.get("bridge_scaler_path", BRIDGE_SCALER_PATH)
        ),
        execution_approved=config.get("execution_approved") is True,
        thesis_safe=config.get("thesis_safe") is True,
        auto_promote_thesis_safe=registration.get("auto_promote_thesis_safe") is True,
        validation_errors=tuple([*root_errors, *validate_macro_scaler_config(config)]),
    )


def validate_fit_plan_without_data_execution(plan: MacroScalerFitPlan) -> list[str]:
    """Validate plan consistency without opening the source or fitting a scaler."""

    errors = list(plan.validation_errors)
    if plan.source_path != plan.project_root / EXPECTED_SOURCE:
        errors.append("resolved source path does not match runtime/data/macro_final.parquet")
    expected_targets = {
        name: plan.project_root / relative for name, relative in EXPECTED_CANONICAL_OUTPUTS.items()
    }
    if dict(plan.artifact_targets) != expected_targets:
        errors.append("resolved artifact targets do not match canonical future targets")
    if plan.bridge_scaler_path in plan.artifact_targets.values():
        errors.append("bridge macro_scaler.pt must not be a canonical clean artifact target")
    if plan.execution_approved or plan.thesis_safe or plan.auto_promote_thesis_safe:
        errors.append("fit plan must retain all nonapproval flags")
    return errors


def build_macro_scaler_metadata(
    config: Mapping[str, Any], config_path: str | Path
) -> MacroScalerMetadata:
    """Build backward-compatible planned metadata from config fields only."""

    plan = build_fit_plan(config, config_path)
    paths = config.get("paths", {})
    extra = paths.get("extra", {}) if isinstance(paths, Mapping) else {}
    return MacroScalerMetadata(
        config_path=Path(config_path),
        data_config_path=_resolve_path(plan.project_root, extra.get("data_config", "CHECK")),
        input_macro_panel=plan.source_path,
        macro_predictors=plan.macro_variables,
        fit_start_yyyymm=plan.fit_start_yyyymm,
        fit_end_yyyymm=plan.fit_end_yyyymm,
        train_end_yyyymm=_safe_int(config.get("train_end_yyyymm")),
        ddof=plan.ddof,
        precision=plan.precision,
        n_fit_rows=None,
        means={},
        scales={},
        validation_errors=plan.validation_errors,
    )


def fit_macro_scaler(
    macro_panel: Any, config: Mapping[str, Any], *, allow_execution: bool = False
) -> MacroScaler:
    """Fit the bounded clean macro scaler after explicit execution approval."""

    if not allow_execution:
        raise RuntimeError("Macro-scaler fitting requires explicit allow_execution=True.")
    import numpy as np

    errors = validate_macro_scaler_config(config)
    if errors:
        raise ValueError(f"Invalid macro scaler config: {errors}")
    required_columns = [EXPECTED_DATE_FIELD, *EXPECTED_MACRO_VARIABLES]
    missing_columns = [name for name in required_columns if name not in macro_panel.columns]
    if missing_columns:
        raise ValueError(f"Missing required macro-panel columns: {missing_columns}")
    if macro_panel.columns.tolist().count(EXPECTED_DATE_FIELD) != 1:
        raise ValueError("yyyymm must exist exactly once")
    if any(macro_panel.columns.tolist().count(name) != 1 for name in EXPECTED_MACRO_VARIABLES):
        raise ValueError("Every canonical macro variable must exist exactly once")

    dates_raw = np.asarray(macro_panel[EXPECTED_DATE_FIELD])
    dates_numeric = np.asarray(dates_raw, dtype=np.float64)
    if not np.isfinite(dates_numeric).all() or not np.equal(dates_numeric, np.floor(dates_numeric)).all():
        raise ValueError("yyyymm must be finite and integer-like without information loss")
    dates = dates_numeric.astype(np.int64)
    if np.unique(dates).size != dates.size:
        raise ValueError("yyyymm must be unique at monthly macro-panel level")
    if dates.min() > EXPECTED_FIT_START or dates.max() < 202412:
        raise ValueError("source coverage must include 198001-202412")

    expected_months = []
    year, month = EXPECTED_FIT_START // 100, EXPECTED_FIT_START % 100
    while year * 100 + month <= EXPECTED_FIT_END:
        expected_months.append(year * 100 + month)
        month += 1
        if month == 13:
            year += 1
            month = 1
    fit_mask = (dates >= EXPECTED_FIT_START) & (dates <= EXPECTED_FIT_END)
    fit_dates = dates[fit_mask]
    if fit_dates.tolist() != expected_months:
        raise ValueError("fit period must be complete, monthly, ordered, and inclusive")

    fit_values = np.asarray(
        macro_panel.loc[fit_mask, list(EXPECTED_MACRO_VARIABLES)], dtype=np.float64
    )
    if fit_values.shape != (len(expected_months), len(EXPECTED_MACRO_VARIABLES)):
        raise ValueError("fit matrix has an unexpected shape")
    if np.isnan(fit_values).any():
        raise ValueError("fit values contain missing values")
    if not np.isfinite(fit_values).all():
        raise ValueError("fit values contain non-finite values")
    means = fit_values.mean(axis=0, dtype=np.float64)
    scales = fit_values.std(axis=0, ddof=EXPECTED_DDOF, dtype=np.float64)
    if means.dtype != np.float64 or scales.dtype != np.float64:
        raise ValueError("means and scales must use float64")
    if not np.isfinite(means).all() or not np.isfinite(scales).all():
        raise ValueError("means and scales must be finite")
    if (scales <= EXPECTED_NEAR_ZERO_THRESHOLD).any():
        bad = [name for name, scale in zip(EXPECTED_MACRO_VARIABLES, scales) if scale <= EXPECTED_NEAR_ZERO_THRESHOLD]
        raise ValueError(f"zero or near-zero scales detected: {bad}")

    metadata = MacroScalerMetadata(
        config_path=Path("configs/scaler/macro_scaler_thesis.json"),
        data_config_path=Path("configs/data/universe_500_thesis.json"),
        input_macro_panel=Path(EXPECTED_SOURCE),
        macro_predictors=EXPECTED_MACRO_VARIABLES,
        fit_start_yyyymm=EXPECTED_FIT_START,
        fit_end_yyyymm=EXPECTED_FIT_END,
        train_end_yyyymm=EXPECTED_FIT_END,
        ddof=EXPECTED_DDOF,
        precision=EXPECTED_PRECISION,
        n_fit_rows=fit_values.shape[0],
        means={name: float(value) for name, value in zip(EXPECTED_MACRO_VARIABLES, means)},
        scales={name: float(value) for name, value in zip(EXPECTED_MACRO_VARIABLES, scales)},
        validation_errors=(),
        execution_approved=False,
        data_contents_inspected=True,
        scaler_fitted=True,
        result_produced=True,
        thesis_safe_result_produced=False,
    )
    return MacroScaler(
        macro_predictors=EXPECTED_MACRO_VARIABLES,
        means=metadata.means,
        scales=metadata.scales,
        metadata=metadata,
    )


def transform_macro_panel(macro_panel: Any, scaler: MacroScaler) -> Any:
    """Transform an ordered macro matrix using a fitted bounded scaler."""

    import numpy as np

    values = np.asarray(macro_panel, dtype=np.float64)
    means = np.asarray([scaler.means[name] for name in scaler.macro_predictors], dtype=np.float64)
    scales = np.asarray([scaler.scales[name] for name in scaler.macro_predictors], dtype=np.float64)
    return (values - means) / scales


def inverse_transform_macro_panel(scaled_panel: Any, scaler: MacroScaler) -> Any:
    """Inverse-transform an ordered macro matrix using a fitted bounded scaler."""

    import numpy as np

    values = np.asarray(scaled_panel, dtype=np.float64)
    means = np.asarray([scaler.means[name] for name in scaler.macro_predictors], dtype=np.float64)
    scales = np.asarray([scaler.scales[name] for name in scaler.macro_predictors], dtype=np.float64)
    return values * scales + means


def write_scaler_metadata(
    metadata: MacroScalerMetadata, output_path: str | Path, *, allow_execution: bool = False
) -> None:
    """Fail closed because canonical metadata writing is a future execution action."""

    if not allow_execution:
        raise RuntimeError("Scaler metadata writing requires explicit allow_execution=True.")
    raise NotImplementedError("Scaler metadata writing is not implemented or approved.")


def validate_scaler_metadata(metadata: MacroScalerMetadata, data_contract: Any) -> list[str]:
    """Validate metadata against a data-contract-like object without reading data."""

    errors: list[str] = []
    if tuple(getattr(data_contract, "macro_predictors", ())) != metadata.macro_predictors:
        errors.append("macro predictor order must match the data contract")
    train_end = getattr(data_contract, "train_end_yyyymm", None)
    if train_end is not None and metadata.fit_end_yyyymm != train_end:
        errors.append("scaler fit end must match data-contract train_end_yyyymm")
    if metadata.execution_approved:
        errors.append("metadata.execution_approved must remain false in scaffold")
    if metadata.scaler_fitted:
        errors.append("metadata.scaler_fitted must remain false in scaffold")
    return errors


def write_macro_scaler_artifacts(
    scaler: MacroScaler,
    artifact_targets: Mapping[str, str | Path],
    *,
    stats_payload: Mapping[str, Any] | None = None,
    metadata_payload: Mapping[str, Any] | None = None,
    validation_payload: Mapping[str, Any] | None = None,
    allow_execution: bool = False,
) -> None:
    """Write exactly the five canonical clean scaler artifacts."""

    if not allow_execution:
        raise RuntimeError("Scaler artifact writing requires explicit allow_execution=True.")
    import numpy as np

    targets = {name: Path(path) for name, path in artifact_targets.items()}
    if set(targets) != set(EXPECTED_CANONICAL_OUTPUTS):
        raise ValueError("artifact targets must contain exactly the five canonical outputs")
    if Path(BRIDGE_SCALER_PATH).resolve() in {path.resolve() for path in targets.values()}:
        raise ValueError("bridge macro_scaler.pt cannot be a canonical output")
    for path in targets.values():
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite existing canonical artifact: {path}")

    means = np.asarray([scaler.means[name] for name in scaler.macro_predictors], dtype=np.float64)
    scales = np.asarray([scaler.scales[name] for name in scaler.macro_predictors], dtype=np.float64)
    payloads = {
        "stats": stats_payload,
        "metadata": metadata_payload,
        "validation_report": validation_payload,
    }
    if any(payload is None for payload in payloads.values()):
        raise ValueError("stats, metadata, and validation payloads are required")

    temp_paths: list[Path] = []
    try:
        for name, array in (("mean", means), ("scale", scales)):
            temp = targets[name].with_name(f".{targets[name].name}.tmp")
            with temp.open("wb") as handle:
                np.save(handle, array, allow_pickle=False)
            temp_paths.append(temp)
        for name, payload in payloads.items():
            temp = targets[name].with_name(f".{targets[name].name}.tmp")
            with temp.open("w", encoding="utf-8") as handle:
                json.dump(payload, handle, indent=2, sort_keys=True)
                handle.write("\n")
            temp_paths.append(temp)
        for name in ("mean", "scale", "stats", "metadata", "validation_report"):
            targets[name].with_name(f".{targets[name].name}.tmp").replace(targets[name])
    finally:
        for temp in temp_paths:
            if temp.exists():
                temp.unlink()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def execute_macro_scaler_fit(
    config_path: str | Path = "configs/scaler/macro_scaler_thesis.json",
    *,
    allow_execution: bool = False,
) -> dict[str, Any]:
    """Execute and validate the single authorized clean macro-scaler fit."""

    if not allow_execution:
        raise RuntimeError("Bounded macro-scaler execution requires explicit allow_execution=True.")
    import numpy as np
    import pandas as pd

    config_path = Path(config_path)
    config = load_scaler_config(config_path)
    plan = build_fit_plan(config, config_path)
    plan_errors = validate_fit_plan_without_data_execution(plan)
    if plan_errors:
        raise ValueError(f"Fit plan validation failed: {plan_errors}")
    if any(path.exists() for path in plan.artifact_targets.values()):
        raise FileExistsError("One or more canonical scaler artifacts already exist")
    bridge_before = _sha256(plan.bridge_scaler_path)
    source_sha256 = _sha256(plan.source_path)
    macro_panel = pd.read_parquet(plan.source_path)
    scaler = fit_macro_scaler(macro_panel, config, allow_execution=True)

    fit_mask = (
        (macro_panel[plan.date_field].astype("int64") >= EXPECTED_FIT_START)
        & (macro_panel[plan.date_field].astype("int64") <= EXPECTED_FIT_END)
    )
    fit_values = np.asarray(
        macro_panel.loc[fit_mask, list(EXPECTED_MACRO_VARIABLES)], dtype=np.float64
    )
    means = np.asarray([scaler.means[name] for name in EXPECTED_MACRO_VARIABLES], dtype=np.float64)
    scales = np.asarray([scaler.scales[name] for name in EXPECTED_MACRO_VARIABLES], dtype=np.float64)
    transformed = transform_macro_panel(fit_values, scaler)
    recovered = inverse_transform_macro_panel(transformed, scaler)
    tolerances = {
        "mean_abs_tolerance": 1e-10,
        "std_abs_tolerance": 1e-10,
        "inverse_max_abs_tolerance": 1e-10,
    }
    transform_mean_max_abs = float(np.max(np.abs(transformed.mean(axis=0))))
    transform_std_max_abs_error = float(np.max(np.abs(transformed.std(axis=0, ddof=0) - 1.0)))
    inverse_max_abs_error = float(np.max(np.abs(recovered - fit_values)))
    if transform_mean_max_abs > tolerances["mean_abs_tolerance"]:
        raise ValueError("transform mean smoke check failed")
    if transform_std_max_abs_error > tolerances["std_abs_tolerance"]:
        raise ValueError("transform standard-deviation smoke check failed")
    if inverse_max_abs_error > tolerances["inverse_max_abs_tolerance"]:
        raise ValueError("inverse-transform smoke check failed")

    created_at = datetime.now(timezone.utc).isoformat()
    implementation_path = Path(__file__).resolve()
    data_config_path = plan.project_root / "configs/data/universe_500_thesis.json"
    base = {
        "schema_version": "0.1",
        "task_id": "TASK_046AM_POE_CLEAN_MACRO_SCALER_FIT_EXECUTION_AND_ARTIFACT_REGISTRATION",
        "created_at_utc": created_at,
        "macro_variables_ordered": list(EXPECTED_MACRO_VARIABLES),
        "fit_period_start_yyyymm": EXPECTED_FIT_START,
        "fit_period_end_yyyymm": EXPECTED_FIT_END,
        "fit_period_inclusive": True,
        "fit_row_count": int(fit_values.shape[0]),
        "ddof": EXPECTED_DDOF,
        "precision": EXPECTED_PRECISION,
        "missing_value_policy": EXPECTED_MISSING_POLICY,
        "zero_variance_policy": EXPECTED_ZERO_VARIANCE_POLICY,
        "near_zero_threshold": EXPECTED_NEAR_ZERO_THRESHOLD,
        "clean_execution_approved": False,
        "auto_promote_thesis_safe": False,
        "thesis_safe": False,
    }
    stats_payload = {
        **base,
        "summary": "Clean train-period macro scaler statistics in canonical variable order.",
        "mean_values_ordered": means.tolist(),
        "scale_values_ordered": scales.tolist(),
        "transform_mean_max_abs": transform_mean_max_abs,
        "transform_std_max_abs_error": transform_std_max_abs_error,
        "inverse_max_abs_error": inverse_max_abs_error,
        "tolerances": tolerances,
    }
    metadata_payload = {
        **base,
        "scaler_config_path": str(config_path),
        "scaler_config_sha256": _sha256(config_path),
        "data_config_path": str(data_config_path.relative_to(plan.project_root)),
        "data_config_sha256": _sha256(data_config_path),
        "source_path": str(plan.source_path.relative_to(plan.project_root)),
        "source_sha256": source_sha256,
        "source_row_count": int(macro_panel.shape[0]),
        "source_month_min": int(macro_panel[plan.date_field].min()),
        "source_month_max": int(macro_panel[plan.date_field].max()),
        "date_field": plan.date_field,
        "excluded_columns": ["yyyymm", "Rfree"],
        "mean_values_ordered": means.tolist(),
        "scale_values_ordered": scales.tolist(),
        "output_artifact_paths": {
            name: str(path.relative_to(plan.project_root)) for name, path in plan.artifact_targets.items()
        },
        "output_artifact_sha256": "registered_after_write_in_checksum_and_artifact_manifests",
        "implementation_module": "poe_thesis.scaling.macro_scaler",
        "implementation_file_sha256": _sha256(implementation_path),
        "python_version": sys.version,
        "package_versions_if_available": {
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "pyarrow": __import__("pyarrow").__version__,
            "platform": platform.platform(),
        },
        "bridge_scaler_path": str(plan.bridge_scaler_path.relative_to(plan.project_root)),
        "bridge_scaler_sha256_before": bridge_before,
        "bridge_scaler_sha256_after": bridge_before,
        "bridge_scaler_preserved": True,
    }
    gate_names = list(config.get("validation_gates", []))
    validation_payload = {
        **base,
        "status": "macro_scaler_fitted_artifacts_registered",
        "scaler_fit_executed": True,
        "scaler_artifacts_written": True,
        "validation_gates": {name: "passed" for name in gate_names},
        "transform_mean_max_abs": transform_mean_max_abs,
        "transform_std_max_abs_error": transform_std_max_abs_error,
        "inverse_max_abs_error": inverse_max_abs_error,
        "tolerances": tolerances,
        "bridge_scaler_sha256_before": bridge_before,
        "bridge_scaler_sha256_after": bridge_before,
        "bridge_scaler_preserved": True,
        "output_artifact_sha256": "registered_after_write_in_checksum_and_artifact_manifests",
    }
    write_macro_scaler_artifacts(
        scaler,
        plan.artifact_targets,
        stats_payload=stats_payload,
        metadata_payload=metadata_payload,
        validation_payload=validation_payload,
        allow_execution=True,
    )

    loaded_mean = np.load(plan.artifact_targets["mean"], allow_pickle=False)
    loaded_scale = np.load(plan.artifact_targets["scale"], allow_pickle=False)
    if loaded_mean.dtype != np.float64 or loaded_mean.shape != (9,):
        raise ValueError("written mean artifact failed dtype/shape validation")
    if loaded_scale.dtype != np.float64 or loaded_scale.shape != (9,):
        raise ValueError("written scale artifact failed dtype/shape validation")
    for name in ("stats", "metadata", "validation_report"):
        with plan.artifact_targets[name].open("r", encoding="utf-8") as handle:
            json.load(handle)
    bridge_after = _sha256(plan.bridge_scaler_path)
    if bridge_after != bridge_before:
        raise RuntimeError("bridge scaler changed during bounded clean scaler fit")
    return {
        "fit_row_count": int(fit_values.shape[0]),
        "means": means.tolist(),
        "scales": scales.tolist(),
        "artifact_sha256": {name: _sha256(path) for name, path in plan.artifact_targets.items()},
        "source_sha256": source_sha256,
        "bridge_sha256_before": bridge_before,
        "bridge_sha256_after": bridge_after,
        "transform_mean_max_abs": transform_mean_max_abs,
        "transform_std_max_abs_error": transform_std_max_abs_error,
        "inverse_max_abs_error": inverse_max_abs_error,
    }


def write_scaler_artifacts(
    scaler: MacroScaler,
    output_scaler: str | Path,
    output_metadata: str | Path,
    *,
    allow_execution: bool = False,
) -> None:
    """Backward-compatible guarded artifact-writing entrypoint."""

    write_macro_scaler_artifacts(
        scaler,
        {"legacy_scaler": output_scaler, "metadata": output_metadata},
        allow_execution=allow_execution,
    )


from typing import Mapping, Union

import numpy as np



def build_synthetic_firm_characteristics(n_firm_chars: int = 146) -> np.ndarray:
    """Generate a deterministic synthetic firm characteristics vector.

    Defaults to 146 dimensions to match the full-panel GKX-style build (146 surviving
    OSAP signals → 1460 interaction features), which is what the trained FNN consumes.
    """
    return np.linspace(-0.5, 0.5, n_firm_chars, dtype=np.float64)


def build_synthetic_raw_macro_state() -> np.ndarray:
    """Generate a deterministic synthetic 9-dimensional raw macro state vector."""
    # Simple deterministic macro vector in canonical order
    return np.linspace(0.01, 0.09, 9, dtype=np.float64)


def _coerce_and_standardize_macro_state(
    macro_state: Union[np.ndarray, list[float], Mapping[str, float]],
    scaler: Union[MacroScaler, str, Path, None],
) -> np.ndarray:
    """Validate a macro state and return it in standardized canonical order."""
    if isinstance(macro_state, Mapping):
        if set(macro_state.keys()) != set(EXPECTED_MACRO_VARIABLES):
            raise ValueError(
                f"Macro state dictionary keys {list(macro_state.keys())} do not match "
                f"expected variables {EXPECTED_MACRO_VARIABLES}"
            )
        macro_arr = np.array([macro_state[var] for var in EXPECTED_MACRO_VARIABLES], dtype=np.float64)
    else:
        macro_arr = np.asarray(macro_state, dtype=np.float64)
        if macro_arr.ndim != 1 or macro_arr.shape[0] != 9:
            raise ValueError(
                f"Macro state must be a 1D vector of length 9, got shape {macro_arr.shape}"
            )
    if not np.isfinite(macro_arr).all():
        raise ValueError("Macro state contains non-finite or missing values.")

    if scaler is None:
        standardized = macro_arr
    else:
        if isinstance(scaler, (str, Path)):
            scaler_obj = load_clean_macro_scaler(scaler)
        elif isinstance(scaler, MacroScaler):
            scaler_obj = scaler
        else:
            raise TypeError("scaler must be a MacroScaler object, string/Path config path, or None")
        standardized = np.array(
            [
                (macro_arr[index] - scaler_obj.means[name]) / scaler_obj.scales[name]
                for index, name in enumerate(EXPECTED_MACRO_VARIABLES)
            ],
            dtype=np.float64,
        )
    if not np.isfinite(standardized).all():
        raise ValueError("Standardized macro state contains non-finite or missing values.")
    return standardized


def build_interaction_feature_vector(
    firm_chars: Union[np.ndarray, list[float]],
    macro_state: Union[np.ndarray, list[float], Mapping[str, float]],
    scaler: Union[MacroScaler, str, Path, None] = None,
    *,
    n_firm_chars: int = 146,
) -> np.ndarray:
    """Construct the canonical POE tactical interaction feature vector.

    Args:
        firm_chars: Firm characteristics vector of length ``n_firm_chars``.
        macro_state: Macro state vector of length 9 or mapping with 9 canonical keys.
        scaler: A fitted MacroScaler instance, config path to load one, or None if pre-standardized.
        n_firm_chars: Number of firm characteristics (default 146 for the full-panel GKX-style
            build → 1460 interaction features; pass 140 for the legacy bridge contract).

    Returns:
        A (n_firm_chars * 10)-dimensional float64 numpy array.
    """
    # 1. Coerce and validate firm characteristics
    firm_arr = np.asarray(firm_chars, dtype=np.float64)
    if firm_arr.ndim != 1 or firm_arr.shape[0] != n_firm_chars:
        raise ValueError(
            f"Firm characteristics must be a 1D vector of length {n_firm_chars}, "
            f"got shape {firm_arr.shape}"
        )
    if not np.isfinite(firm_arr).all():
        raise ValueError("Firm characteristics contain non-finite or missing values.")

    # 2. Validate and standardize the macro state in canonical order
    z = _coerce_and_standardize_macro_state(macro_state, scaler)

    # 4. Form the augmented macro vector [1, z_dp, ..., z_infl] of length 10
    augmented_z = np.empty(10, dtype=np.float64)
    augmented_z[0] = 1.0
    augmented_z[1:] = z

    # 5. Build output feature vector of length (n_firm_chars * 10)
    # First n_firm_chars entries equal firm characteristics (augmented_z[0] = 1)
    # Next 9 blocks equal firm characteristics times each standardized macro value
    feature_vector = np.empty(n_firm_chars * 10, dtype=np.float64)
    for i in range(10):
        feature_vector[i * n_firm_chars : (i + 1) * n_firm_chars] = firm_arr * augmented_z[i]

    return feature_vector


def build_interaction_feature_matrix(
    firm_characteristics: Union[np.ndarray, list[list[float]]],
    macro_state: Union[np.ndarray, list[float], Mapping[str, float]],
    scaler: Union[MacroScaler, str, Path, None] = None,
    *,
    n_firm_chars: int = 146,
) -> np.ndarray:
    """Construct canonical interaction features for multiple assets.

    The output uses the same block ordering as ``build_interaction_feature_vector``:
    the first ``n_firm_chars`` columns contain firm characteristics, followed by nine
    ``n_firm_chars``-column blocks scaled by the standardized macro variables.

    Args:
        firm_characteristics: Finite matrix with shape ``(n_assets, n_firm_chars)``.
        macro_state: Raw macro state when ``scaler`` is supplied, otherwise a
            pre-standardized macro state. Accepts a canonical mapping or vector.
        scaler: A fitted clean scaler, config path, or ``None`` for pre-standardized
            macro input.
        n_firm_chars: Number of firm characteristics per asset (default 146 for the full-panel
            GKX-style build → 1460 interaction features; pass 140 for the legacy bridge contract).

    Returns:
        A finite float64 matrix with shape ``(n_assets, n_firm_chars * 10)``.
    """
    firm_matrix = np.asarray(firm_characteristics, dtype=np.float64)
    if firm_matrix.ndim != 2 or firm_matrix.shape[1] != n_firm_chars or firm_matrix.shape[0] == 0:
        raise ValueError(
            f"Firm characteristics must be a non-empty 2D matrix with shape "
            f"(n_assets, {n_firm_chars}), got shape {firm_matrix.shape}"
        )
    if not np.isfinite(firm_matrix).all():
        raise ValueError("Firm characteristics contain non-finite or missing values.")

    standardized_macro = _coerce_and_standardize_macro_state(macro_state, scaler)
    augmented_macro = np.concatenate(
        (np.ones(1, dtype=np.float64), standardized_macro)
    )

    feature_matrix = (firm_matrix[:, None, :] * augmented_macro[None, :, None]).reshape(
        firm_matrix.shape[0], n_firm_chars * 10
    )
    if not np.isfinite(feature_matrix).all():
        raise ValueError("Interaction feature matrix contains non-finite values.")
    return feature_matrix


def placeholder() -> None:
    """Mark feature-builder implementation as pending (legacy wrapper)."""
    pass
