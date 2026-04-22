"""Собирает единый bundle с метриками, совместимостью и экспериментами для IIS-отчёта."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from reporting.iis_report_common import (
    BUNDLE_CSV_PATH,
    BUNDLE_JSON_PATH,
    DATA_MODE_EXPLANATIONS,
    DYNAMIC_SCENARIOS,
    EXPERIMENT_FILES,
    SCENARIOS,
    VERSION_METADATA,
    VERSION_ORDER,
    VERSION_SHORT,
    build_diagram_specs,
    format_float,
    format_int,
    scenario_filename_slug,
    scenario_key,
)


def _load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _load_csv(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    return pd.read_csv(path, low_memory=False)


def _record_status_bucket(version: str, row: dict[str, Any]) -> str:
    if row.get("availability_status") == "not_available":
        return "danger"
    if row.get("reliability_level") == "high" and row.get("coverage", 0.0) and not row.get("oversmoothing_flag", False):
        return "success"
    if version in {"IISVersion1", "IISVersion2", "IISVersion3"}:
        return "danger"
    return "warning"


def _read_dynamic_files(dataset: str, mode: str) -> dict[str, dict[str, str]]:
    mapping: dict[str, dict[str, str]] = {}
    base_glob = Path("outputs").glob(f"dynamic_iis_{dataset}_{mode}_*.csv")
    for path in sorted(base_glob):
        lower_name = path.name.lower()
        version_name = None
        for candidate in VERSION_ORDER:
            if candidate.lower() in lower_name:
                version_name = candidate
                break
        if version_name is None:
            continue
        entry = mapping.setdefault(version_name, {})
        if lower_name.endswith("_causal.csv"):
            entry["dynamic_causal_csv"] = str(path.resolve())
        else:
            entry["dynamic_csv"] = str(path.resolve())
    return mapping


def _build_compatibility_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for version in VERSION_ORDER:
        metadata = VERSION_METADATA[version]
        rows.append(
            {
                "version": version,
                "short": metadata["short"],
                "title": metadata["title"],
                "era": metadata["era"],
                "static_supported": metadata["static_supported"],
                "dynamic_supported": metadata["dynamic_supported"],
                "proxy_supported": metadata["proxy_supported"],
                "hybrid_supported": metadata["hybrid_supported"],
                "status_note": metadata["status_note"],
                "limitation_flags": metadata["limitation_flags"],
            }
        )
    return rows


def _get_state_payload(data: dict[str, Any] | None, version: str, regime: str = "static") -> dict[str, Any]:
    if not data:
        return {}
    return (data.get("versions", {}).get(version, {}) or {}).get(regime, {}) or {}


def _build_dynamic_support(version: str, dataset: str, mode: str, dynamic_files: dict[str, dict[str, str]]) -> dict[str, Any]:
    required_dynamic_keys = {scenario_key(item["dataset"], item["mode"]) for item in DYNAMIC_SCENARIOS}
    scenario_name = scenario_key(dataset, mode)
    files = dynamic_files.get(version, {})
    if version in {"IISVersion1", "IISVersion2", "IISVersion3"}:
        return {
            "dynamic_state": "not_adapted",
            "dynamic_note": "Версия не адаптирована для динамического сравнения.",
            "dynamic_files": files,
        }
    if files:
        return {
            "dynamic_state": "available",
            "dynamic_note": "Есть сохранённые динамические артефакты.",
            "dynamic_files": files,
        }
    if scenario_name in required_dynamic_keys:
        return {
            "dynamic_state": "missing_required",
            "dynamic_note": "Для этого сценария динамический прогон ожидался, но файлы не найдены.",
            "dynamic_files": files,
        }
    return {
        "dynamic_state": "partial_optional",
        "dynamic_note": "Динамика для этого сценария не является обязательной частью пакета.",
        "dynamic_files": files,
    }


def _build_rows_for_scenario(scenario: dict[str, str]) -> list[dict[str, Any]]:
    dataset = scenario["dataset"]
    mode = scenario["mode"]
    slug = scenario_filename_slug(dataset, mode)
    comparison_path = Path("outputs") / f"comparison_{slug}.csv"
    summary_path = Path("outputs") / f"summary_{slug}.json"
    capacity_path = Path("outputs") / f"state_capacity_{slug}.json"

    comparison_df = _load_csv(comparison_path)
    summary_data = _load_json(summary_path)
    capacity_data = _load_json(capacity_path)
    dynamic_files = _read_dynamic_files(dataset, mode)

    comparison_map: dict[str, dict[str, Any]] = {}
    if comparison_df is not None and not comparison_df.empty:
        comparison_map = {
            str(row["version"]): row for row in comparison_df.to_dict(orient="records")
        }

    rows: list[dict[str, Any]] = []
    missing_parts = [
        name
        for name, condition in (
            ("comparison", comparison_df is None),
            ("summary", summary_data is None),
            ("state_capacity", capacity_data is None),
        )
        if condition
    ]
    availability_status = "ok" if not missing_parts else "not_available"
    availability_reason = "полный набор файлов найден" if not missing_parts else f"не найдены: {', '.join(missing_parts)}"

    for version in VERSION_ORDER:
        metadata = VERSION_METADATA[version]
        comparison_row = comparison_map.get(version, {})
        summary_version = ((summary_data or {}).get("versions", {}) or {}).get(version, {})
        state_static = _get_state_payload(capacity_data, version, regime="static")
        state_dynamic = _get_state_payload(capacity_data, version, regime="dynamic")
        dynamic_support = _build_dynamic_support(version, dataset, mode, dynamic_files)

        coverage = comparison_row.get("coverage")
        valid_segments = comparison_row.get("valid_segments")
        row: dict[str, Any] = {
            "dataset": dataset,
            "mode": mode,
            "scenario_key": scenario_key(dataset, mode),
            "scenario_label": scenario["label"],
            "version": version,
            "version_short": VERSION_SHORT[version],
            "version_title": metadata["title"],
            "version_era": metadata["era"],
            "availability_status": availability_status,
            "availability_reason": availability_reason,
            "coverage": coverage,
            "valid_segments": valid_segments,
            "effect_size": comparison_row.get("effect_size"),
            "distribution_overlap": comparison_row.get("distribution_overlap"),
            "relative_sensitivity": comparison_row.get("relative_sensitivity"),
            "utility_rank": comparison_row.get("utility_rank"),
            "utility_score": comparison_row.get("utility_score"),
            "reliability_level": comparison_row.get("reliability_level"),
            "oversmoothing_flag": comparison_row.get("oversmoothing_flag"),
            "capacity_verdict": state_static.get("capacity_verdict"),
            "capacity_note": state_static.get("capacity_note"),
            "consensus_supported_k": state_static.get("consensus_supported_k"),
            "confident_state_count": state_static.get("confident_state_count"),
            "likely_state_count": state_static.get("likely_state_count"),
            "labeled_state_count": state_static.get("labeled_state_count"),
            "pairwise_distinguishable_pairs": state_static.get("pairwise_distinguishable_pairs"),
            "largest_fully_distinguishable_label_set": state_static.get("largest_fully_distinguishable_label_set"),
            "dynamic_capacity_verdict": state_dynamic.get("capacity_verdict"),
            "dynamic_consensus_supported_k": state_dynamic.get("consensus_supported_k"),
            "dynamic_confident_state_count": state_dynamic.get("confident_state_count"),
            "dynamic_likely_state_count": state_dynamic.get("likely_state_count"),
            "dynamic_state": dynamic_support["dynamic_state"],
            "dynamic_note": dynamic_support["dynamic_note"],
            "dynamic_csv": dynamic_support["dynamic_files"].get("dynamic_csv"),
            "dynamic_causal_csv": dynamic_support["dynamic_files"].get("dynamic_causal_csv"),
            "direct_ratio": comparison_row.get("direct_ratio"),
            "mean_iis": comparison_row.get("mean_iis"),
            "std_iis": comparison_row.get("std_iis"),
            "stress_baseline_diff": comparison_row.get("stress_baseline_diff"),
            "arousal_correlation": comparison_row.get("arousal_correlation"),
            "valence_correlation": comparison_row.get("valence_correlation"),
            "status_note": metadata["status_note"],
            "limitation_flags": metadata["limitation_flags"],
            "feature_availability": (summary_data or {}).get("feature_availability", {}),
            "summary_limitations": (summary_data or {}).get("limitations", []),
            "status_bucket": "",
            "supports_static": metadata["static_supported"],
            "supports_dynamic": metadata["dynamic_supported"],
            "supports_proxy": metadata["proxy_supported"],
            "supports_hybrid": metadata["hybrid_supported"],
            "version_formula_latex": metadata["formula_latex"],
            "version_purpose": metadata["purpose"],
            "version_strength": metadata["strength"],
            "version_weakness": metadata["weakness"],
            "version_universality": metadata["universality"],
            "summary_keys_present": sorted(summary_version.keys()),
            "comparison_path": str(comparison_path.resolve()) if comparison_path.exists() else None,
            "summary_path": str(summary_path.resolve()) if summary_path.exists() else None,
            "state_capacity_path": str(capacity_path.resolve()) if capacity_path.exists() else None,
        }
        row["status_bucket"] = _record_status_bucket(version, row)
        rows.append(row)
    return rows


def _scenario_best_rows(rows_df: pd.DataFrame) -> list[dict[str, Any]]:
    usable = rows_df.copy()
    usable = usable[usable["coverage"].fillna(0).astype(float) > 0]
    if usable.empty:
        return []
    reliability_order = {"high": 0, "medium": 1, "low": 2}
    usable["reliability_order"] = usable["reliability_level"].map(reliability_order).fillna(3)
    usable["effect_abs"] = usable["effect_size"].astype(float).abs()
    usable["inverse_overlap"] = 1.0 - usable["distribution_overlap"].astype(float)
    usable = usable.sort_values(
        [
            "scenario_key",
            "reliability_order",
            "utility_rank",
            "utility_score",
            "effect_abs",
            "inverse_overlap",
        ],
        ascending=[True, True, True, False, False, False],
    )
    winners = usable.groupby("scenario_key", as_index=False).first()
    winners = winners[
        [
            "scenario_key",
            "scenario_label",
            "version",
            "version_short",
            "reliability_level",
            "utility_rank",
            "utility_score",
            "effect_size",
            "distribution_overlap",
            "relative_sensitivity",
            "capacity_verdict",
            "consensus_supported_k",
            "confident_state_count",
            "likely_state_count",
            "oversmoothing_flag",
        ]
    ]
    return winners.to_dict(orient="records")


def _state_confirmation_rows(rows_df: pd.DataFrame) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for scenario in SCENARIOS:
        subset = rows_df.loc[rows_df["scenario_key"] == scenario_key(scenario["dataset"], scenario["mode"])].copy()
        subset = subset[subset["coverage"].fillna(0).astype(float) > 0]
        if subset.empty:
            records.append(
                {
                    "scenario_label": scenario["label"],
                    "best_version": "—",
                    "confirmed_states": "not available",
                    "possible_ceiling": "—",
                    "note": "Нет пригодных результатов для сравнения.",
                }
            )
            continue
        subset["score"] = (
            subset["utility_score"].fillna(0).astype(float)
            + 0.05 * subset["confident_state_count"].fillna(0).astype(float)
            + 0.02 * subset["likely_state_count"].fillna(0).astype(float)
        )
        best = subset.sort_values(["score", "utility_rank"], ascending=[False, True]).iloc[0]
        records.append(
            {
                "scenario_label": scenario["label"],
                "best_version": best["version_short"],
                "confirmed_states": best.get("capacity_verdict") or "—",
                "possible_ceiling": f"k≈{format_int(best.get('consensus_supported_k'))}",
                "note": best.get("capacity_note") or "—",
            }
        )
    return records


def _bundle_experiments() -> dict[str, Any]:
    experiments: dict[str, Any] = {}

    v5_payload = _load_json(EXPERIMENT_FILES["v5_dreamer_calibration"])
    if v5_payload:
        experiments["dreamer_v5_calibration"] = {
            "dataset": v5_payload.get("dataset"),
            "version": v5_payload.get("version"),
            "objective_current_global": (v5_payload.get("baseline_current_global") or {}).get("objective_mean"),
            "objective_domain_recalibrated": (v5_payload.get("baseline_domain_recalibrated") or {}).get("objective_mean"),
            "objective_best_candidate": (v5_payload.get("best_candidate_cv") or {}).get("objective_mean"),
            "recommended_params": v5_payload.get("recommended_v5_params", {}),
            "note": "Калибровка V5 под DREAMER показывает, что часть выигрыша достигается именно доменной подстройкой нелинейных параметров.",
        }

    v7_payload = _load_json(EXPERIMENT_FILES["v7_dreamer_calibration"])
    if v7_payload:
        experiments["dreamer_v7_calibration"] = {
            "dataset": v7_payload.get("dataset"),
            "version": v7_payload.get("version"),
            "candidate_mode": v7_payload.get("candidate_mode"),
            "objective_current_global": (v7_payload.get("baseline_current_global") or {}).get("objective_mean"),
            "objective_domain_recalibrated": (v7_payload.get("baseline_domain_recalibrated") or {}).get("objective_mean"),
            "objective_best_candidate": (v7_payload.get("best_candidate_cv") or {}).get("objective_mean"),
            "recommended_params": v7_payload.get("recommended_v7_params", {}),
            "note": "Для V7 beta зафиксирован отдельный Dreamer-specific override; это улучшение локально, но не делает модель глобально универсальной.",
        }

    synthetic_summary = _load_json(EXPERIMENT_FILES["sumigron_synthetic_summary"])
    synthetic_metrics = _load_csv(EXPERIMENT_FILES["sumigron_synthetic_metrics"])
    if synthetic_summary and synthetic_metrics is not None and not synthetic_metrics.empty:
        def metric_row(scenario_name: str, method_name: str) -> dict[str, Any]:
            row_df = synthetic_metrics.loc[
                (synthetic_metrics["scenario"].astype(str) == scenario_name)
                & (synthetic_metrics["method"].astype(str) == method_name)
            ]
            if row_df.empty:
                return {}
            return row_df.iloc[0].to_dict()

        experiments["sumigron_synthetic"] = {
            "blueprints": synthetic_summary.get("method_blueprints", {}),
            "mean_driven_sumigron": metric_row("mean_driven", "sumigron"),
            "mean_driven_sigmoid": metric_row("mean_driven", "sigmoid_mean"),
            "structure_driven_sumigron": metric_row("structure_driven", "sumigron"),
            "structure_driven_attentive": metric_row("structure_driven", "sumigron_attentive"),
            "structure_driven_sigmoid": metric_row("structure_driven", "sigmoid_mean"),
            "structure_driven_softmax": metric_row("structure_driven", "softmax_weighted_sum"),
            "note": "Synthetic-бенч показывает, что Sumigron особенно силён там, где состояния различаются не только средним уровнем, но и структурой окна.",
        }

    return experiments


def build_bundle() -> dict[str, Any]:
    rows = []
    for scenario in SCENARIOS:
        rows.extend(_build_rows_for_scenario(scenario))
    rows_df = pd.DataFrame(rows)
    compatibility_rows = _build_compatibility_rows()
    best_rows = _scenario_best_rows(rows_df)
    state_rows = _state_confirmation_rows(rows_df)
    experiments = _bundle_experiments()

    strongest_strict = None
    strict_usable = rows_df.loc[
        (rows_df["mode"].astype(str) == "strict") & (rows_df["coverage"].fillna(0).astype(float) > 0)
    ].copy()
    if not strict_usable.empty:
        strict_usable["effect_abs"] = strict_usable["effect_size"].astype(float).abs()
        strict_usable = strict_usable.sort_values(
            ["effect_abs", "utility_score", "reliability_level"],
            ascending=[False, False, True],
        )
        strongest_strict = strict_usable.iloc[0][["scenario_label", "version", "effect_size", "distribution_overlap", "capacity_verdict"]].to_dict()

    bundle = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "report_title": "IIS Full Test + Documentation Pack",
        "versions": [
            {
                "name": version,
                "short": VERSION_METADATA[version]["short"],
                "title": VERSION_METADATA[version]["title"],
                "era": VERSION_METADATA[version]["era"],
                "formula_latex": VERSION_METADATA[version]["formula_latex"],
                "purpose": VERSION_METADATA[version]["purpose"],
                "inputs": VERSION_METADATA[version]["inputs"],
                "strength": VERSION_METADATA[version]["strength"],
                "weakness": VERSION_METADATA[version]["weakness"],
                "universality": VERSION_METADATA[version]["universality"],
                "status_note": VERSION_METADATA[version]["status_note"],
                "limitation_flags": VERSION_METADATA[version]["limitation_flags"],
                "status_bucket": _record_status_bucket(
                    version,
                    {"availability_status": "ok", "reliability_level": "high", "coverage": 1.0, "oversmoothing_flag": False},
                ),
            }
            for version in VERSION_ORDER
        ],
        "scenarios": SCENARIOS,
        "mode_explanations": DATA_MODE_EXPLANATIONS,
        "compatibility": compatibility_rows,
        "scenario_rows": rows_df.to_dict(orient="records"),
        "scenario_best": best_rows,
        "state_confirmation": state_rows,
        "experiments": experiments,
        "diagrams": build_diagram_specs(),
        "overview": {
            "current_best_practical_version": "IISVersion6",
            "current_best_theoretical_line": "IISVersion7 + Sumigron",
            "strongest_real_scenario": strongest_strict,
            "dreamer_note": "DREAMER полезен для выхода за грубые два режима, но пока остаётся слабым по подтверждённому числу состояний.",
            "sumigron_note": "Sumigron остаётся экспериментальным оператором: польза уже видна на synthetic и части V7-прогонов, но это ещё не финальная доказанная основа всей системы.",
        },
    }
    return bundle


def main() -> int:
    bundle = build_bundle()
    BUNDLE_JSON_PATH.parent.mkdir(parents=True, exist_ok=True)
    BUNDLE_JSON_PATH.write_text(json.dumps(bundle, ensure_ascii=False, indent=2), encoding="utf-8")

    rows_df = pd.DataFrame(bundle["scenario_rows"]).copy()
    if not rows_df.empty:
        rows_df["coverage_fmt"] = rows_df["coverage"].apply(format_float)
        rows_df["effect_size_fmt"] = rows_df["effect_size"].apply(format_float)
        rows_df["overlap_fmt"] = rows_df["distribution_overlap"].apply(format_float)
        rows_df["utility_score_fmt"] = rows_df["utility_score"].apply(format_float)
        rows_df["capacity_k_fmt"] = rows_df["consensus_supported_k"].apply(format_int)
    rows_df.to_csv(BUNDLE_CSV_PATH, index=False, encoding="utf-8-sig")

    print(f"bundle_json={BUNDLE_JSON_PATH}")
    print(f"bundle_csv={BUNDLE_CSV_PATH}")
    print(f"scenario_rows={len(bundle['scenario_rows'])}")
    print(f"scenario_best={len(bundle['scenario_best'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
