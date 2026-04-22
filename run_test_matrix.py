"""Сводит сравнение версий ИИС по датасетам в общую таблицу."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from config.settings import OUTPUT_DIR


DETAILED_OUT = OUTPUT_DIR / "test_matrix_detailed.csv"
BEST_OUT = OUTPUT_DIR / "test_matrix_best_by_dataset.csv"


def parse_dataset_mode_from_name(path: Path, prefix: str) -> tuple[str, str]:
    stem = path.stem
    suffix = stem[len(prefix) :]
    for mode in ("strict", "proxy", "hybrid"):
        tail = f"_{mode}"
        if suffix.endswith(tail):
            return suffix[: -len(tail)], mode
    raise ValueError(f"Не удалось выделить dataset/mode из имени {path.name}")


def load_state_capacity() -> dict[tuple[str, str, str], dict[str, Any]]:
    state_map: dict[tuple[str, str, str], dict[str, Any]] = {}
    for path in sorted(OUTPUT_DIR.glob("state_capacity_*_*.json")):
        dataset, mode = parse_dataset_mode_from_name(path, prefix="state_capacity_")
        data = json.loads(path.read_text(encoding="utf-8"))
        for version, payload in data.get("versions", {}).items():
            static_payload = payload.get("static", {})
            state_map[(dataset, mode, version)] = {
                "labeled_state_count": static_payload.get("labeled_state_count"),
                "pairwise_distinguishable_pairs": static_payload.get("pairwise_distinguishable_pairs"),
                "largest_fully_distinguishable_label_set": static_payload.get("largest_fully_distinguishable_label_set"),
                "confident_state_count": static_payload.get("confident_state_count"),
                "likely_state_count": static_payload.get("likely_state_count"),
                "consensus_supported_k": static_payload.get("consensus_supported_k"),
                "consensus_support_strength": static_payload.get("consensus_support_strength"),
                "capacity_verdict": static_payload.get("capacity_verdict"),
                "capacity_note": static_payload.get("capacity_note"),
                "label_alignment_ari": static_payload.get("label_alignment_ari"),
                "label_alignment_nmi": static_payload.get("label_alignment_nmi"),
                "va_alignment_ari": static_payload.get("va_alignment_ari"),
                "va_alignment_nmi": static_payload.get("va_alignment_nmi"),
            }
    return state_map


def build_detailed_table() -> pd.DataFrame:
    state_capacity = load_state_capacity()
    records: list[dict[str, Any]] = []

    for path in sorted(OUTPUT_DIR.glob("comparison_*_*.csv")):
        dataset, mode = parse_dataset_mode_from_name(path, prefix="comparison_")
        comparison_df = pd.read_csv(path)
        for row in comparison_df.to_dict(orient="records"):
            version = row["version"]
            effect_size = row.get("effect_size")
            overlap = row.get("distribution_overlap")
            utility_score = row.get("utility_score")
            valid_segments = row.get("valid_segments")
            reliability = row.get("reliability_level")
            coverage = row.get("coverage")
            effect_abs = abs(float(effect_size)) if pd.notna(effect_size) else np.nan
            inverse_overlap = 1.0 - float(overlap) if pd.notna(overlap) else np.nan
            separation_proxy = (
                0.6 * effect_abs + 0.4 * inverse_overlap
                if pd.notna(effect_abs) and pd.notna(inverse_overlap)
                else np.nan
            )

            merged = {
                "dataset": dataset,
                "mode": mode,
                "version": version,
                "valid_segments": valid_segments,
                "coverage": coverage,
                "reliability_level": reliability,
                "mean_iis": row.get("mean_iis"),
                "std_iis": row.get("std_iis"),
                "stress_baseline_diff": row.get("stress_baseline_diff"),
                "effect_size": effect_size,
                "effect_abs": effect_abs,
                "distribution_overlap": overlap,
                "inverse_overlap": inverse_overlap,
                "arousal_correlation": row.get("arousal_correlation"),
                "valence_correlation": row.get("valence_correlation"),
                "relative_sensitivity": row.get("relative_sensitivity"),
                "direct_ratio": row.get("direct_ratio"),
                "utility_score": utility_score,
                "utility_rank": row.get("utility_rank"),
                "oversmoothing_flag": row.get("oversmoothing_flag"),
                "separation_proxy": separation_proxy,
            }
            merged.update(state_capacity.get((dataset, mode, version), {}))
            records.append(merged)

    detailed_df = pd.DataFrame.from_records(records)
    if detailed_df.empty:
        return detailed_df

    detailed_df["is_usable"] = (
        detailed_df["coverage"].fillna(0.0) > 0
    ) & detailed_df["valid_segments"].fillna(0).astype(float).gt(0)
    return detailed_df.sort_values(
        ["dataset", "mode", "utility_rank", "utility_score", "effect_abs"],
        ascending=[True, True, True, False, False],
    ).reset_index(drop=True)


def build_best_table(detailed_df: pd.DataFrame) -> pd.DataFrame:
    if detailed_df.empty:
        return detailed_df

    usable_df = detailed_df[detailed_df["is_usable"]].copy()
    if usable_df.empty:
        return usable_df

    reliability_order = {"high": 0, "medium": 1, "low": 2}
    usable_df["reliability_order"] = usable_df["reliability_level"].map(reliability_order).fillna(3)
    usable_df = usable_df.sort_values(
        [
            "dataset",
            "mode",
            "reliability_order",
            "utility_rank",
            "utility_score",
            "effect_abs",
            "inverse_overlap",
        ],
        ascending=[True, True, True, True, False, False, False],
    )

    best_rows = usable_df.groupby(["dataset", "mode"], as_index=False).first()
    return best_rows[
        [
            "dataset",
            "mode",
            "version",
            "valid_segments",
            "reliability_level",
            "utility_rank",
            "utility_score",
            "effect_abs",
            "distribution_overlap",
            "relative_sensitivity",
            "confident_state_count",
            "likely_state_count",
            "consensus_supported_k",
            "capacity_verdict",
            "oversmoothing_flag",
        ]
    ].sort_values(["dataset", "mode"]).reset_index(drop=True)


def main() -> int:
    detailed_df = build_detailed_table()
    best_df = build_best_table(detailed_df)

    DETAILED_OUT.parent.mkdir(parents=True, exist_ok=True)
    detailed_df.to_csv(DETAILED_OUT, index=False, encoding="utf-8-sig")
    best_df.to_csv(BEST_OUT, index=False, encoding="utf-8-sig")

    print(f"detailed_rows={len(detailed_df)}")
    print(f"best_rows={len(best_df)}")
    print(f"detailed_out={DETAILED_OUT}")
    print(f"best_out={BEST_OUT}")
    if not best_df.empty:
        print(best_df.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
