"""Псевдо-причинные интервенции над признаками и компонентами ИИС."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from models.dynamic_analysis import CausalDynamicAnalyzer
from models.iis_v1 import IISVersion1
from models.iis_v2 import IISVersion2
from models.iis_v3 import IISVersion3
from models.iis_v4 import IISVersion4
from models.iis_v5 import IISVersion5
from models.iis_v6 import IISVersion6


class IISInterventionSimulator:
    """Позволяет возмущать выбранный признак на интервале времени и пересчитывать динамику."""

    def __init__(self, output_dir: Path, plots_dir: Path) -> None:
        self.output_dir = Path(output_dir)
        self.plots_dir = Path(plots_dir)
        self.dynamic_analyzer = CausalDynamicAnalyzer(output_dir=output_dir, plots_dir=plots_dir)
        self.model_map = {
            "IISVersion1": IISVersion1,
            "IISVersion2": IISVersion2,
            "IISVersion3": IISVersion3,
            "IISVersion4": IISVersion4,
            "IISVersion5": IISVersion5,
            "IISVersion6": IISVersion6,
        }

    def simulate(
        self,
        features_df: pd.DataFrame,
        dataset: str,
        mode: str,
        focus_version: str,
        source_key: str,
        target_column: str,
        start_time_sec: float,
        end_time_sec: float,
        magnitude: float,
        operation: str = "add",
    ) -> dict[str, Any]:
        """Применяет интервенцию к признаку и пересчитывает IIS и IIS_dynamic."""

        if features_df.empty or focus_version not in self.model_map:
            return {}

        record_df = features_df.loc[features_df["source_record_id"].astype(str) == str(source_key)].copy()
        if record_df.empty or target_column not in record_df.columns:
            return {}

        record_df = record_df.sort_values(["window_start_sec", "segment_id"]).reset_index(drop=True)
        mask = (pd.to_numeric(record_df["window_start_sec"], errors="coerce") >= float(start_time_sec)) & (
            pd.to_numeric(record_df["window_start_sec"], errors="coerce") <= float(end_time_sec)
        )
        if not mask.any():
            return {}

        baseline_df = record_df.copy()
        intervened_df = record_df.copy()

        baseline_values = pd.to_numeric(intervened_df[target_column], errors="coerce")
        intervened_values = baseline_values.copy()
        if operation == "scale":
            intervened_values.loc[mask] = intervened_values.loc[mask] * (1.0 + float(magnitude))
        else:
            intervened_values.loc[mask] = intervened_values.loc[mask] + float(magnitude)
        intervened_df[target_column] = intervened_values

        model = self.model_map[focus_version]()
        baseline_results = model.evaluate_dataframe(baseline_df, mode=mode)
        intervened_results = model.evaluate_dataframe(intervened_df, mode=mode)

        baseline_dynamic, _ = self.dynamic_analyzer.build_dynamic_frame(baseline_results, focus_version=focus_version)
        intervention_dynamic, _ = self.dynamic_analyzer.build_dynamic_frame(intervened_results, focus_version=focus_version)
        if intervention_dynamic.empty:
            return {}
        baseline_record = baseline_dynamic.loc[baseline_dynamic["_source_key"].astype(str) == str(source_key)].copy()
        intervention_record = intervention_dynamic.loc[
            intervention_dynamic["_source_key"].astype(str) == str(source_key)
        ].copy()
        if baseline_record.empty or intervention_record.empty:
            return {}

        merged_df = baseline_record.merge(
            intervention_record[
                [
                    "segment_id",
                    "IIS",
                    "IIS_fast",
                    "IIS_smooth_core",
                    "IIS_dynamic",
                    "stress_drive",
                    "response_gain",
                    "recovery_gain",
                    "dynamic_mode",
                ]
            ].rename(
                columns={
                    "IIS": "IIS_intervened",
                    "IIS_fast": "IIS_fast_intervened",
                    "IIS_smooth_core": "IIS_smooth_core_intervened",
                    "IIS_dynamic": "IIS_dynamic_intervened",
                    "stress_drive": "stress_drive_intervened",
                    "response_gain": "response_gain_intervened",
                    "recovery_gain": "recovery_gain_intervened",
                    "dynamic_mode": "dynamic_mode_intervened",
                }
            ),
            on="segment_id",
            how="left",
        )
        merged_df = merged_df.merge(
            baseline_df[["segment_id", target_column]].rename(columns={target_column: target_column}),
            on="segment_id",
            how="left",
        )
        merged_df[f"{target_column}_intervened"] = intervened_values.to_numpy()
        merged_df["intervention_mask"] = mask.to_numpy()

        safe_target = "".join(ch if ch.isalnum() else "_" for ch in target_column.lower())
        safe_source = "".join(ch if ch.isalnum() else "_" for ch in str(source_key).lower())[:80]
        csv_path = self.output_dir / f"intervention_{dataset}_{mode}_{focus_version.lower()}_{safe_source}_{safe_target}.csv"
        plot_path = self.plots_dir / f"intervention_{dataset}_{mode}_{focus_version.lower()}_{safe_source}_{safe_target}.png"
        merged_df.to_csv(csv_path, index=False, encoding="utf-8-sig")
        self.dynamic_analyzer._plot_intervention(  # noqa: SLF001
            merged_df=merged_df,
            dataset=dataset,
            mode=mode,
            focus_version=focus_version,
            source_key=str(source_key),
            component_name=target_column,
            plot_path=plot_path,
        )
        return {
            "intervention_csv": str(csv_path),
            "intervention_plot": str(plot_path),
            "intervention_dataframe": merged_df,
        }
