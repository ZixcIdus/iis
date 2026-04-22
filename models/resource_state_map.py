"""Ресурсная ось и квадрантная карта состояний для версий V4/V5/V6/V7."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config.settings import PLOT_DPI


class IISResourceStateMap:
    """Добавляет к результатам вторую ось ресурсности и 4-квадрантную карту."""

    SUPPORTED_VERSIONS = {"IISVersion4", "IISVersion5", "IISVersion6", "IISVersion7"}

    def __init__(self, plots_dir: Path) -> None:
        self.plots_dir = Path(plots_dir)
        self.plots_dir.mkdir(parents=True, exist_ok=True)

    def augment_results(
        self,
        results_df: pd.DataFrame,
        *,
        dataset: str,
        mode: str,
    ) -> tuple[pd.DataFrame, dict[str, Any]]:
        """Добавляет RES и state-map к уже рассчитанным результатам моделей."""

        if results_df.empty:
            return results_df.copy(), {}

        frame = results_df.copy()
        frame = self._ensure_resource_columns(frame)

        summary: dict[str, Any] = {"versions": {}, "plots": []}
        for version_name, version_df in frame.groupby("version", sort=True):
            if version_name not in self.SUPPORTED_VERSIONS:
                continue

            version_frame = self._compute_resource_columns(version_df)
            if version_frame["RES"].notna().sum() <= 0:
                continue

            valid_rows = version_frame.loc[version_frame["RES"].notna()].copy()
            iis_threshold = float(pd.to_numeric(valid_rows["IIS"], errors="coerce").median())
            res_threshold = float(pd.to_numeric(valid_rows["RES"], errors="coerce").median())

            quadrant_state, quadrant_margin = self._classify_quadrants(
                iis_series=pd.to_numeric(valid_rows["IIS"], errors="coerce"),
                res_series=pd.to_numeric(valid_rows["RES"], errors="coerce"),
                iis_threshold=iis_threshold,
                res_threshold=res_threshold,
            )
            valid_rows.loc[:, "IIS_threshold_empirical"] = iis_threshold
            valid_rows.loc[:, "RES_threshold_empirical"] = res_threshold
            valid_rows.loc[:, "state_map_4"] = quadrant_state.to_numpy()
            valid_rows.loc[:, "state_map_margin"] = quadrant_margin.to_numpy()
            self._write_resource_columns(frame=frame, source_df=valid_rows)

            counts = quadrant_state.value_counts().to_dict()
            by_label = (
                pd.crosstab(valid_rows["label"].fillna("unknown").astype(str), quadrant_state)
                .to_dict(orient="index")
            )
            plot_path = self._plot_state_map(
                valid_rows.assign(RES=valid_rows["RES"], state_map_4=quadrant_state),
                dataset=dataset,
                mode=mode,
                version=version_name,
                iis_threshold=iis_threshold,
                res_threshold=res_threshold,
            )
            summary["versions"][version_name] = {
                "iis_threshold_empirical": iis_threshold,
                "res_threshold_empirical": res_threshold,
                "quadrant_counts": counts,
                "quadrant_by_label": by_label,
                "resource_note": (
                    "RES — не отдельное доказанное физиологическое состояние, а вторая ось карты: "
                    "высокий V повышает ресурс, перегретая Gamma и рассогласование A/V его снижают."
                ),
                "state_map_plot": str(plot_path),
            }
            summary["plots"].append(str(plot_path))

        return frame, summary

    def augment_preview_frame(self, preview_df: pd.DataFrame) -> pd.DataFrame:
        """Добавляет RES и квадранты к промежуточному фрейму без сохранения графиков."""

        if preview_df.empty:
            return self._ensure_resource_columns(preview_df.copy())

        frame = self._ensure_resource_columns(preview_df.copy())
        for version_name, version_df in frame.groupby("version", sort=False):
            if version_name not in self.SUPPORTED_VERSIONS:
                continue
            version_frame = self._compute_resource_columns(version_df)
            valid_rows = version_frame.loc[version_frame["RES"].notna()].copy()
            if valid_rows.empty:
                continue

            iis_threshold = float(pd.to_numeric(valid_rows["IIS"], errors="coerce").median())
            res_threshold = float(pd.to_numeric(valid_rows["RES"], errors="coerce").median())
            quadrant_state, quadrant_margin = self._classify_quadrants(
                iis_series=pd.to_numeric(valid_rows["IIS"], errors="coerce"),
                res_series=pd.to_numeric(valid_rows["RES"], errors="coerce"),
                iis_threshold=iis_threshold,
                res_threshold=res_threshold,
            )
            valid_rows.loc[:, "IIS_threshold_empirical"] = iis_threshold
            valid_rows.loc[:, "RES_threshold_empirical"] = res_threshold
            valid_rows.loc[:, "state_map_4"] = quadrant_state.to_numpy()
            valid_rows.loc[:, "state_map_margin"] = quadrant_margin.to_numpy()
            self._write_resource_columns(frame=frame, source_df=valid_rows)
        return frame

    def _ensure_resource_columns(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Гарантирует наличие служебных колонок ресурсной карты."""

        prepared = frame.copy()
        defaults: dict[str, Any] = {
            "RES": np.nan,
            "RES_core": np.nan,
            "RES_mismatch": np.nan,
            "RES_gamma_load": np.nan,
            "IIS_threshold_empirical": np.nan,
            "RES_threshold_empirical": np.nan,
            "state_map_4": "",
            "state_map_margin": np.nan,
        }
        for column_name, default_value in defaults.items():
            if column_name not in prepared.columns:
                prepared[column_name] = default_value
        prepared["state_map_4"] = prepared["state_map_4"].astype(object)
        return prepared

    def _write_resource_columns(self, *, frame: pd.DataFrame, source_df: pd.DataFrame) -> None:
        """Записывает обратно только ресурсные колонки без конфликтов dtype."""

        columns_to_update = (
            "RES",
            "RES_core",
            "RES_mismatch",
            "RES_gamma_load",
            "IIS_threshold_empirical",
            "RES_threshold_empirical",
            "state_map_4",
            "state_map_margin",
        )
        for column_name in columns_to_update:
            if column_name not in source_df.columns:
                continue
            if column_name == "state_map_4":
                frame.loc[source_df.index, column_name] = source_df[column_name].astype(object).tolist()
            else:
                frame.loc[source_df.index, column_name] = source_df[column_name].to_numpy()

    def _compute_resource_columns(self, version_df: pd.DataFrame) -> pd.DataFrame:
        """Считает непрерывную ресурсную ось без классификации квадрантов."""

        frame = self._ensure_resource_columns(version_df.copy())
        a_value = pd.to_numeric(frame.get("A"), errors="coerce")
        gamma_value = pd.to_numeric(frame.get("Gamma"), errors="coerce")
        v_value = pd.to_numeric(frame.get("V"), errors="coerce")
        iis_value = pd.to_numeric(frame.get("IIS"), errors="coerce")
        valid_mask = a_value.notna() & gamma_value.notna() & v_value.notna() & iis_value.notna()
        if not bool(valid_mask.any()):
            return frame

        gamma_load = gamma_value.clip(lower=0.0)
        mismatch = (a_value - v_value).abs()
        resource_core = 0.70 * v_value - 0.35 * gamma_load - 0.25 * mismatch
        res_score = np.clip(0.5 + 0.5 * np.tanh(1.8 * resource_core), 0.0, 1.0)

        valid_indices = frame.index[valid_mask]
        frame.loc[valid_indices, "RES"] = res_score.loc[valid_mask].to_numpy()
        frame.loc[valid_indices, "RES_core"] = resource_core.loc[valid_mask].to_numpy()
        frame.loc[valid_indices, "RES_mismatch"] = mismatch.loc[valid_mask].to_numpy()
        frame.loc[valid_indices, "RES_gamma_load"] = gamma_load.loc[valid_mask].to_numpy()
        return frame

    def _classify_quadrants(
        self,
        *,
        iis_series: pd.Series,
        res_series: pd.Series,
        iis_threshold: float,
        res_threshold: float,
    ) -> tuple[pd.Series, pd.Series]:
        """Относит точки к четырём макрорежимам по эмпирическим медианам."""

        states: list[str] = []
        margins: list[float] = []
        for iis_value, res_value in zip(iis_series.to_numpy(), res_series.to_numpy(), strict=False):
            iis_distance = float(iis_value - iis_threshold)
            res_distance = float(res_value - res_threshold)
            margins.append(float(min(abs(iis_distance), abs(res_distance))))
            if iis_distance >= 0.0 and res_distance >= 0.0:
                states.append("regulated")
            elif iis_distance < 0.0 and res_distance >= 0.0:
                states.append("mobilized")
            elif iis_distance >= 0.0 and res_distance < 0.0:
                states.append("fragile")
            else:
                states.append("depleted")
        return pd.Series(states, index=iis_series.index, dtype=str), pd.Series(margins, index=iis_series.index, dtype=float)

    def _plot_state_map(
        self,
        version_df: pd.DataFrame,
        *,
        dataset: str,
        mode: str,
        version: str,
        iis_threshold: float,
        res_threshold: float,
    ) -> Path:
        """Строит карту IIS × RES для выбранной версии."""

        color_map = {
            "baseline": "#2E7D32",
            "disbalance": "#F9A825",
            "stress": "#C62828",
            "amusement": "#1565C0",
            "unknown": "#546E7A",
        }
        figure, axes = plt.subplots(1, 2, figsize=(12.5, 5.2))

        for label_name, label_df in version_df.groupby(version_df["label"].fillna("unknown").astype(str), sort=False):
            axes[0].scatter(
                pd.to_numeric(label_df["IIS"], errors="coerce"),
                pd.to_numeric(label_df["RES"], errors="coerce"),
                s=18,
                alpha=0.55,
                color=color_map.get(str(label_name).lower(), "#546E7A"),
                label=str(label_name),
            )
        axes[0].axvline(iis_threshold, color="#37474F", linestyle="--", linewidth=1.0)
        axes[0].axhline(res_threshold, color="#37474F", linestyle="--", linewidth=1.0)
        axes[0].set_title(f"{version}: IIS vs RES по меткам")
        axes[0].set_xlabel("IIS")
        axes[0].set_ylabel("RES")
        axes[0].grid(alpha=0.22)
        axes[0].legend(loc="best", fontsize=8)

        quadrant_colors = {
            "regulated": "#2E7D32",
            "mobilized": "#EF6C00",
            "fragile": "#5E35B1",
            "depleted": "#C62828",
        }
        for state_name, state_df in version_df.groupby(version_df["state_map_4"].astype(str), sort=False):
            axes[1].scatter(
                pd.to_numeric(state_df["IIS"], errors="coerce"),
                pd.to_numeric(state_df["RES"], errors="coerce"),
                s=18,
                alpha=0.60,
                color=quadrant_colors.get(str(state_name).lower(), "#546E7A"),
                label=str(state_name),
            )
        axes[1].axvline(iis_threshold, color="#37474F", linestyle="--", linewidth=1.0)
        axes[1].axhline(res_threshold, color="#37474F", linestyle="--", linewidth=1.0)
        axes[1].set_title(f"{version}: 4-квадрантная карта")
        axes[1].set_xlabel("IIS")
        axes[1].set_ylabel("RES")
        axes[1].grid(alpha=0.22)
        axes[1].legend(loc="best", fontsize=8)

        figure.suptitle(f"Карта IIS × RES: {dataset.upper()} / {mode} / {version}")
        figure.tight_layout()
        safe_version = "".join(ch if ch.isalnum() else "_" for ch in version.lower())
        plot_path = self.plots_dir / f"state_map_v5_res_{dataset}_{mode}_{safe_version}.png"
        figure.savefig(plot_path, dpi=PLOT_DPI)
        plt.close(figure)
        return plot_path
