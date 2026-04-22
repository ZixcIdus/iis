"""Сравнение версий ИИС, агрегирование метрик и построение графиков."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr

from config.settings import (
    DYNAMIC_MAX_RECORDS,
    DYNAMIC_MIN_POINTS,
    DYNAMIC_ROLLING_WINDOW,
    HISTOGRAM_BINS,
    MODEL_COMPONENTS,
    PLOT_DPI,
    RELIABILITY_THRESHOLDS,
    SENSITIVITY_COMPONENTS,
    UTILITY_WEIGHTS,
)
from features.common import safe_json_dumps


class IISComparison:
    """Собирает метрики сравнения моделей и сохраняет графики."""

    def __init__(self, plots_dir: Path) -> None:
        self.plots_dir = Path(plots_dir)
        self.plots_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir = self.plots_dir.parent

    def compare(
        self,
        features_df: pd.DataFrame,
        results_df: pd.DataFrame,
        dataset: str,
        mode: str,
    ) -> tuple[pd.DataFrame, dict[str, Any]]:
        """Строит агрегированные метрики для всех версий."""

        comparison_records: list[dict[str, Any]] = []
        summary: dict[str, Any] = {
            "dataset": dataset,
            "mode": mode,
            "feature_availability": self._feature_availability_summary(features_df),
            "versions": {},
            "limitations": [],
            "reliable_comparisons": [],
        }

        for version_name, version_df in results_df.groupby("version", sort=True):
            metrics = self._metrics_for_version(version_df=version_df)
            comparison_records.append({"dataset": dataset, "mode": mode, "version": version_name, **metrics})
            summary["versions"][version_name] = metrics

        comparison_df = pd.DataFrame.from_records(comparison_records)
        if comparison_df.empty:
            summary["limitations"].append("Нет валидных результатов ИИС для сравнения.")
            return comparison_df, summary

        comparison_df["utility_score"] = self._normalize_utility_score(comparison_df)
        comparison_df["utility_rank"] = comparison_df["utility_score"].rank(ascending=False, method="dense").astype(int)
        comparison_df["oversmoothing_flag"] = comparison_df.apply(self._detect_oversmoothing, axis=1)
        comparison_df["reliability_level"] = comparison_df.apply(self._reliability_level, axis=1)

        for row in comparison_df.to_dict(orient="records"):
            summary["versions"][row["version"]]["utility_score"] = row["utility_score"]
            summary["versions"][row["version"]]["utility_rank"] = int(row["utility_rank"])
            summary["versions"][row["version"]]["oversmoothing_flag"] = bool(row["oversmoothing_flag"])
            summary["versions"][row["version"]]["reliability_level"] = row["reliability_level"]
            if row["reliability_level"] in {"high", "medium"}:
                summary["reliable_comparisons"].append(row["version"])
            if row["oversmoothing_flag"]:
                summary["limitations"].append(
                    f"{row['version']}: возможное избыточное сглаживание по сочетанию overlap/effect size/sensitivity."
                )

        summary["version_ranking"] = (
            comparison_df.sort_values(["utility_rank", "version"], ascending=[True, True])[
                ["version", "utility_rank", "utility_score"]
            ]
            .to_dict(orient="records")
        )

        self._plot_boxplots(results_df=results_df, dataset=dataset, mode=mode)
        self._plot_stress_baseline_difference(comparison_df=comparison_df, dataset=dataset, mode=mode)
        self._plot_scatter(results_df=results_df, dataset=dataset, mode=mode, x_column="valence", y_column="Q")
        self._plot_scatter(results_df=results_df, dataset=dataset, mode=mode, x_column="arousal", y_column="IIS")
        self._plot_component_contributions(results_df=results_df, dataset=dataset, mode=mode)
        self._plot_sensitivity(comparison_df=comparison_df, dataset=dataset, mode=mode)
        dynamic_paths = self._plot_dynamic_trajectories(results_df=results_df, dataset=dataset, mode=mode)
        if dynamic_paths:
            summary["dynamic_outputs"] = dynamic_paths

        return comparison_df, summary

    def _metrics_for_version(self, version_df: pd.DataFrame) -> dict[str, Any]:
        """Считает набор метрик для одной версии модели."""

        valid_iis = version_df["IIS"].dropna()
        label_means = version_df.groupby("label")["IIS"].mean().dropna().to_dict()
        label_variances = version_df.groupby("label")["IIS"].var(ddof=1).dropna().to_dict()

        stress_values = version_df.loc[version_df["label"] == "stress", "IIS"].dropna().to_numpy()
        baseline_values = version_df.loc[version_df["label"] == "baseline", "IIS"].dropna().to_numpy()
        stress_baseline_diff = (
            float(np.mean(stress_values) - np.mean(baseline_values))
            if stress_values.size and baseline_values.size
            else np.nan
        )
        effect_size = self._cohens_d(stress_values, baseline_values)
        overlap_share = self._distribution_overlap(stress_values, baseline_values)

        arousal_corr = self._safe_correlation(version_df["IIS"], version_df["arousal"])
        valence_corr = self._safe_correlation(version_df["Q"], version_df["valence"])
        within_variance = float(np.nanmean(list(label_variances.values()))) if label_variances else np.nan
        coverage = float(valid_iis.size / max(len(version_df), 1))
        provenance_columns = [f"prov_{component}" for component in MODEL_COMPONENTS if f"prov_{component}" in version_df.columns]
        if provenance_columns:
            direct_ratio = float(version_df[provenance_columns].eq("direct").mean().mean())
        else:
            direct_ratio = 0.0

        contribution_means = {
            component: float(version_df[f"contrib_{component}"].dropna().mean())
            if f"contrib_{component}" in version_df
            else np.nan
            for component in MODEL_COMPONENTS
        }
        sensitivity_means = {
            component: float(version_df[f"sens_{component}"].dropna().mean())
            if f"sens_{component}" in version_df
            else np.nan
            for component in SENSITIVITY_COMPONENTS
        }
        available_sensitivities = [abs(value) for value in sensitivity_means.values() if np.isfinite(value)]
        relative_sensitivity = float(np.mean(available_sensitivities)) if available_sensitivities else np.nan

        return {
            "valid_segments": int(valid_iis.size),
            "coverage": coverage,
            "mean_iis": float(valid_iis.mean()) if not valid_iis.empty else np.nan,
            "std_iis": float(valid_iis.std(ddof=1)) if valid_iis.size > 1 else np.nan,
            "within_group_variance": within_variance,
            "stress_baseline_diff": stress_baseline_diff,
            "effect_size": effect_size,
            "distribution_overlap": overlap_share,
            "arousal_correlation": arousal_corr,
            "valence_correlation": valence_corr,
            "relative_sensitivity": relative_sensitivity,
            "direct_ratio": direct_ratio,
            "label_means_json": safe_json_dumps(label_means),
            "label_variances_json": safe_json_dumps(label_variances),
            "component_contributions_json": safe_json_dumps(contribution_means),
            "sensitivity_json": safe_json_dumps(sensitivity_means),
            **{f"mean_contrib_{name}": value for name, value in contribution_means.items()},
            **{f"mean_sens_{name}": value for name, value in sensitivity_means.items()},
        }

    def _normalize_utility_score(self, comparison_df: pd.DataFrame) -> pd.Series:
        """Строит сводный рейтинг полезности версии модели."""

        normalized_components = {
            "effect_size": self._series_minmax(comparison_df["effect_size"].abs()),
            "inverse_overlap": 1.0 - self._series_minmax(comparison_df["distribution_overlap"]),
            "relative_sensitivity": self._series_minmax(comparison_df["relative_sensitivity"]),
            "arousal_correlation": self._series_minmax(comparison_df["arousal_correlation"].abs()),
            "valence_correlation": self._series_minmax(comparison_df["valence_correlation"].abs()),
            "stability": 1.0 - self._series_minmax(comparison_df["within_group_variance"]),
            "coverage": self._series_minmax(comparison_df["coverage"]),
        }

        score = pd.Series(np.zeros(len(comparison_df)), index=comparison_df.index, dtype=float)
        for metric_name, weight in UTILITY_WEIGHTS.items():
            score = score + weight * normalized_components[metric_name].fillna(0.0)
        return score

    def _detect_oversmoothing(self, row: pd.Series) -> bool:
        """Помечает модели с признаками чрезмерного сглаживания."""

        overlap = float(row.get("distribution_overlap", np.nan))
        effect = abs(float(row.get("effect_size", np.nan)))
        sensitivity = float(row.get("relative_sensitivity", np.nan))
        if np.isnan(overlap) or np.isnan(effect) or np.isnan(sensitivity):
            return False
        return overlap > 0.65 and effect < 0.35 and sensitivity < 0.08

    def _reliability_level(self, row: pd.Series) -> str:
        """Оценивает надёжность сравнения по покрытию и объёму данных."""

        valid_segments = int(row.get("valid_segments", 0))
        direct_ratio = float(row.get("direct_ratio", 0.0))
        if (
            valid_segments >= RELIABILITY_THRESHOLDS["min_valid_segments"]
            and direct_ratio >= RELIABILITY_THRESHOLDS["min_direct_ratio_high"]
        ):
            return "high"
        if (
            valid_segments >= RELIABILITY_THRESHOLDS["min_valid_segments"]
            and direct_ratio >= RELIABILITY_THRESHOLDS["min_direct_ratio_medium"]
        ):
            return "medium"
        return "low"

    def _feature_availability_summary(self, features_df: pd.DataFrame) -> dict[str, Any]:
        """Возвращает краткую сводку доступности признаков."""

        columns = [
            "eeg_left_power",
            "eeg_right_power",
            "eeg_gamma_power",
            "eeg_alpha_left",
            "eeg_alpha_right",
            "hrv_hf",
            "hrv_lf",
            "hrv_hf_lf",
            "hrv_lf_hf",
            "stress_label",
            "valence",
            "arousal",
        ]
        summary = {}
        for column in columns:
            if column not in features_df.columns:
                continue
            count = int(features_df[column].notna().sum())
            summary[column] = {
                "valid_segments": count,
                "share": round(count / max(len(features_df), 1), 4),
            }
        return summary

    def _safe_correlation(self, first: pd.Series, second: pd.Series) -> float:
        """Считает корреляцию Пирсона при достаточном числе наблюдений."""

        paired = pd.concat([first, second], axis=1).dropna()
        if len(paired) < 3:
            return np.nan
        if paired.iloc[:, 0].nunique() < 2 or paired.iloc[:, 1].nunique() < 2:
            return np.nan
        try:
            value, _ = pearsonr(paired.iloc[:, 0], paired.iloc[:, 1])
            return float(value)
        except Exception:
            return np.nan

    def _cohens_d(self, first: np.ndarray, second: np.ndarray) -> float:
        """Считает effect size между двумя распределениями."""

        if first.size < 2 or second.size < 2:
            return np.nan
        first_mean = np.mean(first)
        second_mean = np.mean(second)
        pooled_std = np.sqrt(
            ((first.size - 1) * np.var(first, ddof=1) + (second.size - 1) * np.var(second, ddof=1))
            / (first.size + second.size - 2)
        )
        if pooled_std <= 0:
            return np.nan
        return float((first_mean - second_mean) / pooled_std)

    def _distribution_overlap(self, first: np.ndarray, second: np.ndarray) -> float:
        """Оценивает долю перекрытия двух распределений IIS."""

        if first.size < 2 or second.size < 2:
            return np.nan
        combined = np.concatenate([first, second])
        min_value = float(np.min(combined))
        max_value = float(np.max(combined))
        if min_value == max_value:
            return 1.0
        bins = np.linspace(min_value, max_value, HISTOGRAM_BINS + 1)
        first_hist, _ = np.histogram(first, bins=bins, density=True)
        second_hist, _ = np.histogram(second, bins=bins, density=True)
        bin_width = bins[1] - bins[0]
        return float(np.sum(np.minimum(first_hist, second_hist)) * bin_width)

    def _series_minmax(self, series: pd.Series) -> pd.Series:
        """Нормирует серию в диапазон 0..1."""

        numeric = pd.to_numeric(series, errors="coerce")
        valid = numeric.dropna()
        if valid.empty:
            return pd.Series(np.nan, index=series.index, dtype=float)
        min_value = valid.min()
        max_value = valid.max()
        if max_value == min_value:
            return pd.Series(np.ones(len(series)), index=series.index, dtype=float)
        return (numeric - min_value) / (max_value - min_value)

    def _plot_boxplots(self, results_df: pd.DataFrame, dataset: str, mode: str) -> None:
        """Строит boxplot IIS по классам для каждой версии."""

        versions = sorted(results_df["version"].dropna().unique())
        figure, axes = plt.subplots(1, max(len(versions), 1), figsize=(6 * max(len(versions), 1), 5), squeeze=False)

        for axis, version_name in zip(axes[0], versions):
            version_df = results_df.loc[results_df["version"] == version_name]
            labels = [
                label
                for label in sorted(version_df["label"].dropna().unique())
                if not version_df.loc[version_df["label"] == label, "IIS"].dropna().empty
            ]
            if not labels:
                axis.text(0.5, 0.5, "Нет данных", ha="center", va="center")
                axis.set_title(version_name)
                continue
            data = [version_df.loc[version_df["label"] == label, "IIS"].dropna().to_numpy() for label in labels]
            axis.boxplot(data, labels=labels, patch_artist=True)
            axis.set_title(version_name)
            axis.set_ylabel("IIS")
            axis.tick_params(axis="x", rotation=20)

        figure.suptitle(f"IIS по классам: {dataset.upper()} / {mode}")
        figure.tight_layout()
        figure.savefig(self.plots_dir / f"boxplot_iis_{dataset}_{mode}.png", dpi=PLOT_DPI)
        plt.close(figure)

    def _plot_stress_baseline_difference(self, comparison_df: pd.DataFrame, dataset: str, mode: str) -> None:
        """Строит bar chart разницы stress vs baseline."""

        figure, axis = plt.subplots(figsize=(8, 5))
        axis.bar(comparison_df["version"], comparison_df["stress_baseline_diff"])
        axis.set_title(f"Stress - Baseline: {dataset.upper()} / {mode}")
        axis.set_ylabel("Разница IIS")
        axis.set_xlabel("Версия")
        figure.tight_layout()
        figure.savefig(self.plots_dir / f"stress_baseline_diff_{dataset}_{mode}.png", dpi=PLOT_DPI)
        plt.close(figure)

    def _plot_scatter(
        self,
        results_df: pd.DataFrame,
        dataset: str,
        mode: str,
        x_column: str,
        y_column: str,
    ) -> None:
        """Строит scatter-график для каждой версии."""

        versions = sorted(results_df["version"].dropna().unique())
        figure, axes = plt.subplots(1, max(len(versions), 1), figsize=(6 * max(len(versions), 1), 5), squeeze=False)

        for axis, version_name in zip(axes[0], versions):
            version_df = results_df.loc[results_df["version"] == version_name, [x_column, y_column]].dropna()
            axis.set_title(version_name)
            axis.set_xlabel(x_column)
            axis.set_ylabel(y_column)
            if version_df.empty:
                axis.text(0.5, 0.5, "Нет данных", ha="center", va="center")
                continue
            axis.scatter(version_df[x_column], version_df[y_column], alpha=0.65)

        figure.suptitle(f"{y_column} vs {x_column}: {dataset.upper()} / {mode}")
        figure.tight_layout()
        figure.savefig(self.plots_dir / f"scatter_{y_column.lower()}_{x_column.lower()}_{dataset}_{mode}.png", dpi=PLOT_DPI)
        plt.close(figure)

    def _plot_component_contributions(self, results_df: pd.DataFrame, dataset: str, mode: str) -> None:
        """Строит bar chart средних вкладов компонентов."""

        versions = sorted(results_df["version"].dropna().unique())
        positions = np.arange(len(MODEL_COMPONENTS))
        width = 0.22 if versions else 0.2
        figure, axis = plt.subplots(figsize=(10, 5))

        for index, version_name in enumerate(versions):
            version_df = results_df.loc[results_df["version"] == version_name]
            means = [
                float(version_df[f"contrib_{component}"].dropna().mean()) if f"contrib_{component}" in version_df else np.nan
                for component in MODEL_COMPONENTS
            ]
            axis.bar(positions + index * width, means, width=width, label=version_name)

        axis.set_xticks(positions + width * max(len(versions) - 1, 0) / 2)
        axis.set_xticklabels(MODEL_COMPONENTS)
        axis.set_ylabel("Средний вклад")
        axis.set_title(f"Средние вклады компонентов: {dataset.upper()} / {mode}")
        if versions:
            axis.legend()
        figure.tight_layout()
        figure.savefig(self.plots_dir / f"component_contributions_{dataset}_{mode}.png", dpi=PLOT_DPI)
        plt.close(figure)

    def _plot_sensitivity(self, comparison_df: pd.DataFrame, dataset: str, mode: str) -> None:
        """Строит график чувствительности компонентов."""

        positions = np.arange(len(SENSITIVITY_COMPONENTS))
        width = 0.22 if not comparison_df.empty else 0.2
        figure, axis = plt.subplots(figsize=(10, 5))

        for index, row in enumerate(comparison_df.to_dict(orient="records")):
            values = [row.get(f"mean_sens_{component}", np.nan) for component in SENSITIVITY_COMPONENTS]
            axis.bar(positions + index * width, values, width=width, label=row["version"])

        axis.set_xticks(positions + width * max(len(comparison_df) - 1, 0) / 2)
        axis.set_xticklabels(SENSITIVITY_COMPONENTS)
        axis.set_ylabel("Чувствительность IIS")
        axis.set_title(f"Чувствительность компонентов: {dataset.upper()} / {mode}")
        if not comparison_df.empty:
            axis.legend()
        figure.tight_layout()
        figure.savefig(self.plots_dir / f"sensitivity_{dataset}_{mode}.png", dpi=PLOT_DPI)
        plt.close(figure)

    def _plot_dynamic_trajectories(self, results_df: pd.DataFrame, dataset: str, mode: str) -> dict[str, str]:
        """Строит наглядную динамику IIS и компонентов во времени."""

        if results_df.empty or "window_start_sec" not in results_df.columns:
            return {}

        versions = sorted(results_df["version"].dropna().unique())
        if not versions:
            return {}

        if "IISVersion6" in versions:
            focus_version = "IISVersion6"
        elif "IISVersion5" in versions:
            focus_version = "IISVersion5"
        elif "IISVersion4" in versions:
            focus_version = "IISVersion4"
        else:
            focus_version = versions[0]
        version_df = results_df.loc[results_df["version"] == focus_version].copy()
        version_df["window_start_sec"] = pd.to_numeric(version_df["window_start_sec"], errors="coerce")
        version_df["window_end_sec"] = pd.to_numeric(version_df["window_end_sec"], errors="coerce")
        version_df["IIS"] = pd.to_numeric(version_df["IIS"], errors="coerce")
        version_df = version_df.dropna(subset=["window_start_sec", "IIS"])
        if version_df.empty:
            return {}

        if "source_record_id" in version_df.columns:
            source_series = version_df["source_record_id"].fillna("").astype(str)
        else:
            source_series = pd.Series([""] * len(version_df), index=version_df.index, dtype=object)

        counts = (
            version_df.assign(_source_key=source_series.replace("", np.nan))
            .dropna(subset=["_source_key"])
            .groupby("_source_key")["segment_id"]
            .count()
            .sort_values(ascending=False)
        )

        selected_records = counts.loc[counts >= DYNAMIC_MIN_POINTS].head(DYNAMIC_MAX_RECORDS).index.tolist()
        if not selected_records:
            fallback_key = f"{dataset}_{focus_version}_all"
            version_df["_source_key"] = fallback_key
            selected_records = [fallback_key]
        else:
            version_df["_source_key"] = source_series.replace("", np.nan)
            version_df.loc[version_df["_source_key"].isna(), "_source_key"] = version_df["subject_id"].fillna("unknown")

        plot_df = version_df.loc[version_df["_source_key"].isin(selected_records)].copy()
        plot_df = plot_df.sort_values(["_source_key", "window_start_sec", "segment_id"])
        plot_df["time_sec"] = plot_df["window_start_sec"]
        plot_df["IIS_rolling_mean"] = (
            plot_df.groupby("_source_key")["IIS"]
            .transform(lambda series: series.rolling(window=DYNAMIC_ROLLING_WINDOW, min_periods=1).mean())
        )
        plot_df["IIS_rolling_median"] = (
            plot_df.groupby("_source_key")["IIS"]
            .transform(lambda series: series.rolling(window=DYNAMIC_ROLLING_WINDOW, min_periods=1).median())
        )

        csv_columns = [
            "version",
            "subject_id",
            "source_record_id",
            "segment_id",
            "label",
            "time_sec",
            "window_start_sec",
            "window_end_sec",
            "IIS",
            "IIS_rolling_mean",
            "IIS_rolling_median",
            "A",
            "Gamma",
            "V",
            "Q",
        ]
        available_csv_columns = [column for column in csv_columns if column in plot_df.columns]
        dynamic_csv_path = self.output_dir / f"dynamic_iis_{dataset}_{mode}_{focus_version.lower()}.csv"
        plot_df[available_csv_columns].to_csv(dynamic_csv_path, index=False, encoding="utf-8-sig")

        color_map = {
            "baseline": "#2E7D32",
            "disbalance": "#F9A825",
            "stress": "#C62828",
            "amusement": "#1565C0",
        }
        component_colors = {
            "A": "#1565C0",
            "Gamma": "#8E24AA",
            "V": "#00897B",
            "Q": "#6D4C41",
        }

        figure, axes = plt.subplots(
            len(selected_records),
            1,
            figsize=(12, 4.8 * len(selected_records)),
            squeeze=False,
            sharex=False,
        )

        for axis, source_key in zip(axes[:, 0], selected_records):
            record_df = plot_df.loc[plot_df["_source_key"] == source_key].sort_values(["time_sec", "segment_id"])
            axis.plot(record_df["time_sec"], record_df["IIS"], color="#B0BEC5", linewidth=1.2, alpha=0.9, label="IIS raw")
            axis.plot(
                record_df["time_sec"],
                record_df["IIS_rolling_mean"],
                color="#1E88E5",
                linewidth=2.1,
                label=f"causal mean ({DYNAMIC_ROLLING_WINDOW})",
            )
            axis.plot(
                record_df["time_sec"],
                record_df["IIS_rolling_median"],
                color="#F4511E",
                linewidth=1.8,
                linestyle="--",
                label=f"causal median ({DYNAMIC_ROLLING_WINDOW})",
            )

            for component in ("A", "Gamma", "V", "Q"):
                if component in record_df.columns and record_df[component].notna().any():
                    axis.plot(
                        record_df["time_sec"],
                        record_df[component],
                        color=component_colors[component],
                        linewidth=0.9,
                        alpha=0.55,
                        linestyle=":",
                        label=component,
                    )

            for label_name, label_df in record_df.groupby("label", sort=False):
                axis.scatter(
                    label_df["time_sec"],
                    label_df["IIS"],
                    s=20,
                    color=color_map.get(str(label_name).lower(), "#455A64"),
                    alpha=0.9,
                )

            axis.axhline(0.0, color="#CFD8DC", linewidth=0.8)
            axis.set_title(f"{focus_version} | {source_key}")
            axis.set_xlabel("Время окна, сек")
            axis.set_ylabel("IIS / компоненты")
            axis.grid(alpha=0.22)
            handles, labels = axis.get_legend_handles_labels()
            unique = dict(zip(labels, handles))
            axis.legend(unique.values(), unique.keys(), loc="best", fontsize=8)

        figure.suptitle(f"Динамика IIS (только прошлые окна): {dataset.upper()} / {mode} / {focus_version}")
        figure.tight_layout()
        dynamic_plot_path = self.plots_dir / f"dynamic_iis_{dataset}_{mode}_{focus_version.lower()}.png"
        figure.savefig(dynamic_plot_path, dpi=PLOT_DPI)
        plt.close(figure)

        return {
            "focus_version": focus_version,
            "dynamic_csv": str(dynamic_csv_path),
            "dynamic_plot": str(dynamic_plot_path),
        }
