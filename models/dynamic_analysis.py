"""Причинно-следственный динамический анализ IIS без подглядывания в будущее."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config.settings import (
    DYNAMIC_EWMA_ALPHA,
    DYNAMIC_FAST_ALPHA,
    DYNAMIC_GAIN_MAX,
    DYNAMIC_GAIN_MIN,
    DYNAMIC_MAX_RECORDS,
    DYNAMIC_MIN_POINTS,
    DYNAMIC_RECOVERY_ALPHA_BASE,
    DYNAMIC_RECOVERY_ALPHA_SCALE,
    DYNAMIC_ROLLING_WINDOW,
    DYNAMIC_RESPONSE_ALPHA_BASE,
    DYNAMIC_RESPONSE_ALPHA_SCALE,
    DYNAMIC_SMOOTHING_BLEND,
    DYNAMIC_SMOOTH_WEIGHT_MAX,
    DYNAMIC_SMOOTH_WEIGHT_MIN,
    DYNAMIC_VOLATILITY_SCALE,
    DYNAMIC_VOLATILITY_WINDOW,
    MODEL_CONFIGS,
    PLOT_DPI,
    VERSION5_CALIBRATION,
    VERSION6_CALIBRATION,
    VERSION7_CALIBRATION,
    VERSION7_DATASET_OVERRIDES,
)


class CausalDynamicAnalyzer:
    """Строит динамический IIS только из прошлых наблюдений."""

    def __init__(self, output_dir: Path, plots_dir: Path) -> None:
        self.output_dir = Path(output_dir)
        self.plots_dir = Path(plots_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.plots_dir.mkdir(parents=True, exist_ok=True)

    def analyze(
        self,
        results_df: pd.DataFrame,
        dataset: str,
        mode: str,
        focus_version: str = "IISVersion6",
    ) -> dict[str, Any]:
        """Строит причинную динамическую траекторию для выбранной версии."""

        dynamic_df, selected_records = self.build_dynamic_frame(results_df=results_df, focus_version=focus_version)
        if dynamic_df.empty:
            return {}

        csv_path = self.output_dir / f"dynamic_iis_{dataset}_{mode}_{focus_version.lower()}_causal.csv"
        dynamic_df.to_csv(csv_path, index=False, encoding="utf-8-sig")

        plot_path = self._plot_dynamic(dynamic_df=dynamic_df, dataset=dataset, mode=mode, focus_version=focus_version, selected_records=selected_records)
        return {
            "focus_version": focus_version,
            "dynamic_csv": str(csv_path),
            "dynamic_plot": str(plot_path),
            "dynamic_mode": "causal",
        }

    def build_dynamic_frame(
        self,
        results_df: pd.DataFrame,
        focus_version: str = "IISVersion6",
    ) -> tuple[pd.DataFrame, list[str]]:
        """Готовит причинную динамическую таблицу без сохранения на диск."""

        if results_df.empty or focus_version not in results_df["version"].astype(str).unique():
            return pd.DataFrame(), []

        version_df = results_df.loc[results_df["version"] == focus_version].copy()
        if version_df.empty:
            return pd.DataFrame(), []

        version_df["window_start_sec"] = pd.to_numeric(version_df.get("window_start_sec"), errors="coerce")
        version_df["window_end_sec"] = pd.to_numeric(version_df.get("window_end_sec"), errors="coerce")
        version_df["IIS"] = pd.to_numeric(version_df.get("IIS"), errors="coerce")
        version_df = version_df.dropna(subset=["window_start_sec", "IIS"])
        if version_df.empty:
            return pd.DataFrame(), []

        source_key = version_df.get("source_record_id")
        if source_key is None:
            version_df["_source_key"] = version_df["subject_id"].fillna("unknown").astype(str)
        else:
            version_df["_source_key"] = source_key.fillna("").astype(str)
            version_df.loc[version_df["_source_key"] == "", "_source_key"] = version_df["subject_id"].fillna("unknown").astype(str)

        version_df = version_df.sort_values(["_source_key", "window_start_sec", "segment_id"])

        processed_frames: list[pd.DataFrame] = []
        counts = version_df.groupby("_source_key")["segment_id"].count().sort_values(ascending=False)
        selected_records = counts.loc[counts >= DYNAMIC_MIN_POINTS].head(DYNAMIC_MAX_RECORDS).index.tolist()
        if not selected_records:
            selected_records = counts.head(DYNAMIC_MAX_RECORDS).index.tolist()

        for _, group_df in version_df.groupby("_source_key", sort=False):
            processed_frames.append(self._prepare_causal_group(group_df))

        dynamic_df = pd.concat(processed_frames, ignore_index=True) if processed_frames else pd.DataFrame()
        return dynamic_df, selected_records

    def simulate_component_intervention(
        self,
        results_df: pd.DataFrame,
        dataset: str,
        mode: str,
        source_key: str,
        component_name: str,
        start_time_sec: float,
        end_time_sec: float,
        magnitude: float,
        operation: str = "add",
        focus_version: str = "IISVersion6",
    ) -> dict[str, Any]:
        """Применяет псевдо-причинное возмущение к ряду компонентов и пересчитывает динамику."""

        dynamic_df, _ = self.build_dynamic_frame(results_df=results_df, focus_version=focus_version)
        if dynamic_df.empty:
            return {}

        record_df = dynamic_df.loc[dynamic_df["_source_key"].astype(str) == str(source_key)].copy()
        if record_df.empty:
            return {}

        record_df = record_df.sort_values(["time_sec", "segment_id"]).reset_index(drop=True)
        if component_name not in record_df.columns:
            return {}

        mask = (pd.to_numeric(record_df["time_sec"], errors="coerce") >= float(start_time_sec)) & (
            pd.to_numeric(record_df["time_sec"], errors="coerce") <= float(end_time_sec)
        )
        if not mask.any():
            return {}

        original_values = pd.to_numeric(record_df[component_name], errors="coerce")
        perturbed_values = original_values.copy()
        if operation == "scale":
            perturbed_values.loc[mask] = perturbed_values.loc[mask] * (1.0 + float(magnitude))
        else:
            perturbed_values.loc[mask] = perturbed_values.loc[mask] + float(magnitude)

        record_df[f"{component_name}_intervened"] = perturbed_values
        recomputed_df = record_df.copy()
        recomputed_df[component_name] = perturbed_values
        recomputed_df["IIS"] = self._recompute_iis(recomputed_df, focus_version=focus_version)
        recomputed_df = self._prepare_causal_group(recomputed_df)
        recomputed_df = recomputed_df.rename(
            columns={
                "IIS": "IIS_intervened",
                "IIS_causal_mean": "IIS_causal_mean_intervened",
                "IIS_causal_median": "IIS_causal_median_intervened",
                "IIS_causal_ewma": "IIS_causal_ewma_intervened",
                "IIS_fast": "IIS_fast_intervened",
                "IIS_volatility": "IIS_volatility_intervened",
                "IIS_smooth_weight": "IIS_smooth_weight_intervened",
                "IIS_smooth_core": "IIS_smooth_core_intervened",
                "stress_drive": "stress_drive_intervened",
                "response_gain": "response_gain_intervened",
                "recovery_gain": "recovery_gain_intervened",
                "dynamic_mode": "dynamic_mode_intervened",
                "IIS_dynamic": "IIS_dynamic_intervened",
                "dIIS_raw_dt": "dIIS_raw_dt_intervened",
                "dIIS_dynamic_dt": "dIIS_dynamic_dt_intervened",
            }
        )

        merged_df = record_df.merge(
            recomputed_df[
                [
                    "segment_id",
                    f"{component_name}_intervened",
                    "IIS_intervened",
                    "IIS_fast_intervened",
                    "IIS_smooth_core_intervened",
                    "IIS_dynamic_intervened",
                    "stress_drive_intervened",
                    "response_gain_intervened",
                    "recovery_gain_intervened",
                    "dynamic_mode_intervened",
                ]
            ],
            on="segment_id",
            how="left",
        )
        merged_df["intervention_mask"] = mask.to_numpy()

        safe_component = "".join(ch if ch.isalnum() else "_" for ch in component_name.lower())
        safe_source = "".join(ch if ch.isalnum() else "_" for ch in str(source_key).lower())[:80]
        csv_path = self.output_dir / f"intervention_{dataset}_{mode}_{focus_version.lower()}_{safe_source}_{safe_component}.csv"
        plot_path = self.plots_dir / f"intervention_{dataset}_{mode}_{focus_version.lower()}_{safe_source}_{safe_component}.png"
        merged_df.to_csv(csv_path, index=False, encoding="utf-8-sig")
        self._plot_intervention(
            merged_df=merged_df,
            dataset=dataset,
            mode=mode,
            focus_version=focus_version,
            source_key=str(source_key),
            component_name=component_name,
            plot_path=plot_path,
        )
        return {
            "intervention_csv": str(csv_path),
            "intervention_plot": str(plot_path),
            "intervention_dataframe": merged_df,
        }

    def _plot_dynamic(
        self,
        dynamic_df: pd.DataFrame,
        dataset: str,
        mode: str,
        focus_version: str,
        selected_records: list[str],
    ) -> Path:
        """Рисует причинную динамику без использования будущих окон."""

        color_map = {
            "baseline": "#2E7D32",
            "disbalance": "#F9A825",
            "stress": "#C62828",
            "amusement": "#1565C0",
        }
        figure, axes = plt.subplots(
            len(selected_records),
            1,
            figsize=(12, 4.8 * max(len(selected_records), 1)),
            squeeze=False,
            sharex=False,
        )

        for axis, source_key in zip(axes[:, 0], selected_records):
            group_df = dynamic_df.loc[dynamic_df["_source_key"] == source_key].sort_values(["time_sec", "segment_id"])
            axis.plot(group_df["time_sec"], group_df["IIS"], color="#B0BEC5", linewidth=1.1, alpha=0.85, label="IIS raw")
            axis.plot(group_df["time_sec"], group_df["IIS_causal_median"], color="#FB8C00", linewidth=1.4, linestyle="--", label="past median")
            axis.plot(group_df["time_sec"], group_df["IIS_fast"], color="#3949AB", linewidth=1.7, label="IIS fast")
            axis.plot(group_df["time_sec"], group_df["IIS_smooth_core"], color="#8E24AA", linewidth=1.5, linestyle="-.", label="smooth core")
            axis.plot(group_df["time_sec"], group_df["IIS_dynamic"], color="#00897B", linewidth=2.3, label="IIS dynamic")

            for label_name, label_df in group_df.groupby("label", sort=False):
                axis.scatter(
                    label_df["time_sec"],
                    label_df["IIS"],
                    s=18,
                    color=color_map.get(str(label_name).lower(), "#546E7A"),
                    alpha=0.75,
                )

            axis.set_title(f"{focus_version} causal | {source_key}")
            axis.set_xlabel("Время окна, сек")
            axis.set_ylabel("IIS")
            axis.grid(alpha=0.22)
            axis.legend(loc="best", fontsize=8)

        figure.suptitle(f"Причинная динамика IIS: {dataset.upper()} / {mode} / {focus_version}")
        figure.tight_layout()
        plot_path = self.plots_dir / f"dynamic_iis_{dataset}_{mode}_{focus_version.lower()}_causal.png"
        figure.savefig(plot_path, dpi=PLOT_DPI)
        plt.close(figure)
        return plot_path

    def _prepare_causal_group(self, group_df: pd.DataFrame) -> pd.DataFrame:
        """Строит причинные признаки и динамику для одной записи."""

        group_df = group_df.copy().sort_values(["window_start_sec", "segment_id"])
        group_df["time_sec"] = group_df["window_start_sec"]
        group_df["delta_t_sec"] = group_df["time_sec"].diff().replace(0.0, np.nan)
        group_df["IIS_causal_mean"] = group_df["IIS"].rolling(window=DYNAMIC_ROLLING_WINDOW, min_periods=1).mean()
        group_df["IIS_causal_median"] = group_df["IIS"].rolling(window=DYNAMIC_ROLLING_WINDOW, min_periods=1).median()
        group_df["IIS_causal_ewma"] = group_df["IIS"].ewm(alpha=DYNAMIC_EWMA_ALPHA, adjust=False).mean()
        group_df["IIS_fast"] = group_df["IIS"].ewm(alpha=DYNAMIC_FAST_ALPHA, adjust=False).mean()

        raw_diff = group_df["IIS"].diff().abs().fillna(0.0)
        group_df["IIS_volatility"] = raw_diff.rolling(window=DYNAMIC_VOLATILITY_WINDOW, min_periods=1).mean()
        smooth_weight = (group_df["IIS_volatility"] / DYNAMIC_VOLATILITY_SCALE).clip(
            lower=DYNAMIC_SMOOTH_WEIGHT_MIN,
            upper=DYNAMIC_SMOOTH_WEIGHT_MAX,
        )
        group_df["IIS_smooth_weight"] = smooth_weight
        group_df["IIS_smooth_core"] = (
            DYNAMIC_SMOOTHING_BLEND * group_df["IIS_fast"]
            + (1.0 - DYNAMIC_SMOOTHING_BLEND) * group_df["IIS_causal_median"]
        )
        if "RES" in group_df.columns:
            group_df["RES"] = pd.to_numeric(group_df.get("RES"), errors="coerce")
            group_df["RES_causal_mean"] = group_df["RES"].rolling(window=DYNAMIC_ROLLING_WINDOW, min_periods=1).mean()
            group_df["RES_fast"] = group_df["RES"].ewm(alpha=max(DYNAMIC_EWMA_ALPHA * 0.85, 0.1), adjust=False).mean()
            group_df["RES_dynamic"] = 0.65 * group_df["RES_fast"] + 0.35 * group_df["RES_causal_mean"]
        group_df = self._apply_asymmetric_state_update(group_df)
        if "RES_dynamic" in group_df.columns:
            iis_threshold = pd.to_numeric(group_df.get("IIS_threshold_empirical"), errors="coerce").dropna()
            res_threshold = pd.to_numeric(group_df.get("RES_threshold_empirical"), errors="coerce").dropna()
            if not iis_threshold.empty and not res_threshold.empty:
                group_df["state_map_4_dynamic"] = self._quadrant_state(
                    iis_series=pd.to_numeric(group_df["IIS_dynamic"], errors="coerce"),
                    res_series=pd.to_numeric(group_df["RES_dynamic"], errors="coerce"),
                    iis_threshold=float(iis_threshold.iloc[0]),
                    res_threshold=float(res_threshold.iloc[0]),
                )
        group_df["dIIS_raw_dt"] = group_df["IIS"].diff() / group_df["delta_t_sec"]
        group_df["dIIS_dynamic_dt"] = group_df["IIS_dynamic"].diff() / group_df["delta_t_sec"]
        group_df["dynamic_formula_note"] = (
            "Причинная динамика IISD: только прошлые окна; быстрый вход в ухудшение, медленное восстановление, "
            "коэффициент перехода зависит от прошлой волатильности и стресс-драйва Gamma/V/Q."
        )
        return group_df

    def _recompute_iis(self, frame: pd.DataFrame, focus_version: str) -> pd.Series:
        """Пересчитывает статический IIS по компонентам для интервенционного сценария."""

        a_term = pd.to_numeric(frame.get("A"), errors="coerce").fillna(0.0)
        gamma_term = pd.to_numeric(frame.get("Gamma"), errors="coerce").fillna(0.0)
        v_term = pd.to_numeric(frame.get("V"), errors="coerce").fillna(0.0)
        q_term = pd.to_numeric(frame.get("Q"), errors="coerce").fillna(0.0)
        if focus_version == "IISVersion7":
            dataset_key = ""
            if "dataset" in frame.columns and not frame["dataset"].dropna().empty:
                dataset_key = str(frame["dataset"].dropna().iloc[0]).strip().lower()
            v7_params = dict(VERSION7_CALIBRATION)
            override = VERSION7_DATASET_OVERRIDES.get(dataset_key, {})
            for key, value in override.items():
                if key != "weights":
                    v7_params[key] = value
            shape_power = VERSION6_CALIBRATION["shape_power"]
            gamma_shape_power = VERSION6_CALIBRATION["gamma_shape_power"]

            def signed_power_series(series: pd.Series, power: float) -> pd.Series:
                return np.sign(series) * np.abs(series) ** power

            def bounded_positive(series: pd.Series) -> pd.Series:
                positive = series.clip(lower=0.0)
                return positive / (1.0 + positive)

            def sumigron_series(columns: list[pd.Series]) -> pd.Series:
                terms = pd.concat(columns, axis=1).astype(float)
                local_mean = terms.mean(axis=1)
                local_std = terms.std(axis=1, ddof=0)
                local_z = terms.sub(local_mean, axis=0).div(local_std + 1e-6, axis=0)
                drive = 0.60 * local_z + 0.40 * local_z.abs()
                shifted = drive.sub(drive.max(axis=1), axis=0) * float(v7_params["sumigron_temperature"])
                exp_values = np.exp(np.clip(shifted, -50.0, 50.0))
                weights = exp_values.div(exp_values.sum(axis=1).replace(0.0, np.nan), axis=0).fillna(1.0 / terms.shape[1])
                attentive_mean = (weights * terms).sum(axis=1)
                attentive_var = (weights * terms.sub(attentive_mean, axis=0).pow(2)).sum(axis=1)
                attentive_std = np.sqrt(attentive_var)
                attentive_energy = np.log1p((weights * terms.pow(2)).sum(axis=1))
                sign = np.sign(attentive_mean)
                sign = sign.where(sign != 0.0, np.sign(terms.sum(axis=1)))
                sign = sign.replace(0.0, 1.0)
                structured = sign * (
                    float(v7_params["sumigron_structure_weight"]) * bounded_positive(attentive_std)
                    + float(v7_params["sumigron_energy_weight"]) * bounded_positive(attentive_energy)
                )
                return float(v7_params["sumigron_level_weight"]) * attentive_mean + structured

            shaped_a = signed_power_series(a_term, shape_power)
            shaped_gamma = signed_power_series(gamma_term, gamma_shape_power)
            shaped_v = signed_power_series(v_term, shape_power)
            shaped_q = signed_power_series(q_term, shape_power)
            gamma_pos = shaped_gamma.clip(lower=0.0)

            reg_signature = sumigron_series([0.72 * shaped_q, 0.52 * shaped_v, 0.18 * shaped_a, -0.24 * gamma_pos])
            mob_signature = sumigron_series([-0.06 * shaped_q, 0.28 * shaped_v, -0.48 * gamma_pos, 0.10 * shaped_a])
            dep_signature = sumigron_series([-0.60 * shaped_q, -0.50 * shaped_v, 0.36 * gamma_pos, -0.20 * shaped_a])

            logits = pd.DataFrame({"reg": reg_signature, "mob": mob_signature, "dep": dep_signature})
            shifted = logits.sub(logits.max(axis=1), axis=0) * float(v7_params["gate_temperature"])
            exp_values = np.exp(np.clip(shifted, -50.0, 50.0))
            gates = exp_values.div(exp_values.sum(axis=1).replace(0.0, np.nan), axis=0).fillna(1.0 / 3.0)

            synergy_reg = np.sign(shaped_q + shaped_v) * np.sqrt(np.abs(shaped_q * shaped_v) + 1e-9)
            conflict = (shaped_a - shaped_v).abs() + 0.65 * (shaped_q - shaped_v).abs()
            reg_level = float(v7_params["reg_iis_base"]) + float(v7_params["reg_iis_amp"]) * np.tanh(
                reg_signature + 0.20 * synergy_reg
            )
            mob_level = float(v7_params["mob_iis_base"]) + float(v7_params["mob_iis_amp"]) * np.tanh(mob_signature)
            dep_level = float(v7_params["dep_iis_base"]) + float(v7_params["dep_iis_amp"]) * np.tanh(dep_signature)

            regime_score = gates["reg"] * reg_level + gates["mob"] * mob_level + gates["dep"] * dep_level
            entropy = -(gates * np.log(gates + 1e-9)).sum(axis=1) / np.log(3.0)
            regime_balance = gates["reg"] - gates["dep"]
            iis_score = (
                regime_score
                - float(v7_params["transition_entropy_weight"]) * entropy
                + float(v7_params["regime_balance_weight"]) * regime_balance
                - float(v7_params["conflict_penalty"]) * conflict
            )
            return np.clip(iis_score, 0.0, 1.0)

        if focus_version == "IISVersion6":
            weights = MODEL_CONFIGS["IISVersion6"]["weights"]
            shape_power = VERSION6_CALIBRATION["shape_power"]
            gamma_shape_power = VERSION6_CALIBRATION["gamma_shape_power"]

            def signed_power_series(series: pd.Series, power: float) -> pd.Series:
                return np.sign(series) * np.abs(series) ** power

            shaped_a = signed_power_series(a_term, shape_power)
            shaped_gamma = signed_power_series(gamma_term, gamma_shape_power)
            shaped_v = signed_power_series(v_term, shape_power)
            shaped_q = signed_power_series(q_term, shape_power)
            gamma_pos = shaped_gamma.clip(lower=0.0)

            reg_logit = 0.60 * shaped_q + 0.45 * shaped_v + 0.15 * shaped_a - 0.30 * gamma_pos
            mob_logit = -0.05 * shaped_q + 0.20 * shaped_v - 0.55 * gamma_pos - 0.08 * shaped_a
            dep_logit = -0.70 * shaped_q - 0.55 * shaped_v - 0.20 * shaped_a + 0.35 * gamma_pos
            logits = pd.DataFrame({"reg": reg_logit, "mob": mob_logit, "dep": dep_logit})
            shifted = logits.sub(logits.max(axis=1), axis=0) * VERSION6_CALIBRATION["gate_temperature"]
            exp_values = np.exp(np.clip(shifted, -50.0, 50.0))
            gates = exp_values.div(exp_values.sum(axis=1).replace(0.0, np.nan), axis=0).fillna(1.0 / 3.0)

            synergy_reg = np.sign(shaped_q + shaped_v) * np.sqrt(np.abs(shaped_q * shaped_v) + 1e-9)
            conflict = (shaped_a - shaped_v).abs() + 0.75 * (shaped_q - shaped_v).abs()
            reg_level = VERSION6_CALIBRATION["reg_iis_base"] + VERSION6_CALIBRATION["reg_iis_amp"] * np.tanh(
                0.75 * shaped_q + 0.55 * shaped_v + 0.18 * shaped_a - 0.25 * gamma_pos + 0.25 * synergy_reg
            )
            mob_level = VERSION6_CALIBRATION["mob_iis_base"] + VERSION6_CALIBRATION["mob_iis_amp"] * np.tanh(
                -0.08 * shaped_q + 0.32 * shaped_v - 0.52 * gamma_pos + 0.08 * shaped_a
            )
            dep_level = VERSION6_CALIBRATION["dep_iis_base"] + VERSION6_CALIBRATION["dep_iis_amp"] * np.tanh(
                -0.62 * shaped_q - 0.52 * shaped_v - 0.18 * shaped_a + 0.38 * gamma_pos
            )
            regime_score = gates["reg"] * reg_level + gates["mob"] * mob_level + gates["dep"] * dep_level
            entropy = -(gates * np.log(gates + 1e-9)).sum(axis=1) / np.log(3.0)
            regime_balance = gates["reg"] - gates["dep"]
            iis_score = (
                regime_score
                - VERSION6_CALIBRATION["transition_entropy_weight"] * entropy
                + VERSION6_CALIBRATION["regime_balance_weight"] * regime_balance
                - VERSION6_CALIBRATION["conflict_penalty"] * conflict
            )
            return np.clip(iis_score, 0.0, 1.0)

        if focus_version == "IISVersion5":
            weights = MODEL_CONFIGS["IISVersion5"]["weights"]
            core_linear = (
                weights.get("A", 0.0) * a_term
                - weights.get("Gamma", 0.0) * gamma_term
                + weights.get("V", 0.0) * v_term
                + weights.get("Q", 0.0) * q_term
            )
            centered = core_linear - VERSION5_CALIBRATION["output_center"]
            contrast_drive = (
                VERSION5_CALIBRATION["output_gain"] * centered
                + VERSION5_CALIBRATION["output_curve"] * (centered**3)
                + VERSION5_CALIBRATION["output_balance_coupling"] * (q_term - v_term)
                - VERSION5_CALIBRATION["output_gamma_brake"] * gamma_term.clip(lower=0.0)
            )
            base_score = 1.0 / (1.0 + np.exp(-core_linear))
            contrast_score = 0.5 + 0.5 * np.tanh(contrast_drive)
            mix = VERSION5_CALIBRATION["output_contrast_mix"]
            return np.clip((1.0 - mix) * base_score + mix * contrast_score, 0.0, 1.0)

        weights = MODEL_CONFIGS["IISVersion4"]["weights"]
        raw_score = (
            weights.get("A", 0.0) * a_term
            - weights.get("Gamma", 0.0) * gamma_term
            + weights.get("V", 0.0) * v_term
            + weights.get("Q", 0.0) * q_term
        )
        return 1.0 / (1.0 + np.exp(-raw_score))

    def _plot_intervention(
        self,
        merged_df: pd.DataFrame,
        dataset: str,
        mode: str,
        focus_version: str,
        source_key: str,
        component_name: str,
        plot_path: Path,
    ) -> None:
        """Рисует исходную и возмущённую траектории для интервенционного сценария."""

        figure, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
        axes[0].plot(merged_df["time_sec"], merged_df["IIS"], color="#78909C", linewidth=1.6, label="IIS raw")
        axes[0].plot(merged_df["time_sec"], merged_df["IIS_intervened"], color="#E53935", linewidth=1.7, label="IIS after intervention")
        axes[0].plot(merged_df["time_sec"], merged_df["IIS_dynamic"], color="#00897B", linewidth=2.1, label="IIS dynamic")
        axes[0].plot(
            merged_df["time_sec"],
            merged_df["IIS_dynamic_intervened"],
            color="#5E35B1",
            linewidth=2.1,
            label="IIS dynamic after intervention",
        )
        axes[0].set_ylabel("IIS")
        axes[0].grid(alpha=0.22)
        axes[0].legend(loc="best")

        axes[1].plot(merged_df["time_sec"], merged_df[component_name], color="#1E88E5", linewidth=1.5, label=component_name)
        axes[1].plot(
            merged_df["time_sec"],
            merged_df[f"{component_name}_intervened"],
            color="#FB8C00",
            linewidth=1.5,
            label=f"{component_name} intervened",
        )
        affected = merged_df.loc[merged_df["intervention_mask"] == True, "time_sec"]  # noqa: E712
        if not affected.empty:
            axes[1].axvspan(float(affected.min()), float(affected.max()), color="#FFF59D", alpha=0.3, label="intervention window")
        axes[1].set_xlabel("Время окна, сек")
        axes[1].set_ylabel(component_name)
        axes[1].grid(alpha=0.22)
        axes[1].legend(loc="best")

        figure.suptitle(f"Интервенция {component_name}: {dataset.upper()} / {mode} / {focus_version} / {source_key}")
        figure.tight_layout()
        figure.savefig(plot_path, dpi=PLOT_DPI)
        plt.close(figure)

    def _apply_asymmetric_state_update(self, group_df: pd.DataFrame) -> pd.DataFrame:
        """Строит динамический индекс с быстрым ответом и более медленным восстановлением."""

        stress_drive_series = self._stress_drive(group_df)
        response_gains = []
        recovery_gains = []
        dynamic_values = []
        modes = []

        previous_dynamic = np.nan
        for index, row in group_df.iterrows():
            raw_value = float(row["IIS"])
            core_value = float(row["IIS_smooth_core"])
            volatility = float(row["IIS_volatility"])
            volatility_norm = float(np.clip(volatility / DYNAMIC_VOLATILITY_SCALE, 0.0, 1.0))
            stress_drive = float(stress_drive_series.loc[index])

            response_gain = float(
                np.clip(
                    DYNAMIC_RESPONSE_ALPHA_BASE + DYNAMIC_RESPONSE_ALPHA_SCALE * stress_drive + 0.10 * volatility_norm,
                    DYNAMIC_GAIN_MIN,
                    DYNAMIC_GAIN_MAX,
                )
            )
            recovery_gain = float(
                np.clip(
                    DYNAMIC_RECOVERY_ALPHA_BASE + DYNAMIC_RECOVERY_ALPHA_SCALE * (1.0 - stress_drive) - 0.04 * volatility_norm,
                    DYNAMIC_GAIN_MIN,
                    DYNAMIC_GAIN_MAX,
                )
            )

            if not np.isfinite(previous_dynamic):
                dynamic_value = raw_value
                mode = "init"
            else:
                target_value = (1.0 - row["IIS_smooth_weight"]) * raw_value + row["IIS_smooth_weight"] * core_value
                if target_value <= previous_dynamic:
                    gain = response_gain
                    mode = "response"
                else:
                    gain = recovery_gain
                    mode = "recovery"
                dynamic_value = previous_dynamic + gain * (target_value - previous_dynamic)

            response_gains.append(response_gain)
            recovery_gains.append(recovery_gain)
            dynamic_values.append(dynamic_value)
            modes.append(mode)
            previous_dynamic = dynamic_value

        group_df["stress_drive"] = stress_drive_series.to_numpy()
        group_df["response_gain"] = response_gains
        group_df["recovery_gain"] = recovery_gains
        group_df["dynamic_mode"] = modes
        group_df["IIS_dynamic"] = dynamic_values
        return group_df

    def _quadrant_state(
        self,
        *,
        iis_series: pd.Series,
        res_series: pd.Series,
        iis_threshold: float,
        res_threshold: float,
    ) -> pd.Series:
        """Классифицирует точки динамической карты по четырём квадрантам."""

        states: list[str] = []
        for iis_value, res_value in zip(iis_series.to_numpy(), res_series.to_numpy(), strict=False):
            if not np.isfinite(iis_value) or not np.isfinite(res_value):
                states.append("")
            elif iis_value >= iis_threshold and res_value >= res_threshold:
                states.append("regulated")
            elif iis_value < iis_threshold and res_value >= res_threshold:
                states.append("mobilized")
            elif iis_value >= iis_threshold and res_value < res_threshold:
                states.append("fragile")
            else:
                states.append("depleted")
        return pd.Series(states, index=iis_series.index, dtype=str)

    def _stress_drive(self, group_df: pd.DataFrame) -> pd.Series:
        """Строит безразмерную оценку текущей стрессовой нагрузки по прямым компонентам."""

        gamma_term = self._safe_series(group_df.get("Gamma"), len(group_df))
        v_term = self._safe_series(group_df.get("V"), len(group_df))
        q_term = self._safe_series(group_df.get("Q"), len(group_df))

        gamma_drive = np.clip((gamma_term + 1.0) / 2.0, 0.0, 1.0)
        v_drive = np.clip((1.0 - v_term) / 2.0, 0.0, 1.0)
        q_drive = np.clip((1.0 - q_term) / 2.0, 0.0, 1.0)
        return pd.Series(0.45 * gamma_drive + 0.35 * v_drive + 0.20 * q_drive, index=group_df.index, dtype=float)

    def _safe_series(self, series: pd.Series | None, length: int) -> np.ndarray:
        """Возвращает числовой массив с заменой пропусков нейтральным нулём."""

        if series is None:
            return np.zeros(length, dtype=float)
        numeric = pd.to_numeric(series, errors="coerce").fillna(0.0)
        return numeric.to_numpy(dtype=float)
