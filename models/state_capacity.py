"""Анализ ёмкости модели по числу статистически различимых состояний."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
from scipy.stats import kruskal, mannwhitneyu
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import (
    adjusted_rand_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    normalized_mutual_info_score,
    silhouette_score,
)
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import RobustScaler

from config.settings import OUTPUT_DIR, PLOT_DPI, PLOTS_DIR

LOGGER = logging.getLogger("iis_state_capacity")

MODEL_SPACE_CANDIDATES = ("IIS", "Q", "A", "Gamma", "V")
PAIRWISE_CANDIDATES = ("IIS", "Q", "A", "Gamma", "V")
PREFERRED_LABEL_ORDER = ("baseline", "amusement", "disbalance", "stress", "meditation")


@dataclass(slots=True)
class RegimeResult:
    """Сводка по одной версии модели в одном режиме."""

    dataset: str
    mode: str
    version: str
    regime: str
    labeled_state_count: int
    pairwise_distinguishable_pairs: int
    largest_fully_distinguishable_label_set: int
    supported_labels: list[str]
    supported_pair_labels: list[str]
    primary_measure: str
    feature_space: list[str]
    best_k_kmeans: int | None
    best_k_gmm: int | None
    consensus_supported_k: int | None
    consensus_support_strength: str
    label_alignment_ari: float | None
    label_alignment_nmi: float | None
    va_alignment_ari: float | None
    va_alignment_nmi: float | None
    confident_state_count: int
    likely_state_count: int
    no_evidence_for_above: int
    capacity_verdict: str
    capacity_note: str
    dynamic_vs_static_note: str | None = None


class StateCapacityAnalyzer:
    """Оценивает, сколько состояний реально поддерживается выходами модели."""

    def __init__(
        self,
        output_dir: Path | None = None,
        plots_dir: Path | None = None,
        *,
        bootstrap_iterations: int = 24,
        random_state: int = 42,
        k_min: int = 2,
        k_max: int = 5,
        min_class_samples: int = 20,
        max_scatter_points: int = 2500,
    ) -> None:
        self.output_dir = Path(output_dir or OUTPUT_DIR)
        self.plots_dir = Path(plots_dir or (PLOTS_DIR / "state_capacity"))
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.plots_dir.mkdir(parents=True, exist_ok=True)
        self.bootstrap_iterations = int(max(4, bootstrap_iterations))
        self.random_state = int(random_state)
        self.k_min = int(max(2, k_min))
        self.k_max = int(max(self.k_min, k_max))
        self.min_class_samples = int(max(6, min_class_samples))
        self.max_scatter_points = int(max(200, max_scatter_points))

    def analyze_dataset_mode(
        self,
        *,
        dataset: str,
        mode: str,
        versions: list[str] | None = None,
        include_dynamic: bool = True,
    ) -> dict[str, Any]:
        """Полностью анализирует один dataset/mode по всем доступным версиям и режимам."""

        results_path = self.output_dir / f"results_{dataset}_{mode}.csv"
        if not results_path.exists():
            raise FileNotFoundError(f"Не найден файл результатов: {results_path}")

        results_df = pd.read_csv(results_path, low_memory=False)
        if results_df.empty:
            raise ValueError(f"Файл результатов пуст: {results_path}")

        available_versions = [str(value) for value in results_df.get("version", pd.Series(dtype=str)).dropna().unique()]
        selected_versions = [version for version in (versions or available_versions) if version in available_versions]
        if not selected_versions:
            raise ValueError(f"В {results_path.name} нет ни одной запрошенной версии.")

        dynamic_frames = self._load_dynamic_frames(dataset=dataset, mode=mode) if include_dynamic else {}

        separation_records: list[dict[str, Any]] = []
        cluster_records: list[dict[str, Any]] = []
        regime_results: list[RegimeResult] = []
        plot_paths: list[str] = []

        for version in selected_versions:
            static_df = results_df.loc[results_df["version"].astype(str) == version].copy()
            regime_result, regime_pairs, regime_clusters, regime_plots = self._analyze_regime(
                dataset=dataset,
                mode=mode,
                version=version,
                regime="static",
                frame=static_df,
            )
            if regime_result is not None:
                regime_results.append(regime_result)
                separation_records.extend(regime_pairs)
                cluster_records.extend(regime_clusters)
                plot_paths.extend(regime_plots)

            dynamic_df = dynamic_frames.get(version)
            if dynamic_df is not None and not dynamic_df.empty:
                regime_result, regime_pairs, regime_clusters, regime_plots = self._analyze_regime(
                    dataset=dataset,
                    mode=mode,
                    version=version,
                    regime="dynamic",
                    frame=dynamic_df,
                )
                if regime_result is not None:
                    regime_results.append(regime_result)
                    separation_records.extend(regime_pairs)
                    cluster_records.extend(regime_clusters)
                    plot_paths.extend(regime_plots)

        regime_results = self._annotate_dynamic_differences(regime_results)
        separation_df = pd.DataFrame.from_records(separation_records)
        cluster_df = pd.DataFrame.from_records(cluster_records)

        separation_path = self.output_dir / f"state_separation_{dataset}_{mode}.csv"
        cluster_path = self.output_dir / f"state_clusters_{dataset}_{mode}.csv"
        capacity_path = self.output_dir / f"state_capacity_{dataset}_{mode}.json"
        separation_df.to_csv(separation_path, index=False, encoding="utf-8-sig")
        cluster_df.to_csv(cluster_path, index=False, encoding="utf-8-sig")

        payload = self._build_payload(
            dataset=dataset,
            mode=mode,
            regime_results=regime_results,
            separation_df=separation_df,
            cluster_df=cluster_df,
            separation_path=separation_path,
            cluster_path=cluster_path,
            capacity_path=capacity_path,
            plot_paths=plot_paths,
        )
        capacity_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        self._augment_existing_summary(dataset=dataset, mode=mode, payload=payload)
        self._print_console_report(dataset=dataset, mode=mode, regime_results=regime_results)
        return payload

    def _load_dynamic_frames(self, *, dataset: str, mode: str) -> dict[str, pd.DataFrame]:
        """Находит уже сохранённые динамические CSV для разных версий."""

        dynamic_map: dict[str, pd.DataFrame] = {}
        pattern = f"dynamic_iis_{dataset}_{mode}_*_causal.csv"
        for csv_path in sorted(self.output_dir.glob(pattern)):
            lower_name = csv_path.name.lower()
            version_name = None
            for candidate in ("IISVersion1", "IISVersion2", "IISVersion3", "IISVersion4", "IISVersion5", "IISVersion6", "IISVersion7"):
                if candidate.lower() in lower_name:
                    version_name = candidate
                    break
            if version_name is None:
                continue
            frame = pd.read_csv(csv_path, low_memory=False)
            if frame.empty:
                continue
            dynamic_map[version_name] = frame
        return dynamic_map

    def _analyze_regime(
        self,
        *,
        dataset: str,
        mode: str,
        version: str,
        regime: str,
        frame: pd.DataFrame,
    ) -> tuple[RegimeResult | None, list[dict[str, Any]], list[dict[str, Any]], list[str]]:
        """Анализирует одну конкретную версию модели в одном режиме."""

        if frame.empty:
            return None, [], [], []

        primary_measure = "IIS_dynamic" if regime == "dynamic" else "IIS"
        if primary_measure not in frame.columns:
            return None, [], [], []

        local_df = frame.copy()
        local_df["label"] = local_df.get("label", pd.Series(dtype=str)).fillna("unknown").astype(str)
        local_df[primary_measure] = pd.to_numeric(local_df.get(primary_measure), errors="coerce")
        local_df = local_df.dropna(subset=[primary_measure])
        if local_df.empty:
            return None, [], [], []

        measure_columns = [primary_measure]
        measure_columns.extend(
            column
            for column in PAIRWISE_CANDIDATES[1:]
            if column in local_df.columns and pd.to_numeric(local_df[column], errors="coerce").notna().sum() >= self.min_class_samples
        )
        feature_space = [primary_measure]
        feature_space.extend(
            column
            for column in MODEL_SPACE_CANDIDATES[1:]
            if column in local_df.columns and pd.to_numeric(local_df[column], errors="coerce").notna().sum() >= self.min_class_samples
        )

        pairwise_records, aggregate_pairs, label_counts = self._pairwise_analysis(
            local_df=local_df,
            dataset=dataset,
            mode=mode,
            version=version,
            regime=regime,
            measure_columns=measure_columns,
            feature_space=feature_space,
            primary_measure=primary_measure,
        )

        cluster_records, cluster_summary = self._cluster_analysis(
            local_df=local_df,
            dataset=dataset,
            mode=mode,
            version=version,
            regime=regime,
            feature_space=feature_space,
        )

        plot_paths = self._plot_geometry(
            local_df=local_df,
            dataset=dataset,
            mode=mode,
            version=version,
            regime=regime,
            primary_measure=primary_measure,
            feature_space=feature_space,
            cluster_summary=cluster_summary,
        )

        supported_pairs = aggregate_pairs.loc[aggregate_pairs["pair_supported"] == True].copy() if not aggregate_pairs.empty else pd.DataFrame()  # noqa: E712
        supported_labels = self._largest_supported_label_set(supported_pairs)
        verdict, confident_count, likely_count, no_evidence_above, support_strength, note = self._build_capacity_verdict(
            supported_labels=supported_labels,
            aggregate_pairs=aggregate_pairs,
            cluster_summary=cluster_summary,
        )

        label_alignment_ari = cluster_summary.get("label_alignment_ari")
        label_alignment_nmi = cluster_summary.get("label_alignment_nmi")
        va_alignment_ari = cluster_summary.get("va_alignment_ari")
        va_alignment_nmi = cluster_summary.get("va_alignment_nmi")
        best_k_kmeans = cluster_summary.get("best_k_kmeans")
        best_k_gmm = cluster_summary.get("best_k_gmm")
        consensus_supported_k = cluster_summary.get("consensus_supported_k")

        regime_result = RegimeResult(
            dataset=dataset,
            mode=mode,
            version=version,
            regime=regime,
            labeled_state_count=int(len(label_counts)),
            pairwise_distinguishable_pairs=int(len(supported_pairs)),
            largest_fully_distinguishable_label_set=int(len(supported_labels)),
            supported_labels=list(supported_labels),
            supported_pair_labels=sorted(set(supported_pairs["pair_key"].astype(str).tolist())) if not supported_pairs.empty else [],
            primary_measure=primary_measure,
            feature_space=feature_space,
            best_k_kmeans=int(best_k_kmeans) if best_k_kmeans is not None else None,
            best_k_gmm=int(best_k_gmm) if best_k_gmm is not None else None,
            consensus_supported_k=int(consensus_supported_k) if consensus_supported_k is not None else None,
            consensus_support_strength=support_strength,
            label_alignment_ari=float(label_alignment_ari) if label_alignment_ari is not None and np.isfinite(label_alignment_ari) else None,
            label_alignment_nmi=float(label_alignment_nmi) if label_alignment_nmi is not None and np.isfinite(label_alignment_nmi) else None,
            va_alignment_ari=float(va_alignment_ari) if va_alignment_ari is not None and np.isfinite(va_alignment_ari) else None,
            va_alignment_nmi=float(va_alignment_nmi) if va_alignment_nmi is not None and np.isfinite(va_alignment_nmi) else None,
            confident_state_count=int(confident_count),
            likely_state_count=int(likely_count),
            no_evidence_for_above=int(no_evidence_above),
            capacity_verdict=verdict,
            capacity_note=note,
        )
        return regime_result, pairwise_records, cluster_records, plot_paths

    def _pairwise_analysis(
        self,
        *,
        local_df: pd.DataFrame,
        dataset: str,
        mode: str,
        version: str,
        regime: str,
        measure_columns: list[str],
        feature_space: list[str],
        primary_measure: str,
    ) -> tuple[list[dict[str, Any]], pd.DataFrame, dict[str, int]]:
        """Считает попарную различимость по всем доступным выходам модели."""

        records: list[dict[str, Any]] = []
        label_counts = local_df["label"].value_counts()
        label_counts = label_counts.loc[label_counts >= self.min_class_samples].to_dict()
        labels = self._ordered_labels(list(label_counts.keys()))
        if len(labels) < 2:
            return records, pd.DataFrame(), label_counts

        multivariate_distances = self._multivariate_pair_distances(local_df=local_df, labels=labels, feature_space=feature_space)

        for measure in measure_columns:
            measure_df = local_df[["label", measure]].copy()
            measure_df[measure] = pd.to_numeric(measure_df[measure], errors="coerce")
            measure_df = measure_df.dropna(subset=[measure])
            measure_df = measure_df.loc[measure_df["label"].isin(labels)].copy()
            if measure_df.empty:
                continue

            groups = [measure_df.loc[measure_df["label"] == label, measure].to_numpy(dtype=float) for label in labels]
            global_statistic = np.nan
            global_p_value = np.nan
            if len(groups) >= 3 and all(group.size >= 2 for group in groups):
                try:
                    global_statistic, global_p_value = kruskal(*groups, nan_policy="omit")
                except Exception:
                    global_statistic, global_p_value = np.nan, np.nan

            records.append(
                {
                    "dataset": dataset,
                    "mode": mode,
                    "version": version,
                    "regime": regime,
                    "record_type": "global_test",
                    "measure": measure,
                    "label_a": "",
                    "label_b": "",
                    "pair_key": "",
                    "n_a": np.nan,
                    "n_b": np.nan,
                    "mean_a": np.nan,
                    "mean_b": np.nan,
                    "mean_diff": np.nan,
                    "abs_mean_diff": np.nan,
                    "effect_size": np.nan,
                    "distribution_overlap": np.nan,
                    "centroid_distance": np.nan,
                    "test_name": "kruskal",
                    "test_statistic": float(global_statistic) if np.isfinite(global_statistic) else np.nan,
                    "p_value": float(global_p_value) if np.isfinite(global_p_value) else np.nan,
                    "p_value_holm": np.nan,
                    "separation_grade": "global",
                    "grade_score": np.nan,
                    "supported_measures": np.nan,
                    "pair_supported": np.nan,
                    "primary_measure": primary_measure,
                    "primary_measure_grade": np.nan,
                    "centroid_grade": np.nan,
                }
            )

            pair_buffer: list[dict[str, Any]] = []
            for label_a, label_b in combinations(labels, 2):
                first = measure_df.loc[measure_df["label"] == label_a, measure].to_numpy(dtype=float)
                second = measure_df.loc[measure_df["label"] == label_b, measure].to_numpy(dtype=float)
                if first.size < 2 or second.size < 2:
                    continue
                test_statistic, p_value = mannwhitneyu(first, second, alternative="two-sided")
                mean_a = float(np.mean(first))
                mean_b = float(np.mean(second))
                effect_size = self._cohens_d(second, first)
                overlap = self._distribution_overlap(first, second)
                mean_diff = mean_b - mean_a
                centroid_distance = multivariate_distances.get(f"{label_a}__{label_b}", np.nan)
                pair_buffer.append(
                    {
                        "dataset": dataset,
                        "mode": mode,
                        "version": version,
                        "regime": regime,
                        "record_type": "pairwise_measure",
                        "measure": measure,
                        "label_a": label_a,
                        "label_b": label_b,
                        "pair_key": f"{label_a}__{label_b}",
                        "n_a": int(first.size),
                        "n_b": int(second.size),
                        "mean_a": mean_a,
                        "mean_b": mean_b,
                        "mean_diff": mean_diff,
                        "abs_mean_diff": abs(mean_diff),
                        "effect_size": effect_size,
                        "distribution_overlap": overlap,
                        "centroid_distance": centroid_distance,
                        "test_name": "mannwhitneyu",
                        "test_statistic": float(test_statistic) if np.isfinite(test_statistic) else np.nan,
                        "p_value": float(p_value) if np.isfinite(p_value) else np.nan,
                        "p_value_holm": np.nan,
                        "separation_grade": "none",
                        "grade_score": 0,
                        "supported_measures": np.nan,
                        "pair_supported": False,
                        "primary_measure": primary_measure,
                        "primary_measure_grade": np.nan,
                        "centroid_grade": np.nan,
                    }
                )

            holm_values = self._holm_correction([record["p_value"] for record in pair_buffer])
            for record, corrected in zip(pair_buffer, holm_values):
                record["p_value_holm"] = corrected
                grade_name, grade_score = self._grade_pair(
                    effect_size=record["effect_size"],
                    overlap=record["distribution_overlap"],
                    p_value_holm=record["p_value_holm"],
                )
                record["separation_grade"] = grade_name
                record["grade_score"] = grade_score
            records.extend(pair_buffer)

        pair_measure_df = pd.DataFrame.from_records([record for record in records if record["record_type"] == "pairwise_measure"])
        if pair_measure_df.empty:
            return records, pd.DataFrame(), label_counts

        aggregate_records: list[dict[str, Any]] = []
        for pair_key, pair_df in pair_measure_df.groupby("pair_key", sort=True):
            first_row = pair_df.iloc[0]
            supported_measures = int((pair_df["grade_score"] >= 2).sum())
            primary_row = pair_df.loc[pair_df["measure"] == primary_measure]
            primary_grade = int(primary_row["grade_score"].iloc[0]) if not primary_row.empty else 0
            centroid_distance = float(pair_df["centroid_distance"].dropna().iloc[0]) if pair_df["centroid_distance"].notna().any() else np.nan
            centroid_grade = 3 if np.isfinite(centroid_distance) and centroid_distance >= 1.25 else 2 if np.isfinite(centroid_distance) and centroid_distance >= 0.80 else 1 if np.isfinite(centroid_distance) and centroid_distance >= 0.45 else 0
            pair_supported = bool(primary_grade >= 2 or supported_measures >= 2 or (supported_measures >= 1 and centroid_grade >= 2))
            if primary_grade >= 3 or (supported_measures >= 3 and centroid_grade >= 2):
                aggregate_grade = "strong"
                aggregate_score = 3
            elif pair_supported:
                aggregate_grade = "moderate"
                aggregate_score = 2
            elif primary_grade >= 1 or supported_measures >= 1 or centroid_grade >= 1:
                aggregate_grade = "weak"
                aggregate_score = 1
            else:
                aggregate_grade = "none"
                aggregate_score = 0

            aggregate_records.append(
                {
                    "dataset": dataset,
                    "mode": mode,
                    "version": version,
                    "regime": regime,
                    "record_type": "pairwise_aggregate",
                    "measure": "aggregate",
                    "label_a": first_row["label_a"],
                    "label_b": first_row["label_b"],
                    "pair_key": pair_key,
                    "n_a": int(first_row["n_a"]),
                    "n_b": int(first_row["n_b"]),
                    "mean_a": np.nan,
                    "mean_b": np.nan,
                    "mean_diff": np.nan,
                    "abs_mean_diff": np.nan,
                    "effect_size": np.nan,
                    "distribution_overlap": np.nan,
                    "centroid_distance": centroid_distance,
                    "test_name": "aggregate",
                    "test_statistic": np.nan,
                    "p_value": np.nan,
                    "p_value_holm": np.nan,
                    "separation_grade": aggregate_grade,
                    "grade_score": aggregate_score,
                    "supported_measures": supported_measures,
                    "pair_supported": pair_supported,
                    "primary_measure": primary_measure,
                    "primary_measure_grade": primary_grade,
                    "centroid_grade": centroid_grade,
                }
            )

        records.extend(aggregate_records)
        aggregate_pairs = pd.DataFrame.from_records(aggregate_records)
        return records, aggregate_pairs, label_counts

    def _cluster_analysis(
        self,
        *,
        local_df: pd.DataFrame,
        dataset: str,
        mode: str,
        version: str,
        regime: str,
        feature_space: list[str],
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        """Оценивает внутреннее число поддерживаемых кластеров по пространству модели."""

        if len(feature_space) < 2:
            return [], {}

        base_columns = ["label", *feature_space]
        for optional_column in ("subject_id", "source_record_id", "valence", "arousal"):
            if optional_column in local_df.columns and optional_column not in base_columns:
                base_columns.append(optional_column)
        cluster_df = local_df[base_columns].copy()
        for column in feature_space:
            cluster_df[column] = pd.to_numeric(cluster_df[column], errors="coerce")
        cluster_df = cluster_df.dropna(subset=feature_space)
        if len(cluster_df) < max(40, self.k_min * 10):
            return [], {}

        scaler = RobustScaler()
        matrix = scaler.fit_transform(cluster_df[feature_space].to_numpy(dtype=float))
        labels_true = cluster_df["label"].astype(str).to_numpy()
        groups = self._bootstrap_groups(cluster_df)
        va_quadrant = self._make_va_quadrant(cluster_df)

        records: list[dict[str, Any]] = []
        cluster_assignments: dict[tuple[str, int], np.ndarray] = {}

        for algorithm in ("kmeans", "gmm"):
            for k_value in range(self.k_min, min(self.k_max, len(cluster_df) - 1) + 1):
                try:
                    model, predicted = self._fit_cluster_model(algorithm=algorithm, k_value=k_value, matrix=matrix)
                except Exception as error:
                    LOGGER.warning(
                        "Кластеризация не сошлась: dataset=%s mode=%s version=%s regime=%s algorithm=%s k=%s error=%s",
                        dataset,
                        mode,
                        version,
                        regime,
                        algorithm,
                        k_value,
                        error,
                    )
                    continue

                unique_clusters = np.unique(predicted)
                if unique_clusters.size < 2:
                    continue

                silhouette = float(silhouette_score(matrix, predicted))
                davies_bouldin = float(davies_bouldin_score(matrix, predicted))
                calinski_harabasz = float(calinski_harabasz_score(matrix, predicted))
                label_ari = self._safe_adjusted_rand(labels_true, predicted)
                label_nmi = self._safe_nmi(labels_true, predicted)
                va_ari = self._safe_adjusted_rand(va_quadrant, predicted) if va_quadrant is not None else np.nan
                va_nmi = self._safe_nmi(va_quadrant, predicted) if va_quadrant is not None else np.nan
                proportions = self._cluster_proportions(predicted, k_value)
                tiny_cluster_rate = float(np.min(proportions) < 0.10)
                cluster_size_cv = float(np.std(proportions) / max(np.mean(proportions), 1e-8))
                bootstrap_summary = self._bootstrap_cluster_stability(
                    matrix=matrix,
                    algorithm=algorithm,
                    k_value=k_value,
                    groups=groups,
                    reference_model=model,
                    reference_labels=predicted,
                )

                record = {
                    "dataset": dataset,
                    "mode": mode,
                    "version": version,
                    "regime": regime,
                    "algorithm": algorithm,
                    "k": int(k_value),
                    "n_samples": int(len(cluster_df)),
                    "n_features": int(len(feature_space)),
                    "feature_space": json.dumps(feature_space, ensure_ascii=False),
                    "silhouette_score": silhouette,
                    "davies_bouldin_index": davies_bouldin,
                    "calinski_harabasz_score": calinski_harabasz,
                    "label_ari": label_ari,
                    "label_nmi": label_nmi,
                    "va_ari": va_ari,
                    "va_nmi": va_nmi,
                    "bootstrap_mean_ari": bootstrap_summary["mean_ari"],
                    "bootstrap_std_ari": bootstrap_summary["std_ari"],
                    "centroid_drift_mean": bootstrap_summary["mean_drift"],
                    "centroid_drift_std": bootstrap_summary["std_drift"],
                    "tiny_cluster_rate": bootstrap_summary["tiny_rate"],
                    "mean_cluster_size_cv": bootstrap_summary["mean_size_cv"],
                    "effective_bootstrap_runs": bootstrap_summary["effective_runs"],
                    "full_sample_tiny_cluster": tiny_cluster_rate,
                    "full_sample_cluster_size_cv": cluster_size_cv,
                    "aic": float(model.aic(matrix)) if algorithm == "gmm" else np.nan,
                    "bic": float(model.bic(matrix)) if algorithm == "gmm" else np.nan,
                }
                records.append(record)
                cluster_assignments[(algorithm, k_value)] = predicted

        if not records:
            return [], {}

        cluster_metrics_df = pd.DataFrame.from_records(records)
        cluster_metrics_df = self._score_cluster_candidates(cluster_metrics_df)
        records = cluster_metrics_df.to_dict(orient="records")

        consensus_summary = self._build_cluster_summary(cluster_metrics_df)
        best_key = consensus_summary.get("best_assignment_key")
        if best_key is not None and best_key in cluster_assignments:
            consensus_summary["best_cluster_labels"] = cluster_assignments[best_key]
        return records, consensus_summary

    def _fit_cluster_model(self, *, algorithm: str, k_value: int, matrix: np.ndarray) -> tuple[Any, np.ndarray]:
        """Обучает один кластерный алгоритм."""

        if algorithm == "kmeans":
            model = KMeans(n_clusters=k_value, n_init=20, random_state=self.random_state)
            labels = model.fit_predict(matrix)
            return model, labels
        if algorithm == "gmm":
            model = GaussianMixture(
                n_components=k_value,
                covariance_type="full",
                random_state=self.random_state,
                n_init=4,
                reg_covar=1e-6,
            )
            labels = model.fit_predict(matrix)
            return model, labels
        raise ValueError(f"Неизвестный алгоритм кластеризации: {algorithm}")

    def _bootstrap_cluster_stability(
        self,
        *,
        matrix: np.ndarray,
        algorithm: str,
        k_value: int,
        groups: np.ndarray,
        reference_model: Any,
        reference_labels: np.ndarray,
    ) -> dict[str, float]:
        """Оценивает устойчивость кластеров по бутстрэп-подвыборкам."""

        rng = np.random.default_rng(self.random_state)
        unique_groups = np.unique(groups)
        reference_centers = self._extract_cluster_centers(reference_model)

        ari_values: list[float] = []
        drift_values: list[float] = []
        tiny_flags: list[float] = []
        size_cvs: list[float] = []

        for _ in range(self.bootstrap_iterations):
            if unique_groups.size >= max(k_value * 2, 8):
                sampled_groups = rng.choice(unique_groups, size=unique_groups.size, replace=True)
                sampled_indices = np.concatenate([np.flatnonzero(groups == group_value) for group_value in sampled_groups])
            else:
                sampled_indices = rng.integers(0, len(matrix), size=len(matrix))

            sampled_matrix = matrix[sampled_indices]
            if sampled_matrix.shape[0] < max(k_value * 8, 30):
                continue
            try:
                model, _ = self._fit_cluster_model(algorithm=algorithm, k_value=k_value, matrix=sampled_matrix)
            except Exception:
                continue

            predicted_full = model.predict(matrix)
            if np.unique(predicted_full).size < 2:
                continue

            ari_values.append(float(adjusted_rand_score(reference_labels, predicted_full)))
            drift_values.append(self._centroid_drift(reference_centers, self._extract_cluster_centers(model)))
            proportions = self._cluster_proportions(predicted_full, k_value)
            tiny_flags.append(float(np.min(proportions) < 0.10))
            size_cvs.append(float(np.std(proportions) / max(np.mean(proportions), 1e-8)))

        if not ari_values:
            return {
                "mean_ari": np.nan,
                "std_ari": np.nan,
                "mean_drift": np.nan,
                "std_drift": np.nan,
                "tiny_rate": np.nan,
                "mean_size_cv": np.nan,
                "effective_runs": 0,
            }

        return {
            "mean_ari": float(np.mean(ari_values)),
            "std_ari": float(np.std(ari_values, ddof=1)) if len(ari_values) > 1 else 0.0,
            "mean_drift": float(np.mean(drift_values)) if drift_values else np.nan,
            "std_drift": float(np.std(drift_values, ddof=1)) if len(drift_values) > 1 else 0.0,
            "tiny_rate": float(np.mean(tiny_flags)) if tiny_flags else np.nan,
            "mean_size_cv": float(np.mean(size_cvs)) if size_cvs else np.nan,
            "effective_runs": int(len(ari_values)),
        }

    def _score_cluster_candidates(self, cluster_df: pd.DataFrame) -> pd.DataFrame:
        """Строит сводный внутренний скор кандидатов k."""

        scored = cluster_df.copy()
        scored["internal_score"] = 0.0
        scored["alignment_score"] = 0.0
        scored["stable_support"] = False
        scored["best_for_algorithm"] = False

        for _, group_index in scored.groupby(["version", "regime", "algorithm"]).groups.items():
            group = scored.loc[group_index]
            silhouette_norm = self._series_minmax(group["silhouette_score"])
            inverse_dbi = 1.0 - self._series_minmax(group["davies_bouldin_index"])
            ch_norm = self._series_minmax(group["calinski_harabasz_score"])
            stability_norm = self._series_minmax(group["bootstrap_mean_ari"])
            inverse_tiny = 1.0 - self._series_minmax(group["tiny_cluster_rate"])
            scored.loc[group_index, "internal_score"] = (
                0.30 * silhouette_norm
                + 0.20 * inverse_dbi
                + 0.20 * ch_norm
                + 0.20 * stability_norm
                + 0.10 * inverse_tiny
            )

            label_support = 0.50 * self._series_minmax(group["label_ari"]) + 0.50 * self._series_minmax(group["label_nmi"])
            va_support = 0.50 * self._series_minmax(group["va_ari"]) + 0.50 * self._series_minmax(group["va_nmi"])
            scored.loc[group_index, "alignment_score"] = 0.70 * label_support + 0.30 * va_support.fillna(0.0)

            stable_mask = (
                (group["bootstrap_mean_ari"].fillna(-1.0) >= 0.55)
                & (group["tiny_cluster_rate"].fillna(1.0) <= 0.35)
                & (group["silhouette_score"].fillna(-1.0) > 0.0)
            )
            scored.loc[group_index, "stable_support"] = stable_mask.to_numpy(dtype=bool)
            best_index = group.assign(stable_rank=stable_mask.astype(int)).sort_values(
                ["stable_rank", "internal_score", "alignment_score", "k"],
                ascending=[False, False, False, True],
            ).index[0]
            scored.loc[group_index, "best_for_algorithm"] = False
            scored.loc[best_index, "best_for_algorithm"] = True
        return scored

    def _build_cluster_summary(self, cluster_df: pd.DataFrame) -> dict[str, Any]:
        """Выбирает лучший внутренний k и собирает краткую сводку по кластерам."""

        summary: dict[str, Any] = {}
        if cluster_df.empty:
            return summary

        summary["best_k_kmeans"] = self._best_k_for_algorithm(cluster_df, "kmeans")
        summary["best_k_gmm"] = self._best_k_for_algorithm(cluster_df, "gmm")

        stable_df = cluster_df.loc[cluster_df["stable_support"] == True].copy()  # noqa: E712
        source_df = stable_df if not stable_df.empty else cluster_df
        consensus = (
            source_df.groupby("k", as_index=False)
            .agg(
                mean_internal_score=("internal_score", "mean"),
                mean_alignment_score=("alignment_score", "mean"),
                mean_bootstrap_ari=("bootstrap_mean_ari", "mean"),
                mean_label_ari=("label_ari", "mean"),
                mean_label_nmi=("label_nmi", "mean"),
                mean_va_ari=("va_ari", "mean"),
                mean_va_nmi=("va_nmi", "mean"),
                stable_algorithm_count=("stable_support", "sum"),
            )
            .sort_values(
                ["stable_algorithm_count", "mean_internal_score", "mean_alignment_score", "k"],
                ascending=[False, False, False, True],
            )
        )
        if consensus.empty:
            return summary

        top_row = consensus.iloc[0]
        supported_k = int(top_row["k"])
        summary["consensus_supported_k"] = supported_k
        summary["consensus_mean_bootstrap_ari"] = float(top_row["mean_bootstrap_ari"]) if np.isfinite(top_row["mean_bootstrap_ari"]) else np.nan
        summary["label_alignment_ari"] = float(top_row["mean_label_ari"]) if np.isfinite(top_row["mean_label_ari"]) else np.nan
        summary["label_alignment_nmi"] = float(top_row["mean_label_nmi"]) if np.isfinite(top_row["mean_label_nmi"]) else np.nan
        summary["va_alignment_ari"] = float(top_row["mean_va_ari"]) if np.isfinite(top_row["mean_va_ari"]) else np.nan
        summary["va_alignment_nmi"] = float(top_row["mean_va_nmi"]) if np.isfinite(top_row["mean_va_nmi"]) else np.nan

        support_strength = "weak"
        if int(top_row["stable_algorithm_count"]) >= 2 and float(top_row["mean_bootstrap_ari"]) >= 0.70:
            support_strength = "strong"
        elif int(top_row["stable_algorithm_count"]) >= 1 and float(top_row["mean_bootstrap_ari"]) >= 0.55:
            support_strength = "moderate"
        summary["consensus_support_strength"] = support_strength

        best_assignment_row = source_df.loc[source_df["k"] == supported_k].sort_values(
            ["stable_support", "internal_score", "alignment_score", "algorithm"],
            ascending=[False, False, False, True],
        ).iloc[0]
        summary["best_assignment_key"] = (str(best_assignment_row["algorithm"]), int(best_assignment_row["k"]))
        return summary

    def _plot_geometry(
        self,
        *,
        local_df: pd.DataFrame,
        dataset: str,
        mode: str,
        version: str,
        regime: str,
        primary_measure: str,
        feature_space: list[str],
        cluster_summary: dict[str, Any],
    ) -> list[str]:
        """Сохраняет геометрические 2D-визуализации без претензии на доказательство."""

        plot_paths: list[str] = []
        plot_df = self._sample_for_plot(local_df.copy())
        if plot_df.empty:
            return plot_paths

        label_colors = {
            "baseline": "#2E7D32",
            "stress": "#C62828",
            "disbalance": "#F9A825",
            "amusement": "#1565C0",
            "meditation": "#00897B",
        }

        if primary_measure in plot_df.columns and "Q" in plot_df.columns:
            path = self.plots_dir / f"state_capacity_{dataset}_{mode}_{version.lower()}_{regime}_iis_q.png"
            self._scatter_plot(
                plot_df=plot_df,
                x_column=primary_measure,
                y_column="Q",
                color_column="label",
                color_map=label_colors,
                title=f"{version} {regime}: {primary_measure} vs Q",
                x_label=primary_measure,
                y_label="Q",
                output_path=path,
            )
            plot_paths.append(str(path))

        if primary_measure in plot_df.columns and "arousal" in plot_df.columns:
            path = self.plots_dir / f"state_capacity_{dataset}_{mode}_{version.lower()}_{regime}_iis_arousal.png"
            self._scatter_plot(
                plot_df=plot_df,
                x_column=primary_measure,
                y_column="arousal",
                color_column="label",
                color_map=label_colors,
                title=f"{version} {regime}: {primary_measure} vs arousal",
                x_label=primary_measure,
                y_label="arousal",
                output_path=path,
            )
            plot_paths.append(str(path))

        if "Q" in plot_df.columns and "valence" in plot_df.columns:
            path = self.plots_dir / f"state_capacity_{dataset}_{mode}_{version.lower()}_{regime}_q_valence.png"
            self._scatter_plot(
                plot_df=plot_df,
                x_column="Q",
                y_column="valence",
                color_column="label",
                color_map=label_colors,
                title=f"{version} {regime}: Q vs valence",
                x_label="Q",
                y_label="valence",
                output_path=path,
            )
            plot_paths.append(str(path))

        if len(feature_space) >= 2 and cluster_summary.get("best_cluster_labels") is not None:
            pca_path = self.plots_dir / f"state_capacity_{dataset}_{mode}_{version.lower()}_{regime}_pca_clusters.png"
            self._plot_pca_clusters(
                full_df=local_df,
                feature_space=feature_space,
                cluster_labels=np.asarray(cluster_summary["best_cluster_labels"]),
                title=f"{version} {regime}: PCA of model space",
                output_path=pca_path,
            )
            plot_paths.append(str(pca_path))
        return plot_paths

    def _plot_pca_clusters(
        self,
        *,
        full_df: pd.DataFrame,
        feature_space: list[str],
        cluster_labels: np.ndarray,
        title: str,
        output_path: Path,
    ) -> None:
        """Рисует PCA-проекцию с цветом по лучшей кластеризации."""

        full_matrix_df = full_df[feature_space].apply(pd.to_numeric, errors="coerce").dropna()
        if full_matrix_df.empty or len(cluster_labels) != len(full_matrix_df):
            return
        pca = PCA(n_components=2, random_state=self.random_state)
        components = pca.fit_transform(RobustScaler().fit_transform(full_matrix_df.to_numpy(dtype=float)))
        pca_df = pd.DataFrame({"pc1": components[:, 0], "pc2": components[:, 1], "cluster": cluster_labels.astype(int)})
        pca_df = self._sample_for_plot(pca_df)
        if pca_df.empty:
            return

        figure, axis = plt.subplots(figsize=(8, 6))
        scatter = axis.scatter(
            pca_df["pc1"],
            pca_df["pc2"],
            c=pca_df["cluster"],
            cmap="tab10",
            s=18,
            alpha=0.7,
        )
        axis.set_title(title)
        axis.set_xlabel("PC1")
        axis.set_ylabel("PC2")
        axis.grid(alpha=0.22)
        figure.colorbar(scatter, ax=axis, label="cluster")
        figure.tight_layout()
        figure.savefig(output_path, dpi=PLOT_DPI)
        plt.close(figure)

    def _scatter_plot(
        self,
        *,
        plot_df: pd.DataFrame,
        x_column: str,
        y_column: str,
        color_column: str,
        color_map: dict[str, str],
        title: str,
        x_label: str,
        y_label: str,
        output_path: Path,
    ) -> None:
        """Рисует обычный 2D scatter."""

        scatter_df = plot_df[[x_column, y_column, color_column]].copy()
        scatter_df[x_column] = pd.to_numeric(scatter_df[x_column], errors="coerce")
        scatter_df[y_column] = pd.to_numeric(scatter_df[y_column], errors="coerce")
        scatter_df = scatter_df.dropna(subset=[x_column, y_column])
        if scatter_df.empty:
            return

        figure, axis = plt.subplots(figsize=(8, 6))
        for label, label_df in scatter_df.groupby(color_column, sort=False):
            axis.scatter(
                label_df[x_column],
                label_df[y_column],
                s=18,
                alpha=0.65,
                color=color_map.get(str(label).lower(), "#546E7A"),
                label=str(label),
            )
        axis.set_title(title)
        axis.set_xlabel(x_label)
        axis.set_ylabel(y_label)
        axis.grid(alpha=0.22)
        axis.legend(loc="best", fontsize=8)
        figure.tight_layout()
        figure.savefig(output_path, dpi=PLOT_DPI)
        plt.close(figure)

    def _build_capacity_verdict(
        self,
        *,
        supported_labels: list[str],
        aggregate_pairs: pd.DataFrame,
        cluster_summary: dict[str, Any],
    ) -> tuple[str, int, int, int, str, str]:
        """Формулирует итоговую интерпретацию числа состояний."""

        pairwise_label_count = len(supported_labels)
        consensus_k = cluster_summary.get("consensus_supported_k")
        support_strength = str(cluster_summary.get("consensus_support_strength", "weak"))
        bootstrap_ari = float(cluster_summary.get("consensus_mean_bootstrap_ari", np.nan))
        label_alignment_ari = float(cluster_summary.get("label_alignment_ari", np.nan))
        label_alignment_nmi = float(cluster_summary.get("label_alignment_nmi", np.nan))
        alignment_support = (
            (np.isfinite(label_alignment_ari) and label_alignment_ari >= 0.10)
            or (np.isfinite(label_alignment_nmi) and label_alignment_nmi >= 0.10)
        )

        confident_count = 1
        likely_count = 1
        if pairwise_label_count >= 2:
            confident_count = 2
            likely_count = 2
        elif consensus_k is not None and consensus_k >= 2 and alignment_support:
            confident_count = 2
            likely_count = 2
        elif consensus_k is not None and consensus_k >= 2:
            likely_count = 2

        third_mode_supported = (
            consensus_k is not None
            and consensus_k >= 3
            and pairwise_label_count >= 3
            and support_strength in {"strong", "moderate"}
        )
        four_plus_supported = (
            consensus_k is not None
            and consensus_k >= 4
            and pairwise_label_count >= 4
            and support_strength == "strong"
            and np.isfinite(bootstrap_ari)
            and bootstrap_ari >= 0.70
        )

        if four_plus_supported:
            confident_count = int(consensus_k)
            likely_count = int(consensus_k)
            no_evidence_above = int(consensus_k + 1)
            verdict = f"evidence for {consensus_k} states"
            note = "И попарная различимость, и внутренняя кластерная устойчивость поддерживают число состояний выше 3."
            return verdict, confident_count, likely_count, no_evidence_above, support_strength, note

        if third_mode_supported:
            likely_count = 3
            no_evidence_above = 4
            verdict = "likely 3 states"
            note = "Третий режим поддерживается и попарной статистикой по меткам, и внутренней геометрией модели, но слабее первых двух."
            return verdict, confident_count, likely_count, no_evidence_above, support_strength, note

        no_evidence_above = 3 if likely_count >= 2 else 2
        if confident_count >= 2:
            if pairwise_label_count >= 3 or (consensus_k is not None and consensus_k >= 3):
                verdict = "confidently 2 states, limited evidence for 3rd mode"
                note = "Есть намёки на переходный или промежуточный режим, но устойчивости недостаточно для честного подтверждения трёх состояний."
            else:
                verdict = "confidently 2 states"
                note = "Бинарное разделение устойчиво подтверждается, а дополнительные режимы не получают достаточно сильной кластерной и попарной поддержки."
        elif likely_count >= 2:
            verdict = "tentative evidence for 2 states"
            note = "Внутренняя геометрия модели дробится минимум на два режима, но размеченные состояния пока не разделяются достаточно уверенно."
        else:
            verdict = "weak evidence even for 2 states"
            note = "Модель даёт слишком слабое разделение состояний для уверенного вывода даже о двух устойчивых режимах."
        return verdict, confident_count, likely_count, no_evidence_above, support_strength, note

    def _annotate_dynamic_differences(self, regime_results: list[RegimeResult]) -> list[RegimeResult]:
        """Добавляет комментарий, сглаживает ли динамика различимость по сравнению со статикой."""

        indexed: dict[tuple[str, str, str], RegimeResult] = {
            (item.dataset, item.mode, item.version): item
            for item in regime_results
            if item.regime == "static"
        }
        for item in regime_results:
            if item.regime != "dynamic":
                continue
            static_item = indexed.get((item.dataset, item.mode, item.version))
            if static_item is None:
                continue
            if item.likely_state_count < static_item.likely_state_count:
                item.dynamic_vs_static_note = "Динамика сглаживает режимы по сравнению со статикой."
            elif item.likely_state_count > static_item.likely_state_count:
                item.dynamic_vs_static_note = "Динамика добавляет поддерживаемый переходный режим."
            else:
                item.dynamic_vs_static_note = "Динамика не увеличивает число подтверждаемых режимов относительно статики."
        return regime_results

    def _build_payload(
        self,
        *,
        dataset: str,
        mode: str,
        regime_results: list[RegimeResult],
        separation_df: pd.DataFrame,
        cluster_df: pd.DataFrame,
        separation_path: Path,
        cluster_path: Path,
        capacity_path: Path,
        plot_paths: list[str],
    ) -> dict[str, Any]:
        """Собирает JSON-отчёт."""

        versions_payload: dict[str, dict[str, Any]] = {}
        for item in regime_results:
            versions_payload.setdefault(item.version, {})
            versions_payload[item.version][item.regime] = {
                "labeled_state_count": item.labeled_state_count,
                "pairwise_distinguishable_pairs": item.pairwise_distinguishable_pairs,
                "largest_fully_distinguishable_label_set": item.largest_fully_distinguishable_label_set,
                "supported_labels": item.supported_labels,
                "supported_pair_labels": item.supported_pair_labels,
                "primary_measure": item.primary_measure,
                "feature_space": item.feature_space,
                "best_k_kmeans": item.best_k_kmeans,
                "best_k_gmm": item.best_k_gmm,
                "consensus_supported_k": item.consensus_supported_k,
                "consensus_support_strength": item.consensus_support_strength,
                "label_alignment_ari": item.label_alignment_ari,
                "label_alignment_nmi": item.label_alignment_nmi,
                "va_alignment_ari": item.va_alignment_ari,
                "va_alignment_nmi": item.va_alignment_nmi,
                "confident_state_count": item.confident_state_count,
                "likely_state_count": item.likely_state_count,
                "no_evidence_for_above": item.no_evidence_for_above,
                "capacity_verdict": item.capacity_verdict,
                "capacity_note": item.capacity_note,
                "dynamic_vs_static_note": item.dynamic_vs_static_note,
            }

        return {
            "dataset": dataset,
            "mode": mode,
            "versions": versions_payload,
            "generated_files": {
                "state_separation_csv": str(separation_path),
                "state_clusters_csv": str(cluster_path),
                "state_capacity_json": str(capacity_path),
                "plots_dir": str(self.plots_dir),
            },
            "plot_files": sorted(set(plot_paths)),
            "separation_record_count": int(len(separation_df)),
            "cluster_record_count": int(len(cluster_df)),
        }

    def _augment_existing_summary(self, *, dataset: str, mode: str, payload: dict[str, Any]) -> None:
        """Дописывает краткую сводку в уже существующий summary_<dataset>_<mode>.json."""

        summary_path = self.output_dir / f"summary_{dataset}_{mode}.json"
        if not summary_path.exists():
            return
        try:
            summary_payload = json.loads(summary_path.read_text(encoding="utf-8"))
        except Exception:
            return
        summary_payload["state_capacity_outputs"] = payload["generated_files"]
        summary_payload["state_capacity_overview"] = payload["versions"]
        summary_path.write_text(json.dumps(summary_payload, ensure_ascii=False, indent=2), encoding="utf-8")

    def _print_console_report(self, *, dataset: str, mode: str, regime_results: list[RegimeResult]) -> None:
        """Печатает короткий итог по dataset/mode."""

        LOGGER.info("===== State Capacity: %s / %s =====", dataset.upper(), mode)
        if not regime_results:
            LOGGER.info("  Нет валидных режимов для анализа.")
            return
        for item in regime_results:
            LOGGER.info(
                "  %s %s | confident=%s | likely=%s | best_k=(kmeans=%s, gmm=%s, consensus=%s) | verdict=%s",
                item.version,
                item.regime,
                item.confident_state_count,
                item.likely_state_count,
                item.best_k_kmeans,
                item.best_k_gmm,
                item.consensus_supported_k,
                item.capacity_verdict,
            )
            LOGGER.info(
                "  %s %s | labels=%s | distinguishable=%s | note=%s",
                item.version,
                item.regime,
                item.labeled_state_count,
                item.largest_fully_distinguishable_label_set,
                item.capacity_note,
            )
            if item.dynamic_vs_static_note:
                LOGGER.info("  %s %s | dynamic_vs_static=%s", item.version, item.regime, item.dynamic_vs_static_note)

    def _multivariate_pair_distances(self, *, local_df: pd.DataFrame, labels: list[str], feature_space: list[str]) -> dict[str, float]:
        """Считает евклидовы расстояния между центрами классов в пространстве модели."""

        if len(feature_space) < 2:
            return {}
        work_df = local_df[["label", *feature_space]].copy()
        for column in feature_space:
            work_df[column] = pd.to_numeric(work_df[column], errors="coerce")
        work_df = work_df.dropna(subset=feature_space)
        work_df = work_df.loc[work_df["label"].isin(labels)].copy()
        if work_df.empty:
            return {}

        scaler = RobustScaler()
        scaled = scaler.fit_transform(work_df[feature_space].to_numpy(dtype=float))
        scaled_df = pd.DataFrame(scaled, columns=feature_space, index=work_df.index)
        scaled_df["label"] = work_df["label"].to_numpy()
        centroids = scaled_df.groupby("label")[feature_space].mean()

        distances: dict[str, float] = {}
        for label_a, label_b in combinations(labels, 2):
            if label_a not in centroids.index or label_b not in centroids.index:
                continue
            distance = float(np.linalg.norm(centroids.loc[label_b].to_numpy() - centroids.loc[label_a].to_numpy()))
            distances[f"{label_a}__{label_b}"] = distance
        return distances

    def _largest_supported_label_set(self, aggregate_pairs: pd.DataFrame) -> list[str]:
        """Ищет максимальный набор меток, попарно различимых по агрегированной поддержке."""

        if aggregate_pairs.empty:
            return []
        labels = sorted(set(aggregate_pairs["label_a"].astype(str)) | set(aggregate_pairs["label_b"].astype(str)), key=self._label_sort_key)
        supported_edges = {
            tuple(sorted((str(row["label_a"]), str(row["label_b"]))))
            for row in aggregate_pairs.to_dict(orient="records")
            if bool(row.get("pair_supported"))
        }
        for subset_size in range(len(labels), 1, -1):
            for subset in combinations(labels, subset_size):
                if all(tuple(sorted(pair)) in supported_edges for pair in combinations(subset, 2)):
                    return list(subset)
        return []

    def _best_k_for_algorithm(self, cluster_df: pd.DataFrame, algorithm: str) -> int | None:
        """Возвращает лучший k для указанного алгоритма."""

        subset = cluster_df.loc[cluster_df["algorithm"] == algorithm].copy()
        if subset.empty:
            return None
        best = subset.sort_values(
            ["stable_support", "internal_score", "alignment_score", "k"],
            ascending=[False, False, False, True],
        ).iloc[0]
        return int(best["k"])

    def _bootstrap_groups(self, frame: pd.DataFrame) -> np.ndarray:
        """Выбирает идентификатор группировки для бутстрэпа."""

        if "source_record_id" in frame.columns and frame["source_record_id"].fillna("").astype(str).nunique() > 1:
            return frame["source_record_id"].fillna("").astype(str).to_numpy()
        if "subject_id" in frame.columns and frame["subject_id"].fillna("").astype(str).nunique() > 1:
            return frame["subject_id"].fillna("").astype(str).to_numpy()
        return np.asarray([f"row_{index}" for index in range(len(frame))], dtype=object)

    def _make_va_quadrant(self, frame: pd.DataFrame) -> np.ndarray | None:
        """Строит вспомогательные квадранты valence/arousal."""

        if "valence" not in frame.columns or "arousal" not in frame.columns:
            return None
        va_df = frame[["valence", "arousal"]].copy()
        va_df["valence"] = pd.to_numeric(va_df["valence"], errors="coerce")
        va_df["arousal"] = pd.to_numeric(va_df["arousal"], errors="coerce")
        if va_df["valence"].notna().sum() < self.min_class_samples or va_df["arousal"].notna().sum() < self.min_class_samples:
            return None
        valence_median = float(va_df["valence"].median())
        arousal_median = float(va_df["arousal"].median())
        quadrant = np.where(va_df["valence"].fillna(valence_median) >= valence_median, "high_valence", "low_valence")
        quadrant = np.where(
            va_df["arousal"].fillna(arousal_median) >= arousal_median,
            np.char.add(quadrant.astype(str), "_high_arousal"),
            np.char.add(quadrant.astype(str), "_low_arousal"),
        )
        if len(np.unique(quadrant)) < 2:
            return None
        return quadrant.astype(object)

    def _cluster_proportions(self, labels: np.ndarray, k_value: int) -> np.ndarray:
        """Возвращает доли кластеров."""

        counts = np.bincount(labels.astype(int), minlength=k_value).astype(float)
        return counts / max(np.sum(counts), 1.0)

    def _extract_cluster_centers(self, model: Any) -> np.ndarray:
        """Возвращает матрицу центров кластера."""

        if hasattr(model, "cluster_centers_"):
            return np.asarray(model.cluster_centers_, dtype=float)
        if hasattr(model, "means_"):
            return np.asarray(model.means_, dtype=float)
        raise ValueError("У модели нет доступных центров кластера.")

    def _centroid_drift(self, first: np.ndarray, second: np.ndarray) -> float:
        """Оценивает средний дрейф центров после выравнивания перестановки кластеров."""

        if first.shape != second.shape:
            return np.nan
        distance_matrix = cdist(first, second, metric="euclidean")
        row_ids, col_ids = linear_sum_assignment(distance_matrix)
        return float(np.mean(distance_matrix[row_ids, col_ids]))

    def _safe_adjusted_rand(self, truth: np.ndarray | None, predicted: np.ndarray) -> float:
        """Безопасный ARI."""

        if truth is None:
            return np.nan
        truth_array = np.asarray(truth)
        if truth_array.size != predicted.size or np.unique(truth_array).size < 2:
            return np.nan
        try:
            return float(adjusted_rand_score(truth_array, predicted))
        except Exception:
            return np.nan

    def _safe_nmi(self, truth: np.ndarray | None, predicted: np.ndarray) -> float:
        """Безопасный NMI."""

        if truth is None:
            return np.nan
        truth_array = np.asarray(truth)
        if truth_array.size != predicted.size or np.unique(truth_array).size < 2:
            return np.nan
        try:
            return float(normalized_mutual_info_score(truth_array, predicted))
        except Exception:
            return np.nan

    def _holm_correction(self, p_values: list[float]) -> list[float]:
        """Коррекция Холма для набора попарных тестов."""

        if not p_values:
            return []
        clean = [1.0 if value is None or not np.isfinite(value) else float(value) for value in p_values]
        ordered = sorted(enumerate(clean), key=lambda item: item[1])
        corrected = [1.0] * len(clean)
        running_max = 0.0
        total = len(clean)
        for rank, (original_index, value) in enumerate(ordered, start=1):
            adjusted = min(1.0, value * (total - rank + 1))
            running_max = max(running_max, adjusted)
            corrected[original_index] = running_max
        return corrected

    def _grade_pair(self, *, effect_size: float, overlap: float, p_value_holm: float) -> tuple[str, int]:
        """Качественная оценка попарной различимости."""

        if not np.isfinite(effect_size) or not np.isfinite(overlap) or not np.isfinite(p_value_holm):
            return "none", 0
        abs_effect = abs(effect_size)
        if p_value_holm < 0.01 and abs_effect >= 0.75 and overlap <= 0.70:
            return "strong", 3
        if p_value_holm < 0.05 and abs_effect >= 0.35 and overlap <= 0.85:
            return "moderate", 2
        if p_value_holm < 0.10 and (abs_effect >= 0.20 or overlap <= 0.92):
            return "weak", 1
        return "none", 0

    def _ordered_labels(self, labels: list[str]) -> list[str]:
        """Возвращает предсказуемый порядок меток."""

        return sorted(labels, key=self._label_sort_key)

    def _label_sort_key(self, value: str) -> tuple[int, str]:
        """Ключ сортировки для меток."""

        lower = str(value).lower()
        if lower in PREFERRED_LABEL_ORDER:
            return PREFERRED_LABEL_ORDER.index(lower), lower
        return len(PREFERRED_LABEL_ORDER), lower

    def _sample_for_plot(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Ограничивает число точек на графике."""

        if len(frame) <= self.max_scatter_points:
            return frame
        return frame.sample(n=self.max_scatter_points, random_state=self.random_state)

    def _distribution_overlap(self, first: np.ndarray, second: np.ndarray, bins: int = 60) -> float:
        """Доля перекрытия двух одномерных распределений."""

        first = np.asarray(first, dtype=float)
        second = np.asarray(second, dtype=float)
        first = first[np.isfinite(first)]
        second = second[np.isfinite(second)]
        if first.size < 2 or second.size < 2:
            return np.nan
        low = float(min(np.min(first), np.min(second)))
        high = float(max(np.max(first), np.max(second)))
        if not np.isfinite(low) or not np.isfinite(high) or low == high:
            return np.nan
        hist_first, edges = np.histogram(first, bins=bins, range=(low, high), density=True)
        hist_second, _ = np.histogram(second, bins=bins, range=(low, high), density=True)
        widths = np.diff(edges)
        return float(np.sum(np.minimum(hist_first, hist_second) * widths))

    def _cohens_d(self, first: np.ndarray, second: np.ndarray) -> float:
        """Effect size между двумя выборками."""

        first = np.asarray(first, dtype=float)
        second = np.asarray(second, dtype=float)
        first = first[np.isfinite(first)]
        second = second[np.isfinite(second)]
        if first.size < 2 or second.size < 2:
            return np.nan
        variance = ((first.size - 1) * np.var(first, ddof=1) + (second.size - 1) * np.var(second, ddof=1)) / max(
            first.size + second.size - 2,
            1,
        )
        pooled_std = float(np.sqrt(variance))
        if pooled_std == 0.0 or not np.isfinite(pooled_std):
            return np.nan
        return float((np.mean(first) - np.mean(second)) / pooled_std)

    def _series_minmax(self, series: pd.Series) -> pd.Series:
        """Min-max нормировка с защитой от константных рядов."""

        numeric = pd.to_numeric(series, errors="coerce")
        finite = numeric[np.isfinite(numeric)]
        if finite.empty:
            return pd.Series(np.zeros(len(series)), index=series.index, dtype=float)
        minimum = float(np.min(finite))
        maximum = float(np.max(finite))
        if minimum == maximum:
            return pd.Series(np.ones(len(series)), index=series.index, dtype=float)
        return ((numeric - minimum) / (maximum - minimum)).fillna(0.0).clip(0.0, 1.0)
