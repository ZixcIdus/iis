"""Кросс-валидационная калибровка IISVersion7 beta под DREAMER."""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import kruskal

from config.settings import OUTPUT_DIR, VERSION4_CALIBRATION, VERSION6_CALIBRATION, VERSION7_CALIBRATION

EPSILON = 1e-9
DEFAULT_INPUT = OUTPUT_DIR / "features_dreamer.csv"
DEFAULT_JSON = OUTPUT_DIR / "dreamer_v7_calibration.json"
DEFAULT_CSV = OUTPUT_DIR / "dreamer_v7_calibration_candidates.csv"


@dataclass
class FoldResult:
    fold_index: int
    objective: float
    avg_pairwise_d: float
    baseline_stress_d: float
    epsilon_squared: float
    middle_score: float
    std_iis: float
    label_means: dict[str, float]
    gamma_std: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Калибровка IISVersion7 на DREAMER с holdout по субъектам.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="Путь к features_dreamer.csv")
    parser.add_argument("--trials", type=int, default=140, help="Число random-search кандидатов.")
    parser.add_argument("--folds", type=int, default=4, help="Число subject-level CV фолдов.")
    parser.add_argument("--seed", type=int, default=42, help="Seed для воспроизводимости.")
    parser.add_argument("--top-k", type=int, default=12, help="Сколько лучших кандидатов сохранить в CSV.")
    parser.add_argument(
        "--candidate-mode",
        choices=("global", "domain"),
        default="global",
        help="Режим поиска кандидатов: глобовая база или fold-specific DREAMER-recalibration.",
    )
    parser.add_argument("--json-out", type=Path, default=DEFAULT_JSON, help="JSON-отчёт.")
    parser.add_argument("--csv-out", type=Path, default=DEFAULT_CSV, help="CSV лучших кандидатов.")
    return parser.parse_args()


def to_numeric_array(values: Any) -> np.ndarray:
    if isinstance(values, (pd.Series, pd.Index)):
        return pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
    return np.asarray(values, dtype=float)


def robust_center_width(values: Any) -> tuple[float, float]:
    arr = to_numeric_array(values)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan"), float("nan")
    q05, q50, q95 = np.quantile(arr, [0.05, 0.50, 0.95])
    width = max((q95 - q05) / 2.0, 1e-6)
    return float(q50), float(width)


def derive_base_calibration(train_df: pd.DataFrame) -> dict[str, float]:
    total_asymmetry = (
        (to_numeric_array(train_df["eeg_left_power"]) - to_numeric_array(train_df["eeg_right_power"]))
        / (to_numeric_array(train_df["eeg_left_power"]) + to_numeric_array(train_df["eeg_right_power"]) + EPSILON)
    )
    gamma_log = np.log1p(np.maximum(to_numeric_array(train_df["eeg_gamma_power"]), 0.0) * 1e12)
    gamma_alpha_log = np.log1p(np.maximum(to_numeric_array(train_df["gamma_alpha_ratio"]), 0.0))
    hf_lf_log = np.log1p(np.maximum(to_numeric_array(train_df["hrv_hf_lf"]), 0.0))

    alpha_center, alpha_width = robust_center_width(train_df["alpha_asymmetry"])
    total_center, total_width = robust_center_width(total_asymmetry)
    gamma_center, gamma_width = robust_center_width(gamma_log)
    gamma_alpha_center, gamma_alpha_width = robust_center_width(gamma_alpha_log)
    hr_center, hr_width = robust_center_width(train_df["heart_rate"])
    hf_lf_center, hf_lf_width = robust_center_width(hf_lf_log)

    return {
        "alpha_asymmetry_center": alpha_center,
        "alpha_asymmetry_width": alpha_width,
        "total_asymmetry_center": total_center,
        "total_asymmetry_width": total_width,
        "gamma_log_center": gamma_center,
        "gamma_log_width": gamma_width,
        "gamma_alpha_log_center": gamma_alpha_center,
        "gamma_alpha_log_width": gamma_alpha_width,
        "heart_rate_center": hr_center,
        "heart_rate_width": hr_width,
        "hf_lf_log_center": hf_lf_center,
        "hf_lf_log_width": hf_lf_width,
    }


def robust_tanh(values: Any, center: float, width: float) -> np.ndarray:
    arr = to_numeric_array(values)
    result = np.full(arr.shape, np.nan, dtype=float)
    valid_mask = np.isfinite(arr) & np.isfinite(center) & np.isfinite(width) & (width > 0)
    if not np.any(valid_mask):
        return result
    result[valid_mask] = np.tanh((arr[valid_mask] - center) / width)
    return result


def signed_power(values: np.ndarray, power: float) -> np.ndarray:
    result = np.full(values.shape, np.nan, dtype=float)
    valid_mask = np.isfinite(values)
    if not np.any(valid_mask):
        return result
    result[valid_mask] = np.sign(values[valid_mask]) * (np.abs(values[valid_mask]) ** power)
    return result


def bounded_positive(values: np.ndarray) -> np.ndarray:
    positive = np.maximum(values, 0.0)
    return positive / (1.0 + positive)


def rowwise_softmax(logits: np.ndarray, temperature: float) -> np.ndarray:
    scaled = np.asarray(logits, dtype=float) * float(max(temperature, 1e-6))
    finite_mask = np.isfinite(scaled)
    safe_logits = np.where(finite_mask, scaled, -np.inf)
    has_finite = np.any(finite_mask, axis=1, keepdims=True)
    row_max = np.max(safe_logits, axis=1, keepdims=True)
    row_max = np.where(has_finite, row_max, 0.0)
    shifted = np.where(finite_mask, safe_logits - row_max, -np.inf)
    exp_values = np.where(finite_mask, np.exp(np.clip(shifted, -50.0, 50.0)), 0.0)
    totals = exp_values.sum(axis=1, keepdims=True)
    totals[totals <= 0.0] = 1.0
    return exp_values / totals


def sumigron_reduce(columns: list[np.ndarray], params: dict[str, float]) -> np.ndarray:
    matrix = np.column_stack([np.asarray(column, dtype=float) for column in columns])
    finite_mask = np.isfinite(matrix)
    safe_matrix = np.where(finite_mask, matrix, np.nan)
    local_mean = np.nanmean(safe_matrix, axis=1, keepdims=True)
    local_mean = np.where(np.isfinite(local_mean), local_mean, 0.0)
    local_std = np.nanstd(safe_matrix, axis=1, keepdims=True)
    local_std = np.where(np.isfinite(local_std), local_std, 0.0)
    local_z = np.where(finite_mask, (matrix - local_mean) / (local_std + 1e-6), np.nan)
    drive = 0.60 * local_z + 0.40 * np.abs(local_z)
    weights = rowwise_softmax(drive, params["sumigron_temperature"])
    safe_values = np.where(finite_mask, matrix, 0.0)
    attentive_mean = np.sum(weights * safe_values, axis=1)
    attentive_std = np.sqrt(np.sum(weights * np.where(finite_mask, (matrix - attentive_mean[:, None]) ** 2, 0.0), axis=1))
    attentive_energy = np.log1p(np.sum(weights * np.where(finite_mask, matrix**2, 0.0), axis=1))
    sign = np.sign(attentive_mean)
    fallback_sign = np.sign(np.sum(matrix, axis=1))
    sign = np.where(sign == 0.0, fallback_sign, sign)
    sign = np.where(sign == 0.0, 1.0, sign)
    structured = sign * (
        params["sumigron_structure_weight"] * bounded_positive(attentive_std)
        + params["sumigron_energy_weight"] * bounded_positive(attentive_energy)
    )
    return params["sumigron_level_weight"] * attentive_mean + structured


def build_v7_params(candidate: dict[str, float] | None = None) -> dict[str, float]:
    params = dict(VERSION7_CALIBRATION)
    params["weights"] = {
        "A": 0.15,
        "Gamma": 0.08,
        "V": 0.24,
        "Q": 0.53,
    }
    if candidate:
        weight_keys = ("A", "Gamma", "V", "Q")
        if any(key.startswith("weight_") for key in candidate):
            weights = {name: max(candidate.get(f"weight_{name}", params["weights"][name]), 1e-6) for name in weight_keys}
            total = sum(weights.values())
            params["weights"] = {name: value / total for name, value in weights.items()}
        for key, value in candidate.items():
            if key.startswith("weight_"):
                continue
            params[key] = float(value)
    return params


def score_v7(df: pd.DataFrame, base_calibration: dict[str, float], v7_params: dict[str, float]) -> pd.DataFrame:
    left_power = to_numeric_array(df["eeg_left_power"])
    right_power = to_numeric_array(df["eeg_right_power"])
    total_asymmetry = (left_power - right_power) / (left_power + right_power + EPSILON)

    alpha_term = robust_tanh(
        df["alpha_asymmetry"],
        base_calibration["alpha_asymmetry_center"],
        base_calibration["alpha_asymmetry_width"],
    )
    total_term = robust_tanh(
        total_asymmetry,
        base_calibration["total_asymmetry_center"],
        base_calibration["total_asymmetry_width"],
    )
    a_synergy = np.sign(alpha_term + total_term) * np.sqrt(np.abs(alpha_term * total_term) + 1e-9)
    a_raw = sumigron_reduce([1.00 * alpha_term, 0.55 * total_term, 0.35 * a_synergy], v7_params)
    a_value = np.tanh(v7_params["component_gain"] * a_raw)

    gamma_log = np.log1p(np.maximum(to_numeric_array(df["eeg_gamma_power"]), 0.0) * 1e12)
    gamma_term = robust_tanh(
        gamma_log,
        base_calibration["gamma_log_center"],
        base_calibration["gamma_log_width"],
    )
    gamma_alpha_log = np.log1p(np.maximum(to_numeric_array(df["gamma_alpha_ratio"]), 0.0))
    burden_term = robust_tanh(
        gamma_alpha_log,
        base_calibration["gamma_alpha_log_center"],
        base_calibration["gamma_alpha_log_width"],
    )
    gamma_synergy = np.sign(gamma_term + burden_term) * np.sqrt(np.abs(gamma_term * burden_term) + 1e-9)
    gamma_raw = sumigron_reduce([0.35 * gamma_term, 0.75 * burden_term, 0.20 * gamma_synergy], v7_params)
    gamma_value = np.tanh(v7_params["component_gain"] * gamma_raw)

    heart_rate_term = robust_tanh(
        base_calibration["heart_rate_center"] - to_numeric_array(df["heart_rate"]),
        0.0,
        base_calibration["heart_rate_width"],
    )
    hrv_ratio_log = np.log1p(np.maximum(to_numeric_array(df["hrv_hf_lf"]), 0.0))
    ratio_term = robust_tanh(
        hrv_ratio_log,
        base_calibration["hf_lf_log_center"],
        base_calibration["hf_lf_log_width"],
    )
    v_synergy = np.sign(heart_rate_term + ratio_term) * np.sqrt(np.abs(heart_rate_term * ratio_term) + 1e-9)
    v_raw = sumigron_reduce([0.85 * heart_rate_term, 0.30 * ratio_term, 0.20 * v_synergy], v7_params)
    v_value = np.tanh(v7_params["component_gain"] * v_raw)

    q_linear = 0.52 * a_value + 0.44 * v_value - 0.32 * gamma_value
    q_coherence = (
        a_value * v_value
        - v7_params["q_coherence_penalty"] * np.abs(a_value - v_value)
        - v7_params["q_gamma_penalty"] * np.maximum(gamma_value, 0.0)
    )
    q_synergy = np.sign(q_linear + q_coherence) * np.sqrt(np.abs(q_linear * q_coherence) + 1e-9)
    q_raw = sumigron_reduce([q_linear, q_coherence, 0.55 * q_synergy], v7_params)
    q_value = np.clip(0.5 + 0.5 * np.tanh(v7_params["q_gain"] * q_raw), 0.0, 1.0)

    shaped_a = signed_power(a_value, VERSION6_CALIBRATION["shape_power"])
    shaped_gamma = signed_power(gamma_value, VERSION6_CALIBRATION["gamma_shape_power"])
    shaped_v = signed_power(v_value, VERSION6_CALIBRATION["shape_power"])
    shaped_q = signed_power(q_value, VERSION6_CALIBRATION["shape_power"])
    gamma_pos = np.maximum(shaped_gamma, 0.0)

    reg_signature = sumigron_reduce([0.72 * shaped_q, 0.52 * shaped_v, 0.18 * shaped_a, -0.24 * gamma_pos], v7_params)
    mob_signature = sumigron_reduce([-0.06 * shaped_q, 0.28 * shaped_v, -0.48 * gamma_pos, 0.10 * shaped_a], v7_params)
    dep_signature = sumigron_reduce([-0.60 * shaped_q, -0.50 * shaped_v, 0.36 * gamma_pos, -0.20 * shaped_a], v7_params)

    logits = np.column_stack([reg_signature, mob_signature, dep_signature])
    gates = rowwise_softmax(logits, v7_params["gate_temperature"])
    synergy_reg = np.sign(shaped_q + shaped_v) * np.sqrt(np.abs(shaped_q * shaped_v) + 1e-9)
    conflict = np.abs(shaped_a - shaped_v) + 0.65 * np.abs(shaped_q - shaped_v)

    reg_level = v7_params["reg_iis_base"] + v7_params["reg_iis_amp"] * np.tanh(reg_signature + 0.20 * synergy_reg)
    mob_level = v7_params["mob_iis_base"] + v7_params["mob_iis_amp"] * np.tanh(mob_signature)
    dep_level = v7_params["dep_iis_base"] + v7_params["dep_iis_amp"] * np.tanh(dep_signature)

    regime_score = gates[:, 0] * reg_level + gates[:, 1] * mob_level + gates[:, 2] * dep_level
    entropy = -(gates * np.log(gates + 1e-9)).sum(axis=1) / np.log(3.0)
    regime_balance = gates[:, 0] - gates[:, 2]
    iis = regime_score - v7_params["transition_entropy_weight"] * entropy + v7_params["regime_balance_weight"] * regime_balance
    iis -= v7_params["conflict_penalty"] * conflict
    iis = np.clip(iis, 0.0, 1.0)

    scored = df.loc[:, ["subject_id", "segment_id", "label", "valence", "arousal"]].copy()
    scored["A"] = a_value
    scored["Gamma"] = gamma_value
    scored["V"] = v_value
    scored["Q"] = q_value
    scored["IIS"] = iis
    return scored


def cohen_d(left: np.ndarray, right: np.ndarray) -> float:
    if left.size < 2 or right.size < 2:
        return float("nan")
    left_var = float(np.var(left, ddof=1))
    right_var = float(np.var(right, ddof=1))
    pooled = ((left.size - 1) * left_var + (right.size - 1) * right_var) / max(left.size + right.size - 2, 1)
    if pooled <= 0 or not np.isfinite(pooled):
        return float("nan")
    return float((float(np.mean(left)) - float(np.mean(right))) / math.sqrt(pooled))


def middle_state_score(label_means: dict[str, float]) -> float:
    baseline_mean = label_means.get("baseline")
    disbalance_mean = label_means.get("disbalance")
    stress_mean = label_means.get("stress")
    if not all(np.isfinite(value) for value in (baseline_mean, disbalance_mean, stress_mean)):
        return 0.0
    lower = min(baseline_mean, stress_mean)
    upper = max(baseline_mean, stress_mean)
    if upper - lower <= 1e-9:
        return 0.0
    midpoint = 0.5 * (baseline_mean + stress_mean)
    distance = abs(disbalance_mean - midpoint) / (upper - lower)
    inside_bonus = 1.0 if lower <= disbalance_mean <= upper else 0.0
    return float(max(0.0, inside_bonus * (1.0 - distance)))


def compute_metrics(scored_df: pd.DataFrame) -> dict[str, Any]:
    labels = ["baseline", "disbalance", "stress"]
    label_means = {
        label: float(pd.to_numeric(scored_df.loc[scored_df["label"] == label, "IIS"], errors="coerce").mean())
        for label in labels
    }

    pairwise_ds: list[float] = []
    for left_label, right_label in combinations(labels, 2):
        left_values = pd.to_numeric(scored_df.loc[scored_df["label"] == left_label, "IIS"], errors="coerce").dropna().to_numpy(dtype=float)
        right_values = pd.to_numeric(scored_df.loc[scored_df["label"] == right_label, "IIS"], errors="coerce").dropna().to_numpy(dtype=float)
        d_value = cohen_d(left_values, right_values)
        if np.isfinite(d_value):
            pairwise_ds.append(abs(d_value))

    baseline_values = pd.to_numeric(scored_df.loc[scored_df["label"] == "baseline", "IIS"], errors="coerce").dropna().to_numpy(dtype=float)
    stress_values = pd.to_numeric(scored_df.loc[scored_df["label"] == "stress", "IIS"], errors="coerce").dropna().to_numpy(dtype=float)
    baseline_stress_d = abs(cohen_d(baseline_values, stress_values))

    kruskal_groups = [
        pd.to_numeric(scored_df.loc[scored_df["label"] == label, "IIS"], errors="coerce").dropna().to_numpy(dtype=float)
        for label in labels
    ]
    epsilon_squared = float("nan")
    try:
        h_statistic, _ = kruskal(*kruskal_groups)
        n_total = int(sum(group.size for group in kruskal_groups))
        if n_total > len(labels):
            epsilon_squared = float(max((h_statistic - len(labels) + 1) / (n_total - len(labels)), 0.0))
    except Exception:
        epsilon_squared = float("nan")

    avg_pairwise_d = float(np.mean(pairwise_ds)) if pairwise_ds else float("nan")
    middle_score = middle_state_score(label_means)
    std_iis = float(pd.to_numeric(scored_df["IIS"], errors="coerce").std())
    gamma_std = float(pd.to_numeric(scored_df["Gamma"], errors="coerce").std())

    objective = (
        1.00 * (avg_pairwise_d if np.isfinite(avg_pairwise_d) else 0.0)
        + 0.80 * (baseline_stress_d if np.isfinite(baseline_stress_d) else 0.0)
        + 1.50 * (epsilon_squared if np.isfinite(epsilon_squared) else 0.0)
        + 0.30 * middle_score
        + 0.10 * (std_iis if np.isfinite(std_iis) else 0.0)
    )
    return {
        "objective": float(objective),
        "avg_pairwise_d": avg_pairwise_d,
        "baseline_stress_d": float(baseline_stress_d) if np.isfinite(baseline_stress_d) else float("nan"),
        "epsilon_squared": epsilon_squared,
        "middle_score": float(middle_score),
        "std_iis": std_iis,
        "label_means": label_means,
        "gamma_std": gamma_std,
    }


def split_subjects(subjects: list[str], n_folds: int) -> list[list[str]]:
    folds = [[] for _ in range(max(n_folds, 2))]
    for index, subject_id in enumerate(subjects):
        folds[index % len(folds)].append(subject_id)
    return folds


def random_candidate(rng: np.random.Generator) -> dict[str, float]:
    return {
        "weight_A": float(rng.uniform(0.10, 0.22)),
        "weight_Gamma": float(rng.uniform(0.04, 0.12)),
        "weight_V": float(rng.uniform(0.14, 0.32)),
        "weight_Q": float(rng.uniform(0.38, 0.72)),
        "sumigron_temperature": float(rng.uniform(0.90, 3.20)),
        "sumigron_structure_weight": float(rng.uniform(0.10, 0.90)),
        "sumigron_energy_weight": float(rng.uniform(0.05, 0.60)),
        "component_gain": float(rng.uniform(0.85, 1.80)),
        "q_gain": float(rng.uniform(0.90, 3.00)),
        "q_coherence_penalty": float(rng.uniform(0.20, 0.90)),
        "q_gamma_penalty": float(rng.uniform(0.05, 0.35)),
        "gate_temperature": float(rng.uniform(0.90, 2.60)),
        "reg_iis_base": float(rng.uniform(0.72, 0.90)),
        "reg_iis_amp": float(rng.uniform(0.08, 0.22)),
        "mob_iis_base": float(rng.uniform(0.42, 0.62)),
        "mob_iis_amp": float(rng.uniform(0.06, 0.18)),
        "dep_iis_base": float(rng.uniform(0.10, 0.30)),
        "dep_iis_amp": float(rng.uniform(0.08, 0.22)),
        "transition_entropy_weight": float(rng.uniform(0.02, 0.16)),
        "regime_balance_weight": float(rng.uniform(0.00, 0.12)),
        "conflict_penalty": float(rng.uniform(0.04, 0.18)),
    }


def evaluate_across_folds(
    features_df: pd.DataFrame,
    fold_subjects: list[list[str]],
    global_base_calibration: dict[str, float],
    v7_params: dict[str, float],
    mode: str,
) -> dict[str, Any]:
    fold_results: list[FoldResult] = []
    for fold_index, fold_subject_ids in enumerate(fold_subjects):
        validation_df = features_df[features_df["subject_id"].isin(fold_subject_ids)].copy()
        train_df = features_df[~features_df["subject_id"].isin(fold_subject_ids)].copy()
        if mode == "global":
            base_calibration = global_base_calibration
        else:
            base_calibration = derive_base_calibration(train_df)
        scored = score_v7(validation_df, base_calibration=base_calibration, v7_params=v7_params)
        metrics = compute_metrics(scored)
        fold_results.append(
            FoldResult(
                fold_index=fold_index,
                objective=metrics["objective"],
                avg_pairwise_d=metrics["avg_pairwise_d"],
                baseline_stress_d=metrics["baseline_stress_d"],
                epsilon_squared=metrics["epsilon_squared"],
                middle_score=metrics["middle_score"],
                std_iis=metrics["std_iis"],
                label_means=metrics["label_means"],
                gamma_std=metrics["gamma_std"],
            )
        )

    return {
        "objective_mean": float(np.mean([fold.objective for fold in fold_results])),
        "objective_std": float(np.std([fold.objective for fold in fold_results])),
        "avg_pairwise_d_mean": float(np.nanmean([fold.avg_pairwise_d for fold in fold_results])),
        "baseline_stress_d_mean": float(np.nanmean([fold.baseline_stress_d for fold in fold_results])),
        "epsilon_squared_mean": float(np.nanmean([fold.epsilon_squared for fold in fold_results])),
        "middle_score_mean": float(np.nanmean([fold.middle_score for fold in fold_results])),
        "std_iis_mean": float(np.nanmean([fold.std_iis for fold in fold_results])),
        "gamma_std_mean": float(np.nanmean([fold.gamma_std for fold in fold_results])),
        "folds": [
            {
                "fold_index": fold.fold_index,
                "objective": fold.objective,
                "avg_pairwise_d": fold.avg_pairwise_d,
                "baseline_stress_d": fold.baseline_stress_d,
                "epsilon_squared": fold.epsilon_squared,
                "middle_score": fold.middle_score,
                "std_iis": fold.std_iis,
                "gamma_std": fold.gamma_std,
                "label_means": fold.label_means,
            }
            for fold in fold_results
        ],
    }


def main() -> int:
    args = parse_args()
    features_df = pd.read_csv(args.input, low_memory=False)
    subjects = sorted(features_df["subject_id"].dropna().unique().tolist())
    fold_subjects = split_subjects(subjects, n_folds=args.folds)

    global_v7_params = build_v7_params()
    global_base = dict(VERSION4_CALIBRATION)
    dreamer_full_base = derive_base_calibration(features_df)

    current_summary = evaluate_across_folds(
        features_df=features_df,
        fold_subjects=fold_subjects,
        global_base_calibration=global_base,
        v7_params=global_v7_params,
        mode="global",
    )
    recalibrated_summary = evaluate_across_folds(
        features_df=features_df,
        fold_subjects=fold_subjects,
        global_base_calibration=global_base,
        v7_params=global_v7_params,
        mode="domain",
    )

    rng = np.random.default_rng(args.seed)
    candidates: list[dict[str, Any]] = []
    for trial_index in range(args.trials):
        candidate = random_candidate(rng)
        params = build_v7_params(candidate)
        summary = evaluate_across_folds(
            features_df=features_df,
            fold_subjects=fold_subjects,
            global_base_calibration=global_base,
            v7_params=params,
            mode=args.candidate_mode,
        )
        candidates.append(
            {
                "trial_index": trial_index,
                "objective_mean": summary["objective_mean"],
                "objective_std": summary["objective_std"],
                "avg_pairwise_d_mean": summary["avg_pairwise_d_mean"],
                "baseline_stress_d_mean": summary["baseline_stress_d_mean"],
                "epsilon_squared_mean": summary["epsilon_squared_mean"],
                "middle_score_mean": summary["middle_score_mean"],
                "std_iis_mean": summary["std_iis_mean"],
                "gamma_std_mean": summary["gamma_std_mean"],
                **candidate,
            }
        )

    candidates_df = pd.DataFrame.from_records(candidates).sort_values(
        ["objective_mean", "epsilon_squared_mean", "avg_pairwise_d_mean"],
        ascending=[False, False, False],
    )
    top_candidates_df = candidates_df.head(args.top_k).copy()
    best_candidate = top_candidates_df.iloc[0].to_dict()
    best_params = build_v7_params(best_candidate)
    best_summary = evaluate_across_folds(
        features_df=features_df,
        fold_subjects=fold_subjects,
        global_base_calibration=global_base,
        v7_params=best_params,
        mode=args.candidate_mode,
    )

    args.csv_out.parent.mkdir(parents=True, exist_ok=True)
    top_candidates_df.to_csv(args.csv_out, index=False)

    report = {
        "dataset": "dreamer",
        "version": "IISVersion7",
        "input_csv": str(args.input),
        "trials": args.trials,
        "folds": args.folds,
        "seed": args.seed,
        "candidate_mode": args.candidate_mode,
        "baseline_current_global": current_summary,
        "baseline_domain_recalibrated": recalibrated_summary,
        "best_candidate_cv": best_summary,
        "recommended_base_calibration_all_data": dreamer_full_base,
        "recommended_v7_params": best_params,
        "best_candidate_flat": best_candidate,
        "top_candidates_csv": str(args.csv_out),
    }
    args.json_out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print("current_global:", json.dumps(current_summary, ensure_ascii=False))
    print("domain_recalibrated:", json.dumps(recalibrated_summary, ensure_ascii=False))
    print("best_candidate_cv:", json.dumps(best_summary, ensure_ascii=False))
    print("json_out:", args.json_out)
    print("csv_out:", args.csv_out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
