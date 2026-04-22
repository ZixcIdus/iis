"""Сравнение сумигрона с базовыми нелинейностями на синтетических окнах."""

from __future__ import annotations

import json
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import kruskal, spearmanr

from config.settings import OUTPUT_DIR

EPSILON = 1e-9
PLOTS_DIR = OUTPUT_DIR / "plots"
DETAIL_OUT = OUTPUT_DIR / "sumigron_synthetic_metrics.csv"
SUMMARY_OUT = OUTPUT_DIR / "sumigron_synthetic_summary.json"
AMPLITUDE_PLOT_OUT = PLOTS_DIR / "sumigron_synthetic_amplitude.png"
STATE_PLOT_OUT = PLOTS_DIR / "sumigron_synthetic_states.png"
SOFTMAX_TEMPERATURE = 1.0
ATTENTION_LINEAR_GAIN = 0.60
ATTENTION_ABS_GAIN = 0.40
SUMIGRON_WEIGHTS = (0.35, 0.30, 0.35)
SUMIGRON_ATTENTIVE_WEIGHTS = (0.32, 0.28, 0.40)


@dataclass(frozen=True)
class StateConfig:
    name: str
    mean_level: float
    oscillation_amp: float
    noise_scale: float
    burst_scale: float = 0.0


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def softsign(x: np.ndarray) -> np.ndarray:
    return x / (1.0 + np.abs(x))


def bounded_arctan(x: np.ndarray) -> np.ndarray:
    return (2.0 / np.pi) * np.arctan(x)


def softmax(values: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    scaled = values / max(temperature, 1e-6)
    shifted = scaled - np.max(scaled)
    exp_values = np.exp(shifted)
    return exp_values / np.sum(exp_values)


def zscore(values: np.ndarray, center: float, scale: float) -> np.ndarray:
    return (values - center) / max(scale, EPSILON)


def rowwise_softmax(values: np.ndarray, temperature: float = SOFTMAX_TEMPERATURE) -> np.ndarray:
    scaled = values / max(temperature, 1e-6)
    shifted = scaled - np.max(scaled, axis=1, keepdims=True)
    exp_values = np.exp(shifted)
    return exp_values / np.sum(exp_values, axis=1, keepdims=True)


def window_stats(windows: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    means = np.mean(windows, axis=1)
    stds = np.std(windows, axis=1)
    energies = np.log1p(np.mean(windows**2, axis=1))
    return means, stds, energies


def cohen_d(left: np.ndarray, right: np.ndarray) -> float:
    if left.size < 2 or right.size < 2:
        return float("nan")
    left_var = float(np.var(left, ddof=1))
    right_var = float(np.var(right, ddof=1))
    pooled = ((left.size - 1) * left_var + (right.size - 1) * right_var) / max(left.size + right.size - 2, 1)
    if pooled <= 0 or not np.isfinite(pooled):
        return float("nan")
    return float((float(np.mean(left)) - float(np.mean(right))) / np.sqrt(pooled))


def normalized_eps2(values: np.ndarray, labels: np.ndarray) -> float:
    groups = [values[labels == label] for label in sorted(np.unique(labels))]
    try:
        h_stat, _ = kruskal(*groups)
    except ValueError:
        return float("nan")
    n_total = int(sum(group.size for group in groups))
    k = len(groups)
    if n_total <= k:
        return float("nan")
    return float(max((h_stat - k + 1) / (n_total - k), 0.0))


def make_window(config: StateConfig, rng: np.random.Generator, length: int) -> np.ndarray:
    time = np.linspace(0.0, 1.0, length, endpoint=False)
    frequency = rng.uniform(1.0, 4.0)
    phase = rng.uniform(0.0, 2.0 * np.pi)
    envelope_phase = rng.uniform(0.0, 2.0 * np.pi)
    envelope = 0.55 + 0.45 * np.sin(2.0 * np.pi * time * rng.uniform(0.4, 1.2) + envelope_phase) ** 2
    oscillation = config.oscillation_amp * envelope * np.sin(2.0 * np.pi * frequency * time + phase)
    noise = rng.normal(loc=0.0, scale=config.noise_scale, size=length)
    bursts = np.zeros(length, dtype=float)
    if config.burst_scale > 0:
        burst_count = int(rng.integers(1, 4))
        burst_positions = rng.integers(0, length, size=burst_count)
        bursts[burst_positions] = rng.normal(loc=config.burst_scale, scale=config.burst_scale * 0.25, size=burst_count)
    return config.mean_level + oscillation + noise + bursts


def generate_scenario(
    scenario_name: str,
    state_configs: list[StateConfig],
    samples_per_state: int,
    length: int,
    seed: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    records: list[dict[str, object]] = []
    for state_index, config in enumerate(state_configs):
        for sample_index in range(samples_per_state):
            window = make_window(config=config, rng=rng, length=length)
            records.append(
                {
                    "scenario": scenario_name,
                    "state": config.name,
                    "state_order": state_index,
                    "sample_index": sample_index,
                    "window": window,
                    "target_mean": float(np.mean(window)),
                    "target_std": float(np.std(window)),
                    "target_rms": float(np.sqrt(np.mean(window**2))),
                }
            )
    return pd.DataFrame.from_records(records)


def mean_driven_states() -> list[StateConfig]:
    return [
        StateConfig("mean_low", mean_level=-0.35, oscillation_amp=0.25, noise_scale=0.12),
        StateConfig("mean_midlow", mean_level=0.00, oscillation_amp=0.28, noise_scale=0.12),
        StateConfig("mean_midhigh", mean_level=0.35, oscillation_amp=0.25, noise_scale=0.12),
        StateConfig("mean_high", mean_level=0.70, oscillation_amp=0.22, noise_scale=0.12),
    ]


def structure_driven_states() -> list[StateConfig]:
    return [
        StateConfig("stable", mean_level=0.12, oscillation_amp=0.18, noise_scale=0.08),
        StateConfig("rhythmic", mean_level=0.12, oscillation_amp=0.55, noise_scale=0.10),
        StateConfig("volatile", mean_level=0.12, oscillation_amp=0.30, noise_scale=0.34),
        StateConfig("bursty", mean_level=0.12, oscillation_amp=0.38, noise_scale=0.28, burst_scale=1.25),
    ]


def compose_sumigron(means: np.ndarray, stds: np.ndarray, energies: np.ndarray, weights: tuple[float, float, float]) -> np.ndarray:
    mu_z = zscore(means, center=float(np.mean(means)), scale=float(np.std(means)))
    sigma_z = zscore(stds, center=float(np.mean(stds)), scale=float(np.std(stds)))
    energy_z = zscore(energies, center=float(np.mean(energies)), scale=float(np.std(energies)))
    return weights[0] * mu_z + weights[1] * sigma_z + weights[2] * energy_z


def softmax_weighted_sum_values(normalized_windows: np.ndarray) -> np.ndarray:
    weights = rowwise_softmax(normalized_windows, temperature=SOFTMAX_TEMPERATURE)
    return np.sum(weights * normalized_windows, axis=1)


def attentive_sumigron_values(windows: np.ndarray, means: np.ndarray, stds: np.ndarray) -> np.ndarray:
    local_means = means[:, None]
    local_stds = np.maximum(stds[:, None], EPSILON)
    local_z = (windows - local_means) / local_stds
    peak_drive = ATTENTION_LINEAR_GAIN * local_z + ATTENTION_ABS_GAIN * np.abs(local_z)
    attentive_weights = rowwise_softmax(peak_drive, temperature=SOFTMAX_TEMPERATURE)

    attentive_mean = np.sum(attentive_weights * windows, axis=1)
    attentive_std = np.sqrt(np.sum(attentive_weights * (windows - attentive_mean[:, None]) ** 2, axis=1))
    attentive_energy = np.log1p(np.sum(attentive_weights * (windows**2), axis=1))
    return compose_sumigron(
        attentive_mean,
        attentive_std,
        attentive_energy,
        weights=SUMIGRON_ATTENTIVE_WEIGHTS,
    )


def build_methods(windows: np.ndarray) -> dict[str, np.ndarray]:
    global_mean = float(np.mean(windows))
    global_std = float(np.std(windows))
    normalized = zscore(windows, center=global_mean, scale=global_std)

    means, stds, energies = window_stats(windows)

    methods = {
        "linear_mean": np.mean(normalized, axis=1),
        "sigmoid_mean": np.mean(sigmoid(normalized), axis=1),
        "tanh_mean": np.mean(np.tanh(normalized), axis=1),
        "softsign_mean": np.mean(softsign(normalized), axis=1),
        "arctan_mean": np.mean(bounded_arctan(normalized), axis=1),
        "softmax_weighted_sum": softmax_weighted_sum_values(normalized),
        "sumigron": compose_sumigron(means, stds, energies, weights=SUMIGRON_WEIGHTS),
        "sumigron_attentive": attentive_sumigron_values(windows, means, stds),
    }
    return methods


def method_metrics(values: np.ndarray, amplitude_target: np.ndarray, state_labels: np.ndarray) -> dict[str, object]:
    pearson_corr = float(np.corrcoef(values, amplitude_target)[0, 1])
    spearman_corr = float(spearmanr(values, amplitude_target).correlation)
    dynamic_range = float(np.quantile(values, 0.95) - np.quantile(values, 0.05))
    epsilon_squared = normalized_eps2(values, labels=state_labels)

    pairwise_ds: list[float] = []
    state_means: dict[str, float] = {}
    for state_name in sorted(np.unique(state_labels)):
        state_means[state_name] = float(np.mean(values[state_labels == state_name]))
    for left_state, right_state in combinations(sorted(np.unique(state_labels)), 2):
        left_values = values[state_labels == left_state]
        right_values = values[state_labels == right_state]
        d_value = cohen_d(left_values, right_values)
        if np.isfinite(d_value):
            pairwise_ds.append(abs(d_value))

    return {
        "pearson_target_corr": pearson_corr,
        "spearman_target_corr": spearman_corr,
        "dynamic_range_p95_p05": dynamic_range,
        "epsilon_squared": epsilon_squared,
        "avg_pairwise_d": float(np.mean(pairwise_ds)) if pairwise_ds else float("nan"),
        "state_means": state_means,
    }


def normalize_for_plot(values: np.ndarray) -> np.ndarray:
    lower = float(np.quantile(values, 0.05))
    upper = float(np.quantile(values, 0.95))
    if upper - lower <= EPSILON:
        return np.full_like(values, 0.5, dtype=float)
    scaled = (values - lower) / (upper - lower)
    return np.clip(scaled, 0.0, 1.0)


def save_plots(plot_df: pd.DataFrame) -> None:
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    amplitude_plot_df = plot_df.copy()
    amplitude_plot_df["target_bin"] = pd.qcut(amplitude_plot_df["target_rms"], q=12, duplicates="drop")
    amplitude_line_df = (
        amplitude_plot_df.groupby(["scenario", "method", "target_bin"], observed=True)
        .agg(target_rms=("target_rms", "mean"), output=("plot_value", "mean"))
        .reset_index()
    )

    fig, axes = plt.subplots(1, 2, figsize=(15, 6), constrained_layout=True)
    for axis, scenario_name in zip(axes, ["mean_driven", "structure_driven"], strict=True):
        scenario_df = amplitude_line_df[amplitude_line_df["scenario"] == scenario_name]
        for method_name, method_df in scenario_df.groupby("method"):
            axis.plot(method_df["target_rms"], method_df["output"], marker="o", linewidth=1.8, markersize=3, label=method_name)
        axis.set_title(f"{scenario_name}: target RMS vs normalized output")
        axis.set_xlabel("Target RMS")
        axis.set_ylabel("Normalized method output")
        axis.grid(alpha=0.25)
    axes[1].legend(loc="center left", bbox_to_anchor=(1.02, 0.5))
    fig.savefig(AMPLITUDE_PLOT_OUT, dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(2, 1, figsize=(15, 10), constrained_layout=True)
    for axis, scenario_name in zip(axes, ["mean_driven", "structure_driven"], strict=True):
        scenario_df = plot_df[plot_df["scenario"] == scenario_name].copy()
        method_order = list(dict.fromkeys(scenario_df["method"].tolist()))
        state_order = list(dict.fromkeys(scenario_df["state"].tolist()))
        positions = []
        values = []
        labels = []
        tick_positions = []
        tick_labels = []
        x_position = 1
        for method_name in method_order:
            method_df = scenario_df[scenario_df["method"] == method_name]
            block_positions = []
            for state_name in state_order:
                state_values = method_df.loc[method_df["state"] == state_name, "plot_value"].to_numpy(dtype=float)
                positions.append(x_position)
                values.append(state_values)
                labels.append(f"{method_name}\n{state_name}")
                block_positions.append(x_position)
                x_position += 1
            tick_positions.append(float(np.mean(block_positions)))
            tick_labels.append(method_name)
            x_position += 1
        axis.boxplot(values, positions=positions, widths=0.6, patch_artist=False, showfliers=False)
        axis.set_title(f"{scenario_name}: normalized outputs by synthetic state")
        axis.set_ylabel("Normalized output")
        axis.set_xticks(tick_positions)
        axis.set_xticklabels(tick_labels, rotation=20, ha="right")
        axis.grid(axis="y", alpha=0.25)
    fig.savefig(STATE_PLOT_OUT, dpi=160)
    plt.close(fig)


def main() -> int:
    summary: dict[str, object] = {
        "method_blueprints": {
            "softmax_weighted_sum": {
                "input": "global_zscore(window)",
                "core": "softmax(row) * row",
                "effect": "focuses on salient high points inside each window",
            },
            "sumigron": {
                "input": "window statistics",
                "core": "0.35*mu_z + 0.30*sigma_z + 0.35*energy_z",
                "effect": "keeps level, variability and energy explicitly separated",
            },
            "sumigron_attentive": {
                "input": "window statistics + local attention",
                "core": "softmax(0.60*local_z + 0.40*abs(local_z)) -> attentive mu/std/energy -> 0.32*mu_z + 0.28*sigma_z + 0.40*energy_z",
                "effect": "borrows peak sensitivity from softmax while preserving interpretable sumigron channels",
            },
        }
    }
    scenarios = {
        "mean_driven": generate_scenario(
            scenario_name="mean_driven",
            state_configs=mean_driven_states(),
            samples_per_state=320,
            length=160,
            seed=41,
        ),
        "structure_driven": generate_scenario(
            scenario_name="structure_driven",
            state_configs=structure_driven_states(),
            samples_per_state=320,
            length=160,
            seed=77,
        ),
    }

    metric_records: list[dict[str, object]] = []
    plot_records: list[dict[str, object]] = []

    for scenario_name, scenario_df in scenarios.items():
        windows = np.stack(scenario_df["window"].to_numpy())
        methods = build_methods(windows=windows)
        amplitude_target = scenario_df["target_rms"].to_numpy(dtype=float)
        state_labels = scenario_df["state"].to_numpy(dtype=str)
        summary[scenario_name] = {}

        for method_name, method_values in methods.items():
            metrics = method_metrics(method_values, amplitude_target=amplitude_target, state_labels=state_labels)
            metric_records.append(
                {
                    "scenario": scenario_name,
                    "method": method_name,
                    **{key: value for key, value in metrics.items() if key != "state_means"},
                    "state_means_json": json.dumps(metrics["state_means"], ensure_ascii=False),
                }
            )
            summary[scenario_name][method_name] = metrics

            normalized_output = normalize_for_plot(method_values)
            for state_name, target_rms, plot_value in zip(state_labels, amplitude_target, normalized_output, strict=True):
                plot_records.append(
                    {
                        "scenario": scenario_name,
                        "method": method_name,
                        "state": state_name,
                        "target_rms": target_rms,
                        "plot_value": plot_value,
                    }
                )

    metrics_df = pd.DataFrame.from_records(metric_records).sort_values(
        ["scenario", "spearman_target_corr", "avg_pairwise_d", "dynamic_range_p95_p05"],
        ascending=[True, False, False, False],
    )
    plot_df = pd.DataFrame.from_records(plot_records)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    metrics_df.to_csv(DETAIL_OUT, index=False, encoding="utf-8-sig")
    SUMMARY_OUT.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    save_plots(plot_df)

    print(f"metrics_out={DETAIL_OUT}")
    print(f"summary_out={SUMMARY_OUT}")
    print(f"amplitude_plot={AMPLITUDE_PLOT_OUT}")
    print(f"state_plot={STATE_PLOT_OUT}")
    print(metrics_df.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
