"""Извлечение EEG-признаков из сегментов."""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.signal import welch

from config.settings import (
    BCMI_EEG_CHANNELS,
    BCMI_LEFT_CHANNELS,
    BCMI_RIGHT_CHANNELS,
    DEAP_EEG_CHANNELS,
    DEAP_LEFT_CHANNELS,
    DEAP_RIGHT_CHANNELS,
    DREAMER_EEG_CHANNELS,
    DREAMER_LEFT_CHANNELS,
    DREAMER_RIGHT_CHANNELS,
    EEG_BANDS,
    EVA_MED_LEFT_CHANNELS,
    EVA_MED_RIGHT_CHANNELS,
    EPSILON,
    MIN_SIGNAL_SAMPLES,
)

def _ensure_2d(signal: Any) -> np.ndarray:
    """Приводит входной сигнал к двумерному массиву float."""

    array = np.asarray(signal, dtype=float)
    array = np.squeeze(array)
    if array.ndim == 1:
        return array[np.newaxis, :]
    if array.ndim == 0:
        return np.empty((0, 0), dtype=float)
    return array.astype(float, copy=False)


def _dataset_channel_groups(dataset: str) -> tuple[list[str], list[str], list[str]]:
    """Возвращает список всех, левых и правых каналов для датасета."""

    dataset = dataset.lower()
    if dataset == "dreamer":
        return DREAMER_EEG_CHANNELS, DREAMER_LEFT_CHANNELS, DREAMER_RIGHT_CHANNELS
    if dataset == "deap":
        return DEAP_EEG_CHANNELS, DEAP_LEFT_CHANNELS, DEAP_RIGHT_CHANNELS
    if dataset in {"ds002722", "ds002724"}:
        return BCMI_EEG_CHANNELS, BCMI_LEFT_CHANNELS, BCMI_RIGHT_CHANNELS
    if dataset == "eva_med":
        all_channels = list(dict.fromkeys(EVA_MED_LEFT_CHANNELS + EVA_MED_RIGHT_CHANNELS))
        return all_channels, EVA_MED_LEFT_CHANNELS, EVA_MED_RIGHT_CHANNELS
    return [], [], []


def _band_power(signal: np.ndarray, sampling_rate: float, band: tuple[float, float]) -> float:
    """Оценивает среднюю мощность сигнала в полосе частот."""

    if signal.size < MIN_SIGNAL_SAMPLES or sampling_rate <= 0:
        return np.nan

    freqs, power = welch(signal, fs=sampling_rate, nperseg=min(signal.size, int(max(64, sampling_rate * 2))))
    mask = (freqs >= band[0]) & (freqs <= band[1])
    if not np.any(mask):
        return np.nan
    return float(np.trapezoid(power[mask], freqs[mask]))


def _multichannel_band_power(eeg: np.ndarray, sampling_rate: float, band: tuple[float, float]) -> float:
    """Считает среднюю мощность полосы по нескольким каналам."""

    if eeg.ndim != 2 or eeg.shape[1] < MIN_SIGNAL_SAMPLES:
        return np.nan

    channel_values: list[float] = []
    for channel in eeg:
        value = _band_power(channel, sampling_rate=sampling_rate, band=band)
        if np.isfinite(value):
            channel_values.append(value)

    if not channel_values:
        return np.nan
    return float(np.mean(channel_values))


def _resolve_eeg(signals: dict[str, Any], dataset: str) -> tuple[np.ndarray, list[str]]:
    """Нормализует EEG-массив и список каналов."""

    eeg = signals.get("eeg")
    if eeg is None:
        return np.empty((0, 0), dtype=float), []

    eeg_array = _ensure_2d(eeg)
    channel_names = list(signals.get("eeg_channels", []) or [])
    if channel_names and len(channel_names) == eeg_array.shape[0]:
        return eeg_array, channel_names

    all_channels, _, _ = _dataset_channel_groups(dataset)
    if len(all_channels) >= eeg_array.shape[0]:
        return eeg_array, all_channels[: eeg_array.shape[0]]

    return eeg_array, [f"EEG_{index + 1}" for index in range(eeg_array.shape[0])]


def _pick_channels(eeg: np.ndarray, channel_names: list[str], target_channels: list[str]) -> np.ndarray:
    """Извлекает подмножество каналов по именам."""

    if not target_channels or eeg.size == 0 or not channel_names:
        return np.empty((0, 0), dtype=float)

    index_lookup = {name: index for index, name in enumerate(channel_names)}
    indices = [index_lookup[name] for name in target_channels if name in index_lookup]
    if not indices:
        return np.empty((0, 0), dtype=float)
    return eeg[indices, :]


def _empty_eeg_feature_block() -> dict[str, Any]:
    """Возвращает шаблон пустого блока EEG-признаков."""

    features = {
        "eeg_left_power": np.nan,
        "eeg_right_power": np.nan,
        "eeg_gamma_power": np.nan,
        "eeg_alpha_left": np.nan,
        "eeg_alpha_right": np.nan,
        "alpha_asymmetry": np.nan,
        "gamma_alpha_ratio": np.nan,
    }
    for name in list(features):
        features[f"prov_{name}"] = "unavailable"
        features[f"source_{name}"] = ""
    return features


def extract_eeg_features(
    signals: dict[str, Any],
    sampling_rates: dict[str, float],
    dataset: str,
) -> dict[str, Any]:
    """Извлекает набор EEG-признаков и происхождение каждого признака."""

    feature_block = _empty_eeg_feature_block()
    eeg, channel_names = _resolve_eeg(signals=signals, dataset=dataset)
    sampling_rate = float(sampling_rates.get("eeg", np.nan))

    if eeg.size == 0 or not np.isfinite(sampling_rate):
        return feature_block

    _, left_channels, right_channels = _dataset_channel_groups(dataset)
    eeg_left = _pick_channels(eeg=eeg, channel_names=channel_names, target_channels=left_channels)
    eeg_right = _pick_channels(eeg=eeg, channel_names=channel_names, target_channels=right_channels)

    if eeg_left.size == 0 or eeg_right.size == 0:
        return feature_block

    eeg_left_power = _multichannel_band_power(eeg_left, sampling_rate=sampling_rate, band=EEG_BANDS["full"])
    eeg_right_power = _multichannel_band_power(eeg_right, sampling_rate=sampling_rate, band=EEG_BANDS["full"])
    eeg_gamma_power = _multichannel_band_power(eeg, sampling_rate=sampling_rate, band=EEG_BANDS["gamma"])
    eeg_alpha_left = _multichannel_band_power(eeg_left, sampling_rate=sampling_rate, band=EEG_BANDS["alpha"])
    eeg_alpha_right = _multichannel_band_power(eeg_right, sampling_rate=sampling_rate, band=EEG_BANDS["alpha"])

    alpha_asymmetry = np.nan
    gamma_alpha_ratio = np.nan
    if np.isfinite(eeg_alpha_left) and np.isfinite(eeg_alpha_right):
        alpha_asymmetry = float(np.log(eeg_alpha_right + EPSILON) - np.log(eeg_alpha_left + EPSILON))
    if np.isfinite(eeg_gamma_power) and np.isfinite(eeg_alpha_left) and np.isfinite(eeg_alpha_right):
        gamma_alpha_ratio = float(eeg_gamma_power / (eeg_alpha_left + eeg_alpha_right + EPSILON))

    direct_map = {
        "eeg_left_power": eeg_left_power,
        "eeg_right_power": eeg_right_power,
        "eeg_gamma_power": eeg_gamma_power,
        "eeg_alpha_left": eeg_alpha_left,
        "eeg_alpha_right": eeg_alpha_right,
        "alpha_asymmetry": alpha_asymmetry,
        "gamma_alpha_ratio": gamma_alpha_ratio,
    }

    for name, value in direct_map.items():
        feature_block[name] = value
        feature_block[f"prov_{name}"] = "direct" if np.isfinite(value) else "unavailable"
        feature_block[f"source_{name}"] = "eeg"

    return feature_block
