"""Извлечение HRV-признаков из ECG или PPG."""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.interpolate import interp1d
from scipy.signal import butter, detrend, filtfilt, find_peaks, welch

from config.settings import EPSILON, HRV_BANDS, MIN_RR_COUNT, RR_INTERPOLATION_FREQUENCY

def _ensure_1d(signal: Any) -> np.ndarray:
    """Приводит входной сигнал к одномерному массиву float."""

    array = np.asarray(signal, dtype=float).squeeze()
    if array.ndim == 0:
        return np.asarray([], dtype=float)
    if array.ndim > 1:
        return np.asarray(array.reshape(-1), dtype=float)
    return array.astype(float, copy=False)


def _empty_hrv_feature_block() -> dict[str, Any]:
    """Возвращает шаблон пустого блока HRV-признаков."""

    features = {
        "hrv_hf": np.nan,
        "hrv_lf": np.nan,
        "hrv_hf_lf": np.nan,
        "hrv_lf_hf": np.nan,
        "heart_rate": np.nan,
        "hrv_rmssd": np.nan,
        "hrv_sdnn": np.nan,
    }
    for name in list(features):
        features[f"prov_{name}"] = "unavailable"
        features[f"source_{name}"] = ""
    return features


def _bandpass_filter(signal: np.ndarray, sampling_rate: float, low: float, high: float) -> np.ndarray:
    """Применяет полосовой фильтр Баттерворта."""

    nyquist = 0.5 * sampling_rate
    if nyquist <= 0 or low >= high or high >= nyquist:
        return signal
    b_coeff, a_coeff = butter(3, [low / nyquist, high / nyquist], btype="bandpass")
    min_required = 3 * max(len(a_coeff), len(b_coeff))
    if signal.size <= min_required:
        return signal
    return filtfilt(b_coeff, a_coeff, signal)


def _integrate_band(freqs: np.ndarray, power: np.ndarray, band: tuple[float, float]) -> float:
    """Интегрирует спектральную мощность по частотной полосе."""

    mask = (freqs >= band[0]) & (freqs <= band[1])
    if not np.any(mask):
        return np.nan
    return float(np.trapezoid(power[mask], freqs[mask]))


def _rr_frequency_features(rr_intervals_sec: np.ndarray) -> tuple[float, float]:
    """Считает LF и HF мощность по последовательности RR-интервалов."""

    if rr_intervals_sec.size < MIN_RR_COUNT:
        return np.nan, np.nan

    rr_times = np.cumsum(rr_intervals_sec)
    rr_times -= rr_times[0]
    if rr_times[-1] <= 0:
        return np.nan, np.nan

    interpolation_time = np.arange(0.0, rr_times[-1], 1.0 / RR_INTERPOLATION_FREQUENCY)
    if interpolation_time.size < 8:
        return np.nan, np.nan

    interpolator = interp1d(rr_times, rr_intervals_sec, kind="linear", fill_value="extrapolate")
    resampled_rr = detrend(interpolator(interpolation_time))
    freqs, power = welch(
        resampled_rr,
        fs=RR_INTERPOLATION_FREQUENCY,
        nperseg=min(resampled_rr.size, 256),
    )

    lf_power = _integrate_band(freqs=freqs, power=power, band=HRV_BANDS["lf"])
    hf_power = _integrate_band(freqs=freqs, power=power, band=HRV_BANDS["hf"])
    return lf_power, hf_power


def _detect_intervals(signal: np.ndarray, sampling_rate: float, signal_kind: str) -> np.ndarray:
    """Находит интервалы между ударами сердца по ECG или PPG."""

    if signal_kind == "ecg":
        filtered = _bandpass_filter(signal, sampling_rate=sampling_rate, low=5.0, high=18.0)
        distance = int(max(1, 0.3 * sampling_rate))
        prominence = max(np.std(filtered) * 0.4, EPSILON)
    else:
        filtered = _bandpass_filter(signal, sampling_rate=sampling_rate, low=0.5, high=5.0)
        distance = int(max(1, 0.4 * sampling_rate))
        prominence = max(np.std(filtered) * 0.25, EPSILON)

    peaks, _ = find_peaks(filtered, distance=distance, prominence=prominence)
    if peaks.size < MIN_RR_COUNT + 1:
        return np.asarray([], dtype=float)

    intervals = np.diff(peaks) / sampling_rate
    intervals = intervals[(intervals >= 0.3) & (intervals <= 2.0)]
    return intervals.astype(float, copy=False)


def _hrv_from_signal(signal: np.ndarray, sampling_rate: float, signal_kind: str) -> dict[str, float]:
    """Считает набор HRV-признаков по сердечному сигналу."""

    rr_intervals = _detect_intervals(signal=signal, sampling_rate=sampling_rate, signal_kind=signal_kind)
    if rr_intervals.size < MIN_RR_COUNT:
        return {
            "hrv_hf": np.nan,
            "hrv_lf": np.nan,
            "hrv_hf_lf": np.nan,
            "hrv_lf_hf": np.nan,
            "heart_rate": np.nan,
            "hrv_rmssd": np.nan,
            "hrv_sdnn": np.nan,
        }

    lf_power, hf_power = _rr_frequency_features(rr_intervals_sec=rr_intervals)
    diff_rr = np.diff(rr_intervals)
    rmssd = np.sqrt(np.mean(np.square(diff_rr))) * 1000.0 if diff_rr.size else np.nan
    sdnn = np.std(rr_intervals, ddof=1) * 1000.0 if rr_intervals.size > 1 else np.nan
    mean_rr = np.mean(rr_intervals)
    heart_rate = 60.0 / mean_rr if mean_rr > 0 else np.nan

    return {
        "hrv_hf": hf_power,
        "hrv_lf": lf_power,
        "hrv_hf_lf": hf_power / (lf_power + EPSILON) if np.isfinite(hf_power) and np.isfinite(lf_power) else np.nan,
        "hrv_lf_hf": lf_power / (hf_power + EPSILON) if np.isfinite(hf_power) and np.isfinite(lf_power) else np.nan,
        "heart_rate": heart_rate,
        "hrv_rmssd": rmssd,
        "hrv_sdnn": sdnn,
    }


def _populate_feature_block(
    values: dict[str, float],
    provenance: str,
    source: str,
) -> dict[str, Any]:
    """Строит словарь признаков и метаданных происхождения."""

    block = _empty_hrv_feature_block()
    for name, value in values.items():
        block[name] = value
        block[f"prov_{name}"] = provenance if np.isfinite(value) else "unavailable"
        block[f"source_{name}"] = source if np.isfinite(value) else ""
    return block


def extract_hrv_features(signals: dict[str, Any], sampling_rates: dict[str, float]) -> dict[str, Any]:
    """Извлекает HRV-признаки, предпочитая ECG как прямой источник."""

    ecg_signal = _ensure_1d(signals.get("ecg", []))
    ecg_rate = float(sampling_rates.get("ecg", np.nan))
    if ecg_signal.size > 0 and np.isfinite(ecg_rate) and ecg_rate > 0:
        ecg_values = _hrv_from_signal(signal=ecg_signal, sampling_rate=ecg_rate, signal_kind="ecg")
        if any(np.isfinite(value) for value in ecg_values.values()):
            return _populate_feature_block(values=ecg_values, provenance="direct", source="ecg")

    ppg_signal = _ensure_1d(signals.get("ppg", []))
    ppg_rate = float(sampling_rates.get("ppg", np.nan))
    if ppg_signal.size > 0 and np.isfinite(ppg_rate) and ppg_rate > 0:
        ppg_values = _hrv_from_signal(signal=ppg_signal, sampling_rate=ppg_rate, signal_kind="ppg")
        if any(np.isfinite(value) for value in ppg_values.values()):
            return _populate_feature_block(values=ppg_values, provenance="proxy", source="ppg")

    return _empty_hrv_feature_block()
