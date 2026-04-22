"""Общие функции сегментации и формирования таблицы признаков."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

from config.settings import EPSILON, LABEL_ENCODING, STATE_THRESHOLDS
from features.eeg_features import extract_eeg_features
from features.hrv_features import extract_hrv_features

LOGGER = logging.getLogger(__name__)
_LAZY_SIGNAL_CACHE: dict[str, Any] = {
    "path": None,
    "channels": tuple(),
    "data": None,
    "channel_lookup": {},
}


def safe_float(value: Any) -> float:
    """Преобразует значение в float и возвращает NaN при ошибке."""

    if value is None:
        return np.nan
    if isinstance(value, (float, int, np.floating, np.integer)):
        return float(value)
    try:
        return float(value)
    except (TypeError, ValueError):
        return np.nan


def ensure_1d(signal: Any) -> np.ndarray:
    """Приводит входной сигнал к одномерному массиву float."""

    array = np.asarray(signal, dtype=float).squeeze()
    if array.ndim == 0:
        return np.asarray([], dtype=float)
    if array.ndim > 1:
        return np.asarray(array.reshape(-1), dtype=float)
    return array.astype(float, copy=False)


def ensure_2d(signal: Any) -> np.ndarray:
    """Приводит входной сигнал к двумерному массиву float."""

    array = np.asarray(signal, dtype=float)
    array = np.squeeze(array)
    if array.ndim == 1:
        return array[np.newaxis, :]
    if array.ndim == 0:
        return np.empty((0, 0), dtype=float)
    return array.astype(float, copy=False)


def safe_json_dumps(payload: Any) -> str:
    """Сериализует объект в JSON с поддержкой NaN и numpy-типов."""

    def _default(value: Any) -> Any:
        if isinstance(value, (np.floating, np.integer)):
            return value.item()
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, set):
            return sorted(value)
        return str(value)

    return json.dumps(payload, ensure_ascii=False, default=_default)


def _materialize_lazy_signals(signals: dict[str, Any]) -> dict[str, Any]:
    """Лениво извлекает срезы сигналов из EDF-файла по метаданным сегмента."""

    if signals.get("__lazy_loader__") != "edf_slice":
        return signals

    recording_path = Path(str(signals.get("recording_path", ""))).expanduser().resolve()
    if not recording_path.exists():
        raise FileNotFoundError(f"Не найден EDF-файл сегмента: {recording_path}")

    start_sample = max(int(signals.get("start_sample", 0)), 0)
    end_sample = max(int(signals.get("end_sample", 0)), start_sample)
    eeg_channels = list(signals.get("eeg_channels", []) or [])
    ecg_channel = str(signals.get("ecg_channel", "") or "")
    ppg_channel = str(signals.get("ppg_channel", "") or "")
    gsr_channel = str(signals.get("gsr_channel", "") or "")

    requested_channels = list(
        dict.fromkeys(
            [*eeg_channels]
            + ([ecg_channel] if ecg_channel else [])
            + ([ppg_channel] if ppg_channel else [])
            + ([gsr_channel] if gsr_channel else [])
        )
    )

    if not requested_channels:
        return {}

    cache_key = str(recording_path)
    cache_channels = tuple(requested_channels)
    if _LAZY_SIGNAL_CACHE["path"] != cache_key or _LAZY_SIGNAL_CACHE["channels"] != cache_channels:
        try:
            import mne  # type: ignore
        except ImportError as error:
            raise ImportError(
                "Для чтения OpenNeuro EDF-сегментов требуется пакет mne. "
                "Установите его командой: python -m pip install mne"
            ) from error

        raw = mne.io.read_raw_edf(recording_path, preload=False, verbose="ERROR")
        available_channels = [name for name in requested_channels if name in raw.ch_names]
        if not available_channels:
            if hasattr(raw, "close"):
                raw.close()
            return {}

        data = raw.get_data(picks=available_channels).astype(np.float32, copy=False)
        if hasattr(raw, "close"):
            raw.close()

        _LAZY_SIGNAL_CACHE["path"] = cache_key
        _LAZY_SIGNAL_CACHE["channels"] = tuple(available_channels)
        _LAZY_SIGNAL_CACHE["data"] = data
        _LAZY_SIGNAL_CACHE["channel_lookup"] = {
            name: index for index, name in enumerate(available_channels)
        }

    data = _LAZY_SIGNAL_CACHE["data"]
    channel_lookup = _LAZY_SIGNAL_CACHE["channel_lookup"]
    if data is None:
        return {}

    upper_bound = data.shape[1]
    start_sample = min(start_sample, upper_bound)
    end_sample = min(end_sample, upper_bound)

    resolved: dict[str, Any] = {}
    available_eeg_channels = [name for name in eeg_channels if name in channel_lookup]
    if available_eeg_channels:
        eeg_indices = [channel_lookup[name] for name in available_eeg_channels]
        resolved["eeg"] = data[eeg_indices, start_sample:end_sample]
        resolved["eeg_channels"] = available_eeg_channels

    if ecg_channel and ecg_channel in channel_lookup:
        resolved["ecg"] = data[channel_lookup[ecg_channel], start_sample:end_sample]
    if ppg_channel and ppg_channel in channel_lookup:
        resolved["ppg"] = data[channel_lookup[ppg_channel], start_sample:end_sample]
    if gsr_channel and gsr_channel in channel_lookup:
        resolved["gsr"] = data[channel_lookup[gsr_channel], start_sample:end_sample]

    return resolved


def segment_bounds(
    signal_length: int,
    sampling_rate: float,
    window_seconds: float,
    step_seconds: float,
) -> list[tuple[int, int, float, float]]:
    """Возвращает границы сегментов в отсчётах и секундах."""

    if signal_length <= 0 or sampling_rate <= 0 or window_seconds <= 0 or step_seconds <= 0:
        return []

    window_samples = int(round(window_seconds * sampling_rate))
    step_samples = int(round(step_seconds * sampling_rate))
    if window_samples <= 0 or step_samples <= 0 or signal_length < window_samples:
        return []

    bounds: list[tuple[int, int, float, float]] = []
    start = 0
    while start + window_samples <= signal_length:
        end = start + window_samples
        bounds.append((start, end, start / sampling_rate, end / sampling_rate))
        start += step_samples
    return bounds


def slice_signal(signal: Any, start: int, end: int) -> np.ndarray:
    """Вырезает временное окно из одномерного или двумерного сигнала."""

    array = np.asarray(signal)
    if array.ndim == 1:
        return np.asarray(array[start:end], dtype=float)
    if array.ndim == 2:
        return np.asarray(array[:, start:end], dtype=float)
    squeezed = np.squeeze(array)
    if squeezed.ndim == 1:
        return np.asarray(squeezed[start:end], dtype=float)
    if squeezed.ndim == 2:
        return np.asarray(squeezed[:, start:end], dtype=float)
    raise ValueError("Ожидался одномерный или двумерный сигнал.")


def build_segment_record(
    subject_id: str,
    segment_id: str,
    dataset: str,
    label: str,
    signals: dict[str, Any],
    sampling_rates: dict[str, float],
    self_report: dict[str, Any] | None,
    source_record_id: str,
    window_start_sec: float,
    window_end_sec: float,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Формирует унифицированную запись сегмента."""

    return {
        "subject_id": subject_id,
        "segment_id": segment_id,
        "dataset": dataset,
        "label": label,
        "signals": signals,
        "sampling_rates": sampling_rates,
        "self_report": self_report or {},
        "source_record_id": source_record_id,
        "window_start_sec": safe_float(window_start_sec),
        "window_end_sec": safe_float(window_end_sec),
        "segment_duration_sec": safe_float(window_end_sec) - safe_float(window_start_sec),
        "metadata": metadata or {},
    }


def expand_segments_dataframe(
    segments_df: pd.DataFrame,
    window_seconds: float,
    step_seconds: float,
) -> pd.DataFrame:
    """Дробит крупные сегменты на причинно-упорядоченные подокна для динамического анализа."""

    if segments_df.empty or window_seconds <= 0 or step_seconds <= 0:
        return segments_df.copy()

    expanded_records: list[dict[str, Any]] = []
    for row in segments_df.to_dict(orient="records"):
        duration_seconds = _resolve_segment_duration_seconds(row)
        if not np.isfinite(duration_seconds) or duration_seconds <= 0 or duration_seconds <= window_seconds:
            expanded_records.append(row)
            continue

        bounds = segment_bounds(
            signal_length=int(round(duration_seconds * 1000.0)),
            sampling_rate=1000.0,
            window_seconds=window_seconds,
            step_seconds=step_seconds,
        )
        if not bounds:
            expanded_records.append(row)
            continue

        parent_start_sec = safe_float(row.get("window_start_sec"))
        if not np.isfinite(parent_start_sec):
            parent_start_sec = 0.0
        parent_end_sec = safe_float(row.get("window_end_sec"))
        if not np.isfinite(parent_end_sec):
            parent_end_sec = parent_start_sec + duration_seconds

        for window_index, (_, _, local_start_sec, local_end_sec) in enumerate(bounds):
            absolute_start_sec = parent_start_sec + local_start_sec
            absolute_end_sec = min(parent_start_sec + local_end_sec, parent_end_sec)

            expanded_record = dict(row)
            expanded_record["segment_id"] = f"{row.get('segment_id', 'segment')}_dyn{window_index:03d}"
            expanded_record["window_start_sec"] = absolute_start_sec
            expanded_record["window_end_sec"] = absolute_end_sec
            expanded_record["segment_duration_sec"] = absolute_end_sec - absolute_start_sec

            metadata = dict(row.get("metadata", {}) or {})
            metadata["dynamic_parent_segment_id"] = row.get("segment_id")
            metadata["dynamic_window_index"] = window_index
            metadata["dynamic_window_seconds"] = window_seconds
            metadata["dynamic_step_seconds"] = step_seconds
            expanded_record["metadata"] = metadata

            signals = row.get("signals", {}) or {}
            sampling_rates = row.get("sampling_rates", {}) or {}
            expanded_record["signals"] = _slice_segment_signals(
                signals=signals,
                sampling_rates=sampling_rates,
                local_start_sec=local_start_sec,
                local_end_sec=local_end_sec,
            )
            expanded_records.append(expanded_record)

    return pd.DataFrame.from_records(expanded_records)


def _resolve_segment_duration_seconds(row: dict[str, Any]) -> float:
    """Оценивает длительность сегмента в секундах."""

    duration = safe_float(row.get("segment_duration_sec"))
    if np.isfinite(duration) and duration > 0:
        return duration

    start_sec = safe_float(row.get("window_start_sec"))
    end_sec = safe_float(row.get("window_end_sec"))
    if np.isfinite(start_sec) and np.isfinite(end_sec) and end_sec > start_sec:
        return end_sec - start_sec

    signals = row.get("signals", {}) or {}
    sampling_rates = row.get("sampling_rates", {}) or {}
    for signal_name, signal_value in signals.items():
        if signal_name.startswith("__") or signal_name.endswith("_channels") or signal_name in {"recording_path"}:
            continue
        rate = safe_float(sampling_rates.get(signal_name))
        if not np.isfinite(rate) or rate <= 0:
            continue
        array = np.asarray(signal_value)
        if array.ndim == 1 and array.size > 0:
            return float(array.size / rate)
        if array.ndim == 2 and array.shape[-1] > 0:
            return float(array.shape[-1] / rate)

    if signals.get("__lazy_loader__") == "edf_slice":
        start_sample = safe_float(signals.get("start_sample"))
        end_sample = safe_float(signals.get("end_sample"))
        candidate_rates = [
            safe_float(sampling_rates.get(name))
            for name in ("eeg", "ecg", "ppg", "gsr")
            if np.isfinite(safe_float(sampling_rates.get(name)))
        ]
        if np.isfinite(start_sample) and np.isfinite(end_sample) and end_sample > start_sample and candidate_rates:
            return float((end_sample - start_sample) / candidate_rates[0])

    return np.nan


def _slice_segment_signals(
    signals: dict[str, Any],
    sampling_rates: dict[str, float],
    local_start_sec: float,
    local_end_sec: float,
) -> dict[str, Any]:
    """Вырезает подокно из исходных сигналов сегмента."""

    if signals.get("__lazy_loader__") == "edf_slice":
        return _slice_lazy_signals(signals=signals, sampling_rates=sampling_rates, local_start_sec=local_start_sec, local_end_sec=local_end_sec)

    sliced: dict[str, Any] = {}
    for key, value in signals.items():
        if key.endswith("_channels"):
            sliced[key] = list(value) if isinstance(value, (list, tuple)) else value
            continue
        if key.startswith("__") or key == "recording_path":
            sliced[key] = value
            continue

        rate = safe_float(sampling_rates.get(key))
        if not np.isfinite(rate) or rate <= 0:
            sliced[key] = value
            continue

        start_sample = max(int(round(local_start_sec * rate)), 0)
        end_sample = max(int(round(local_end_sec * rate)), start_sample)
        try:
            sliced[key] = slice_signal(value, start_sample, end_sample)
        except Exception:
            sliced[key] = value

    return sliced


def _slice_lazy_signals(
    signals: dict[str, Any],
    sampling_rates: dict[str, float],
    local_start_sec: float,
    local_end_sec: float,
) -> dict[str, Any]:
    """Сдвигает границы ленивого EDF-среза без загрузки полного файла."""

    sliced = dict(signals)
    base_rate = np.nan
    for key in ("eeg", "ecg", "ppg", "gsr"):
        rate = safe_float(sampling_rates.get(key))
        if np.isfinite(rate) and rate > 0:
            base_rate = rate
            break

    if not np.isfinite(base_rate) or base_rate <= 0:
        return sliced

    start_sample = int(safe_float(signals.get("start_sample")))
    end_sample = int(safe_float(signals.get("end_sample")))
    offset_start = max(int(round(local_start_sec * base_rate)), 0)
    offset_end = max(int(round(local_end_sec * base_rate)), offset_start)
    sliced["start_sample"] = min(start_sample + offset_start, end_sample)
    sliced["end_sample"] = min(start_sample + offset_end, end_sample)
    return sliced


def derive_state_label(dataset: str, valence: float | None, arousal: float | None) -> str:
    """Строит трёхсостоянийную метку по self-report для DREAMER и DEAP."""

    dataset = dataset.lower()
    valence_value = safe_float(valence)
    arousal_value = safe_float(arousal)
    if np.isnan(valence_value) or np.isnan(arousal_value):
        return "disbalance"

    thresholds = STATE_THRESHOLDS.get(dataset)
    if thresholds is None:
        return "disbalance"

    low = thresholds["low"]
    high = thresholds["high"]

    if arousal_value >= high and valence_value <= low:
        return "stress"
    if arousal_value <= low and valence_value >= high:
        return "baseline"
    return "disbalance"


def encode_stress_label(label: str, dataset: str, self_report: dict[str, Any] | None = None) -> tuple[float, str]:
    """Возвращает численный индикатор нагрузки и происхождение метки."""

    label_key = str(label).strip().lower()
    if label_key in LABEL_ENCODING:
        provenance = "direct" if dataset.lower() == "wesad" else "proxy"
        return LABEL_ENCODING[label_key], provenance

    self_report = self_report or {}
    valence = safe_float(self_report.get("valence"))
    arousal = safe_float(self_report.get("arousal"))
    derived_label = derive_state_label(dataset, valence, arousal)
    return LABEL_ENCODING.get(derived_label, np.nan), "proxy"


def feature_columns() -> list[str]:
    """Возвращает базовый список числовых признаков для отчётности."""

    return [
        "eeg_left_power",
        "eeg_right_power",
        "eeg_gamma_power",
        "eeg_alpha_left",
        "eeg_alpha_right",
        "alpha_asymmetry",
        "gamma_alpha_ratio",
        "hrv_hf",
        "hrv_lf",
        "hrv_hf_lf",
        "hrv_lf_hf",
        "heart_rate",
        "hrv_rmssd",
        "hrv_sdnn",
        "stress_label",
        "valence",
        "arousal",
        "dominance",
        "liking",
    ]


def summarize_available_features(features_df: pd.DataFrame) -> dict[str, Any]:
    """Формирует краткую сводку доступности признаков."""

    summary: dict[str, Any] = {}
    for column in feature_columns():
        if column not in features_df.columns:
            continue
        valid_count = int(features_df[column].notna().sum())
        summary[column] = {
            "valid_segments": valid_count,
            "share": round(valid_count / max(len(features_df), 1), 4),
        }
    return summary


def extract_features_dataframe(
    segments_df: pd.DataFrame,
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
) -> pd.DataFrame:
    """Извлекает все доступные признаки для таблицы сегментов."""

    records: list[dict[str, Any]] = []
    rows = segments_df.to_dict(orient="records")
    total_rows = len(rows)

    for row_index, row in enumerate(rows, start=1):
        if progress_callback is not None and (row_index == 1 or row_index % 100 == 0 or row_index == total_rows):
            progress_callback(
                {
                    "stage": "features",
                    "current": row_index,
                    "total": total_rows,
                    "message": f"Извлечение признаков {row_index}/{total_rows}",
                }
            )

        signals = _materialize_lazy_signals(row.get("signals", {}) or {})
        sampling_rates = row.get("sampling_rates", {}) or {}
        dataset = str(row.get("dataset", "")).lower()
        self_report = row.get("self_report", {}) or {}

        eeg_features = extract_eeg_features(signals=signals, sampling_rates=sampling_rates, dataset=dataset)
        hrv_features = extract_hrv_features(signals=signals, sampling_rates=sampling_rates)

        valence = safe_float(self_report.get("valence"))
        arousal = safe_float(self_report.get("arousal"))
        dominance = safe_float(self_report.get("dominance"))
        liking = safe_float(self_report.get("liking"))
        stress_score, stress_provenance = encode_stress_label(
            label=str(row.get("label", "")),
            dataset=dataset,
            self_report=self_report,
        )

        record: dict[str, Any] = {
            "subject_id": row.get("subject_id"),
            "segment_id": row.get("segment_id"),
            "dataset": dataset,
            "label": row.get("label"),
            "source_record_id": row.get("source_record_id"),
            "window_start_sec": safe_float(row.get("window_start_sec")),
            "window_end_sec": safe_float(row.get("window_end_sec")),
            "segment_duration_sec": safe_float(row.get("segment_duration_sec")),
            "valence": valence,
            "arousal": arousal,
            "dominance": dominance,
            "liking": liking,
            "stress_label": stress_score,
            "prov_stress_label": stress_provenance,
            "source_stress_label": "dataset_label" if dataset == "wesad" else "self_report_label_logic",
            "self_report_json": safe_json_dumps(self_report),
            "metadata_json": safe_json_dumps(row.get("metadata", {})),
        }

        record.update(eeg_features)
        record.update(hrv_features)

        provenance_summary = {
            key.replace("prov_", ""): value
            for key, value in record.items()
            if key.startswith("prov_")
        }
        direct_count = sum(1 for value in provenance_summary.values() if value == "direct")
        proxy_count = sum(1 for value in provenance_summary.values() if value == "proxy")
        unavailable_count = sum(1 for value in provenance_summary.values() if value == "unavailable")

        record["direct_feature_count"] = direct_count
        record["proxy_feature_count"] = proxy_count
        record["unavailable_feature_count"] = unavailable_count
        record["feature_provenance_json"] = safe_json_dumps(provenance_summary)

        records.append(record)

    features_df = pd.DataFrame.from_records(records)
    if features_df.empty:
        LOGGER.warning("Таблица признаков пуста.")
        return features_df

    numeric_columns = [
        "window_start_sec",
        "window_end_sec",
        "segment_duration_sec",
        "valence",
        "arousal",
        "dominance",
        "liking",
        "stress_label",
        "direct_feature_count",
        "proxy_feature_count",
        "unavailable_feature_count",
    ] + feature_columns()

    for column in numeric_columns:
        if column in features_df.columns:
            features_df[column] = pd.to_numeric(features_df[column], errors="coerce")

    return features_df
