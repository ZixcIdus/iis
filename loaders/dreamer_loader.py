"""Загрузчик датасета DREAMER."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.io import loadmat

from config.settings import DATASET_DEFAULTS, DREAMER_EEG_CHANNELS
from features.common import build_segment_record, derive_state_label, segment_bounds, slice_signal, safe_float

LOGGER = logging.getLogger(__name__)


class DREAMERLoader:
    """Загружает DREAMER и разбивает данные на EEG/ECG-сегменты."""

    def __init__(self, root: Path, window_seconds: float | None = None, step_seconds: float | None = None) -> None:
        self.root = Path(root)
        self.window_seconds = window_seconds or DATASET_DEFAULTS["dreamer"]["window_seconds"]
        self.step_seconds = step_seconds or DATASET_DEFAULTS["dreamer"]["step_seconds"]
        self.eeg_sampling_rate = 128.0
        self.ecg_sampling_rate = 256.0

    def load(self) -> pd.DataFrame:
        """Читает файл DREAMER.mat и возвращает таблицу сегментов."""

        mat_path = self._resolve_mat_path()
        mat_data = loadmat(mat_path, verify_compressed_data_integrity=False)
        if "DREAMER" not in mat_data:
            raise KeyError(f"В файле {mat_path} отсутствует ключ DREAMER.")

        dreamer_root = self._unwrap(mat_data["DREAMER"])
        data_field = self._get_field(dreamer_root, "Data")
        subject_entries = self._to_list(data_field)

        records: list[dict[str, Any]] = []
        for subject_index, subject_entry in enumerate(subject_entries):
            subject_id = f"S{subject_index + 1:02d}"
            try:
                records.extend(self._load_subject(subject_entry=subject_entry, subject_id=subject_id))
            except Exception:
                LOGGER.exception("Не удалось обработать субъекта DREAMER %s", subject_id)

        return pd.DataFrame.from_records(records)

    def _resolve_mat_path(self) -> Path:
        """Находит основной mat-файл DREAMER."""

        exact_matches = sorted(self.root.rglob("DREAMER.mat"))
        if exact_matches:
            return exact_matches[0]

        any_matches = sorted(self.root.rglob("*.mat"))
        if any_matches:
            return any_matches[0]

        raise FileNotFoundError(
            f"Файл DREAMER.mat не найден в {self.root}. Ожидается структура вроде data/DREAMER/DREAMER.mat."
        )

    def _load_subject(self, subject_entry: Any, subject_id: str) -> list[dict[str, Any]]:
        """Разворачивает данные одного испытуемого."""

        eeg_struct = self._get_field(subject_entry, "EEG")
        ecg_struct = self._safe_get_field(subject_entry, "ECG")

        eeg_baseline_trials = self._to_list(self._get_field(eeg_struct, "baseline"))
        eeg_stimuli_trials = self._to_list(self._get_field(eeg_struct, "stimuli"))
        ecg_baseline_trials = self._to_list(self._get_field(ecg_struct, "baseline")) if ecg_struct is not None else []
        ecg_stimuli_trials = self._to_list(self._get_field(ecg_struct, "stimuli")) if ecg_struct is not None else []

        valence_scores = self._to_numeric_list(self._get_field(subject_entry, "ScoreValence"))
        arousal_scores = self._to_numeric_list(self._get_field(subject_entry, "ScoreArousal"))
        dominance_scores = self._to_numeric_list(self._safe_get_field(subject_entry, "ScoreDominance"))
        liking_scores = self._to_numeric_list(
            self._safe_get_field(subject_entry, "ScoreLiking", "ScoreLike", "ScoreLikability")
        )

        trial_count = len(eeg_stimuli_trials)
        records: list[dict[str, Any]] = []

        for trial_index in range(trial_count):
            eeg_stimulus = self._prepare_trial_matrix(eeg_stimuli_trials[trial_index], expected_channels=14)
            if eeg_stimulus.size == 0:
                continue

            ecg_stimulus = self._prepare_ecg_signal(ecg_stimuli_trials[trial_index] if trial_index < len(ecg_stimuli_trials) else None)
            valence = valence_scores[trial_index] if trial_index < len(valence_scores) else np.nan
            arousal = arousal_scores[trial_index] if trial_index < len(arousal_scores) else np.nan
            dominance = dominance_scores[trial_index] if trial_index < len(dominance_scores) else np.nan
            liking = liking_scores[trial_index] if trial_index < len(liking_scores) else np.nan

            self_report = {
                "valence": valence,
                "arousal": arousal,
                "dominance": dominance,
                "liking": liking,
            }
            state_label = derive_state_label("dreamer", valence=valence, arousal=arousal)

            baseline_eeg = (
                self._prepare_trial_matrix(eeg_baseline_trials[trial_index], expected_channels=14)
                if trial_index < len(eeg_baseline_trials)
                else np.empty((0, 0), dtype=float)
            )
            baseline_ecg = (
                self._prepare_ecg_signal(ecg_baseline_trials[trial_index])
                if trial_index < len(ecg_baseline_trials)
                else np.asarray([], dtype=float)
            )

            records.extend(
                self._segment_trial(
                    subject_id=subject_id,
                    trial_index=trial_index,
                    label="baseline",
                    eeg_signal=baseline_eeg,
                    ecg_signal=baseline_ecg,
                    self_report={},
                    source_record_id=f"{subject_id}_trial_{trial_index:02d}_baseline",
                )
            )
            records.extend(
                self._segment_trial(
                    subject_id=subject_id,
                    trial_index=trial_index,
                    label=state_label,
                    eeg_signal=eeg_stimulus,
                    ecg_signal=ecg_stimulus,
                    self_report=self_report,
                    source_record_id=f"{subject_id}_trial_{trial_index:02d}_stimulus",
                )
            )

        LOGGER.info("DREAMER: субъект %s, сегментов: %d", subject_id, len(records))
        return records

    def _segment_trial(
        self,
        subject_id: str,
        trial_index: int,
        label: str,
        eeg_signal: np.ndarray,
        ecg_signal: np.ndarray,
        self_report: dict[str, Any],
        source_record_id: str,
    ) -> list[dict[str, Any]]:
        """Делит одно испытание DREAMER на сегменты по EEG-времени."""

        if eeg_signal.size == 0:
            return []

        records: list[dict[str, Any]] = []
        bounds = segment_bounds(
            signal_length=eeg_signal.shape[1],
            sampling_rate=self.eeg_sampling_rate,
            window_seconds=self.window_seconds,
            step_seconds=self.step_seconds,
        )

        for segment_number, (start_index, end_index, start_sec, end_sec) in enumerate(bounds):
            eeg_window = eeg_signal[:, start_index:end_index]
            ecg_window = self._slice_by_seconds(
                signal=ecg_signal,
                sampling_rate=self.ecg_sampling_rate,
                start_sec=start_sec,
                end_sec=end_sec,
            )

            segment_id = f"{subject_id}_trial_{trial_index:02d}_{label}_{segment_number:03d}"
            signals = {
                "eeg": eeg_window,
                "eeg_channels": DREAMER_EEG_CHANNELS,
                "ecg": ecg_window,
            }
            sampling_rates = {
                "eeg": self.eeg_sampling_rate,
                "ecg": self.ecg_sampling_rate,
            }
            metadata = {
                "trial_index": trial_index,
                "segment_origin": "dreamer_mat",
            }

            records.append(
                build_segment_record(
                    subject_id=subject_id,
                    segment_id=segment_id,
                    dataset="dreamer",
                    label=label,
                    signals=signals,
                    sampling_rates=sampling_rates,
                    self_report=self_report,
                    source_record_id=source_record_id,
                    window_start_sec=start_sec,
                    window_end_sec=end_sec,
                    metadata=metadata,
                )
            )

        return records

    def _slice_by_seconds(self, signal: np.ndarray, sampling_rate: float, start_sec: float, end_sec: float) -> np.ndarray:
        """Вырезает окно сигнала по секундам."""

        if signal.size == 0:
            return np.asarray([], dtype=float)
        start_index = int(round(start_sec * sampling_rate))
        end_index = int(round(end_sec * sampling_rate))
        return slice_signal(signal, start=start_index, end=end_index)

    def _prepare_trial_matrix(self, trial_entry: Any, expected_channels: int) -> np.ndarray:
        """Нормализует trial-массив до формы каналы x время."""

        if trial_entry is None:
            return np.empty((0, 0), dtype=float)

        array = np.asarray(self._unwrap(trial_entry), dtype=float)
        array = np.squeeze(array)
        if array.ndim != 2:
            return np.empty((0, 0), dtype=float)

        if array.shape[1] == expected_channels:
            return array[:, :expected_channels].T
        if array.shape[0] == expected_channels:
            return array[:expected_channels, :]
        if array.shape[0] > array.shape[1]:
            return array[:, :expected_channels].T
        return array[:expected_channels, :]

    def _prepare_ecg_signal(self, trial_entry: Any) -> np.ndarray:
        """Преобразует ECG-траекторию в один усреднённый канал."""

        if trial_entry is None:
            return np.asarray([], dtype=float)

        array = np.asarray(self._unwrap(trial_entry), dtype=float)
        array = np.squeeze(array)
        if array.ndim == 1:
            return array.astype(float, copy=False)
        if array.ndim != 2:
            return np.asarray([], dtype=float)
        if array.shape[0] <= array.shape[1]:
            return np.mean(array, axis=0).astype(float, copy=False)
        return np.mean(array, axis=1).astype(float, copy=False)

    def _unwrap(self, value: Any) -> Any:
        """Убирает лишние numpy-обёртки Matlab-структур."""

        current = value
        while isinstance(current, np.ndarray) and current.size == 1:
            current = current.reshape(-1)[0]
        return current

    def _get_field(self, value: Any, field_name: str) -> Any:
        """Извлекает поле из Matlab-структуры."""

        current = self._unwrap(value)
        if current is None:
            raise KeyError(field_name)
        if isinstance(current, np.void) and current.dtype.names and field_name in current.dtype.names:
            return current[field_name]
        if isinstance(current, np.ndarray) and current.dtype.names and field_name in current.dtype.names:
            return current[field_name]
        if isinstance(current, dict) and field_name in current:
            return current[field_name]
        if hasattr(current, field_name):
            return getattr(current, field_name)
        raise KeyError(f"Поле {field_name} отсутствует.")

    def _safe_get_field(self, value: Any, *field_names: str) -> Any | None:
        """Возвращает первое найденное поле или None."""

        for field_name in field_names:
            try:
                return self._get_field(value, field_name)
            except KeyError:
                continue
        return None

    def _to_list(self, value: Any) -> list[Any]:
        """Преобразует Matlab cell-array к линейному python-списку."""

        if value is None:
            return []
        array = np.asarray(value, dtype=object)
        if array.size == 0:
            return []
        return [self._unwrap(item) for item in array.reshape(-1)]

    def _to_numeric_list(self, value: Any) -> list[float]:
        """Преобразует массив оценок к списку float."""

        if value is None:
            return []
        array = np.asarray(value, dtype=float).reshape(-1)
        return [safe_float(item) for item in array]

