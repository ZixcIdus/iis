"""Загрузчик датасета DEAP."""

from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from config.settings import (
    DATASET_DEFAULTS,
    DEAP_EEG_CHANNELS,
    DEAP_PERIPHERAL_CHANNELS,
)
from features.common import build_segment_record, derive_state_label, segment_bounds, safe_float

LOGGER = logging.getLogger(__name__)


class DEAPLoader:
    """Загружает DEAP и разбивает каждое испытание на сегменты."""

    def __init__(self, root: Path, window_seconds: float | None = None, step_seconds: float | None = None) -> None:
        self.root = Path(root)
        self.window_seconds = window_seconds or DATASET_DEFAULTS["deap"]["window_seconds"]
        self.step_seconds = step_seconds or DATASET_DEFAULTS["deap"]["step_seconds"]
        self.sampling_rate = 128.0
        self.pretrial_seconds = 3.0

    def load(self) -> pd.DataFrame:
        """Читает все доступные .dat-файлы DEAP."""

        subject_files = sorted(self.root.rglob("s*.dat"))
        if not subject_files:
            raise FileNotFoundError(
                f"Файлы DEAP не найдены в {self.root}. Ожидается структура вроде data/DEAP/s01.dat."
            )

        records: list[dict[str, Any]] = []
        for subject_path in subject_files:
            try:
                records.extend(self._load_subject(subject_path))
            except Exception:
                LOGGER.exception("Не удалось обработать DEAP-файл %s", subject_path)

        return pd.DataFrame.from_records(records)

    def _load_subject(self, subject_path: Path) -> list[dict[str, Any]]:
        """Обрабатывает одного испытуемого DEAP."""

        with subject_path.open("rb") as file_pointer:
            payload = pickle.load(file_pointer, encoding="latin1")

        data = np.asarray(payload.get("data"), dtype=float)
        labels = np.asarray(payload.get("labels"), dtype=float)
        if data.ndim != 3 or labels.ndim != 2:
            LOGGER.warning("Файл %s имеет неожиданный формат DEAP.", subject_path)
            return []

        subject_id = subject_path.stem.upper()
        records: list[dict[str, Any]] = []
        eeg_channel_count = len(DEAP_EEG_CHANNELS)
        peripheral_channel_count = len(DEAP_PERIPHERAL_CHANNELS)
        expected_channels = eeg_channel_count + peripheral_channel_count
        if data.shape[1] < expected_channels:
            LOGGER.warning("В файле %s ожидается 40 каналов, найдено %d.", subject_path, data.shape[1])
            return []

        pretrial_samples = int(round(self.pretrial_seconds * self.sampling_rate))

        for trial_index in range(data.shape[0]):
            trial_matrix = data[trial_index]
            eeg = trial_matrix[:eeg_channel_count, pretrial_samples:]
            peripheral = trial_matrix[eeg_channel_count : eeg_channel_count + peripheral_channel_count, pretrial_samples:]
            valence = safe_float(labels[trial_index, 0] if labels.shape[1] > 0 else np.nan)
            arousal = safe_float(labels[trial_index, 1] if labels.shape[1] > 1 else np.nan)
            dominance = safe_float(labels[trial_index, 2] if labels.shape[1] > 2 else np.nan)
            liking = safe_float(labels[trial_index, 3] if labels.shape[1] > 3 else np.nan)

            self_report = {
                "valence": valence,
                "arousal": arousal,
                "dominance": dominance,
                "liking": liking,
            }
            state_label = derive_state_label("deap", valence=valence, arousal=arousal)

            bounds = segment_bounds(
                signal_length=eeg.shape[1],
                sampling_rate=self.sampling_rate,
                window_seconds=self.window_seconds,
                step_seconds=self.step_seconds,
            )

            for segment_number, (start_index, end_index, start_sec, end_sec) in enumerate(bounds):
                segment_id = f"{subject_id}_trial_{trial_index:02d}_{segment_number:03d}"
                signals = {
                    "eeg": eeg[:, start_index:end_index],
                    "eeg_channels": DEAP_EEG_CHANNELS,
                    "resp": peripheral[5, start_index:end_index],
                    "ppg": peripheral[6, start_index:end_index],
                    "gsr": peripheral[4, start_index:end_index],
                    "temperature": peripheral[7, start_index:end_index],
                    "peripheral_channels": DEAP_PERIPHERAL_CHANNELS,
                }
                sampling_rates = {
                    "eeg": self.sampling_rate,
                    "resp": self.sampling_rate,
                    "ppg": self.sampling_rate,
                    "gsr": self.sampling_rate,
                    "temperature": self.sampling_rate,
                }
                metadata = {
                    "trial_index": trial_index,
                    "segment_origin": "deap_preprocessed_python",
                }

                records.append(
                    build_segment_record(
                        subject_id=subject_id,
                        segment_id=segment_id,
                        dataset="deap",
                        label=state_label,
                        signals=signals,
                        sampling_rates=sampling_rates,
                        self_report=self_report,
                        source_record_id=f"{subject_id}_trial_{trial_index:02d}",
                        window_start_sec=start_sec + self.pretrial_seconds,
                        window_end_sec=end_sec + self.pretrial_seconds,
                        metadata=metadata,
                    )
                )

        LOGGER.info("DEAP: субъект %s, сегментов: %d", subject_id, len(records))
        return records

