"""Загрузчик датасета WESAD."""

from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from config.settings import DATASET_DEFAULTS, WESAD_LABEL_MAP, WESAD_SAMPLING_RATES
from features.common import build_segment_record, ensure_1d, segment_bounds

LOGGER = logging.getLogger(__name__)


class WESADLoader:
    """Загружает WESAD и возвращает сегменты в унифицированном формате."""

    def __init__(self, root: Path, window_seconds: float | None = None, step_seconds: float | None = None) -> None:
        self.root = Path(root)
        self.window_seconds = window_seconds or DATASET_DEFAULTS["wesad"]["window_seconds"]
        self.step_seconds = step_seconds or DATASET_DEFAULTS["wesad"]["step_seconds"]

    def load(self) -> pd.DataFrame:
        """Читает все доступные файлы WESAD."""

        subject_files = sorted(self.root.rglob("S*.pkl"))
        if not subject_files:
            raise FileNotFoundError(
                f"Файлы WESAD не найдены в {self.root}. Ожидается структура вроде data/WESAD/S2/S2.pkl."
            )

        records: list[dict[str, Any]] = []
        for subject_path in subject_files:
            try:
                records.extend(self._load_subject(subject_path))
            except Exception:
                LOGGER.exception("Не удалось обработать WESAD-файл %s", subject_path)

        return pd.DataFrame.from_records(records)

    def _load_subject(self, subject_path: Path) -> list[dict[str, Any]]:
        """Читает одного испытуемого WESAD."""

        with subject_path.open("rb") as file_pointer:
            payload = pickle.load(file_pointer, encoding="latin1")

        subject_id = subject_path.stem.upper()
        labels = ensure_1d(payload.get("label", []))
        signal_block = payload.get("signal", {}) or {}
        chest = signal_block.get("chest", {}) or {}

        ecg = ensure_1d(chest.get("ECG", []))
        if labels.size == 0 or ecg.size == 0:
            LOGGER.warning("В WESAD-файле %s отсутствуют ECG или labels.", subject_path)
            return []

        min_length = min(labels.size, ecg.size)
        labels = labels[:min_length]
        ecg = ecg[:min_length]
        resp = ensure_1d(chest.get("Resp", []))[:min_length]
        eda = ensure_1d(chest.get("EDA", []))[:min_length]
        emg = ensure_1d(chest.get("EMG", []))[:min_length]
        temp = ensure_1d(chest.get("Temp", []))[:min_length]

        records: list[dict[str, Any]] = []
        start_index = 0
        run_index = 0
        sampling_rate = float(WESAD_SAMPLING_RATES["ecg"])

        while start_index < labels.size:
            label_code = int(labels[start_index])
            end_index = start_index + 1
            while end_index < labels.size and int(labels[end_index]) == label_code:
                end_index += 1

            label_name = WESAD_LABEL_MAP.get(label_code, "other")
            if label_name in {"baseline", "stress", "amusement"}:
                run_records = self._segment_run(
                    subject_id=subject_id,
                    run_index=run_index,
                    label_code=label_code,
                    label_name=label_name,
                    start_index=start_index,
                    end_index=end_index,
                    ecg=ecg,
                    resp=resp,
                    eda=eda,
                    emg=emg,
                    temp=temp,
                    sampling_rate=sampling_rate,
                )
                records.extend(run_records)
                run_index += 1

            start_index = end_index

        LOGGER.info("WESAD: субъект %s, сегментов: %d", subject_id, len(records))
        return records

    def _segment_run(
        self,
        subject_id: str,
        run_index: int,
        label_code: int,
        label_name: str,
        start_index: int,
        end_index: int,
        ecg: np.ndarray,
        resp: np.ndarray,
        eda: np.ndarray,
        emg: np.ndarray,
        temp: np.ndarray,
        sampling_rate: float,
    ) -> list[dict[str, Any]]:
        """Делит непрерывный участок одной метки на окна."""

        segment_records: list[dict[str, Any]] = []
        bounds = segment_bounds(
            signal_length=end_index - start_index,
            sampling_rate=sampling_rate,
            window_seconds=self.window_seconds,
            step_seconds=self.step_seconds,
        )

        for segment_number, (local_start, local_end, start_sec, end_sec) in enumerate(bounds):
            global_start = start_index + local_start
            global_end = start_index + local_end
            segment_id = f"{subject_id}_{label_name}_{run_index:02d}_{segment_number:03d}"

            signals = {
                "ecg": ecg[global_start:global_end],
                "resp": resp[global_start:global_end],
                "eda": eda[global_start:global_end],
                "emg": emg[global_start:global_end],
                "temp": temp[global_start:global_end],
            }
            sampling_rates = dict(WESAD_SAMPLING_RATES)
            metadata = {
                "label_code": label_code,
                "segment_origin": "wrist_chest_preprocessed",
            }

            segment_records.append(
                build_segment_record(
                    subject_id=subject_id,
                    segment_id=segment_id,
                    dataset="wesad",
                    label=label_name,
                    signals=signals,
                    sampling_rates=sampling_rates,
                    self_report={},
                    source_record_id=f"{subject_id}_run_{run_index:02d}",
                    window_start_sec=(start_index / sampling_rate) + start_sec,
                    window_end_sec=(start_index / sampling_rate) + end_sec,
                    metadata=metadata,
                )
            )

        return segment_records

