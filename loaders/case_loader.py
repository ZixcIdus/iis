"""Загрузчик датасета CASE."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from config.settings import CASE_SAMPLING_RATES, CASE_VIDEO_LABEL_MAP, DATASET_DEFAULTS
from features.common import build_segment_record

LOGGER = logging.getLogger(__name__)


class CASELoader:
    """Загружает CASE из raw-логов и восстанавливает сегменты по метаданным."""

    def __init__(self, root: Path, window_seconds: float | None = None, step_seconds: float | None = None) -> None:
        self.root = Path(root)
        self.window_seconds = window_seconds or DATASET_DEFAULTS["case"]["window_seconds"]
        self.step_seconds = step_seconds or DATASET_DEFAULTS["case"]["step_seconds"]
        self.sampling_rate = float(CASE_SAMPLING_RATES["ecg"])

    def load(self) -> pd.DataFrame:
        """Читает доступные raw-файлы CASE и возвращает таблицу сегментов."""

        case_root = self._resolve_case_root()
        sequences = self._load_sequences(case_root / "metadata" / "seqs_order.txt")
        durations = self._load_durations(case_root / "metadata" / "videos_duration.txt")

        records: list[dict[str, Any]] = []
        for subject_number in sorted(sequences):
            try:
                records.extend(
                    self._load_subject(
                        case_root=case_root,
                        subject_number=subject_number,
                        video_sequence=sequences[subject_number],
                        durations_ms=durations,
                    )
                )
            except Exception:
                LOGGER.exception("Не удалось обработать CASE для субъекта %02d", subject_number)

        return pd.DataFrame.from_records(records)

    def _resolve_case_root(self) -> Path:
        """Находит корень CASE с папками data/metadata/scripts."""

        if (self.root / "metadata" / "seqs_order.txt").exists():
            return self.root

        matches = sorted(self.root.rglob("seqs_order.txt"))
        if matches:
            return matches[0].parent.parent

        raise FileNotFoundError(
            f"Файлы CASE не найдены в {self.root}. Ожидается структура вроде data/CASE/case_dataset-ver_SciData_0/."
        )

    def _load_sequences(self, path: Path) -> dict[int, list[str]]:
        """Читает порядок предъявления видео для всех участников."""

        table = pd.read_csv(path, sep=r"\t+", dtype=str, engine="python")
        table.columns = [self._normalize_text(column) for column in table.columns]

        sequences: dict[int, list[str]] = {}
        for subject_number in range(1, 31):
            column_name = f"seq_sub{subject_number}"
            if column_name not in table.columns:
                continue
            values = [
                self._normalize_text(value)
                for value in table[column_name].dropna().tolist()
                if self._normalize_text(value)
            ]
            if values:
                sequences[subject_number] = values
        return sequences

    def _load_durations(self, path: Path) -> dict[str, float]:
        """Читает длительности видео в миллисекундах."""

        table = pd.read_csv(path, sep=r"\t+", dtype=str, engine="python")
        table.columns = [self._normalize_text(column) for column in table.columns]
        if "video_name" not in table.columns or "video_duration" not in table.columns:
            raise KeyError(f"В файле {path} ожидаются столбцы video_name и video_duration.")

        durations: dict[str, float] = {}
        for row in table.to_dict(orient="records"):
            name = self._normalize_text(row.get("video_name"))
            if not name:
                continue
            durations[name] = float(row.get("video_duration"))
        return durations

    def _load_subject(
        self,
        case_root: Path,
        subject_number: int,
        video_sequence: list[str],
        durations_ms: dict[str, float],
    ) -> list[dict[str, Any]]:
        """Загружает raw-логи одного участника CASE."""

        phys_path = case_root / "data" / "raw" / "physiological" / f"sub{subject_number}_DAQ.txt"
        ann_path = case_root / "data" / "raw" / "annotations" / f"sub{subject_number}_joystick.txt"
        if not phys_path.exists() or not ann_path.exists():
            LOGGER.warning("CASE: отсутствуют raw-файлы для субъекта %02d", subject_number)
            return []

        daq_data = np.loadtxt(phys_path, delimiter="\t", dtype=float)
        annotation_data = np.loadtxt(ann_path, delimiter="\t", dtype=float)
        if daq_data.ndim != 2 or daq_data.shape[1] < 9:
            LOGGER.warning("CASE: неожиданный формат физиологии в %s", phys_path)
            return []
        if annotation_data.ndim != 2 or annotation_data.shape[1] < 3:
            LOGGER.warning("CASE: неожиданный формат аннотаций в %s", ann_path)
            return []

        daq_time_ms, ecg, bvp, gsr, resp, temperature = self._transform_physiology(daq_data)
        ann_time_ms, valence, arousal = self._transform_annotations(annotation_data)

        records: list[dict[str, Any]] = []
        current_start_ms = 0.0
        subject_id = f"S{subject_number:02d}"

        for video_index, video_name in enumerate(video_sequence):
            duration_ms = durations_ms.get(video_name)
            if duration_ms is None:
                LOGGER.warning("CASE: для видео %s нет длительности, субъект %s", video_name, subject_id)
                continue
            video_end_ms = current_start_ms + duration_ms
            state_label = CASE_VIDEO_LABEL_MAP.get(video_name, "disbalance")

            records.extend(
                self._segment_video(
                    subject_id=subject_id,
                    video_name=video_name,
                    state_label=state_label,
                    video_index=video_index,
                    video_start_ms=current_start_ms,
                    video_end_ms=video_end_ms,
                    daq_time_ms=daq_time_ms,
                    ecg=ecg,
                    bvp=bvp,
                    gsr=gsr,
                    resp=resp,
                    temperature=temperature,
                    ann_time_ms=ann_time_ms,
                    valence=valence,
                    arousal=arousal,
                )
            )
            current_start_ms = video_end_ms

        LOGGER.info("CASE: субъект %s, сегментов: %d", subject_id, len(records))
        return records

    def _transform_physiology(
        self,
        daq_data: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Переводит raw-DAQ из вольт в физические единицы по официальным формулам."""

        time_ms = daq_data[:, 0] * 1000.0
        ecg = ((daq_data[:, 1] - 2.8) / 50.0) * 1000.0
        bvp = (58.962 * daq_data[:, 2]) - 115.09
        gsr = (24.0 * daq_data[:, 3]) - 49.2
        resp = (58.923 * daq_data[:, 4]) - 115.01
        temperature = (21.341 * daq_data[:, 5]) - 32.085
        return time_ms, ecg, bvp, gsr, resp, temperature

    def _transform_annotations(self, annotation_data: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Переводит joystick-координаты в шкалу 0.5..9.5."""

        time_ms = annotation_data[:, 0] * 1000.0
        valence = 0.5 + 9.0 * (annotation_data[:, 1] + 26225.0) / 52450.0
        arousal = 0.5 + 9.0 * (annotation_data[:, 2] + 26225.0) / 52450.0
        return time_ms, valence, arousal

    def _segment_video(
        self,
        subject_id: str,
        video_name: str,
        state_label: str,
        video_index: int,
        video_start_ms: float,
        video_end_ms: float,
        daq_time_ms: np.ndarray,
        ecg: np.ndarray,
        bvp: np.ndarray,
        gsr: np.ndarray,
        resp: np.ndarray,
        temperature: np.ndarray,
        ann_time_ms: np.ndarray,
        valence: np.ndarray,
        arousal: np.ndarray,
    ) -> list[dict[str, Any]]:
        """Разбивает одно видео на окна фиксированной длины."""

        window_ms = float(self.window_seconds * 1000.0)
        step_ms = float(self.step_seconds * 1000.0)
        if window_ms <= 0 or step_ms <= 0 or video_end_ms - video_start_ms < window_ms:
            return []

        records: list[dict[str, Any]] = []
        segment_index = 0
        current_start_ms = video_start_ms
        safe_video_name = video_name.replace("-", "_")

        while current_start_ms + window_ms <= video_end_ms + 1e-6:
            current_end_ms = current_start_ms + window_ms

            daq_start = int(np.searchsorted(daq_time_ms, current_start_ms, side="left"))
            daq_end = int(np.searchsorted(daq_time_ms, current_end_ms, side="right"))
            ann_start = int(np.searchsorted(ann_time_ms, current_start_ms, side="left"))
            ann_end = int(np.searchsorted(ann_time_ms, current_end_ms, side="right"))

            if daq_end - daq_start < 256:
                current_start_ms += step_ms
                segment_index += 1
                continue

            segment_valence = self._mean_or_nan(valence[ann_start:ann_end])
            segment_arousal = self._mean_or_nan(arousal[ann_start:ann_end])

            signals = {
                "ecg": ecg[daq_start:daq_end],
                "ppg": bvp[daq_start:daq_end],
                "resp": resp[daq_start:daq_end],
                "gsr": gsr[daq_start:daq_end],
                "temperature": temperature[daq_start:daq_end],
            }
            sampling_rates = {
                "ecg": CASE_SAMPLING_RATES["ecg"],
                "ppg": CASE_SAMPLING_RATES["ppg"],
                "resp": CASE_SAMPLING_RATES["resp"],
                "gsr": CASE_SAMPLING_RATES["gsr"],
                "temperature": CASE_SAMPLING_RATES["temperature"],
            }
            metadata = {
                "video_name": video_name,
                "video_index": video_index,
                "segment_origin": "case_raw_reconstructed",
            }

            records.append(
                build_segment_record(
                    subject_id=subject_id,
                    segment_id=f"{subject_id}_{safe_video_name}_{video_index:02d}_{segment_index:03d}",
                    dataset="case",
                    label=state_label,
                    signals=signals,
                    sampling_rates=sampling_rates,
                    self_report={
                        "valence": segment_valence,
                        "arousal": segment_arousal,
                    },
                    source_record_id=f"{subject_id}_{safe_video_name}_{video_index:02d}",
                    window_start_sec=current_start_ms / 1000.0,
                    window_end_sec=current_end_ms / 1000.0,
                    metadata=metadata,
                )
            )

            current_start_ms += step_ms
            segment_index += 1

        return records

    def _normalize_text(self, value: Any) -> str:
        """Нормализует текстовое значение из metadata-файлов."""

        return str(value).strip().strip('"').strip().lower()

    def _mean_or_nan(self, values: np.ndarray) -> float:
        """Возвращает среднее по сегменту или NaN при отсутствии данных."""

        if values.size == 0:
            return float("nan")
        return float(np.mean(values))
