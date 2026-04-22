"""Загрузчик открытых OpenNeuro BCMI-датасетов ds002722 и ds002724."""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

from config.settings import BCMI_EEG_CHANNELS, BCMI_TARGET_LABELS, DATASET_DEFAULTS
from features.common import build_segment_record, derive_state_label, safe_float

LOGGER = logging.getLogger(__name__)


class OpenNeuroBCMILoader:
    """Загружает trial-сегменты из открытых BCMI-датасетов OpenNeuro."""

    def __init__(self, root: Path, dataset_name: str, progress_callback: Callable[[dict[str, Any]], None] | None = None) -> None:
        self.root = Path(root)
        self.dataset_name = dataset_name.lower()
        self.progress_callback = progress_callback
        if self.dataset_name not in {"ds002722", "ds002724"}:
            raise ValueError(f"Неподдерживаемый BCMI-датасет: {dataset_name}")
        self.window_seconds = float(DATASET_DEFAULTS[self.dataset_name]["window_seconds"])
        self.step_seconds = float(DATASET_DEFAULTS[self.dataset_name]["step_seconds"])

    def load(self) -> pd.DataFrame:
        """Читает все записи датасета и возвращает унифицированную таблицу сегментов."""

        data_root = self._resolve_root()
        edf_files = self._find_recordings(data_root)

        try:
            import mne  # type: ignore
        except ImportError as error:
            raise ImportError(
                "Для чтения ds002722/ds002724 требуется пакет mne. "
                "Установите его командой: python -m pip install mne"
            ) from error

        records: list[dict[str, Any]] = []
        total_files = len(edf_files)
        for file_index, edf_path in enumerate(edf_files, start=1):
            if self.progress_callback is not None:
                self.progress_callback(
                    {
                        "stage": "loading_files",
                        "current": file_index,
                        "total": total_files,
                        "message": f"Загрузка EDF {file_index}/{total_files}: {edf_path.name}",
                        "file_name": edf_path.name,
                    }
                )
            try:
                records.extend(self._load_recording(mne=mne, edf_path=edf_path))
            except Exception:
                LOGGER.exception("Не удалось обработать BCMI-запись %s", edf_path)

        return pd.DataFrame.from_records(records)

    def _resolve_root(self) -> Path:
        """Определяет корень датасета по dataset_description.json."""

        if (self.root / "dataset_description.json").exists():
            return self.root

        matches = sorted(self.root.rglob("dataset_description.json"))
        if matches:
            return matches[0].parent

        raise FileNotFoundError(
            f"В {self.root} не найден dataset_description.json для {self.dataset_name}."
        )

    def _find_recordings(self, data_root: Path) -> list[Path]:
        """Ищет EDF-файлы с EEG-записями."""

        edf_files = sorted(data_root.rglob("*_eeg.edf"))
        if not edf_files:
            raise FileNotFoundError(f"В {data_root} не найдено ни одного файла *_eeg.edf.")
        return edf_files

    def _load_recording(self, mne: Any, edf_path: Path) -> list[dict[str, Any]]:
        """Извлекает trial-сегменты из одной EDF-записи."""

        events_path = edf_path.with_name(edf_path.name.replace("_eeg.edf", "_events.tsv"))
        if not events_path.exists():
            LOGGER.warning("Для файла %s отсутствует events.tsv", edf_path)
            return []

        events = pd.read_csv(events_path, sep="\t")
        trial_events = self._select_target_events(events)
        if trial_events.empty:
            LOGGER.warning("В %s нет событий trial_type 1..9", events_path)
            return []

        raw = mne.io.read_raw_edf(edf_path, preload=False, verbose="ERROR")
        sampling_rate = float(raw.info["sfreq"])
        n_times = int(raw.n_times)
        available_channels = set(raw.ch_names)

        eeg_channels = [name for name in BCMI_EEG_CHANNELS if name in available_channels]
        ecg_channel = "ECG" if "ECG" in available_channels else ""
        gsr_channel = "GSR" if "GSR" in available_channels else ""
        auxiliary_channels = [name for name in ("VA1", "VA2", "VAtarg") if name in available_channels]

        aux_data = (
            raw.get_data(picks=auxiliary_channels).astype(np.float32, copy=False)
            if auxiliary_channels
            else np.empty((0, 0), dtype=np.float32)
        )
        if hasattr(raw, "close"):
            raw.close()

        aux_lookup = {name: index for index, name in enumerate(auxiliary_channels)}
        subject_id, session_id, run_id = self._parse_identifiers(edf_path)

        records: list[dict[str, Any]] = []
        for trial_index, row in enumerate(trial_events.to_dict(orient="records")):
            onset = safe_float(row.get("onset"))
            duration = safe_float(row.get("duration"))
            target_code = int(safe_float(row.get("trial_type")))
            if not np.isfinite(onset) or not np.isfinite(duration) or duration <= 0:
                continue

            start_sample = max(int(round(onset * sampling_rate)), 0)
            end_sample = min(int(round((onset + duration) * sampling_rate)), n_times)
            if end_sample - start_sample < 256:
                continue

            valence = self._segment_mean(aux_data, aux_lookup, "VA1", start_sample, end_sample, scale=1e6)
            arousal = self._segment_mean(aux_data, aux_lookup, "VA2", start_sample, end_sample, scale=1e6)
            vatarg = self._segment_mean(aux_data, aux_lookup, "VAtarg", start_sample, end_sample, scale=1e6)

            fallback_valence, fallback_arousal = self._target_centroid(target_code)
            if not np.isfinite(valence):
                valence = fallback_valence
            if not np.isfinite(arousal):
                arousal = fallback_arousal

            label = derive_state_label(self.dataset_name, valence=valence, arousal=arousal)
            target_name = BCMI_TARGET_LABELS.get(target_code, "")

            session_token = session_id or "ses-0"
            run_token = run_id or "run-0"
            segment_id = f"{subject_id}_{session_token}_{run_token}_{trial_index:03d}"

            records.append(
                build_segment_record(
                    subject_id=subject_id,
                    segment_id=segment_id,
                    dataset=self.dataset_name,
                    label=label,
                    signals={
                        "__lazy_loader__": "edf_slice",
                        "recording_path": str(edf_path),
                        "start_sample": start_sample,
                        "end_sample": end_sample,
                        "eeg_channels": eeg_channels,
                        "ecg_channel": ecg_channel,
                        "gsr_channel": gsr_channel,
                    },
                    sampling_rates={
                        "eeg": sampling_rate,
                        "ecg": sampling_rate if ecg_channel else np.nan,
                        "gsr": sampling_rate if gsr_channel else np.nan,
                    },
                    self_report={
                        "valence": valence,
                        "arousal": arousal,
                    },
                    source_record_id=edf_path.stem,
                    window_start_sec=onset,
                    window_end_sec=onset + duration,
                    metadata={
                        "session_id": session_id,
                        "run_id": run_id,
                        "target_code": target_code,
                        "target_label": target_name,
                        "vatarg_mean": vatarg,
                        "segment_origin": "openneuro_bcmi_trial",
                        "source_file": edf_path.name,
                    },
                )
            )

        LOGGER.info("%s: файл %s, сегментов: %d", self.dataset_name, edf_path.name, len(records))
        return records

    def _select_target_events(self, events: pd.DataFrame) -> pd.DataFrame:
        """Оставляет только события начала trial с кодами 1..9."""

        if "trial_type" not in events.columns:
            return pd.DataFrame()

        table = events.copy()
        table["trial_type"] = pd.to_numeric(table["trial_type"], errors="coerce")
        table["onset"] = pd.to_numeric(table.get("onset"), errors="coerce")
        table["duration"] = pd.to_numeric(table.get("duration"), errors="coerce")
        table = table.loc[
            table["trial_type"].between(1, 9, inclusive="both")
            & table["onset"].notna()
            & table["duration"].notna()
        ]
        return table.sort_values(["onset", "trial_type"]).reset_index(drop=True)

    def _parse_identifiers(self, edf_path: Path) -> tuple[str, str, str]:
        """Извлекает идентификаторы субъекта, сессии и run из пути BIDS."""

        subject_id = next((part for part in edf_path.parts if part.startswith("sub-")), edf_path.stem)
        session_id = next((part for part in edf_path.parts if part.startswith("ses-")), "")
        run_match = re.search(r"task-(run\d+)", edf_path.stem)
        run_id = run_match.group(1) if run_match else ""
        return subject_id, session_id, run_id

    def _segment_mean(
        self,
        aux_data: np.ndarray,
        aux_lookup: dict[str, int],
        channel_name: str,
        start_sample: int,
        end_sample: int,
        scale: float = 1.0,
    ) -> float:
        """Возвращает среднее значение auxiliary-канала на trial-отрезке."""

        if channel_name not in aux_lookup or aux_data.size == 0:
            return np.nan
        values = aux_data[aux_lookup[channel_name], start_sample:end_sample]
        if values.size == 0:
            return np.nan
        return float(np.mean(values) * scale)

    def _target_centroid(self, target_code: int) -> tuple[float, float]:
        """Возвращает центры valence/arousal для target-кода, если self-report недоступен."""

        mapping = {
            1: (1.5, 1.5),
            2: (4.5, 1.5),
            3: (7.5, 1.5),
            4: (1.5, 4.5),
            5: (4.5, 4.5),
            6: (7.5, 4.5),
            7: (1.5, 7.5),
            8: (4.5, 7.5),
            9: (7.5, 7.5),
        }
        return mapping.get(target_code, (np.nan, np.nan))
