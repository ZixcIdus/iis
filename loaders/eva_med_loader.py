"""Загрузчик датасета EVA-MED."""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from config.settings import DATASET_DEFAULTS
from features.common import build_segment_record, derive_state_label, safe_float

LOGGER = logging.getLogger(__name__)


class EVAMEDLoader:
    """Загружает EVA-MED из предобработанных epoch-файлов."""

    def __init__(self, root: Path, window_seconds: float | None = None, step_seconds: float | None = None) -> None:
        self.root = Path(root)
        self.window_seconds = window_seconds or DATASET_DEFAULTS["eva_med"]["window_seconds"]
        self.step_seconds = step_seconds or DATASET_DEFAULTS["eva_med"]["step_seconds"]

    def load(self) -> pd.DataFrame:
        """Читает предобработанные EEG/ECG epoch-файлы EVA-MED."""

        eva_root = self._resolve_root()
        epoch_files = self._find_epoch_files(eva_root)
        if not epoch_files:
            raise FileNotFoundError(
                f"В {eva_root} не найдены файлы EVA-MED формата .fif. Ожидаются папки ECG_EEG_emo или EEG_preprocessed."
            )

        try:
            import mne  # type: ignore
        except ImportError as error:
            raise ImportError(
                "Для чтения EVA-MED требуется пакет mne. Установите его командой: python -m pip install mne"
            ) from error

        sam_lookup = self._load_sam_lookup(eva_root)
        records: list[dict[str, Any]] = []
        for epoch_path in epoch_files:
            try:
                records.extend(self._load_epoch_file(mne=mne, epoch_path=epoch_path, eva_root=eva_root, sam_lookup=sam_lookup))
            except Exception:
                LOGGER.exception("Не удалось обработать EVA-MED файл %s", epoch_path)

        return pd.DataFrame.from_records(records)

    def _resolve_root(self) -> Path:
        """Находит корень EVA-MED."""

        if any((self.root / folder).exists() for folder in ("ECG_EEG_emo", "EEG_preprocessed", "Physiological Data")):
            return self.root

        matches = sorted(self.root.rglob("SAM.xlsx"))
        if matches:
            return matches[0].parent

        fif_matches = sorted(self.root.rglob("*.fif"))
        if fif_matches:
            return fif_matches[0].parent.parent

        return self.root

    def _find_epoch_files(self, eva_root: Path) -> list[Path]:
        """Ищет наиболее полезные epoch-файлы EVA-MED."""

        candidates = sorted((eva_root / "ECG_EEG_emo").rglob("*.fif")) if (eva_root / "ECG_EEG_emo").exists() else []
        if candidates:
            return candidates

        candidates = sorted((eva_root / "EEG_preprocessed").rglob("*.fif")) if (eva_root / "EEG_preprocessed").exists() else []
        if candidates:
            return candidates

        return sorted(eva_root.rglob("*.fif"))

    def _load_sam_lookup(self, eva_root: Path) -> dict[str, dict[str, float]]:
        """Пытается загрузить self-report SAM для дальнейшего сопоставления."""

        sam_candidates = sorted(eva_root.rglob("SAM.xlsx")) + sorted(eva_root.rglob("SAM.csv"))
        if not sam_candidates:
            return {}

        sam_path = sam_candidates[0]
        try:
            if sam_path.suffix.lower() == ".csv":
                table = pd.read_csv(sam_path)
            else:
                table = pd.read_excel(sam_path)
        except Exception:
            LOGGER.warning("Не удалось прочитать SAM-файл EVA-MED: %s", sam_path)
            return {}

        normalized_columns = {column: self._normalize_text(column) for column in table.columns}
        table = table.rename(columns=normalized_columns)

        subject_column = self._find_first_column(table, ("subject", "participant", "name", "id"))
        valence_column = self._find_first_column(table, ("valence",))
        arousal_column = self._find_first_column(table, ("arousal",))
        phase_column = self._find_first_column(table, ("phase", "condition", "label", "emotion"))
        if subject_column is None or valence_column is None or arousal_column is None:
            return {}

        lookup: dict[str, dict[str, float]] = {}
        for row in table.to_dict(orient="records"):
            subject_key = self._normalize_text(row.get(subject_column))
            phase_key = self._normalize_text(row.get(phase_column)) if phase_column else ""
            lookup_key = f"{subject_key}|{phase_key}"
            lookup[lookup_key] = {
                "valence": safe_float(row.get(valence_column)),
                "arousal": safe_float(row.get(arousal_column)),
            }
        return lookup

    def _load_epoch_file(
        self,
        mne: Any,
        epoch_path: Path,
        eva_root: Path,
        sam_lookup: dict[str, dict[str, float]],
    ) -> list[dict[str, Any]]:
        """Читает один epoch-файл EVA-MED."""

        epochs = mne.read_epochs(epoch_path, preload=True, verbose="ERROR")
        subject_id = self._infer_subject_id(epoch_path)
        epoch_duration = self._infer_epoch_duration(epochs=epochs)
        eeg_picks = mne.pick_types(epochs.info, eeg=True, exclude=[])
        ecg_picks = mne.pick_types(epochs.info, ecg=True, exclude=[])

        all_data = epochs.get_data(copy=False)
        if all_data.ndim != 3 or all_data.shape[0] == 0:
            return []

        eeg_channel_names = [epochs.ch_names[index] for index in eeg_picks]
        ecg_fallback = self._load_ecg_fallback(eva_root=eva_root, epoch_path=epoch_path, subject_id=subject_id)
        labels = self._extract_epoch_labels(epochs=epochs, epoch_path=epoch_path)
        phase_names = self._extract_phase_names(epochs=epochs, epoch_path=epoch_path)

        records: list[dict[str, Any]] = []
        for epoch_index in range(all_data.shape[0]):
            eeg_signal = all_data[epoch_index, eeg_picks, :] if len(eeg_picks) else np.empty((0, 0), dtype=float)
            ecg_signal = self._extract_ecg_epoch(
                all_data=all_data,
                epoch_index=epoch_index,
                ecg_picks=ecg_picks,
                ecg_fallback=ecg_fallback,
            )

            phase_name = phase_names[epoch_index] if epoch_index < len(phase_names) else ""
            sam_values = sam_lookup.get(f"{subject_id.lower()}|{phase_name.lower()}", {})
            valence = safe_float(sam_values.get("valence"))
            arousal = safe_float(sam_values.get("arousal"))

            label = labels[epoch_index] if epoch_index < len(labels) else ""
            if not label:
                label = derive_state_label("eva_med", valence=valence, arousal=arousal)

            records.append(
                build_segment_record(
                    subject_id=subject_id,
                    segment_id=f"{subject_id}_{epoch_path.stem}_{epoch_index:04d}",
                    dataset="eva_med",
                    label=label,
                    signals={
                        "eeg": eeg_signal,
                        "eeg_channels": eeg_channel_names,
                        "ecg": ecg_signal,
                    },
                    sampling_rates={
                        "eeg": float(epochs.info["sfreq"]),
                        "ecg": float(len(ecg_signal) / max(epoch_duration, 1e-6)) if ecg_signal.size else float(epochs.info["sfreq"]),
                    },
                    self_report={
                        "valence": valence,
                        "arousal": arousal,
                    },
                    source_record_id=f"{subject_id}_{epoch_path.stem}",
                    window_start_sec=epoch_index * epoch_duration,
                    window_end_sec=(epoch_index + 1) * epoch_duration,
                    metadata={
                        "phase": phase_name,
                        "segment_origin": "eva_med_epochs",
                        "source_file": str(epoch_path.name),
                    },
                )
            )

        LOGGER.info("EVA-MED: файл %s, сегментов: %d", epoch_path.name, len(records))
        return records

    def _load_ecg_fallback(self, eva_root: Path, epoch_path: Path, subject_id: str) -> np.ndarray | None:
        """Пытается найти отдельный NPY-файл с ECG, если его нет в .fif."""

        ecg_root = eva_root / "ECG_preprocessed"
        if not ecg_root.exists():
            return None

        subject_token = self._normalize_text(subject_id)
        stem_token = self._normalize_text(epoch_path.stem)
        candidates = sorted(
            path
            for path in ecg_root.rglob("*.npy")
            if subject_token in self._normalize_text(path.stem) or stem_token in self._normalize_text(path.stem)
        )
        if not candidates:
            return None

        try:
            array = np.load(candidates[0], allow_pickle=True)
        except Exception:
            LOGGER.warning("Не удалось прочитать ECG NPY для EVA-MED: %s", candidates[0])
            return None

        array = np.asarray(array, dtype=float)
        if array.ndim == 3:
            return np.mean(array, axis=1)
        if array.ndim == 2:
            return array
        return None

    def _extract_ecg_epoch(
        self,
        all_data: np.ndarray,
        epoch_index: int,
        ecg_picks: np.ndarray,
        ecg_fallback: np.ndarray | None,
    ) -> np.ndarray:
        """Возвращает одномерный ECG-сигнал для одной эпохи."""

        if len(ecg_picks):
            return np.mean(all_data[epoch_index, ecg_picks, :], axis=0).astype(float, copy=False)
        if ecg_fallback is not None and ecg_fallback.ndim == 2 and epoch_index < ecg_fallback.shape[0]:
            return np.asarray(ecg_fallback[epoch_index], dtype=float)
        return np.asarray([], dtype=float)

    def _extract_epoch_labels(self, epochs: Any, epoch_path: Path) -> list[str]:
        """Пытается получить трёхсостоянийные labels для эпох."""

        metadata = getattr(epochs, "metadata", None)
        if metadata is not None and not metadata.empty:
            label_column = self._find_first_column(metadata, ("label", "emotion", "condition", "phase", "state"))
            valence_column = self._find_first_column(metadata, ("valence",))
            arousal_column = self._find_first_column(metadata, ("arousal",))

            if label_column is not None:
                return [self._state_label_from_text(value) for value in metadata[label_column].tolist()]
            if valence_column is not None and arousal_column is not None:
                return [
                    derive_state_label("eva_med", valence=safe_float(valence), arousal=safe_float(arousal))
                    for valence, arousal in zip(metadata[valence_column].tolist(), metadata[arousal_column].tolist(), strict=False)
                ]

        if getattr(epochs, "event_id", None):
            reverse_event_id = {code: name for name, code in epochs.event_id.items()}
            return [self._state_label_from_text(reverse_event_id.get(int(event_code), "")) for event_code in epochs.events[:, 2]]

        return [self._state_label_from_text(epoch_path.stem)] * len(epochs)

    def _extract_phase_names(self, epochs: Any, epoch_path: Path) -> list[str]:
        """Пытается получить названия фаз для связи с SAM."""

        metadata = getattr(epochs, "metadata", None)
        if metadata is not None and not metadata.empty:
            phase_column = self._find_first_column(metadata, ("phase", "condition", "label", "emotion"))
            if phase_column is not None:
                return [self._normalize_text(value) for value in metadata[phase_column].tolist()]

        if getattr(epochs, "event_id", None):
            reverse_event_id = {code: name for name, code in epochs.event_id.items()}
            return [self._normalize_text(reverse_event_id.get(int(event_code), "")) for event_code in epochs.events[:, 2]]

        return [self._normalize_text(epoch_path.stem)] * len(epochs)

    def _infer_subject_id(self, path: Path) -> str:
        """Извлекает идентификатор субъекта из имени файла."""

        normalized = self._normalize_text(path.stem)
        match = re.search(r"(?:sub|subject|s)[_-]?(\d{1,3})", normalized)
        if match:
            return f"S{int(match.group(1)):02d}"
        return path.stem

    def _infer_epoch_duration(self, epochs: Any) -> float:
        """Оценивает длительность одной эпохи в секундах."""

        if hasattr(epochs, "times") and len(epochs.times) > 1:
            return float(epochs.times[-1] - epochs.times[0] + (1.0 / epochs.info["sfreq"]))
        return float(self.window_seconds)

    def _find_first_column(self, table: pd.DataFrame, candidates: tuple[str, ...]) -> str | None:
        """Находит первый столбец, содержащий один из кандидатов."""

        normalized_columns = {column: self._normalize_text(column) for column in table.columns}
        for candidate in candidates:
            for column, normalized in normalized_columns.items():
                if candidate in normalized:
                    return column
        return None

    def _state_label_from_text(self, value: Any) -> str:
        """Переводит текстовую фазу EVA-MED в три состояния проекта."""

        text = self._normalize_text(value)
        if any(token in text for token in ("stress", "mmst", "arousal_high")):
            return "stress"
        if any(token in text for token in ("calm", "baseline", "rest")):
            return "baseline"
        if any(token in text for token in ("positive", "negative", "neutral", "emotion")):
            return "disbalance"
        return "disbalance"

    def _normalize_text(self, value: Any) -> str:
        """Нормализует строку для поиска по именам файлов и колонок."""

        return str(value).strip().strip('"').replace(" ", "_").lower()
