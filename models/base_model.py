"""Базовый интерфейс и общая логика вычисления ИИС."""

from __future__ import annotations

import copy
import logging
from abc import ABC, abstractmethod
from typing import Any

import numpy as np
import pandas as pd

from config.settings import (
    EEG_POWER_UV2_SCALE,
    MODEL_COMPONENTS,
    MODEL_CONFIGS,
    PROXY_RANGES,
    SELF_REPORT_RANGES,
    SENSITIVITY_COMPONENTS,
    SENSITIVITY_DELTA,
)
from features.common import safe_json_dumps, safe_float

LOGGER = logging.getLogger(__name__)

ComponentResult = dict[str, Any]


class BaseIISModel(ABC):
    """Базовый класс для всех версий модели ИИС."""

    def __init__(self) -> None:
        self.name = self.__class__.__name__
        if self.name not in MODEL_CONFIGS:
            raise KeyError(f"Для модели {self.name} нет конфигурации в settings.py.")
        self.config = MODEL_CONFIGS[self.name]
        self.weights = dict(self.config["weights"])
        self.required_components = tuple(self.weights.keys())

    def prepare_inputs(self, feature_row: pd.Series | dict[str, Any]) -> dict[str, Any]:
        """Нормализует строку признаков для дальнейших расчётов."""

        if isinstance(feature_row, pd.Series):
            return feature_row.to_dict()
        return dict(feature_row)

    def compute_components(self, inputs: dict[str, Any], mode: str) -> dict[str, ComponentResult]:
        """Последовательно считает все компоненты модели."""

        components: dict[str, ComponentResult] = {}
        components["A"] = self.calculate_A(inputs, mode)
        components["Gamma"] = self.calculate_Gamma(inputs, mode)
        components["H"] = self.calculate_H(inputs, mode)
        components["V"] = self.calculate_V(inputs, mode)
        components["Q"] = self.calculate_Q(inputs, mode)
        components["K"] = self.calculate_K(inputs, mode, components)
        return components

    def compute_score(self, components: dict[str, ComponentResult], mode: str) -> dict[str, Any]:
        """Считает итоговый индекс ИИС."""

        return self.calculate_IIS(components, mode)

    def explain_score(self, score_payload: dict[str, Any]) -> dict[str, Any]:
        """Строит краткое текстовое объяснение рассчитанного индекса."""

        contributions = score_payload.get("contributions", {}) or {}
        ordered = sorted(contributions.items(), key=lambda item: abs(item[1]), reverse=True)
        top_components = [name for name, _ in ordered[:3]]
        return {
            "active_components": score_payload.get("active_components", []),
            "missing_components": score_payload.get("missing_components", []),
            "top_contributors": top_components,
            "formula_note": score_payload.get("formula_note", ""),
            "insufficient_reason": score_payload.get("insufficient_reason", ""),
        }

    def evaluate_dataframe(
        self,
        features_df: pd.DataFrame,
        mode: str,
        progress_callback: Any | None = None,
        preview_every: int = 120,
    ) -> pd.DataFrame:
        """Вычисляет модель для всей таблицы признаков."""

        rows = features_df.to_dict(orient="records")
        total_rows = len(rows)
        records: list[dict[str, Any]] = []
        preview_step = max(int(preview_every), 1)
        preview_size = max(min(preview_step, 240), 40)

        for row_index, row in enumerate(rows, start=1):
            records.append(self.evaluate_row(row=row, mode=mode))
            if progress_callback is not None and (
                row_index == 1 or row_index % preview_step == 0 or row_index == total_rows
            ):
                progress_callback(
                    {
                        "stage": "model_rows",
                        "model_name": self.name,
                        "current": row_index,
                        "total": total_rows,
                        "message": f"{self.name}: расчёт {row_index}/{total_rows}",
                        "preview_rows": records[-preview_size:],
                    }
                )
        return pd.DataFrame.from_records(records)

    def evaluate_row(self, row: dict[str, Any], mode: str) -> dict[str, Any]:
        """Вычисляет все компоненты и итоговый индекс для одного сегмента."""

        inputs = self.prepare_inputs(row)
        components = self.compute_components(inputs, mode)
        score_payload = self.compute_score(components, mode)
        explanation = self.explain_score(score_payload)
        sensitivities = self.compute_sensitivity(components=components, mode=mode, base_score=score_payload.get("IIS"))

        result: dict[str, Any] = {
            "subject_id": inputs.get("subject_id"),
            "segment_id": inputs.get("segment_id"),
            "dataset": inputs.get("dataset"),
            "label": inputs.get("label"),
            "source_record_id": inputs.get("source_record_id"),
            "window_start_sec": safe_float(inputs.get("window_start_sec")),
            "window_end_sec": safe_float(inputs.get("window_end_sec")),
            "segment_duration_sec": safe_float(inputs.get("segment_duration_sec")),
            "version": self.name,
            "mode": mode,
            "IIS": safe_float(score_payload.get("IIS")),
            "raw_score": safe_float(score_payload.get("raw_score")),
            "coverage_ratio": safe_float(score_payload.get("coverage_ratio")),
            "active_component_count": int(score_payload.get("active_component_count", 0)),
            "active_components_json": safe_json_dumps(score_payload.get("active_components", [])),
            "missing_components_json": safe_json_dumps(score_payload.get("missing_components", [])),
            "component_provenance_json": safe_json_dumps(score_payload.get("active_component_provenance", {})),
            "contributions_json": safe_json_dumps(score_payload.get("contributions", {})),
            "formula_note": score_payload.get("formula_note", ""),
            "insufficient_reason": score_payload.get("insufficient_reason", ""),
            "score_explanation_json": safe_json_dumps(explanation),
            "stress_label": safe_float(inputs.get("stress_label")),
            "valence": safe_float(inputs.get("valence")),
            "arousal": safe_float(inputs.get("arousal")),
            "dominance": safe_float(inputs.get("dominance")),
            "liking": safe_float(inputs.get("liking")),
        }

        for component_name in MODEL_COMPONENTS:
            component = components.get(component_name, self._component_result(np.nan, "unavailable", "", ""))
            result[component_name] = safe_float(component.get("value"))
            result[f"prov_{component_name}"] = component.get("provenance", "unavailable")
            result[f"source_{component_name}"] = component.get("source", "")
            result[f"note_{component_name}"] = component.get("note", "")
            result[f"contrib_{component_name}"] = safe_float(score_payload.get("contributions", {}).get(component_name))

        result.update(sensitivities)
        return result

    def compute_sensitivity(
        self,
        components: dict[str, ComponentResult],
        mode: str,
        base_score: float | None,
    ) -> dict[str, float]:
        """Считает чувствительность ИИС к малому изменению компонентов."""

        sensitivity_payload = {f"sens_{name}": np.nan for name in SENSITIVITY_COMPONENTS}
        base_value = safe_float(base_score)
        if not np.isfinite(base_value):
            return sensitivity_payload

        for component_name in SENSITIVITY_COMPONENTS:
            component = components.get(component_name)
            if component is None or not self._is_component_accepted(component, mode):
                continue

            current_value = safe_float(component.get("value"))
            if not np.isfinite(current_value):
                continue
            delta = max(abs(current_value) * SENSITIVITY_DELTA, 0.01)
            perturbed_components = copy.deepcopy(components)
            perturbed_components[component_name]["value"] = current_value + delta
            perturbed_score = safe_float(self.calculate_IIS(perturbed_components, mode).get("IIS"))
            if np.isfinite(perturbed_score):
                sensitivity_payload[f"sens_{component_name}"] = (perturbed_score - base_value) / delta

        return sensitivity_payload

    def _component_result(self, value: float, provenance: str, source: str, note: str) -> ComponentResult:
        """Создаёт унифицированное представление компонента."""

        return {
            "value": float(value) if np.isfinite(value) else np.nan,
            "provenance": provenance,
            "source": source,
            "note": note,
        }

    def _feature_value(
        self,
        inputs: dict[str, Any],
        feature_name: str,
        mode: str,
        allow_proxy: bool = False,
    ) -> tuple[float, str, str]:
        """Возвращает значение признака с учётом режима обработки пропусков."""

        value = safe_float(inputs.get(feature_name))
        provenance = str(inputs.get(f"prov_{feature_name}", "unavailable"))
        source = str(inputs.get(f"source_{feature_name}", ""))

        if not np.isfinite(value):
            return np.nan, "unavailable", ""
        if provenance == "direct":
            return value, "direct", source
        if allow_proxy and mode == "proxy" and provenance == "proxy":
            return value, "proxy", source
        return np.nan, "unavailable", ""

    def _merge_provenance(self, provenances: list[str]) -> str:
        """Сводит происхождение по нескольким подпризнакам к одному статусу."""

        valid = [value for value in provenances if value != "unavailable"]
        if not valid:
            return "unavailable"
        if all(value == "direct" for value in valid):
            return "direct"
        return "proxy"

    def _merge_sources(self, sources: list[str]) -> str:
        """Собирает читаемое описание источников компонента."""

        unique_sources = [source for source in dict.fromkeys(source for source in sources if source)]
        return ", ".join(unique_sources)

    def _safe_sigmoid(self, value: float) -> float:
        """Вычисляет сигмоиду."""

        if not np.isfinite(value):
            return np.nan
        return float(1.0 / (1.0 + np.exp(-value)))

    def _calibrated_gamma_log_power(self, gamma_power: float) -> float:
        """Переводит мощность gamma из В² в мкВ² и логарифмирует для устойчивости."""

        if not np.isfinite(gamma_power):
            return np.nan
        return float(np.log1p(max(gamma_power * EEG_POWER_UV2_SCALE, 0.0)))

    def _calibrated_hrv_ratio(self, ratio_value: float) -> float:
        """Сжимает HF/LF в лог-шкалу, чтобы редкие выбросы не доминировали в IIS."""

        if not np.isfinite(ratio_value):
            return np.nan
        return float(np.log1p(max(ratio_value, 0.0)))

    def _bounded_positive_gate(self, value: float) -> float:
        """Нормирует положительную величину в диапазон 0..1 без жёсткой отсечки."""

        if not np.isfinite(value):
            return np.nan
        positive = max(value, 0.0)
        return float(positive / (1.0 + positive))

    def _robust_tanh(self, value: float, center: float, width: float) -> float:
        """Мягко центрирует показатель относительно калиброванного диапазона."""

        if not np.isfinite(value) or not np.isfinite(center) or not np.isfinite(width) or width <= 0:
            return np.nan
        return float(np.tanh((value - center) / width))

    def _clip_score(self, value: float) -> float:
        """Ограничивает итоговый ИИС диапазоном 0..1."""

        if not np.isfinite(value):
            return np.nan
        return float(np.clip(value, 0.0, 1.0))

    def _is_component_accepted(self, component: ComponentResult, mode: str) -> bool:
        """Проверяет, можно ли включать компонент в расчёт итогового индекса."""

        value = safe_float(component.get("value"))
        provenance = component.get("provenance", "unavailable")
        if not np.isfinite(value):
            return False
        if mode == "proxy":
            return provenance in {"direct", "proxy"}
        return provenance == "direct"

    def _prepare_score_inputs(self, components: dict[str, ComponentResult], mode: str) -> dict[str, Any]:
        """Подготавливает активные компоненты и веса для итоговой формулы."""

        active_components: dict[str, float] = {}
        active_component_provenance: dict[str, str] = {}
        missing_components: list[str] = []

        for component_name in self.required_components:
            component = components.get(component_name, {})
            if self._is_component_accepted(component, mode):
                active_components[component_name] = safe_float(component.get("value"))
                active_component_provenance[component_name] = component.get("provenance", "unavailable")
            else:
                missing_components.append(component_name)

        total_weight = float(sum(self.weights.values()))
        active_weight = float(sum(self.weights[name] for name in active_components))
        coverage_ratio = active_weight / total_weight if total_weight > 0 else 0.0

        if mode in {"strict", "proxy"} and missing_components:
            return {
                "can_score": False,
                "active_components": list(active_components.keys()),
                "active_component_values": active_components,
                "active_component_provenance": active_component_provenance,
                "missing_components": missing_components,
                "coverage_ratio": coverage_ratio,
                "insufficient_reason": "Для режима требуется полный набор допустимых компонентов.",
            }

        if not active_components:
            return {
                "can_score": False,
                "active_components": [],
                "active_component_values": {},
                "active_component_provenance": {},
                "missing_components": list(self.required_components),
                "coverage_ratio": coverage_ratio,
                "insufficient_reason": "Нет компонентов, пригодных для расчёта ИИС.",
            }

        if mode == "hybrid":
            weight_sum = float(sum(self.weights[name] for name in active_components))
            normalized_weights = {
                name: (self.weights[name] / weight_sum if weight_sum > 0 else 0.0)
                for name in active_components
            }
        else:
            normalized_weights = {name: self.weights[name] for name in active_components}

        return {
            "can_score": True,
            "active_components": list(active_components.keys()),
            "active_component_values": active_components,
            "active_component_provenance": active_component_provenance,
            "missing_components": missing_components,
            "coverage_ratio": coverage_ratio,
            "normalized_weights": normalized_weights,
            "insufficient_reason": "",
        }

    def _finalize_score_payload(
        self,
        prepared: dict[str, Any],
        raw_score: float,
        iis_score: float,
        contributions: dict[str, float],
        formula_note: str,
    ) -> dict[str, Any]:
        """Собирает итоговый словарь результата вычисления ИИС."""

        return {
            "IIS": self._clip_score(iis_score),
            "raw_score": raw_score,
            "coverage_ratio": prepared.get("coverage_ratio", 0.0),
            "active_component_count": len(prepared.get("active_components", [])),
            "active_components": prepared.get("active_components", []),
            "active_component_provenance": prepared.get("active_component_provenance", {}),
            "missing_components": prepared.get("missing_components", []),
            "contributions": contributions,
            "formula_note": formula_note,
            "insufficient_reason": prepared.get("insufficient_reason", ""),
        }

    def _empty_score_payload(self, prepared: dict[str, Any], formula_note: str) -> dict[str, Any]:
        """Возвращает пустой результат, если ИИС посчитать нельзя."""

        return {
            "IIS": np.nan,
            "raw_score": np.nan,
            "coverage_ratio": prepared.get("coverage_ratio", 0.0),
            "active_component_count": len(prepared.get("active_components", [])),
            "active_components": prepared.get("active_components", []),
            "active_component_provenance": prepared.get("active_component_provenance", {}),
            "missing_components": prepared.get("missing_components", list(self.required_components)),
            "contributions": {},
            "formula_note": formula_note,
            "insufficient_reason": prepared.get("insufficient_reason", ""),
        }

    def _normalize_self_report(self, inputs: dict[str, Any], field_name: str) -> tuple[float, str]:
        """Нормирует self-report в диапазон 0..1."""

        value = safe_float(inputs.get(field_name))
        if not np.isfinite(value):
            return np.nan, ""

        dataset = str(inputs.get("dataset", "")).lower()
        if dataset in SELF_REPORT_RANGES:
            low, high = SELF_REPORT_RANGES[dataset]
            if high <= low:
                return np.nan, ""
            normalized = (value - low) / (high - low)
        else:
            normalized = value
        return float(np.clip(normalized, 0.0, 1.0)), field_name

    def _proxy_motivation_level(self, inputs: dict[str, Any]) -> tuple[float, str]:
        """Оценивает мотивацию как прокси для дофамина."""

        valence_norm, source = self._normalize_self_report(inputs, "valence")
        if np.isfinite(valence_norm):
            return valence_norm, source

        label = str(inputs.get("label", "")).lower()
        label_map = {
            "amusement": 0.90,
            "baseline": 0.65,
            "disbalance": 0.40,
            "stress": 0.20,
        }
        if label in label_map:
            return label_map[label], "label"

        stress_label = safe_float(inputs.get("stress_label"))
        if np.isfinite(stress_label):
            return float(np.clip(1.0 - stress_label, 0.0, 1.0)), "stress_label"
        return np.nan, ""

    def _proxy_stress_level(self, inputs: dict[str, Any]) -> tuple[float, str]:
        """Оценивает уровень стрессовой нагрузки как прокси для кортизола."""

        stress_label = safe_float(inputs.get("stress_label"))
        if np.isfinite(stress_label):
            return float(np.clip(stress_label, 0.0, 1.0)), "stress_label"

        arousal_norm, source = self._normalize_self_report(inputs, "arousal")
        if np.isfinite(arousal_norm):
            return arousal_norm, source

        label = str(inputs.get("label", "")).lower()
        label_map = {
            "baseline": 0.15,
            "amusement": 0.25,
            "disbalance": 0.55,
            "stress": 0.90,
        }
        if label in label_map:
            return label_map[label], "label"
        return np.nan, ""

    def _proxy_dopamine(self, inputs: dict[str, Any], mode: str) -> tuple[float, str, str]:
        """Возвращает прокси-оценку дофамина."""

        if mode != "proxy":
            return np.nan, "unavailable", ""
        level, source = self._proxy_motivation_level(inputs)
        if not np.isfinite(level):
            return np.nan, "unavailable", ""
        low, high = PROXY_RANGES["dopamine"]
        value = low + (high - low) * level
        return value, "proxy", source

    def _proxy_cortisol(self, inputs: dict[str, Any], mode: str) -> tuple[float, str, str]:
        """Возвращает прокси-оценку кортизола."""

        if mode != "proxy":
            return np.nan, "unavailable", ""
        level, source = self._proxy_stress_level(inputs)
        if not np.isfinite(level):
            return np.nan, "unavailable", ""
        low, high = PROXY_RANGES["cortisol"]
        value = low + (high - low) * level
        return value, "proxy", source

    @abstractmethod
    def calculate_A(self, inputs: dict[str, Any], mode: str) -> ComponentResult:
        """Считает компонент A."""

    @abstractmethod
    def calculate_Gamma(self, inputs: dict[str, Any], mode: str) -> ComponentResult:
        """Считает компонент Gamma."""

    @abstractmethod
    def calculate_H(self, inputs: dict[str, Any], mode: str) -> ComponentResult:
        """Считает компонент H."""

    @abstractmethod
    def calculate_V(self, inputs: dict[str, Any], mode: str) -> ComponentResult:
        """Считает компонент V."""

    @abstractmethod
    def calculate_Q(self, inputs: dict[str, Any], mode: str) -> ComponentResult:
        """Считает компонент Q."""

    @abstractmethod
    def calculate_K(
        self,
        inputs: dict[str, Any],
        mode: str,
        components: dict[str, ComponentResult],
    ) -> ComponentResult:
        """Считает компонент K."""

    @abstractmethod
    def calculate_IIS(self, components: dict[str, ComponentResult], mode: str) -> dict[str, Any]:
        """Считает итоговый ИИС."""
