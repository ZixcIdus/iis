"""Универсальная тестовая версия ИИС 6.0 с мягким переключением между режимами."""

from __future__ import annotations

import numpy as np

from config.settings import VERSION6_CALIBRATION
from models.base_model import ComponentResult
from models.iis_v4 import IISVersion4


class IISVersion6(IISVersion4):
    """Версия 6.0 моделирует состояние как мягкую смесь нескольких физиологических режимов."""

    @staticmethod
    def _signed_power(value: float, power: float) -> float:
        """Нелинейно раскрывает сигнал без потери знака."""

        if not np.isfinite(value):
            return float(np.nan)
        return float(np.sign(value) * (abs(value) ** power))

    @staticmethod
    def _softmax(logits: list[float], temperature: float) -> np.ndarray:
        """Считает мягкие веса режимов."""

        raw = np.asarray(logits, dtype=float)
        if not np.isfinite(raw).any():
            return np.full(len(logits), 1.0 / max(len(logits), 1), dtype=float)
        centered = raw - np.nanmax(raw)
        scaled = centered * float(max(temperature, 1e-6))
        exp_values = np.exp(np.clip(scaled, -50.0, 50.0))
        total = float(exp_values.sum())
        if total <= 0.0 or not np.isfinite(total):
            return np.full(len(logits), 1.0 / max(len(logits), 1), dtype=float)
        return exp_values / total

    def _shape_components(self, a_value: float, gamma_value: float, v_value: float, q_value: float | None = None) -> dict[str, float]:
        """Готовит нелинейно преобразованные компоненты."""

        power = VERSION6_CALIBRATION["shape_power"]
        gamma_power = VERSION6_CALIBRATION["gamma_shape_power"]
        shaped = {
            "A": self._signed_power(a_value, power) if np.isfinite(a_value) else 0.0,
            "Gamma": self._signed_power(gamma_value, gamma_power) if np.isfinite(gamma_value) else 0.0,
            "V": self._signed_power(v_value, power) if np.isfinite(v_value) else 0.0,
            "Q": self._signed_power(float(q_value), power) if q_value is not None and np.isfinite(q_value) else 0.0,
        }
        shaped["GammaPos"] = max(shaped["Gamma"], 0.0)
        return shaped

    def calculate_Q(self, inputs: dict[str, object], mode: str) -> ComponentResult:
        """Считает Q как смесь регуляторного, мобилизационного и истощающего режимов."""

        component_map = {
            "A": self.calculate_A(inputs, mode),
            "Gamma": self.calculate_Gamma(inputs, mode),
            "V": self.calculate_V(inputs, mode),
        }

        a_value = float(component_map["A"].get("value", np.nan))
        gamma_value = float(component_map["Gamma"].get("value", np.nan))
        v_value = float(component_map["V"].get("value", np.nan))
        if not any(np.isfinite(value) for value in (a_value, gamma_value, v_value)):
            return self._component_result(np.nan, "unavailable", "", "Q 6.0 недоступен без базовых EEG/ECG-компонентов.")

        provenances = [
            component_map[name].get("provenance", "unavailable")
            for name, value in (("A", a_value), ("Gamma", gamma_value), ("V", v_value))
            if np.isfinite(value)
        ]
        sources = [
            component_map[name].get("source", "")
            for name, value in (("A", a_value), ("Gamma", gamma_value), ("V", v_value))
            if np.isfinite(value)
        ]

        shaped = self._shape_components(a_value, gamma_value, v_value)
        gate_logits = [
            0.95 * shaped["A"] + 1.05 * shaped["V"] - 0.55 * shaped["GammaPos"],
            -0.18 * shaped["A"] + 0.48 * shaped["V"] - 1.05 * shaped["GammaPos"],
            -0.80 * shaped["A"] - 0.95 * shaped["V"] + 0.72 * shaped["GammaPos"],
        ]
        gates = self._softmax(gate_logits, VERSION6_CALIBRATION["gate_temperature"])

        synergy = np.sign(shaped["A"] + shaped["V"]) * np.sqrt(abs(shaped["A"] * shaped["V"]) + 1e-9)
        q_reg = VERSION6_CALIBRATION["reg_q_base"] + VERSION6_CALIBRATION["reg_q_amp"] * np.tanh(
            1.10 * shaped["A"] + 1.15 * shaped["V"] - 0.45 * shaped["GammaPos"] + 0.30 * synergy
        )
        q_mob = VERSION6_CALIBRATION["mob_q_base"] + VERSION6_CALIBRATION["mob_q_amp"] * np.tanh(
            0.10 * shaped["A"] + 0.55 * shaped["V"] - 1.20 * shaped["GammaPos"]
        )
        q_dep = VERSION6_CALIBRATION["dep_q_base"] + VERSION6_CALIBRATION["dep_q_amp"] * np.tanh(
            -0.65 * abs(shaped["A"]) - 0.95 * abs(shaped["V"]) + 0.95 * shaped["GammaPos"]
        )
        q_value = float(np.clip(gates[0] * q_reg + gates[1] * q_mob + gates[2] * q_dep, 0.0, 1.0))

        provenance = self._merge_provenance(provenances)
        source = self._merge_sources(sources)
        note = (
            "Версия 6.0: Q вычисляется как мягкая смесь трёх режимов: regulation, mobilization и depletion; "
            "вес режима задаётся softmax-гейтом по A, V и Gamma."
        )

        if mode == "proxy":
            valence_norm, valence_source = self._normalize_self_report(inputs, "valence")
            arousal_norm, arousal_source = self._normalize_self_report(inputs, "arousal")
            if np.isfinite(valence_norm) and np.isfinite(arousal_norm):
                q_anchor = float(np.clip(0.70 * valence_norm + 0.30 * (1.0 - arousal_norm), 0.0, 1.0))
                proxy_weight = VERSION6_CALIBRATION["q_proxy_weight"]
                q_value = float((1.0 - proxy_weight) * q_value + proxy_weight * q_anchor)
                provenance = "proxy"
                source = self._merge_sources([source, valence_source, arousal_source])
                note = (
                    "Версия 6.0: Q как gated-физиологическое ядро с ослабленной proxy-якоризацией по valence/arousal."
                )

        return self._component_result(q_value, provenance, source, note)

    def calculate_IIS(self, components: dict[str, ComponentResult], mode: str) -> dict[str, object]:
        """Считает итоговый IIS как универсальную смесь режимов regulation, mobilization и depletion."""

        prepared = self._prepare_score_inputs(components, mode)
        note = (
            "Версия 6.0: IIS как soft mixture of regimes; модель переключается между regulation, mobilization и depletion "
            "через мягкий gate вместо одной глобальной почти линейной формулы."
        )
        if not prepared.get("can_score"):
            return self._empty_score_payload(prepared, note)

        values = prepared["active_component_values"]
        a_value = float(values.get("A", 0.0))
        gamma_value = float(values.get("Gamma", 0.0))
        v_value = float(values.get("V", 0.0))
        q_value = float(values.get("Q", 0.0))
        shaped = self._shape_components(a_value, gamma_value, v_value, q_value=q_value)

        reg_logit = 0.60 * shaped["Q"] + 0.45 * shaped["V"] + 0.15 * shaped["A"] - 0.30 * shaped["GammaPos"]
        mob_logit = -0.05 * shaped["Q"] + 0.20 * shaped["V"] - 0.55 * shaped["GammaPos"] - 0.08 * shaped["A"]
        dep_logit = -0.70 * shaped["Q"] - 0.55 * shaped["V"] - 0.20 * shaped["A"] + 0.35 * shaped["GammaPos"]
        gates = self._softmax([reg_logit, mob_logit, dep_logit], VERSION6_CALIBRATION["gate_temperature"])

        synergy_reg = np.sign(shaped["Q"] + shaped["V"]) * np.sqrt(abs(shaped["Q"] * shaped["V"]) + 1e-9)
        conflict = abs(shaped["A"] - shaped["V"]) + 0.75 * abs(shaped["Q"] - shaped["V"])
        reg_level = VERSION6_CALIBRATION["reg_iis_base"] + VERSION6_CALIBRATION["reg_iis_amp"] * np.tanh(
            0.75 * shaped["Q"] + 0.55 * shaped["V"] + 0.18 * shaped["A"] - 0.25 * shaped["GammaPos"] + 0.25 * synergy_reg
        )
        mob_level = VERSION6_CALIBRATION["mob_iis_base"] + VERSION6_CALIBRATION["mob_iis_amp"] * np.tanh(
            -0.08 * shaped["Q"] + 0.32 * shaped["V"] - 0.52 * shaped["GammaPos"] + 0.08 * shaped["A"]
        )
        dep_level = VERSION6_CALIBRATION["dep_iis_base"] + VERSION6_CALIBRATION["dep_iis_amp"] * np.tanh(
            -0.62 * shaped["Q"] - 0.52 * shaped["V"] - 0.18 * shaped["A"] + 0.38 * shaped["GammaPos"]
        )

        regime_score = float(gates[0] * reg_level + gates[1] * mob_level + gates[2] * dep_level)
        entropy = float(-np.sum(gates * np.log(gates + 1e-9)) / np.log(3.0))
        regime_balance = float(gates[0] - gates[2])
        iis_score = regime_score - VERSION6_CALIBRATION["transition_entropy_weight"] * entropy + VERSION6_CALIBRATION["regime_balance_weight"] * regime_balance
        iis_score -= VERSION6_CALIBRATION["conflict_penalty"] * conflict
        iis_score = self._clip_score(iis_score)

        raw_score = 0.14 * shaped["A"] - 0.08 * shaped["Gamma"] + 0.24 * shaped["V"] + 0.54 * shaped["Q"]
        contributions = {
            "A": 0.14 * shaped["A"],
            "Gamma": -0.08 * shaped["Gamma"],
            "V": 0.24 * shaped["V"],
            "Q": 0.54 * shaped["Q"],
        }
        return self._finalize_score_payload(prepared, raw_score, iis_score, contributions, note)
