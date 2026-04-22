"""Контрастная версия ИИС 5.0 с менее плоской интерпретацией итоговой шкалы."""

from __future__ import annotations

import numpy as np

from config.settings import VERSION5_CALIBRATION
from models.base_model import ComponentResult
from models.iis_v4 import IISVersion4


class IISVersion5(IISVersion4):
    """Версия 5.0 усиливает контраст между режимами без гормонального блока."""

    @staticmethod
    def _signed_power(value: float, power: float) -> float:
        """Нелинейно раскрывает умеренные значения, не ломая знак компонента."""

        if not np.isfinite(value):
            return float(np.nan)
        magnitude = float(abs(value) ** power)
        return float(np.sign(value) * magnitude)

    def calculate_Q(self, inputs: dict[str, object], mode: str) -> ComponentResult:
        """Считает Q как контрастную физиологическую согласованность A, V и Gamma."""

        component_map = {
            "A": self.calculate_A(inputs, mode),
            "Gamma": self.calculate_Gamma(inputs, mode),
            "V": self.calculate_V(inputs, mode),
        }

        a_value = float(component_map["A"].get("value", np.nan))
        gamma_value = float(component_map["Gamma"].get("value", np.nan))
        v_value = float(component_map["V"].get("value", np.nan))

        direct_terms: list[float] = []
        provenances: list[str] = []
        sources: list[str] = []

        if np.isfinite(a_value):
            provenances.append(component_map["A"].get("provenance", "unavailable"))
            sources.append(component_map["A"].get("source", ""))
        if np.isfinite(gamma_value):
            provenances.append(component_map["Gamma"].get("provenance", "unavailable"))
            sources.append(component_map["Gamma"].get("source", ""))
        if np.isfinite(v_value):
            provenances.append(component_map["V"].get("provenance", "unavailable"))
            sources.append(component_map["V"].get("source", ""))

        if np.isfinite(a_value):
            direct_terms.append(0.50 * a_value)
        if np.isfinite(v_value):
            direct_terms.append(0.42 * v_value)
        if np.isfinite(gamma_value):
            direct_terms.append(-0.30 * gamma_value)

        q_phys = np.nan
        if direct_terms:
            shaped_a = self._signed_power(a_value, VERSION5_CALIBRATION["q_signed_power"]) if np.isfinite(a_value) else np.nan
            shaped_v = self._signed_power(v_value, VERSION5_CALIBRATION["q_signed_power"]) if np.isfinite(v_value) else np.nan
            shaped_gamma = self._signed_power(gamma_value, 1.05) if np.isfinite(gamma_value) else np.nan
            coherence = 0.0
            if np.isfinite(a_value) and np.isfinite(v_value):
                coherence += a_value * v_value
                coherence -= VERSION5_CALIBRATION["q_mismatch_penalty"] * abs(a_value - v_value)
            if np.isfinite(gamma_value):
                coherence -= VERSION5_CALIBRATION["q_gamma_penalty"] * max(gamma_value, 0.0)

            synergy = 0.0
            energy = 0.0
            conflict = 0.0
            if np.isfinite(shaped_a) and np.isfinite(shaped_v):
                synergy += np.sign(shaped_a + shaped_v) * np.sqrt(abs(shaped_a * shaped_v) + 1e-9)
                energy += 0.5 * (abs(shaped_a) + abs(shaped_v))
                conflict += abs(shaped_a - shaped_v)
            if np.isfinite(shaped_gamma) and np.isfinite(shaped_v):
                conflict += VERSION5_CALIBRATION["q_gamma_v_penalty"] * max(shaped_gamma, 0.0) * (1.0 - max(shaped_v, 0.0))

            q_phys = float(
                np.tanh(
                    VERSION5_CALIBRATION["q_linear_gain"] * sum(direct_terms)
                    + VERSION5_CALIBRATION["q_coherence_gain"] * coherence
                    + VERSION5_CALIBRATION["q_synergy_gain"] * synergy
                    + VERSION5_CALIBRATION["q_energy_gain"] * energy * np.sign(sum(direct_terms) if direct_terms else 0.0)
                    - VERSION5_CALIBRATION["q_conflict_gain"] * conflict
                )
            )

        q_value = q_phys
        provenance = self._merge_provenance(provenances)
        source = self._merge_sources(sources)
        note = (
            "Версия 5.0: Q как контрастная согласованность A и V с штрафом за mismatch и перегретую Gamma; "
            "это делает середину шкалы менее плоской."
        )

        if mode == "proxy":
            valence_norm, valence_source = self._normalize_self_report(inputs, "valence")
            arousal_norm, arousal_source = self._normalize_self_report(inputs, "arousal")
            if np.isfinite(valence_norm) and np.isfinite(arousal_norm):
                q_anchor = float(np.tanh(2.4 * (0.65 * valence_norm + 0.35 * (1.0 - arousal_norm) - 0.5)))
                proxy_weight = VERSION5_CALIBRATION["q_proxy_weight"]
                if np.isfinite(q_phys):
                    q_value = float((1.0 - proxy_weight) * q_phys + proxy_weight * q_anchor)
                else:
                    q_value = q_anchor
                provenance = "proxy"
                source = self._merge_sources([source, valence_source, arousal_source])
                note = (
                    "Версия 5.0: Q как контрастное физиологическое ядро с мягкой proxy-якоризацией по valence/arousal; "
                    "вес proxy уменьшен относительно V4, чтобы не терять собственную физиологическую геометрию."
                )

        if not np.isfinite(q_value):
            return self._component_result(np.nan, "unavailable", "", "Q 5.0 недоступен без опорных EEG/ECG или self-report.")

        return self._component_result(q_value, provenance, source, note)

    def calculate_IIS(self, components: dict[str, ComponentResult], mode: str) -> dict[str, object]:
        """Считает итоговый IIS 5.0 как смесь базовой шкалы и контрастной tanh-интерпретации."""

        prepared = self._prepare_score_inputs(components, mode)
        note = (
            "Версия 5.0: IIS = (1-m)·σ(core) + m·(0.5 + 0.5·tanh(contrast)); "
            "слой contrast расширяет рабочий диапазон и уменьшает плоскость интерпретации без гормональных proxy."
        )
        if not prepared.get("can_score"):
            return self._empty_score_payload(prepared, note)

        values = prepared["active_component_values"]
        weights = prepared["normalized_weights"]
        contributions: dict[str, float] = {}
        core_linear = 0.0

        for name, value in values.items():
            direction = -1.0 if name == "Gamma" else 1.0
            contribution = direction * weights[name] * value
            contributions[name] = contribution
            core_linear += contribution

        a_value = float(values.get("A", 0.0))
        q_value = float(values.get("Q", 0.0))
        v_value = float(values.get("V", 0.0))
        gamma_value = float(values.get("Gamma", 0.0))
        shaped_a = self._signed_power(a_value, VERSION5_CALIBRATION["output_signed_power"])
        shaped_v = self._signed_power(v_value, VERSION5_CALIBRATION["output_signed_power"])
        shaped_q = self._signed_power(q_value, VERSION5_CALIBRATION["output_signed_power"])
        centered = (
            0.14 * shaped_a
            - 0.08 * gamma_value
            + 0.22 * shaped_v
            + 0.56 * shaped_q
        ) - VERSION5_CALIBRATION["output_center"]

        regime_synergy = 0.0
        if np.isfinite(shaped_a) and np.isfinite(shaped_v):
            regime_synergy += np.sqrt(abs(shaped_a * shaped_v) + 1e-9) * np.sign(shaped_a + shaped_v)
        if np.isfinite(shaped_q) and np.isfinite(shaped_v):
            regime_synergy += 0.8 * np.sqrt(abs(shaped_q * shaped_v) + 1e-9) * np.sign(shaped_q + shaped_v)

        phase_energy = float(np.sqrt(np.mean([shaped_a**2, shaped_v**2, shaped_q**2])))
        conflict = abs(shaped_a - shaped_v) + 0.75 * abs(shaped_q - shaped_v)
        contrast_drive = (
            VERSION5_CALIBRATION["output_gain"] * centered
            + VERSION5_CALIBRATION["output_curve"] * (centered**3)
            + VERSION5_CALIBRATION["output_balance_coupling"] * (q_value - v_value)
            - VERSION5_CALIBRATION["output_gamma_brake"] * max(gamma_value, 0.0)
            + VERSION5_CALIBRATION["output_regime_gain"] * regime_synergy
            + VERSION5_CALIBRATION["output_energy_gain"] * phase_energy * np.sign(centered)
            - VERSION5_CALIBRATION["output_conflict_gain"] * conflict
            - VERSION5_CALIBRATION["output_gamma_cross_penalty"] * max(gamma_value, 0.0) * (1.0 - max(v_value, 0.0))
        )

        base_score = self._safe_sigmoid(core_linear + 0.22 * regime_synergy - 0.10 * conflict)
        contrast_score = 0.5 + 0.5 * float(np.tanh(contrast_drive))
        contrast_mix = VERSION5_CALIBRATION["output_contrast_mix"]
        iis_score = self._clip_score((1.0 - contrast_mix) * base_score + contrast_mix * contrast_score)
        return self._finalize_score_payload(prepared, core_linear, iis_score, contributions, note)
