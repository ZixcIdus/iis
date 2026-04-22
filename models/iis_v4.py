"""Наглядная версия ИИС 4.0 без гормонального блока."""

from __future__ import annotations

import numpy as np

from config.settings import EPSILON, VERSION4_CALIBRATION
from models.base_model import BaseIISModel, ComponentResult


class IISVersion4(BaseIISModel):
    """Калиброванная по реальным EEG/ECG данным версия ИИС без гормонов."""

    def calculate_A(self, inputs: dict[str, object], mode: str) -> ComponentResult:
        """Считает A как устойчивую асимметрию по alpha и общей мощности."""

        alpha_asymmetry, prov_1, source_1 = self._feature_value(inputs, "alpha_asymmetry", mode, allow_proxy=False)
        left_power, prov_2, source_2 = self._feature_value(inputs, "eeg_left_power", mode, allow_proxy=False)
        right_power, prov_3, source_3 = self._feature_value(inputs, "eeg_right_power", mode, allow_proxy=False)

        components: list[float] = []
        provenances: list[str] = []
        sources: list[str] = []

        if np.isfinite(alpha_asymmetry):
            alpha_term = self._robust_tanh(
                alpha_asymmetry,
                VERSION4_CALIBRATION["alpha_asymmetry_center"],
                VERSION4_CALIBRATION["alpha_asymmetry_width"],
            )
            components.append(1.10 * alpha_term)
            provenances.append(prov_1)
            sources.append(source_1)

        if np.isfinite(left_power) and np.isfinite(right_power):
            total_asymmetry = (left_power - right_power) / (left_power + right_power + EPSILON)
            total_term = self._robust_tanh(
                total_asymmetry,
                VERSION4_CALIBRATION["total_asymmetry_center"],
                VERSION4_CALIBRATION["total_asymmetry_width"],
            )
            components.append(0.40 * total_term)
            provenances.extend([prov_2, prov_3])
            sources.extend([source_2, source_3])

        if not components:
            return self._component_result(np.nan, "unavailable", "", "A 4.0 недоступен без EEG-асимметрии.")

        value = float(np.tanh(sum(components)))
        return self._component_result(
            value,
            self._merge_provenance(provenances),
            self._merge_sources(sources),
            "Версия 4.0: A=tanh(1.10·Aalpha + 0.40·Atotal), где обе части центрированы по реальным диапазонам ds002722/ds002724.",
        )

    def calculate_Gamma(self, inputs: dict[str, object], mode: str) -> ComponentResult:
        """Считает Gamma как нагрузку активации по gamma и gamma/alpha."""

        gamma_power, prov_1, source_1 = self._feature_value(inputs, "eeg_gamma_power", mode, allow_proxy=False)
        gamma_alpha_ratio, prov_2, source_2 = self._feature_value(inputs, "gamma_alpha_ratio", mode, allow_proxy=False)

        components: list[float] = []
        provenances: list[str] = []
        sources: list[str] = []

        if np.isfinite(gamma_power):
            gamma_log = self._calibrated_gamma_log_power(gamma_power)
            gamma_term = self._robust_tanh(
                gamma_log,
                VERSION4_CALIBRATION["gamma_log_center"],
                VERSION4_CALIBRATION["gamma_log_width"],
            )
            components.append(0.30 * gamma_term)
            provenances.append(prov_1)
            sources.append(source_1)

        if np.isfinite(gamma_alpha_ratio):
            gamma_alpha_log = float(np.log1p(max(gamma_alpha_ratio, 0.0)))
            burden_term = self._robust_tanh(
                gamma_alpha_log,
                VERSION4_CALIBRATION["gamma_alpha_log_center"],
                VERSION4_CALIBRATION["gamma_alpha_log_width"],
            )
            components.append(0.70 * burden_term)
            provenances.append(prov_2)
            sources.append(source_2)

        if not components:
            return self._component_result(np.nan, "unavailable", "", "Gamma 4.0 недоступна без gamma-признаков.")

        value = float(np.tanh(sum(components)))
        return self._component_result(
            value,
            self._merge_provenance(provenances),
            self._merge_sources(sources),
            "Версия 4.0: Gamma отражает перегрев активации и затем вычитается из итогового IIS.",
        )

    def calculate_H(self, inputs: dict[str, object], mode: str) -> ComponentResult:
        """В версии 4.0 гормональный блок исключён из модели."""

        return self._component_result(np.nan, "unavailable", "", "Версия 4.0 не использует гормональный блок H.")

    def calculate_V(self, inputs: dict[str, object], mode: str) -> ComponentResult:
        """Считает V как автономную устойчивость с упором на ЧСС."""

        heart_rate, prov_1, source_1 = self._feature_value(inputs, "heart_rate", mode, allow_proxy=True)
        hf_lf_ratio, prov_2, source_2 = self._feature_value(inputs, "hrv_hf_lf", mode, allow_proxy=True)

        if not np.isfinite(heart_rate) and not np.isfinite(hf_lf_ratio):
            return self._component_result(np.nan, "unavailable", "", "V 4.0 недоступен без ECG/HRV.")

        hr_term = np.nan
        ratio_term = np.nan
        if np.isfinite(heart_rate):
            hr_term = self._robust_tanh(
                VERSION4_CALIBRATION["heart_rate_center"] - heart_rate,
                0.0,
                VERSION4_CALIBRATION["heart_rate_width"],
            )

        if np.isfinite(hf_lf_ratio):
            ratio_log = self._calibrated_hrv_ratio(hf_lf_ratio)
            ratio_term = self._robust_tanh(
                ratio_log,
                VERSION4_CALIBRATION["hf_lf_log_center"],
                VERSION4_CALIBRATION["hf_lf_log_width"],
            )

        if np.isfinite(hr_term) and np.isfinite(ratio_term):
            value = float(np.tanh(0.80 * hr_term + 0.20 * ratio_term))
            provenance = self._merge_provenance([prov_1, prov_2])
            source = self._merge_sources([source_1, source_2])
            note = "Версия 4.0: V=tanh(0.80·Vhr + 0.20·Vhrv); вклад HF/LF ослаблен из-за шумности открытых HRV."
            return self._component_result(value, provenance, source, note)

        if np.isfinite(hr_term):
            return self._component_result(
                float(hr_term),
                prov_1,
                source_1,
                "Версия 4.0: V по обратной ЧСС, потому что именно она оказалась наиболее устойчивой на реальных открытых наборах.",
            )

        return self._component_result(
            float(ratio_term),
            prov_2,
            source_2,
            "Версия 4.0: V по калиброванному HF/LF в лог-шкале.",
        )

    def calculate_Q(self, inputs: dict[str, object], mode: str) -> ComponentResult:
        """Считает Q как интегральное качество состояния с опциональной прокси-якоризацией."""

        component_map = {
            "A": self.calculate_A(inputs, mode),
            "Gamma": self.calculate_Gamma(inputs, mode),
            "V": self.calculate_V(inputs, mode),
        }

        direct_terms: list[float] = []
        provenances: list[str] = []
        sources: list[str] = []

        a_value = float(component_map["A"].get("value", np.nan))
        gamma_value = float(component_map["Gamma"].get("value", np.nan))
        v_value = float(component_map["V"].get("value", np.nan))

        if np.isfinite(a_value):
            direct_terms.append(0.55 * a_value)
            provenances.append(component_map["A"].get("provenance", "unavailable"))
            sources.append(component_map["A"].get("source", ""))
        if np.isfinite(v_value):
            direct_terms.append(0.45 * v_value)
            provenances.append(component_map["V"].get("provenance", "unavailable"))
            sources.append(component_map["V"].get("source", ""))
        if np.isfinite(gamma_value):
            direct_terms.append(-0.25 * gamma_value)
            provenances.append(component_map["Gamma"].get("provenance", "unavailable"))
            sources.append(component_map["Gamma"].get("source", ""))

        q_phys = float(np.tanh(sum(direct_terms))) if direct_terms else np.nan
        q_value = q_phys
        provenance = self._merge_provenance(provenances)
        source = self._merge_sources(sources)
        note = "Версия 4.0: Q как физиологическая интеграция A/V/Gamma."

        if mode == "proxy":
            valence_norm, valence_source = self._normalize_self_report(inputs, "valence")
            arousal_norm, arousal_source = self._normalize_self_report(inputs, "arousal")
            if np.isfinite(valence_norm) and np.isfinite(arousal_norm):
                q_anchor = float(np.tanh(2.2 * (0.62 * valence_norm + 0.38 * (1.0 - arousal_norm) - 0.5)))
                proxy_weight = VERSION4_CALIBRATION["q_proxy_weight"]
                if np.isfinite(q_phys):
                    q_value = float((1.0 - proxy_weight) * q_phys + proxy_weight * q_anchor)
                else:
                    q_value = q_anchor
                provenance = "proxy"
                source = self._merge_sources([source, valence_source, arousal_source])
                note = (
                    "Версия 4.0: Q как смесь физиологического ядра и мягкого proxy-якоря по valence/arousal; "
                    "это нужно, чтобы шкала модели не отрывалась от реальных эмоциональных меток."
                )

        if not np.isfinite(q_value):
            return self._component_result(np.nan, "unavailable", "", "Q 4.0 недоступен без опорных EEG/ECG или self-report.")

        return self._component_result(q_value, provenance, source, note)

    def calculate_K(
        self,
        inputs: dict[str, object],
        mode: str,
        components: dict[str, ComponentResult],
    ) -> ComponentResult:
        """В версии 4.0 коэффициент K исключён из модели."""

        return self._component_result(np.nan, "unavailable", "", "Версия 4.0 не использует коэффициент K.")

    def calculate_IIS(self, components: dict[str, ComponentResult], mode: str) -> dict[str, object]:
        """Считает итоговый IIS по калиброванной на открытых данных формуле 4.0."""

        prepared = self._prepare_score_inputs(components, mode)
        note = (
            "Версия 4.0: IIS=σ(0.10·A - 0.05·Gamma + 0.25·V + 0.60·Q); "
            "формула откалибрована на ds002722/ds002724 и исключает гормональный блок."
        )
        if not prepared.get("can_score"):
            return self._empty_score_payload(prepared, note)

        values = prepared["active_component_values"]
        weights = prepared["normalized_weights"]
        contributions: dict[str, float] = {}
        raw_score = 0.0

        for name, value in values.items():
            direction = -1.0 if name == "Gamma" else 1.0
            contribution = direction * weights[name] * value
            contributions[name] = contribution
            raw_score += contribution

        iis_score = self._safe_sigmoid(raw_score)
        return self._finalize_score_payload(prepared, raw_score, iis_score, contributions, note)
