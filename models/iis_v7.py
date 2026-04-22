"""Экспериментальная версия ИИС 7.0 с Sumigron-агрегацией на уровне подтермов признаков."""

from __future__ import annotations

import numpy as np

from config.settings import VERSION4_CALIBRATION, VERSION7_CALIBRATION, VERSION7_DATASET_OVERRIDES
from models.base_model import ComponentResult
from models.iis_v6 import IISVersion6


class IISVersion7(IISVersion6):
    """V7-beta: структурная Sumigron-агрегация подтермов + dataset-aware gated-логика режимов."""

    def _dataset_key(self, inputs: dict[str, object] | None) -> str:
        return str((inputs or {}).get("dataset", "")).strip().lower()

    def _v7_params(self, dataset_key: str = "") -> dict[str, object]:
        params: dict[str, object] = dict(VERSION7_CALIBRATION)
        override = VERSION7_DATASET_OVERRIDES.get(dataset_key, {})
        for key, value in override.items():
            if key == "weights":
                continue
            params[key] = value
        params["weights"] = dict(self.weights)
        if isinstance(override.get("weights"), dict):
            params["weights"] = dict(override["weights"])
        return params

    def compute_components(self, inputs: dict[str, object], mode: str) -> dict[str, ComponentResult]:
        components = super().compute_components(inputs, mode)
        components["_context"] = {"dataset": self._dataset_key(inputs)}
        return components

    def _sumigron_scalar(self, values: list[float], params: dict[str, object] | None = None) -> float:
        """Собирает несколько подтермов в один структурный скаляр без раннего насыщения."""

        v7_params = params or VERSION7_CALIBRATION
        finite = np.asarray([value for value in values if np.isfinite(value)], dtype=float)
        if finite.size == 0:
            return float(np.nan)
        if finite.size == 1:
            return float(finite[0])

        local_mean = float(np.mean(finite))
        local_std = float(np.std(finite))
        local_z = (finite - local_mean) / (local_std + 1e-6)
        drive = 0.60 * local_z + 0.40 * np.abs(local_z)
        weights = self._softmax(drive.tolist(), float(v7_params["sumigron_temperature"]))

        attentive_mean = float(np.sum(weights * finite))
        attentive_std = float(np.sqrt(np.sum(weights * (finite - attentive_mean) ** 2)))
        attentive_energy = float(np.log1p(np.sum(weights * (finite**2))))

        sign = float(np.sign(attentive_mean))
        if sign == 0.0:
            sign = float(np.sign(np.sum(finite)))
        if sign == 0.0:
            sign = 1.0

        structured_boost = sign * (
            float(v7_params["sumigron_structure_weight"]) * self._bounded_positive_gate(attentive_std)
            + float(v7_params["sumigron_energy_weight"]) * self._bounded_positive_gate(attentive_energy)
        )
        return float(float(v7_params["sumigron_level_weight"]) * attentive_mean + structured_boost)

    def calculate_A(self, inputs: dict[str, object], mode: str) -> ComponentResult:
        """Считает A как Sumigron-агрегацию alpha- и total-asymmetry подтермов."""

        v7_params = self._v7_params(self._dataset_key(inputs))
        alpha_asymmetry, prov_1, source_1 = self._feature_value(inputs, "alpha_asymmetry", mode, allow_proxy=False)
        left_power, prov_2, source_2 = self._feature_value(inputs, "eeg_left_power", mode, allow_proxy=False)
        right_power, prov_3, source_3 = self._feature_value(inputs, "eeg_right_power", mode, allow_proxy=False)

        terms: list[float] = []
        provenances: list[str] = []
        sources: list[str] = []

        alpha_term = np.nan
        total_term = np.nan

        if np.isfinite(alpha_asymmetry):
            alpha_term = self._robust_tanh(
                alpha_asymmetry,
                VERSION4_CALIBRATION["alpha_asymmetry_center"],
                VERSION4_CALIBRATION["alpha_asymmetry_width"],
            )
            if np.isfinite(alpha_term):
                terms.append(1.00 * alpha_term)
                provenances.append(prov_1)
                sources.append(source_1)

        if np.isfinite(left_power) and np.isfinite(right_power):
            total_asymmetry = (left_power - right_power) / (left_power + right_power + 1e-8)
            total_term = self._robust_tanh(
                total_asymmetry,
                VERSION4_CALIBRATION["total_asymmetry_center"],
                VERSION4_CALIBRATION["total_asymmetry_width"],
            )
            if np.isfinite(total_term):
                terms.append(0.55 * total_term)
                provenances.extend([prov_2, prov_3])
                sources.extend([source_2, source_3])

        if np.isfinite(alpha_term) and np.isfinite(total_term):
            synergy = float(np.sign(alpha_term + total_term) * np.sqrt(abs(alpha_term * total_term) + 1e-9))
            terms.append(0.35 * synergy)

        if not terms:
            return self._component_result(np.nan, "unavailable", "", "A 7.0 недоступен без EEG-асимметрии.")

        raw_value = self._sumigron_scalar(terms, params=v7_params)
        value = float(np.tanh(float(v7_params["component_gain"]) * raw_value))
        return self._component_result(
            value,
            self._merge_provenance(provenances),
            self._merge_sources(sources),
            "Версия 7.0 beta: A строится через Sumigron по alpha- и total-asymmetry подтермам вместо раннего плоского сжатия окна.",
        )

    def calculate_Gamma(self, inputs: dict[str, object], mode: str) -> ComponentResult:
        """Считает Gamma как структурную нагрузку по gamma power и gamma/alpha."""

        v7_params = self._v7_params(self._dataset_key(inputs))
        gamma_power, prov_1, source_1 = self._feature_value(inputs, "eeg_gamma_power", mode, allow_proxy=False)
        gamma_alpha_ratio, prov_2, source_2 = self._feature_value(inputs, "gamma_alpha_ratio", mode, allow_proxy=False)

        terms: list[float] = []
        provenances: list[str] = []
        sources: list[str] = []

        gamma_term = np.nan
        burden_term = np.nan

        if np.isfinite(gamma_power):
            gamma_log = self._calibrated_gamma_log_power(gamma_power)
            gamma_term = self._robust_tanh(
                gamma_log,
                VERSION4_CALIBRATION["gamma_log_center"],
                VERSION4_CALIBRATION["gamma_log_width"],
            )
            if np.isfinite(gamma_term):
                terms.append(0.35 * gamma_term)
                provenances.append(prov_1)
                sources.append(source_1)

        if np.isfinite(gamma_alpha_ratio):
            gamma_alpha_log = float(np.log1p(max(gamma_alpha_ratio, 0.0)))
            burden_term = self._robust_tanh(
                gamma_alpha_log,
                VERSION4_CALIBRATION["gamma_alpha_log_center"],
                VERSION4_CALIBRATION["gamma_alpha_log_width"],
            )
            if np.isfinite(burden_term):
                terms.append(0.75 * burden_term)
                provenances.append(prov_2)
                sources.append(source_2)

        if np.isfinite(gamma_term) and np.isfinite(burden_term):
            synergy = float(np.sign(gamma_term + burden_term) * np.sqrt(abs(gamma_term * burden_term) + 1e-9))
            terms.append(0.20 * synergy)

        if not terms:
            return self._component_result(np.nan, "unavailable", "", "Gamma 7.0 недоступна без gamma-признаков.")

        raw_value = self._sumigron_scalar(terms, params=v7_params)
        value = float(np.tanh(float(v7_params["component_gain"]) * raw_value))
        return self._component_result(
            value,
            self._merge_provenance(provenances),
            self._merge_sources(sources),
            "Версия 7.0 beta: Gamma строится через Sumigron по gamma power и gamma/alpha, сохраняя структуру нагрузки активации.",
        )

    def calculate_V(self, inputs: dict[str, object], mode: str) -> ComponentResult:
        """Считает V как Sumigron-агрегацию HR и HF/LF-подтермов."""

        v7_params = self._v7_params(self._dataset_key(inputs))
        heart_rate, prov_1, source_1 = self._feature_value(inputs, "heart_rate", mode, allow_proxy=True)
        hf_lf_ratio, prov_2, source_2 = self._feature_value(inputs, "hrv_hf_lf", mode, allow_proxy=True)

        if not np.isfinite(heart_rate) and not np.isfinite(hf_lf_ratio):
            return self._component_result(np.nan, "unavailable", "", "V 7.0 недоступен без ECG/HRV.")

        terms: list[float] = []
        provenances: list[str] = []
        sources: list[str] = []

        hr_term = np.nan
        ratio_term = np.nan

        if np.isfinite(heart_rate):
            hr_term = self._robust_tanh(
                VERSION4_CALIBRATION["heart_rate_center"] - heart_rate,
                0.0,
                VERSION4_CALIBRATION["heart_rate_width"],
            )
            if np.isfinite(hr_term):
                terms.append(0.85 * hr_term)
                provenances.append(prov_1)
                sources.append(source_1)

        if np.isfinite(hf_lf_ratio):
            ratio_log = self._calibrated_hrv_ratio(hf_lf_ratio)
            ratio_term = self._robust_tanh(
                ratio_log,
                VERSION4_CALIBRATION["hf_lf_log_center"],
                VERSION4_CALIBRATION["hf_lf_log_width"],
            )
            if np.isfinite(ratio_term):
                terms.append(0.30 * ratio_term)
                provenances.append(prov_2)
                sources.append(source_2)

        if np.isfinite(hr_term) and np.isfinite(ratio_term):
            synergy = float(np.sign(hr_term + ratio_term) * np.sqrt(abs(hr_term * ratio_term) + 1e-9))
            terms.append(0.20 * synergy)

        raw_value = self._sumigron_scalar(terms, params=v7_params)
        value = float(np.tanh(float(v7_params["component_gain"]) * raw_value))
        return self._component_result(
            value,
            self._merge_provenance(provenances),
            self._merge_sources(sources),
            "Версия 7.0 beta: V строится через Sumigron по HR и HF/LF-подтермам, чтобы не терять структурную информацию окна.",
        )

    def calculate_Q(self, inputs: dict[str, object], mode: str) -> ComponentResult:
        """Считает Q как структурную Sumigron-интеграцию A, V и Gamma."""

        v7_params = self._v7_params(self._dataset_key(inputs))
        component_map = {
            "A": self.calculate_A(inputs, mode),
            "Gamma": self.calculate_Gamma(inputs, mode),
            "V": self.calculate_V(inputs, mode),
        }

        a_value = float(component_map["A"].get("value", np.nan))
        gamma_value = float(component_map["Gamma"].get("value", np.nan))
        v_value = float(component_map["V"].get("value", np.nan))
        if not any(np.isfinite(value) for value in (a_value, gamma_value, v_value)):
            return self._component_result(np.nan, "unavailable", "", "Q 7.0 недоступен без базовых EEG/ECG-компонентов.")

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

        q_linear = 0.52 * a_value + 0.44 * v_value - 0.32 * gamma_value
        q_coherence = (
            a_value * v_value
            - float(v7_params["q_coherence_penalty"]) * abs(a_value - v_value)
            - float(v7_params["q_gamma_penalty"]) * max(gamma_value, 0.0)
        )
        q_synergy = float(np.sign(q_linear + q_coherence) * np.sqrt(abs(q_linear * q_coherence) + 1e-9))
        q_raw = self._sumigron_scalar([q_linear, q_coherence, 0.55 * q_synergy], params=v7_params)
        q_value = float(np.clip(0.5 + 0.5 * np.tanh(float(v7_params["q_gain"]) * q_raw), 0.0, 1.0))

        provenance = self._merge_provenance(provenances)
        source = self._merge_sources(sources)
        note = (
            "Версия 7.0 beta: Q строится как Sumigron по линейной, когерентной и синергетической подструктурам A/V/Gamma; "
            "это beta-реализация Sumigron на уровне признаков, а не сырых отсчётов."
        )

        if mode == "proxy":
            valence_norm, valence_source = self._normalize_self_report(inputs, "valence")
            arousal_norm, arousal_source = self._normalize_self_report(inputs, "arousal")
            if np.isfinite(valence_norm) and np.isfinite(arousal_norm):
                q_anchor = float(np.clip(0.65 * valence_norm + 0.35 * (1.0 - arousal_norm), 0.0, 1.0))
                proxy_weight = float(v7_params["q_proxy_weight"])
                q_value = float((1.0 - proxy_weight) * q_value + proxy_weight * q_anchor)
                provenance = "proxy"
                source = self._merge_sources([source, valence_source, arousal_source])
                note = "Версия 7.0 beta: Q как Sumigron-ядро с мягкой proxy-якоризацией по valence/arousal."

        return self._component_result(q_value, provenance, source, note)

    def calculate_IIS(self, components: dict[str, ComponentResult], mode: str) -> dict[str, object]:
        """Считает IIS как gated-смесь режимов, но с Sumigron-сигнатурами."""

        context = components.get("_context", {}) if isinstance(components.get("_context"), dict) else {}
        dataset_key = str(context.get("dataset", "")).lower()
        v7_params = self._v7_params(dataset_key)
        weights = v7_params.get("weights", self.weights)
        prepared = self._prepare_score_inputs(components, mode)
        note = (
            "Версия 7.0 beta: IIS строится как gated mixture of regimes, но сигнатуры режимов и Q "
            "получаются через Sumigron-подобную структурную агрегацию признаковых подтермов с dataset-aware профилем."
        )
        if not prepared.get("can_score"):
            return self._empty_score_payload(prepared, note)

        values = prepared["active_component_values"]
        a_value = float(values.get("A", 0.0))
        gamma_value = float(values.get("Gamma", 0.0))
        v_value = float(values.get("V", 0.0))
        q_value = float(values.get("Q", 0.0))
        shaped = self._shape_components(a_value, gamma_value, v_value, q_value=q_value)

        reg_signature = self._sumigron_scalar(
            [0.72 * shaped["Q"], 0.52 * shaped["V"], 0.18 * shaped["A"], -0.24 * shaped["GammaPos"]],
            params=v7_params,
        )
        mob_signature = self._sumigron_scalar(
            [-0.06 * shaped["Q"], 0.28 * shaped["V"], -0.48 * shaped["GammaPos"], 0.10 * shaped["A"]],
            params=v7_params,
        )
        dep_signature = self._sumigron_scalar(
            [-0.60 * shaped["Q"], -0.50 * shaped["V"], 0.36 * shaped["GammaPos"], -0.20 * shaped["A"]],
            params=v7_params,
        )

        gates = self._softmax(
            [reg_signature, mob_signature, dep_signature],
            float(v7_params["gate_temperature"]),
        )

        synergy_reg = float(np.sign(shaped["Q"] + shaped["V"]) * np.sqrt(abs(shaped["Q"] * shaped["V"]) + 1e-9))
        conflict = abs(shaped["A"] - shaped["V"]) + 0.65 * abs(shaped["Q"] - shaped["V"])

        reg_level = float(v7_params["reg_iis_base"]) + float(v7_params["reg_iis_amp"]) * np.tanh(
            reg_signature + 0.20 * synergy_reg
        )
        mob_level = float(v7_params["mob_iis_base"]) + float(v7_params["mob_iis_amp"]) * np.tanh(mob_signature)
        dep_level = float(v7_params["dep_iis_base"]) + float(v7_params["dep_iis_amp"]) * np.tanh(dep_signature)

        regime_score = float(gates[0] * reg_level + gates[1] * mob_level + gates[2] * dep_level)
        entropy = float(-np.sum(gates * np.log(gates + 1e-9)) / np.log(3.0))
        regime_balance = float(gates[0] - gates[2])
        iis_score = regime_score
        iis_score -= float(v7_params["transition_entropy_weight"]) * entropy
        iis_score += float(v7_params["regime_balance_weight"]) * regime_balance
        iis_score -= float(v7_params["conflict_penalty"]) * conflict
        iis_score = self._clip_score(iis_score)

        raw_score = (
            weights.get("A", 0.0) * a_value
            - weights.get("Gamma", 0.0) * gamma_value
            + weights.get("V", 0.0) * v_value
            + weights.get("Q", 0.0) * q_value
        )
        contributions = {
            "A": weights.get("A", 0.0) * a_value,
            "Gamma": -weights.get("Gamma", 0.0) * gamma_value,
            "V": weights.get("V", 0.0) * v_value,
            "Q": weights.get("Q", 0.0) * q_value,
        }
        return self._finalize_score_payload(prepared, raw_score, iis_score, contributions, note)
