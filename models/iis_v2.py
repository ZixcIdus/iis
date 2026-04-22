"""Версия ИИС 1.0 из основного документа."""

from __future__ import annotations

import numpy as np

from config.settings import EPSILON, VERSION3_K_PARAMS
from models.base_model import BaseIISModel, ComponentResult


class IISVersion2(BaseIISModel):
    """Версия 1.0 с той же базовой интеграцией, но A считается по общей мощности полушарий."""

    def calculate_A(self, inputs: dict[str, object], mode: str) -> ComponentResult:
        """Считает A по общей лево-правой мощности ЭЭГ."""

        left_power, prov_1, source_1 = self._feature_value(inputs, "eeg_left_power", mode, allow_proxy=False)
        right_power, prov_2, source_2 = self._feature_value(inputs, "eeg_right_power", mode, allow_proxy=False)
        if not np.isfinite(left_power) or not np.isfinite(right_power):
            return self._component_result(np.nan, "unavailable", "", "A недоступен без общей мощности левого и правого полушария.")

        value = (left_power - right_power) / (left_power + right_power + EPSILON)
        provenance = self._merge_provenance([prov_1, prov_2])
        source = self._merge_sources([source_1, source_2])
        return self._component_result(value, provenance, source, "Формула версии 1.0: A=(NL-NR)/(NL+NR).")

    def calculate_Gamma(self, inputs: dict[str, object], mode: str) -> ComponentResult:
        """Считает Γ = ln(1 + Pγ)."""

        gamma_power, provenance, source = self._feature_value(inputs, "eeg_gamma_power", mode, allow_proxy=False)
        if not np.isfinite(gamma_power):
            return self._component_result(np.nan, "unavailable", "", "Gamma недоступна без мощности γ-ритма.")

        value = self._calibrated_gamma_log_power(gamma_power)
        return self._component_result(
            value,
            provenance,
            source,
            "Калиброванная реализация для открытых данных: Γ=ln(1+Pγ·10^12), где мощность переведена из В² в мкВ².",
        )

    def calculate_H(self, inputs: dict[str, object], mode: str) -> ComponentResult:
        """Считает H = DA/50 - C/15."""

        dopamine, prov_1, source_1 = self._proxy_dopamine(inputs, mode)
        cortisol, prov_2, source_2 = self._proxy_cortisol(inputs, mode)
        if not np.isfinite(dopamine) or not np.isfinite(cortisol):
            return self._component_result(np.nan, "unavailable", "", "H требует дофамин и кортизол; на датасетах доступен только proxy-режим.")

        value = (dopamine / 50.0) - (cortisol / 15.0)
        provenance = self._merge_provenance([prov_1, prov_2])
        source = self._merge_sources([source_1, source_2])
        return self._component_result(value, provenance, source, "Формула версии 1.0: H=DA/50-C/15.")

    def calculate_V(self, inputs: dict[str, object], mode: str) -> ComponentResult:
        """Считает V = HF/LF."""

        hf, prov_1, source_1 = self._feature_value(inputs, "hrv_hf", mode, allow_proxy=True)
        lf, prov_2, source_2 = self._feature_value(inputs, "hrv_lf", mode, allow_proxy=True)
        ratio, prov_3, source_3 = self._feature_value(inputs, "hrv_hf_lf", mode, allow_proxy=True)

        if np.isfinite(hf) and np.isfinite(lf):
            value = self._calibrated_hrv_ratio(hf / (lf + EPSILON))
            provenance = self._merge_provenance([prov_1, prov_2])
            source = self._merge_sources([source_1, source_2])
            return self._component_result(
                value,
                provenance,
                source,
                "Калиброванная реализация для открытых данных: V=ln(1+HF/LF), чтобы редкие выбросы HRV не насыщали IIS.",
            )
        if np.isfinite(ratio):
            value = self._calibrated_hrv_ratio(ratio)
            return self._component_result(
                value,
                prov_3,
                source_3,
                "V получен из готового отношения HF/LF и переведён в калиброванную форму ln(1+HF/LF).",
            )
        return self._component_result(np.nan, "unavailable", "", "V недоступен без HRV.")

    def calculate_Q(self, inputs: dict[str, object], mode: str) -> ComponentResult:
        """Считает диагностический Q для сравнения с valence."""

        alpha_left, prov_1, source_1 = self._feature_value(inputs, "eeg_alpha_left", mode, allow_proxy=False)
        alpha_right, prov_2, source_2 = self._feature_value(inputs, "eeg_alpha_right", mode, allow_proxy=False)
        if not np.isfinite(alpha_left) or not np.isfinite(alpha_right):
            return self._component_result(np.nan, "unavailable", "", "Q недоступен без α-мощности слева и справа.")

        q_raw = (alpha_left - alpha_right) / (alpha_left + alpha_right + EPSILON)
        value = float(np.tanh(VERSION3_K_PARAMS["valence_beta"] * q_raw))
        provenance = self._merge_provenance([prov_1, prov_2])
        source = self._merge_sources([source_1, source_2])
        return self._component_result(value, provenance, source, "Диагностический Q добавлен для сопоставления с валентностью; в ИИС версии 2 не входит.")

    def calculate_K(
        self,
        inputs: dict[str, object],
        mode: str,
        components: dict[str, ComponentResult],
    ) -> ComponentResult:
        """В версии 1.0 коэффициент K отсутствовал."""

        return self._component_result(np.nan, "unavailable", "", "В версии 1.0 документа коэффициент K отсутствует.")

    def calculate_IIS(self, components: dict[str, ComponentResult], mode: str) -> dict[str, object]:
        """Считает итоговый ИИС по формуле версии 1.0."""

        prepared = self._prepare_score_inputs(components, mode)
        note = "Версия 1.0 документа: ИИС=σ(0.35A+0.25Γ+0.30H+0.10V)."
        if not prepared.get("can_score"):
            return self._empty_score_payload(prepared, note)

        values = prepared["active_component_values"]
        weights = prepared["normalized_weights"]
        contributions = {name: weights[name] * values[name] for name in values}
        raw_score = float(sum(contributions.values()))
        iis_score = self._safe_sigmoid(raw_score)
        return self._finalize_score_payload(prepared, raw_score, iis_score, contributions, note)
