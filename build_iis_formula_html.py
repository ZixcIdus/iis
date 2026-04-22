"""Генерирует автономный HTML с расширенными формулами IISVersion1–IISVersion7."""

from __future__ import annotations

import html
import json
import math
from pathlib import Path
from typing import Any

import pandas as pd

from config.settings import VERSION6_CALIBRATION, VERSION7_CALIBRATION, VERSION7_DATASET_OVERRIDES
from reporting.iis_report_common import VERSION_METADATA, VERSION_ORDER, VERSION_SHORT

PROJECT_ROOT = Path(__file__).resolve().parent
DOCS_DIR = PROJECT_ROOT / "docs"
OUTPUT_HTML = DOCS_DIR / "iis_formula_expanded.html"
OUTPUT_HTML_MOCHA = DOCS_DIR / "iis_formula_expanded_mocha.html"

EXAMPLE_SOURCES: dict[str, dict[str, str]] = {
    "IISVersion1": {
        "path": str(PROJECT_ROOT / "outputs" / "results_ds002722_proxy.csv"),
        "scenario": "ds002722 / proxy",
    },
    "IISVersion2": {
        "path": str(PROJECT_ROOT / "outputs" / "results_ds002724_proxy.csv"),
        "scenario": "ds002724 / proxy",
    },
    "IISVersion3": {
        "path": str(PROJECT_ROOT / "outputs" / "results_ds002722_proxy.csv"),
        "scenario": "ds002722 / proxy",
    },
    "IISVersion4": {
        "path": str(PROJECT_ROOT / "outputs" / "results_ds002724_strict.csv"),
        "scenario": "ds002724 / strict",
    },
    "IISVersion5": {
        "path": str(PROJECT_ROOT / "outputs" / "results_ds002724_strict.csv"),
        "scenario": "ds002724 / strict",
    },
    "IISVersion6": {
        "path": str(PROJECT_ROOT / "outputs" / "results_ds002724_strict.csv"),
        "scenario": "ds002724 / strict",
    },
    "IISVersion7": {
        "path": str(PROJECT_ROOT / "outputs" / "results_dreamer_strict.csv"),
        "scenario": "dreamer / strict",
    },
}

FORMULA_SECTIONS: dict[str, dict[str, Any]] = {
    "IISVersion1": {
        "headline": "Ранняя четырёхкомпонентная proxy-модель",
        "formula_blocks": [
            (
                "Итоговый индекс",
                """
IIS_1 = σ(0.35·A + 0.25·Γ + 0.30·H + 0.10·V)
σ(x) = 1 / (1 + e^(-x))
                """.strip(),
            ),
            (
                "Компоненты",
                """
A = (α_L - α_R) / (α_L + α_R + ε)
Γ = ln(1 + P_γ · 10^12)
H = DA / 50 - C / 15
V = ln(1 + HF / LF)
Q_raw = (α_L - α_R) / (α_L + α_R + ε)
Q = tanh(2.0 · Q_raw)
                """.strip(),
            ),
        ],
        "example_mode": "direct_linear",
        "notes": [
            "Полная формула простая: итог получается прямой линейной суммой четырёх блоков с последующей сигмоидой.",
            "На открытых данных блок H в этой версии обычно proxy-оценочный, а Q считается как диагностический, но не входит в IIS.",
        ],
    },
    "IISVersion2": {
        "headline": "Историческая версия с другой осью A",
        "formula_blocks": [
            (
                "Итоговый индекс",
                """
IIS_2 = σ(0.35·A + 0.25·Γ + 0.30·H + 0.10·V)
                """.strip(),
            ),
            (
                "Изменение относительно V1",
                """
A = (N_L - N_R) / (N_L + N_R + ε)
Γ, H, V и диагностический Q совпадают с V1
                """.strip(),
            ),
        ],
        "example_mode": "direct_linear",
        "notes": [
            "По структуре это та же интеграция, что и V1; меняется только определение A.",
            "Версия историческая и удобна в HTML как контраст к более поздним калиброванным моделям.",
        ],
    },
    "IISVersion3": {
        "headline": "Историческая версия с коэффициентом K",
        "formula_blocks": [
            (
                "Итоговый индекс",
                """
IIS_3 = σ(0.35·A + 0.25·Γ + 0.30·H + 0.10·V)
H     = K·DA / 50 - C / 15
                """.strip(),
            ),
            (
                "Коэффициент K",
                """
K = (R · G) / (1 + α·S² + β·F(S))
F(S) = 1 / (1 + exp(-k_f·(S - S_0)))

Рабочие константы:
α = 1.2, β = 0.24, k_f = 5.0, S_0 = 0.6
                """.strip(),
            ),
            (
                "Компоненты",
                """
A = (N_L - N_R) / (N_L + N_R + ε)
Γ = ln(1 + P_γ · 10^12)
V = ln(1 + HF / LF)
Q = tanh(2.0 · Q_raw)
                """.strip(),
            ),
        ],
        "example_mode": "direct_linear_with_k",
        "notes": [
            "В результирующих CSV уже сохраняется готовый H; прямые DA и C в открытых данных отдельно не восстанавливаются.",
            "Поэтому для числовой подстановки ниже показан честный внешний слой и уже полученный H.",
        ],
    },
    "IISVersion4": {
        "headline": "Первое негормональное практическое ядро",
        "formula_blocks": [
            (
                "Калибровочная нормировка",
                """
T(x; c, w) = tanh((x - c) / w)
                """.strip(),
            ),
            (
                "Компоненты",
                """
A_α     = T(α_asym; c_α, w_α)
A_total = T((N_L - N_R) / (N_L + N_R + ε); c_N, w_N)
A       = tanh(1.10·A_α + 0.40·A_total)

G_log     = T(ln(1 + P_γ · 10^12); c_γ, w_γ)
G_γ/α     = T(ln(1 + γ/α); c_γ/α, w_γ/α)
Γ         = tanh(0.30·G_log + 0.70·G_γ/α)

V_hr      = T(HR; c_HR, w_HR)
V_hf/lf   = T(ln(1 + HF / LF); c_HRV, w_HRV)
V         = tanh(0.45·V_hr + 0.55·V_hf/lf)
                """.strip(),
            ),
            (
                "Интегральное качество и итог",
                """
Q_4   = tanh(0.55·A + 0.45·V - 0.25·Γ)
IIS_4 = σ(0.10·A - 0.05·Γ + 0.25·V + 0.60·Q_4)
                """.strip(),
            ),
        ],
        "example_mode": "v4_exact",
        "notes": [
            "Это первая версия, которую имеет смысл показывать как практическую основу strict-режима.",
            "Для V4 ниже даётся уже точная числовая подстановка в Q_4 и IIS_4.",
        ],
    },
    "IISVersion5": {
        "headline": "Контрастная нелинейная версия",
        "formula_blocks": [
            (
                "Компонент Q_5",
                """
Q_lin = 0.50·A + 0.42·V - 0.30·Γ
Q_coh = A·V - 0.50·|A - V| - 0.18·Γ_+

Â = φ_0.82(A)
V̂ = φ_0.82(V)
Γ̂ = φ_1.05(Γ)

S_AV = sign(Â + V̂)·sqrt(|Â·V̂| + ε)
E_AV = (|Â| + |V̂|) / 2
C_AV = |Â - V̂| + 0.16·Γ̂_+·(1 - V̂_+)

Q_5 = tanh(
  1.40·Q_lin
  + 0.55·Q_coh
  + 0.42·S_AV
  + 0.28·E_AV·sign(Q_lin)
  - 0.24·C_AV
)
                """.strip(),
            ),
            (
                "Итоговый индекс",
                """
core_5     = 0.12·A - 0.07·Γ + 0.24·V + 0.57·Q_5
z_5        = 0.14·Ã - 0.08·Γ + 0.22·Ṽ + 0.56·Q̃ - 0.08
IIS_5,base = σ(core_5 + 0.22·S_reg - 0.10·C_phase)
IIS_5,ctr  = 0.5 + 0.5·tanh(contrast_5)
IIS_5      = clip(0.65·IIS_5,base + 0.35·IIS_5,ctr, 0, 1)
                """.strip(),
            ),
        ],
        "example_mode": "engine_trace",
        "notes": [
            "Формула уже многоуровневая: в результирующем CSV сохраняются компоненты A/Γ/V/Q и итоговый internal raw_score.",
            "Ниже показан честный engine trace: какие именно компонентные вклады дали итоговый internal scalar перед финальным bounded-выходом.",
        ],
    },
    "IISVersion6": {
        "headline": "Gated-модель режимов",
        "formula_blocks": [
            (
                "Гейты и локальные уровни для Q_6",
                """
Â = φ_0.78(A), V̂ = φ_0.78(V), Γ̂ = φ_1.05(Γ), Q̂ = φ_0.78(Q_6)
Γ̂_+ = max(Γ̂, 0)

z_reg^(Q) =  0.95·Â + 1.05·V̂ - 0.55·Γ̂_+
z_mob^(Q) = -0.18·Â + 0.48·V̂ - 1.05·Γ̂_+
z_dep^(Q) = -0.80·Â - 0.95·V̂ + 0.72·Γ̂_+

(g_reg^(Q), g_mob^(Q), g_dep^(Q)) = softmax(
  1.65·z_reg^(Q), 1.65·z_mob^(Q), 1.65·z_dep^(Q)
)

Q_6 = clip(g_reg^(Q)·q_reg + g_mob^(Q)·q_mob + g_dep^(Q)·q_dep, 0, 1)
                """.strip(),
            ),
            (
                "Итоговый gated IIS",
                """
z_reg =  0.60·Q̂ + 0.45·V̂ + 0.15·Â - 0.30·Γ̂_+
z_mob = -0.05·Q̂ + 0.20·V̂ - 0.55·Γ̂_+ - 0.08·Â
z_dep = -0.70·Q̂ - 0.55·V̂ - 0.20·Â + 0.35·Γ̂_+

(g_reg, g_mob, g_dep) = softmax(1.65·z_reg, 1.65·z_mob, 1.65·z_dep)

R_6  = g_reg·i_reg + g_mob·i_mob + g_dep·i_dep
H(g) = -(g_reg ln g_reg + g_mob ln g_mob + g_dep ln g_dep) / ln 3
B_6  = g_reg - g_dep

IIS_6 = clip(R_6 - 0.08·H(g) + 0.05·B_6 - 0.10·C_QV^(6), 0, 1)
                """.strip(),
            ),
        ],
        "example_mode": "engine_trace",
        "notes": [
            "V6 уже лучше читать как систему режимов, а не как одну плоскую формулу.",
            "Ниже поэтому показан и symbolic block, и реальный engine trace на одном и том же сегменте, что и у V4/V5.",
        ],
    },
    "IISVersion7": {
        "headline": "Экспериментальная Sumigron-бета",
        "formula_blocks": [
            (
                "Оператор Sumigron",
                """
sgop(W_i,c) = w_μ·μ̃_i,c^sg + w_σ·σ̃_i,c^sg + w_E·Ẽ_i,c^sg

μ_i,c      = (1/L_i)·Σ_t x_i,c,t
σ_i,c      = sqrt((1/L_i)·Σ_t (x_i,c,t - μ_i,c)^2)
x̂_i,c,t   = (x_i,c,t - μ_i,c) / (σ_i,c + ε)
u_i,c,t    = 0.60·x̂_i,c,t + 0.40·|x̂_i,c,t|
α_i,c,t    = exp(u_i,c,t / τ) / Σ_s exp(u_i,c,s / τ)
μ_i,c^sg   = Σ_t α_i,c,t·x_i,c,t
σ_i,c^sg   = sqrt(Σ_t α_i,c,t·(x_i,c,t - μ_i,c^sg)^2)
E_i,c^sg   = log(1 + Σ_t α_i,c,t·x_i,c,t²)

Базовые веса:
w_μ = 0.32, w_σ = 0.28, w_E = 0.40
                """.strip(),
            ),
            (
                "Компоненты и gated-слой V7",
                """
A_7 = tanh(a_1·sgop(W_i^α asym) + a_2·sgop(W_i^total asym))
Γ_7 = tanh(g_1·sgop(W_i^γ) + g_2·sgop(W_i^γ/α))
V_7 = tanh(v_1·sgop(W_i^HR) + v_2·sgop(W_i^HF/LF))

Q_7^lin = q_1·A_7 + q_2·V_7 - q_3·Γ_7
Q_7^coh = A_7·V_7 - λ_1·|A_7 - V_7| - λ_2·max(Γ_7, 0)
Q_7     = tanh(η_1·Q_7^lin + η_2·Q_7^coh)

(g_7^reg, g_7^mob, g_7^dep) = softmax(z_7^reg, z_7^mob, z_7^dep)
IIS_7 = g_7^reg·s_7^reg + g_7^mob·s_7^mob + g_7^dep·s_7^dep

RES_7 = clip(0.5 + 0.5·tanh(ρ_1·V_7 - ρ_2·max(Γ_7,0) - ρ_3·|A_7 - V_7|), 0, 1)
DYN_7 = tanh(δ_1·ΔIIS_7 + δ_2·ΔRES_7 - δ_3·Vol_i)
                """.strip(),
            ),
        ],
        "example_mode": "engine_trace_with_res",
        "notes": [
            "Это уже beta-линия: полноценная symbolic форма дана полностью, а реальный пример берётся из DREAMER strict, где для V7 используется dataset-aware override.",
            "Ниже поэтому показан и trace по компонентам, и дополнительные выходы RES/state map.",
        ],
    },
}

USECOLS = [
    "subject_id",
    "segment_id",
    "dataset",
    "label",
    "version",
    "mode",
    "IIS",
    "raw_score",
    "coverage_ratio",
    "active_component_count",
    "formula_note",
    "score_explanation_json",
    "contributions_json",
    "A",
    "Gamma",
    "H",
    "V",
    "Q",
    "K",
    "RES",
    "RES_core",
    "RES_mismatch",
    "RES_gamma_load",
    "state_map_4",
    "state_map_margin",
]


def _fmt(value: Any, digits: int = 4) -> str:
    if value is None:
        return "—"
    if isinstance(value, str):
        return value
    try:
        if pd.isna(value):
            return "—"
    except Exception:
        pass
    if isinstance(value, (int,)):
        return str(value)
    try:
        return f"{float(value):.{digits}f}"
    except Exception:
        return str(value)


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def _clip01(x: float) -> float:
    return max(0.0, min(1.0, x))


def _signed_power(value: float, power: float) -> float:
    if not math.isfinite(value):
        return float("nan")
    return math.copysign(abs(value) ** power, value)


def _softmax(logits: list[float], temperature: float) -> list[float]:
    finite = [float(x) for x in logits]
    if not any(math.isfinite(x) for x in finite):
        return [1.0 / max(len(finite), 1)] * len(finite)
    max_value = max(finite)
    centered = [x - max_value for x in finite]
    scaled = [max(-50.0, min(50.0, x * max(temperature, 1e-6))) for x in centered]
    exp_values = [math.exp(x) for x in scaled]
    total = sum(exp_values)
    if total <= 0.0 or not math.isfinite(total):
        return [1.0 / max(len(finite), 1)] * len(finite)
    return [x / total for x in exp_values]


def _bounded_positive_gate(value: float) -> float:
    if not math.isfinite(value):
        return float("nan")
    positive = max(value, 0.0)
    return positive / (1.0 + positive)


def _merge_v7_params(dataset_key: str) -> dict[str, Any]:
    params: dict[str, Any] = dict(VERSION7_CALIBRATION)
    params["weights"] = dict(VERSION7_CALIBRATION.get("weights", {}))
    override = VERSION7_DATASET_OVERRIDES.get(dataset_key.lower(), {})
    for key, value in override.items():
        if key == "weights" and isinstance(value, dict):
            merged_weights = dict(params.get("weights", {}))
            merged_weights.update(value)
            params["weights"] = merged_weights
        else:
            params[key] = value
    return params


def _load_jsonish(value: Any) -> Any:
    if not isinstance(value, str) or not value.strip():
        return {}
    try:
        return json.loads(value)
    except Exception:
        return {}


def _pick_example_row(csv_path: str, version: str) -> dict[str, Any]:
    best_row: dict[str, Any] | None = None
    best_key: tuple[float, float, float] | None = None
    for chunk in pd.read_csv(csv_path, usecols=USECOLS, chunksize=50000, low_memory=False):
        subset = chunk[chunk["version"] == version]
        if subset.empty:
            continue
        for _, row in subset.iterrows():
            key = (
                float(row.get("coverage_ratio") or 0.0),
                float(row.get("active_component_count") or 0.0),
                float(row.get("IIS") or 0.0),
            )
            if best_key is None or key > best_key:
                best_key = key
                best_row = row.to_dict()
    if best_row is None:
        raise ValueError(f"Не найдена строка версии {version} в {csv_path}")
    best_row["contributions_json"] = _load_jsonish(best_row.get("contributions_json"))
    best_row["score_explanation_json"] = _load_jsonish(best_row.get("score_explanation_json"))
    return best_row


def _build_metadata_table(row: dict[str, Any], scenario: str) -> str:
    meta = [
        ("Сценарий", scenario),
        ("Датасет", row.get("dataset")),
        ("Метка", row.get("label")),
        ("Субъект", row.get("subject_id")),
        ("Сегмент", row.get("segment_id")),
        ("Покрытие", _fmt(row.get("coverage_ratio"))),
        ("Активных блоков", _fmt(row.get("active_component_count"), digits=0)),
        ("IIS", _fmt(row.get("IIS"))),
        ("raw_score", _fmt(row.get("raw_score"))),
    ]
    cells = "".join(
        f"<tr><th>{html.escape(str(label))}</th><td>{html.escape(str(value))}</td></tr>"
        for label, value in meta
    )
    return f'<table class="meta-table">{cells}</table>'


def _build_component_table(row: dict[str, Any]) -> str:
    comps = ["A", "Gamma", "H", "V", "Q", "K", "RES"]
    rows = "".join(
        f"<tr><th>{html.escape(name)}</th><td>{html.escape(_fmt(row.get(name)))}</td></tr>"
        for name in comps
    )
    return f'<table class="component-table">{rows}</table>'


def _build_contrib_table(contrib: dict[str, Any]) -> str:
    if not contrib:
        return "<p class='muted'>Разложение вкладов не сохранено.</p>"
    rows = "".join(
        f"<tr><th>{html.escape(str(key))}</th><td>{html.escape(_fmt(value))}</td></tr>"
        for key, value in contrib.items()
    )
    total = sum(float(value) for value in contrib.values())
    rows += f"<tr class='total'><th>Σ вкладов</th><td>{html.escape(_fmt(total))}</td></tr>"
    return f'<table class="contrib-table">{rows}</table>'


def _sumigron_trace(values: list[float], temperature: float, structure_weight: float, energy_weight: float, level_weight: float = 1.0) -> dict[str, Any]:
    finite = [float(value) for value in values if math.isfinite(value)]
    if not finite:
        return {
            "values": [],
            "local_mean": float("nan"),
            "local_std": float("nan"),
            "local_z": [],
            "drive": [],
            "weights": [],
            "attentive_mean": float("nan"),
            "attentive_std": float("nan"),
            "attentive_energy": float("nan"),
            "structured_boost": float("nan"),
            "scalar": float("nan"),
        }
    if len(finite) == 1:
        return {
            "values": finite,
            "local_mean": finite[0],
            "local_std": 0.0,
            "local_z": [0.0],
            "drive": [0.0],
            "weights": [1.0],
            "attentive_mean": finite[0],
            "attentive_std": 0.0,
            "attentive_energy": math.log1p(finite[0] ** 2),
            "structured_boost": 0.0,
            "scalar": finite[0],
        }

    local_mean = sum(finite) / len(finite)
    local_std = math.sqrt(sum((x - local_mean) ** 2 for x in finite) / len(finite))
    local_z = [(x - local_mean) / (local_std + 1e-6) for x in finite]
    drive = [0.60 * z + 0.40 * abs(z) for z in local_z]
    weights = _softmax(drive, temperature)
    attentive_mean = sum(w * x for w, x in zip(weights, finite))
    attentive_std = math.sqrt(sum(w * (x - attentive_mean) ** 2 for w, x in zip(weights, finite)))
    attentive_energy = math.log1p(sum(w * (x**2) for w, x in zip(weights, finite)))
    sign = 1.0
    if attentive_mean != 0.0:
        sign = math.copysign(1.0, attentive_mean)
    else:
        total = sum(finite)
        if total != 0.0:
            sign = math.copysign(1.0, total)
    structured_boost = sign * (
        structure_weight * _bounded_positive_gate(attentive_std)
        + energy_weight * _bounded_positive_gate(attentive_energy)
    )
    scalar = level_weight * attentive_mean + structured_boost
    return {
        "values": finite,
        "local_mean": local_mean,
        "local_std": local_std,
        "local_z": local_z,
        "drive": drive,
        "weights": weights,
        "attentive_mean": attentive_mean,
        "attentive_std": attentive_std,
        "attentive_energy": attentive_energy,
        "structured_boost": structured_boost,
        "scalar": scalar,
    }


def _build_sumigron_trace_html(title: str, labels: list[str], trace: dict[str, Any], scalar_label: str) -> str:
    rows = []
    for idx, value in enumerate(trace["values"]):
        label = labels[idx] if idx < len(labels) else f"t{idx+1}"
        rows.append(
            f"<tr><th>{html.escape(label)}</th><td>{html.escape(_fmt(value))}</td>"
            f"<td>{html.escape(_fmt(trace['local_z'][idx]))}</td>"
            f"<td>{html.escape(_fmt(trace['drive'][idx]))}</td>"
            f"<td>{html.escape(_fmt(trace['weights'][idx]))}</td></tr>"
        )
    meta = f"""
    <div class="trace-meta">
      <p><strong>local_mean:</strong> {_fmt(trace['local_mean'])}</p>
      <p><strong>local_std:</strong> {_fmt(trace['local_std'])}</p>
      <p><strong>attentive_mean:</strong> {_fmt(trace['attentive_mean'])}</p>
      <p><strong>attentive_std:</strong> {_fmt(trace['attentive_std'])}</p>
      <p><strong>attentive_energy:</strong> {_fmt(trace['attentive_energy'])}</p>
      <p><strong>structured_boost:</strong> {_fmt(trace['structured_boost'])}</p>
      <p><strong>{html.escape(scalar_label)}:</strong> {_fmt(trace['scalar'])}</p>
    </div>
    """
    return f"""
    <article class="example-card">
      <h3>{html.escape(title)}</h3>
      <table class="trace-table">
        <thead>
          <tr><th>Терм</th><th>value</th><th>z</th><th>drive</th><th>softmax weight</th></tr>
        </thead>
        <tbody>{''.join(rows)}</tbody>
      </table>
      {meta}
    </article>
    """


def _v6_gate_trace(row: dict[str, Any]) -> dict[str, Any] | None:
    try:
        a = float(row["A"])
        g = float(row["Gamma"])
        v = float(row["V"])
        q = float(row["Q"])
    except Exception:
        return None

    shape_power = float(VERSION6_CALIBRATION["shape_power"])
    gamma_power = float(VERSION6_CALIBRATION["gamma_shape_power"])
    shaped_a = _signed_power(a, shape_power)
    shaped_g = _signed_power(g, gamma_power)
    shaped_v = _signed_power(v, shape_power)
    shaped_q = _signed_power(q, shape_power)
    gamma_pos = max(shaped_g, 0.0)

    q_logits = [
        0.95 * shaped_a + 1.05 * shaped_v - 0.55 * gamma_pos,
        -0.18 * shaped_a + 0.48 * shaped_v - 1.05 * gamma_pos,
        -0.80 * shaped_a - 0.95 * shaped_v + 0.72 * gamma_pos,
    ]
    q_gates = _softmax(q_logits, float(VERSION6_CALIBRATION["gate_temperature"]))
    q_synergy = math.copysign(1.0, shaped_a + shaped_v) * math.sqrt(abs(shaped_a * shaped_v) + 1e-9)
    q_reg = float(VERSION6_CALIBRATION["reg_q_base"]) + float(VERSION6_CALIBRATION["reg_q_amp"]) * math.tanh(
        1.10 * shaped_a + 1.15 * shaped_v - 0.45 * gamma_pos + 0.30 * q_synergy
    )
    q_mob = float(VERSION6_CALIBRATION["mob_q_base"]) + float(VERSION6_CALIBRATION["mob_q_amp"]) * math.tanh(
        0.10 * shaped_a + 0.55 * shaped_v - 1.20 * gamma_pos
    )
    q_dep = float(VERSION6_CALIBRATION["dep_q_base"]) + float(VERSION6_CALIBRATION["dep_q_amp"]) * math.tanh(
        -0.65 * abs(shaped_a) - 0.95 * abs(shaped_v) + 0.95 * gamma_pos
    )
    q_mix = _clip01(q_gates[0] * q_reg + q_gates[1] * q_mob + q_gates[2] * q_dep)

    iis_logits = [
        0.60 * shaped_q + 0.45 * shaped_v + 0.15 * shaped_a - 0.30 * gamma_pos,
        -0.05 * shaped_q + 0.20 * shaped_v - 0.55 * gamma_pos - 0.08 * shaped_a,
        -0.70 * shaped_q - 0.55 * shaped_v - 0.20 * shaped_a + 0.35 * gamma_pos,
    ]
    iis_gates = _softmax(iis_logits, float(VERSION6_CALIBRATION["gate_temperature"]))
    synergy_reg = math.copysign(1.0, shaped_q + shaped_v) * math.sqrt(abs(shaped_q * shaped_v) + 1e-9)
    conflict = abs(shaped_a - shaped_v) + 0.75 * abs(shaped_q - shaped_v)
    reg_level = float(VERSION6_CALIBRATION["reg_iis_base"]) + float(VERSION6_CALIBRATION["reg_iis_amp"]) * math.tanh(
        0.75 * shaped_q + 0.55 * shaped_v + 0.18 * shaped_a - 0.25 * gamma_pos + 0.25 * synergy_reg
    )
    mob_level = float(VERSION6_CALIBRATION["mob_iis_base"]) + float(VERSION6_CALIBRATION["mob_iis_amp"]) * math.tanh(
        -0.08 * shaped_q + 0.32 * shaped_v - 0.52 * gamma_pos + 0.08 * shaped_a
    )
    dep_level = float(VERSION6_CALIBRATION["dep_iis_base"]) + float(VERSION6_CALIBRATION["dep_iis_amp"]) * math.tanh(
        -0.62 * shaped_q - 0.52 * shaped_v - 0.18 * shaped_a + 0.38 * gamma_pos
    )
    regime_score = iis_gates[0] * reg_level + iis_gates[1] * mob_level + iis_gates[2] * dep_level
    entropy = -(sum(x * math.log(x + 1e-9) for x in iis_gates)) / math.log(3.0)
    balance = iis_gates[0] - iis_gates[2]
    iis_final = _clip01(
        regime_score
        - float(VERSION6_CALIBRATION["transition_entropy_weight"]) * entropy
        + float(VERSION6_CALIBRATION["regime_balance_weight"]) * balance
        - float(VERSION6_CALIBRATION["conflict_penalty"]) * conflict
    )
    return {
        "shaped": {"A": shaped_a, "Gamma": shaped_g, "GammaPos": gamma_pos, "V": shaped_v, "Q": shaped_q},
        "q_logits": q_logits,
        "q_gates": q_gates,
        "q_levels": {"reg": q_reg, "mob": q_mob, "dep": q_dep},
        "q_mix": q_mix,
        "iis_logits": iis_logits,
        "iis_gates": iis_gates,
        "iis_levels": {"reg": reg_level, "mob": mob_level, "dep": dep_level},
        "conflict": conflict,
        "entropy": entropy,
        "balance": balance,
        "regime_score": regime_score,
        "iis_final": iis_final,
    }


def _v7_gate_trace(row: dict[str, Any]) -> dict[str, Any] | None:
    try:
        a = float(row["A"])
        g = float(row["Gamma"])
        v = float(row["V"])
        q = float(row["Q"])
        dataset = str(row.get("dataset", "")).lower()
    except Exception:
        return None

    params = _merge_v7_params(dataset)
    shape_power = float(VERSION6_CALIBRATION["shape_power"])
    gamma_power = float(VERSION6_CALIBRATION["gamma_shape_power"])
    shaped_a = _signed_power(a, shape_power)
    shaped_g = _signed_power(g, gamma_power)
    shaped_v = _signed_power(v, shape_power)
    shaped_q = _signed_power(q, shape_power)
    gamma_pos = max(shaped_g, 0.0)

    q_linear = 0.52 * a + 0.44 * v - 0.32 * g
    q_coherence = a * v - float(params["q_coherence_penalty"]) * abs(a - v) - float(params["q_gamma_penalty"]) * max(g, 0.0)
    q_synergy = math.copysign(1.0, q_linear + q_coherence) * math.sqrt(abs(q_linear * q_coherence) + 1e-9)
    q_trace = _sumigron_trace(
        [q_linear, q_coherence, 0.55 * q_synergy],
        float(params["sumigron_temperature"]),
        float(params["sumigron_structure_weight"]),
        float(params["sumigron_energy_weight"]),
        float(params["sumigron_level_weight"]),
    )
    q_value = _clip01(0.5 + 0.5 * math.tanh(float(params["q_gain"]) * q_trace["scalar"]))

    reg_terms = [0.72 * shaped_q, 0.52 * shaped_v, 0.18 * shaped_a, -0.24 * gamma_pos]
    mob_terms = [-0.06 * shaped_q, 0.28 * shaped_v, -0.48 * gamma_pos, 0.10 * shaped_a]
    dep_terms = [-0.60 * shaped_q, -0.50 * shaped_v, 0.36 * gamma_pos, -0.20 * shaped_a]
    reg_trace = _sumigron_trace(reg_terms, float(params["sumigron_temperature"]), float(params["sumigron_structure_weight"]), float(params["sumigron_energy_weight"]), float(params["sumigron_level_weight"]))
    mob_trace = _sumigron_trace(mob_terms, float(params["sumigron_temperature"]), float(params["sumigron_structure_weight"]), float(params["sumigron_energy_weight"]), float(params["sumigron_level_weight"]))
    dep_trace = _sumigron_trace(dep_terms, float(params["sumigron_temperature"]), float(params["sumigron_structure_weight"]), float(params["sumigron_energy_weight"]), float(params["sumigron_level_weight"]))
    gates = _softmax([reg_trace["scalar"], mob_trace["scalar"], dep_trace["scalar"]], float(params["gate_temperature"]))

    synergy_reg = math.copysign(1.0, shaped_q + shaped_v) * math.sqrt(abs(shaped_q * shaped_v) + 1e-9)
    conflict = abs(shaped_a - shaped_v) + 0.65 * abs(shaped_q - shaped_v)
    reg_level = float(params["reg_iis_base"]) + float(params["reg_iis_amp"]) * math.tanh(reg_trace["scalar"] + 0.20 * synergy_reg)
    mob_level = float(params["mob_iis_base"]) + float(params["mob_iis_amp"]) * math.tanh(mob_trace["scalar"])
    dep_level = float(params["dep_iis_base"]) + float(params["dep_iis_amp"]) * math.tanh(dep_trace["scalar"])
    regime_score = gates[0] * reg_level + gates[1] * mob_level + gates[2] * dep_level
    entropy = -(sum(x * math.log(x + 1e-9) for x in gates)) / math.log(3.0)
    balance = gates[0] - gates[2]
    iis_final = _clip01(
        regime_score
        - float(params["transition_entropy_weight"]) * entropy
        + float(params["regime_balance_weight"]) * balance
        - float(params["conflict_penalty"]) * conflict
    )
    return {
        "params": params,
        "shaped": {"A": shaped_a, "Gamma": shaped_g, "GammaPos": gamma_pos, "V": shaped_v, "Q": shaped_q},
        "q_linear": q_linear,
        "q_coherence": q_coherence,
        "q_synergy": q_synergy,
        "q_trace": q_trace,
        "q_value": q_value,
        "reg_trace": reg_trace,
        "mob_trace": mob_trace,
        "dep_trace": dep_trace,
        "gates": gates,
        "levels": {"reg": reg_level, "mob": mob_level, "dep": dep_level},
        "conflict": conflict,
        "entropy": entropy,
        "balance": balance,
        "regime_score": regime_score,
        "iis_final": iis_final,
    }


def _build_numeric_example(version: str, row: dict[str, Any]) -> str:
    a = float(row.get("A")) if row.get("A") == row.get("A") else None
    g = float(row.get("Gamma")) if row.get("Gamma") == row.get("Gamma") else None
    h = float(row.get("H")) if row.get("H") == row.get("H") else None
    v = float(row.get("V")) if row.get("V") == row.get("V") else None
    q = float(row.get("Q")) if row.get("Q") == row.get("Q") else None
    k = float(row.get("K")) if row.get("K") == row.get("K") else None
    contrib = row.get("contributions_json") or {}

    if version in {"IISVersion1", "IISVersion2", "IISVersion3"} and None not in (a, g, h, v):
        raw = 0.35 * a + 0.25 * g + 0.30 * h + 0.10 * v
        iis = _sigmoid(raw)
        extra = ""
        if version == "IISVersion3":
            extra = (
                f"\nK уже сохранён в строке как K = {_fmt(k)};\n"
                f"H в CSV уже рассчитан и равен {_fmt(h)}, поэтому внешний слой можно подставить напрямую."
            )
        return (
            "<pre class='eq example-block'>"
            f"raw = 0.35·A + 0.25·Γ + 0.30·H + 0.10·V\n"
            f"    = 0.35·({_fmt(a)}) + 0.25·({_fmt(g)}) + 0.30·({_fmt(h)}) + 0.10·({_fmt(v)})\n"
            f"    = {_fmt(raw)}\n\n"
            f"IIS = σ(raw) = σ({_fmt(raw)}) = {_fmt(iis)}\n"
            f"stored raw_score = {_fmt(row.get('raw_score'))}\n"
            f"stored IIS       = {_fmt(row.get('IIS'))}"
            f"{extra}"
            "</pre>"
        )

    if version == "IISVersion4" and None not in (a, g, v, q):
        q_inner = 0.55 * a + 0.45 * v - 0.25 * g
        q_calc = math.tanh(q_inner)
        raw = 0.10 * a - 0.05 * g + 0.25 * v + 0.60 * q
        iis = _sigmoid(raw)
        return (
            "<pre class='eq example-block'>"
            f"Q_4 = tanh(0.55·A + 0.45·V - 0.25·Γ)\n"
            f"    = tanh(0.55·({_fmt(a)}) + 0.45·({_fmt(v)}) - 0.25·({_fmt(g)}))\n"
            f"    = tanh({_fmt(q_inner)}) = {_fmt(q_calc)}\n"
            f"stored Q = {_fmt(q)}\n\n"
            f"raw = 0.10·A - 0.05·Γ + 0.25·V + 0.60·Q_4\n"
            f"    = 0.10·({_fmt(a)}) - 0.05·({_fmt(g)}) + 0.25·({_fmt(v)}) + 0.60·({_fmt(q)})\n"
            f"    = {_fmt(raw)}\n\n"
            f"IIS = σ(raw) = σ({_fmt(raw)}) = {_fmt(iis)}\n"
            f"stored raw_score = {_fmt(row.get('raw_score'))}\n"
            f"stored IIS       = {_fmt(row.get('IIS'))}"
            "</pre>"
        )

    if version in {"IISVersion5", "IISVersion6", "IISVersion7"}:
        contrib_lines = "\n".join(
            f"{key:>7s}: {_fmt(value)}" for key, value in contrib.items()
        )
        extra = ""
        if version == "IISVersion7":
            extra = (
                f"\nRES            = {_fmt(row.get('RES'))}"
                f"\nRES_core       = {_fmt(row.get('RES_core'))}"
                f"\nstate_map_4    = {row.get('state_map_4') or '—'}"
                f"\nstate_margin   = {_fmt(row.get('state_map_margin'))}"
                "\nПример взят из DREAMER strict, где для V7 действует dataset-aware Dreamer override."
            )
        return (
            "<pre class='eq example-block'>"
            "В этой версии внутренние нелинейные/гейтовые слои глубже, поэтому для живого примера\n"
            "ниже показан exact engine trace, который уже сохранил движок в results_*.csv.\n\n"
            f"A = {_fmt(a)}\nΓ = {_fmt(g)}\nV = {_fmt(v)}\nQ = {_fmt(q)}\n\n"
            "Разложение stored raw_score по компонентам:\n"
            f"{contrib_lines}\n"
            f"{'-'*24}\n"
            f"raw_score(engine) = {_fmt(row.get('raw_score'))}\n"
            f"IIS(engine)       = {_fmt(row.get('IIS'))}"
            f"{extra}"
            "</pre>"
        )

    return "<p class='muted'>Для этой версии не удалось собрать числовую подстановку.</p>"


def _build_v6_gate_html(row: dict[str, Any]) -> str:
    trace = _v6_gate_trace(row)
    if trace is None:
        return ""
    shaped = trace["shaped"]
    q_rows = "".join(
        f"<tr><th>{label}</th><td>{html.escape(_fmt(logit))}</td><td>{html.escape(_fmt(gate))}</td><td>{html.escape(_fmt(level))}</td></tr>"
        for label, logit, gate, level in [
            ("reg", trace["q_logits"][0], trace["q_gates"][0], trace["q_levels"]["reg"]),
            ("mob", trace["q_logits"][1], trace["q_gates"][1], trace["q_levels"]["mob"]),
            ("dep", trace["q_logits"][2], trace["q_gates"][2], trace["q_levels"]["dep"]),
        ]
    )
    iis_rows = "".join(
        f"<tr><th>{label}</th><td>{html.escape(_fmt(logit))}</td><td>{html.escape(_fmt(gate))}</td><td>{html.escape(_fmt(level))}</td></tr>"
        for label, logit, gate, level in [
            ("reg", trace["iis_logits"][0], trace["iis_gates"][0], trace["iis_levels"]["reg"]),
            ("mob", trace["iis_logits"][1], trace["iis_gates"][1], trace["iis_levels"]["mob"]),
            ("dep", trace["iis_logits"][2], trace["iis_gates"][2], trace["iis_levels"]["dep"]),
        ]
    )
    return f"""
    <div class="trace-grid">
      <article class="example-card">
        <h3>V6 · shaped-компоненты</h3>
        <table class="trace-table">
          <tbody>
            <tr><th>A^shape</th><td>{html.escape(_fmt(shaped['A']))}</td></tr>
            <tr><th>Γ^shape</th><td>{html.escape(_fmt(shaped['Gamma']))}</td></tr>
            <tr><th>Γ^+</th><td>{html.escape(_fmt(shaped['GammaPos']))}</td></tr>
            <tr><th>V^shape</th><td>{html.escape(_fmt(shaped['V']))}</td></tr>
            <tr><th>Q^shape</th><td>{html.escape(_fmt(shaped['Q']))}</td></tr>
          </tbody>
        </table>
      </article>
      <article class="example-card">
        <h3>V6 · гейты для Q</h3>
        <table class="trace-table">
          <thead><tr><th>режим</th><th>logit</th><th>softmax</th><th>q_level</th></tr></thead>
          <tbody>{q_rows}</tbody>
        </table>
        <p><strong>Q_mix(calc):</strong> {html.escape(_fmt(trace['q_mix']))}</p>
        <p><strong>Q(stored):</strong> {html.escape(_fmt(row.get('Q')))}</p>
      </article>
      <article class="example-card full">
        <h3>V6 · финальные гейты IIS</h3>
        <table class="trace-table">
          <thead><tr><th>режим</th><th>logit</th><th>softmax</th><th>iis_level</th></tr></thead>
          <tbody>{iis_rows}</tbody>
        </table>
        <div class="trace-meta">
          <p><strong>conflict:</strong> {_fmt(trace['conflict'])}</p>
          <p><strong>entropy:</strong> {_fmt(trace['entropy'])}</p>
          <p><strong>balance:</strong> {_fmt(trace['balance'])}</p>
          <p><strong>regime_score:</strong> {_fmt(trace['regime_score'])}</p>
          <p><strong>IIS(calc):</strong> {_fmt(trace['iis_final'])}</p>
          <p><strong>IIS(stored):</strong> {_fmt(row.get('IIS'))}</p>
        </div>
      </article>
    </div>
    """


def _build_v7_gate_html(row: dict[str, Any]) -> str:
    trace = _v7_gate_trace(row)
    if trace is None:
        return ""
    reg_html = _build_sumigron_trace_html(
        "V7 · regulation signature",
        ["0.72·Q^shape", "0.52·V^shape", "0.18·A^shape", "-0.24·Γ^+"],
        trace["reg_trace"],
        "reg_signature",
    )
    mob_html = _build_sumigron_trace_html(
        "V7 · mobilization signature",
        ["-0.06·Q^shape", "0.28·V^shape", "-0.48·Γ^+", "0.10·A^shape"],
        trace["mob_trace"],
        "mob_signature",
    )
    dep_html = _build_sumigron_trace_html(
        "V7 · depletion signature",
        ["-0.60·Q^shape", "-0.50·V^shape", "0.36·Γ^+", "-0.20·A^shape"],
        trace["dep_trace"],
        "dep_signature",
    )
    gate_rows = "".join(
        f"<tr><th>{label}</th><td>{html.escape(_fmt(signature))}</td><td>{html.escape(_fmt(gate))}</td><td>{html.escape(_fmt(level))}</td></tr>"
        for label, signature, gate, level in [
            ("reg", trace["reg_trace"]["scalar"], trace["gates"][0], trace["levels"]["reg"]),
            ("mob", trace["mob_trace"]["scalar"], trace["gates"][1], trace["levels"]["mob"]),
            ("dep", trace["dep_trace"]["scalar"], trace["gates"][2], trace["levels"]["dep"]),
        ]
    )
    params = trace["params"]
    return f"""
    <div class="trace-grid">
      {reg_html}
      {mob_html}
      {dep_html}
      <article class="example-card full">
        <h3>V7 · финальные гейты IIS</h3>
        <table class="trace-table">
          <thead><tr><th>режим</th><th>signature</th><th>softmax</th><th>iis_level</th></tr></thead>
          <tbody>{gate_rows}</tbody>
        </table>
        <div class="trace-meta">
          <p><strong>gate_temperature:</strong> {_fmt(params['gate_temperature'])}</p>
          <p><strong>transition_entropy_weight:</strong> {_fmt(params['transition_entropy_weight'])}</p>
          <p><strong>regime_balance_weight:</strong> {_fmt(params['regime_balance_weight'])}</p>
          <p><strong>conflict_penalty:</strong> {_fmt(params['conflict_penalty'])}</p>
          <p><strong>conflict:</strong> {_fmt(trace['conflict'])}</p>
          <p><strong>entropy:</strong> {_fmt(trace['entropy'])}</p>
          <p><strong>balance:</strong> {_fmt(trace['balance'])}</p>
          <p><strong>regime_score:</strong> {_fmt(trace['regime_score'])}</p>
          <p><strong>IIS(calc):</strong> {_fmt(trace['iis_final'])}</p>
          <p><strong>IIS(stored):</strong> {_fmt(row.get('IIS'))}</p>
        </div>
      </article>
    </div>
    """


def _build_sumigron_deep_dive(row: dict[str, Any]) -> str:
    trace = _v7_gate_trace(row)
    if trace is None:
        return ""
    q_trace = trace["q_trace"]
    q_html = _build_sumigron_trace_html(
        "Q7 через Sumigron",
        ["q_linear", "q_coherence", "0.55·q_synergy"],
        q_trace,
        "q_raw",
    )
    params = trace["params"]
    return f"""
    <section class="version-section" id="sumigron-deep-dive">
      <div class="version-header">
        <div>
          <h2>Sumigron · отдельный пошаговый разбор</h2>
          <p class="lead">Ниже не общая идея, а реальный trace по строке V7 из DREAMER strict.</p>
        </div>
        <div class="version-badges">
          <span class="badge badge-era">Экспериментальный оператор</span>
          <span class="badge badge-univ">dataset-aware</span>
        </div>
      </div>
      <div class="summary-grid">
        <article class="summary-card">
          <h3>Строка</h3>
          <p><strong>dataset:</strong> {html.escape(str(row.get('dataset')))}</p>
          <p><strong>subject:</strong> {html.escape(str(row.get('subject_id')))}</p>
          <p><strong>segment:</strong> {html.escape(str(row.get('segment_id')))}</p>
        </article>
        <article class="summary-card">
          <h3>Параметры Sumigron</h3>
          <p><strong>temperature:</strong> {_fmt(params['sumigron_temperature'])}</p>
          <p><strong>level_weight:</strong> {_fmt(params['sumigron_level_weight'])}</p>
          <p><strong>structure_weight:</strong> {_fmt(params['sumigron_structure_weight'])}</p>
          <p><strong>energy_weight:</strong> {_fmt(params['sumigron_energy_weight'])}</p>
        </article>
        <article class="summary-card">
          <h3>Входы в Q7</h3>
          <p><strong>q_linear:</strong> {_fmt(trace['q_linear'])}</p>
          <p><strong>q_coherence:</strong> {_fmt(trace['q_coherence'])}</p>
          <p><strong>q_synergy:</strong> {_fmt(trace['q_synergy'])}</p>
          <p><strong>Q7(calc):</strong> {_fmt(trace['q_value'])}</p>
          <p><strong>Q7(stored):</strong> {_fmt(row.get('Q'))}</p>
        </article>
      </div>
      <div class="trace-grid">
        {q_html}
      </div>
      <article class="notes-card">
        <h3>Как это читать</h3>
        <ul>
          <li>Сначала Sumigron не насыщает входы сразу, а переводит их в локальную структуру: mean, std, energy через attentive-веса.</li>
          <li>Потом из этой структуры собирается скаляр <code>q_raw</code>.</li>
          <li>Уже после этого на него навешивается bounded-нелинейность для получения <code>Q7</code>.</li>
        </ul>
      </article>
    </section>
    """

def _build_section(version: str, row: dict[str, Any]) -> str:
    meta = VERSION_METADATA[version]
    formula = FORMULA_SECTIONS[version]
    scenario = EXAMPLE_SOURCES[version]["scenario"]
    formula_blocks = "".join(
        f"""
        <article class="formula-card">
          <h3>{html.escape(title)}</h3>
          <pre class="eq">{html.escape(block.strip())}</pre>
        </article>
        """
        for title, block in formula["formula_blocks"]
    )
    notes = "".join(f"<li>{html.escape(note)}</li>" for note in formula["notes"])
    explanation = row.get("score_explanation_json") or {}
    top_contributors = ", ".join(explanation.get("top_contributors", [])) or "—"
    deep_trace_html = ""
    if version == "IISVersion6":
        deep_trace_html = _build_v6_gate_html(row)
    elif version == "IISVersion7":
        deep_trace_html = _build_v7_gate_html(row)
    return f"""
    <section class="version-section" id="{html.escape(meta['short'].lower())}">
      <div class="version-header">
        <div>
          <h2>{html.escape(meta['short'])} · {html.escape(meta['title'])}</h2>
          <p class="lead">{html.escape(formula['headline'])}</p>
        </div>
        <div class="version-badges">
          <span class="badge badge-era">{html.escape(meta['era'])}</span>
          <span class="badge badge-univ">{html.escape(meta['universality'])}</span>
        </div>
      </div>

      <div class="summary-grid">
        <article class="summary-card">
          <h3>Назначение</h3>
          <p>{html.escape(meta['purpose'])}</p>
        </article>
        <article class="summary-card">
          <h3>Сильная сторона</h3>
          <p>{html.escape(meta['strength'])}</p>
        </article>
        <article class="summary-card">
          <h3>Слабое место</h3>
          <p>{html.escape(meta['weakness'])}</p>
        </article>
      </div>

      <div class="formula-grid">
        {formula_blocks}
      </div>

      <div class="example-grid">
        <article class="example-card">
          <h3>Пример строки из данных</h3>
          {_build_metadata_table(row, scenario)}
          <p class="muted">{html.escape(str(row.get('formula_note') or ''))}</p>
        </article>
        <article class="example-card">
          <h3>Компоненты строки</h3>
          {_build_component_table(row)}
        </article>
      </div>

      <article class="example-card full">
        <h3>Как это подставляется на данных</h3>
        {_build_numeric_example(version, row)}
      </article>

      <div class="example-grid">
        <article class="example-card">
          <h3>Разложение вкладов</h3>
          {_build_contrib_table(row.get('contributions_json') or {})}
        </article>
        <article class="example-card">
          <h3>След движка</h3>
          <p><strong>Активные блоки:</strong> {html.escape(', '.join(explanation.get('active_components', [])) or '—')}</p>
          <p><strong>Отсутствующие блоки:</strong> {html.escape(', '.join(explanation.get('missing_components', [])) or '—')}</p>
          <p><strong>Top contributors:</strong> {html.escape(top_contributors)}</p>
          <p><strong>stored raw_score:</strong> {html.escape(_fmt(row.get('raw_score')))}</p>
          <p><strong>stored IIS:</strong> {html.escape(_fmt(row.get('IIS')))}</p>
        </article>
      </div>

      {deep_trace_html}

      <article class="notes-card">
        <h3>Пояснения</h3>
        <ul>{notes}</ul>
      </article>
    </section>
    """


def build_html() -> str:
    example_rows = {
        version: _pick_example_row(source["path"], version)
        for version, source in EXAMPLE_SOURCES.items()
    }
    sections = "".join(_build_section(version, example_rows[version]) for version in VERSION_ORDER)
    sumigron_deep_dive = _build_sumigron_deep_dive(example_rows["IISVersion7"])
    nav_links = "".join(
        f'<a href="#{VERSION_METADATA[version]["short"].lower()}">{html.escape(VERSION_METADATA[version]["short"])}</a>'
        for version in VERSION_ORDER
    )
    return f"""<!doctype html>
<html lang="ru">
<head>
  <meta charset="utf-8">
  <title>IIS Expanded Formula HTML</title>
  <style>
    :root {{
      --bg: #11111b;
      --paper: rgba(30, 30, 46, 0.74);
      --ink: #cdd6f4;
      --muted: #a6adc8;
      --line: rgba(205, 214, 244, 0.10);
      --accent: #89b4fa;
      --accent-soft: rgba(137, 180, 250, 0.16);
      --warn: #f9e2af;
      --warn-soft: rgba(249, 226, 175, 0.12);
      --ok: #a6e3a1;
      --ok-soft: rgba(166, 227, 161, 0.12);
      --rose: #f5c2e7;
      --surface: rgba(49, 50, 68, 0.82);
      --shadow: 0 24px 64px rgba(0, 0, 0, 0.42);
      --glow-accent: 0 0 0 1px rgba(137, 180, 250, 0.10), 0 0 28px rgba(137, 180, 250, 0.10);
      --glow-rose: 0 0 24px rgba(245, 194, 231, 0.08);
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font: 16px/1.55 "Segoe UI", "Noto Sans", sans-serif;
      color: var(--ink);
      background:
        radial-gradient(circle at top left, rgba(203, 166, 247, 0.18), transparent 26%),
        radial-gradient(circle at top right, rgba(137, 180, 250, 0.16), transparent 24%),
        linear-gradient(180deg, #11111b 0%, #181825 100%);
      color-scheme: dark;
    }}
    .layout {{
      display: grid;
      grid-template-columns: 260px minmax(0, 1fr);
      min-height: 100vh;
    }}
    .sidebar {{
      position: sticky;
      top: 0;
      align-self: start;
      height: 100vh;
      padding: 28px 22px;
      border-right: 1px solid var(--line);
      background: rgba(17, 17, 27, .58);
      backdrop-filter: blur(22px) saturate(140%);
    }}
    .sidebar h1 {{
      margin: 0 0 10px 0;
      font-size: 1.2rem;
    }}
    .sidebar p {{
      margin: 0 0 16px 0;
      color: var(--muted);
      font-size: .95rem;
    }}
    .sidebar nav {{
      display: grid;
      gap: 8px;
    }}
    .sidebar a {{
      color: var(--accent);
      text-decoration: none;
      padding: 7px 10px;
      border-radius: 10px;
      transition: background .18s ease, box-shadow .18s ease, color .18s ease, transform .18s ease;
    }}
    .sidebar a:hover {{
      background: var(--accent-soft);
      box-shadow: var(--glow-accent);
      transform: translateX(2px);
    }}
    main {{
      padding: 34px 38px 52px 38px;
    }}
    .hero, .version-section, .notes-card, .summary-card, .example-card, .formula-card {{
      background: var(--paper);
      border: 1px solid var(--line);
      border-radius: 18px;
      box-shadow: var(--shadow), var(--glow-rose);
      backdrop-filter: blur(18px) saturate(125%);
    }}
    .hero {{
      padding: 28px 30px;
      margin-bottom: 26px;
    }}
    .hero h2 {{
      margin: 0 0 12px 0;
      font-size: 2rem;
      text-shadow: 0 0 24px rgba(137, 180, 250, 0.14);
    }}
    .hero p {{
      margin: 8px 0;
      max-width: 980px;
    }}
    .hero-grid {{
      display: grid;
      grid-template-columns: repeat(3, minmax(0, 1fr));
      gap: 14px;
      margin-top: 18px;
    }}
    .hero-chip {{
      padding: 14px 16px;
      border-radius: 14px;
      background: var(--surface);
      border: 1px solid var(--line);
      box-shadow: inset 0 1px 0 rgba(255,255,255,0.04), 0 0 20px rgba(137, 180, 250, 0.05);
    }}
    .version-section {{
      padding: 24px;
      margin-bottom: 22px;
    }}
    .version-header {{
      display: flex;
      justify-content: space-between;
      gap: 16px;
      align-items: start;
      margin-bottom: 16px;
    }}
    .version-header h2 {{
      margin: 0 0 6px 0;
    }}
    .lead {{
      margin: 0;
      color: var(--muted);
    }}
    .version-badges {{
      display: flex;
      gap: 8px;
      flex-wrap: wrap;
    }}
    .badge {{
      padding: 7px 10px;
      border-radius: 999px;
      font-size: .82rem;
      font-weight: 600;
      white-space: nowrap;
    }}
    .badge-era {{
      background: var(--accent-soft);
      color: var(--accent);
      box-shadow: 0 0 18px rgba(137, 180, 250, 0.10);
    }}
    .badge-univ {{
      background: var(--warn-soft);
      color: var(--warn);
      box-shadow: 0 0 18px rgba(249, 226, 175, 0.08);
    }}
    .summary-grid, .formula-grid, .example-grid {{
      display: grid;
      gap: 14px;
      margin-top: 14px;
    }}
    .summary-grid {{
      grid-template-columns: repeat(3, minmax(0, 1fr));
    }}
    .formula-grid {{
      grid-template-columns: repeat(2, minmax(0, 1fr));
    }}
    .example-grid {{
      grid-template-columns: repeat(2, minmax(0, 1fr));
    }}
    .summary-card, .formula-card, .example-card, .notes-card {{
      padding: 18px;
    }}
    .example-card.full {{
      margin-top: 14px;
    }}
    h3 {{
      margin-top: 0;
      margin-bottom: 12px;
    }}
    .eq {{
      margin: 0;
      padding: 14px 16px;
      overflow-x: auto;
      white-space: pre-wrap;
      font: 14px/1.55 "Cascadia Mono", "Consolas", monospace;
      border-radius: 14px;
      background: rgba(24, 24, 37, 0.92);
      border: 1px solid var(--line);
      box-shadow: inset 0 1px 0 rgba(255,255,255,0.04), 0 0 24px rgba(137, 180, 250, 0.06);
    }}
    .example-block {{
      background: rgba(17, 17, 27, 0.96);
      box-shadow: inset 0 1px 0 rgba(255,255,255,0.03), 0 0 30px rgba(245, 194, 231, 0.05);
    }}
    .muted {{
      color: var(--muted);
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      font-size: .95rem;
    }}
    th, td {{
      padding: 8px 10px;
      border-bottom: 1px solid var(--line);
      text-align: left;
      vertical-align: top;
    }}
    th {{
      width: 38%;
      color: var(--muted);
      font-weight: 600;
    }}
    .total th, .total td {{
      font-weight: 700;
    }}
    .trace-grid {{
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 14px;
      margin-top: 14px;
    }}
    .trace-table thead th {{
      color: var(--ink);
      background: var(--surface);
      box-shadow: inset 0 -1px 0 rgba(255,255,255,0.03);
    }}
    .trace-meta {{
      margin-top: 12px;
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 8px 16px;
    }}
    .trace-meta p {{
      margin: 0;
    }}
    ul {{
      margin: 0;
      padding-left: 18px;
    }}
    @media (max-width: 1120px) {{
      .layout {{ grid-template-columns: 1fr; }}
      .sidebar {{
        position: static;
        height: auto;
        border-right: 0;
        border-bottom: 1px solid var(--line);
      }}
      main {{ padding: 22px; }}
      .summary-grid, .formula-grid, .example-grid, .hero-grid, .trace-grid, .trace-meta {{
        grid-template-columns: 1fr;
      }}
      .version-header {{
        flex-direction: column;
      }}
    }}
  </style>
</head>
<body>
  <div class="layout">
    <aside class="sidebar">
      <h1>IIS · Expanded Formulas</h1>
      <p>Автономный формульный HTML: полные symbolic-блоки и числовые подстановки по реальным строкам из результатов.</p>
      <nav>
        <a href="#top">Верх</a>
        <a href="#sumigron-deep-dive">Sumigron</a>
        {nav_links}
      </nav>
    </aside>
    <main id="top">
      <section class="hero">
        <h2>Расширенный формульный HTML по IISVersion1–IISVersion7</h2>
        <p>Этот файл не использует MathJax и внешние зависимости. Формулы записаны в читаемом ASCII/Unicode-виде, чтобы они открывались локально без сети. Для каждой версии ниже есть два уровня: полная symbolic-запись и живая подстановка по реальной строке из <code>results_*.csv</code>.</p>
        <p>Для простых версий V1–V4 показана прямолинейная числовая подстановка. Для V5–V7 ниже показан честный <em>engine trace</em>: полные формулы даны symbolic, а числовой пример строится через сохранённые в CSV компонентные вклады и итоговый <code>raw_score</code>, потому что внутренние нелинейные слои глубже и не разворачиваются полностью в одном столбце.</p>
        <div class="hero-grid">
          <div class="hero-chip"><strong>V1–V3</strong><br>Исторические proxy-ориентированные линии.</div>
          <div class="hero-chip"><strong>V4–V6</strong><br>Практическое негормональное ядро и gated-линия.</div>
          <div class="hero-chip"><strong>V7 + Sumigron</strong><br>Экспериментальная beta-ветка с dataset-aware Dreamer override.</div>
        </div>
      </section>
      {sumigron_deep_dive}
      {sections}
    </main>
  </div>
</body>
</html>
"""


def main() -> int:
    OUTPUT_HTML.parent.mkdir(parents=True, exist_ok=True)
    html_text = build_html()
    OUTPUT_HTML.write_text(html_text, encoding="utf-8")
    OUTPUT_HTML_MOCHA.write_text(html_text, encoding="utf-8")
    print(f"html={OUTPUT_HTML}")
    print(f"html_mocha={OUTPUT_HTML_MOCHA}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
