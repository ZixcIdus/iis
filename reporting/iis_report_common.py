"""Общие константы, метаданные и рендеринг для полного IIS-отчёта."""

from __future__ import annotations

import html
import math
from pathlib import Path
from typing import Any

from config.settings import OUTPUT_DIR

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DOCS_DIR = PROJECT_ROOT / "docs"

VERSION_ORDER = [
    "IISVersion1",
    "IISVersion2",
    "IISVersion3",
    "IISVersion4",
    "IISVersion5",
    "IISVersion6",
    "IISVersion7",
]

VERSION_SHORT = {name: f"V{index}" for index, name in enumerate(VERSION_ORDER, start=1)}

SCENARIOS: list[dict[str, str]] = [
    {"dataset": "case", "mode": "hybrid", "label": "CASE / hybrid"},
    {"dataset": "wesad", "mode": "hybrid", "label": "WESAD / hybrid"},
    {"dataset": "ds002722", "mode": "strict", "label": "ds002722 / strict"},
    {"dataset": "ds002722", "mode": "proxy", "label": "ds002722 / proxy"},
    {"dataset": "ds002724", "mode": "strict", "label": "ds002724 / strict"},
    {"dataset": "ds002724", "mode": "proxy", "label": "ds002724 / proxy"},
    {"dataset": "dreamer", "mode": "strict", "label": "DREAMER / strict"},
]

DYNAMIC_SCENARIOS: list[dict[str, str]] = [
    {"dataset": "ds002722", "mode": "strict", "label": "ds002722 / strict"},
    {"dataset": "ds002724", "mode": "strict", "label": "ds002724 / strict"},
    {"dataset": "dreamer", "mode": "strict", "label": "DREAMER / strict"},
]

EXPERIMENT_FILES = {
    "v5_dreamer_calibration": OUTPUT_DIR / "dreamer_v5_calibration.json",
    "v7_dreamer_calibration": OUTPUT_DIR / "dreamer_v7_calibration_global.json",
    "sumigron_synthetic_summary": OUTPUT_DIR / "sumigron_synthetic_summary.json",
    "sumigron_synthetic_metrics": OUTPUT_DIR / "sumigron_synthetic_metrics.csv",
}

BUNDLE_JSON_PATH = OUTPUT_DIR / "iis_full_report_bundle.json"
BUNDLE_CSV_PATH = OUTPUT_DIR / "iis_full_report_bundle_rows.csv"
PDF_TEX_PATH = DOCS_DIR / "iis_full_report.tex"
PDF_PATH = DOCS_DIR / "iis_full_report.pdf"
HTML_PATH = DOCS_DIR / "iis_full_report.html"


def scenario_key(dataset: str, mode: str) -> str:
    return f"{dataset}/{mode}"


def scenario_filename_slug(dataset: str, mode: str) -> str:
    return f"{dataset}_{mode}"


VERSION_METADATA: dict[str, dict[str, Any]] = {
    "IISVersion1": {
        "short": "V1",
        "title": "Тестовое раннее ядро",
        "era": "Историческая",
        "formula_latex": r"IIS_1=\sigma(0.35A+0.25\Gamma+0.30H+0.10V)",
        "purpose": "Проверить саму идею интегрального индекса на четырёх базовых блоках.",
        "inputs": ["A по alpha-асимметрии", "Gamma", "H", "V"],
        "strength": "Максимально простая и прозрачная структура.",
        "weakness": "Сильно зависит от гормонального proxy-блока и плохо переносится в strict-режим.",
        "universality": "Низкая",
        "static_supported": True,
        "dynamic_supported": False,
        "proxy_supported": True,
        "hybrid_supported": True,
        "status_note": "Историческая proxy-ориентированная версия. Полезна как точка отсчёта, но не как современное универсальное ядро.",
        "limitation_flags": ["историческая версия", "не универсальна", "proxy-ориентирована", "не адаптирована к динамике"],
    },
    "IISVersion2": {
        "short": "V2",
        "title": "Версия 1.0 с общей лево-правой мощностью",
        "era": "Историческая",
        "formula_latex": r"IIS_2=\sigma(0.35A+0.25\Gamma+0.30H+0.10V)",
        "purpose": "Сохранить простую интеграцию V1, но заменить A на более грубую межполушарную мощность.",
        "inputs": ["A по общей мощности полушарий", "Gamma", "H", "V"],
        "strength": "Чуть ближе к общим EEG-признакам, чем V1.",
        "weakness": "Остаётся завязанной на гормональный блок и в реальных strict-сценариях почти неработоспособна.",
        "universality": "Низкая",
        "static_supported": True,
        "dynamic_supported": False,
        "proxy_supported": True,
        "hybrid_supported": True,
        "status_note": "Историческая промежуточная версия. Применимость ограничена proxy/hybrid-сценариями.",
        "limitation_flags": ["историческая версия", "не универсальна", "proxy-ориентирована", "не адаптирована к динамике"],
    },
    "IISVersion3": {
        "short": "V3",
        "title": "Версия 2.0 с K и отдельной валентностью Q",
        "era": "Историческая",
        "formula_latex": r"IIS_3=\sigma(0.35A+0.25\Gamma+0.30H+0.10V),\quad H=\frac{K\cdot DA}{50}-\frac{C}{15}",
        "purpose": "Добавить коэффициент проницаемости K и отделить диагностическую валентность Q от основного IIS.",
        "inputs": ["A", "Gamma", "H через K", "V", "Q как отдельный диагностический блок"],
        "strength": "Содержательно богаче V1/V2 за счёт K и Q.",
        "weakness": "Слишком тяжело опирается на proxy-нормировки и не подтверждается как универсальная практическая модель.",
        "universality": "Низкая",
        "static_supported": True,
        "dynamic_supported": False,
        "proxy_supported": True,
        "hybrid_supported": True,
        "status_note": "Историческая расширенная версия. Полезна для истории идеи и сопоставления, но не для основной практической линии.",
        "limitation_flags": ["историческая версия", "не универсальна", "proxy-ориентирована", "не адаптирована к динамике"],
    },
    "IISVersion4": {
        "short": "V4",
        "title": "Первое негормональное практическое ядро",
        "era": "Практическая",
        "formula_latex": r"IIS_4=\sigma(0.10A-0.05\Gamma+0.25V+0.60Q_4)",
        "purpose": "Убрать гормоны и собрать рабочую EEG/ECG-модель для открытых данных.",
        "inputs": ["A по alpha/total asymmetry", "Gamma по gamma и gamma/alpha", "V по HR и HF/LF", "Q как физиологическая интеграция"],
        "strength": "Первая реально переносимая строгая версия без выдуманных биохимических блоков.",
        "weakness": "Шкала ещё довольно плоская; промежуточные режимы часто сглаживаются.",
        "universality": "Средняя",
        "static_supported": True,
        "dynamic_supported": True,
        "proxy_supported": True,
        "hybrid_supported": True,
        "status_note": "Рабочая практическая база. Хороша как baseline, но по внутренней геометрии уступает V5/V6.",
        "limitation_flags": [],
    },
    "IISVersion5": {
        "short": "V5",
        "title": "Контрастная нелинейная версия",
        "era": "Практическая",
        "formula_latex": r"IIS_5=\operatorname{clip}\!\left((1-m)IIS_{5,\mathrm{base}}+mIIS_{5,\mathrm{contrast}},0,1\right)",
        "purpose": "Раскрыть контраст режимов и уменьшить плоскость середины шкалы без возврата к гормонам.",
        "inputs": ["A", "Gamma", "V", "Q с coherence/synergy/conflict"],
        "strength": "Часто лучше выделяет скрытую структуру и средние режимы, чем V4.",
        "weakness": "Не универсальна по датасетам: на части наборов выигрывает, на части нет; чувствительна к калибровке.",
        "universality": "Средняя",
        "static_supported": True,
        "dynamic_supported": True,
        "proxy_supported": True,
        "hybrid_supported": True,
        "status_note": "Нелинейная практическая версия. Хорошо работает там, где помогает контрастная геометрия, но не является окончательно универсальной.",
        "limitation_flags": ["не универсальна"],
    },
    "IISVersion6": {
        "short": "V6",
        "title": "Gated-модель режимов",
        "era": "Практическая",
        "formula_latex": r"IIS_6=g_{\mathrm{reg}}i_{\mathrm{reg}}+g_{\mathrm{mob}}i_{\mathrm{mob}}+g_{\mathrm{dep}}i_{\mathrm{dep}}-\lambda_H H(g)-\lambda_C C",
        "purpose": "Сделать модель смесью режимов regulation / mobilization / depletion вместо одной глобальной почти линейной формулы.",
        "inputs": ["A", "Gamma", "V", "Q", "softmax-gates режимов"],
        "strength": "Лучшее текущее универсальное практическое strict-ядро, особенно на ds002724.",
        "weakness": "Число базовых режимов зашито в gate-архитектуру; не всегда раскрывает более богатую карту состояний.",
        "universality": "Высокая",
        "static_supported": True,
        "dynamic_supported": True,
        "proxy_supported": True,
        "hybrid_supported": True,
        "status_note": "Текущая лучшая практическая линия для честных негормональных strict-сценариев.",
        "limitation_flags": [],
    },
    "IISVersion7": {
        "short": "V7",
        "title": "Экспериментальная Sumigron-бета",
        "era": "Экспериментальная",
        "formula_latex": r"IIS_7=g^{reg}_7 s^{reg}_7 + g^{mob}_7 s^{mob}_7 + g^{dep}_7 s^{dep}_7,\quad \sgop=\sum+\int+\bowtie",
        "purpose": "Перенести структурную нелинейность раньше по цепочке через оператор Sumigron и сохранить 3D-слой IIS/RES/DYN.",
        "inputs": ["A, Gamma, V через Sumigron-подтермы", "Q как структурная интеграция", "gated IIS", "RES", "DYN"],
        "strength": "Даёт новую архитектурную линию и уже показывает пользу на части наборов и synthetic-тестах.",
        "weakness": "Пока beta-версия; есть dataset-specific override для DREAMER, поэтому она ещё не универсальна.",
        "universality": "Экспериментальная",
        "static_supported": True,
        "dynamic_supported": True,
        "proxy_supported": True,
        "hybrid_supported": True,
        "status_note": "Экспериментальная V7-beta. Есть Dreamer-specific override, поэтому её нельзя пока подавать как окончательно универсальную.",
        "limitation_flags": ["не универсальна", "dataset-specific calibration"],
    },
}


DATA_MODE_EXPLANATIONS = [
    {
        "mode": "strict",
        "title": "Strict",
        "body": "Используются только реально измеренные прямые признаки. Ничего не додумывается и не якорится proxy-метками.",
    },
    {
        "mode": "proxy",
        "title": "Proxy",
        "body": "Разрешены proxy-замены и мягкие якоря по доступным косвенным признакам и self-report. Это полезно для исторических V1–V3 и части ранних экспериментов.",
    },
    {
        "mode": "hybrid",
        "title": "Hybrid",
        "body": "Модель использует всё, что реально доступно: прямые признаки остаются прямыми, недостающие части могут компенсироваться proxy-слоем.",
    },
]


def version_status_bucket(version_name: str) -> str:
    metadata = VERSION_METADATA[version_name]
    if "историческая версия" in metadata["limitation_flags"]:
        return "danger"
    if version_name == "IISVersion7":
        return "warning"
    if version_name in {"IISVersion4", "IISVersion5"}:
        return "warning"
    return "success"


def format_float(value: Any, digits: int = 3) -> str:
    if value is None:
        return "—"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "—"
    if math.isnan(number) or math.isinf(number):
        return "—"
    return f"{number:.{digits}f}"


def format_int(value: Any) -> str:
    if value is None:
        return "—"
    try:
        number = int(value)
    except (TypeError, ValueError):
        return "—"
    return str(number)


def bool_word(value: bool) -> str:
    return "да" if value else "нет"


def escape_tex(text: str) -> str:
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
        "—": r"---",
        "≈": r"\ensuremath{\approx}",
        "→": r"\ensuremath{\rightarrow}",
        "↓": r"\ensuremath{\downarrow}",
    }
    output = []
    for character in text:
        output.append(replacements.get(character, character))
    return "".join(output)


def tex_multiline(items: list[str]) -> str:
    return r"\\ ".join(escape_tex(item) for item in items)


def html_badge(text: str, bucket: str) -> str:
    return f'<span class="badge badge-{bucket}">{html.escape(text)}</span>'


def tex_badge(text: str, bucket: str) -> str:
    color = {"success": "green!20!black", "warning": "orange!60!black", "danger": "red!70!black"}.get(bucket, "black")
    return rf"\textcolor{{{color}}}{{\textbf{{{escape_tex(text)}}}}}"


def _render_flow_html_rows(rows: list[list[dict[str, Any] | str]]) -> str:
    rendered_rows: list[str] = []
    for row_index, row in enumerate(rows):
        node_fragments: list[str] = []
        for node_index, node in enumerate(row):
            payload = node if isinstance(node, dict) else {"label": str(node), "bucket": "warning"}
            label = html.escape(str(payload["label"]))
            bucket = payload.get("bucket", "warning")
            node_fragments.append(f'<div class="flow-node flow-node-{bucket}">{label}</div>')
            if node_index < len(row) - 1:
                node_fragments.append('<div class="flow-arrow">→</div>')
        row_html = f'<div class="flow-row">{"".join(node_fragments)}</div>'
        rendered_rows.append(row_html)
        if row_index < len(rows) - 1:
            rendered_rows.append('<div class="flow-row flow-row-down"><div class="flow-down">↓</div></div>')
    return "".join(rendered_rows)


def _render_flow_tex_rows(rows: list[list[dict[str, Any] | str]]) -> str:
    rendered_rows: list[str] = []
    for row_index, row in enumerate(rows):
        node_fragments: list[str] = []
        for node_index, node in enumerate(row):
            payload = node if isinstance(node, dict) else {"label": str(node), "bucket": "warning"}
            label = escape_tex(str(payload["label"]))
            node_fragments.append(rf"\fbox{{\parbox{{2.7cm}}{{\centering {label}}}}}")
            if node_index < len(row) - 1:
                node_fragments.append(r"$\rightarrow$")
        rendered_rows.append(" ".join(node_fragments))
        if row_index < len(rows) - 1:
            rendered_rows.append(r"$\downarrow$")
    return "\\\\[0.6em]\n".join(rendered_rows)


def render_diagram_html(spec: dict[str, Any]) -> str:
    title = html.escape(spec["title"])
    description = html.escape(spec.get("description", ""))
    if spec["type"] == "flow":
        body = _render_flow_html_rows(spec["rows"])
    elif spec["type"] == "matrix":
        headers = "".join(f"<th>{html.escape(str(column))}</th>" for column in spec["columns"])
        rows_html: list[str] = []
        for row in spec["rows"]:
            label = html.escape(str(row["label"]))
            cells = []
            for value in row["values"]:
                bucket = "success" if value == "да" else "warning" if value == "частично" else "danger"
                cells.append(f'<td class="matrix-cell matrix-cell-{bucket}">{html.escape(str(value))}</td>')
            rows_html.append(f"<tr><th>{label}</th>{''.join(cells)}</tr>")
        body = f'<table class="diagram-matrix"><thead><tr><th>Версия</th>{headers}</tr></thead><tbody>{"".join(rows_html)}</tbody></table>'
    else:
        raise ValueError(f"Неизвестный тип схемы: {spec['type']}")
    return (
        f'<section class="diagram-block" id="{html.escape(spec["id"])}">'
        f"<h4>{title}</h4>"
        f"<div class=\"diagram-description\">{description}</div>"
        f"{body}"
        f"</section>"
    )


def render_diagram_tex(spec: dict[str, Any]) -> str:
    title = escape_tex(spec["title"])
    description = escape_tex(spec.get("description", ""))
    if spec["type"] == "flow":
        body = _render_flow_tex_rows(spec["rows"])
        rendered = rf"\begin{{center}}{body}\end{{center}}"
    elif spec["type"] == "matrix":
        column_spec = "l" + "c" * len(spec["columns"])
        header = " & ".join(["Версия", *[escape_tex(str(column)) for column in spec["columns"]]]) + r" \\"
        row_lines = []
        for row in spec["rows"]:
            values = [escape_tex(str(value)) for value in row["values"]]
            row_lines.append(" & ".join([escape_tex(str(row["label"])), *values]) + r" \\")
        rendered = (
            rf"\begin{{center}}\begin{{tabular}}{{{column_spec}}}\toprule "
            + header
            + r"\midrule "
            + " ".join(row_lines)
            + r"\bottomrule\end{tabular}\end{center}"
        )
    else:
        raise ValueError(f"Неизвестный тип схемы: {spec['type']}")
    return rf"\subsubsection*{{{title}}}{description and f' {description}' or ''}" + "\n" + rendered


def build_diagram_specs() -> list[dict[str, Any]]:
    version_nodes = [
        {"label": f"{VERSION_SHORT[name]}: {VERSION_METADATA[name]['title']}", "bucket": version_status_bucket(name)}
        for name in VERSION_ORDER
    ]
    applicability_rows = []
    for name in VERSION_ORDER:
        metadata = VERSION_METADATA[name]
        applicability_rows.append(
            {
                "label": VERSION_SHORT[name],
                "values": [
                    "да" if metadata["static_supported"] else "нет",
                    "да" if metadata["dynamic_supported"] else "нет",
                    "да" if metadata["proxy_supported"] else "нет",
                    "да" if metadata["hybrid_supported"] else "нет",
                    "да" if "dataset-specific calibration" in metadata["limitation_flags"] else "нет",
                ],
            }
        )

    return [
        {
            "id": "diag-evolution",
            "type": "flow",
            "title": "Эволюция версий V1→V7",
            "description": "Переход от исторических proxy-ориентированных формул к негормональным и затем к gated/Sumigron-архитектурам.",
            "rows": [version_nodes[:4], version_nodes[4:]],
        },
        {
            "id": "diag-pipeline",
            "type": "flow",
            "title": "Общий pipeline",
            "description": "Общая цепочка обработки для всех современных версий.",
            "rows": [[
                {"label": "dataset", "bucket": "warning"},
                {"label": "segments / windows", "bucket": "warning"},
                {"label": "features", "bucket": "warning"},
                {"label": "components", "bucket": "warning"},
                {"label": "IIS", "bucket": "success"},
            ]],
        },
        {
            "id": "diag-modes",
            "type": "flow",
            "title": "Режимы данных",
            "description": "Три режима доступа к данным и компенсации пропусков.",
            "rows": [[
                {"label": "strict", "bucket": "success"},
                {"label": "proxy", "bucket": "danger"},
                {"label": "hybrid", "bucket": "warning"},
            ]],
        },
        {
            "id": "diag-arch",
            "type": "matrix",
            "title": "Сравнение архитектур V4/V5/V6/V7",
            "description": "Упрощённая карта того, какие архитектурные идеи добавлялись поверх практического ядра.",
            "columns": ["Гормоны", "Нелинейная геометрия", "Gated-режимы", "3D слой", "Sumigron"],
            "rows": [
                {"label": "V4", "values": ["нет", "частично", "нет", "частично", "нет"]},
                {"label": "V5", "values": ["нет", "да", "нет", "частично", "нет"]},
                {"label": "V6", "values": ["нет", "да", "да", "частично", "нет"]},
                {"label": "V7 beta", "values": ["нет", "да", "да", "да", "да"]},
            ],
        },
        {
            "id": "diag-v7",
            "type": "flow",
            "title": "Архитектура V7 beta",
            "description": "Новая ветка: структурный оператор до gated-слоя и 3D-интерпретации.",
            "rows": [[
                {"label": "window / subterms", "bucket": "warning"},
                {"label": "Sumigron", "bucket": "warning"},
                {"label": "A / Gamma / V / Q", "bucket": "warning"},
                {"label": "gated IIS", "bucket": "warning"},
                {"label": "RES / DYN", "bucket": "success"},
            ]],
        },
        {
            "id": "diag-sumigron",
            "type": "flow",
            "title": "Внутренняя логика Sumigron",
            "description": "Разделение окна на уровень, структуру и энергию с attentive-взвешиванием.",
            "rows": [[
                {"label": "уровень", "bucket": "warning"},
                {"label": "структура", "bucket": "warning"},
                {"label": "энергия", "bucket": "warning"},
                {"label": "attentive weighting", "bucket": "warning"},
                {"label": "структурный скаляр", "bucket": "success"},
            ]],
        },
        {
            "id": "diag-applicability",
            "type": "matrix",
            "title": "Матрица применимости версий",
            "description": "Явное разделение статической, динамической и proxy/hybrid-применимости.",
            "columns": ["static", "dynamic", "proxy", "hybrid", "calibrated"],
            "rows": applicability_rows,
        },
    ]
