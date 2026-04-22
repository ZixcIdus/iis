"""Генерирует единый PDF и автономный HTML-отчёт по IISVersion1–IISVersion7."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

from build_iis_report_bundle import build_bundle
from reporting.iis_report_common import (
    BUNDLE_JSON_PATH,
    HTML_PATH,
    PDF_PATH,
    PDF_TEX_PATH,
    VERSION_ORDER,
    VERSION_SHORT,
    escape_tex,
    format_float,
    format_int,
    html_badge,
    render_diagram_html,
    render_diagram_tex,
    tex_badge,
)

HTML_PATH_MOCHA = Path(__file__).resolve().parent / "docs" / "iis_full_report_mocha.html"


def _load_bundle() -> dict[str, Any]:
    if BUNDLE_JSON_PATH.exists():
        return json.loads(BUNDLE_JSON_PATH.read_text(encoding="utf-8"))
    return build_bundle()


def _status_badge(bucket: str) -> str:
    mapping = {
        "success": ("рабочее", "success"),
        "warning": ("ограничено", "warning"),
        "danger": ("legacy/слабое", "danger"),
    }
    text, actual_bucket = mapping.get(bucket, ("статус", "warning"))
    return html_badge(text, actual_bucket)


def _tex_status_badge(bucket: str) -> str:
    mapping = {
        "success": ("рабочее", "success"),
        "warning": ("ограничено", "warning"),
        "danger": ("legacy/слабое", "danger"),
    }
    text, actual_bucket = mapping.get(bucket, ("статус", "warning"))
    return tex_badge(text, actual_bucket)


def _build_html(bundle: dict[str, Any]) -> str:
    versions = {item["name"]: item for item in bundle["versions"]}
    scenario_rows = bundle["scenario_rows"]
    scenario_best = bundle["scenario_best"]
    state_rows = bundle["state_confirmation"]
    compatibility = bundle["compatibility"]
    experiments = bundle["experiments"]

    nav = """
    <nav class="sidebar">
      <h1>IIS Pack</h1>
      <a href="#overview">Сводка</a>
      <a href="#modes">Что к чему</a>
      <a href="#diagrams">Схемы</a>
      <a href="#versions">Версии V1–V7</a>
      <a href="#compatibility">Совместимость</a>
      <a href="#tests">Матрица тестов</a>
      <a href="#best">Лучшие версии</a>
      <a href="#states">Число состояний</a>
      <a href="#experiments">Эксперименты</a>
      <a href="#limits">Ограничения</a>
      <a href="#conclusion">Заключение</a>
    </nav>
    """

    overview_cards = []
    strongest = bundle["overview"].get("strongest_real_scenario") or {}
    overview_cards.append(
        f"""
        <div class="summary-card">
          <div class="summary-label">Лучшая практическая линия</div>
          <div class="summary-value">{versions[bundle['overview']['current_best_practical_version']]['short']}</div>
          <div class="summary-note">{versions[bundle['overview']['current_best_practical_version']]['title']}</div>
        </div>
        """
    )
    overview_cards.append(
        f"""
        <div class="summary-card">
          <div class="summary-label">Лучшая теоретическая линия</div>
          <div class="summary-value">{bundle['overview']['current_best_theoretical_line']}</div>
          <div class="summary-note">Экспериментальная ветка со структурным оператором.</div>
        </div>
        """
    )
    overview_cards.append(
        f"""
        <div class="summary-card">
          <div class="summary-label">Самый сильный реальный сценарий</div>
          <div class="summary-value">{strongest.get('scenario_label', '—')}</div>
          <div class="summary-note">{VERSION_SHORT.get(strongest.get('version'), strongest.get('version', '—'))} / effect={format_float(strongest.get('effect_size'))}</div>
        </div>
        """
    )

    modes_html = "".join(
        f"""
        <div class="mode-card">
          <h3>{item['title']}</h3>
          <p>{item['body']}</p>
        </div>
        """
        for item in bundle["mode_explanations"]
    )

    diagrams_html = "".join(render_diagram_html(spec) for spec in bundle["diagrams"])

    version_cards = []
    for version_name in VERSION_ORDER:
        version = versions[version_name]
        bucket = version["status_bucket"]
        flags = "".join(f"<li>{flag}</li>" for flag in version["limitation_flags"]) or "<li>Явных флагов нет.</li>"
        inputs = "".join(f"<li>{entry}</li>" for entry in version["inputs"])
        version_cards.append(
            f"""
            <details class="version-card version-card-{bucket}" id="{version['short'].lower()}">
              <summary>
                <span class="version-head">{version['short']} · {version['title']}</span>
                {_status_badge(bucket)}
              </summary>
              <div class="version-body">
                <div class="formula-box"><code>{version['formula_latex']}</code></div>
                <p><strong>Назначение:</strong> {version['purpose']}</p>
                <p><strong>Сильная сторона:</strong> {version['strength']}</p>
                <p><strong>Слабое место:</strong> {version['weakness']}</p>
                <p><strong>Уровень универсальности:</strong> {version['universality']}</p>
                <p><strong>Статус:</strong> {version['status_note']}</p>
                <div class="two-col">
                  <div>
                    <h4>Входные компоненты</h4>
                    <ul>{inputs}</ul>
                  </div>
                  <div>
                    <h4>Ограничения</h4>
                    <ul>{flags}</ul>
                  </div>
                </div>
              </div>
            </details>
            """
        )

    compatibility_rows_html = []
    for row in compatibility:
        compatibility_rows_html.append(
            "<tr>"
            f"<th>{row['short']}</th>"
            f"<td>{'да' if row['static_supported'] else 'нет'}</td>"
            f"<td>{'да' if row['dynamic_supported'] else 'нет'}</td>"
            f"<td>{'да' if row['proxy_supported'] else 'нет'}</td>"
            f"<td>{'да' if row['hybrid_supported'] else 'нет'}</td>"
            f"<td>{', '.join(row['limitation_flags']) or '—'}</td>"
            f"<td>{row['status_note']}</td>"
            "</tr>"
        )

    matrix_rows_html = []
    for row in scenario_rows:
        matrix_rows_html.append(
            "<tr>"
            f"<td>{row['scenario_label']}</td>"
            f"<td>{row['version_short']}</td>"
            f"<td>{format_float(row.get('coverage'))}</td>"
            f"<td>{format_int(row.get('valid_segments'))}</td>"
            f"<td>{format_float(row.get('effect_size'))}</td>"
            f"<td>{format_float(row.get('distribution_overlap'))}</td>"
            f"<td>{format_float(row.get('relative_sensitivity'))}</td>"
            f"<td>{row.get('reliability_level') or '—'}</td>"
            f"<td>{format_int(row.get('consensus_supported_k'))}</td>"
            f"<td>{row.get('capacity_verdict') or '—'}</td>"
            f"<td>{row.get('dynamic_state') or '—'}</td>"
            "</tr>"
        )

    best_rows_html = []
    for row in scenario_best:
        best_rows_html.append(
            "<tr>"
            f"<td>{row['scenario_label']}</td>"
            f"<td>{row['version_short']}</td>"
            f"<td>{row.get('reliability_level') or '—'}</td>"
            f"<td>{format_float(row.get('effect_size'))}</td>"
            f"<td>{format_float(row.get('distribution_overlap'))}</td>"
            f"<td>{format_float(row.get('relative_sensitivity'))}</td>"
            f"<td>{row.get('capacity_verdict') or '—'}</td>"
            "</tr>"
        )

    state_rows_html = []
    for row in state_rows:
        state_rows_html.append(
            "<tr>"
            f"<td>{row['scenario_label']}</td>"
            f"<td>{row['best_version']}</td>"
            f"<td>{row['confirmed_states']}</td>"
            f"<td>{row['possible_ceiling']}</td>"
            f"<td>{row['note']}</td>"
            "</tr>"
        )

    experiments_html = f"""
    <div class="experiment-grid">
      <article class="experiment-card">
        <h3>DREAMER · V5 calibration</h3>
        <p>Текущий global objective: {format_float(experiments.get('dreamer_v5_calibration', {}).get('objective_current_global'))}</p>
        <p>Best candidate: {format_float(experiments.get('dreamer_v5_calibration', {}).get('objective_best_candidate'))}</p>
        <p>{experiments.get('dreamer_v5_calibration', {}).get('note', 'Нет данных.')}</p>
      </article>
      <article class="experiment-card">
        <h3>DREAMER · V7 beta calibration</h3>
        <p>Candidate mode: {experiments.get('dreamer_v7_calibration', {}).get('candidate_mode', '—')}</p>
        <p>Current global objective: {format_float(experiments.get('dreamer_v7_calibration', {}).get('objective_current_global'))}</p>
        <p>Best candidate: {format_float(experiments.get('dreamer_v7_calibration', {}).get('objective_best_candidate'))}</p>
        <p>{experiments.get('dreamer_v7_calibration', {}).get('note', 'Нет данных.')}</p>
      </article>
      <article class="experiment-card">
        <h3>Sumigron synthetic benchmark</h3>
        <p>Structure-driven / Sumigron ε²: {format_float(experiments.get('sumigron_synthetic', {}).get('structure_driven_sumigron', {}).get('epsilon_squared'))}</p>
        <p>Structure-driven / sigmoid ε²: {format_float(experiments.get('sumigron_synthetic', {}).get('structure_driven_sigmoid', {}).get('epsilon_squared'))}</p>
        <p>{experiments.get('sumigron_synthetic', {}).get('note', 'Нет данных.')}</p>
      </article>
    </div>
    """

    limit_items = []
    for version_name in VERSION_ORDER:
        version = versions[version_name]
        flags = ", ".join(version["limitation_flags"]) or "без специальных флагов"
        limit_items.append(f"<li><strong>{version['short']}:</strong> {version['weakness']} Ограничения: {flags}.</li>")

    html_text = f"""<!doctype html>
<html lang="ru">
<head>
  <meta charset="utf-8">
  <title>{bundle['report_title']}</title>
  <style>
    :root {{
      --bg: #11111b;
      --paper: rgba(30, 30, 46, 0.76);
      --ink: #cdd6f4;
      --muted: #a6adc8;
      --line: rgba(205, 214, 244, 0.10);
      --accent: #89b4fa;
      --accent-soft: rgba(137, 180, 250, 0.16);
      --success: #a6e3a1;
      --success-soft: rgba(166, 227, 161, 0.12);
      --warning: #f9e2af;
      --warning-soft: rgba(249, 226, 175, 0.12);
      --danger: #f38ba8;
      --danger-soft: rgba(243, 139, 168, 0.14);
      --surface: rgba(49, 50, 68, 0.82);
      --sidebar: rgba(17, 17, 27, 0.68);
      --sidebar-ink: #cdd6f4;
      --shadow: 0 22px 60px rgba(0, 0, 0, 0.42);
      --glow-accent: 0 0 0 1px rgba(137, 180, 250, 0.10), 0 0 28px rgba(137, 180, 250, 0.10);
      --glow-rose: 0 0 24px rgba(245, 194, 231, 0.08);
      --radius: 18px;
      font-family: "Segoe UI", "Noto Sans", sans-serif;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      background:
        radial-gradient(circle at top left, rgba(203, 166, 247, 0.18), transparent 26%),
        radial-gradient(circle at top right, rgba(137, 180, 250, 0.16), transparent 24%),
        linear-gradient(180deg, #11111b 0%, #181825 100%);
      color: var(--ink);
      color-scheme: dark;
    }}
    .layout {{ display: grid; grid-template-columns: 270px 1fr; min-height: 100vh; }}
    .sidebar {{
      position: sticky; top: 0; align-self: start; height: 100vh; padding: 28px 22px; background: var(--sidebar);
      color: var(--sidebar-ink); border-right: 1px solid rgba(255,255,255,0.08); backdrop-filter: blur(22px) saturate(140%);
    }}
    .sidebar h1 {{ margin: 0 0 24px; font-size: 1.35rem; letter-spacing: 0.04em; text-transform: uppercase; }}
    .sidebar a {{ display: block; color: #e6e9ef; text-decoration: none; margin: 10px 0; font-size: 0.98rem; padding: 6px 10px; border-radius: 10px; transition: background .18s ease, box-shadow .18s ease, transform .18s ease; }}
    .sidebar a:hover {{ color: #ffffff; background: rgba(137, 180, 250, 0.10); box-shadow: var(--glow-accent); transform: translateX(2px); }}
    .content {{ padding: 34px; max-width: 1400px; }}
    .hero {{
      padding: 30px; background: var(--paper); border: 1px solid var(--line); border-radius: calc(var(--radius) + 4px);
      box-shadow: var(--shadow), var(--glow-rose); margin-bottom: 28px; backdrop-filter: blur(18px) saturate(125%);
    }}
    .hero h2 {{ margin: 0 0 12px; font-size: 2rem; text-shadow: 0 0 24px rgba(137, 180, 250, 0.14); }}
    .hero p {{ margin: 10px 0; color: var(--muted); line-height: 1.55; }}
    .summary-grid, .experiment-grid, .mode-grid, .two-col {{
      display: grid; gap: 18px;
    }}
    .summary-grid {{ grid-template-columns: repeat(auto-fit, minmax(230px, 1fr)); margin-top: 20px; }}
    .mode-grid {{ grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); }}
    .experiment-grid {{ grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); }}
    .two-col {{ grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); }}
    .summary-card, .mode-card, .experiment-card, .section-card {{
      background: var(--paper); border: 1px solid var(--line); border-radius: var(--radius); padding: 20px; box-shadow: var(--shadow), var(--glow-rose); backdrop-filter: blur(18px) saturate(125%);
    }}
    .summary-label {{ color: var(--muted); font-size: 0.85rem; text-transform: uppercase; letter-spacing: 0.08em; }}
    .summary-value {{ font-size: 1.4rem; font-weight: 700; margin-top: 10px; }}
    .summary-note, .experiment-card p, .mode-card p, .section-card p {{ color: var(--muted); line-height: 1.55; }}
    .badge {{ display: inline-block; padding: 4px 10px; border-radius: 999px; font-size: 0.78rem; font-weight: 700; }}
    .badge-success {{ background: var(--success-soft); color: var(--success); }}
    .badge-warning {{ background: var(--warning-soft); color: var(--warning); }}
    .badge-danger {{ background: var(--danger-soft); color: var(--danger); }}
    section.anchor {{ margin-top: 24px; }}
    h3.section-title {{ margin: 30px 0 16px; font-size: 1.55rem; }}
    .diagram-block {{
      background: var(--paper); border: 1px solid var(--line); border-radius: var(--radius); padding: 20px; margin-bottom: 16px; box-shadow: var(--shadow), var(--glow-rose);
    }}
    .diagram-description {{ color: var(--muted); margin-bottom: 14px; }}
    .flow-row {{ display: flex; flex-wrap: wrap; align-items: center; justify-content: center; gap: 10px; }}
    .flow-row-down {{ margin: 8px 0; }}
    .flow-down, .flow-arrow {{ font-size: 1.2rem; color: var(--muted); }}
    .flow-node {{
      min-width: 160px; max-width: 240px; padding: 12px 14px; border-radius: 14px; text-align: center; font-weight: 600;
      border: 1px solid var(--line);
    }}
    .flow-node-success {{ background: var(--success-soft); color: var(--success); }}
    .flow-node-warning {{
      background: rgba(249, 226, 175, 0.18);
      color: #f9e2af;
      border-color: rgba(249, 226, 175, 0.22);
      text-shadow: 0 0 18px rgba(249, 226, 175, 0.16);
      box-shadow: inset 0 1px 0 rgba(255,255,255,0.03), 0 0 24px rgba(249, 226, 175, 0.08);
    }}
    .flow-node-danger {{ background: var(--danger-soft); color: var(--danger); }}
    .diagram-matrix, .data-table {{
      width: 100%; border-collapse: collapse; overflow: hidden; border-radius: 14px; background: var(--paper);
    }}
    .diagram-matrix th, .diagram-matrix td, .data-table th, .data-table td {{
      border: 1px solid var(--line); padding: 10px 12px; vertical-align: top; text-align: left;
    }}
    .diagram-matrix th, .data-table th {{ background: var(--surface); }}
    .matrix-cell-success {{ background: var(--success-soft); }}
    .matrix-cell-warning {{ background: var(--warning-soft); }}
    .matrix-cell-danger {{ background: var(--danger-soft); }}
    .version-card {{
      margin-bottom: 16px; background: var(--paper); border: 1px solid var(--line); border-radius: var(--radius); box-shadow: var(--shadow), var(--glow-rose);
      overflow: hidden;
    }}
    .version-card summary {{
      list-style: none; cursor: pointer; padding: 18px 20px; display: flex; align-items: center; justify-content: space-between; gap: 18px;
      background: rgba(49, 50, 68, 0.92); font-weight: 700;
    }}
    .version-card summary::-webkit-details-marker {{ display: none; }}
    .version-card-danger summary {{ background: rgba(243, 139, 168, 0.16); }}
    .version-card-warning summary {{ background: rgba(249, 226, 175, 0.16); }}
    .version-card-success summary {{ background: rgba(166, 227, 161, 0.16); }}
    .version-body {{ padding: 20px; }}
    .version-head {{ font-size: 1.1rem; }}
    .formula-box {{
      background: rgba(17, 17, 27, 0.96); color: #f5e0dc; border-radius: 14px; padding: 14px 16px; overflow-x: auto; margin-bottom: 14px;
      box-shadow: inset 0 1px 0 rgba(255,255,255,0.04), 0 0 28px rgba(137, 180, 250, 0.06);
      font-family: "Consolas", "Courier New", monospace;
    }}
    ul {{ margin-top: 8px; }}
    @media (max-width: 980px) {{
      .layout {{ grid-template-columns: 1fr; }}
      .sidebar {{ position: relative; height: auto; }}
      .content {{ padding: 20px; }}
    }}
  </style>
</head>
<body>
  <div class="layout">
    {nav}
    <main class="content">
      <section id="overview" class="hero">
        <h2>IIS Full Test + Documentation Pack</h2>
        <p>Единый локальный отчёт по версиям IISVersion1–IISVersion7. Здесь собраны статические и частично динамические тесты, state capacity, эксперименты V5/V7 на DREAMER и synthetic-бенч Sumigron. Слабые места сохранены как есть, без выравнивания под красивые выводы.</p>
        <p><strong>Сильная реальная точка сейчас:</strong> {bundle['overview']['strongest_real_scenario'].get('scenario_label', '—')}.</p>
        <div class="summary-grid">
          {''.join(overview_cards)}
        </div>
      </section>

      <section id="modes" class="anchor">
        <h3 class="section-title">Что К Чему</h3>
        <div class="mode-grid">{modes_html}</div>
      </section>

      <section id="diagrams" class="anchor">
        <h3 class="section-title">Схемы</h3>
        {diagrams_html}
      </section>

      <section id="versions" class="anchor">
        <h3 class="section-title">Версии V1–V7</h3>
        {''.join(version_cards)}
      </section>

      <section id="compatibility" class="anchor">
        <h3 class="section-title">Совместимость</h3>
        <table class="data-table">
          <thead>
            <tr>
              <th>Версия</th><th>static</th><th>dynamic</th><th>proxy</th><th>hybrid</th><th>Флаги</th><th>Статус</th>
            </tr>
          </thead>
          <tbody>{''.join(compatibility_rows_html)}</tbody>
        </table>
      </section>

      <section id="tests" class="anchor">
        <h3 class="section-title">Полная Матрица Тестов</h3>
        <table class="data-table">
          <thead>
            <tr>
              <th>Scenario</th><th>Версия</th><th>Coverage</th><th>Valid</th><th>Effect</th><th>Overlap</th><th>Sens</th><th>Reliability</th><th>k</th><th>Verdict</th><th>Dynamic</th>
            </tr>
          </thead>
          <tbody>{''.join(matrix_rows_html)}</tbody>
        </table>
      </section>

      <section id="best" class="anchor">
        <h3 class="section-title">Лучшие Версии По Сценариям</h3>
        <table class="data-table">
          <thead>
            <tr>
              <th>Scenario</th><th>Лучшая версия</th><th>Reliability</th><th>Effect</th><th>Overlap</th><th>Sens</th><th>Verdict</th>
            </tr>
          </thead>
          <tbody>{''.join(best_rows_html)}</tbody>
        </table>
      </section>

      <section id="states" class="anchor">
        <h3 class="section-title">Что Реально Подтверждается По Числу Состояний</h3>
        <table class="data-table">
          <thead>
            <tr>
              <th>Scenario</th><th>Лучшая версия</th><th>Подтверждённое</th><th>Внутренний потолок</th><th>Примечание</th>
            </tr>
          </thead>
          <tbody>{''.join(state_rows_html)}</tbody>
        </table>
      </section>

      <section id="experiments" class="anchor">
        <h3 class="section-title">Экспериментальные Проверки</h3>
        {experiments_html}
      </section>

      <section id="limits" class="anchor">
        <h3 class="section-title">Ограничения И Честные Пометки</h3>
        <div class="section-card">
          <ul>{''.join(limit_items)}</ul>
          <p><strong>DREAMER:</strong> {bundle['overview']['dreamer_note']}</p>
          <p><strong>Sumigron:</strong> {bundle['overview']['sumigron_note']}</p>
        </div>
      </section>

      <section id="conclusion" class="anchor">
        <h3 class="section-title">Заключение</h3>
        <div class="section-card">
          <p><strong>Текущая лучшая практическая версия:</strong> {versions[bundle['overview']['current_best_practical_version']]['short']} — {versions[bundle['overview']['current_best_practical_version']]['title']}.</p>
          <p><strong>Текущая лучшая теоретическая линия:</strong> {bundle['overview']['current_best_theoretical_line']}.</p>
          <p><strong>Что остаётся гипотезой:</strong> уверенное подтверждение 3+ устойчивых состояний и окончательная универсальность V7/Sumigron на разнородных датасетах.</p>
        </div>
      </section>
    </main>
  </div>
</body>
</html>
"""
    return html_text


def _build_tex(bundle: dict[str, Any]) -> str:
    versions = {item["name"]: item for item in bundle["versions"]}
    diagrams_tex = "\n".join(render_diagram_tex(spec) for spec in bundle["diagrams"])

    version_sections = []
    for version_name in VERSION_ORDER:
        version = versions[version_name]
        inputs_tex = "\n".join(rf"\item {escape_tex(item)}" for item in version["inputs"])
        flags = version["limitation_flags"] or ["явных специальных флагов нет"]
        flags_tex = "\n".join(rf"\item {escape_tex(flag)}" for flag in flags)
        version_sections.append(
            rf"""
\subsection*{{{escape_tex(version['short'])}. {escape_tex(version['title'])} {_tex_status_badge(version['status_bucket'])}}}
\[
{version['formula_latex']}
\]
\textbf{{Назначение.}} {escape_tex(version['purpose'])}

\textbf{{Сильная сторона.}} {escape_tex(version['strength'])}

\textbf{{Слабое место.}} {escape_tex(version['weakness'])}

\textbf{{Уровень универсальности.}} {escape_tex(version['universality'])}

\textbf{{Статус.}} {escape_tex(version['status_note'])}

\textbf{{Входные компоненты:}}
\begin{{itemize}}[leftmargin=1.2cm]
{inputs_tex}
\end{{itemize}}

\textbf{{Ограничения:}}
\begin{{itemize}}[leftmargin=1.2cm]
{flags_tex}
\end{{itemize}}
"""
        )

    comparison_rows = []
    for row in bundle["scenario_rows"]:
        comparison_rows.append(
            " & ".join(
                [
                    escape_tex(str(row["scenario_label"])),
                    escape_tex(str(row["version_short"])),
                    escape_tex(format_float(row.get("coverage"))),
                    escape_tex(format_int(row.get("valid_segments"))),
                    escape_tex(format_float(row.get("effect_size"))),
                    escape_tex(format_float(row.get("distribution_overlap"))),
                    escape_tex(format_float(row.get("relative_sensitivity"))),
                    escape_tex(str(row.get("reliability_level") or "—")),
                    escape_tex(format_int(row.get("consensus_supported_k"))),
                    escape_tex(str(row.get("capacity_verdict") or "—")),
                    escape_tex(str(row.get("dynamic_state") or "—")),
                ]
            )
            + r" \\"
        )

    best_rows = []
    for row in bundle["scenario_best"]:
        best_rows.append(
            " & ".join(
                [
                    escape_tex(str(row["scenario_label"])),
                    escape_tex(str(row["version_short"])),
                    escape_tex(str(row.get("reliability_level") or "—")),
                    escape_tex(format_float(row.get("effect_size"))),
                    escape_tex(format_float(row.get("distribution_overlap"))),
                    escape_tex(format_float(row.get("relative_sensitivity"))),
                    escape_tex(str(row.get("capacity_verdict") or "—")),
                ]
            )
            + r" \\"
        )

    state_rows = []
    for row in bundle["state_confirmation"]:
        state_rows.append(
            " & ".join(
                [
                    escape_tex(str(row["scenario_label"])),
                    escape_tex(str(row["best_version"])),
                    escape_tex(str(row["confirmed_states"])),
                    escape_tex(str(row["possible_ceiling"])),
                    escape_tex(str(row["note"])),
                ]
            )
            + r" \\"
        )

    compatibility_rows = []
    for row in bundle["compatibility"]:
        compatibility_rows.append(
            " & ".join(
                [
                    escape_tex(str(row["short"])),
                    escape_tex("да" if row["static_supported"] else "нет"),
                    escape_tex("да" if row["dynamic_supported"] else "нет"),
                    escape_tex("да" if row["proxy_supported"] else "нет"),
                    escape_tex("да" if row["hybrid_supported"] else "нет"),
                    escape_tex(", ".join(row["limitation_flags"]) or "—"),
                ]
            )
            + r" \\"
        )

    mode_rows = []
    for mode in bundle["mode_explanations"]:
        mode_rows.append(
            " & ".join([escape_tex(mode["title"]), escape_tex(mode["body"])]) + r" \\"
        )

    experiments = bundle["experiments"]
    strongest = bundle["overview"].get("strongest_real_scenario") or {}

    tex = rf"""\documentclass[11pt,a4paper]{{article}}
\usepackage[T2A]{{fontenc}}
\usepackage[utf8]{{inputenc}}
\usepackage[russian]{{babel}}
\usepackage{{amsmath,amssymb,amsfonts}}
\usepackage{{geometry}}
\usepackage{{array}}
\usepackage{{booktabs}}
\usepackage{{longtable}}
\usepackage{{pdflscape}}
\usepackage{{enumitem}}
\usepackage{{hyperref}}
\usepackage[table]{{xcolor}}
\geometry{{margin=1.8cm}}
\setlength{{\parindent}}{{0pt}}
\setlength{{\parskip}}{{0.45em}}
\newcommand{{\sgop}}{{\mathop{{\sum\!\!\!\int_{{\bowtie}}}}}}

\title{{IIS Full Test + Documentation Pack}}
\author{{Проект анализа интегрального индекса состояния}}
\date{{\today}}

\begin{{document}}
\maketitle

\section{{Введение}}
Этот отчёт объединяет в одном документе версии \texttt{{IISVersion1}}--\texttt{{IISVersion7}}, полную матрицу доступных тестовых сценариев, state capacity, Dreamer-калибровки и synthetic-бенч оператора Sumigron.

Главный принцип документа: слабые результаты не скрываются. Если версия не универсальна, если режим не адаптирован к динамике или если на конкретном датасете видны только tentative-сигналы, это фиксируется явно.

\textbf{{Лучшая практическая линия сейчас:}} {escape_tex(bundle['overview']['current_best_practical_version'])}.\\
\textbf{{Лучшая теоретическая линия:}} {escape_tex(bundle['overview']['current_best_theoretical_line'])}.\\
\textbf{{Самый сильный реальный сценарий:}} {escape_tex(str(strongest.get('scenario_label', '—')))} ({escape_tex(VERSION_SHORT.get(strongest.get('version', ''), str(strongest.get('version', '—'))))}).

\section{{Что такое strict / proxy / hybrid}}
\begin{{center}}
\begin{{tabular}}{{p{{2.2cm}}p{{11.8cm}}}}
\toprule
Режим & Смысл \\
\midrule
{chr(10).join(mode_rows)}
\bottomrule
\end{{tabular}}
\end{{center}}

\section{{Карта версий и схемы}}
{diagrams_tex}

\section{{Полная документация по каждой версии}}
{chr(10).join(version_sections)}

\section{{Матрица совместимости}}
\begin{{center}}
\begin{{tabular}}{{lccccc}}
\toprule
Версия & static & dynamic & proxy & hybrid & Флаги \\
\midrule
{chr(10).join(compatibility_rows)}
\bottomrule
\end{{tabular}}
\end{{center}}

\section{{Сравнение версий на тестах}}
\subsection*{{Полная матрица по всем dataset/mode}}
\begin{{landscape}}
\scriptsize
\begin{{longtable}}{{p{{2.8cm}}p{{0.8cm}}rrrrrp{{1.2cm}}rp{{4.4cm}}p{{1.6cm}}}}
\toprule
Scenario & V & Cov & Valid & Eff & Ovl & Sens & Rel & k & Verdict & Dyn \\
\midrule
\endfirsthead
\toprule
Scenario & V & Cov & Valid & Eff & Ovl & Sens & Rel & k & Verdict & Dyn \\
\midrule
\endhead
{chr(10).join(comparison_rows)}
\bottomrule
\end{{longtable}}
\normalsize
\end{{landscape}}

\subsection*{{Лучшие версии по сценариям}}
\begin{{center}}
\begin{{tabular}}{{p{{2.8cm}}p{{0.8cm}}p{{1.3cm}}rrrrp{{4.2cm}}}}
\toprule
Scenario & V & Rel & Eff & Ovl & Sens & Verdict \\
\midrule
{chr(10).join(best_rows)}
\bottomrule
\end{{tabular}}
\end{{center}}

\subsection*{{Что реально подтверждается по числу состояний}}
\begin{{center}}
\begin{{tabular}}{{p{{2.8cm}}p{{0.8cm}}p{{4.2cm}}p{{1.3cm}}p{{6.0cm}}}}
\toprule
Scenario & V & Подтверждённое & Потолок & Примечание \\
\midrule
{chr(10).join(state_rows)}
\bottomrule
\end{{tabular}}
\end{{center}}

\section{{V7 beta и Sumigron}}
\textbf{{Зачем появился оператор.}} Sumigron нужен как попытка переставить нелинейность раньше по цепочке: не сжимать окно в один ранний bounded-ответ, а сначала сохранить уровень, структуру и энергию.

\textbf{{Что делает V7 beta.}} Архитектура \(\text{{window}} \to \sgop \to A,\Gamma,V,Q \to IIS,RES,DYN\) сохраняет gated-слой и трёхмерную интерпретацию, но меняет раннюю агрегацию.

\textbf{{Где уже дал пользу.}} На части строгих сценариев и в synthetic-бенче Sumigron/V7 показывают осмысленный выигрыш.

\textbf{{Где пока не дал прорыва.}} На DREAMER V7 полезна, но не делает резкого скачка по подтверждённому числу состояний; кроме того, для DREAMER уже используется dataset-specific override, поэтому версия ещё не универсальна.

\section{{Экспериментальные проверки}}
\subsection*{{Dreamer V5 calibration}}
Current global objective: {escape_tex(format_float(experiments.get('dreamer_v5_calibration', {}).get('objective_current_global')))}.\\
Best candidate objective: {escape_tex(format_float(experiments.get('dreamer_v5_calibration', {}).get('objective_best_candidate')))}.\\
{escape_tex(experiments.get('dreamer_v5_calibration', {}).get('note', 'Нет данных.'))}

\subsection*{{Dreamer V7 beta calibration}}
Candidate mode: {escape_tex(str(experiments.get('dreamer_v7_calibration', {}).get('candidate_mode', '—')))}.\\
Current global objective: {escape_tex(format_float(experiments.get('dreamer_v7_calibration', {}).get('objective_current_global')))}.\\
Best candidate objective: {escape_tex(format_float(experiments.get('dreamer_v7_calibration', {}).get('objective_best_candidate')))}.\\
{escape_tex(experiments.get('dreamer_v7_calibration', {}).get('note', 'Нет данных.'))}

\subsection*{{Synthetic Sumigron benchmark}}
Structure-driven / Sumigron \(\epsilon^2\): {escape_tex(format_float(experiments.get('sumigron_synthetic', {}).get('structure_driven_sumigron', {}).get('epsilon_squared')))}.\\
Structure-driven / sigmoid \(\epsilon^2\): {escape_tex(format_float(experiments.get('sumigron_synthetic', {}).get('structure_driven_sigmoid', {}).get('epsilon_squared')))}.\\
{escape_tex(experiments.get('sumigron_synthetic', {}).get('note', 'Нет данных.'))}

\section{{Ограничения и честные пометки}}
\begin{{itemize}}[leftmargin=1.2cm]
{chr(10).join(rf"\item \textbf{{{escape_tex(versions[name]['short'])}}}: {escape_tex(versions[name]['weakness'])} Ограничения: {escape_tex(', '.join(versions[name]['limitation_flags']) or 'без специальных флагов')}." for name in VERSION_ORDER)}
\end{{itemize}}

\textbf{{DREAMER.}} {escape_tex(bundle['overview']['dreamer_note'])}

\textbf{{Sumigron.}} {escape_tex(bundle['overview']['sumigron_note'])}

\section{{Заключение}}
\textbf{{Текущая лучшая практическая версия.}} {escape_tex(versions[bundle['overview']['current_best_practical_version']]['short'])} --- {escape_tex(versions[bundle['overview']['current_best_practical_version']]['title'])}.

\textbf{{Текущая лучшая теоретическая линия.}} {escape_tex(bundle['overview']['current_best_theoretical_line'])}.

\textbf{{Что остаётся гипотезой.}} Подтверждение 3+ устойчивых состояний на разнородных наборах, а также финальная универсальность V7/Sumigron без dataset-specific override.

\end{{document}}
"""
    return tex


def _compile_pdf(tex_path: Path) -> None:
    command = ["pdflatex", "-interaction=nonstopmode", "-halt-on-error", tex_path.name]
    for _ in range(2):
        subprocess.run(command, cwd=tex_path.parent, check=True, capture_output=True, text=True)


def main() -> int:
    bundle = _load_bundle()
    html_text = _build_html(bundle)
    tex_text = _build_tex(bundle)

    HTML_PATH.write_text(html_text, encoding="utf-8")
    HTML_PATH_MOCHA.write_text(html_text, encoding="utf-8")
    PDF_TEX_PATH.write_text(tex_text, encoding="utf-8")
    _compile_pdf(PDF_TEX_PATH)

    print(f"html={HTML_PATH}")
    print(f"html_mocha={HTML_PATH_MOCHA}")
    print(f"tex={PDF_TEX_PATH}")
    print(f"pdf={PDF_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
