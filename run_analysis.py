"""CLI-скрипт для сравнения трёх версий ИИС на физиологических датасетах."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

import pandas as pd

from config.settings import (
    DATASET_DEFAULTS,
    DYNAMIC_WINDOW_DEFAULTS,
    LOG_DIR,
    OUTPUT_DIR,
    PLOTS_DIR,
    PROCESSING_MODES,
    SUPPORTED_DATASETS,
    ensure_runtime_directories,
)
from features.common import expand_segments_dataframe, extract_features_dataframe, summarize_available_features
from loaders.case_loader import CASELoader
from loaders.deap_loader import DEAPLoader
from loaders.dreamer_loader import DREAMERLoader
from loaders.eva_med_loader import EVAMEDLoader
from loaders.openneuro_bcmi_loader import OpenNeuroBCMILoader
from loaders.wesad_loader import WESADLoader
from models.comparison import IISComparison
from models.dynamic_analysis import CausalDynamicAnalyzer
from models.iis_v1 import IISVersion1
from models.iis_v2 import IISVersion2
from models.iis_v3 import IISVersion3
from models.iis_v4 import IISVersion4
from models.iis_v5 import IISVersion5
from models.iis_v6 import IISVersion6
from models.iis_v7 import IISVersion7
from models.resource_state_map import IISResourceStateMap

LOGGER = logging.getLogger("iis_analysis")
CURRENT_LOG_PATH: Path | None = None


def _sanitize_log_label(value: str) -> str:
    """Делает безопасный суффикс имени лог-файла."""

    cleaned = "".join(character.lower() if character.isalnum() else "_" for character in value)
    while "__" in cleaned:
        cleaned = cleaned.replace("__", "_")
    return cleaned.strip("_") or "run"


def get_current_log_path() -> Path | None:
    """Возвращает путь к лог-файлу текущего запуска."""

    return CURRENT_LOG_PATH


def emit_progress(
    progress_callback: Callable[[dict[str, Any]], None] | None,
    *,
    stage: str,
    message: str,
    stage_start: float,
    stage_end: float,
    current: float | None = None,
    total: float | None = None,
    extra: dict[str, Any] | None = None,
) -> None:
    """Передаёт в GUI нормализованный прогресс текущего этапа."""

    if progress_callback is None:
        return

    fraction = stage_end
    if current is not None and total is not None and total > 0:
        local_fraction = min(max(float(current) / float(total), 0.0), 1.0)
        fraction = stage_start + (stage_end - stage_start) * local_fraction
    payload = {
        "stage": stage,
        "message": message,
        "fraction": min(max(fraction, 0.0), 1.0),
        "current": current,
        "total": total,
    }
    if extra:
        payload.update(extra)
    progress_callback(payload)


def parse_args() -> argparse.Namespace:
    """Разбирает аргументы командной строки."""

    parser = argparse.ArgumentParser(description="Сравнение версий модели ИИС.")
    parser.add_argument("--dataset", required=True, choices=[*SUPPORTED_DATASETS, "all"], help="Название датасета.")
    parser.add_argument("--mode", required=True, choices=PROCESSING_MODES, help="Режим обработки пропусков.")
    parser.add_argument(
        "--data-root",
        default=None,
        help="Путь к корню датасета. Для --dataset all ожидается каталог с подпапками WESAD/DREAMER/DEAP/CASE/EVA-MED/DS002722/DS002724.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        help="Уровень логирования.",
    )
    parser.add_argument(
        "--dynamic",
        action="store_true",
        help="Включить динамическую причинную обработку с разбиением на подокна.",
    )
    parser.add_argument(
        "--window-seconds",
        type=float,
        default=None,
        help="Размер динамического окна в секундах. Если не задан, используется dataset-specific default.",
    )
    parser.add_argument(
        "--step-seconds",
        type=float,
        default=None,
        help="Шаг динамического окна в секундах. Если не задан, используется dataset-specific default.",
    )
    parser.add_argument(
        "--focus-version",
        default="IISVersion6",
        choices=("IISVersion1", "IISVersion2", "IISVersion3", "IISVersion4", "IISVersion5", "IISVersion6", "IISVersion7"),
        help="Версия модели для динамического причинного анализа.",
    )
    return parser.parse_args()


def configure_logging(log_level: str, run_label: str | None = None, use_stdout: bool = True) -> Path:
    """Настраивает консольное и файловое логирование."""

    ensure_runtime_directories()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    label_suffix = f"_{_sanitize_log_label(run_label)}" if run_label else ""
    log_path = LOG_DIR / f"analysis_{timestamp}{label_suffix}.log"
    latest_log_path = LOG_DIR / "analysis.log"
    handlers: list[logging.Handler] = [
        logging.FileHandler(log_path, encoding="utf-8"),
        logging.FileHandler(latest_log_path, mode="w", encoding="utf-8"),
    ]
    if use_stdout:
        handlers.insert(0, logging.StreamHandler(sys.stdout))
    logging.basicConfig(
        level=getattr(logging, log_level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        handlers=handlers,
        force=True,
    )
    global CURRENT_LOG_PATH
    CURRENT_LOG_PATH = log_path
    LOGGER.info("Лог текущего запуска: %s", log_path)
    LOGGER.info("Последний запуск дублируется в: %s", latest_log_path)
    return log_path


def resolve_dataset_path(dataset: str, cli_data_root: str | None, global_mode: bool) -> Path:
    """Определяет путь до конкретного датасета."""

    if cli_data_root is None:
        return Path(DATASET_DEFAULTS[dataset]["default_root"]).resolve()

    root = Path(cli_data_root).expanduser().resolve()
    if global_mode:
        return root / Path(DATASET_DEFAULTS[dataset]["default_root"]).name
    return root


def build_loader(
    dataset: str,
    dataset_path: Path,
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
) -> Any:
    """Возвращает загрузчик для выбранного датасета."""

    if dataset == "wesad":
        return WESADLoader(root=dataset_path)
    if dataset == "dreamer":
        return DREAMERLoader(root=dataset_path)
    if dataset == "deap":
        return DEAPLoader(root=dataset_path)
    if dataset == "case":
        return CASELoader(root=dataset_path)
    if dataset == "eva_med":
        return EVAMEDLoader(root=dataset_path)
    if dataset in {"ds002722", "ds002724"}:
        return OpenNeuroBCMILoader(root=dataset_path, dataset_name=dataset, progress_callback=progress_callback)
    raise ValueError(f"Неподдерживаемый датасет: {dataset}")


def save_json(path: Path, payload: dict[str, Any]) -> None:
    """Сохраняет JSON-файл в UTF-8."""

    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def summarize_component_provenance(version_df: pd.DataFrame) -> dict[str, dict[str, int]]:
    """Подсчитывает direct/proxy/unavailable по компонентам."""

    summary: dict[str, dict[str, int]] = {}
    for component in ("A", "Gamma", "H", "V", "Q", "K"):
        series = version_df.get(f"prov_{component}")
        if series is None:
            continue
        summary[component] = {
            "direct": int((series == "direct").sum()),
            "proxy": int((series == "proxy").sum()),
            "unavailable": int((series == "unavailable").sum()),
        }
    return summary


def print_console_report(
    dataset: str,
    mode: str,
    features_df: pd.DataFrame,
    results_df: pd.DataFrame,
    comparison_df: pd.DataFrame,
    summary: dict[str, Any],
) -> None:
    """Печатает краткий отчёт в консоль."""

    LOGGER.info("===== Отчёт по датасету %s / режим %s =====", dataset.upper(), mode)
    LOGGER.info("Доступные признаки:")
    for feature_name, stats in summarize_available_features(features_df).items():
        LOGGER.info("  %s: %d сегм. (доля %.2f)", feature_name, stats["valid_segments"], stats["share"])

    LOGGER.info("Происхождение блоков моделей:")
    for version_name, version_df in results_df.groupby("version", sort=True):
        provenance = summarize_component_provenance(version_df)
        direct_blocks = [name for name, stats in provenance.items() if stats["direct"] > 0]
        proxy_blocks = [name for name, stats in provenance.items() if stats["proxy"] > 0]
        unavailable_blocks = [name for name, stats in provenance.items() if stats["direct"] == 0 and stats["proxy"] == 0]
        LOGGER.info("  %s | direct: %s", version_name, ", ".join(direct_blocks) if direct_blocks else "нет")
        LOGGER.info("  %s | proxy: %s", version_name, ", ".join(proxy_blocks) if proxy_blocks else "нет")
        LOGGER.info("  %s | unavailable: %s", version_name, ", ".join(unavailable_blocks) if unavailable_blocks else "нет")

    LOGGER.info("Надёжность сравнения версий:")
    if comparison_df.empty:
        LOGGER.info("  Сравнение недоступно: отсутствуют валидные результаты.")
    else:
        for row in comparison_df.sort_values("utility_rank").to_dict(orient="records"):
            LOGGER.info(
                "  %s | rank=%s | reliability=%s | effect=%.3f | overlap=%.3f | rel_sens=%.3f",
                row["version"],
                row["utility_rank"],
                row["reliability_level"],
                row.get("effect_size", float("nan")),
                row.get("distribution_overlap", float("nan")),
                row.get("relative_sensitivity", float("nan")),
            )

    limitations = summary.get("limitations", [])
    if limitations:
        LOGGER.info("Ограничения сравнения:")
        for limitation in limitations:
            LOGGER.info("  %s", limitation)

    reliable = summary.get("reliable_comparisons", [])
    if reliable:
        LOGGER.info("Где сравнение трёх версий наиболее надёжно: %s", ", ".join(sorted(set(reliable))))
    else:
        LOGGER.info("Где сравнение трёх версий наиболее надёжно: недостаточно данных для уверенного вывода.")

    resource_map = summary.get("resource_state_map", {})
    if resource_map:
        LOGGER.info("Карта IIS x RES:")
        for version_name, version_payload in resource_map.get("versions", {}).items():
            counts = version_payload.get("quadrant_counts", {})
            LOGGER.info(
                "  %s | regulated=%s | mobilized=%s | fragile=%s | depleted=%s",
                version_name,
                counts.get("regulated", 0),
                counts.get("mobilized", 0),
                counts.get("fragile", 0),
                counts.get("depleted", 0),
            )


def run_dataset(
    dataset: str,
    mode: str,
    dataset_path: Path,
    dynamic: bool = False,
    window_seconds: float | None = None,
    step_seconds: float | None = None,
    focus_version: str = "IISVersion6",
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """Полностью обрабатывает один датасет."""

    LOGGER.info("Запуск обработки %s из каталога %s", dataset.upper(), dataset_path)
    emit_progress(
        progress_callback,
        stage="setup",
        message=f"Подготовка сценария для {dataset.upper()}",
        stage_start=0.0,
        stage_end=0.02,
    )

    def loader_progress(event: dict[str, Any]) -> None:
        emit_progress(
            progress_callback,
            stage="loading_files",
            message=event.get("message", "Загрузка файлов датасета"),
            stage_start=0.02,
            stage_end=0.22,
            current=event.get("current"),
            total=event.get("total"),
            extra=event,
        )

    loader = build_loader(dataset=dataset, dataset_path=dataset_path, progress_callback=loader_progress)
    segments_df = loader.load()
    if segments_df.empty:
        raise ValueError(f"Загрузчик {dataset} не вернул ни одного сегмента.")
    emit_progress(
        progress_callback,
        stage="loading_complete",
        message=f"Сегменты загружены: {len(segments_df)}",
        stage_start=0.22,
        stage_end=0.22,
        current=1,
        total=1,
    )

    if dynamic:
        dynamic_defaults = DYNAMIC_WINDOW_DEFAULTS.get(dataset, {})
        actual_window_seconds = float(window_seconds if window_seconds is not None else dynamic_defaults.get("window_seconds", 5.0))
        actual_step_seconds = float(step_seconds if step_seconds is not None else dynamic_defaults.get("step_seconds", max(actual_window_seconds / 2.0, 1.0)))
        emit_progress(
            progress_callback,
            stage="windowing",
            message=f"Дробление на окна {actual_window_seconds:.1f} c / шаг {actual_step_seconds:.1f} c",
            stage_start=0.23,
            stage_end=0.28,
        )
        LOGGER.info(
            "Динамический режим: дробление сегментов на окна %.2f с, шаг %.2f с",
            actual_window_seconds,
            actual_step_seconds,
        )
        segments_df = expand_segments_dataframe(
            segments_df=segments_df,
            window_seconds=actual_window_seconds,
            step_seconds=actual_step_seconds,
        )
        LOGGER.info("После динамического дробления сегментов: %d", len(segments_df))
        emit_progress(
            progress_callback,
            stage="windowing_complete",
            message=f"Получено окон: {len(segments_df)}",
            stage_start=0.28,
            stage_end=0.28,
            current=1,
            total=1,
        )

    def feature_progress(event: dict[str, Any]) -> None:
        emit_progress(
            progress_callback,
            stage="features",
            message=event.get("message", "Извлечение признаков"),
            stage_start=0.30,
            stage_end=0.68,
            current=event.get("current"),
            total=event.get("total"),
            extra=event,
        )

    features_df = extract_features_dataframe(segments_df, progress_callback=feature_progress)
    if features_df.empty:
        raise ValueError(f"Не удалось извлечь признаки для датасета {dataset}.")
    emit_progress(
        progress_callback,
        stage="features_complete",
        message=f"Признаки извлечены: {len(features_df)} строк",
        stage_start=0.68,
        stage_end=0.68,
        current=1,
        total=1,
    )

    feature_path = OUTPUT_DIR / f"features_{dataset}.csv"
    features_df.to_csv(feature_path, index=False, encoding="utf-8-sig")

    resource_engine = IISResourceStateMap(plots_dir=PLOTS_DIR)
    models = [IISVersion1(), IISVersion2(), IISVersion3(), IISVersion4(), IISVersion5(), IISVersion6(), IISVersion7()]
    result_frames = []
    total_models = len(models)
    for model_index, model in enumerate(models, start=1):
        model_stage_start = 0.70 + (model_index - 1) * (0.18 / total_models)
        model_stage_end = 0.70 + model_index * (0.18 / total_models)
        emit_progress(
            progress_callback,
            stage="models",
            message=f"Расчёт {model.name} ({model_index}/{total_models})",
            stage_start=model_stage_start,
            stage_end=model_stage_end,
            current=model_index - 1,
            total=total_models,
            extra={"model_name": model.name},
        )
        def model_progress(event: dict[str, Any]) -> None:
            preview_payload: dict[str, Any] = {"model_name": model.name}
            preview_rows = event.get("preview_rows")
            if isinstance(preview_rows, list) and preview_rows:
                preview_frame = pd.DataFrame.from_records(preview_rows)
                if model.name in resource_engine.SUPPORTED_VERSIONS:
                    preview_frame = resource_engine.augment_preview_frame(preview_frame)
                preview_columns = [
                    column_name
                    for column_name in (
                        "source_record_id",
                        "window_start_sec",
                        "window_end_sec",
                        "label",
                        "version",
                        "IIS",
                        "RES",
                        "state_map_4",
                        "A",
                        "Gamma",
                        "V",
                        "Q",
                        "H",
                        "K",
                    )
                    if column_name in preview_frame.columns
                ]
                y_column = "RES" if "RES" in preview_columns and preview_frame["RES"].notna().any() else ("V" if "V" in preview_columns else "Q")
                z_column_candidates = (
                    "V",
                    "Q",
                    "Gamma",
                    "A",
                    "RES",
                    "IIS",
                )
                z_column = next(
                    (
                        column_name
                        for column_name in z_column_candidates
                        if column_name in preview_columns and column_name not in ("IIS", y_column)
                    ),
                    "V",
                )
                preview_payload["live_preview"] = {
                    "version": model.name,
                    "rows": preview_frame[preview_columns].tail(180).to_dict(orient="records"),
                    "count": int(event.get("current", len(preview_frame)) or len(preview_frame)),
                    "total": int(event.get("total", len(features_df)) or len(features_df)),
                    "x_column": "IIS",
                    "y_column": y_column,
                    "z_column": z_column,
                }
            emit_progress(
                progress_callback,
                stage="models",
                message=event.get("message", f"{model.name}: расчёт"),
                stage_start=model_stage_start,
                stage_end=model_stage_end,
                current=event.get("current"),
                total=event.get("total"),
                extra=preview_payload,
            )

        result_frames.append(
            model.evaluate_dataframe(
                features_df,
                mode=mode,
                progress_callback=model_progress if model.name == focus_version else None,
            )
        )
        emit_progress(
            progress_callback,
            stage="models",
            message=f"Готово: {model.name} ({model_index}/{total_models})",
            stage_start=model_stage_start,
            stage_end=model_stage_end,
            current=model_index,
            total=total_models,
            extra={"model_name": model.name},
        )
    results_df = pd.concat(result_frames, ignore_index=True)
    results_df, resource_summary = resource_engine.augment_results(results_df, dataset=dataset, mode=mode)

    results_path = OUTPUT_DIR / f"results_{dataset}_{mode}.csv"
    results_df.to_csv(results_path, index=False, encoding="utf-8-sig")

    comparison_engine = IISComparison(plots_dir=PLOTS_DIR)
    emit_progress(
        progress_callback,
        stage="comparison",
        message="Сравнение версий и построение графиков",
        stage_start=0.90,
        stage_end=0.96,
    )
    comparison_df, summary = comparison_engine.compare(
        features_df=features_df,
        results_df=results_df,
        dataset=dataset,
        mode=mode,
    )

    comparison_path = OUTPUT_DIR / f"comparison_{dataset}_{mode}.csv"
    comparison_df.to_csv(comparison_path, index=False, encoding="utf-8-sig")

    if dynamic:
        dynamic_engine = CausalDynamicAnalyzer(output_dir=OUTPUT_DIR, plots_dir=PLOTS_DIR)
        emit_progress(
            progress_callback,
            stage="dynamic",
            message=f"Причинная динамика {focus_version}",
            stage_start=0.96,
            stage_end=0.99,
        )
        dynamic_summary = dynamic_engine.analyze(
            results_df=results_df,
            dataset=dataset,
            mode=mode,
            focus_version=focus_version,
        )
        if dynamic_summary:
            summary["dynamic_outputs"] = dynamic_summary

    if resource_summary:
        summary["resource_state_map"] = resource_summary

    summary_path = OUTPUT_DIR / f"summary_{dataset}_{mode}.json"
    summary["generated_files"] = {
        "features_csv": str(feature_path),
        "results_csv": str(results_path),
        "comparison_csv": str(comparison_path),
        "plots_dir": str(PLOTS_DIR),
        "summary_json": str(summary_path),
    }
    current_log_path = get_current_log_path()
    if current_log_path is not None:
        summary["generated_files"]["log_file"] = str(current_log_path)
        summary["generated_files"]["latest_log_file"] = str(LOG_DIR / "analysis.log")
    if resource_summary.get("plots"):
        summary["generated_files"]["resource_state_map_plots"] = resource_summary["plots"]
    save_json(summary_path, summary)
    emit_progress(
        progress_callback,
        stage="done",
        message="Анализ завершён",
        stage_start=1.0,
        stage_end=1.0,
        current=1,
        total=1,
    )

    print_console_report(
        dataset=dataset,
        mode=mode,
        features_df=features_df,
        results_df=results_df,
        comparison_df=comparison_df,
        summary=summary,
    )

    return summary


def main() -> int:
    """Точка входа CLI."""

    args = parse_args()
    run_label = f"{args.dataset}_{args.mode}_{'dynamic' if args.dynamic else 'static'}"
    configure_logging(args.log_level, run_label=run_label)

    ensure_runtime_directories()
    datasets = list(SUPPORTED_DATASETS) if args.dataset == "all" else [args.dataset]
    global_mode = args.dataset == "all"

    failures: list[str] = []
    for dataset in datasets:
        try:
            dataset_path = resolve_dataset_path(dataset=dataset, cli_data_root=args.data_root, global_mode=global_mode)
            run_dataset(
                dataset=dataset,
                mode=args.mode,
                dataset_path=dataset_path,
                dynamic=args.dynamic,
                window_seconds=args.window_seconds,
                step_seconds=args.step_seconds,
                focus_version=args.focus_version,
            )
        except Exception as error:
            LOGGER.exception("Ошибка при обработке %s: %s", dataset, error)
            failures.append(dataset)

    if failures and len(failures) == len(datasets):
        LOGGER.error("Все датасеты завершились ошибкой: %s", ", ".join(failures))
        return 1

    if failures:
        LOGGER.warning("Часть датасетов завершилась ошибкой: %s", ", ".join(failures))
    else:
        LOGGER.info("Все выбранные датасеты обработаны успешно.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
