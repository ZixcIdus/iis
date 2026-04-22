"""CLI для анализа ёмкости модели по числу различимых состояний."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from config.settings import OUTPUT_DIR, PROCESSING_MODES, SUPPORTED_DATASETS
from models.state_capacity import StateCapacityAnalyzer
from run_analysis import configure_logging

LOGGER = logging.getLogger("iis_state_capacity_cli")


def parse_args() -> argparse.Namespace:
    """Разбирает аргументы командной строки."""

    parser = argparse.ArgumentParser(description="Анализ числа различимых состояний модели ИИС.")
    parser.add_argument("--dataset", required=True, choices=[*SUPPORTED_DATASETS, "all"], help="Название датасета.")
    parser.add_argument(
        "--mode",
        default="auto",
        choices=[*PROCESSING_MODES, "all", "auto"],
        help="Режим обработки. auto = взять все найденные режимы для датасета.",
    )
    parser.add_argument("--versions", nargs="*", default=None, help="Список версий ИИС. По умолчанию используются все найденные.")
    parser.add_argument("--bootstrap", type=int, default=24, help="Число бутстрэп-итераций для устойчивости кластеров.")
    parser.add_argument("--k-min", type=int, default=2, help="Минимальное k для кластеров.")
    parser.add_argument("--k-max", type=int, default=5, help="Максимальное k для кластеров.")
    parser.add_argument("--min-class-samples", type=int, default=20, help="Минимальный размер класса для попарной статистики.")
    parser.add_argument("--include-dynamic", action="store_true", help="Включить анализ динамических CSV, если они существуют.")
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        help="Уровень логирования.",
    )
    return parser.parse_args()


def discover_dataset_modes(dataset: str, mode: str) -> list[tuple[str, str]]:
    """Определяет, какие результаты реально есть в outputs/."""

    available_pairs: set[tuple[str, str]] = set()
    for results_path in OUTPUT_DIR.glob("results_*_*.csv"):
        stem = results_path.stem
        if not stem.startswith("results_"):
            continue
        suffix = stem[len("results_") :]
        for candidate_mode in PROCESSING_MODES:
            tail = f"_{candidate_mode}"
            if suffix.endswith(tail):
                candidate_dataset = suffix[: -len(tail)]
                available_pairs.add((candidate_dataset, candidate_mode))
                break

    if dataset == "all":
        if mode == "all" or mode == "auto":
            return sorted(available_pairs)
        return sorted(pair for pair in available_pairs if pair[1] == mode)

    if mode == "auto":
        return sorted(pair for pair in available_pairs if pair[0] == dataset)
    if mode == "all":
        return sorted(pair for pair in available_pairs if pair[0] == dataset)
    return [(dataset, mode)]


def main() -> int:
    """Точка входа CLI."""

    args = parse_args()
    run_label = f"state_capacity_{args.dataset}_{args.mode}"
    configure_logging(args.log_level, run_label=run_label)

    pairs = discover_dataset_modes(dataset=args.dataset, mode=args.mode)
    if not pairs:
        LOGGER.error("Не найдено ни одного results_<dataset>_<mode>.csv для запрошенного набора.")
        return 1

    analyzer = StateCapacityAnalyzer(
        bootstrap_iterations=args.bootstrap,
        k_min=args.k_min,
        k_max=args.k_max,
        min_class_samples=args.min_class_samples,
    )

    failures: list[str] = []
    for dataset_name, mode_name in pairs:
        try:
            analyzer.analyze_dataset_mode(
                dataset=dataset_name,
                mode=mode_name,
                versions=args.versions,
                include_dynamic=args.include_dynamic,
            )
        except Exception as error:
            LOGGER.exception("Ошибка анализа state capacity для %s/%s: %s", dataset_name, mode_name, error)
            failures.append(f"{dataset_name}/{mode_name}")

    if failures and len(failures) == len(pairs):
        LOGGER.error("Анализ завершился ошибкой для всех наборов: %s", ", ".join(failures))
        return 1
    if failures:
        LOGGER.warning("Часть наборов завершилась ошибкой: %s", ", ".join(failures))
    else:
        LOGGER.info("Анализ state capacity завершён успешно для всех выбранных наборов.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
