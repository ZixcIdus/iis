"""Полный оркестратор IIS Full Test + Documentation Pack."""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

from build_iis_full_report import main as build_full_report_main
from build_iis_report_bundle import main as build_bundle_main
from reporting.iis_report_common import DYNAMIC_SCENARIOS, SCENARIOS, VERSION_ORDER
from run_analysis import configure_logging, resolve_dataset_path, run_dataset
from run_test_matrix import main as run_test_matrix_main
from models.state_capacity import StateCapacityAnalyzer

LOGGER = logging.getLogger("iis_full_pack")
MANIFEST_PATH = Path("outputs") / "iis_full_pack_manifest.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Полный прогон IIS Full Test + Documentation Pack.")
    parser.add_argument("--log-level", default="INFO", choices=("DEBUG", "INFO", "WARNING", "ERROR"))
    parser.add_argument("--bootstrap", type=int, default=16, help="Число bootstrap-итераций для state capacity.")
    parser.add_argument("--calibration-trials", type=int, default=240, help="Число trial-итераций для DREAMER calibration.")
    parser.add_argument("--skip-static", action="store_true", help="Не перезапускать статические сценарии.")
    parser.add_argument("--skip-dynamic", action="store_true", help="Не перезапускать динамические сценарии.")
    parser.add_argument("--skip-state-capacity", action="store_true", help="Не пересчитывать state capacity.")
    parser.add_argument("--skip-experiments", action="store_true", help="Не запускать calibration/synthetic блоки.")
    parser.add_argument("--skip-report", action="store_true", help="Не собирать bundle/PDF/HTML.")
    return parser.parse_args()


def _python_command(args: list[str]) -> list[str]:
    return [sys.executable, *args]


def _run_subprocess(args: list[str], *, label: str) -> dict[str, Any]:
    command = _python_command(args)
    LOGGER.info("Запуск подпроцесса %s: %s", label, " ".join(command))
    completed = subprocess.run(command, check=True, capture_output=True)
    stdout = completed.stdout.decode("utf-8", errors="replace").strip() if completed.stdout else ""
    stderr = completed.stderr.decode("utf-8", errors="replace").strip() if completed.stderr else ""
    return {
        "label": label,
        "command": command,
        "returncode": completed.returncode,
        "stdout": stdout,
        "stderr": stderr,
    }


def _run_static_scenarios(manifest: dict[str, Any]) -> None:
    manifest["static_runs"] = []
    for scenario in SCENARIOS:
        dataset = scenario["dataset"]
        mode = scenario["mode"]
        dataset_path = resolve_dataset_path(dataset=dataset, cli_data_root=None, global_mode=False)
        LOGGER.info("Статический прогон %s / %s", dataset, mode)
        result = run_dataset(
            dataset=dataset,
            mode=mode,
            dataset_path=dataset_path,
            dynamic=False,
            focus_version="IISVersion7",
        )
        manifest["static_runs"].append(
            {
                "dataset": dataset,
                "mode": mode,
                "result_files": result.get("generated_files", {}),
                "summary_path": result.get("summary_path"),
            }
        )


def _run_dynamic_scenarios(manifest: dict[str, Any]) -> None:
    manifest["dynamic_runs"] = []
    focus_versions = ["IISVersion4", "IISVersion5", "IISVersion6", "IISVersion7"]
    for scenario in DYNAMIC_SCENARIOS:
        dataset = scenario["dataset"]
        mode = scenario["mode"]
        dataset_path = resolve_dataset_path(dataset=dataset, cli_data_root=None, global_mode=False)
        for version in focus_versions:
            LOGGER.info("Динамический прогон %s / %s / %s", dataset, mode, version)
            result = run_dataset(
                dataset=dataset,
                mode=mode,
                dataset_path=dataset_path,
                dynamic=True,
                focus_version=version,
            )
            manifest["dynamic_runs"].append(
                {
                    "dataset": dataset,
                    "mode": mode,
                    "focus_version": version,
                    "dynamic_outputs": result.get("dynamic_outputs"),
                    "summary_path": result.get("summary_path"),
                }
            )


def _run_state_capacity(manifest: dict[str, Any], bootstrap: int) -> None:
    analyzer = StateCapacityAnalyzer(bootstrap_iterations=bootstrap)
    manifest["state_capacity_runs"] = []
    for scenario in SCENARIOS:
        dataset = scenario["dataset"]
        mode = scenario["mode"]
        LOGGER.info("State capacity %s / %s", dataset, mode)
        payload = analyzer.analyze_dataset_mode(
            dataset=dataset,
            mode=mode,
            versions=list(VERSION_ORDER),
            include_dynamic=True,
        )
        manifest["state_capacity_runs"].append(
            {
                "dataset": dataset,
                "mode": mode,
                "capacity_json": payload.get("generated_files", {}).get("state_capacity_json"),
            }
        )


def _run_experiments(manifest: dict[str, Any], calibration_trials: int) -> None:
    manifest["experiments"] = []
    experiment_calls = [
        (
            "dreamer_v5_calibration",
            [
                "run_dreamer_v5_calibration.py",
                "--input",
                "outputs/features_dreamer.csv",
                "--trials",
                str(calibration_trials),
                "--folds",
                "4",
                "--seed",
                "42",
                "--json-out",
                "outputs/dreamer_v5_calibration.json",
                "--csv-out",
                "outputs/dreamer_v5_calibration_candidates.csv",
            ],
        ),
        (
            "dreamer_v7_calibration_global",
            [
                "run_dreamer_v7_calibration.py",
                "--input",
                "outputs/features_dreamer.csv",
                "--trials",
                str(calibration_trials),
                "--folds",
                "4",
                "--seed",
                "42",
                "--candidate-mode",
                "global",
                "--json-out",
                "outputs/dreamer_v7_calibration_global.json",
                "--csv-out",
                "outputs/dreamer_v7_calibration_global_candidates.csv",
            ],
        ),
        (
            "sumigron_synthetic",
            ["run_sumigron_synthetic.py"],
        ),
    ]
    for label, command in experiment_calls:
        manifest["experiments"].append(_run_subprocess(command, label=label))


def _run_report_build(manifest: dict[str, Any]) -> None:
    manifest["report_build"] = []
    bundle_code = build_bundle_main()
    manifest["report_build"].append({"step": "bundle", "returncode": bundle_code})
    matrix_code = run_test_matrix_main()
    manifest["report_build"].append({"step": "test_matrix", "returncode": matrix_code})
    report_code = build_full_report_main()
    manifest["report_build"].append({"step": "report", "returncode": report_code})


def main() -> int:
    args = parse_args()
    configure_logging(args.log_level, run_label="full_test_pack")
    manifest: dict[str, Any] = {
        "started_at": datetime.now().isoformat(timespec="seconds"),
        "config": {
            "bootstrap": args.bootstrap,
            "calibration_trials": args.calibration_trials,
            "skip_static": args.skip_static,
            "skip_dynamic": args.skip_dynamic,
            "skip_state_capacity": args.skip_state_capacity,
            "skip_experiments": args.skip_experiments,
            "skip_report": args.skip_report,
        },
        "status": "running",
    }

    try:
        if not args.skip_static:
            _run_static_scenarios(manifest)
        if not args.skip_dynamic:
            _run_dynamic_scenarios(manifest)
        if not args.skip_state_capacity:
            _run_state_capacity(manifest, bootstrap=args.bootstrap)
        if not args.skip_experiments:
            _run_experiments(manifest, calibration_trials=args.calibration_trials)
        if not args.skip_report:
            _run_report_build(manifest)
        manifest["status"] = "completed"
        returncode = 0
    except Exception as error:  # noqa: BLE001
        LOGGER.exception("Полный пакет завершился с ошибкой: %s", error)
        manifest["status"] = "failed"
        manifest["error"] = repr(error)
        returncode = 1
    finally:
        manifest["finished_at"] = datetime.now().isoformat(timespec="seconds")
        MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
        MANIFEST_PATH.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        LOGGER.info("Manifest сохранён в %s", MANIFEST_PATH)
    return returncode


if __name__ == "__main__":
    raise SystemExit(main())
