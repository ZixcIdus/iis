# Карта репозитория

Ниже краткая карта того, как устроен проект на уровне каталогов и основных сценариев запуска.

## Каталоги

### `config/`

Базовые настройки проекта.

Ключевой файл:

- `settings.py`

### `features/`

Извлечение признаков из физиологических сигналов.

Ключевые модули:

- `eeg_features.py`
- `hrv_features.py`
- `common.py`

### `loaders/`

Загрузка и нормализация данных из разных открытых наборов.

Ключевые модули:

- `wesad_loader.py`
- `dreamer_loader.py`
- `case_loader.py`
- `openneuro_bcmi_loader.py`
- `deap_loader.py`
- `eva_med_loader.py`

### `models/`

Главное математическое ядро проекта.

Ключевые модули:

- `iis_v1.py` ... `iis_v7.py`
- `comparison.py`
- `dynamic_analysis.py`
- `intervention_analysis.py`
- `resource_state_map.py`
- `state_capacity.py`
- `base_model.py`

### `reporting/`

Общие функции для сборки отчетов и итоговых пакетов.

### `docs/`

Исходные научные документы проекта.

В том числе:

- математические документы по версиям;
- документы по гипотезам и валидации;
- отдельный документ по `Diamond State Model`;
- документ по `V7` и `Sumigron`.

### `data/`

Локальные датасеты. Каталог не включается в Git.

### `outputs/`

Локальные результаты анализа, таблицы, графики и логи. Каталог не включается в Git.

### `temp/`

Временные рабочие артефакты. Каталог не включается в Git.

## Основные сценарии запуска

### `run_analysis.py`

Главная CLI-точка входа для сравнения версий модели ИИС на выбранном датасете и в выбранном режиме.

Пример:

```bash
python run_analysis.py --dataset ds002724 --mode strict --focus-version IISVersion6
```

### `run_full_test_pack.py`

Полный прогон пакета тестов и документации.

Пример:

```bash
python run_full_test_pack.py
```

Поддерживает флаги для пропуска отдельных частей, например:

- `--skip-static`
- `--skip-dynamic`
- `--skip-state-capacity`
- `--skip-experiments`
- `--skip-report`

### `build_iis_full_report.py`

Сборка итогового расширенного отчета по уже подготовленному bundle.

### `build_iis_report_bundle.py`

Формирование сводного bundle, на котором строятся итоговые документы.

### `run_dreamer_v5_calibration.py` и `run_dreamer_v7_calibration.py`

Отдельные сценарии калибровки для `DREAMER`.

### `run_sumigron_synthetic.py`

Synthetic-проверка для экспериментальной ветки `V7`.

### `run_test_matrix.py`

Сборка сравнительной матрицы по версиям и датасетам.

### `run_state_capacity.py`

Отдельный сценарий анализа state capacity.

### `run_gui.py`

Локальный графический интерфейс для работы с проектом.

## Рекомендуемая логика чтения кода

Если вы заходите в репозиторий впервые, лучше идти в таком порядке:

1. `README.md`
2. `docs/github/model-line-v4-v6-v7.md`
3. `docs/github/formulas-v4-v6-v7.md`
4. `run_analysis.py`
5. `models/iis_v4.py`, `models/iis_v6.py`, `models/iis_v7.py`

Так структура проекта читается как исследовательская система, а не как набор несвязанных скриптов.
