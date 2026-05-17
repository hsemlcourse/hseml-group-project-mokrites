[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/kOqwghv0)
# ML Project — Предсказание типа астрономического объекта по данным наблюдений

**Студент:** Артюшин Алексей Владимирович  
**Группа:** БИВ 235

## Оглавление
1. [Описание задачи](#описание-задачи)
2. [Структура репозитория](#структура-репозитория)
3. [Быстрый старт](#быстрый-старт)
4. [Запуск пайплайна CP1 и CP2](#запуск-пайплайна-cp1-и-cp2)
5. [Проверки качества](#проверки-качества)
6. [Ноутбуки](#ноутбуки)
7. [Данные](#данные)
8. [Результаты](#результаты)
9. [Отчёт](#отчёт)

## Описание задачи
**Задача:** multiclass-классификация — по наблюдательным признакам SDSS предсказать тип объекта (`STAR`, `GALAXY`, `QSO`).

**Датасет:** Kaggle, Stellar Classification Dataset (SDSS17): [ссылка](https://www.kaggle.com/datasets/fedesoriano/stellar-classification-dataset-sdss17)

**Основная метрика:** `Macro-F1`  
**Дополнительные метрики:** `Accuracy`, `Weighted-F1`, per-class precision/recall, confusion matrix.

## Структура репозитория
```text
.
├── data
│   ├── raw
│   │   └── star_classification.csv        # Исходный датасет
│   └── processed
│       ├── train.csv                      # Train после preprocessing
│       ├── val.csv                        # Validation после preprocessing
│       ├── test.csv                       # Test после preprocessing
│       └── preprocessing_meta.json        # Метаданные preprocessing
├── models
│   ├── cp1_baseline.joblib                # Baseline CP1 (LogReg)
│   ├── cp1_best_model.joblib              # Лучшая модель CP1
│   ├── cp1_results.csv                    # Таблица экспериментов CP1
│   ├── cp1_metrics.json                   # Подробные метрики CP1
│   ├── cp2_best_model.joblib              # Лучшая модель CP2
│   ├── cp2_results.csv                    # Таблица экспериментов CP2 (part1 + part2)
│   └── cp2_metrics.json                   # Подробные метрики CP2 + tuning summary
├── notebooks
│   ├── 01_eda.ipynb                       # EDA и анализ данных
│   ├── 02_baseline.ipynb                  # Baseline + CP2 add-on
│   └── 03_experiments.ipynb               # Эксперименты CP1 + CP2
├── presentation                            # Материалы для защиты
├── report
│   └── report.md                           # Текст отчёта
├── src
│   ├── __init__.py
│   ├── preprocessing.py                    # Подготовка данных и split
│   ├── modeling.py                         # CP1: baseline + experiment
│   └── cp2_experiments.py                  # CP2: новые модели, ансамбли, tuning
├── tests
│   ├── test.py                             # Тесты preprocessing + CP1 smoke
│   └── test_cp2.py                         # CP2 smoke-тест
├── pytest.ini
├── requirements.txt
└── README.md
```

## Быстрый старт
```bash
# 1) Клонировать репозиторий
git clone https://github.com/hsemlcourse/hseml-group-project-mokrites.git
cd hseml-group-project-mokrites

# 2) Создать и активировать виртуальное окружение
python3 -m venv .venv
source .venv/bin/activate

# 3) Установить зависимости
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Запуск пайплайна CP1 и CP2
```bash
# 1) Preprocessing (создаёт train/val/test)
python3 -m src.preprocessing \
  --input data/raw/star_classification.csv \
  --output data/processed \
  --random-state 42

# 2) CP1 (baseline + experiment)
python3 -m src.modeling \
  --processed-dir data/processed \
  --models-dir models \
  --random-state 42

# 3) CP2 (part1: новые модели/ансамбли, part2: RandomizedSearchCV)
python3 -m src.cp2_experiments \
  --processed-dir data/processed \
  --models-dir models \
  --random-state 42 \
  --tune-max-samples 25000 \
  --tune-n-iter 8
```

## Проверки качества
```bash
ruff check src tests
flake8 src tests --max-line-length 120
pytest -q
```

## Ноутбуки
Запускать по порядку:
1. `notebooks/01_eda.ipynb`
2. `notebooks/02_baseline.ipynb`
3. `notebooks/03_experiments.ipynb`

## Данные
- `data/raw/star_classification.csv` — исходные данные (100000 строк, 18 колонок).
- `data/processed/train.csv`, `val.csv`, `test.csv` — подготовленные сплиты.

## Результаты
### CP1
| Модель | Macro-F1 (val) | Accuracy (val) | Macro-F1 (test) | Accuracy (test) |
|---|---:|---:|---:|---:|
| LogisticRegression (baseline, base features) | 0.9508 | 0.9571 | 0.9479 | 0.9555 |
| RandomForestClassifier (best CP1, engineered features) | **0.9758** | **0.9791** | **0.9747** | **0.9783** |

### CP2
| Эксперимент | Phase | Macro-F1 (val) | Accuracy (val) | Macro-F1 (test) | Accuracy (test) |
|---|---|---:|---:|---:|---:|
| HistGradientBoostingClassifier (`hist_gb_exp`) | part1 | **0.9746** | **0.9780** | **0.9739** | **0.9775** |
| ExtraTreesClassifier (`extra_trees_tuned`) | part2_tuning | 0.9743 | 0.9777 | 0.9725 | 0.9761 |
| HistGradientBoostingClassifier (`hist_gb_tuned`) | part2_tuning | 0.9738 | 0.9773 | 0.9738 | 0.9773 |

Итог: в CP2 проведены дополнительные модели, ансамбли и тюнинг; лучший результат CP2 достигнут моделью `hist_gb_exp`.

## Отчёт
Финальный отчёт: `report/report.md`
