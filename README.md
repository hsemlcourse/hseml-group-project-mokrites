[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/kOqwghv0)
# ML Project — Предсказание типа астрономического объекта по данным наблюдений

**Студент:** Артюшин Алексей Владимирович

**Группа:** БИВ 235


## Оглавление

1. [Описание задачи](#описание-задачи)
2. [Структура репозитория](#структура-репозитория)
3. [Запуски](#быстрый-старт)
4. [Данные](#данные)
5. [Результаты](#результаты)
7. [Отчёт](#отчёт)


## Описание задачи

**Задача:** Классификация – по наблюдательным признакам SDSS предсказать тип объекта (STAR, GALAXY, QSO)

**Датасет:** Kaggle, Stellar Classification Dataset (SDSS17): [ссылка](https://www.kaggle.com/datasets/fedesoriano/stellar-classification-dataset-sdss17)

**Целевая метрика:** Macro-F1, Accuracy, Weighted-F1, confusion matrix, per-class precision/recall


## Структура репозитория

```
.
├── data
│   ├── raw
│   │   └── star_classification.csv      # Исходный датасет Kaggle (SDSS17)
│   └── processed
│       ├── train.csv                    # Train после preprocessing
│       ├── val.csv                      # Validation после preprocessing
│       ├── test.csv                     # Test после preprocessing
│       └── preprocessing_meta.json      # Метаданные preprocessing (clip bounds, колонки, размеры)
├── models
│   ├── cp1_baseline.joblib              # Baseline-модель (Logistic Regression)
│   ├── cp1_best_model.joblib            # Лучшая модель CP1 (по Macro-F1 на validation)
│   ├── cp1_results.csv                  # Таблица сравнения моделей и метрик на validation
│   └── cp1_metrics.json                 # Подробные метрики/отчеты/confusion matrix для лучшей модели
├── notebooks
│   ├── 01_eda.ipynb                     # EDA: проверки данных, визуализации, анализ классов
│   ├── 02_baseline.ipynb                # Baseline-модель и метрики на val/test
│   └── 03_experiments.ipynb             # Сравнение baseline и experiment-модели
├── presentation                         # Материалы для защиты (будут добавлены позже)
├── report
│   └── report.md                        # Черновик/финальная версия отчета
├── src
│   ├── __init__.py
│   ├── preprocessing.py                 # Пайплайн обработки данных и формирования split
│   └── modeling.py                      # CP1 моделирование: baseline + experiment + сохранение артефактов
├── tests
│   └── test.py                          # Тесты preprocessing и smoke-тест modeling
├── requirements.txt                     # Зависимости проекта
└── README.md

```

## Запуск


```bash
# 1. Клонировать репозиторий
git clone https://github.com/hsemlcourse/hseml-group-project-mokrites.git
cd hseml-group-project-mokrites

# 2. Создать виртуальное окружение
python -m venv .venv
source .venv/bin/activate   # Linux/macOS
# .venv\Scripts\activate    # Windows

# 3. Установить зависимости
pip install -r requirements.txt

# 4. Открыть и запустить блокноты по порядку:

notebooks/01_eda.ipynb
notebooks/02_baseline.ipynb
notebooks/03_experiments.ipynb
```

## Данные
- `data/raw/star_classification.csv` — исходный файлы
- `data/processed/train.csv`, `data/processed/val.csv` и `data/processed/test.csv`— предобработанные данные

## Результаты
| Модель | Macro-F1 (val) | Accuracy (val) | Примечание |
|--------|-----------------|----------------|------------|
| Baseline (Logistic Regression, base features) | 0.9508 | 0.9571 | Базовая модель без feature engineering в baseline-режиме |
| Лучшая модель (RandomForestClassifier, with engineered features) | 0.9758 | 0.9791 | Лучшая по validation; на test: Macro-F1 = 0.9747, Accuracy = 0.9783 |

## Отчёт

Финальный отчёт: [`report/report.md`](report/report.md)
