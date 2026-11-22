# Financial News Sentiment Classification

## О проекте

Учебный ML mini-product для классификации
тональности англоязычных финансовых новостей.

Решается задача многоклассовой классификации:

- negative (0)
- neutral (1)
- positive (2)

Проект охватывает полный цикл ML:
EDA, preprocessing, training, evaluation,
error analysis, model saving и inference.

## Датасет

Используется Financial PhraseBank
(Financial Sentiment Analysis).

Источник:
https://www.kaggle.com/datasets/ankurzing/sentiment-analysis-for-financial-news

Файл: `all-data.csv`

- Исходных записей: 4 846
- После очистки: 4836
- Классов: 3
- Язык: английский

Исходный CSV не имеет заголовков и читается
с параметрами encoding='latin-1', header=None.

## Технологии

- Python
- pandas, NumPy
- scikit-learn
- TF-IDF
- Logistic Regression
- matplotlib, seaborn
- joblib
- Jupyter Notebook

## Установка

Создайте виртуальное окружение:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

Установите зависимости:

```powershell
python -m pip install -r requirements.txt
```

## Обучение модели

Поместите `all-data.csv` в корень проекта.

Запустите:

```powershell
jupyter lab
```

Откройте Notebook:

`Sentiment Analysis for Financial News.ipynb`

В меню выберите Kernel -> Restart & Run All.

Notebook выполняет:
1. EDA
2. Preprocessing
3. Baseline training
4. Cross-validation и выбор модели
5. Error analysis
6. Сохранение модели
7. Финальные проверки

Для воспроизводимости используется random_state=42.

## Baseline

TF-IDF:
- max_features = 5000
- ngram_range = (1, 2)

Classifier: LogisticRegression

Baseline Macro F1 (test): 0.6888

## Улучшения

Для сравнения использовалась
Stratified 5-Fold Cross-Validation
на обучающей выборке.

| Модель | Macro F1 (CV) | Изменение |
|---|---:|---:|
| LogReg Balanced | 0.7200 | +9.82% |
| LinearSVC | 0.7124 | +8.66% |
| Baseline LogReg | 0.6556 | +0.00% |
| TF-IDF Trigrams | 0.6431 | -1.92% |

Лучшая модель: **LogReg Balanced**

Best Macro F1 (CV): 0.7200

## Результаты на Test Set

Baseline Macro F1: 0.6888

Best Model Macro F1: 0.7242

Best Model Accuracy: 0.7665

Test size: 968

CV и test-метрики не являются
взаимозаменяемыми.

## Error Analysis

Всего тестовых примеров: 968

Ошибочных предсказаний: 226

Основной тип ошибки:
neutral -> positive: 71 ошибок

Выполнены:
- Confusion Matrix
- Анализ ошибок по классам
- Разбор реальных финансовых новостей

Результаты находятся в папке `reports/`.

## Inference без переобучения

Обученный Pipeline:

`models/day06_inference_pipeline.joblib`

Запуск из терминала:

```powershell
python predict_cli.py --text "Company profits increased by 20%."
```

Несколько текстов:

```powershell
python predict_cli.py --text "Profits rose." --text "Sales declined."
```

Input: одна или несколько английских новостей.

Output: JSON с полями:
- text
- label
- sentiment
- confidence (если доступен)

Для inference не требуется исходный CSV.

Загружайте joblib только из доверенных источников.

## Demo

Примеры находятся в Notebook и файле:

`reports/day07_demo.csv`

## Ограничения модели

1. Модель обучена на английских текстах.
2. Русскоязычные новости надёжно не поддерживаются.
3. Возможны ошибки на неоднозначных формулировках.
4. Модель ограничена тематикой и периодом
   исходного датасета.
5. TF-IDF не обладает полноценным пониманием
   экономического контекста.
6. Confidence не гарантирует правильность ответа.
7. Модель не предназначена для автоматического
   принятия инвестиционных решений.
8. Для загрузки joblib желательно использовать
   совместимые версии Python и scikit-learn.

Права на датасет и условия его распространения
необходимо проверять у источника.

## Итог

Реализован end-to-end classification workflow:
от загрузки данных до inference новых текстов.
