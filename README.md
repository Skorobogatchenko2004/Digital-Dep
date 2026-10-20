# Кредитный скоринг

Учебный проект по бинарной классификации риска дефолта заемщика.

## Датасет

Используется [Credit Risk Dataset на Kaggle](https://www.kaggle.com/datasets/laotse/credit-risk-dataset): 32 581 заявка, целевая переменная `loan_status` (1 — дефолт, 0 — погашение). CSV находится в `data/credit_risk_dataset.csv`. Признаки включают доход, сумму и цель займа, кредитный рейтинг, занятость и кредитную историю.

## Запуск

```bash
python -m pip install -r requirements.txt
python credit_scoring.py
```

Скрипт делит данные на обучающую и тестовую выборки с сохранением доли классов, обрабатывает пропуски и категориальные признаки внутри pipeline и сравнивает Dummy baseline, логистическую регрессию, дерево решений, случайный лес и HistGradientBoosting. Он выводит Accuracy, Precision, Recall, F1, ROC-AUC, PR-AUC и матрицу ошибок лучшей модели. Таблица метрик сохраняется в `model_metrics.csv`.

Можно передать собственный CSV и путь для результатов: `python credit_scoring.py --data path/to/data.csv --output results/metrics.csv`.

## Запуск в Google Colab

Создайте новый notebook в Google Colab и выполните ячейки по порядку. Репозиторий должен быть доступен для клонирования.

1. Склонируйте проект и перейдите в его папку:

```python
!git clone https://github.com/Skorobogatchenko2004/Digital-Dep.git
%cd Digital-Dep
```

2. Установите зависимости:

```python
!pip install -r requirements.txt
```

3. Обучите модели и выведите метрики:

```python
!python credit_scoring.py
```

Датасет уже находится в репозитории, поэтому загружать его отдельно не нужно. Результаты сохранятся в `model_metrics.csv`.

4. При необходимости скачайте таблицу метрик на свой компьютер:

```python
from google.colab import files
files.download("model_metrics.csv")
```

Результаты предназначены для учебного эксперимента и не должны использоваться для реальных решений о выдаче кредита без отдельной проверки качества, справедливости и соответствия требованиям регулирования.