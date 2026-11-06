"""Задачи первой части лабораторной: Pandas и Titanic."""
from __future__ import annotations

import pandas as pd

from grader_contracts.pandas_tasks import TitanicInput, TitanicSummary


def analyze_titanic(data: TitanicInput) -> TitanicSummary:
    """Выполните загрузку и анализ датасета Titanic."""

    # 1. Загружаем датасет
    df = pd.read_csv(data.csv_path)

    # 2. Общее количество строк
    row_count = len(df)

    # 3. Пропуски по колонкам
    missing_by_column = df.isnull().sum().to_dict()

    # 4. Пассажиры старше 30 лет
    adults_over_30_count = int((df['Age'] > 30).sum())

    # 5. Средний возраст по классам
    mean_age_dict = df.groupby('Pclass')['Age'].mean().to_dict()
    mean_age_by_pclass = {int(k): float(v) for k, v in mean_age_dict.items()}

    # 6. Доля выживших по классам
    surv_dict = df.groupby('Pclass')['Survived'].mean().to_dict()
    survival_rate_by_pclass = {int(k): float(v) for k, v in surv_dict.items()}

    # 7. Пять наибольших значений Fare
    highest_fares = df['Fare'].nlargest(5).tolist()

    return TitanicSummary(
        row_count=row_count,
        missing_by_column=missing_by_column,
        adults_over_30_count=adults_over_30_count,
        mean_age_by_pclass=mean_age_by_pclass,
        survival_rate_by_pclass=survival_rate_by_pclass,
        highest_fares=highest_fares
    )