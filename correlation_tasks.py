"""Задачи второй части лабораторной: корреляционный анализ."""
from __future__ import annotations

import pandas as pd

from grader_contracts.correlation_tasks import BrainCorrelationSummary, BrainDataInput


def analyze_brain_correlations(data: BrainDataInput) -> BrainCorrelationSummary:
    """Проанализируйте brainsize.txt."""
    
    # Загружаем таблицу
    df = pd.read_csv(data.csv_path, sep='\t', na_values='NA')

    # Разделяем по полу
    men_df = df[df['Gender'] == 'Male']
    women_df = df[df['Gender'] == 'Female']

    men_count = len(men_df)
    women_count = len(women_df)

    # Признаки для анализа
    features = ['FSIQ', 'VIQ', 'PIQ', 'Weight', 'Height']

    # Считаем корреляцию Пирсона
    men_corr_series = men_df[features + ['MRI_Count']].corr()['MRI_Count'].drop('MRI_Count')
    women_corr_series = women_df[features + ['MRI_Count']].corr()['MRI_Count'].drop('MRI_Count')

    men_mri_correlation = men_corr_series.to_dict()
    women_mri_correlation = women_corr_series.to_dict()

    # Ищем признак с максимальным модулем корреляции среди обеих групп
    all_correlations = {**men_mri_correlation, **women_mri_correlation}
    strongest_mri_feature = max(all_correlations, key=lambda k: abs(all_correlations[k]))

    return BrainCorrelationSummary(
        men_count=men_count,
        women_count=women_count,
        women_mri_correlation=women_mri_correlation,
        men_mri_correlation=men_mri_correlation,
        strongest_mri_feature=strongest_mri_feature
    )