import pandas as pd
import numpy as np
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset, TargetDriftPreset
from evidently.test_suite import TestSuite
from evidently.tests import TestValueDrift, TestShareOfDriftedFeatures
import warnings
warnings.filterwarnings('ignore')

def load_data_for_drift():
    """Загрузка и подготовка данных для анализа дрейфа"""
    from sklearn.datasets import load_iris
    data = load_iris()
    df = pd.DataFrame(data.data, columns=[col.replace(' (cm)', '').replace(' ', '_') for col in data.feature_names])
    df['target'] = data.target
    
    # Создаем искусственный дрейф для демонстрации
    reference = df.sample(frac=0.5, random_state=42)
    current = df.drop(reference.index)
    
    # Добавляем немного шума в current данные для симуляции дрейфа
    np.random.seed(42)
    for col in current.columns[:-1]:  # все кроме target
        current[col] = current[col] * (1 + np.random.normal(0, 0.1, len(current)))
    
    return reference, current

def run_drift_analysis():
    """Запуск анализа дрейфа данных"""
    print("📊 Запуск анализа дрейфа с EvidentlyAI...")
    
    # Загрузка данных
    reference, current = load_data_for_drift()
    
    # Создание отчета о дрейфе данных
    data_drift_report = Report(metrics=[DataDriftPreset()])
    data_drift_report.run(
        reference_data=reference,
        current_data=current
    )
    
    # Создание отчета о дрейфе целевой переменной
    target_drift_report = Report(metrics=[TargetDriftPreset()])
    target_drift_report.run(
        reference_data=reference,
        current_data=current
    )
    
    # Сохранение отчетов
    data_drift_report.save_html('reports/data_drift_report.html')
    target_drift_report.save_html('reports/target_drift_report.html')
    
    print("✅ Отчеты о дрейфе сохранены в папке reports/")
    
    # Вывод основных результатов
    print("\n📈 Основные метрики дрейфа:")
    result = data_drift_report.as_dict()
    n_drifted_features = result['metrics'][0]['result']['number_of_drifted_features']
    share_drifted_features = result['metrics'][0]['result']['share_of_drifted_features']
    
    print(f"Количество признаков с дрейфом: {n_drifted_features}")
    print(f"Доля признаков с дрейфом: {share_drifted_features:.2%}")
    
    return data_drift_report, target_drift_report

if __name__ == "__main__":
    run_drift_analysis()