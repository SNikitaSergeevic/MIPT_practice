import pandas as pd
import numpy as np
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
    
    try:
        # Пробуем импорт для evidently 0.4.x
        from evidently.report import Report
        from evidently.metric_preset import DataDriftPreset
    except ImportError:
        try:
            # Пробуем импорт для evidently 0.3.x
            from evidently.dashboard import Dashboard
            from evidently.tabs import DataDriftTab
            print("⚠️ Используется evidently 0.3.x")
        except ImportError as e:
            print(f"❌ Ошибка при создании отчета: {e}")
            return None, None
    
    # Загрузка данных
    reference, current = load_data_for_drift()
    
    try:
        # Для evidently 0.4.x
        data_drift_report = Report(metrics=[DataDriftPreset()])
        data_drift_report.run(
            reference_data=reference,
            current_data=current
        )
        data_drift_report.save_html('reports/data_drift_report.html')
        print("✅ Отчет о дрейфе сохранен (v0.4.x)")
        
    except (NameError, TypeError):
        try:
            # Для evidently 0.3.x
            data_drift_dashboard = Dashboard(tabs=[DataDriftTab()])
            data_drift_dashboard.calculate(
                reference_data=reference,
                current_data=current
            )
            data_drift_dashboard.save('reports/data_drift_report.html')
            print("✅ Отчет о дрейфе сохранен (v0.3.x)")
        except Exception as e:
            print(f"❌ Ошибка при создании отчета: {e}")
            return None, None
    
    print("📈 Анализ дрейфа завершен!")
    return reference, current

if __name__ == "__main__":
    run_drift_analysis()