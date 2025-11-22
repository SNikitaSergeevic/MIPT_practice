import pandas as pd
import numpy as np
from deepchecks.tabular import Dataset
from deepchecks.tabular.suites import data_integrity, train_test_validation
import warnings
warnings.filterwarnings('ignore')

def load_sample_data():
    """Загрузка примерных данных для демонстрации"""
    from sklearn.datasets import load_iris
    data = load_iris()
    df = pd.DataFrame(data.data, columns=data.feature_names)
    df['target'] = data.target
    return df

def create_deepchecks_dataset(df, target_col='target'):
    """Создание Dataset объекта для Deepchecks"""
    categorical_features = df.select_dtypes(include=['object']).columns.tolist()
    dataset = Dataset(
        df, 
        label=target_col,
        cat_features=categorical_features
    )
    return dataset

def run_data_validation():
    """Запуск проверки качества данных"""
    print("🔍 Запуск проверки данных с Deepchecks...")
    
    # Загрузка данных
    df = load_sample_data()
    
    # Разделение на train/test для демонстрации
    train_df = df.sample(frac=0.7, random_state=42)
    test_df = df.drop(train_df.index)
    
    # Создание datasets
    train_dataset = create_deepchecks_dataset(train_df)
    test_dataset = create_deepchecks_dataset(test_df)
    
    # Запуск проверок
    print("Выполнение проверки целостности данных...")
    integrity_suite = data_integrity()
    integrity_result = integrity_suite.run(train_dataset)
    
    print("Выполнение train-test validation...")
    validation_suite = train_test_validation()
    validation_result = validation_suite.run(train_dataset, test_dataset)
    
    # Сохранение отчетов
    integrity_result.save_as_html('reports/deepchecks_integrity_report.html')
    validation_result.save_as_html('reports/deepchecks_validation_report.html')
    
    print("✅ Отчеты сохранены в папке reports/")
    
    return integrity_result, validation_result

if __name__ == "__main__":
    run_data_validation()