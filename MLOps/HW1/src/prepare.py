import pandas as pd
import yaml
from sklearn.model_selection import train_test_split
from pathlib import Path

def main():
    # Загрузка параметров
    with open('params.yaml', 'r') as f:
        params = yaml.safe_load(f)
    
    split_ratio = params['prepare']['split_ratio']
    random_state = params['prepare']['random_state']
    
    # Загрузка данных
    data = pd.read_csv('data/raw/data.csv')
    
    # Автоматическое определение целевой переменной
    # Для Iris: последний столбец обычно target
    # Для других датасетов может потребоваться настройка
    target_column = data.columns[-1]  # берем последний столбец как target
    
    print(f"Using '{target_column}' as target variable")
    print(f"Data shape: {data.shape}")
    print(f"Columns: {data.columns.tolist()}")
    
    # Базовая предобработка
    data_cleaned = data.dropna()
    
    # Разделение на train/test
    train_data, test_data = train_test_split(
        data_cleaned, 
        test_size=split_ratio, 
        random_state=random_state,
        stratify=data_cleaned[target_column]
    )
    
    # Сохранение обработанных данных
    Path('data/processed').mkdir(exist_ok=True)
    train_data.to_csv('data/processed/train.csv', index=False)
    test_data.to_csv('data/processed/test.csv', index=False)
    
    # Сохраняем имя целевой переменной для использования в train.py
    with open('data/processed/target_column.txt', 'w') as f:
        f.write(target_column)
    
    print(f"Data preparation completed successfully! Target: {target_column}")

if __name__ == "__main__":
    main()