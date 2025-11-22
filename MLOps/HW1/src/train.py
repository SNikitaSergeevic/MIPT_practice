import pandas as pd
import yaml
import mlflow
import mlflow.sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import json
from pathlib import Path

def main():
    # Загрузка параметров
    with open('params.yaml', 'r') as f:
        params = yaml.safe_load(f)
    
    model_type = params['train']['model_type']
    random_state = params['train']['random_state']
    
    # Загрузка данных
    train_data = pd.read_csv('data/processed/train.csv')
    test_data = pd.read_csv('data/processed/test.csv')
    
    # Загрузка имени целевой переменной
    try:
        with open('data/processed/target_column.txt', 'r') as f:
            target_column = f.read().strip()
    except FileNotFoundError:
        # Если файла нет, используем последний столбец
        target_column = train_data.columns[-1]
        print(f"Warning: target_column.txt not found, using '{target_column}' as target")
    
    print(f"Using target column: '{target_column}'")
    print(f"Train data shape: {train_data.shape}")
    print(f"Test data shape: {test_data.shape}")
    
    # Проверяем, что целевая переменная существует
    if target_column not in train_data.columns:
        available_cols = train_data.columns.tolist()
        raise KeyError(f"Target column '{target_column}' not found. Available columns: {available_cols}")
    
    # Разделение на признаки и целевую переменную
    X_train = train_data.drop(target_column, axis=1)
    y_train = train_data[target_column]
    X_test = test_data.drop(target_column, axis=1)
    y_test = test_data[target_column]
    
    # Настройка MLflow
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("mlops_hw1")
    
    with mlflow.start_run():
        # Выбор и обучение модели
        if model_type == "LogisticRegression":
            model = LogisticRegression(random_state=random_state, max_iter=1000)
        elif model_type == "RandomForest":
            model = RandomForestClassifier(random_state=random_state)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        model.fit(X_train, y_train)
        
        # Предсказания и метрики
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        # Логирование в MLflow
        mlflow.log_param("model_type", model_type)
        mlflow.log_param("target_column", target_column)
        mlflow.log_param("random_state", random_state)
        mlflow.log_metric("accuracy", accuracy)
        mlflow.sklearn.log_model(model, "model")
        
        # Сохранение дополнительных артефактов
        report = classification_report(y_test, y_pred, output_dict=True)
        with open('classification_report.json', 'w') as f:
            json.dump(report, f)
        mlflow.log_artifact('classification_report.json')
        
        # Сохранение модели для DVC
        import joblib
        joblib.dump(model, 'model.pkl')
        mlflow.log_artifact('model.pkl')
        
        print(f"Training completed! Accuracy: {accuracy:.4f}")
        print(f"Target column used: {target_column}")

if __name__ == "__main__":
    main()