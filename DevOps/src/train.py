import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report
import json
import os

def prepare_data():
    """Подготовка данных для обучения"""
    print("📊 Подготовка данных...")
    data = load_iris()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = data.target
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    return X_train, X_test, y_train, y_test, data.feature_names

def train_model(X_train, y_train, params):
    """Обучение модели с заданными параметрами"""
    print("🤖 Обучение модели...")
    model = RandomForestClassifier(**params)
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    """Оценка модели"""
    print("📈 Оценка модели...")
    y_pred = model.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    
    metrics = {
        'accuracy': accuracy,
        'f1_score': f1
    }
    
    # Детальный отчет
    report = classification_report(y_test, y_pred, output_dict=True)
    
    return metrics, report, y_pred

def run_experiment():
    """Запуск полного эксперимента с логированием в MLflow"""
    
    # Параметры эксперимента
    params = {
        'n_estimators': 100,
        'max_depth': 5,
        'random_state': 42,
        'min_samples_split': 2
    }
    
    # Настройка MLflow
    mlflow.set_tracking_uri("./mlruns")
    mlflow.set_experiment("Iris_Classification")
    
    with mlflow.start_run():
        print("🚀 Запуск эксперимента MLflow...")
        
        # Подготовка данных
        X_train, X_test, y_train, y_test, feature_names = prepare_data()
        
        # Логирование параметров
        mlflow.log_params(params)
        mlflow.log_param("dataset", "iris")
        mlflow.log_param("n_features", len(feature_names))
        
        # Обучение модели
        model = train_model(X_train, y_train, params)
        
        # Оценка модели
        metrics, report, y_pred = evaluate_model(model, X_test, y_test)
        
        # Логирование метрик
        mlflow.log_metrics(metrics)
        
        # Логирование модели
        mlflow.sklearn.log_model(model, "model")
        
        # Логирование дополнительных артефактов
        # Сохраняем отчет о классификации
        with open("classification_report.json", "w") as f:
            json.dump(report, f, indent=2)
        mlflow.log_artifact("classification_report.json")
        
        # Сохраняем важность признаков
        feature_importance = pd.DataFrame({
            'feature': feature_names,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        feature_importance.to_csv("feature_importance.csv", index=False)
        mlflow.log_artifact("feature_importance.csv")
        
        # Логирование тестовых предсказаний
        test_results = pd.DataFrame({
            'actual': y_test,
            'predicted': y_pred
        })
        test_results.to_csv("test_predictions.csv", index=False)
        mlflow.log_artifact("test_predictions.csv")
        
        print("✅ Эксперимент успешно завершен!")
        print(f"📊 Метрики модели: {metrics}")
        
        return model, metrics

if __name__ == "__main__":
    # Создаем папки для артефактов
    os.makedirs("reports", exist_ok=True)
    os.makedirs("models", exist_ok=True)
    
    # Запускаем эксперимент
    model, metrics = run_experiment()
    
    # Сохраняем модель локально
    import joblib
    joblib.dump(model, "models/iris_model.joblib")
    print("💾 Модель сохранена в models/iris_model.joblib")