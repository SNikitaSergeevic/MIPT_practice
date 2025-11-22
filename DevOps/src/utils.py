import pandas as pd
import numpy as np
import yaml
import json
import os

def load_config(config_path="configs/config.yaml"):
    """Загрузка конфигурации из YAML файла"""
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def save_metrics(metrics, filepath):
    """Сохранение метрик в JSON файл"""
    with open(filepath, 'w') as f:
        json.dump(metrics, f, indent=2)

def setup_directories():
    """Создание необходимых директорий"""
    directories = ['data', 'models', 'reports', 'mlruns']
    for dir in directories:
        os.makedirs(dir, exist_ok=True)
    print("✅ Директории созданы")

def log_message(message, level="INFO"):
    """Простое логирование"""
    print(f"[{level}] {message}")