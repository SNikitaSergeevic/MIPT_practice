import sys
import os

# Добавляем src в путь для импортов
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data_validation import run_data_validation
from drift_analysis import run_drift_analysis
from train import run_experiment
from utils import setup_directories, log_message

def run_full_pipeline():
    """Запуск полного ML пайплайна"""
    log_message("🚀 Запуск полного ML пайплайна...")
    
    # Настройка окружения
    setup_directories()
    
    try:
        # 1. Проверка данных
        log_message("Этап 1: Проверка качества данных")
        run_data_validation()
        
        # 2. Анализ дрейфа
        log_message("Этап 2: Анализ дрейфа данных")
        run_drift_analysis()
        
        # 3. Обучение и логирование
        log_message("Этап 3: Обучение модели и логирование")
        model, metrics = run_experiment()
        
        log_message("✅ Все этапы пайплайна успешно завершены!")
        
    except Exception as e:
        log_message(f"❌ Ошибка в пайплайне: {e}", "ERROR")
        raise

if __name__ == "__main__":
    run_full_pipeline()