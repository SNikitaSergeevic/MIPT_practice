# ML Pipeline Project

Проект машинного обучения с воспроизводимым пайплайном и мониторингом качества.

## Структура проекта

ml-pipeline-project/

├── data/ # Исходные данные (не в Git)

├── models/ # Обученные модели (Git LFS)

├── src/ # Исходный код

│ ├── init.py

│ ├── data_validation.py

│ ├── drift_analysis.py

│ └── train.py

├── configs/ # Конфигурации

│ └── config.yaml

├── notebooks/ # Исследовательский анализ

├── tests/ # Тесты

├── requirements.txt # Зависимости

└── README.md