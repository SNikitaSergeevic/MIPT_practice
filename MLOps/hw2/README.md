# ML gRPC Service - Домашнее задание 2

Минимальный gRPC сервис для обслуживания ML-модели с эндпоинтами `/health` и `/predict`. Проект реализован в рамках модуля 2 «Архитектурные паттерны для обслуживания ML-моделей».

## 📋 Требования задания

- ✅ Реализация gRPC-сервиса с методами `/health` и `/predict`
- ✅ Описание контракта API в Protocol Buffers
- ✅ Генерация Python-кода из proto файла
- ✅ Подключение ML-модели (Random Forest на Iris dataset)
- ✅ Контейнеризация с Docker
- ✅ Локальное тестирование эндпоинтов

## 🏗️ Структура проекта

```
ml_grpc_service/
├── protos/
│   └── model.proto              # gRPC контракт
├── server/
│   ├── __init__.py
│   └── server.py                # gRPC сервер
├── client/
│   ├── __init__.py
│   └── client.py                # gRPC клиент для тестирования
├── models/
│   └── model.pkl                # Обученная модель
├── requirements.txt             # Зависимости Python
├── Dockerfile                   # Конфигурация Docker
├── .dockerignore                # Исключения для Docker
├── train_model.py               # скрипт обучения модели
├── generate_proto.py            # генерация gRPC кода
├── model_pb2.py                 # protobuf код
├── model_pb2_grpc.py            # gRPC код
└── README.md                    # Документация
```

## 🚀 Команды сборки и запуска

### 1. Локальная установка

```bash
# Клонирование репозитория
git clone <repository-url>
cd ml_grpc_service

# Установка зависимостей
pip install -r requirements.txt

# gRPC код
python generate_proto.py

# Обучение модели
python train_model.py

# запуск сервера
python -m server.server
```

### 2. Запуск через Docker

```bash
# Сборка Docker образа
docker build -t grpc-ml-service .

# Запуск контейнера
docker run -p 50051:50051 grpc-ml-service
```

## 📡 Примеры вызовов эндпоинтов

### 1. Проверка /health через grpcurl

```bash
grpcurl -plaintext localhost:50051 mlservice.vl.PredictionService/Health
```

**Ожидаемый ответ:**
```json
{
  "status": "ok",
  "modelVersion": "v1.0.0"
}
```

### 2. Проверка /predict через клиент

```bash
python -m client.client
```

**Ожидаемый вывод:**
```
Starting gRPC client tests...
Testing /health endpoint...
Health Response: status=ok, version=v1.0.0

Testing /predict endpoint...
Predict Response: prediction=Iris-setosa, confidence=0.9200, version=v1.0.0

All tests passed successfully!
```

### 3. Проверка /predict через grpcurl

```bash
grpcurl -plaintext -d '{"features": [5.1, 3.5, 1.4, 0.2]}' \
  localhost:50051 mlservice.vl.PredictionService/Predict
```

**Ожидаемый ответ:**
```json
{
  "prediction": "Iris-setosa",
  "confidence": 0.92,
  "modelVersion": "v1.0.0"
}
```

## 🔧 Технические детали

### Переменные окружения
- `PORT=50051` - порт gRPC сервера
- `MODEL_PATH=/app/models/model.pkl` - путь к модели
- `MODEL_VERSION=v1.0.0` - версия модели

## 📊 Модель данных

**Датасет:** Iris (150 samples, 3 classes)
**Признаки (4 числовых):**
1. sepal length (см)
2. sepal width (см)
3. petal length (см)
4. petal width (см)

**Классы:**
- Iris-setosa
- Iris-versicolor
- Iris-virginica

**Модель:** RandomForestClassifier (100 деревьев)
**Точность:** ~96.7%

## 🧪 Тестирование

### Интеграционное тестирование
```bash
# Запуск всех тестов
python -m client.client

# Индивидуальное тестирование
python -c "
import grpc
import model_pb2
import model_pb2_grpc

channel = grpc.insecure_channel('localhost:50051')
stub = model_pb2_grpc.PredictionServiceStub(channel)

# Test Health
health_response = stub.Health(model_pb2.HealthRequest())
print(f'Health Status: {health_response.status}')

# Test Predict
test_cases = [
    ([5.1, 3.5, 1.4, 0.2], 'Iris-setosa'),
    ([6.0, 2.7, 5.1, 1.6], 'Iris-versicolor'),
    ([6.7, 3.0, 5.2, 2.3], 'Iris-virginica')
]

for features, expected in test_cases:
    request = model_pb2.PredictRequest(features=features)
    response = stub.Predict(request)
    print(f'Features: {features} -> Prediction: {response.prediction}, Confidence: {response.confidence:.2f}')
"
```


## 🎯 Примеры использования

### Пример: Использование в Python приложении
```python
import grpc
import model_pb2
import model_pb2_grpc

class MLServiceClient:
    def __init__(self, host='localhost:50051'):
        self.channel = grpc.insecure_channel(host)
        self.stub = model_pb2_grpc.PredictionServiceStub(self.channel)
    
    def check_health(self):
        response = self.stub.Health(model_pb2.HealthRequest())
        return response.status, response.model_version
    
    def predict(self, features):
        request = model_pb2.PredictRequest(features=features)
        response = self.stub.Predict(request)
        return response.prediction, response.confidence

# Использование
client = MLServiceClient()
status, version = client.check_health()
prediction, confidence = client.predict([5.1, 3.5, 1.4, 0.2])
```

---

**Автор:** [Никита С.]  
**Курс:** MLOps  
**Дата:** 2025  
**Версия:** 1.0.0