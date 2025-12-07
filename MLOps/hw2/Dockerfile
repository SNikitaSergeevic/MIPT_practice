FROM python:3.11-slim

# Установка системных зависимостей
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Установка переменных окружения
ENV PORT=50051 \
    MODEL_PATH=/app/models/model.pkl \
    MODEL_VERSION=v1.0.0 \
    PYTHONPATH=/app

WORKDIR /app

# Копирование зависимостей
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Копирование protos и генерация gRPC кода
COPY protos/ ./protos/
RUN python -m grpc_tools.protoc \
    -I./protos \
    --python_out=. \
    --grpc_python_out=. \
    ./protos/model.proto

# Копирование остального кода
COPY . .

# Создание директорий
RUN mkdir -p /app/models

EXPOSE 50051

CMD ["python", "-m", "server.server"]
