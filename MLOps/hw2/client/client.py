import grpc
import sys
import numpy as np

# Добавляем текущую директорию в путь для импорта
sys.path.append('.')

import model_pb2
import model_pb2_grpc

def test_health():
    """Тестирование Health endpoint"""
    print("Testing /health endpoint...")
    
    with grpc.insecure_channel('localhost:50051') as channel:
        stub = model_pb2_grpc.PredictionServiceStub(channel)
        try:
            response = stub.Health(model_pb2.HealthRequest(), timeout=10)
            print(f"Health Response: status={response.status}, version={response.model_version}")
            return True
        except grpc.RpcError as e:
            print(f"Health Error: {e}")
            return False

def test_predict():
    """Тестирование Predict endpoint"""
    print("\nTesting /predict endpoint...")
    
    # Пример признаков для ириса (можно изменить)
    # Для Iris dataset: [sepal_length, sepal_width, petal_length, petal_width]
    features = [5.1, 3.5, 1.4, 0.2]  # Пример для Iris-setosa
    
    with grpc.insecure_channel('localhost:50051') as channel:
        stub = model_pb2_grpc.PredictionServiceStub(channel)
        try:
            request = model_pb2.PredictRequest(features=features)
            response = stub.Predict(request, timeout=10)
            print(f"Predict Response: prediction={response.prediction}, "
                  f"confidence={response.confidence:.4f}, "
                  f"version={response.model_version}")
            return True
        except grpc.RpcError as e:
            print(f"Predict Error: {e}")
            return False

def run():
    """Основная функция для тестирования"""
    print("Starting gRPC client tests...")
    
    # Тестируем Health
    if not test_health():
        print("Health test failed!")
        return
    
    # Тестируем Predict
    if not test_predict():
        print("Predict test failed!")
        return
    
    print("\nAll tests passed successfully!")

if __name__ == '__main__':
    run()