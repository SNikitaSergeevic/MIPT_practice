# server/server.py
import grpc
from concurrent import futures
import pickle
import os
import numpy as np  # <-- ДОБАВЬТЕ ЭТУ СТРОКУ!
import sys

# Добавляем текущую директорию в путь Python
sys.path.append('.')

# Импортируем сгенерированные gRPC файлы
import model_pb2
import model_pb2_grpc

class PredictionServicer(model_pb2_grpc.PredictionServiceServicer):
    def __init__(self):
        self.model_version = os.getenv('MODEL_VERSION', 'v1.0.0')
        model_path = os.getenv('MODEL_PATH', 'models/model.pkl')
        
        print(f"Loading model from: {model_path}")
        print(f"Model version: {self.model_version}")
        
        try:
            with open(model_path, 'rb') as f:
                self.model = pickle.load(f)
            print("Model loaded successfully")
            
            # Проверяем, что модель загружена правильно
            print(f"Model type: {type(self.model)}")
            print(f"Model classes: {self.model.classes_ if hasattr(self.model, 'classes_') else 'N/A'}")
            
        except Exception as e:
            print(f"Error loading model: {e}")
            raise
        
        # Названия классов для Iris dataset
        self.class_names = ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']
        print(f"Class names: {self.class_names}")
    
    def Health(self, request, context):
        """Health check endpoint"""
        print("Health check called")
        return model_pb2.HealthResponse(
            status="ok",
            model_version=self.model_version
        )
    
    def Predict(self, request, context):
        """Predict endpoint"""
        try:
            print(f"Predict called with features: {request.features}")
            
            # Проверяем входные данные
            if len(request.features) != 4:
                error_msg = f"Expected 4 features for Iris dataset, got {len(request.features)}"
                print(error_msg)
                context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
                context.set_details(error_msg)
                return model_pb2.PredictResponse()
            
            # Преобразуем в numpy массив
            features_array = np.array(request.features, dtype=np.float32).reshape(1, -1)
            print(f"Features array shape: {features_array.shape}")
            
            # Делаем предсказание
            prediction_idx = self.model.predict(features_array)[0]
            print(f"Prediction index: {prediction_idx}")
            
            # Получаем вероятности
            confidence = 0.0
            if hasattr(self.model, 'predict_proba'):
                try:
                    probabilities = self.model.predict_proba(features_array)[0]
                    confidence = float(np.max(probabilities))
                    print(f"Probabilities: {probabilities}")
                except Exception as e:
                    print(f"Error getting probabilities: {e}")
                    confidence = 1.0
            else:
                confidence = 1.0
            
            print(f"Confidence: {confidence}")
            
            # Получаем название класса
            if prediction_idx < len(self.class_names):
                prediction_name = self.class_names[prediction_idx]
            else:
                prediction_name = str(prediction_idx)
            
            print(f"Prediction name: {prediction_name}")
            
            response = model_pb2.PredictResponse(
                prediction=prediction_name,
                confidence=confidence,
                model_version=self.model_version
            )
            
            print(f"Response: {response}")
            return response
            
        except Exception as e:
            print(f"Prediction error: {e}")
            import traceback
            traceback.print_exc()
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(f"Prediction error: {str(e)}")
            return model_pb2.PredictResponse()

def serve():
    """Запуск gRPC сервера"""
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    model_pb2_grpc.add_PredictionServiceServicer_to_server(
        PredictionServicer(), server
    )
    
    port = os.getenv('PORT', '50051')
    server_address = f'[::]:{port}'
    server.add_insecure_port(server_address)
    
    print(f"Starting gRPC server on {server_address}")
    print(f"Model path: {os.getenv('MODEL_PATH', 'models/model.pkl')}")
    
    server.start()
    print("Server started successfully")
    
    try:
        server.wait_for_termination()
    except KeyboardInterrupt:
        print("\nShutting down server...")
        server.stop(0)

if __name__ == '__main__':
    serve()