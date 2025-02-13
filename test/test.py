import tensorflow as tf
import numpy as np
import json
import os

class WifiGesturePredictor:
    def __init__(self, model_dir='saved_model'):
        # 모델 로드
        self.model_path = os.path.join(model_dir, 'wifi_gesture_model')
        self.model = tf.keras.models.load_model(self.model_path)
        
        # 메타데이터 로드
        metadata_path = os.path.join(model_dir, 'model_metadata.json')
        with open(metadata_path, 'r') as f:
            self.metadata = json.load(f)
        
        self.class_names = self.metadata['class_names']
        self.input_shape = self.metadata['input_shape']
        
        print("Model loaded successfully")
        print(f"Input shape: {self.input_shape}")
        print(f"Classes: {self.class_names}")
    
    def preprocess_data(self, csi_data):
        """CSI 데이터 전처리"""
        # 데이터 shape 맞추기
        if len(csi_data) > self.input_shape[0]:
            csi_data = csi_data[:self.input_shape[0]]
        elif len(csi_data) < self.input_shape[0]:
            pad_width = self.input_shape[0] - len(csi_data)
            csi_data = np.pad(csi_data, (0, pad_width))
        
        # 정규화
        csi_data = (csi_data - np.mean(csi_data)) / (np.std(csi_data) + 1e-10)
        
        # 모델 입력 shape에 맞게 reshape
        csi_data = csi_data.reshape(1, self.input_shape[0], 1)
        
        return csi_data
    
    def predict(self, csi_data):
        """제스처 예측"""
        # 데이터 전처리
        processed_data = self.preprocess_data(csi_data)
        
        # 예측
        predictions = self.model.predict(processed_data)
        predicted_class_idx = np.argmax(predictions[0])
        confidence = predictions[0][predicted_class_idx]
        
        return {
            'class': self.class_names[predicted_class_idx],
            'confidence': float(confidence),
            'probabilities': {
                class_name: float(prob)
                for class_name, prob in zip(self.class_names, predictions[0])
            }
        }

# 사용 예시
if __name__ == "__main__":
    # 예측기 초기화
    predictor = WifiGesturePredictor()
    
    # 테스트 데이터로 예측
    test_data = np.random.random(300)  # 테스트용 더미 데이터
    result = predictor.predict(test_data)
    
    print("\nPrediction result:")
    print(f"Predicted class: {result['class']}")
    print(f"Confidence: {result['confidence']:.4f}")
    print("\nClass probabilities:")
    for class_name, prob in result['probabilities'].items():
        print(f"{class_name}: {prob:.4f}")