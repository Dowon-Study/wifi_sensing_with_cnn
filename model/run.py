import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

# 데이터 전처리 함수 (예: 리쉐이핑 및 정규화)
def preprocess_data(file_path):
    data = pd.read_csv(file_path)
    csi_values = data['CSI_VALUES'].apply(lambda x: np.fromstring(x.strip('[]'), sep=' '))  # 문자열로 저장된 리스트를 numpy 배열로 변환
    X = np.array(list(csi_values))
    X = X / 255.0  # 정규화
    X = X.reshape((-1, 64, 64, 1))  # 리쉐이핑 (필요에 따라 크기 조정)
    return X

# 예측 및 출력 함수
def predict_and_print(model_path, data_file):
    model = load_model(model_path)
    X_new = preprocess_data(data_file)
    predictions = model.predict(X_new)
    labels = ['empty', 'water']  # 레이블 순서가 'empty'와 'water'라고 가정
    for i, pred in enumerate(predictions):
        label = labels[np.argmax(pred)]
        print(f"샘플 {i + 1}: {label}")

# 예측 실행 예제
predict_and_print('cnn_model.h5', 'new_csi_data.csv')