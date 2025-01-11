import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer

# 1. CSV 데이터 불러오기 및 전처리
def load_and_preprocess_data(file_path):
    data = pd.read_csv(file_path)
    X = data.drop(columns=['label']).values  # 'label' 열을 제외한 데이터
    y = data['label'].values  # 'label' 열
    X = X / 255.0
    X = X.reshape((-1, 64, 64, 1))  # 데이터 리쉐이핑
    lb = LabelBinarizer()
    y = lb.fit_transform(y)
    return X, y

# 2. CNN 모델 설계
def create_cnn_model(input_shape):
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(2, activation='softmax'))
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# 3. 학습 및 저장
def train_and_save_model(X, y, model_save_path):
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1, random_state=42)
    model = create_cnn_model(X_train.shape[1:])
    model.fit(X_train, y_train, epochs=10, validation_data=(X_val, y_val))
    model.save(model_save_path)  # 모델 저장
    return model

# 실행 예제
X, y = load_and_preprocess_data('train_data.csv')
train_and_save_model(X, y, 'cnn_model.h5')  # 모델 저장 경로 지정