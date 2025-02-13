import pandas as pd
import numpy as np
from ast import literal_eval
import os
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import json

def preprocess_csi_data(data_path):
    print(f"Loading data from: {data_path}")
    df = pd.read_csv(data_path)
    print("Original data shape:", df.shape)
    
    # CSI 데이터 추출 및 전처리
    processed_data = []
    valid_labels = []
    
    for idx, row in df.iterrows():
        try:
            # Csi 컬럼에서 데이터 추출
            csi_str = row['Csi']
            if pd.isna(csi_str):
                continue
            
            # 문자열을 리스트로 변환 (큰따옴표와 대괄호 제거)
            csi_str = csi_str.strip('"[]')
            csi_values = [float(x) for x in csi_str.split(',') if x.strip()]
            csi_values = np.array(csi_values)
            
            # 데이터 검증
            if len(csi_values) < 100:  # 최소 길이 체크
                continue
                
            processed_data.append(csi_values)
            valid_labels.append(row['Label'])
            
            if idx % 100 == 0:
                print(f"Processed {idx} rows, valid samples: {len(processed_data)}")
                if len(processed_data) > 0:
                    print(f"Sample data length: {len(processed_data[-1])}")
                    
        except Exception as e:
            print(f"Error processing row {idx}: {e}")
            continue
    
    if len(processed_data) == 0:
        raise ValueError("No valid samples found in the dataset")
    
    # 데이터 길이 통일 (모든 샘플을 300개 포인트로 맞춤)
    target_length = 300
    processed_data_adjusted = []
    valid_labels_adjusted = []
    
    for data, label in zip(processed_data, valid_labels):
        if len(data) >= target_length:
            processed_data_adjusted.append(data[:target_length])
            valid_labels_adjusted.append(label)
        else:
            # 짧은 데이터는 패딩
            padded = np.pad(data, (0, target_length - len(data)))
            processed_data_adjusted.append(padded)
            valid_labels_adjusted.append(label)
    
    # numpy 배열로 변환
    X = np.array(processed_data_adjusted)
    
    # 데이터 정규화
    X = (X - np.mean(X)) / (np.std(X) + 1e-10)
    
    # CNN 입력을 위한 reshape
    X = X.reshape(-1, target_length, 1)
    
    # 레이블 처리
    unique_labels = sorted(set(valid_labels_adjusted))
    print(f"\nUnique labels in data: {unique_labels}")
    
    label_mapping = {label: idx for idx, label in enumerate(unique_labels)}
    y = np.array([label_mapping[label] for label in valid_labels_adjusted])
    
    print("\nFinal processed data:")
    print("X shape:", X.shape)
    print("y shape:", y.shape)
    print("Label mapping:", label_mapping)
    print("Samples per class:", {label: count for label, count in zip(unique_labels, np.bincount(y))})
    
    return X, y

def create_cnn_model(input_shape, num_classes):
    model = models.Sequential([
        # 입력 레이어
        layers.Input(shape=input_shape),
        
        # 첫 번째 Conv 블록
        layers.Conv1D(64, kernel_size=5, activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling1D(pool_size=2),
        
        # 두 번째 Conv 블록
        layers.Conv1D(128, kernel_size=5, activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling1D(pool_size=2),
        
        # 세 번째 Conv 블록
        layers.Conv1D(256, kernel_size=3, activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling1D(pool_size=2),
        
        # Dense 레이어
        layers.Flatten(),
        layers.Dropout(0.3),  # 드롭아웃 비율 조정
        layers.Dense(256, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        layers.Dense(num_classes, activation='softmax')
    ])
    
    return model

def plot_training_history(history):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # 정확도 그래프
    ax1.plot(history.history['accuracy'], label='train')
    ax1.plot(history.history['val_accuracy'], label='validation')
    ax1.set_title('Model Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.legend()
    
    # 손실 그래프
    ax2.plot(history.history['loss'], label='train')
    ax2.plot(history.history['val_loss'], label='validation')
    ax2.set_title('Model Loss')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()
    
    plt.tight_layout()
    plt.show()

def plot_confusion_matrix(y_true, y_pred, classes):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=classes, yticklabels=classes)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.show()

def save_model_and_metadata(model, class_names, save_dir='saved_model'):
    """모델과 메타데이터를 저장하는 함수"""
    try:
        # 저장 디렉토리의 절대 경로 얻기
        current_dir = os.path.dirname(os.path.abspath(__file__))
        save_dir = os.path.join(current_dir, save_dir)
        
        # 저장 디렉토리 생성
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
            print(f"Created directory: {save_dir}")
        
        # 모델 저장
        model_path = os.path.join(save_dir, 'wifi_gesture_model.h5')
        model.save(model_path)
        print(f"Model saved successfully to: {model_path}")
        
        # 클래스 정보 저장
        metadata = {
            'class_names': class_names,
            'input_shape': list(model.input_shape[1:]),
            'num_classes': len(class_names)
        }
        
        metadata_path = os.path.join(save_dir, 'model_metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f)
        print(f"Metadata saved successfully to: {metadata_path}")
        
        # 저장된 파일 확인
        saved_files = os.listdir(save_dir)
        print("\nFiles in save directory:")
        for file in saved_files:
            file_path = os.path.join(save_dir, file)
            file_size = os.path.getsize(file_path) / (1024 * 1024)  # MB로 변환
            print(f"- {file} ({file_size:.2f} MB)")
            
    except Exception as e:
        print(f"Error saving model and metadata: {e}")
        raise

def main():
    try:
        # 데이터 로드 및 전처리
        current_dir = os.path.dirname(os.path.abspath(__file__))
        data_path = os.path.join(current_dir, "rock_paper_data.csv")
        X, y = preprocess_csi_data(data_path)
        
        # 데이터 분할 (train: 70%, validation: 15%, test: 15%)
        X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
        X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
        
        print("\nData split sizes:")
        print(f"Train: {X_train.shape[0]} samples")
        print(f"Validation: {X_val.shape[0]} samples")
        print(f"Test: {X_test.shape[0]} samples")
        
        # 모델 생성
        num_classes = len(np.unique(y))
        input_shape = (X.shape[1], 1)
        model = create_cnn_model(input_shape, num_classes)
        
        # 모델 컴파일
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        print("\nModel Summary:")
        model.summary()
        
        # Early Stopping 설정 수정
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=20,  # 더 긴 patience
            restore_best_weights=True,
            verbose=1
        )
        
        # Learning Rate 조정 추가
        reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=10,
            min_lr=0.00001,
            verbose=1
        )
        
        # 모델 학습
        print("\nTraining model...")
        history = model.fit(
            X_train, y_train,
            epochs=200,  # 더 많은 에폭
            batch_size=32,
            validation_data=(X_val, y_val),
            callbacks=[early_stopping, reduce_lr],
            class_weight={  # 클래스 가중치 추가
                0: 1.0,
                1: 1.0
            },
            verbose=1
        )
        
        # 학습 결과 시각화
        plot_training_history(history)
        
        # 테스트 세트에서 성능 평가
        print("\nEvaluating on test set:")
        test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
        print(f"Test accuracy: {test_acc:.4f}")
        
        # 예측 및 분류 보고서
        y_pred = model.predict(X_test)
        y_pred_classes = np.argmax(y_pred, axis=1)
        
        print("\nClassification Report:")
        class_names = [f"Class {i}" for i in range(num_classes)]
        print(classification_report(y_test, y_pred_classes, target_names=class_names))
        
        # 혼동 행렬 시각화
        plot_confusion_matrix(y_test, y_pred_classes, class_names)
        
        # 모델 저장
        save_model_and_metadata(model, class_names)
        
    except Exception as e:
        print(f"Error during processing: {e}")
        raise

if __name__ == "__main__":
    main()
