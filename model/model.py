import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

# CNN 모델 설계
def create_cnn_model(input_shape):
    model = Sequential()

    # 첫 번째 컨볼루션 레이어와 풀링 레이어
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # 두 번째 컨볼루션 레이어와 풀링 레이어
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # 세 번째 컨볼루션 레이어와 풀링 레이어
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # 드롭아웃을 통해 과적합 방지
    model.add(Dropout(0.5))

    # Flatten 레이어
    model.add(Flatten())

    # 완전 연결 레이어
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))

    # 출력 레이어 (소프트맥스 활성화)
    model.add(Dense(2, activation='softmax'))

    # 모델 컴파일
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    return model

# 입력 데이터의 형태 (예: 64x64 이미지와 유사한 2차원 CSI 데이터를 사용한다고 가정)
input_shape = (64, 64, 1)  # 가로, 세로, 채널 수 (채널 수가 1인 흑백 이미지 형태로 가정)
cnn_model = create_cnn_model(input_shape)

# 모델 요약 출력
cnn_model.summary()