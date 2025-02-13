import pandas as pd
from sklearn.model_selection import train_test_split
import os

def create_train_data():
    # 현재 파일의 디렉토리 경로 가져오기
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(current_dir))
    
    # 원본 데이터 파일 경로
    original_data_path = os.path.join(project_root, 'data', 'rock_paper_data.csv')
    
    # 저장할 train 데이터 파일 경로
    train_data_path = os.path.join(project_root, 'data', 'train_data.csv')
    
    # 데이터 읽기
    data = pd.read_csv(original_data_path)
    
    # 데이터를 특성(X)과 레이블(y)로 분리
    X = data.drop('label', axis=1)  # label 열을 제외한 모든 열이 특성
    y = data['label']  # label 열이 타겟 변수
    
    # 데이터 분할 (80% 훈련용, 20% 테스트용)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # 훈련 데이터를 다시 합치기
    train_data = pd.concat([X_train, y_train], axis=1)
    
    # train_data.csv로 저장
    train_data.to_csv(train_data_path, index=False)
    print(f"Training data saved to {train_data_path}")
    
    return train_data_path

# 기존의 load_and_preprocess_data 함수 수정
def load_and_preprocess_data(file_path):
    # 먼저 train_data.csv 생성
    train_data_path = create_train_data()
    
    # 생성된 train_data.csv 파일 로드
    data = pd.read_csv(train_data_path)
    return data