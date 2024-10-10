import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from Model import DiceCoefficient  # 제공된 Model.py에서 임포트

# Keras 모델 파일로부터 예측 및 실제 값 불러오기 함수
def load_model_predictions(keras_file_path, test_data):
    model = load_model(keras_file_path, custom_objects={'DiceCoefficient': DiceCoefficient})  # 모델 로드
    y_pred = model.predict(test_data)  # 모델로 예측 수행
    y_true = np.array([y for _, y in test_data])  # 실제 라벨
    return y_true, y_pred

# 성능 데이터를 불러오기 (keras 파일 경로를 리스트로 설정)
model_results_paths = [
    'best_dice_model.keras',
    'best_dice_model_VNet.keras',
    'Not Aug.keras',
    'Elastic- 0.1.keras'
]

augmentations = ['Best Dice Model', 'Best Dice Model VNet', 'No Augmentation', 'Elastic 0.1']
dice_coefficients_all = []

# 테스트 데이터 로드 (예시로 임의의 테스트 데이터 설정)
# 실제로는 test_data에 사용할 데이터셋을 불러와야 합니다.
# 예시: test_data = some_data_loading_function()
test_data = ...  # 여기에 테스트 데이터 로드

# 각 keras 파일에서 y_true와 y_pred를 불러와 Dice Coefficient 계산
for path in model_results_paths:
    y_true, y_pred = load_model_predictions(path, test_data)
    
    # 제공된 DiceCoefficient 함수를 사용하여 계산 (채널별로 계산)
    dice_coefficient = DiceCoefficient()
    dice_coefficients = []
    for i in range(y_true.shape[-1]):  # 채널별로 Dice Coefficient 계산
        dice_coefficient.update_state(y_true[..., i], y_pred[..., i])
        dice_coefficients.append(dice_coefficient.result().numpy())
    dice_coefficient.reset_state()  # 상태 초기화
    
    dice_coefficients_all.append(dice_coefficients)

# Dice Coefficient 시각화 함수
def plot_dice_coefficients(dice_coefficients, augmentations):
    fig, ax = plt.subplots(figsize=(10, 6))

    # Dice Coefficient 시각화
    for i, label in enumerate(['ET', 'WT', 'TC']):
        ax.plot(augmentations, [coeff[i] for coeff in dice_coefficients], marker='o', label=label)
    
    ax.set_title('Dice Coefficient [%]')
    ax.set_ylabel('Dice Coefficient')
    ax.set_xlabel('Model Names')
    ax.legend()
    
    plt.tight_layout()
    plt.show()

# Rank Score 시각화 함수
def plot_rank_scores(dice_coefficients, augmentations):
    avg_dice_coefficients = [np.mean(coeff) for coeff in dice_coefficients]  # 각 증강 기법의 평균 Dice Coefficient
    rank_scores = np.argsort(-np.array(avg_dice_coefficients)) + 1  # 순위 계산

    # Rank Score 시각화
    plt.figure(figsize=(10, 6))
    bars = plt.bar(augmentations, rank_scores, color=['blue', 'orange'], width=0.6)

    # 막대 그래프에 순위 라벨 추가
    for bar, rank in zip(bars, rank_scores):
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval - 0.2, int(rank), ha='center', va='bottom', color='white', fontsize=12)

    plt.title('Rank Scores of Models')
    plt.xlabel('Model Names')
    plt.ylabel('Rank (Lower is Better)')
    plt.show()

# 불러온 데이터로 시각화
plot_dice_coefficients(dice_coefficients_all, augmentations)
plot_rank_scores(dice_coefficients_all, augmentations)
