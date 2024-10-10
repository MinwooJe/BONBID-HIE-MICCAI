import tensorflow as tf
from tensorflow.keras.losses import Loss

# Dice Coefficient 및 Loss를 함께 계산하는 클래스
class DiceMetricAndLoss(Loss):
    def __init__(self, name="dice_metric_and_loss", **kwargs):
        super().__init__(name=name, **kwargs)

    def call(self, y_true, y_pred):
        # Flatten the tensors
        y_true_f = tf.keras.backend.flatten(y_true)
        y_pred_f = tf.keras.backend.flatten(y_pred)

        intersection = tf.keras.backend.sum(y_true_f * y_pred_f)
        union = tf.keras.backend.sum(y_true_f) + tf.keras.backend.sum(y_pred_f)

        dice_coefficient = (2. * intersection + 1) / (union + 1)

        return dice_coefficient

    # Loss를 구하는 함수 추가
    def compute_loss(self, y_true, y_pred):
        dice_coefficient = self.call(y_true, y_pred)
        dice_loss = 1 - dice_coefficient
        return dice_loss


# import numpy as np

# # 테스트 1: 전부 0인 값과 전부 1인 값 비교
# y_true_test_1 = tf.constant(np.array([[0, 0, 0], [0, 0, 0]]), dtype=tf.float32)
# y_pred_test_1 = tf.constant(np.array([[1, 1, 1], [1, 1, 1]]), dtype=tf.float32)

# # DiceCoefficient 클래스 인스턴스 생성
# dice_coefficient_fn = DiceCoefficient()

# # Dice Coefficient 계산
# dice_coeff_1 = dice_coefficient_fn(y_true_test_1, y_pred_test_1)
# print("Dice Coefficient (Test 1):", dice_coeff_1.numpy())  # 결과: 0에 가까운 값

# # 테스트 2: 전부 0인 값과 전부 0인 값 비교
# y_true_test_2 = tf.constant(np.array([[0, 0, 0], [0, 0, 0]]), dtype=tf.float32)
# y_pred_test_2 = tf.constant(np.array([[0, 0, 0], [0, 0, 0]]), dtype=tf.float32)

# # Dice Coefficient 계산
# dice_coeff_2 = dice_coefficient_fn(y_true_test_2, y_pred_test_2)
# print("Dice Coefficient (Test 2):", dice_coeff_2.numpy())  # 결과: 1에 가까운 값
