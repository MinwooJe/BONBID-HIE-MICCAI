import tensorflow as tf
import os
import matplotlib.pyplot as plt
from preprocessing import *
from sklearn.model_selection import ParameterGrid
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.layers import Input, Conv3D, MaxPooling3D, Conv3DTranspose, BatchNormalization, Activation, Dropout, concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import MeanIoU

# 커스텀 DiceCoefficient 메트릭
class DiceCoefficient(tf.keras.metrics.Metric):
    def __init__(self, name='dice_coefficient', **kwargs):
        super(DiceCoefficient, self).__init__(name=name, **kwargs)
        self.intersection = self.add_weight(name='intersection', initializer='zeros')
        self.union = self.add_weight(name='union', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true_f = tf.keras.backend.flatten(y_true)
        y_pred_f = tf.keras.backend.flatten(y_pred)

        mask = tf.logical_or(tf.not_equal(y_true_f, 0), tf.not_equal(y_pred_f, 0))
        y_true_f = tf.boolean_mask(y_true_f, mask)
        y_pred_f = tf.boolean_mask(y_pred_f, mask)
        
        intersection = tf.keras.backend.sum(y_true_f * y_pred_f)
        union = tf.keras.backend.sum(y_true_f) + tf.keras.backend.sum(y_pred_f)
        
        self.intersection.assign_add(intersection)
        self.union.assign_add(union)

    def result(self):
        return (2. * self.intersection + 1) / (self.union + 1)

    def reset_state(self):
        self.intersection.assign(0.0)
        self.union.assign(0.0)

# 커스텀 DiceLoss 손실 함수
class DiceLoss(tf.keras.losses.Loss):
    def call(self, y_true, y_pred):
        y_true_f = tf.keras.backend.flatten(y_true)
        y_pred_f = tf.keras.backend.flatten(y_pred)

        mask = tf.logical_or(tf.not_equal(y_true_f, 0), tf.not_equal(y_pred_f, 0))
        y_true_f = tf.boolean_mask(y_true_f, mask)
        y_pred_f = tf.boolean_mask(y_pred_f, mask)

        y_pred_f = tf.keras.backend.clip(y_pred_f, 0, 1)
        intersection = tf.keras.backend.sum(y_true_f * y_pred_f)
        return 1 - (2. * intersection + 1) / (tf.keras.backend.sum(y_true_f) + tf.keras.backend.sum(y_pred_f) + 1)

kernel_initializer = 'he_uniform'

class Unet3DModel:
    def __init__(self, img_height, img_width, img_depth, img_channels, kernel_initializer='he_uniform'):
        self.img_height = img_height
        self.img_width = img_width
        self.img_depth = img_depth
        self.img_channels = img_channels
        self.kernel_initializer = kernel_initializer
        self.model = self.Unet_model()

    def CBR3D(self, x, filters, kernel_size=(3, 3, 3), activation='relu', padding='same', dropout_rate=0.0):
        x = Conv3D(filters, kernel_size, padding=padding, kernel_initializer=self.kernel_initializer)(x)
        x = BatchNormalization()(x)
        x = Activation(activation)(x)
        if dropout_rate > 0.0:
            x = Dropout(dropout_rate)(x)
        return x

    def Unet_model(self):
        inputs_image = Input((self.img_height, self.img_width, self.img_depth, 2), name='image')

        c1_image = self.CBR3D(inputs_image, 16, dropout_rate=0.1)
        c1_image = self.CBR3D(c1_image, 16)
        p1_image = MaxPooling3D((2, 2, 2))(c1_image)

        c2 = self.CBR3D(p1_image, 32, dropout_rate=0.1)
        c2 = self.CBR3D(c2, 32)
        p2 = MaxPooling3D((2, 2, 2))(c2)

        c3 = self.CBR3D(p2, 64, dropout_rate=0.2)
        c3 = self.CBR3D(c3, 64)
        p3 = MaxPooling3D((2, 2, 2))(c3)

        c4 = self.CBR3D(p3, 128, dropout_rate=0.2)
        c4 = self.CBR3D(c4, 128)
        p4 = MaxPooling3D(pool_size=(2, 2, 2))(c4)

        c5 = self.CBR3D(p4, 256, dropout_rate=0.3)
        c5 = self.CBR3D(c5, 256)

        # Expansive path
        u6 = Conv3DTranspose(128, (2, 2, 2), strides=(2, 2, 2), padding='same')(c5)
        u6 = concatenate([u6, c4])
        c6 = self.CBR3D(u6, 128, dropout_rate=0.2)
        c6 = self.CBR3D(c6, 128)

        u7 = Conv3DTranspose(64, (2, 2, 2), strides=(2, 2, 2), padding='same')(c6)
        u7 = concatenate([u7, c3])
        c7 = self.CBR3D(u7, 64, dropout_rate=0.2)
        c7 = self.CBR3D(c7, 64)

        u8 = Conv3DTranspose(32, (2, 2, 2), strides=(2, 2, 2), padding='same')(c7)
        u8 = concatenate([u8, c2])
        c8 = self.CBR3D(u8, 32, dropout_rate=0.1)
        c8 = self.CBR3D(c8, 32)

        u9 = Conv3DTranspose(16, (2, 2, 2), strides=(2, 2, 2), padding='same')(c8)
        u9 = concatenate([u9, c1_image])
        c9 = self.CBR3D(u9, 16, dropout_rate=0.1)
        c9 = self.CBR3D(c9, 16)

        outputs = Conv3D(1, (1, 1, 1), activation='sigmoid')(c9)

        model = Model(inputs=[inputs_image], outputs=[outputs])
        model.summary()

        model.compile(optimizer='adam', loss=DiceLoss(), metrics=['accuracy', MeanIoU(num_classes=2), DiceCoefficient()])

        return model

    def train(self, train_tf_dataloader, val_tf_dataloader, batch_size=1, epochs=100):
        BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        saved_model_path = os.path.join(BASE_DIR, 'models')
        checkpoint_path = os.path.join(saved_model_path, 'best_dice_model.keras')

        checkpoint_dice = ModelCheckpoint(checkpoint_path, monitor='val_dice_coefficient', verbose=1,
                                          save_best_only=True, mode='max')
        early_stopping = EarlyStopping(monitor='val_loss', patience=10, verbose=1, mode='min')

        history = self.model.fit(
            train_tf_dataloader,
            validation_data=val_tf_dataloader,
            epochs=epochs,
            callbacks=[checkpoint_dice, early_stopping]
        )

        return history
    
    def plot_prob_performance(self, prob_history):
        # prob 값에 따른 성능 그래프 그리기
        probs_contrast, probs_elastic, scores = zip(*prob_history)
        plt.figure(figsize=(10, 6))
        plt.scatter(probs_contrast, probs_elastic, c=scores, cmap='viridis', marker='o')
        plt.colorbar(label='Validation Dice Coefficient')
        plt.xlabel('RandAdjustContrastd Prob Value')
        plt.ylabel('Rand3DElasticd Prob Value')
        plt.title('Grid Search Prob Tuning')
        plt.grid(True)
        plt.show()

    def plot_epoch_performance(self, all_histories):
        # 모든 prob 값에 대한 성능 변화 그래프 그리기
        plt.figure(figsize=(12, 8))
        for (prob_contrast, prob_elastic), history in all_histories.items():
            plt.plot(range(1, len(history) + 1), history, label=f'Prob contrast: {prob_contrast}, Prob elastic: {prob_elastic}')
        plt.xlabel('Epoch')
        plt.ylabel('Validation Dice Coefficient')
        plt.title('Validation Dice Coefficient Across Epochs for Different Prob Values')
        plt.legend()
        plt.grid(True)
        plt.show()

    def grid_search_prob_tuning(self, train_path, val_path, param_grid, epochs=100):
        param_list = list(ParameterGrid(param_grid))
        best_score = float('-inf')
        best_params = None

        prob_history = []  # prob 값과 성능을 저장할 리스트
        all_histories = {}  # 각 prob 값에 대한 모든 모델의 성능 기록

        for params in param_list:
            prob_contrast = params['prob_contrast']
            prob_elastic = params['prob_elastic']

            print(f"Training with prob_contrast={prob_contrast}, prob_elastic={prob_elastic}")

            train_dataset = preprocess_dataset(train_path, prob_contrast=prob_contrast, prob_elastic=prob_elastic, is_train=True)
            val_dataset = preprocess_dataset(val_path, prob_contrast=prob_contrast, prob_elastic=prob_elastic, is_train=False)

            train_tf_dataloader = convert_monai_to_tf_dataset(train_dataset, batch_size=1)
            val_tf_dataloader = convert_monai_to_tf_dataset(val_dataset, batch_size=1)

            model = Unet3DModel(img_height=256, img_width=256, img_depth=64, img_channels=2)

            history = model.train(train_tf_dataloader, val_tf_dataloader, batch_size=1, epochs=epochs)

            val_dice_coefficient = history.history['val_dice_coefficient'][-1]

            # prob 값과 해당 성능을 저장
            prob_history.append((prob_contrast, prob_elastic, val_dice_coefficient))

            # 각 prob 값에 대한 모든 에포크의 성능 기록
            all_histories[(prob_contrast, prob_elastic)] = history.history['val_dice_coefficient']

            if val_dice_coefficient > best_score:
                best_score = val_dice_coefficient
                best_params = params

        self.plot_prob_performance(prob_history)
        self.plot_epoch_performance(all_histories)

        return best_params, best_score

