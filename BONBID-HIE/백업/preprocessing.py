import os
import monai 

from monai.transforms import (
    LoadImaged,
    EnsureChannelFirstd,
    Compose,
    Resized,
    ConcatItemsd,
    DeleteItemsd,
    Rand3DElasticd
)
import tensorflow as tf
import numpy as np
from monai.data import Dataset

import numpy as np
from monai.transforms import Transform
from monai.transforms.utils import ensure_tuple

class RandomGammaIntensity(Transform):
    def __init__(self, gamma_range=(0.8, 1.2), gain_range=(0.8, 1.2), prob=0.5):
        self.gamma_range = ensure_tuple(gamma_range)
        self.gain_range = ensure_tuple(gain_range)
        self.prob = prob

    def __call__(self, img):
        if np.random.rand() > self.prob:
            return img
        gamma = np.random.uniform(self.gamma_range[0], self.gamma_range[1])
        gain = np.random.uniform(self.gain_range[0], self.gain_range[1])
        
        img = img.astype(np.float32)
        img = gain * np.power(img, gamma)
        
        return img.astype(np.float32)

def combine_paths_to_dic(adc_dir, z_adc_dir, label_dir):
    adc_paths = [{'adc_ss': os.path.join(adc_dir, filename)} for filename in os.listdir(adc_dir) if filename.endswith('.nii.gz')]
    z_adc_paths = [{'z_adc': os.path.join(z_adc_dir, filename)} for filename in os.listdir(z_adc_dir) if filename.endswith('.nii.gz')]
    label_paths = [{'label': os.path.join(label_dir, filename)} for filename in os.listdir(label_dir) if filename.endswith('.nii.gz')]
    
    adc_paths.sort(key=lambda x: os.path.basename(x['adc_ss']))
    z_adc_paths.sort(key=lambda x: os.path.basename(x['z_adc']))
    label_paths.sort(key=lambda x: os.path.basename(x['label']))

    combined_paths = []
    for adc, z_adc, label in zip(adc_paths, z_adc_paths, label_paths):
        combined_path = adc.copy()
        combined_path.update(z_adc)
        combined_path.update(label)
        combined_paths.append(combined_path)

    return combined_paths

def get_transforms(prob_contrast=0.5, prob_elastic=0.5, is_train=True):
    transforms = [
        LoadImaged(keys=["adc_ss", "z_adc", "label"], reader="NibabelReader"),
        EnsureChannelFirstd(keys=["adc_ss", "z_adc", "label"]),
        Resized(keys=["adc_ss", "z_adc", "label"], spatial_size=(256, 256, 64)),
        ConcatItemsd(keys=["adc_ss", "z_adc"], name="image", dim=0),
        DeleteItemsd(keys=["adc_ss", "z_adc"]),
    ]
    if is_train:
        transforms += [
            RandomGammaIntensity(gamma_range=(0.8, 1.2), gain_range=(0.8, 1.2), prob=prob_contrast),
            Rand3DElasticd(keys=["image"], sigma_range=(2, 2), magnitude_range=(100, 200), prob=prob_elastic, spatial_size=(256, 256, 64))
        ]
    return Compose(transforms)

# ## 학습 할 때 사용
# def create_and_preprocess_dataset(adc_path, z_adc_path, label_path, prob=0.5, is_train=True):
#     data_paths = combine_paths_to_dic(adc_path, z_adc_path, label_path)
#     transforms = get_transforms(prob=prob, is_train=is_train)
#     dataset = Dataset(data=data_paths, transform=transforms)
#     return dataset

# # Grid Search 할 때 사용
def create_dataset(adc_path, z_adc_path, label_path):
    data_paths = combine_paths_to_dic(adc_path, z_adc_path, label_path)
    return data_paths

def preprocess_dataset(data_paths, prob_contrast=0.5, prob_elastic=0.5, is_train=True):
    transforms = get_transforms(prob_contrast=prob_contrast, prob_elastic=prob_elastic, is_train=is_train)
    dataset = Dataset(data=data_paths, transform=transforms)
    return dataset

##

def data_generator(monai_dataset):
    for data in monai_dataset:
        image = data['image']
        label = data['label']
        
        image = np.transpose(image, (1, 2, 3, 0))
        label = np.transpose(label, (1, 2, 3, 0))
        
        yield image, label

def convert_monai_to_tf_dataset(monai_dataset, batch_size=1):
    output_signature = (
        tf.TensorSpec(shape=(256, 256, 64, 2), dtype=tf.float32),  # Image
        tf.TensorSpec(shape=(256, 256, 64, 1), dtype=tf.float32)   # Label
    )
    
    tf_dataset = tf.data.Dataset.from_generator(
        lambda: data_generator(monai_dataset),
        output_signature=output_signature
    )
    tf_dataset = tf_dataset.batch(batch_size)
    tf_dataset = tf_dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
    
    return tf_dataset