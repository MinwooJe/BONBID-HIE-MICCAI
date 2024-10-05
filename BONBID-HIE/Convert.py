import os
import itk
from glob import glob

def split_filename(filepath):
    path = os.path.dirname(filepath)
    filename = os.path.basename(filepath)
    base, extension = os.path.splitext(filename)
    if extension == '.gz':
        base, ext2 = os.path.splitext(base)
        extension = ext2 + extension
    return path, base, extension

def mha_to_nii(input_dir, out_dir):
    try:
        filenames = glob(os.path.join(input_dir, '*.mha'))
        if len(filenames) == 0:
            raise Exception(f'Could not find .mha files in {input_dir}')
        for filename in filenames:
            print(f'Converting image: {filename}')
            img = itk.imread(filename)
            _, base, _ = split_filename(filename)
            out_filename = os.path.join(out_dir, base + '.nii.gz')
            itk.imwrite(img, out_filename)
            print(f'Saved to: {out_filename}')
        return 0
    except Exception as e:
        print(e)
        return 1
def convert_directories(img_dir_dic):
    for input_dir, out_dir in img_dir_dic.items():
        os.makedirs(out_dir, exist_ok=True)
        mha_to_nii(input_dir, out_dir)

if __name__ == "__main__":
    # Training data paths
    ADC_INPUT_PATH = '../data/train/raw_data/1ADC_ss'
    Z_ADC_INPUT_PATH = '../data/train/raw_data/2Z_ADC'
    LABEL_INPUT_PATH = '../data/train/raw_data/3LABEL'

    ADC_OUTPUT_PATH = '../data/train/convert/1ADC_SS'
    Z_ADC_OUTPUT_PATH = '../data/train/convert/2Z_ADC'
    LABEL_OUTPUT_PATH = '../data/train/convert/3LABEL'

    # Validation data paths
    VAL_ADC_INPUT_PATH = '../data/val/1ADC_ss'
    VAL_Z_ADC_INPUT_PATH = '../data/val/2Z_ADC'
    VAL_LABEL_INPUT_PATH = '../data/val/3LABEL'

    VAL_ADC_OUTPUT_PATH = '../data/val/convert/1ADC_SS'
    VAL_Z_ADC_OUTPUT_PATH = '../data/val/convert/2Z_ADC'
    VAL_LABEL_OUTPUT_PATH = '../data/val/convert/3LABEL'

    os.makedirs(ADC_OUTPUT_PATH, exist_ok=True)
    os.makedirs(Z_ADC_OUTPUT_PATH, exist_ok=True)
    os.makedirs(LABEL_OUTPUT_PATH, exist_ok=True)

    os.makedirs(VAL_ADC_OUTPUT_PATH, exist_ok=True)
    os.makedirs(VAL_Z_ADC_OUTPUT_PATH, exist_ok=True)
    os.makedirs(VAL_LABEL_OUTPUT_PATH, exist_ok=True)

    img_dir_dic = {
        ADC_INPUT_PATH: ADC_OUTPUT_PATH,
        Z_ADC_INPUT_PATH: Z_ADC_OUTPUT_PATH,
        LABEL_INPUT_PATH: LABEL_OUTPUT_PATH,
        VAL_ADC_INPUT_PATH: VAL_ADC_OUTPUT_PATH,
        VAL_Z_ADC_INPUT_PATH: VAL_Z_ADC_OUTPUT_PATH,
        VAL_LABEL_INPUT_PATH: VAL_LABEL_OUTPUT_PATH
    }

    for input_dir, out_dir in img_dir_dic.items():
        mha_to_nii(input_dir, out_dir)