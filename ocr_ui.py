import os
import pathlib
import tensorflow as tf
import numpy as np
import turtle

from model import preprocess_image, OCRModel


def main():
    model = tf.keras.models.load_model('models/ocr_model_xs_diff_para')
    ocr_model = OCRModel(model)

    # image = preprocess_image('sample_input.png', 'processed.png')
    # result = ocr_model(image)

    result = ocr_model('images/random_scale_img_f.png')

    print(f"Prediction result: {result}")


if __name__ == '__main__':
    main()