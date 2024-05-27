#%%

import os
import pathlib
import shutil
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import cv2

# from tensorflow_datasets import load
from scipy.ndimage import gaussian_filter
from IPython.display import display, clear_output
from keras import models, layers


# GLOBAL VARIABLES
SEED = 42
tf.random.set_seed(SEED)
np.random.seed(SEED)

# DOWNLOAD_DATA_DIR = 'C:\\Users\\User\\Desktop\\Python\\OCR_Workshop\\data'
MERGE_DATA_DIR = 'D:\\training_data\\data\\by_merge'
DATA_DIR = 'D:\\training_data\\balanced_MNIST'

EPOCH = 5


def main():
    #
    # download data, by_merge, unzip, NIST
    #    
    # download_data_dir = pathlib.Path(DOWNLOAD_DATA_DIR)
    # if not data_dir.exists():
    #     tf.keras.utils.get_file(
    #         'by_merge.zip',
    #         origin='https://s3.amazonaws.com/nist-srd/SD19/by_merge.zip',
    #         extract=True,
    #         cache_dir='.',
    #         cache_subdir='data'
    #     )

    #
    # reorganize folder, rename data files
    #
    #
    # rename_and_move_file()

    # data_dir = pathlib.Path(DATA_DIR)
    # class_names = tf.io.gfile.listdir(str(data_dir))
    # print(class_names)

    #
    # kaggle, EMNIST balanced
    #
    train_set = pd.read_csv("./kaggle/emnist-balanced-train.csv")
    test_set = pd.read_csv("./kaggle/emnist-balanced-test.csv")
    mapping = pd.read_csv("./kaggle/emnist-balanced-mapping.txt", sep = ' ', header = None)

    display(train_set.head())
    train_set.info()
    # last element is labels
    print(f"\nTrain set shape:  {train_set.shape}")
    print(f"Test set shape:  {test_set.shape}\n")

    # see class frequency
    labels1 = train_set["45"].values
    labels=set(labels1)
    labels=list(labels)
    labels = [str(label) for label in labels]
    print(f"Labels: {labels}\n")

    plt.figure(figsize=(20,6))
    sns.countplot(x=labels1)

    mapping.head()
    class_mapping = get_class_mapping(mapping)
    print(f"{class_mapping}\n")

    # extract label and training data
    y_train = np.array(train_set.iloc[:,0].values)
    x_train = np.array(train_set.iloc[:,1:].values)

    y_test = np.array(test_set.iloc[:,0].values)
    x_test = np.array(test_set.iloc[:,1:].values)
    print(f"Extracted Labels: {y_train}\n")
    print(f"Train data shape: {x_train.shape}")
    print(f"Train data shape: {x_test.shape}\n")

    fig, axes = plt.subplots(4, 5,figsize=(12,12))

    for i, j in enumerate(axes.flat):
        j.set_title(class_mapping.get(y_train[i+2]))
        j.imshow(x_train[i+2].reshape([28,28]))

    # normalize
    x_train = x_train.astype('float32') / 255
    x_test = x_test.astype('float32') / 255

    # Reshape the data to have a single color channel (since EMNIST is grayscale)
    # and match the input shape expected by the model
    x_train = x_train.reshape((-1, 28, 28, 1))
    x_test = x_test.reshape((-1, 28, 28, 1))
    print(f"Shape before transpose: {x_train.shape}")
    print(f"Mean before transpose: {np.mean(x_train[116])}\n")

    # transpose to reverse effect from converting png to csv (2D -> 1D)
    def transpose_data(x):
        return np.transpose(np.squeeze(x))[..., np.newaxis]
    
    x_train = np.array([transpose_data(x) for x in x_train])
    x_test = np.array([transpose_data(x) for x in x_test])
    print(f"Shape after transpose: {x_train.shape}")
    print(f"Mean after transpose: {np.mean(x_train[116])}\n")

    # show image after transpose (readable)
    fig2, axes2 = plt.subplots(4, 5,figsize=(12,12))

    for i, j in enumerate(axes2.flat):
        j.set_title(class_mapping.get(y_train[i+2]))
        j.imshow(x_train[i+2])

    # compare input preprocessing pipeline with training image data
    fig3, axes3 = plt.subplots(1, 2,figsize=(8,4))

    for i, j in enumerate(axes3.flat):
        j.set_title(class_mapping.get(y_train[116]))
        j.imshow(x_train[116])

    preprocess_image('sample_input.png', 'processed.png')

    # train_ds, test_ds = tf.keras.utils.image_dataset_from_directory(
    #     directory=data_dir,
    #     batch_size=32,
    #     color_mode='grayscale',
    #     image_size=(128, 128),
    #     seed=0,
    #     validation_split=0.3,
    #     subset='both'
    # )

    # print(train_ds.element_spec, test_ds.element_spec)


def get_class_mapping(mapping):
    class_mapping = {}
    ascii_code = mapping[1].values
    for i, code in enumerate(ascii_code):
        class_mapping[i] = chr(code)
    return class_mapping


def preprocess_image(input_image_path, output_image_path):
    # Step 1: Read the original 128x128 binary image
    image = cv2.imread(input_image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError("Image not found or the path is incorrect")
    
    # Invert the colors of the image
    image = 255 - image

    # Step 2: Apply Gaussian filter with σ = 1
    image = gaussian_filter(image, sigma=1)

    # Step 3: Extract the region around the character
    # Find non-zero pixels (characters)
    coords = cv2.findNonZero(image)
    x, y, w, h = cv2.boundingRect(coords)

    # Crop the image to the bounding box
    cropped_image = image[y:y+h, x:x+w]

    # Step 4: Center the character in a square image
    # Calculate the size of the new image (keeping the aspect ratio)
    max_side = max(w, h)
    square_image = np.zeros((max_side, max_side), dtype=np.uint8)

    # Compute the offset to center the character
    x_offset = (max_side - w) // 2
    y_offset = (max_side - h) // 2

    # Place the cropped image in the center of the square image
    square_image[y_offset:y_offset+h, x_offset:x_offset+w] = cropped_image

    # Step 5: Add a 2-pixel border
    padded_image = cv2.copyMakeBorder(square_image, 2, 2, 2, 2, cv2.BORDER_CONSTANT, value=0)

    # Step 6: Down-sample to 28x28 using bi-cubic interpolation
    downsampled_image = cv2.resize(padded_image, (28, 28), interpolation=cv2.INTER_CUBIC)

    # Step 7: Scale intensity values to [0, 255]
    # Convert image to have values in range [0, 255]
    final_image = cv2.normalize(downsampled_image, None, 0, 255, cv2.NORM_MINMAX)

    final_image = final_image/255.0

    print(f"Mean input image: {np.mean(final_image)}")
    plt.imshow(final_image[..., np.newaxis])

    # Save the final processed image
    cv2.imwrite(output_image_path, final_image[..., np.newaxis])


def rename_and_move_file():
    ALL_CLASS_NAMES = os.listdir(MERGE_DATA_DIR)

    for class_name in ALL_CLASS_NAMES:
        SUB_DIRS_PATH = os.path.join(MERGE_DATA_DIR, class_name)
        SUBDIRS = os.listdir(SUB_DIRS_PATH)

        for sub_dir in SUBDIRS:
            SRC_PARENT_PATH = os.path.join(MERGE_DATA_DIR, class_name, sub_dir)
            DST_PARENT_PATH = os.path.join(DATA_DIR, class_name)

            SRC_FILE_PATH = os.listdir(SRC_PARENT_PATH)

            for file in SRC_FILE_PATH:
                SRC_PATH = os.path.join(SRC_PARENT_PATH, file)

                NEW_FILE_NAME = class_name + "_" + file
                SRC_NEW_FILE_PATH = os.path.join(SRC_PARENT_PATH, NEW_FILE_NAME)
                DST_PATH = os.path.join(DST_PARENT_PATH, NEW_FILE_NAME)

                if os.path.isfile(DST_PATH) or os.path.isfile(SRC_NEW_FILE_PATH):
                    print("FILE EXISTS: {}".format(NEW_FILE_NAME))
                else:
                    os.rename(SRC_PATH, SRC_NEW_FILE_PATH)
                    shutil.move(SRC_NEW_FILE_PATH, DST_PATH)
                    print(DST_PATH)


if __name__ == "__main__":
    main()


# %%