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
from keras.callbacks import EarlyStopping
from keras.models import Sequential


# GLOBAL VARIABLES
SEED = 42
tf.random.set_seed(SEED)
np.random.seed(SEED)

# DOWNLOAD_DATA_DIR = 'C:\\Users\\User\\Desktop\\Python\\OCR_Workshop\\data'
MERGE_DATA_DIR = 'D:\\training_data\\data\\by_merge'
DATA_DIR = 'D:\\training_data\\balanced_MNIST'


def main():

    train_dataset, test_dataset = get_train_test_dataset()

    model = get_model(train_dataset)

    model.summary()

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),
        loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy']
    )

    EPOCHS = 100

    early_stopping = EarlyStopping(
        monitor='val_accuracy',
        min_delta=0,
        restore_best_weights=True,
        patience=3,
        mode='max',
        verbose=0
    )

    history = model.fit(train_dataset,
                        epochs=EPOCHS, 
                        validation_data=test_dataset, 
                        callbacks=[
                            early_stopping, 
                            training_plot
                                   ]
                        )
    
    # model.save(filepath='models/ocr_model_small')
    # # model.save(filepath='models/ocr_model_variant')
    # model.save(filepath='models/ocr_model_large')
    # model.save(filepath='models/ocr_model_xs')
    # # model.save(filepath='models/ocr_model_scce')
    # model.save(filepath='models/ocr_model_xs_v2')
    # model.save(filepath='models/ocr_model_large_v2')
    # model.save(filepath="models/ocr_model_he_xxs")
    # model.save(filepath="models/ocr_model_he_large")
    # model.save(filepath="models/ocr_model_he_xs")

    # model.save(filepath='models/ocr_model_xxs')

    model.evaluate(test_dataset, return_dict=True)

    confusion(model, test_dataset)


def get_class_mapping(mapping):
    class_mapping = {}
    ascii_code = mapping[1].values
    for i, code in enumerate(ascii_code):
        class_mapping[i] = chr(code)
    return class_mapping


def preprocess_image(input_image_path, output_image_path):
    # read the original image
    image = cv2.imread(input_image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError("Image not found or the path is incorrect")
    
    # invert the colors of the image
    image = 255 - image

    # apply Gaussian filter with σ = 1
    image = gaussian_filter(image, sigma=1)

    # extract the region around the character
    # find non-zero pixels (characters)
    coords = cv2.findNonZero(image)
    x, y, w, h = cv2.boundingRect(coords)

    # crop the image to the bounding box
    cropped_image = image[y:y+h, x:x+w]

    # center the character in a square image
    # calculate the size of the new image (keeping the aspect ratio)
    max_side = max(w, h)
    square_image = np.zeros((max_side, max_side), dtype=np.uint8)

    # compute the offset to center the character
    x_offset = (max_side - w) // 2
    y_offset = (max_side - h) // 2

    # place the cropped image in the center of the square image
    square_image[y_offset:y_offset+h, x_offset:x_offset+w] = cropped_image

    # add a 2-pixel border
    padded_image = cv2.copyMakeBorder(square_image, 2, 2, 2, 2, cv2.BORDER_CONSTANT, value=0)

    # down-sample to 28x28 using bi-cubic interpolation
    downsampled_image = cv2.resize(padded_image, (28, 28), interpolation=cv2.INTER_CUBIC)

    # scale intensity values to [0, 255]
    # convert image to have values in range [0, 255]
    final_image = cv2.normalize(downsampled_image, None, 0, 255, cv2.NORM_MINMAX)

    # normalize values between [0, 1]
    final_image = final_image/255.0

    # verify that preprocessing is consistant with data
    print(f"Mean input image: {np.mean(final_image)}\n")

    # save the final processed image
    cv2.imwrite(output_image_path, final_image[..., np.newaxis])

    # add batch shape, and channel
    return final_image[np.newaxis, ..., np.newaxis]

    # depending on tf version, and other dependancies may run into error since data is float64.
    # if error occurs during inference, uncomment line below & comment away return above
    # return (final_image[np.newaxis, ..., np.newaxis]).astype(np.float32)


class OCRModel(tf.Module):
    def __init__(self, model):
        self.model = model
        self.class_mapping = {0: '0', 1: '1', 2: '2', 3: '3', 4: '4', 5: '5', 6: '6', 7: '7', 8: '8', 9: '9', 
                        10: 'A', 11: 'B', 12: 'C', 13: 'D', 14: 'E', 15: 'F', 16: 'G', 17: 'H', 18: 'I', 
                        19: 'J', 20: 'K', 21: 'L', 22: 'M', 23: 'N', 24: 'O', 25: 'P', 26: 'Q', 27: 'R', 
                        28: 'S', 29: 'T', 30: 'U', 31: 'V', 32: 'W', 33: 'X', 34: 'Y', 35: 'Z', 36: 'a', 
                        37: 'b', 38: 'd', 39: 'e', 40: 'f', 41: 'g', 42: 'h', 43: 'n', 44: 'q', 45: 'r', 
                        46: 't'}
        self.classes = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E', 'F', 'G',
                         'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 
                         'Y', 'Z', 'a', 'b', 'd', 'e', 'f', 'g', 'h', 'n', 'q', 'r', 't']

    @tf.function
    def predict(self, data):
        if isinstance(data, tf.Tensor):
            pred = self.model(data, training=False)
        elif isinstance(data, str):
            image = self.preprocess_image(data, "images/processed_image.png")
            pred = self.model(image, training=False)
        else:
            raise ValueError("Unsurported data type.\nPlease pass preprocessed image using preprocess_image function.\nOr pass path to image file")
        
        return pred
    
    def __call__(self, data):
        pred = self.predict(data)
        # pred_index = tf.argmax(pred, axis=-1)[0]
        # pred_index = tf.cast(pred_index, tf.int32)
        return self.class_mapping.get(np.argmax(pred.numpy()[0])), pred
    
    def preprocess_image(self, input_image_path, output_image_path):
        # read the original image
        image = cv2.imread(input_image_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise ValueError("Image not found or the path is incorrect")
        
        # invert the colors of the image
        image = 255 - image

        # apply Gaussian filter with σ = 1
        image = gaussian_filter(image, sigma=1)

        # extract the region around the character
        # find non-zero pixels (characters)
        coords = cv2.findNonZero(image)
        x, y, w, h = cv2.boundingRect(coords)

        # crop the image to the bounding box
        cropped_image = image[y:y+h, x:x+w]

        # center the character in a square image
        # calculate the size of the new image (keeping the aspect ratio)
        max_side = max(w, h)
        square_image = np.zeros((max_side, max_side), dtype=np.uint8)

        # compute the offset to center the character
        x_offset = (max_side - w) // 2
        y_offset = (max_side - h) // 2

        # place the cropped image in the center of the square image
        square_image[y_offset:y_offset+h, x_offset:x_offset+w] = cropped_image

        # add a 2-pixel border
        padded_image = cv2.copyMakeBorder(square_image, 2, 2, 2, 2, cv2.BORDER_CONSTANT, value=0)

        # down-sample to 28x28 using bi-cubic interpolation
        downsampled_image = cv2.resize(padded_image, (28, 28), interpolation=cv2.INTER_CUBIC)

        # scale intensity values to [0, 255]
        # convert image to have values in range [0, 255]
        final_image = cv2.normalize(downsampled_image, None, 0, 255, cv2.NORM_MINMAX)

        # normalize values between [0, 1]
        final_image = final_image/255.0

        # verify that preprocessing is consistant with data
        print(f"Mean input image: {np.mean(final_image)}\n")

        # save the final processed image
        cv2.imwrite(output_image_path, final_image[..., np.newaxis])

        # add batch shape, and channel
        return final_image[np.newaxis, ..., np.newaxis]
    
        # depending on tf version, and other dependancies may run into error since data is float64.
        # if error occurs during inference, uncomment line below & comment away return above
        # return (final_image[np.newaxis, ..., np.newaxis]).astype(np.float32)
    

# graph, plot
class TrainingPlot(tf.keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.loss = []
        self.acc = []
        self.val_loss = []
        self.val_acc = []
        self.logs = []

    def on_epoch_end(self, epoch, logs={}):
        # append latest log
        self.logs.append(logs)
        self.loss.append(logs.get('loss'))
        self.acc.append(logs.get('accuracy'))
        self.val_loss.append(logs.get('val_loss'))
        self.val_acc.append(logs.get('val_accuracy'))

        # at least 2 data points
        if len(self.loss) > 1 and len(self.acc) > 1:

            # plot graph
            clear_output(wait=True)
            N = np.arange(0, len(self.loss))

            plt.style.use("seaborn")

            plt.figure(figsize=(12, 6))
            plt.subplot(1, 2, 1)
            plt.plot(N, self.loss, self.val_loss)
            plt.legend(['loss', 'val_loss'])
            plt.ylim([0, max(plt.ylim())])
            plt.xlabel('Epoch')
            plt.ylabel('Loss [CrossEntrophy]')

            plt.subplot(1, 2, 2)
            plt.plot(N, 100*np.array(self.acc), 100*np.array(self.val_acc))
            plt.legend(['accuracy', 'val_accuracy'])
            plt.ylim([0, 100])
            plt.xlabel('Epoch')
            plt.ylabel('Accuracy [%]')

            plt.show()


def get_train_test_dataset():
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
    train_dataframe = pd.read_csv("./kaggle/emnist-balanced-train.csv")
    test_dataframe = pd.read_csv("./kaggle/emnist-balanced-test.csv")
    mapping = pd.read_csv("./kaggle/emnist-balanced-mapping.txt", sep = ' ', header = None)

    display(train_dataframe.head())
    train_dataframe.info()
    # last element is labels
    print(f"\nTrain set shape:  {train_dataframe.shape}")
    print(f"Test set shape:  {test_dataframe.shape}\n")

    # see class frequency
    labels = train_dataframe["45"].values

    plt.figure(figsize=(20,6))
    sns.countplot(x=labels)

    # get class mapping in dict
    mapping.head()
    class_mapping = get_class_mapping(mapping)
    print(f"{class_mapping}\n")

    # extract label and training data
    train_labels = np.array(train_dataframe.iloc[:,0].values)
    train_images = np.array(train_dataframe.iloc[:,1:].values)

    test_labels = np.array(test_dataframe.iloc[:,0].values)
    test_images = np.array(test_dataframe.iloc[:,1:].values)
    print(f"Extracted Labels: {train_labels}\n")
    print(f"Train data shape: {train_images.shape}")
    print(f"Train data shape: {test_images.shape}\n")

    del train_dataframe
    del test_dataframe

    fig, axes = plt.subplots(4, 5,figsize=(12,12))
    for i, j in enumerate(axes.flat):
        j.set_title(class_mapping.get(train_labels[i]))
        j.imshow(train_images[i].reshape([28,28]))

    # normalize
    train_images = train_images.astype('float32') / 255
    test_images = test_images.astype('float32') / 255

    # Reshape the data to have a single color channel (since EMNIST is grayscale)
    # and match the input shape expected by the model
    train_images = train_images.reshape((-1, 28, 28, 1))
    test_images = test_images.reshape((-1, 28, 28, 1))
    print(f"Shape before transpose: {train_images.shape}")
    print(f"Mean before transpose: {np.mean(train_images[116])}\n")

    # transpose to reverse effect from converting png to csv (2D -> 1D)
    def transpose_data(x):
        return np.transpose(np.squeeze(x))[..., np.newaxis]
    
    train_images = np.array([transpose_data(x) for x in train_images])
    test_images = np.array([transpose_data(x) for x in test_images])
    print(f"Shape after transpose: {train_images.shape}")
    print(f"Mean after transpose: {np.mean(train_images[116])}\n")

    # show image after transpose (readable)
    fig2, axes2 = plt.subplots(4, 5,figsize=(12,12))
    for i, j in enumerate(axes2.flat):
        j.set_title(class_mapping.get(train_labels[i]))
        j.imshow(train_images[i])

    # compare input preprocessing pipeline with training image data
    fig3, axes3 = plt.subplots(1, 2,figsize=(8,4))
    for i, j in enumerate(axes3.flat):
        j.set_title(class_mapping.get(train_labels[116]))
        j.imshow(train_images[116])

    # preprocess_image('sample_input.png', 'processed.png')
    # preprocess_image('random_scale_img.png', 'random_processed.png')
    pre_img = preprocess_image('images/random_scale_img_f.png', 'images/random_processed_f.png')
    plt.imshow(np.squeeze(pre_img, axis=0))

    # # encode labels into one-hot vectors
    # train_labels = tf.keras.utils.to_categorical(train_labels)
    # test_labels = tf.keras.utils.to_categorical(test_labels)
    # print(f"Example label: {train_labels[0]}\n")

    # put images and labels together in tf dataset
    train_dataset = tf.data.Dataset.from_tensor_slices((train_images, train_labels))
    test_dataset = tf.data.Dataset.from_tensor_slices((test_images, test_labels))
    print(f"TF Dataset shape: {train_dataset.element_spec}")

    del train_images
    del train_labels
    del test_images
    del test_labels

    train_dataset = train_dataset.cache().shuffle(10000).batch(16).prefetch(tf.data.AUTOTUNE)
    test_dataset = test_dataset.cache().batch(16)

    return train_dataset, test_dataset


def confusion(model, val_ds):
    classes = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E', 'F', 'G',
                         'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 
                         'Y', 'Z', 'a', 'b', 'd', 'e', 'f', 'g', 'h', 'n', 'q', 'r', 't']
    
    mapping = pd.read_csv("./kaggle/emnist-balanced-mapping.txt", sep = ' ', header = None)
    mapping = mapping[0].values

    # model = tf.keras.models.load_model('models/ocr_model_xs_v2')
    # confusion matrix
    y_pred = model.predict(val_ds)
    y_pred = tf.argmax(y_pred, axis=1)
    y_true = tf.concat(list(val_ds.map(lambda data, label: label)), axis=0)

    confusion_mtx = tf.math.confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(20, 20))
    plt.autoscale(tight=True)
    sns.heatmap(confusion_mtx,
                xticklabels=classes,
                yticklabels=classes,
                annot=True, fmt='g')
    plt.xlabel('Prediction')
    plt.ylabel('Label')

    plt.show()


def get_model(train_dataset):
    mapping = pd.read_csv("./kaggle/emnist-balanced-mapping.txt", sep = ' ', header = None)
    class_mapping = get_class_mapping(mapping)

    # build model
    num_classes = len(class_mapping)

    for x, y in train_dataset.take(1):
        plt.figure()
        plt.imshow(x[0])
        plt.title(class_mapping.get(y[0].numpy()))
        print(f"Input shape: {x.shape[1:]}")
        plt.show()

    input_shape = x.shape[1:]

    # # small
    # model = Sequential([
    #     layers.Input(shape=input_shape),
    #     layers.Conv2D(filters=32, kernel_size=(6, 6), activation='relu', padding='SAME'),
    #     layers.MaxPooling2D((3, 3)),
    #     layers.Conv2D(filters=64, kernel_size=(4, 4), activation='relu', padding='SAME'),
    #     layers.MaxPooling2D((2, 2)),
    #     layers.Conv2D(filters=256, kernel_size=(3, 3), activation='relu'),
    #     layers.Flatten(),
    #     layers.Dense(256, activation='relu'),
    #     layers.Dense(128, activation='relu'),
    #     layers.Dropout(0.3),
    #     layers.Dense(64, activation='relu'),
    #     layers.Dropout(0.5),
    #     layers.Dense(num_classes, activation='softmax')
    # ])

    # # small same para with large
    # model = Sequential([
    #     layers.Input(shape=input_shape),
    #     layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='SAME'),
    #     layers.MaxPooling2D((2, 2)),
    #     layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu'),
    #     layers.MaxPooling2D((2, 2)),
    #     layers.Conv2D(filters=256, kernel_size=(3, 3), activation='relu'),
    #     layers.Flatten(),
    #     layers.Dense(256, activation='relu'),
    #     layers.Dense(128, activation='relu'),
    #     layers.Dropout(0.3),
    #     layers.Dense(64, activation='relu'),
    #     layers.Dropout(0.5),
    #     layers.Dense(num_classes, activation='softmax')
    # ])

    # # # variant
    # # model = Sequential([
    # #     layers.Input(shape=input_shape),
    # #     layers.Conv2D(filters=32, kernel_size=(4, 4), activation='relu', padding='SAME'),
    # #     layers.MaxPooling2D((2, 2)),
    # #     layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu'),
    # #     layers.MaxPooling2D((2, 2)),
    # #     layers.Conv2D(filters=256, kernel_size=(3, 3), activation='relu'),
    # #     layers.Flatten(),
    # #     layers.Dense(256, activation='sigmoid'),
    # #     layers.Dense(128, activation='sigmoid'),
    # #     layers.Dense(64, activation='relu'),
    # #     layers.Dropout(0.5),
    # #     layers.Dense(num_classes, activation='softmax')
    # # ])

    # # ocr model large
    # model = Sequential([
    #     layers.Input(shape=input_shape),
    #     layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='SAME'),
    #     layers.MaxPooling2D((2, 2)),
    #     layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu'),
    #     layers.MaxPooling2D((2, 2)),
    #     layers.Conv2D(filters=256, kernel_size=(3, 3), activation='relu'),
    #     layers.Flatten(),
    #     layers.Dense(256, activation='relu'),
    #     layers.Dropout(0.25),
    #     layers.Dense(128, activation='relu'),
    #     layers.Dropout(0.25),
    #     layers.Dense(64, activation='relu'),
    #     layers.Dropout(0.5),
    #     layers.Dense(num_classes)
    # ])

    # # ocr model large v2
    # model = Sequential([
    #     layers.Input(shape=input_shape),
    #     layers.Conv2D(filters=32, kernel_size=(6, 6), activation='relu', padding='SAME'),
    #     layers.MaxPooling2D((3, 3)),
    #     layers.Conv2D(filters=64, kernel_size=(4, 4), activation='relu', padding='SAME'),
    #     layers.MaxPooling2D((2, 2)),
    #     layers.Conv2D(filters=256, kernel_size=(3, 3), activation='relu'),
    #     layers.Flatten(),
    #     layers.Dense(256, activation='relu'),
    #     layers.Dropout(0.25),
    #     layers.Dense(128, activation='relu'),
    #     layers.Dropout(0.25),
    #     layers.Dense(64, activation='relu'),
    #     layers.Dropout(0.5),
    #     layers.Dense(num_classes)
    # ])

    # # xs v2, extra small/scce
    # model = Sequential([
    #     layers.Input(shape=input_shape),
    #     layers.Conv2D(filters=32, kernel_size=(6, 6), activation='relu', padding='SAME'),
    #     layers.MaxPooling2D((3, 3)),
    #     layers.Conv2D(filters=64, kernel_size=(4, 4), activation='relu', padding='SAME'),
    #     layers.MaxPooling2D((2, 2)),
    #     layers.Conv2D(filters=128, kernel_size=(3, 3), activation='relu'),
    #     layers.Flatten(),
    #     layers.Dense(64, activation='relu'),
    #     layers.Dense(64, activation='relu'),
    #     layers.Dropout(0.5),
    #     layers.Dense(num_classes)
    # ], name='OCR_Model')

    # # xs same para with large
    # model = Sequential([
    #     layers.Input(shape=input_shape),
    #     layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same'),
    #     layers.MaxPooling2D((2, 2)),
    #     layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu'),
    #     layers.MaxPooling2D((2, 2)),
    #     layers.Conv2D(filters=128, kernel_size=(3, 3), activation='relu'),
    #     layers.Flatten(),
    #     layers.Dense(64, activation='relu'),
    #     layers.Dense(64, activation='relu'),
    #     layers.Dropout(0.5),
    #     layers.Dense(num_classes)
    # ])
    
    # xxs
    model = Sequential([
        layers.Input(shape=input_shape),
        layers.Conv2D(filters=16, kernel_size=(3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(filters=16, kernel_size=(3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(filters=16, kernel_size=(3, 3), activation='relu'),
        layers.Flatten(),
        layers.Dense(num_classes, activation='relu'), 
        layers.Dropout(0.25),
        layers.Dense(num_classes)
    ], name='OCR_Model')

    # # xs, he_normal
    # model = Sequential([
    #     layers.Input(shape=input_shape),
    #     layers.Conv2D(filters=32, kernel_size=(6, 6), activation='relu', padding='SAME', 
    #                   kernel_initializer='he_normal'),
    #     layers.MaxPooling2D((3, 3)),
    #     layers.Conv2D(filters=64, kernel_size=(4, 4), activation='relu', padding='SAME', 
    #                   kernel_initializer='he_normal'),
    #     layers.MaxPooling2D((2, 2)),
    #     layers.Conv2D(filters=128, kernel_size=(3, 3), activation='relu', 
    #                   kernel_initializer='he_normal'),
    #     layers.Flatten(),
    #     layers.Dense(64, activation='relu', kernel_initializer='he_normal'),
    #     layers.Dense(64, activation='relu', kernel_initializer='he_normal'),
    #     layers.Dropout(0.5),
    #     layers.Dense(num_classes)
    # ], name='OCR_Model')

    # # he_normal xxs
    # model = Sequential([
    #     layers.Input(shape=input_shape),
    #     layers.Conv2D(filters=16, kernel_size=(3, 3), activation='relu',
    #                   kernel_initializer='he_normal'),
    #     layers.MaxPooling2D((2, 2)),
    #     layers.Conv2D(filters=16, kernel_size=(3, 3), activation='relu',
    #                   kernel_initializer='he_normal'),
    #     layers.MaxPooling2D((2, 2)),
    #     layers.Conv2D(filters=16, kernel_size=(3, 3), activation='relu', 
    #                   kernel_initializer='he_normal'),
    #     layers.Flatten(),
    #     layers.Dense(num_classes, activation='relu', 
    #                  kernel_initializer='he_normal'), 
    #     layers.Dropout(0.25),
    #     layers.Dense(num_classes)
    # ], name='OCR_Model')

    # # he large
    # model = Sequential([
    #     layers.Input(shape=input_shape),
    #     layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu', 
    #                   kernel_initializer='he_normal'),
    #     layers.MaxPooling2D((2, 2)),
    #     layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu', 
    #                   kernel_initializer='he_normal'),
    #     layers.MaxPooling2D((2, 2)),
    #     layers.Conv2D(filters=128, kernel_size=(3, 3), activation='relu', 
    #                   kernel_initializer='he_normal'),
    #     layers.Flatten(),
    #     layers.Dense(units=256, activation='relu', 
    #                  kernel_initializer='he_normal'),
    #     layers.Dense(units=128, activation='relu', 
    #                  kernel_initializer='he_normal'),
    #     layers.Dropout(0.25),
    #     layers.Dense(units=64, activation='relu', 
    #                  kernel_initializer='he_normal'),
    #     layers.Dropout(0.25),
    #     layers.Dense(units=num_classes),
    # ])

    return model

def evaluate_and_plot_models():
    pass

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
    training_plot = TrainingPlot()
    main()


# %%