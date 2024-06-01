import cv2
import numpy as np
import pandas as pd
import tempfile
import os
import tensorflow as tf
import seaborn as sns
import tensorflow_model_optimization as tfmot

from keras import models, layers
from keras.callbacks import EarlyStopping
from keras.models import Sequential
from scipy.ndimage import gaussian_filter
from matplotlib import pyplot as plt
from IPython.display import display

def convert_model():
    converter = tf.lite.TFLiteConverter.from_saved_model('models/ocr_model_xs_v2_q')
    tflite_model = converter.convert()

    with open('ocr_model_q.tflite', 'wb') as f:
        f.write(tflite_model)

def main():
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

    q_train_images = train_images[:2500]
    q_train_labels = train_labels[:2500]

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

    # build model
    num_classes = len(class_mapping)

    for x, y in train_dataset.take(1):
        plt.figure()
        plt.imshow(x[0])
        plt.title(class_mapping.get(y[0].numpy()))
        print(f"Input shape: {x.shape[1:]}")
        plt.show()

    input_shape = x.shape[1:]

    EPOCHS = 3

    # early_stopping = EarlyStopping(
    #     monitor='val_accuracy',
    #     min_delta=0,
    #     restore_best_weights=True,
    #     patience=3,
    #     mode='max',
    #     verbose=0
    # )

    # confusion(test_dataset)

    # model = tf.keras.models.load_model('models/ocr_model_xs_v2')
    model = Sequential([
        layers.Input(shape=input_shape),
        layers.Conv2D(filters=32, kernel_size=(6, 6), activation='relu', padding='SAME'),
        layers.MaxPooling2D((3, 3)),
        layers.Conv2D(filters=64, kernel_size=(4, 4), activation='relu', padding='SAME'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(filters=128, kernel_size=(3, 3), activation='relu'),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(num_classes)
    ], name='OCR_Model')

    model.compile(
        optimizer='adam',
        loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy']
    )

    history = model.fit(train_dataset,
                        epochs=EPOCHS, 
                        validation_data=test_dataset, 
                        )

    quantize_model = tfmot.quantization.keras.quantize_model

    # q_aware stands for for quantization aware.
    q_aware_model = quantize_model(model)

    q_aware_model.summary()

    # model.compile(
    #     optimizer='adam',
    #     loss='categorical_crossentropy',
    #     metrics=['accuracy']
    # )

    q_aware_model.compile(
        optimizer='adam',
        loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy']
    )

    history = q_aware_model.fit(q_train_images,
                                q_train_labels,
                        epochs=EPOCHS, 
                        validation_split=0.1, 
                        )

    q_aware_model.evaluate(test_dataset, return_dict=True)

    # fig4, axes4 = plt.figure(2, 2, figsize=(12, 8))
    # for i, ax in enumerate(axes.flatten()):
    #     ax.plot(history.history)

    # convert_model(q_aware_model)
    q_aware_model.save('models/ocr_model_xs_v2_q')
    convert_model()


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
    # return final_image[np.newaxis, ..., np.newaxis]

    # depending on tf version, and other dependancies may run into error since data is float64.
    # if error occurs during inference, uncomment line below & comment away return above
    return (final_image[np.newaxis, ..., np.newaxis]).astype(np.float32)



if __name__ == '__main__':
    main()