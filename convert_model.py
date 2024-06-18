import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_model_optimization as tfmot

from IPython.display import display

def convert_model(quantized_path, tflite_path):
    converter = tf.lite.TFLiteConverter.from_saved_model(quantized_path)
    tflite_model = converter.convert()

    with open(tflite_path, 'wb') as f:
        f.write(tflite_model)

def main():

    # get train & test dataset
    train_dataset, test_dataset = get_train_test_dataset()

    # load pretrained model
    model = tf.keras.models.load_model('models/ocr_model_xxs')

    # create quantization object
    quantize_model = tfmot.quantization.keras.quantize_model

    # change model to quantized model
    q_aware_model = quantize_model(model)

    q_aware_model.summary()

    q_aware_model.compile(
        optimizer='adam',
        loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy']
    )

    # fine tune quantized model
    EPOCHS = 3
    history = q_aware_model.fit(train_dataset,
                        epochs=EPOCHS, 
                        validation_data=test_dataset
                        )

    q_aware_model.evaluate(test_dataset, return_dict=True)

    # save quantized model
    QUANTIZED_PATH = 'models/quantized_model_xxs'
    TFLITE_PATH = 'quantized_model_xxs.tflite'
    q_aware_model.save(QUANTIZED_PATH)
    
    # convert model to tflite
    convert_model(QUANTIZED_PATH, TFLITE_PATH)


def get_class_mapping(mapping):
    class_mapping = {}
    ascii_code = mapping[1].values
    for i, code in enumerate(ascii_code):
        class_mapping[i] = chr(code)
    return class_mapping


def get_train_test_dataset():
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

    q_train_images = train_images[:2500]
    q_train_labels = train_labels[:2500]
    test_images = test_images[:2500]
    test_labels = test_labels[:2500]

    # put images and labels together in tf dataset
    train_dataset = tf.data.Dataset.from_tensor_slices((q_train_images, q_train_labels))
    test_dataset = tf.data.Dataset.from_tensor_slices((test_images, test_labels))
    print(f"TF Dataset shape: {train_dataset.element_spec}")

    del train_images
    del train_labels
    del test_images
    del test_labels
    del q_train_images
    del q_train_labels

    train_dataset = train_dataset.cache().shuffle(10000).batch(16).prefetch(tf.data.AUTOTUNE)
    test_dataset = test_dataset.cache().batch(16)

    return train_dataset, test_dataset


if __name__ == '__main__':
    main()