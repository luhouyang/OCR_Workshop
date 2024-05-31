import tensorflow as tf

converter = tf.lite.TFLiteConverter.from_saved_model('models/ocr_model_xs_v2')
tflite_model = converter.convert()

with open('ocr_model.tflite', 'wb') as f:
    f.write(tflite_model)