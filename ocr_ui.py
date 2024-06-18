import os
import tensorflow as tf
import numpy as np
import turtle
import matplotlib.pyplot as plt

from model import OCRModel
from PIL import Image


t=turtle


def main():
    # model = tf.keras.models.load_model('models/ocr_model_xs_v2')
    # model = tf.keras.models.load_model('models/ocr_model_large')
    # model = tf.keras.models.load_model('models/ocr_model_he_l2_large')
    model = tf.keras.models.load_model('models/ocr_model_xxs_v2')
    print(np.sum([np.prod(v.shape) for v in model.trainable_variables]))
    ocr_model = OCRModel(model)

    t.title("PyBoard | Shift + s to run prediction")
    t.setup(1000, 1000)
    t.screensize(1000, 1000)
    t.shape("circle")
    t.shapesize(1.5)
    t.pu()
    t.color("black") # dot
    t.bgcolor("white")  # background
    t.pencolor("black") # line
    t.pensize(30)
    t.speed(0)

        # move pen to mouse
    def skc(x,y):
        t.pu()
        t.goto(x,y)
        def sketch(x,y):
            t.pd()
            t.goto(x,y)
        t.ondrag(sketch)
    t.onscreenclick(skc)

    # eraser
    def erase():
        t.speed(0)
        t.pencolor("white")
        t.pensize(40)
        t.shape("square")
        t.shapesize(2)

    # clear canvas
    def clear():
        t.clear()

    # pen
    def backtopen():
        t.speed(0)
        t.color("black")
        t.pensize(30)
        t.shape("circle")
        t.shapesize(1.5)

    # undo last stroke
    def undo():
        t.undo()

    # run prediction
    def predict():
        # get image and convert to png
        DATA_DIR = 'pyboard.png'

        canvas = t.getscreen().getcanvas()
        canvas.postscript(file="foo.ps")
        psimage = Image.open("foo.ps")
        psimage.save(DATA_DIR)
        psimage.close()
        os.remove("foo.ps")

        image = ocr_model.preprocess_image(DATA_DIR, 'preprocessed_img.png')

        # print(repr(image))
        result, pred = ocr_model(image)
        print(f"Prediction result: {result}")

        plt.figure(figsize=(20, 8))
        plt.subplot(1, 2, 1)
        plt.bar(ocr_model.classes, tf.nn.softmax(pred[0]))
        plt.title(result)

        plt.subplot(1, 2, 2)
        np_data = np.asarray(np.squeeze(image, axis=0))
        plt.imshow(np_data)
        plt.title(result)

        plt.show()
    
    # start
    t.onkeypress(undo,"u")
    t.onkey(backtopen,"p")
    t.onkey(clear,"c")
    t.onkey(erase,"e")
    t.onkey(predict, "S")
    t.listen()
    t.mainloop()

    # image = ocr_model.preprocess_image('images/sample_input.png', 'images/processed.png')
    # result = ocr_model(image)
    # print(f"Prediction result: {result}")

    # result = ocr_model('images/random_scale_img_f.png')
    # print(f"Prediction result: {result}")


if __name__ == '__main__':
    main()