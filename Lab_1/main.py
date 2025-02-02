import os.path
import tensorflow as tf
import numpy as np
from PIL import Image, ImageOps

mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

def load_image(plik):
    img = Image.open(plik).convert('L')
    img = img.resize((28, 28))
    img = ImageOps.invert(img)
    img_array = np.array(img)
    img_array = img_array / 255.0
    img_array = img_array.reshape((1, 28, 28))
    return img_array

def main():
    if os.path.exists('model.keras'):
        model = tf.keras.models.load_model('model.keras')
    else:
        model = tf.keras.models.Sequential([
            tf.keras.layers.Flatten(input_shape=(28, 28)),
            tf.keras.layers.Dense(1024, activation='relu'),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(10, activation='softmax')
        ])

        model.compile(optimizer='adam',
                      loss='sparse_categorical_crossentropy',
                      metrics=['accuracy'])
        model.fit(x_train, y_train, epochs=8)

    e = model.evaluate(x_test, y_test)
    print(e)
    model.save('model.keras')

    plik = '9.png'
    testowany_plik = load_image(plik)

    prediction = model.predict(testowany_plik)
    predicted_class = np.argmax(prediction)

    print(f'Rozpoznana cyfra: {predicted_class}')

if __name__ == '__main__':
    main()
