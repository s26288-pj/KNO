import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.layers import Dense, Flatten, MaxPooling2D, Conv2D


CLASSES = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

(ds_train, ds_test), ds_info = tfds.load(
    'mnist',
    split=['train', 'test'],
    shuffle_files=True,
    as_supervised=True,
    with_info=True,
)

def normalize_img(image, label):
  """Normalizes images: `uint8` -> `float32`."""
  return tf.cast(image, tf.float32) / 255., label

def preprocess_image(image_path):
    """
    Preprocess a single image for prediction.

    :param image_path: The path to the image file.
    :return: The preprocessed image.
    """
    img = load_img(image_path, target_size=(28, 28))
    gray_image = img.convert('L')
    img_array = img_to_array(gray_image) / 255.0
    return np.expand_dims(img_array, axis=0)

ds_train = ds_train.map(normalize_img, num_parallel_calls=tf.data.AUTOTUNE)
ds_train = ds_train.cache()
ds_train = ds_train.shuffle(ds_info.splits['train'].num_examples)
ds_train = ds_train.batch(128)
ds_train = ds_train.prefetch(tf.data.AUTOTUNE)

ds_test = ds_test.map(normalize_img, num_parallel_calls=tf.data.AUTOTUNE)
ds_test = ds_test.batch(128)
ds_test = ds_test.cache()
ds_test = ds_test.prefetch(tf.data.AUTOTUNE)

model_k = tf.keras.models.Sequential([
        Conv2D(16, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
])

model_k.compile(
    optimizer=tf.keras.optimizers.Adam(0.001),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],
)

model_k.fit(
    ds_train,
    epochs=6,
    validation_data=ds_test,
)

print("Loading and preprocessing image...")
image = preprocess_image('one.png')

print("Predicting image using convolutional model...")
number_prediction = model_k.predict(image)
number_class = CLASSES[np.argmax(number_prediction)]
print(f"Image was classified as: {number_class}")

model_l = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(28, 28)),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dense(10)
])

model_l.compile(
    optimizer=tf.keras.optimizers.Adam(0.001),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],
)

model_l.fit(
    ds_train,
    epochs=6,
    validation_data=ds_test,
)

print("Predicting image using layer model...")
number_prediction = model_l.predict(image)
number_class = CLASSES[np.argmax(number_prediction)]
print(f"Image was classified as: {number_class}")