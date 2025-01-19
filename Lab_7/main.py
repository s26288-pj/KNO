import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Dense, Flatten, Reshape, Conv2D, Input
from tensorflow.keras import Model, losses

directory = "D:/WORK/s26288/KNO/Lab_7/images"

# Załaduj dane treningowe i walidacyjne oddzielnie
x_train = tf.keras.preprocessing.image_dataset_from_directory(
    directory,
    label_mode=None,
    color_mode='rgb',
    batch_size=32,
    image_size=(128, 128),
    shuffle=True,
    seed=123,
    interpolation='bilinear',
    validation_split=0.2,
    subset='training'
)

x_test = tf.keras.preprocessing.image_dataset_from_directory(
    directory,
    label_mode=None,
    color_mode='rgb',
    batch_size=32,
    image_size=(128, 128),
    shuffle=True,
    seed=123,
    interpolation='bilinear',
    validation_split=0.2,
    subset='validation'
)

def encoder(latent_dim=2):
    layers_list = [
        tf.keras.layers.Conv2D(32, (3, 3), activation="relu", padding="same"),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(128, (3, 3), activation="relu", padding="same"),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(latent_dim, activation="relu")
    ]

    def call(inputs):
        x = inputs
        for layer in layers_list:
            x = layer(x)
        return x

    x = tf.keras.Input(shape=(128, 128, 3))
    return tf.keras.Model(inputs=x, outputs=call(x))

def decoder(latent_dim=2):
    layers_list = [
        tf.keras.layers.Dense(128 * 128 * 3, activation="relu"),
        tf.keras.layers.Reshape((16, 16, 128)),
        tf.keras.layers.Conv2DTranspose(128, (3, 3), activation="relu", padding="same"),
        tf.keras.layers.UpSampling2D((2, 2)),
        tf.keras.layers.Conv2DTranspose(64, (3, 3), activation="relu", padding="same"),
        tf.keras.layers.UpSampling2D((2, 2)),
        tf.keras.layers.Conv2DTranspose(32, (3, 3), activation="relu", padding="same"),
        tf.keras.layers.UpSampling2D((2, 2)),
        tf.keras.layers.Conv2D(3, (3, 3), activation="sigmoid", padding="same")
    ]

    def call(inputs):
        x = inputs
        for layer in layers_list:
            x = layer(x)
        return x

    x = tf.keras.Input(shape=(latent_dim,))
    return tf.keras.Model(inputs=x, outputs=call(x))


def autoencoder(input_shape=(128, 128, 3), latent_dim=2):
    enc = encoder(latent_dim)
    dec = decoder(latent_dim)
    inputs = Input(shape=input_shape)
    encoded = enc(inputs)
    decoded = dec(encoded)
    return Model(inputs, decoded, name="autoencoder"), enc, dec

shape = (128, 128, 3)
latent_dim = 2
autoencoder_model, encoder_model, decoder_model = autoencoder(shape, latent_dim)

autoencoder_model.compile(optimizer='adam', loss=losses.MeanSquaredError())

# Konwersja danych do numpy arrays
def convert_to_numpy(dataset):
    return np.concatenate([x for x in dataset], axis=0)

x_train_np = convert_to_numpy(x_train)
x_test_np = convert_to_numpy(x_test)

# Trenowanie modelu
autoencoder_model.fit(x_train_np, x_train_np,
                      epochs=100,
                      shuffle=True,
                      )

# Pobieranie zakodowanych i zdekodowanych obrazów
encoded_imgs = encoder_model.predict(x_train_np)
decoded_imgs = decoder_model.predict(encoded_imgs)

# Wyświetlanie oryginalnych i zrekonstruowanych obrazów
n = 10  # Docelowa liczba obrazów do wyświetlenia
num_images = x_train_np.shape[0]
n = min(n, num_images)  # Upewnij się, że n nie przekracza liczby dostępnych obrazów

plt.figure(figsize=(20, 4))
for i in range(n):
    # Wyświetl oryginalny obraz
    ax = plt.subplot(2, n, i + 1)
    plt.imshow(x_train_np[i].astype("uint8"))
    plt.title("Oryginalny")
    plt.axis("off")

    # Wyświetl zrekonstruowany obraz
    ax = plt.subplot(2, n, i + 1 + n)
    plt.imshow((decoded_imgs[i] * 255).astype("uint8"))
    plt.title("Zrekonstruowany")
    plt.axis("off")

plt.show()
