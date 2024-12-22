import keras
import matplotlib.pyplot as plt
import pandas as pd
import keras_tuner as kt
import tensorflow as tf
from tensorflow.keras import layers, Model
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle


def dataframe_to_dataset(dataframe):
    dataframe = dataframe.copy()
    labels = dataframe.pop("class")
    ds = tf.data.Dataset.from_tensor_slices((dataframe.values, keras.utils.to_categorical(labels.values - 1, num_classes=3)))
    ds = ds.shuffle(buffer_size=len(dataframe)).batch(32)
    return ds


def create_datasets(file: str):
    dataset = pd.read_csv(file)
    dataset = shuffle(dataset)
    dataset.columns = ["class", "Alcohol", "Malicacid", "Ash", "Alcalinity_of_ash", "Magnesium", "Total_phenols",
                       "Flavanoids", "Nonflavanoid_phenols", "Proanthocyanins", "Color_intensity", "Hue",
                       "0D280_0D315_of_diluted_wines", "Proline"]

    test_dataframe = dataset.sample(frac=0.1, random_state=1337)
    train_dataframe = dataset.drop(test_dataframe.index)
    train_dataframe, val_dataframe = train_test_split(train_dataframe, test_size=0.15, random_state=1337)

    train_ds = dataframe_to_dataset(train_dataframe)
    test_ds = dataframe_to_dataset(test_dataframe)
    val_ds = dataframe_to_dataset(val_dataframe)

    return train_ds, val_ds, test_ds


class CustomModel(Model):
    def __init__(self, num_hidden_units, dropout_rate):
        super(CustomModel, self).__init__()
        self.dense1 = layers.Dense(num_hidden_units, activation='relu')
        self.dropout = layers.Dropout(dropout_rate)
        self.output_layer = layers.Dense(3, activation='softmax')

    def call(self, inputs):
        x = self.dense1(inputs)
        x = self.dropout(x)
        return self.output_layer(x)


def build_model(hp):
    num_hidden_units = hp.Int('num_hidden_units', min_value=32, max_value=512, step=32)
    dropout_rate = hp.Float('dropout_rate', min_value=0.1, max_value=0.5, step=0.1)
    learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])

    model = CustomModel(num_hidden_units, dropout_rate)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model


if __name__ == "__main__":
    train_ds, val_ds, test_ds = create_datasets("wine.data")

    input_shape = 0
    for x, y in train_ds.take(1):
        input_shape = x.shape[1]

    tuner = kt.Hyperband(
        build_model,
        objective='val_accuracy',
        max_epochs=10,
        factor=3,
        directory='my_tuner',
        project_name='wine_model_tuning'
    )

    stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)

    tuner.search(train_ds, epochs=10, validation_data=val_ds, callbacks=[stop_early])

    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
    print(f"Optimal hyperparameters: {best_hps.values}")

    final_model = tuner.hypermodel.build(best_hps)
    history = final_model.fit(train_ds, epochs=10, validation_data=val_ds,
                              callbacks=[stop_early])

    loss, accuracy = final_model.evaluate(test_ds)
    print(f"Test accuracy: {accuracy}")