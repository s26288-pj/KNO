import datetime

import keras
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
from keras.src.layers import LayerNormalization, Dense, Input
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


def create_model(input_shape: int, layer_units: int, layers_count: int = 1, learning_rate: float = 0.01):
  model = keras.Sequential()
  model.add(keras.layers.Dense(10, input_shape=(input_shape,), activation='relu'))
  model.add(LayerNormalization())
  for _ in range(layers_count):
    model.add(keras.layers.Dense(units=layer_units, activation='relu'))
  model.add(keras.layers.Dense(3, activation='softmax'))

  model.compile(optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
                loss="categorical_crossentropy",
                metrics=['accuracy'])

  return model


parameters = [
    [1, 10, 1, 0.01],
    [2, 10, 2, 0.002],
    [3, 13, 1, 0.01],
    [4, 13, 2, 0.002],
    [5, 15, 1, 0.01],
    [6, 15, 2, 0.002],
]


def plot_all_learning_curves(history_dict):
    plt.figure(figsize=(10, 6))

    for model_name, history in history_dict.items():
        train_accuracy = history.history['accuracy']
        val_accuracy = history.history['val_accuracy']
        epochs = range(1, len(train_accuracy) + 1)

        # Plotting training and validation accuracy for each model
        plt.plot(epochs, train_accuracy, label=f'{model_name} (train)', linestyle='--')
        plt.plot(epochs, val_accuracy, label=f'{model_name} (val)', linestyle='-')

    # Customize the plot
    plt.title("Learning Curves for All Models")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    train_ds, val_ds, test_ds = create_datasets("wine.data")
    results = []
    history_dict = {}

    input_shape = 0
    for x, y in train_ds.take(1):
        input_shape = x.shape[1]

    models = {}
    for index, layer_units, layers_count, learnig_rate in parameters:
        model_name = f"model_{index}"
        models[model_name] = create_model(
            input_shape=input_shape,
            layer_units=layer_units,
            layers_count=layers_count,
            learning_rate=learnig_rate,
        )

        log_dir = "logs/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + f"/{model_name}"
        tensorboard_callback = keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

        model = models[model_name]
        history = model.fit(train_ds, epochs=50, batch_size=32, validation_data=val_ds, callbacks=[tensorboard_callback])

        history_dict[model_name] = history

        results.append({
            "model_name": model_name,
            "learning_rate": learnig_rate,
            "layer_units": layer_units,
            "layers_count": layers_count,
            "train_accuracy": history.history['accuracy'][-1],
            "val_accuracy": history.history['val_accuracy'][-1]
        })

    results_df = pd.DataFrame(results)
    print("Results for different models:")
    print(results_df)
    plot_all_learning_curves(history_dict)

    best_model_info = results_df.loc[results_df['val_accuracy'].idxmax()]
    best_model = models[best_model_info['model_name']]
    test_loss, test_accuracy = best_model.evaluate(test_ds)

    baseline_model = keras.Sequential([
        Input(shape=(input_shape,)),
        LayerNormalization(),
        Dense(13 * 2, activation="relu"),
        Dense(13, activation='relu'),
        Dense(3, activation='softmax'),
    ])
    baseline_model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=['accuracy'],
    )
    baseline_model.fit(train_ds, epochs=50, batch_size=32, validation_data=val_ds)
    baseline_test_loss, baseline_test_accuracy = baseline_model.evaluate(test_ds)

    print(f"\nBaseline test accuracy: {baseline_test_accuracy:.4f}")
    print(f"\nBest model test accuracy: {test_accuracy:.4f}")
    print(best_model_info)