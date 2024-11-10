**Pierwszy model**

```python
model_1 = keras.Sequential([
    Input(shape=(input_shape,)),                # Warstwa wejściowa
    Dense(13 * 2, activation="relu"),           # Pierwsza warstwa gęsta
    Dense(13, activation='relu'),         # Druga warstwa gęsta
    Dense(3, activation='softmax')        # Warstwa wyjściowa z softmax dla 3 klas
])
```

Jest to model sekwencyjny 