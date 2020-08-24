import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split


def nn_basic(x, y):
    # print("Is there a GPU available: "),
    # print(tf.config.experimental.list_physical_devices("GPU"))

    print("------------------array type---------------")
    # x = np.expand_dims(np.asarray(x),-1).astype(np.float32)
    # y = np.expand_dims(np.asarray(y),-1).astype(np.float32)
    # x = np.asarray(x).astype(np.float32)
    # y = np.asarray(y).astype(np.float32)

    try:
        x.shape
        x = x.tolist()
    except:
        pass

    try:
        x = tf.convert_to_tensor(x.tolist())
        y = tf.convert_to_tensor(y.tolist())
        input_dim = np.shape(x[0])

    except:
        input_dim = len(x[0])

    x = np.array(x)
    y = np.array(y)

    # normalize
    # y = (y - np.min(y)) / (np.max(y) - np.min(y))
    # standarized
    std = np.std(y)
    mean = np.mean(y)
    y = (y - np.mean(y)) / np.std(y)

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    model = tf.keras.models.Sequential([
        tf.keras.Input(shape=(input_dim,)),
        tf.keras.layers.Dense(1024, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(1024),
        tf.keras.layers.Dense(1)
    ])
    #   tf.keras.layers.Dropout(0.2),
    #    tf.keras.layers.Dense(256),
    mse = tf.keras.losses.MeanSquaredError()
    model.compile(optimizer='adam', loss=mse, metrics=["MeanSquaredError"])

    model.fit(x_train, y_train, epochs=100)
    print(std)
    model.evaluate(x_test, y_test, verbose=2)
    print("MAE:" + str(std * np.mean(tf.keras.losses.MAE(y_test, model.predict(x_test)))))
