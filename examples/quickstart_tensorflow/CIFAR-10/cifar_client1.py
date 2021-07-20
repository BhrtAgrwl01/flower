import os
import numpy as np
import flwr as fl
import tensorflow as tf
import sys

# Make TensorFlow log less verbose
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

if __name__ == "__main__":

    # Define Flower client
    class CifarClient(fl.client.NumPyClient):

        def __init__(self):
            self.alpha = float(alpha_val)
            print("Current value of alpha from the init method : ", self.alpha)
            
        def get_parameters(self):  # type: ignore
            return model.get_weights()

        def fit(self, parameters, config):  # type: ignore
            local_weights = model.get_weights()
            global_weights = parameters
            fedplus_weights = np.add((1 - self.alpha) * np.array(local_weights), self.alpha * np.array(global_weights))
            model.set_weights(fedplus_weights)
            model.fit(X_train, y_train, epochs=3, batch_size=128)
            return model.get_weights(), len(X_train), {}

        def evaluate(self, parameters, config):  # type: ignore
            local_weights = model.get_weights()
            global_weights = parameters
            fedplus_weights = ((1 - self.alpha) * np.array(local_weights)) + (self.alpha * np.array(global_weights))
            model.set_weights(fedplus_weights)
            loss, accuracy = model.evaluate(X_test, y_test)
            print("Accuracy on Test dataset : ", accuracy)
            return loss, len(X_test), {"accuracy": accuracy}

    # Start Flower client
    for alpha_val in sys.argv[1:]:

        print("Current alpha value", float(alpha_val))
        print("Total alpha values :", len(sys.argv)-1)

        data = np.load("cifar_party1.npz")

        X_train = data["X_train"]
        y_train = data["y_train"]
        X_test = data["X_test"]
        y_test = data["y_test"]

        model = tf.keras.models.Sequential()
        model.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu', input_shape=[32, 32, 3]))
        model.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))
        model.add(tf.keras.layers.Conv2D(filters=64, kernel_size=3, activation='relu'))
        model.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))
        model.add(tf.keras.layers.Flatten())
        model.add(tf.keras.layers.Dense(128, activation='relu'))
        model.add(tf.keras.layers.Dropout(0.2))
        model.add(tf.keras.layers.Dense(10, activation='softmax'))
        model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

        fl.client.start_numpy_client("localhost:8080", client=CifarClient())