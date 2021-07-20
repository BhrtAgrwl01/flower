import os
import numpy as np
import flwr as fl
import tensorflow as tf
import sys
import matplotlib.pyplot as plt
from tensorflow.python.keras.engine.base_layer_utils import mark_as_return
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

# Make TensorFlow log less verbose
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

if __name__ == "__main__":

    accuracy_vals=[]
    test_values = []
    test_predictions = []

    # Define Flower client
    class EMNISTClient(fl.client.NumPyClient):

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
            model.fit(X_train, y_train, epochs=3, batch_size=64)
            return model.get_weights(), len(X_train), {}

        def evaluate(self, parameters, config):  # type: ignore
            local_weights = model.get_weights()
            global_weights = parameters
            fedplus_weights = ((1 - self.alpha) * np.array(local_weights)) + (self.alpha * np.array(global_weights))
            model.set_weights(fedplus_weights)
            loss, accuracy = model.evaluate(X_test, y_test)
            accuracy_vals.append(accuracy)
            print("End accuracy : ", accuracy_vals[-1])
            y_pred = np.argmax(model.predict(X_test), axis=-1)
            print("Confusion Matrix : \n", confusion_matrix(y_test, y_pred))
            print("Classification Report : \n", classification_report(y_test, y_pred))
            return loss, len(X_test), {"accuracy": accuracy}

    # Start Flower client
    for alpha_val in sys.argv[1:]:

        print("Current alpha value", float(alpha_val))
        print("Total alpha values :", len(sys.argv)-1)

        data1 = np.load('first_splitted_dataset.npz', allow_pickle=True)
        X_train = data1["X_train"]
        y_train = data1["y_train"]
        X_test = data1["X_test"]
        y_test = data1["y_test"]

        X_train = np.array([tf.convert_to_tensor(arr.reshape(28, 28, 1)) for arr in X_train])
        y_train = np.array([tf.convert_to_tensor(arr) for arr in y_train])
        X_test = np.array([tf.convert_to_tensor(arr.reshape(28, 28, 1)) for arr in X_test])
        y_test = np.array([tf.convert_to_tensor(arr) for arr in y_test])


        model = tf.keras.models.Sequential()
        model.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu', input_shape=[28, 28, 1]))
        model.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))
        model.add(tf.keras.layers.Conv2D(filters=64, kernel_size=3, activation='relu'))
        model.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))
        model.add(tf.keras.layers.Flatten())
        model.add(tf.keras.layers.Dense(128, activation='relu'))
        model.add(tf.keras.layers.Dropout(0.2))
        model.add(tf.keras.layers.Dense(62, activation='softmax'))
        model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

        fl.client.start_numpy_client("localhost:8080", client=EMNISTClient())

    final_accuracies_client1 = [accuracy_vals[11*(n+1)-1] for n in range(len(sys.argv)-1)]
    print(final_accuracies_client1)
    plt.plot(sys.argv[1:], final_accuracies_client1, "b*-")
    plt.xlabel("Alpha")
    plt.ylabel("Accuracy")
    plt.yticks(np.arange(max(min(final_accuracies_client1)-0.1, 0), min(max(final_accuracies_client1)+0.1, 1), 0.05))
    plt.title("Client 1 accuracies")
    plt.show(block=True)