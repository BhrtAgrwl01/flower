#importing modules
import os
import numpy as np
import flwr as fl
import tensorflow as tf
import json
import matplotlib.pyplot as plt
from tensorflow.python.keras.engine.base_layer_utils import mark_as_return
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from tensorflow.keras.applications import ResNet50


physical_devices = tf.config.list_physical_devices('GPU')
try:
  tf.config.set_logical_device_configuration(
    physical_devices[0],
    [tf.config.LogicalDeviceConfiguration(memory_limit=3000),
     tf.config.LogicalDeviceConfiguration(memory_limit=3000),
     tf.config.LogicalDeviceConfiguration(memory_limit=3000)])


  logical_devices = tf.config.list_logical_devices('GPU')
  assert len(logical_devices) == len(physical_devices) + 1


  tf.config.set_logical_device_configuration(
    physical_devices[0],
    [tf.config.LogicalDeviceConfiguration(memory_limit=10),
     tf.config.LogicalDeviceConfiguration(memory_limit=10),
     tf.config.LogicalDeviceConfiguration(memory_limit=10)])
except:
  # Invalid device or cannot modify logical devices once initialized.
  pass


# loading the config file for the parameters - alpha, batch_size and local_epochs
with open("kvasir_config.json", "r") as f:
  params = json.load(f)


# Make TensorFlow log less verbose
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

if __name__ == "__main__":

    # define the initial model
    model = tf.keras.models.Sequential()
    model.add(ResNet50(include_top = False, pooling = 'avg', weights = 'imagenet'))
    model.add(tf.keras.layers.Dropout(0.2))
    model.add(tf.keras.layers.Dense(256, activation='relu'))
    model.add(tf.keras.layers.Dropout(0.2))
    model.add(tf.keras.layers.Dense(128, activation='relu'))
    model.add(tf.keras.layers.Dropout(0.2))
    model.add(tf.keras.layers.Dense(64, activation='relu'))
    model.add(tf.keras.layers.Dropout(0.2))
    model.add(tf.keras.layers.Dense(8, activation = 'softmax'))
    model.layers[0].trainable = False
    adam = tf.keras.optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-07, amsgrad=False, name='Adam')
    model.compile(optimizer = adam, loss = 'categorical_crossentropy', metrics=['accuracy'])

    #loading datasets
    data1 = np.load('kvasir_party_576_personal.npz', allow_pickle=True)
    X_train1 = data1["x_train"]
    y_train1 = data1["y_train"]
    X_test1 = data1["x_test"]
    y_test1 = data1["y_test"]
    data_agg = np.load('kvasir_agg.npz', allow_pickle=True)
    X_test_agg= data_agg["x_test"]
    y_test_agg= data_agg["y_test"]

    # Define Flower client
    class KvasirClient(fl.client.NumPyClient):

        def __init__(self):
            self.alpha = params["alpha"]
            print("Current value of alpha from the init method : ", self.alpha)
            self.epochs = params["local_epochs"]
            print("Epochs from init method : ", self.epochs)
            self.batch_size = params["batch_size"]
            print("Batch size from init method : ", self.batch_size)
            
        def get_parameters(self):  # type: ignore
            return model.get_weights()

        def fit(self, parameters, config):  # type: ignore

            # implementation of fed+ algorithm
            # getting local weights
            local_weights = model.get_weights()
            #getting global weights
            global_weights = parameters
            #creating federated weighted weights as per fedavg+ algorithm
            fedplus_weights = np.add((1 - self.alpha) * np.array(local_weights), self.alpha * np.array(global_weights))
            model.set_weights(fedplus_weights)

            model.fit(X_train1, y_train1, epochs=self.epochs, batch_size=self.batch_size)
            return model.get_weights(), len(X_train1), {}

        def evaluate(self, parameters, config):  # type: ignore

            # implementation of fed+ algorithm
            local_weights = model.get_weights()
            global_weights = parameters
            fedplus_weights = ((1 - self.alpha) * np.array(local_weights)) + (self.alpha * np.array(global_weights))
            model.set_weights(fedplus_weights)


            loss_agg, accuracy_agg = model.evaluate(X_test_agg, y_test_agg)
            print("Accuracy on common test dataset : ", accuracy_agg)
            loss1, accuracy1 = model.evaluate(X_test1, y_test1)
            print("Accuracy on personal test dataset : ", accuracy1)
            y_true_agg = np.argmax(y_test_agg, axis=1)
            y_pred_agg = np.argmax(model.predict(X_test_agg), axis=1)
            print("Confusion Matrix for common test dataset: \n", confusion_matrix(y_true_agg, y_pred_agg))
            print("Classification Report for common test dataset: \n", classification_report(y_true_agg, y_pred_agg))
            y_true1 = np.argmax(y_test1, axis=1)
            y_pred1 = np.argmax(model.predict(X_test1, axis=1))
            print("Confusion Matrix for personal test dataset: \n", confusion_matrix(y_true1, y_pred1))
            print("Classification Report for personal test dataset: \n", classification_report(y_true1, y_pred1))
            return loss_agg, len(X_test_agg), {"accuracy": accuracy_agg}

    # Start Flower client
    fl.client.start_numpy_client("localhost:8080", client=KvasirClient())

        