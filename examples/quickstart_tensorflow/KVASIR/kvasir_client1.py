import os
import numpy as np
import flwr as fl
import tensorflow as tf
import sys
import matplotlib.pyplot as plt
from tensorflow.python.keras.engine.base_layer_utils import mark_as_return
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
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


# Make TensorFlow log less verbose
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

if __name__ == "__main__":

    # Define Flower client
    class KvasirClient(fl.client.NumPyClient):

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
            model.fit(X_train1, y_train1, epochs=3, batch_size=128)
            return model.get_weights(), len(X_train1), {}

        def evaluate(self, parameters, config):  # type: ignore
            local_weights = model.get_weights()
            global_weights = parameters
            fedplus_weights = ((1 - self.alpha) * np.array(local_weights)) + (self.alpha * np.array(global_weights))
            model.set_weights(fedplus_weights)
            loss, accuracy = model.evaluate(X_test, y_test)
            y_true = np.argmax(y_test, axis=1)
            y_pred = np.argmax(model.predict(X_test), axis=1)
            print("Accuracy from evaluate method : ", accuracy)
            print("Confusion Matrix : \n", confusion_matrix(y_true, y_pred))
            print("Classification Report : \n", classification_report(y_true, y_pred))
            return loss, len(X_test), {"accuracy": accuracy}

    # Start Flower client
    for alpha_val in sys.argv[1:]:

        print("Current alpha value", float(alpha_val))
        print("Total alpha values :", len(sys.argv)-1)

        data1 = np.load('kvasir_party_576.npz', allow_pickle=True)
        X_train1 = data1["x_train"]
        y_train1 = data1["y_train"]
        X_test = data1["x_test"]
        y_test = data1["y_test"]


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

        fl.client.start_numpy_client("127.0.0.1:12345", client=KvasirClient())