# pylint: disable=locally-disabled, multiple-statements, fixme, line-too-long, no-name-in-module, wrong-import-order, bad-option-value

from abc import abstractmethod
from models.RainbowModel import RainbowModel
import numpy as np
import matplotlib.pyplot as plt

from tensorflow import keras
from keras.models import Sequential
from keras import layers


class DeepConvLSTM(RainbowModel):

    # General
    batch_size = 100
    n_filters = 64
    kernel_size = 5
    n_lstm_units = 128

    def __init__(self, **kwargs):
        """
        :param kwargs:
            window_size: int
            n_features: int
            n_outputs: int
        """

        super().__init__(**kwargs)
        self.model_name = "DeepConvLSTM"

        # Create model
        self.model = self._create_model(self.n_features, self.n_outputs)
        print(
            f"Building model for {self.window_size} timesteps (window_size) and {kwargs['n_features']} features."
        )


    def _create_model(self, n_features, n_outputs):
        model = keras.Sequential()
        model.add(keras.Input(shape=(self.window_size, n_features, 1)))

        # Random orthogonal weights
        initializer = keras.initializers.Orthogonal()

        # 4 CNN layers
        model.add(layers.Conv2D(self.n_filters, kernel_size=(
            self.kernel_size, 1), activation="relu", kernel_initializer=initializer))
        model.add(layers.Conv2D(self.n_filters, kernel_size=(
            self.kernel_size, 1), activation="relu", kernel_initializer=initializer))
        model.add(layers.Conv2D(self.n_filters, kernel_size=(
            self.kernel_size, 1), activation="relu", kernel_initializer=initializer))
        model.add(layers.Conv2D(self.n_filters, kernel_size=(
            self.kernel_size, 1), activation="relu", kernel_initializer=initializer))

        # (None, window_size, n_features, n_filters) -> (None, n_features, window_size * n_filters)
        model.add(layers.Permute((2, 1, 3)))
        model.add(layers.Reshape((int(model.layers[4].output_shape[1]),
                                  int(model.layers[4].output_shape[2]) * int(model.layers[4].output_shape[3]))))

        # 2 LSTM layers
        model.add(layers.LSTM(self.n_lstm_units, activation="tanh", dropout=0.5,
                  return_sequences=True, kernel_initializer=initializer))
        model.add(layers.LSTM(self.n_lstm_units, activation="tanh", dropout=0.5,
                  return_sequences=True, kernel_initializer=initializer))
        model.add(layers.Flatten())

        model.add(layers.Dense(n_outputs, activation="softmax"))

        print(model.summary())

        model.compile(
            loss='categorical_crossentropy',
            optimizer='RMSprop', # learning_rate = 0.001
            metrics=[keras.metrics.CategoricalAccuracy(), 'accuracy'])

        return model
