from models.RainbowModel import RainbowModel
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import h5py
from tensorflow.keras import regularizers
from tensorflow.keras.layers import (
    Input,
    Conv2D,
    Dense,
    Flatten,
    Dropout,
    LSTM,
    GlobalMaxPooling1D,
    MaxPooling2D,
    BatchNormalization,
)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import itertools

from datetime import datetime
import os


class JensModel(RainbowModel):
    def __init__(self, epochs=10, **kwargs):
        """
        epochs=10
        :param kwargs:
            window_size: int
            stride_size: int
            n_features: int
            n_outputs: int
        """

        # hyper params to instance vars
        super().__init__(**kwargs)
        self.window_size = kwargs["window_size"]
        self.stride_size = kwargs["stride_size"]

        self.epochs = epochs

        # create model
        self.model = self._create_model(kwargs["n_features"], kwargs["n_outputs"])

    def _create_model(self, n_features, n_outputs):
        print(
            f"Building model for {self.window_size} timesteps (window_size) and {n_features} features"
        )

        i = Input(shape=(self.window_size, n_features))  # before: self.x_train[0].shape
        x = Conv2D(
            32,
            (3, 3),
            strides=2,
            activation="relu",
            padding="same",
            kernel_regularizer=regularizers.l2(0.0005),
        )(i)
        x = BatchNormalization()(x)
        x = MaxPooling2D((2, 2))(x)
        x = Dropout(0.2)(x)
        x = Conv2D(
            64,
            (3, 3),
            strides=2,
            activation="relu",
            padding="same",
            kernel_regularizer=regularizers.l2(0.0005),
        )(x)
        x = BatchNormalization()(x)
        x = Dropout(0.4)(x)
        x = Conv2D(
            128,
            (3, 3),
            strides=2,
            activation="relu",
            padding="same",
            kernel_regularizer=regularizers.l2(0.0005),
        )(x)
        x = BatchNormalization()(x)
        x = MaxPooling2D((2, 2))(x)
        x = Dropout(0.2)(x)
        x = Flatten()(x)
        x = Dropout(0.2)(x)
        x = Dense(1024, activation="relu")(x)
        x = Dropout(0.2)(x)
        x = Dense(n_outputs, activation="softmax")(x)
        model = Model(i, x)
        model.compile(
            optimizer=Adam(lr=0.001),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )

        return model
