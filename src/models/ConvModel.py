from random import shuffle
from models.RainbowModelLeaveRecsOut import RainbowModelLeaveRecsOut
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout  # type: ignore
from tensorflow.keras.models import Sequential  # type: ignore
import numpy as np

from utils.Recording import Recording
from utils.array_operations import split_list_by_percentage
from utils.typing import assert_type


class ConvModel(RainbowModelLeaveRecsOut):
    def __init__(self, **kwargs):
        """
        Convolutional model
        :param kwargs:
            window_size: int
            stride_size: int
            test_percentage: float
            n_features: int
            n_outputs: int
        """

        # hyper params to instance vars
        self.window_size = kwargs["window_size"]
        self.stride_size = kwargs["stride_size"]
        self.test_percentage = kwargs["test_percentage"]

        self.verbose = 0
        self.epochs = 10
        self.batch_size = 32

        # create model
        self.model = self.__create_model(kwargs["n_features"], kwargs["n_outputs"])

    def __create_model(self, n_features, n_outputs):
        # window_size, n_features, n_outputs = X.shape[1], X.shape[2], y.shape[1]

        print(
            f"Building model for {self.window_size} timesteps (window_size) and {n_features} features"
        )
        model = Sequential()
        model.add(
            Conv1D(
                filters=64,
                kernel_size=3,
                activation="relu",
                input_shape=(self.window_size, n_features),
            )
        )
        model.add(Conv1D(filters=64, kernel_size=3, activation="relu"))
        model.add(Dropout(0.5))
        model.add(MaxPooling1D(pool_size=2))
        model.add(Flatten())
        model.add(Dense(100, activation="relu"))
        model.add(Dense(n_outputs, activation="softmax"))
        model.compile(
            loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"]
        )
        return model
