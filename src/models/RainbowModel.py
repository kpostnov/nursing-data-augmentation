# pylint: disable=locally-disabled, multiple-statements, fixme, line-too-long, no-name-in-module, unused_import, wrong-import-order, bad-option-value

from abc import ABC, abstractmethod
from math import sqrt
from typing import Any, Union
from numpy.core.numeric import full
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from tensorflow.keras.utils import to_categorical  # type: ignore
from random import shuffle
import numpy as np
import pandas as pd
import tensorflow as tf  # type: ignore
from datetime import datetime
import os

from tensorflow.python.saved_model.utils_impl import get_saved_model_pb_path  # type: ignore

from utils.array_operations import split_list_by_percentage, transform_to_subarrays
from utils.Recording import Recording
from utils.Window import Window

from utils.typing import assert_type
import utils.settings as settings


class RainbowModel(ABC):

    # general
    model_name = None

    # variables that need to be implemented in the child class
    window_size: Union[int, None] = None
    stride_size: Union[int, None] = None
    class_weight = None

    model: Any = None
    batch_size: Union[int, None] = None
    verbose: Union[int, None] = None
    epochs: Union[int, None] = None
    kwargs = None

    @abstractmethod
    def __init__(self, **kwargs):
        """
        Builds a model, assigns it to self.model = ...
        It can take hyper params as arguments that are intended to be varied in the future.
        If hyper params dont directly influence the model creation (e.g. meant for normalisation),
        they need to be stored as instance variable, that they can be accessed, when needed.
        """

        # self.model = None
        # assert (self.model is not None)
        self.kwargs = kwargs

    # @error_after_seconds(600) # after 10 minutes, something is wrong
    def windowize_convert_fit(self, recordings_train: "list[Recording]") -> None:
        """
        For a data efficient comparison between models, the preprocessed data for
        training and evaluation of the model only exists, while this method is running

        shuffles the windows
        """
        assert_type([(recordings_train[0], Recording)])
        X_train, y_train = self.windowize_convert(recordings_train)
        self.fit(X_train, y_train)

    # Preprocess ----------------------------------------------------------------------

    def windowize_convert(
        self, recordings_train: "list[Recording]"
    ) -> "tuple[np.ndarray,np.ndarray]":
        """
        shuffles the windows
        """
        windows_train = self.windowize(recordings_train)
        shuffle(
            windows_train
        )  # many running windows in a row?, one batch too homogenous?, lets shuffle
        X_train, y_train = self.convert(windows_train)
        return X_train, y_train

    def windowize(self, recordings: "list[Recording]") -> "list[Window]":
        """
        based on the hyper param for window size, windowizes the recording_frames
        convertion to numpy arrays
        """
        assert_type([(recordings[0], Recording)])

        assert (
            self.window_size is not None
        ), "window_size has to be set in the constructor of your concrete model class please, you stupid ass"
        assert (
            self.stride_size is not None
        ), "stride_size has to be set in the constructor of your concrete model class, please"

        windows: "list[Window]" = []
        for recording in recordings:
            sensor_array = recording.sensor_frame.to_numpy()
            sensor_subarrays = transform_to_subarrays(
                sensor_array, self.window_size, self.stride_size
            )
            recording_windows = list(
                map(
                    lambda sensor_subarray: Window(
                        sensor_subarray, recording.activity, recording.subject
                    ),
                    sensor_subarrays,
                )
            )
            windows.extend(recording_windows)
        return windows

    def convert(self, windows: "list[Window]") -> "tuple[np.ndarray, np.ndarray]":
        """
        converts the windows to two numpy arrays as needed for the concrete model
        sensor_array (data) and activity_array (labels)
        """
        assert_type([(windows[0], Window)])

        sensor_arrays = list(map(lambda window: window.sensor_array, windows))
        activities = list(map(lambda window: window.activity, windows))

        # to_categorical converts the activity_array to the dimensions needed
        activity_vectors = to_categorical(
            np.array(activities),
            num_classes=len(settings.activity_initial_num_to_activity_str),
        )

        return np.array(sensor_arrays), np.array(activity_vectors)

    # Fit ----------------------------------------------------------------------

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        """
        Fit the self.model to the data
        """
        assert_type(
            [(X_train, (np.ndarray, np.generic)), (y_train, (np.ndarray, np.generic))]
        )
        assert (
            X_train.shape[0] == y_train.shape[0]
        ), "X_train and y_train have to have the same length"
        # print(f"Fitting with class weight: {self.class_weight}")
        history = self.model.fit(
            X_train,
            y_train,
            validation_split=0.2,
            epochs=self.epochs,
            batch_size=self.batch_size,
            verbose=self.verbose,
            class_weight=self.class_weight
        )
        self.history = history

    # Predict ------------------------------------------------------------------------

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        """
        gets a list of windows and returns a list of prediction_vectors
        """
        return self.model.predict(X_test)

    
