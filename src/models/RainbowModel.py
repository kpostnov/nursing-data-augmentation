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
from utils.server_manager import error_after_seconds
from utils.visualizing import visualizeAccuracy, visualizeLoss
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
        """
        assert_type([(recordings_train[0], Recording)])
        X_train, y_train = self.windowize_convert(recordings_train)
        self.fit(X_train, y_train)

    # Preprocess ----------------------------------------------------------------------

    def windowize_convert(
        self, recordings_train: "list[Recording]"
    ) -> "tuple[np.ndarray,np.ndarray]":
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

        def window_sensor_array(window):
            return window.sensor_array

        sensor_arrays = list(map(window_sensor_array, windows))

        def window_activity_number(window):
            return settings.ACTIVITIES[window.activity]

        activity_index_array = list(map(window_activity_number, windows))

        # to_categorical converts the activity_array to the dimensions needed
        activity_vectors = to_categorical(
            np.array(activity_index_array), num_classes=len(settings.ACTIVITIES)
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
            class_weight=self.class_weight,
        )
        self.history = history

    # Predict ------------------------------------------------------------------------

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        """
        gets a list of windows and returns a list of prediction_vectors
        """
        return self.model.predict(X_test)

    # Utils --------------------------------------------------------------------------

    # def add_acceleration_magnitude(self, df: pd.DataFrame) -> None:
    #     for sensor_suffix in settings.SENSOR_SUFFIX_ORDER:
    #         acceleration_column_names = [f"FreeAcc_{axis}_{sensor_suffix}" for axis in ['X', 'Y', 'Z']]
    #         sq_value = 0
    #         for acceleration_column_name in acceleration_column_names:
    #             sq_value += df[acceleration_column_name]**2
    #         df[f'magnitude_{sensor_suffix}'] = sqrt(sq_value)

    def plot_confusion_matrix(self, X_test, y_test) -> None:
        """
        Plots a confusion matrix
        """
        assert_type(
            [(X_test, (np.ndarray, np.generic)), (y_test, (np.ndarray, np.generic))]
        )

        predictions = self.model.predict(X_test)
        predictions = np.argmax(predictions, axis=1)
        y_test = np.argmax(y_test, axis=1)

        self.confusion_matrix = ConfusionMatrixDisplay(
            confusion_matrix=confusion_matrix(y_test, predictions)
        )
        self.confusion_matrix.plot()

    def get_model_name(self) -> str:
        if self.model_name is None:
            currentDT = datetime.now()
            currentDT_str = currentDT.strftime("%y-%m-%d_%H-%M-%S_%f")
            self.model_name = currentDT_str + "---" + type(self).__name__
        return self.model_name

    def save_model(self, path: str, name_suffix: str = "") -> "tuple[str, str]":
        """
        Saves the model to the given path
        """
        assert_type([(path, str)])

        # create directory
        model_name = self.get_model_name() + name_suffix
        model_folder_path = os.path.join(path, model_name)
        model_folder_path_internal = os.path.join(model_folder_path, "model")
        os.makedirs(model_folder_path_internal, exist_ok=True)

        # save normal model
        self.model.save(model_folder_path_internal)

        # save model as .h5
        self.model.save(model_folder_path + "/" + model_name + ".h5", save_format="h5")

        # save model as .tflite
        converter = tf.lite.TFLiteConverter.from_keras_model(self.model)

        # TODO: Optimizations for new tensorflow version
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.experimental_new_converter = True
        converter.target_spec.supported_ops = [
            tf.lite.OpsSet.TFLITE_BUILTINS,
            tf.lite.OpsSet.SELECT_TF_OPS,
        ]

        tflite_model = converter.convert()
        # converter = tf.lite.TFLiteConverter.from_saved_model(model_folder_path_internal)
        # tflite_model = converter.convert()
        print(f"Saving TFLite to {model_folder_path}/{model_name}.tflite")
        with open(f"{model_folder_path}/{model_name}.tflite", "wb") as f:
            f.write(tflite_model)
        return (
            model_folder_path + "/" + model_name + ".h5",
            f"{model_folder_path}/{model_name}.tflite",
        )
