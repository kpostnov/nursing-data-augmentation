# pylint: disable=locally-disabled, multiple-statements, fixme, line-too-long, no-name-in-module, import-error, wrong-import-order, bad-option-value

import os
from abc import ABC, abstractmethod
from typing import Any, Union
import numpy as np
import tensorflow as tf  # type: ignore

from utils.array_operations import transform_to_subarrays
from utils.folder_operations import create_folders_in_path
from utils.typing import assert_type


class RainbowModel(ABC):

    # General
    model_name = None

    # Variables that need to be implemented in the child class
    window_size: Union[int, None] = None
    stride_size: Union[int, None] = None
    class_weight = None

    model: Any = None
    batch_size: Union[int, None] = None
    verbose: Union[int, None] = None
    n_epochs: Union[int, None] = None
    kwargs = None


    @abstractmethod
    def __init__(self, **kwargs):
        """
        Builds a model, assigns it to self.model = ...
        It can take hyper params as arguments that are intended to be varied in the future.
        If hyper params dont directly influence the model creation (e.g. meant for normalisation),
        they need to be stored as instance variable, that they can be accessed, when needed.
        """

        self.kwargs = kwargs

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

        history = self.model.fit(
            X_train,
            y_train,
            validation_split=0.2,
            epochs=self.n_epochs,
            batch_size=self.batch_size,
            verbose=self.verbose,
            class_weight=self.class_weight,
        )
        self.history = history


    # Predict ------------------------------------------------------------------------

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        """
        Gets a list of windows and returns a list of prediction_vectors
        """
        return self.model.predict(X_test)


    def export(self, path: str) -> None:
        """
        Will create an 'export' folder in the path, and save the model there in 3 different formats
        """
        print("Exporting model ...")

        # Define, create folder structure
        export_path = os.path.join(path, "export")
        export_path_raw_model = os.path.join(export_path, "raw_model")
        create_folders_in_path(export_path_raw_model)

        # 1/3 Export raw model ------------------------------------------------------------
        self.model.save(export_path_raw_model)

        # 2/3 Export .h5 model ------------------------------------------------------------
        self.model.save(export_path + "/" + self.model_name + ".h5", save_format="h5")

        # 3/3 Export .h5 model ------------------------------------------------------------
        converter = tf.lite.TFLiteConverter.from_keras_model(self.model)

        converter.optimizations = [
            tf.lite.Optimize.DEFAULT
        ] 
        converter.experimental_new_converter = True
        converter.target_spec.supported_ops = [
            tf.lite.OpsSet.TFLITE_BUILTINS,
            tf.lite.OpsSet.SELECT_TF_OPS,
        ]

        tflite_model = converter.convert()
        with open(f"{export_path}/{self.model_name}.tflite", "wb") as f:
            f.write(tflite_model)

        print("Export finished")
