# pylint: disable=locally-disabled, multiple-statements, fixme, line-too-long, no-name-in-module, wrong-import-order, bad-option-value

from abc import abstractmethod
import itertools
from models.RainbowModel import RainbowModel
import numpy as np
import matplotlib.pyplot as plt

from tensorflow import keras
from keras.models import Sequential
from keras import layers

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from utils.typing import assert_type
from utils.Window import Window
from utils.Recording import Recording
from utils.array_operations import transform_to_subarrays


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
        self.window_size = kwargs["window_size"]
        self.n_features = kwargs["n_features"]
        self.n_outputs = kwargs["n_outputs"]
        self.verbose = kwargs.get("verbose") or True
        self.n_epochs = kwargs.get("n_epochs") or 10
        self.learning_rate = kwargs.get("learning_rate") or 0.001
        self.frequency = kwargs.get("frequency") or 30
        self.stride_size = kwargs.get("stride_size") or self.window_size
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
            optimizer='RMSprop',
            metrics=[keras.metrics.CategoricalAccuracy(), 'accuracy'])

        return model


    @abstractmethod
    def windowize(self, recordings: "list[Recording]") -> "list[Window]":
        raise NotImplementedError


    def convert(self, windows: "list[Window]") -> "tuple[np.ndarray, np.ndarray]":
        X_train, y_train = super().convert(windows)
        return np.expand_dims(X_train, -1), y_train


class SlidingWindowDeepConvLSTM(DeepConvLSTM):
    """
    Windowizes the recording from beginning to end without overlap and skipping time steps.
    The label is determined by the last time step in the window.
    """

    def _windowize_recording(self, recording: "Recording") -> "list[Window]":
        recording_sensor_array = (recording.sensor_frame.to_numpy())
        activities = recording.activities.to_numpy()

        sensor_subarrays = transform_to_subarrays(recording_sensor_array, self.window_size, self.stride_size)
        activity_subarray = activities[self.window_size-1::self.stride_size]

        assert len(sensor_subarrays) == len(activity_subarray)

        recording_windows = [Window(sensor_subarray, int(activity_subarray[i]), recording.subject)
                             for i, sensor_subarray in enumerate(sensor_subarrays)]

        return recording_windows


    def windowize(self, recordings: "list[Recording]") -> "list[Window]":
        assert_type([(recordings[0], Recording)])
        assert (
            self.window_size is not None
        ), "window_size has to be set in the constructor of your concrete model class."

        print("Windowizing in progress...")
        recording_windows = list(map(self._windowize_recording, recordings))
        print("Windowizing done.")

        # Flatten list of lists
        return list(itertools.chain.from_iterable(recording_windows))


class JumpingWindowDeepConvLSTM(DeepConvLSTM):
    """
    Windowizes the recording from beginning to end with overlap.
    Windows that contain multiple activities skip time steps that do not fit or are too short.
    """

    def _windowize_recording(self, recording: "Recording") -> "list[Window]":
        windows = []
        recording_sensor_array = (recording.sensor_frame.to_numpy())
        activities = recording.activities.to_numpy()

        start = 0
        end = 0

        def last_start_stamp_not_reached(start):
            return start + self.window_size - 1 < len(recording_sensor_array)

        while last_start_stamp_not_reached(start):
            end = start + self.window_size - 1

            # Has planned window the same activity in the beginning and the end?
            if (len(set(activities[start: (end + 1)])) == 1):
                window_sensor_array = recording_sensor_array[start: (end + 1), :]
                activity = activities[start]
                start += self.stride_size

                windows.append(Window(window_sensor_array,
                               int(activity), recording.subject))
            else:
                while last_start_stamp_not_reached(start):
                    if activities[start] != activities[start + 1]:
                        start += 1
                        break
                    start += 1

        return windows


    def _print_non_continuous_windowize_monitoring(self, recordings: "list[Recording]"):
        """
        Prints the number of timesteps that cannot be used in the model (discarded). 
        This can happen if a window contains more than one activity.
        """
        def n_wasted_timesteps_windowize(recording: "Recording"):
            activities = recording.activities.to_numpy()
            change_idxs = np.where(activities[:-1] != activities[1:])[0] + 1

            def get_n_wasted_timesteps(label_len):
                return (
                    (label_len - self.window_size) % (self.window_size // 2)
                    if label_len >= self.window_size
                    else label_len
                )

            start_idx = 0
            n_wasted_timesteps = 0
            for change_idx in change_idxs:
                label_len = change_idx - start_idx
                n_wasted_timesteps += get_n_wasted_timesteps(label_len)
                start_idx = change_idx
            last_label_len = (
                len(activities) - change_idxs[-1]
                if len(change_idxs) > 0
                else len(activities)
            )
            n_wasted_timesteps += get_n_wasted_timesteps(last_label_len)
            return n_wasted_timesteps

        def to_hours_str(n_timesteps) -> int:
            hz = self.frequency
            minutes = (n_timesteps / hz) / 60
            hours = int(minutes / 60)
            minutes_remaining = int(minutes % 60)
            return f"{hours}h {minutes_remaining}m"

        n_total_timesteps = sum(
            map(lambda recording: len(recording.activities), recordings))
        n_wasted_timesteps = sum(map(n_wasted_timesteps_windowize, recordings))

        print(
            f"=> Total recording time: \n\t before: {to_hours_str(n_total_timesteps)}\n\t after: {to_hours_str(n_total_timesteps - n_wasted_timesteps)}"
        )
        print(f"n_total_timesteps: {n_total_timesteps}")
        print(f"n_wasted_timesteps: {n_wasted_timesteps}")


    def windowize(self, recordings: "list[Recording]") -> "list[Window]":
        assert_type([(recordings[0], Recording)])
        assert (
            self.window_size is not None
        ), "window_size has to be set in the constructor of your concrete model class."

        self._print_non_continuous_windowize_monitoring(recordings)
        print("Windowizing in progress...")
        recording_windows = list(map(self._windowize_recording, recordings))
        print("Windowizing done.")

        # Flatten list of lists
        return list(itertools.chain.from_iterable(recording_windows))
