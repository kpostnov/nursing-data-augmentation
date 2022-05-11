import itertools
from typing import Callable
import numpy as np
import random
from utils import settings
from datatypes.Recording import Recording
from datatypes.Window import Window
from utils.array_operations import transform_to_subarrays
from utils.typing import assert_type
from tensorflow.keras.utils import to_categorical


class Windowizer:
    def __init__(self, window_size: int, stride_size: int, windowize: Callable, frequency: int = 60):
        self.window_size = window_size
        self.stride_size = stride_size
        self.windowize = windowize
        self.frequency = frequency
    
    def print_non_continuous_windowize_monitoring(self, recordings: "list[Recording]") -> None:
        """
        Prints the number of timesteps that cannot be used in the model (discarded). 
        This can happen if a window contains more than one activity.
        """
        def n_wasted_timesteps_windowize(recording: "Recording"):
            activities = recording.activities.to_numpy()
            change_idxs = np.where(activities[:-1] != activities[1:])[0] + 1

            def get_n_wasted_timesteps(label_len):
                return (
                    (label_len - self.window_size) % (self.stride_size)
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

        n_total_timesteps = sum(map(lambda recording: len(recording.activities), recordings))
        n_wasted_timesteps = sum(map(n_wasted_timesteps_windowize, recordings))

        print(
            f"=> Total recording time: \n\t before: {to_hours_str(n_total_timesteps)}\n\t after: {to_hours_str(n_total_timesteps - n_wasted_timesteps)}"
        )
        print(f"n_total_timesteps: {n_total_timesteps}")
        print(f"n_wasted_timesteps: {n_wasted_timesteps}")


    def windowize_jumping(self, recordings: "list[Recording]") -> "list[Window]":
        """
        Windowizes the recording from beginning to end with overlap.
        Windows that contain multiple activities skip time steps that do not fit or are too short.
        """
        def windowize_recording(recording: "Recording") -> "list[Window]":
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
        
        assert_type([(recordings[0], Recording)])
        assert(self.window_size is not None), "window_size can't be None."

        self.print_non_continuous_windowize_monitoring(recordings)
        print("Windowizing in progress...")
        recording_windows = list(map(windowize_recording, recordings))
        print("Windowizing done.")

        # Flatten list of lists
        return list(itertools.chain.from_iterable(recording_windows))


    def windowize_sliding(self, recordings: "list[Recording]") -> "list[Window]":
        """
        Windowizes the recording from beginning to end without overlap and skipping time steps.
        The label is determined by the last time step in the window.
        """
        def windowize_recording(recording: "Recording") -> "list[Window]":
            recording_sensor_array = (recording.sensor_frame.to_numpy())
            activities = recording.activities.to_numpy()

            sensor_subarrays = transform_to_subarrays(recording_sensor_array, self.window_size, self.stride_size)
            activity_subarray = activities[self.window_size-1::self.stride_size]

            assert len(sensor_subarrays) == len(activity_subarray)

            recording_windows = [Window(sensor_subarray, int(activity_subarray[i]), recording.subject)
                                    for i, sensor_subarray in enumerate(sensor_subarrays)]

            return recording_windows
            
        assert_type([(recordings[0], Recording)])
        assert(self.window_size is not None), "window_size can't be None."

        print("Windowizing in progress...")
        recording_windows = list(map(windowize_recording, recordings))
        print("Windowizing done.")

        # Flatten list of lists
        return list(itertools.chain.from_iterable(recording_windows))


    def windowize_windowize(self, recordings: "list[Recording]") -> "list[Window]":
        """
        Returns windows of the recordings.
        """
        windows = self.windowize(self, recordings)
        return windows

    
    def windowize_convert(self, recordings_train: "list[Recording]") -> "tuple[np.ndarray,np.ndarray]":
        """
        Shuffles the windows
        """
        windows_train = self.windowize(self, recordings_train)
        random.shuffle(windows_train)  # many running windows in a row?, one batch too homogenous?, lets shuffle
        X_train, y_train = self.convert(windows_train)

        return X_train, y_train


    def convert(self, windows: "list[Window]") -> "tuple[np.ndarray, np.ndarray]":
        """
        Converts the windows to two numpy arrays as needed for the model
        sensor_array (data) and activity_array (labels)
        """
        def split(windows: "list[Window]") -> "tuple[np.ndarray, np.ndarray]":
            assert_type([(windows[0], Window)])

            sensor_arrays = list(map(lambda window: window.sensor_array, windows))
            activities = list(map(lambda window: window.activity, windows))

            # to_categorical converts the activity_array to the dimensions needed
            activity_vectors = to_categorical(
                np.array(activities),
                num_classes=len(settings.ACTIVITIES),
            )

            return np.array(sensor_arrays), np.array(activity_vectors)

        X_train, y_train = split(windows)
        return np.expand_dims(X_train, -1), y_train
