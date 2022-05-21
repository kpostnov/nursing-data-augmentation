from sklearn.preprocessing import MinMaxScaler, StandardScaler
import pandas as pd

from utils import settings
from datatypes.Recording import Recording
from utils.typing import assert_type


def preprocess(recordings: "list[Recording]", methods: list) -> "list[Recording]":
    """
    Preprocesses the recordings with the given methods and returns the scaler (or None).
    """
    assert_type([(recordings[0], Recording)])

    for method in methods:
        recordings, scaler = method(recordings)
    return recordings, scaler


def interpolate_ffill(recordings: "list[Recording]") -> "list[Recording]":
    """
    The recordings have None values, this function interpolates them.
    NaN values in the beginning (that cannot be interpolated) are filled with the first non-NaN value.
    """
    assert_type([(recordings[0], Recording)])
    fill_method = "ffill"

    for recording in recordings:
        recording.sensor_frame = recording.sensor_frame.fillna(method=fill_method)
        # Handle NaN values in beginning of recording
        recording.sensor_frame = recording.sensor_frame.fillna(method="bfill")

    return recordings, None


def interpolate_linear(recordings: "list[Recording]") -> "list[Recording]":
    """
    The recordings have None values, this function linearly interpolates them
    """
    assert_type([(recordings[0], Recording)])

    for recording in recordings:
        recording.sensor_frame = recording.sensor_frame.interpolate(method="linear")
        # Handle NaN values in beginning of recording
        recording.sensor_frame = recording.sensor_frame.fillna(method="bfill")

    return recordings, None


def normalize_standardscaler(recordings: "list[Recording]") -> "list[Recording]":
    """
    Normalizes the sensor values using StandardScaler
    """
    assert_type([(recordings[0], Recording)])

    # First fit the scaler on all data
    scaler = StandardScaler()
    for recording in recordings:
        scaler.partial_fit(recording.sensor_frame)

    # Then apply normalization on each recording_frame
    for recording in recordings:
        transformed_array = scaler.transform(recording.sensor_frame)
        recording.sensor_frame = pd.DataFrame(
            transformed_array, columns=recording.sensor_frame.columns)
    return recordings, scaler


def normalize_minmaxscaler(recordings: "list[Recording]") -> "list[Recording]":
    """
    Normalizes the sensor values per recording channel to be in range 0 to 1
    """
    assert_type([(recordings[0], Recording)])

    scaler = MinMaxScaler()
    # complete_sensor_frame = pd.concat([recording.sensor_frame for recording in recordings])
    # scaler.fit(complete_sensor_frame)
    for recording in recordings:
        scaler.partial_fit(recording.sensor_frame)

    for recording in recordings:
        transformed_array = scaler.transform(recording.sensor_frame)
        recording.sensor_frame = pd.DataFrame(
            transformed_array, columns=recording.sensor_frame.columns)

    return recordings, scaler
