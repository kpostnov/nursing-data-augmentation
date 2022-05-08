from sklearn.preprocessing import MinMaxScaler, StandardScaler
import pandas as pd

from utils import settings
from datatypes.Recording import Recording
from utils.typing import assert_type


def preprocess(recordings: "list[Recording]", methods: list) -> "list[Recording]":
    """
    Preprocesses the recordings with the given methods
    """
    assert_type([(recordings[0], Recording)])

    for method in methods:
        recordings = method(recordings)
    return recordings
    

def ordonez_preprocess(recordings: "list[Recording]") -> "list[Recording]":
    """
    1. _interpolate_linear
    2. _normalize_standardscaler
    """
    assert_type([(recordings[0], Recording)])

    recordings = interpolate_linear(recordings)
    recordings = normalize_minmaxscaler(recordings)
    return recordings


def pamap2_preprocess(recordings: "list[Recording]") -> "list[Recording]":
    """
    1. _interpolate_ffill
    """
    assert_type([(recordings[0], Recording)])

    recordings = interpolate_ffill(recordings)
    return recordings


def jens_preprocess(recordings: "list[Recording]") -> "list[Recording]":
    """
    1. _interpolate_linear
    """
    assert_type([(recordings[0], Recording)])

    recordings = interpolate_linear(recordings)
    return recordings

# Preprocess-Library ------------------------------------------------------------


def map_activities_to_id(recordings: "list[Recording]") -> "list[Recording]":
    def map_recording_activities_to_id(recording):
        """
        Converts the string labels of one recording to integers"
        """
        recording.activities = pd.Series(
            [
                settings.ACTIVITIES.get(activity) or settings.ACTIVITIES["invalid"]
                for activity in recording.activities
            ]
        )
        return recording

    # Convert the string labels of all recordings to integers
    return [map_recording_activities_to_id(recording) for recording in recordings]


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

    return recordings


def interpolate_linear(recordings: "list[Recording]") -> "list[Recording]":
    """
    The recordings have None values, this function linearly interpolates them
    """
    assert_type([(recordings[0], Recording)])

    for recording in recordings:
        recording.sensor_frame = recording.sensor_frame.interpolate(method="linear")
        # Handle NaN values in beginning of recording
        recording.sensor_frame = recording.sensor_frame.fillna(method="bfill")

    return recordings


def normalize_standardscaler(recordings: "list[Recording]") -> "list[Recording]":
    """
    Normalizes the sensor values to be in range -1 to 1
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
    return recordings


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

    return recordings
