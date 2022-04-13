from utils.Recording import Recording
import utils.settings as settings
import copy


def filter_acceleration(recordings: "list[Recording]") -> "list[Recording]":
    """
    Removes the Quaterion data from the recordings, only keeps the acceleration data
    """
    acceleration_recordings = copy.deepcopy(recordings)

    acceleration_column_names = [
        f"FreeAcc_{axis}_{sensor_suffix}"
        for sensor_suffix in settings.SENSOR_SUFFIX_ORDER
        for axis in ["X", "Y", "Z"]
    ]

    for recording in acceleration_recordings:
        recording.sensor_frame = recording.sensor_frame[acceleration_column_names]

    return acceleration_recordings
