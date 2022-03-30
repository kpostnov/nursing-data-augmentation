from sklearn.preprocessing import MinMaxScaler
import pandas as pd
from utils.Recording import Recording
from utils.typing import assert_type


class Preprocessor:

    # add parametrization
    def __init__(self, fill_method="ffill"):
        self.fill_method = fill_method

    def preprocess(self, raw_recordings: "list[Recording]") -> "list[Recording]":
        recordings = self.interpolate(raw_recordings)
        recordings = self.normalize(recordings)
        return recordings

    def interpolate(self, raw_recordings: "list[Recording]") -> "list[Recording]":
        """
        the raw_recordings have None values, this function interpolates them
        """
        assert_type([(raw_recordings[0], Recording)])

        for recording in raw_recordings:
            recording.sensor_frame = recording.sensor_frame.fillna(
                method=self.fill_method
            )

        return raw_recordings

    def normalize(self, recordings: "list[Recording]") -> "list[Recording]":
        """
        Normalizes the sensor values to be in range 0 to 1
        """
        assert_type([(recordings[0], Recording)])

        # First fit the scaler on all data
        scaler = MinMaxScaler()
        self.scaler = scaler
        for recording in recordings:
            scaler.fit(recording.sensor_frame)

        # Then apply normalization on each recording_frame
        for recording in recordings:
            transformed_array = scaler.transform(recording.sensor_frame)
            recording.sensor_frame = pd.DataFrame(
                transformed_array, columns=recording.sensor_frame.columns
            )
        return recordings
