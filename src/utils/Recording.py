import pandas as pd
from dataclasses import dataclass

from utils.typing import assert_type


@dataclass
class Recording:
    """
    our base data object, that holds all information about one XSens unit, the recording

    info: activiy might get a list in the future
    """

    def __init__(
        self,
        sensor_frame: pd.DataFrame,
        time_frame: pd.Series,
        activity: str,
        subject: str,
    ) -> None:
        assert_type(
            [
                (sensor_frame, pd.DataFrame),
                (time_frame, pd.Series),
                (activity, str),
                (subject, str),
            ]
        )
        self.sensor_frame = sensor_frame
        self.time_frame = time_frame
        self.activity = activity
        self.subject = subject

    sensor_frame: pd.DataFrame
    time_frame: pd.Series
    activity: str
    subject: str
