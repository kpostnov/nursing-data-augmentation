from utils.typing import assert_type

import pandas as pd
from dataclasses import dataclass



@dataclass
class Recording:
    """
    our base data object
    Multilabel, so expects activity pd.Series

    Future: 
        - add self.recorder
    """

    def __init__(
        self,
        sensor_frame: pd.DataFrame,
        time_frame: pd.Series,
        activities: pd.Series,
        subject: str,
    ) -> None:
        assert_type(
            [
                (sensor_frame, pd.DataFrame),
                (time_frame, pd.Series),
                (activities, pd.Series),
                (subject, str),
            ]
        )
        self.sensor_frame = sensor_frame
        self.time_frame = time_frame
        self.activities = activities
        self.subject = subject

