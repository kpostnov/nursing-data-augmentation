import numpy as np
from dataclasses import dataclass

from utils.typing import assert_type


@dataclass
class Window:
    def __init__(self, sensor_array: np.ndarray, activity: str, subject: str) -> None:
        assert_type(
            [(sensor_array, (np.ndarray, np.generic)), (activity, str), (subject, str)]
        )
        self.sensor_array = sensor_array
        self.activity = activity
        self.subject = subject

    sensor_array: np.ndarray
    activity: str
    subject: str
