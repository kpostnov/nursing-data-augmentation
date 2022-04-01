import numpy as np
from dataclasses import dataclass

from utils.typing import assert_type


@dataclass
class Window:
    sensor_array: np.ndarray
    activity: int
    subject: str

    def __init__(self, sensor_array: np.ndarray, activity: int, subject: str) -> None:
        assert_type(
            [(sensor_array, (np.ndarray, np.generic)), (activity, int), (subject, str)]
        )
        self.sensor_array = sensor_array
        self.activity = activity
        self.subject = subject