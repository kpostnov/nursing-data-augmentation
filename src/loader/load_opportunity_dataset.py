import itertools
import os

import numpy as np
import pandas as pd

from utils.Recording import Recording
import utils.settings as settings


def load_opportunity_dataset(opportunity_dataset_path: str) -> "list[Recording]":
    """
    Returns a list of Recordings from the opportunity dataset
    """
    opportunity_dataset_path += "/dataset"
    subject_ids = range(1, 5)
    recording_ids = range(1, 6)

    # Needed sensor columns
    # 64-76: RLA (13)
    # 90-102: LLA (13)
    # 38-50: BACK (13)
    # 103-118: L-SHOE (16)
    # 119-134: R-SHOE (16)
    sensor_frame_column_indices = np.array(
        [
            *np.arange(38, 51),
            *np.arange(64, 77),
            *np.arange(90, 103),
            *np.arange(103, 119),
            *np.arange(119, 135),
        ]
    )
    # Convert to 0-indexed
    sensor_frame_column_indices -= 1

    label_column_index = 245 - 1
    timeframe_column_index = 1 - 1

    recordings = []
    for sub, rec in itertools.product(subject_ids, recording_ids):
        file_name = f"S{sub}-ADL{rec}.dat"
        file_path = os.path.join(opportunity_dataset_path, file_name)
        print(f"Reading {file_path} ...")
        file_df = pd.read_csv(file_path, delimiter=" ", header=None)
        recording = Recording(
            file_df.iloc[
                :, sensor_frame_column_indices
            ],  # will give us the indices as column names (normally you would give them their name)
            file_df.iloc[:, timeframe_column_index],
            file_df.iloc[:, label_column_index].map(
                lambda label: settings.activity_initial_num_to_activity_idx[label]
            ),  # Use `[0]` to get only one activity | maps 0, 101, 102, 103, 104, 105 to 0, 1, 2, 3, 4, 5
            f"{sub}",
        )

        recordings.append(recording)

    print(f"\n => Total {len(recordings)} recordings read")

    return recordings

