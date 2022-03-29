import itertools
import os

import numpy as np
import pandas

from utils.Recording import Recording


def load_opportunity_dataset(dataset_path: str) -> "list[Recording]":
    """
    Returns a list of Recordings from the opportunity dataset
    """
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
        file_path = os.path.join(dataset_path, file_name)
        p = pandas.read_csv(file_path, delimiter=" ", header=None)
        recording = Recording(
            p.iloc[:, sensor_frame_column_indices],
            p.iloc[:, timeframe_column_index],
            f"{p.iloc[:, label_column_index][0]}",  # TODO: Use all, remove `[0]`
            f"{sub}",
        )

        recordings.append(recording)

    print(f"Read {len(recordings)} recordings")

    return recordings


load_opportunity_dataset(
    "/Users/franz/Projects/dhc-lab/data/OpportunityUCIDataset/dataset"
)
