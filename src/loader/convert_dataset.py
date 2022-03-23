from copy import deepcopy
from typing import List
import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation as R, RotationSpline

from utils import settings
from utils.Recording import Recording


def convert_quaternion_to_matrix(recordings: List[Recording]) -> List[Recording]:
    recordings = deepcopy(recordings)

    # Iterate over all recordings
    for idx, recording in enumerate(recordings):
        if idx % 10 == 0:
            print(f"Converting quaternion to matrix for recording {idx}")

        # Iterate over all sensors - we need to convert this many quaternions
        for sensor_suffix in settings.SENSOR_SUFFIX_ORDER:
            # Build the column names, that we need to select for the quaternion
            quaternion_cols = [f"Quat_{axis}_{sensor_suffix}" for axis in ['W', 'X', 'Y', 'Z']]

            # The matrix is 3x3, so we need 5 more columns. We add these to the Quat-columns:
            added_rotation_cols = [f"Rot_{axis}_{sensor_suffix}" for axis in ['1', '2', '3', '4', '5']]
            # Insert the rows before writing to them is necessary
            for name in added_rotation_cols:
                recording.sensor_frame.insert(recording.sensor_frame.columns.get_loc(quaternion_cols[-1]), name, np.nan)
            # To not get an error when we try to read the quaternion, select only not nan rows
            filled_rows = recording.sensor_frame[quaternion_cols[0]].notnull()

            # Read all quaternions simultaneously
            quaternions = R.from_quat(recording.sensor_frame.loc[filled_rows, quaternion_cols])

            # Convert to matrix and reshape them to be an array of 9 values
            matrices = quaternions.as_matrix().reshape((-1, 9))

            # Write the matrices to the quat + new columns
            recording.sensor_frame.loc[filled_rows, quaternion_cols+added_rotation_cols] = matrices

    return recordings


def convert_quaternion_to_euler(recordings: List[Recording]) -> List[Recording]:
    recordings = deepcopy(recordings)

    # Iterate over all recordings
    for idx, recording in enumerate(recordings):
        if idx % 10 == 0:
            print(f"Converting quaternion to euler angles for recording {idx}", end='\r')

        # Iterate over all sensors - we need to convert this many quaternions
        for sensor_suffix in settings.SENSOR_SUFFIX_ORDER:
            # Build the column names, that we need to select for the quaternion
            quaternion_cols = [f"Quat_{axis}_{sensor_suffix}" for axis in ['W', 'X', 'Y', 'Z']]

            # To not get an error when we try to read the quaternion, select only not nan rows
            filled_rows = recording.sensor_frame[quaternion_cols[0]].notnull()

            # Read all quaternions simultaneously
            quaternions = R.from_quat(recording.sensor_frame.loc[filled_rows, quaternion_cols])

            # Convert to euler angles
            degrees = quaternions.as_euler('zyx')

            # Write them and remove the leftover column
            recording.sensor_frame.loc[filled_rows, quaternion_cols[:3]] = degrees
            recording.sensor_frame.drop(quaternion_cols[3], axis=1, inplace=True)

    print()
    return recordings


def convert_quaternion_to_vector(recordings: List[Recording]) -> List[Recording]:
    recordings = deepcopy(recordings)

    # Iterate over all recordings
    for idx, recording in enumerate(recordings):
        if idx % 10 == 0:
            print(f"Converting quaternion to euler angles for recording {idx}")

        # Iterate over all sensors - we need to convert this many quaternions
        for sensor_suffix in settings.SENSOR_SUFFIX_ORDER:
            # Build the column names, that we need to select for the quaternion
            quaternion_cols = [f"Quat_{axis}_{sensor_suffix}" for axis in ['W', 'X', 'Y', 'Z']]

            # To not get an error when we try to read the quaternion, select only not nan rows
            filled_rows = recording.sensor_frame[quaternion_cols[0]].notnull()

            # Read all quaternions simultaneously
            quaternions = R.from_quat(recording.sensor_frame.loc[filled_rows, quaternion_cols])

            # Convert to euler angles
            vector = quaternions.apply([0, 0, 1])  # quaternions.as_rotvec()

            # Write them and remove the leftover column
            recording.sensor_frame.loc[filled_rows, quaternion_cols[:3]] = vector
            recording.sensor_frame.drop(quaternion_cols[3], axis=1, inplace=True)

    return recordings

def convert_euler_to_vector(recordings: List[Recording]) -> List[Recording]:
    recordings = deepcopy(recordings)

    # Iterate over all recordings
    for idx, recording in enumerate(recordings):
        if idx % 10 == 0:
            print(f"Converting euler to vectors for recording {idx}")

        # Iterate over all sensors - we need to convert this many quaternions
        for sensor_suffix in settings.SENSOR_SUFFIX_ORDER:
            # Build the column names, that we need to select for the quaternion
            euler_cols = [f"Quat_{axis}_{sensor_suffix}" for axis in ['W', 'X', 'Y']]

            # To not get an error when we try to read the quaternion, select only not nan rows
            filled_rows = recording.sensor_frame[euler_cols[0]].notnull()

            # Read all quaternions simultaneously
            quaternions = R.from_euler('zxy', recording.sensor_frame.loc[filled_rows, euler_cols])

            # Convert to euler angles
            vector = quaternions.apply([0, 0, 1])  # quaternions.as_rotvec()

            # Write them and remove the leftover column
            recording.sensor_frame.loc[filled_rows, euler_cols] = vector

    return recordings


def convert_euler_to_velocity(recordings: List[Recording]) -> List[Recording]:
    recordings = deepcopy(recordings)

    # Iterate over all recordings
    for idx, recording in enumerate(recordings):
        if idx % 10 == 0:
            print(f"Converting to velocity for recording {idx}", end='\r')

        # Iterate over all sensors - we need to convert this many quaternions
        for sensor_suffix in settings.SENSOR_SUFFIX_ORDER:
            # Build the column names, that we need to select for the quaternion
            columns = [f"Quat_{axis}_{sensor_suffix}" for axis in ['W', 'X', 'Y']]

            # To not get an error when we try to read the quaternion, select only not nan rows
            filled_rows = recording.sensor_frame[columns[0]].notnull()

            angles = recording.sensor_frame.loc[filled_rows, columns].values
            times = recording.time_frame.loc[filled_rows].values
            rotations = R.from_euler('zyx', angles)
            spline = RotationSpline(times, rotations)

            # Convert to euler angles
            velocity = spline(times, 1)

            # Write them and remove the leftover column
            recording.sensor_frame.loc[filled_rows, columns[:3]] = velocity

    print()
    return recordings


def convert_quat_to_velocity(recordings: List[Recording]) -> List[Recording]:
    recordings = deepcopy(recordings)

    # Iterate over all recordings
    for idx, recording in enumerate(recordings):
        if idx % 10 == 0:
            print(f"Converting to velocity for recording {idx}", end='\r')

        # Iterate over all sensors - we need to convert this many quaternions
        for sensor_suffix in settings.SENSOR_SUFFIX_ORDER:
            # Build the column names, that we need to select for the quaternion
            columns = [f"Quat_{axis}_{sensor_suffix}" for axis in ['W', 'X', 'Y', 'Z']]

            # To not get an error when we try to read the quaternion, select only not nan rows
            filled_rows = recording.sensor_frame[columns[0]].notnull()

            quaternions = recording.sensor_frame.loc[filled_rows, columns].values
            times = recording.time_frame.loc[filled_rows].values
            rotations = R.from_quat(quaternions)
            spline = RotationSpline(times, rotations)

            # Convert to euler angles
            velocity = spline(times, 1)

            # Write them and remove the leftover column
            recording.sensor_frame.loc[filled_rows, columns[:3]] = velocity
            recording.sensor_frame.drop(columns[3], axis=1, inplace=True)

    print()
    return recordings


def convert_recording_speed(recordings: List[Recording], multiplier: float) -> List[Recording]:
    recordings = deepcopy(recordings)

    # Iterate over all recordings
    for idx, recording in enumerate(recordings):
        if idx % 10 == 0:
            print(f"Changing speed for recording {idx}", end='\r')
        # Merge the time and sensor frames
        combined_frame = pd.merge(recording.time_frame, recording.sensor_frame, left_index=True, right_index=True)
        combined_frame.resample('12ms').interpolate(method='spline')

    print()
    return recordings
