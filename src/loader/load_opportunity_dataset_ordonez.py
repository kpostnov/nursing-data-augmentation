import itertools
import os

import numpy as np
import pandas as pd

from utils.Recording import Recording
import utils.settings as settings


def load_opportunity_dataset_ordonez(opportunity_dataset_path: str) -> tuple("list[Recording]"):
    """
    Returns a list of Recordings from the opportunity dataset.
    Returns only those recordings that were mentioned in Ordonez et al. (2015)
    """
    print("Reading the opportunity dataset...")
    opportunity_dataset_path += "/dataset"

    recording_files_train = ['S1-ADL1.dat', 'S1-ADL2.dat', 'S1-ADL3.dat', 'S1-ADL4.dat', 
        'S1-ADL5.dat', 'S1-Drill.dat', 'S2-ADL1.dat', 'S2-ADL2.dat', 'S2-ADL3.dat', 
        'S2-Drill.dat', 'S3-ADL1.dat', 'S3-ADL2.dat', 
        'S3-ADL3.dat', 'S3-Drill.dat',]
    recording_files_test = ['S2-ADL4.dat', 'S2-ADL5.dat', 'S3-ADL4.dat', 'S3-ADL5.dat',]

    selected_feature_names = [
        "Acc-RKN^-accX",
        "Acc-RKN^-accY",
        "Acc-RKN^-accZ",
        "Acc-HIP-accX",
        "Acc-HIP-accY",
        "Acc-HIP-accZ",
        "Acc-LUA^-accX",
        "Acc-LUA^-accY",
        "Acc-LUA^-accZ",
        "Acc-RUA_-accX",
        "Acc-RUA_-accY",
        "Acc-RUA_-accZ",
        "Acc-LH-accX",
        "Acc-LH-accY",
        "Acc-LH-accZ",
        "Acc-BACK-accX",
        "Acc-BACK-accY",
        "Acc-BACK-accZ",
        "Acc-RKN_-accX",
        "Acc-RKN_-accY",
        "Acc-RKN_-accZ",
        "Acc-RWR-accX",
        "Acc-RWR-accY",
        "Acc-RWR-accZ",
        "Acc-RUA^-accX",
        "Acc-RUA^-accY",
        "Acc-RUA^-accZ",
        "Acc-LUA_-accX",
        "Acc-LUA_-accY",
        "Acc-LUA_-accZ",
        "Acc-LWR-accX",
        "Acc-LWR-accY",
        "Acc-LWR-accZ",
        "Acc-RH-accX",
        "Acc-RH-accY",
        "Acc-RH-accZ",
        "IMU-BACK-accX",
        "IMU-BACK-accY",
        "IMU-BACK-accZ",
        "IMU-BACK-gyroX",
        "IMU-BACK-gyroY",
        "IMU-BACK-gyroZ",
        "IMU-BACK-magneticX",
        "IMU-BACK-magneticY",
        "IMU-BACK-magneticZ",
        "IMU-RUA-accX",
        "IMU-RUA-accY",
        "IMU-RUA-accZ",
        "IMU-RUA-gyroX",
        "IMU-RUA-gyroY",
        "IMU-RUA-gyroZ",
        "IMU-RUA-magneticX",
        "IMU-RUA-magneticY",
        "IMU-RUA-magneticZ",
        "IMU-RLA-accX",
        "IMU-RLA-accY",
        "IMU-RLA-accZ",
        "IMU-RLA-gyroX",
        "IMU-RLA-gyroY",
        "IMU-RLA-gyroZ",
        "IMU-RLA-magneticX",
        "IMU-RLA-magneticY",
        "IMU-RLA-magneticZ",
        "IMU-LUA-accX",
        "IMU-LUA-accY",
        "IMU-LUA-accZ",
        "IMU-LUA-gyroX",
        "IMU-LUA-gyroY",
        "IMU-LUA-gyroZ",
        "IMU-LUA-magneticX",
        "IMU-LUA-magneticY",
        "IMU-LUA-magneticZ",
        "IMU-LLA-accX",
        "IMU-LLA-accY",
        "IMU-LLA-accZ",
        "IMU-LLA-gyroX",
        "IMU-LLA-gyroY",
        "IMU-LLA-gyroZ",
        "IMU-LLA-magneticX",
        "IMU-LLA-magneticY",
        "IMU-LLA-magneticZ",
        "IMU-L-SHOE-EuX",
        "IMU-L-SHOE-EuY",
        "IMU-L-SHOE-EuZ",
        "IMU-L-SHOE-Nav_Ax",
        "IMU-L-SHOE-Nav_Ay",
        "IMU-L-SHOE-Nav_Az",
        "IMU-L-SHOE-Body_Ax",
        "IMU-L-SHOE-Body_Ay",
        "IMU-L-SHOE-Body_Az",
        "IMU-L-SHOE-AngVelBodyFrameX",
        "IMU-L-SHOE-AngVelBodyFrameY",
        "IMU-L-SHOE-AngVelBodyFrameZ",
        "IMU-L-SHOE-AngVelNavFrameX",
        "IMU-L-SHOE-AngVelNavFrameY",
        "IMU-L-SHOE-AngVelNavFrameZ",
        "IMU-L-SHOE-Compass",
        "IMU-R-SHOE-EuX",
        "IMU-R-SHOE-EuY",
        "IMU-R-SHOE-EuZ",
        "IMU-R-SHOE-Nav_Ax",
        "IMU-R-SHOE-Nav_Ay",
        "IMU-R-SHOE-Nav_Az",
        "IMU-R-SHOE-Body_Ax",
        "IMU-R-SHOE-Body_Ay",
        "IMU-R-SHOE-Body_Az",
        "IMU-R-SHOE-AngVelBodyFrameX",
        "IMU-R-SHOE-AngVelBodyFrameY",
        "IMU-R-SHOE-AngVelBodyFrameZ",
        "IMU-R-SHOE-AngVelNavFrameX",
        "IMU-R-SHOE-AngVelNavFrameY",
        "IMU-R-SHOE-AngVelNavFrameZ",
        "IMU-R-SHOE-Compass",
    ]

    print(f"Selected features (n_features: {len(selected_feature_names)}):\n", "\n".join(
        ["\t" + str(feature_name) for feature_name in selected_feature_names]))

    # Get column names
    col_names = []
    current_directory_path = os.path.dirname(os.path.realpath(__file__))
    opportunity_col_names_path = os.path.join(current_directory_path, "opportunity_col_names")
    with open(opportunity_col_names_path, "r") as file:
        lines = file.read().splitlines()
        for line in lines:
            col_names.append(line)

    recordings_train = []
    for file_name in recording_files_train:
        sub = file_name[1]
        file_path = os.path.join(os.path.dirname(
            __file__), opportunity_dataset_path, file_name)
        print(f"Reading {file_path} ...")
        file_df = pd.read_csv(file_path, delimiter=" ", header=None)
        file_df.columns = col_names  # give them the real column names

        recordings_train.append(Recording(
            sensor_frame=file_df.loc[:, selected_feature_names],
            time_frame=file_df.loc[:, 'MILLISEC'],
            activities=file_df.loc[:, 'ML_Both_Arms'].map(
                lambda label: settings.activity_initial_num_to_activity_idx[label]
            ),
            subject=f"{sub}",
        ))

    recordings_test = []
    for file_name in recording_files_test:
        sub = file_name[1]
        file_path = os.path.join(os.path.dirname(
            __file__), opportunity_dataset_path, file_name)
        print(f"Reading {file_path} ...")
        file_df = pd.read_csv(file_path, delimiter=" ", header=None)
        file_df.columns = col_names  # give them the real column names

        recordings_test.append(Recording(
            sensor_frame=file_df.loc[:, selected_feature_names],
            time_frame=file_df.loc[:, 'MILLISEC'],
            activities=file_df.loc[:, 'ML_Both_Arms'].map(
                lambda label: settings.activity_initial_num_to_activity_idx[label]
            ),
            subject=f"{sub}",
        ))

    print(f"\n => Total {len(recordings_train)} training recordings read.")
    print(f" => Total {len(recordings_test)} test recordings read.")

    return (recordings_train, recordings_test)
