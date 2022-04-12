import os
import pandas as pd

from utils.Recording import Recording
import utils.settings as settings


def load_pamap2_dataset(
        pamap2_dataset_path: str,
        include_heart_rate: bool = True,
        lin_interpolate: bool = False) -> "list[Recording]":
    """
    Returns a list of Recordings from the PAMAP2 dataset. 
    Each Recording corresponds to one subject.
    NOTE: Returns only those subjects and recordings that were mentioned in the paper. (Hoelzmann et al., 2021)
    """
    print("Reading the PAMAP2 dataset")
    pamap2_dataset_path += "/Protocol"
    subject_ids = range(1, 9)

    # acc_x_6, acc_y_6, acc_z_6 will be filtered out as they contain faulty data
    col_names = [
        "timestamp",
        "activity_id",
        "heart_rate",
        "temperature",
        "acc_x",
        "acc_y",
        "acc_z",
        "acc_x_6",
        "acc_y_6",
        "acc_z_6",
        "gyro_x",
        "gyro_y",
        "gyro_z",
        "mag_x",
        "mag_y",
        "mag_z"
    ]

    recordings = []
    for subject_id in subject_ids:
        file_name = f"subject10{subject_id}.dat"
        file_path = os.path.join(os.path.dirname(__file__), pamap2_dataset_path, file_name)

        print(f"Reading {file_name}")
        df = pd.read_csv(file_path, sep=" ", header=None)

        df = df.iloc[:, :16]
        df.columns = col_names
        df = df.drop(columns=["acc_x_6", "acc_y_6", "acc_z_6"])

        df = filter_data_frame(df, include_heart_rate, lin_interpolate)

        recordings.append(Recording(
            sensor_frame=df.iloc[:, 2:],
            time_frame=df.loc[:, "timestamp"],
            activities=df.loc[:, "activity_id"].map(
                lambda label: settings.pamap2_activity_map[label]
            ),
            subject=str(subject_id)
        ))

    print(f"\n => Total {len(recordings)} recordings read.")
    return recordings


def filter_data_frame(df: pd.DataFrame, include_heart_rate: bool, lin_interpolate: bool) -> pd.DataFrame:
    """
    Filters the data frame according to the paper, 
    i.e. keep only needed activities (lying, sitting, standing, walking, vacuum cleaning, ironing),
    ffill missing values
    """

    if lin_interpolate:
        df = df.interpolate(method="linear")
    else:
        df = df.fillna(method="ffill")

    df = df[df.activity_id.isin([1, 2, 3, 4, 16, 17])]

    if not include_heart_rate:
        df = df.drop(columns=["heart_rate"])

    return df
