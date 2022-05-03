import os
import pandas as pd

from datatypes.Recording import Recording
import utils.settings as settings


def load_nursing_field_dataset(nursing_dataset_path: str) -> "list[Recording]":
    """
    Returns a list of Recordings from the NURSING_2020 dataset (field data). 
    """
    nursing_dataset_path += "/Field"

    label_file = "field_label_train.csv"
    training_files = os.listdir(os.path.join(
        os.path.dirname(__file__), nursing_dataset_path))

    training_files.remove(label_file)

    # training_files = ["raw_field_acc_user04.csv"]
    training_files = []

    recordings = []

    label_df = pd.read_csv(os.path.join(os.path.dirname(
        __file__), nursing_dataset_path, label_file))

    label_df = label_df.sort_values(by=["user_id", "start"])

    print(label_df.head())

    for file in training_files:
        print(f"Reading {file}")
        subject = file.split(".")[0][-2:]

        df = pd.read_csv(os.path.join(os.path.dirname(
            __file__), nursing_dataset_path, file))

        df = df.sort_values(by=["datetime"])

        recordings.append(Recording(
            sensor_frame=df.loc[:, ['x', 'y', 'z']],
            time_frame=df.loc[:, "datetime"],

        ))

    return recordings


def load_nursing_lab_dataset(nursing_dataset_path: str) -> "list[Recording]":
    """
    Returns a list of Recordings from the NURSING_2020 dataset (lab data). 
    """
    nursing_dataset_path += "/Lab"
    label_file = "labels_lab_2users.csv"
    training_file = "bigact_raw_lab_acc.csv"

    recordings = []

    return recordings


def load_nursing_train_dataset(nursing_dataset_path: str) -> "list[Recording]":
    """
    Returns a list of Recordings from the NURSING_2020 dataset (all training data). 
    """
    print("Reading the NURSING dataset...")

    recordings = []
    recordings += load_nursing_field_dataset(nursing_dataset_path)
    recordings += load_nursing_lab_dataset(nursing_dataset_path)

    return recordings


def load_nursing_test_dataset(nursing_dataset_path: str) -> "list[Recording]":
    """
    Returns a list of Recordings from the NURSING_2020 dataset (test data). 
    """
