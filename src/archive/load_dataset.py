import os
from random import shuffle
from utils.file_functions import get_subfolder_names
from loader.XSensRecordingReader import XSensRecordingReader
import pandas as pd
import utils.settings as settings
from utils.Recording import Recording


def load_dataset(dataset_path: str) -> "list[Recording]":
    """
    Returns a list of the raw recordings (activities, subjects included, None values) (different representaion of dataset)
    directory structure bias! not shuffled!


    This function knows the structure of the XSens dataset.
    It will call the create_recording_frame function on every recording folder.

    bp/data
        dataset_01
            activity_01
                subject_01
                    recording_01
                        random_bullshit_folder
                        sensor_01.csv
                        sensor_02.csv
                        ...
                    recording_02
                        ...
                subject_02
                    ....
            activity_02
                ...
        data_set_02
            ...

    """

    if not os.path.exists(dataset_path):
        raise Exception("The dataset_path does not exist")

    recordings: list[Recording] = []

    # activity
    activity_folder_names = get_subfolder_names(dataset_path)
    for activity_folder_name in activity_folder_names:
        activity_folder_path = os.path.join(dataset_path, activity_folder_name)

        # subject
        subject_folder_names = get_subfolder_names(activity_folder_path)
        for subject_folder_name in subject_folder_names:
            subject_folder_path = os.path.join(
                activity_folder_path, subject_folder_name
            )

            # recording
            recording_folder_names = get_subfolder_names(subject_folder_path)
            for recording_folder_name in recording_folder_names:
                if recording_folder_name.startswith("_"):
                    continue
                recording_folder_path = os.path.join(
                    subject_folder_path, recording_folder_name
                )
                # print("Reading recording: {}".format(recording_folder_path))

                recordings.append(
                    create_recording(
                        recording_folder_path, activity_folder_name, subject_folder_name
                    )
                )

    return recordings


def create_recording(
    recording_folder_path: str, activity: str, subject: str
) -> Recording:
    """
    Returns a recording
    Gets a XSens recorind folder path, loops over sensor files, concatenates them, adds activity and subject, returns a recording
    """

    raw_recording_frame = XSensRecordingReader.get_recording_frame(
        recording_folder_path
    )

    time_column_name = "SampleTimeFine"
    time_frame = raw_recording_frame[time_column_name]

    sensor_frame = raw_recording_frame.drop([time_column_name], axis=1)
    sensor_frame = reorder_sensor_columns(sensor_frame)

    return Recording(sensor_frame, time_frame, activity, subject)


def reorder_sensor_columns(sensor_frame: pd.DataFrame) -> pd.DataFrame:
    """
    reorders according to global settings
    """

    column_suffix_dict = {}
    for column_name in sensor_frame.columns:
        ending = column_name[-2:]
        if ending in column_suffix_dict:
            column_suffix_dict[ending].append(column_name)
        else:
            column_suffix_dict[ending] = [column_name]

    # assert list(column_suffix_dict.keys()) == settings.SENSOR_SUFFIX_ORDER ... only same elements

    column_names_ordered = []
    for sensor_suffix in settings.SENSOR_SUFFIX_ORDER:
        column_names_ordered.extend(column_suffix_dict[sensor_suffix])

    return sensor_frame[column_names_ordered]
