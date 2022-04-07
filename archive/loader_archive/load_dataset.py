import os
import json
from utils.file_functions import get_subfolder_names
from archive.loader_archive.XSensRecordingReader import XSensRecordingReader
import pandas as pd
import utils.settings as settings
from utils.Recording import Recording


def load_dataset(dataset_path: str) -> 'list[Recording]':
    """
    Returns a list of the raw recordings (activities, subjects included, None values) (different representaion of dataset)
    directory structure bias! not shuffled!


    This function knows the structure of the XSens dataset.
    It will call the create_recording_frame function on every recording folder.

    bp/data
        dataset_01
            recording_01
                metadata.json
                sensor_01.csv
                sensor_02.csv

            recording_01
                metadata.json
                sensor_01.csv
                sensor_02.csv

        data_set_02
            ...

    """

    if not os.path.exists(dataset_path):
        raise Exception("The dataset_path does not exist")

    recordings: list[Recording] = []

    # recording
    recording_folder_names = get_subfolder_names(dataset_path)
    for recording_folder_name in recording_folder_names:
        recording_folder_path = os.path.join(dataset_path, recording_folder_name)
        subject_folder_name = get_subject_folder_name(recording_folder_path)
        recordings.append(create_recording(recording_folder_path, subject_folder_name))

    return recordings


def get_subject_folder_name(recording_folder_path: str) -> str:
    with open(recording_folder_path + os.path.sep + 'metadata.json', 'r') as f:
        data = json.load(f)
    return data['person']


def get_activity_dataframe(time_frame, recording_folder_path: str) -> pd.DataFrame:
    with open(recording_folder_path + os.path.sep + 'metadata.json', 'r') as f:
        data = json.load(f)
    activities = data['activities']
    arr = time_frame.to_frame()
    arr['activities'] = ""
    activity = pd.Series([], dtype=str)
    counter = 1

    # to microseconds
    first_activity_timestamp = int(activities[0]["timeStarted"]) * 1000

    # normalize next activity timestamp (from metadata) by
    # substracting with first activity timestamp (from metadata)
    # and adding that to SampleFineTime (.csv) of first column
    # in microseconds
    next_activity_timestamp = int(activities[1]["timeStarted"]) * 1000 - first_activity_timestamp + arr.iloc[0]['SampleTimeFine']
    for idx in range(len(arr)):
        if(len(activities) <= counter):
            activity = pd.concat([activity, pd.Series([activities[counter-1]["label"]])])
        else:

            if(arr.iloc[idx]['SampleTimeFine'] < next_activity_timestamp):
                activity = pd.concat([activity, pd.Series([activities[counter-1]["label"]])])
            else:
                counter += 1
                if(len(activities) > counter):
                    next_activity_timestamp = int(activities[counter]["timeStarted"]) * 1000 - first_activity_timestamp + arr.iloc[0]['SampleTimeFine']
                activity = pd.concat([activity, pd.Series([activities[counter-1]["label"]])])
    return activity


def create_recording(recording_folder_path: str, subject: str) -> Recording:
    """
    Returns a recording
    Gets a XSens recorind folder path, loops over sensor files, concatenates them, adds activity and subject, returns a recording
    """

    raw_recording_frame = XSensRecordingReader.get_recording_frame(recording_folder_path)

    time_column_name = 'SampleTimeFine'
    time_frame = raw_recording_frame[time_column_name]

    activity = get_activity_dataframe(time_frame, recording_folder_path)

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
