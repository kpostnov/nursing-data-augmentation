import os
import pandas as pd
from utils.Recording import Recording

def save_recordings(recordings: 'list[Recording]', path: str) -> None:
    """
    Save each recording to a csv file.
    """
    if not os.path.exists(path):
        os.makedirs(path)

    for recording in recordings:
        recording.activities.index = recording.sensor_frame.index

        recording_dataframe = recording.sensor_frame.copy()
        recording_dataframe['SampleTimeFine'] = recording.time_frame
        recording_dataframe['activity'] = recording.activities

        filename = recording.subject + '_' + str(recording_dataframe.iloc[0, 0]) + '.csv'
        recording_dataframe.to_csv(os.path.join(path, filename), index=False)

    print('Saved recordings to ' + path)


def load_recordings(path: str) -> 'list[Recording]':
    """
    Load the recordings from a folder containing csv files.
    """
    recordings = []
    for file in os.listdir(path):
        if file.endswith(".csv"):
            recording_dataframe = pd.read_csv(os.path.join(path, file))
            time_frame = recording_dataframe.loc[:, 'SampleTimeFine']
            activities = recording_dataframe.loc[:, 'activity']
            sensor_frame = recording_dataframe.loc[:, 
                recording_dataframe.columns.difference(['SampleTimeFine', 'activity'])]
            subject = file.split('_')[0]

            recordings.append(Recording(sensor_frame, time_frame, activities, subject))

    print(f'Loaded {len(recordings)} recordings from {path}')
    
    return recordings
