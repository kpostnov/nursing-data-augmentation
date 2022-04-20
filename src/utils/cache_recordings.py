import os
import pandas as pd
from utils.Recording import Recording

def save_recordings(recordings: 'list[Recording]', path: str) -> None:
    """
    Save each recording to a csv file.
    """
    if not os.path.exists(path):
        os.makedirs(path)

    for (index, recording) in enumerate(recordings):
        print(f'Saving recording {index} / {len(recordings)}')

        recording.activities.index = recording.sensor_frame.index

        recording_dataframe = recording.sensor_frame.copy()
        recording_dataframe['SampleTimeFine'] = recording.time_frame
        recording_dataframe['activity'] = recording.activities

        filename = index + '_' + recording.subject + '_' + str(recording_dataframe.iloc[0, 0]) + '.csv'
        recording_dataframe.to_csv(os.path.join(path, filename), index=False)

    print('Saved recordings to ' + path)


def load_recordings(path: str, limit: int = None) -> 'list[Recording]':
    """
    Load the recordings from a folder containing csv files.
    """
    recordings = []

    recording_files = os.listdir(path)
    recording_files = list(filter(lambda file: file.endswith('.csv'), recording_files))
    
    if limit is not None:
        recording_files = recording_files[:limit]
        recording_files.sort()

    for (index, file) in enumerate(recording_files):
        print(f'Loading recording {file}, {index} / {len(recording_files)}')

        recording_dataframe = pd.read_csv(os.path.join(path, file))
        time_frame = recording_dataframe.loc[:, 'SampleTimeFine']
        activities = recording_dataframe.loc[:, 'activity']
        sensor_frame = recording_dataframe.loc[:, 
            recording_dataframe.columns.difference(['SampleTimeFine', 'activity'])]
        subject = file.split('_')[0]

        recordings.append(Recording(sensor_frame, time_frame, activities, subject))

    print(f'Loaded {len(recordings)} recordings from {path}')
    
    return recordings
