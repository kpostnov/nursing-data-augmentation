from multiprocessing import Pool
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

        filename = str(index) + '_' + recording.subject + '.csv'
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

    recording_files = sorted(recording_files, key=lambda file: int(file.split('_')[0]))
    recording_files = list(map(lambda file_name: os.path.join(path, file_name), recording_files))

    pool = Pool()
    recordings = pool.imap_unordered(read_recording_from_csv, enumerate(recording_files), 10)
    pool.close()
    pool.join()
    recordings = list(recordings)

    print(f'Loaded {len(recordings)} recordings from {path}')

    return recordings


def read_recording_from_csv(data: 'tuple[int, str]') -> Recording:
    index, file = data

    print(f'Loading recording {file}, {index + 1}')

    recording_dataframe = pd.read_csv(file)
    time_frame = recording_dataframe.loc[:, 'SampleTimeFine']
    activities = recording_dataframe.loc[:, 'activity']
    sensor_frame = recording_dataframe.loc[:, recording_dataframe.columns.difference(['SampleTimeFine', 'activity'])]
    filename = os.path.basename(file)
    subject = filename.split('_')[1].split('.')[0]

    return Recording(sensor_frame, time_frame, activities, subject)
