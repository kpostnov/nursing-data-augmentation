import os
from loader.load_pamap2_dataset import load_pamap2_dataset
import utils.settings as settings


# Load data
recordings = load_pamap2_dataset(settings.pamap2_dataset_path)

for recording in recordings:
    print(recording.sensor_frame.shape)
    print(recording.sensor_frame.head())
    print(recording.activities.value_counts())

