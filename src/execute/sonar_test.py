import os
import random

from loader.load_dataset import load_dataset
from datatypes.Recording import Recording
from utils.save_all_recordings import save_all_recordings
from utils.cache_recordings import save_recordings, load_recordings
import utils.settings as settings
from utils.Windowizer import Windowizer
from models.DeepConvLSTM import DeepConvLSTM
from visualization.visualize import plot_pca
from loader.preprocessing import interpolate_linear
from scripts.plot_people import count_activities_per_person
import numpy as np
import matplotlib.pyplot as plt


# Kann in Recording (sowie evtl. mapping int zu labels)
def get_index_map_by_column(recording: Recording, columns: 'list[str]'):
    return {recording.sensor_frame.columns.get_loc(column): column for column in columns}

# Load data
recordings = load_recordings("D:\dataset\ML Prototype Recordings\without_null_activities", limit=3)

values = count_activities_per_person(recordings)
values.to_csv('actvities_per_person.csv')
values.plot.bar()
plt.savefig('activities_per_person.png')

# random.seed(1678978086101)
# random.shuffle(recordings)

# save_recordings(recordings, "D:\dataset\ML Prototype Recordings\without_null_activities")


