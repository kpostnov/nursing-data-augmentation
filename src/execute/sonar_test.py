import os
import random

from numpy import rec
from loader.load_dataset import load_dataset
from utils import Recording
from utils.save_all_recordings import save_all_recordings
from utils.cache_recordings import save_recordings, load_recordings
import utils.settings as settings
from utils.Windowizer import Windowizer
from models.DeepConvLSTM import DeepConvLSTM
from visualization.visualize import plot_pca
from loader.preprocessing import interpolate_linear



def get_index_map_by_column(recording: Recording, columns: 'list[str]'):
    return {recording.sensor_frame.columns.get_loc(column): column for column in columns}

# Load data
recordings = load_recordings("D:\dataset\ML Prototype Recordings\without_null_activities", limit=1)

# random.seed(1678978086101)
# random.shuffle(recordings)

# save_recordings(recordings, "D:\dataset\ML Prototype Recordings\without_null_activities")

recordings = interpolate_linear(recordings)
# Calculate magnitude
for rec in recordings:
    rec.activities = rec.activities.map(settings.ACTIVITIES)
    
    dv_x = rec.sensor_frame['dv[1]_LW'] ** 2
    dv_y = rec.sensor_frame['dv[2]_LW'] ** 2
    dv_z = rec.sensor_frame['dv[3]_LW'] ** 2
    rec.sensor_frame['magnitude'] = (dv_x + dv_y + dv_z) ** 0.5

windowizer = Windowizer(100, 100, Windowizer.windowize_sliding)
windows = windowizer.windowize_sliding(recordings)
columns = get_index_map_by_column(recordings[0], ['magnitude', 'dv[1]_LW', 'dv[2]_LW', 'dv[3]_LW'])


plot_pca(windows, columns_to_plot=columns)
