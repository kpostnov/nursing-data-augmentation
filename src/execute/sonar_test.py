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
import numpy as np
import matplotlib.pyplot as plt


# Kann in Recording (sowie evtl. mapping int zu labels)
def get_index_map_by_column(recording: Recording, columns: 'list[str]'):
    return {recording.sensor_frame.columns.get_loc(column): column for column in columns}

# Load data
recordings = load_recordings("D:\dataset\ML Prototype Recordings\without_null_activities", limit=1)

# random.seed(1678978086101)
# random.shuffle(recordings)

# save_recordings(recordings, "D:\dataset\ML Prototype Recordings\without_null_activities")

# recordings = interpolate_linear(recordings)
# Calculate magnitude
for rec in recordings:
    rec.activities = rec.activities.map(settings.ACTIVITIES)

windowizer = Windowizer(100, 100, Windowizer.windowize_sliding)
windows = windowizer.windowize_sliding(recordings)
# columns = get_index_map_by_column(recordings[0], ['dv[1]_LW', 'dv[2]_LW', 'dv[3]_LW'])
# X_train, y_train = windowizer.windowize_convert(recordings)

# X_train = np.squeeze(X_train, -1)


fig, ax = plt.subplots(nrows=2)
plot_pca(windows[:200], ax=ax[0])
plot_pca(windows[200:400], ax=ax[1])
plt.show()
