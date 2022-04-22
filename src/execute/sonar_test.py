import os
import random
from loader.load_dataset import load_dataset
from utils.save_all_recordings import save_all_recordings
from utils.cache_recordings import save_recordings, load_recordings
import utils.settings as settings


# Load data
recordings = load_recordings("D:\dataset\ML Prototype Recordings\combined_dataset")

# random.seed(1678978086101)
# random.shuffle(recordings)

save_recordings(recordings, "D:\dataset\ML Prototype Recordings\without_null_activities")
