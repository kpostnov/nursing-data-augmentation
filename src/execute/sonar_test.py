import os
import random
from loader.Preprocessor import Preprocessor
from loader.load_dataset import load_dataset
from utils.save_all_recordings import save_all_recordings
from utils.cache_recordings import save_recordings
import utils.settings as settings


# Load data
recordings = load_dataset("D:\dataset\ML Prototype Recordings")

random.seed(1678978086101)
random.shuffle(recordings)

save_recordings(recordings, "D:\dataset\ML Prototype Recordings\combined_dataset")
