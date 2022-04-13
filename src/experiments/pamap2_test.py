import os
import random

from numpy import record
from loader.load_pamap2_dataset import load_pamap2_dataset
import utils.settings as settings
from utils.array_operations import split_list_by_percentage
from models.JensModel import JensModel
from utils.folder_operations import new_saved_experiment_folder
from evaluation.conf_matrix import create_conf_matrix
from evaluation.text_metrics import create_text_metrics
from evaluation.metrics import accuracy, f1_score


settings.init()

# Load data
recordings = load_pamap2_dataset(settings.pamap2_dataset_path)

for recording in recordings:
    print(recording.sensor_frame.shape)
    print(recording.sensor_frame.head())
    print(recording.activities.value_counts())

