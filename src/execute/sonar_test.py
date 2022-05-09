import os
import random
from evaluation.conf_matrix import create_conf_matrix
from evaluation.metrics import accuracy
from evaluation.save_configuration import save_model_configuration
from evaluation.text_metrics import create_text_metrics

from loader.load_dataset import load_dataset
from datatypes.Recording import Recording
from utils.array_operations import split_list_by_percentage
from utils.filter_activities import filter_activities, filter_activities_negative
from utils.folder_operations import new_saved_experiment_folder
from utils.save_all_recordings import save_all_recordings
from utils.cache_recordings import save_recordings, load_recordings
import utils.settings as settings
from utils.Windowizer import Windowizer
from models.AdaptedDeepConvLSTM import AdaptedDeepConvLSTM
from visualization.visualize import plot_pca
from loader.preprocessing import interpolate_linear, normalize_standardscaler, preprocess
from scripts.plot_people import count_activities_per_person, count_recordings_per_person
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# Create datasets
# Load data
# recordings = load_recordings("D:\dataset\ML Prototype Recordings\without_null_activities", limit=3)


# activities_to_remove = ['föhnen', 'essen reichen', 'haare waschen', 'accessoires anlegen', 'aufwischen (staub)', 'medikamente stellen', 'küchenvorbereitung']
# subjects_to_remove = ['mathias', 'christine']

# reduced = filter_activities_negative(recordings, activities_to_remove)
# idx = [i for i in range(len(reduced)) if reduced[i].subject in subjects_to_remove]
# reduced = [reduced[i] for i in range(len(reduced)) if i not in idx]

# save_recordings(reduced, "/dhc/groups/bp2021ba1/data/reduced_data")


# rare_activities = ['föhnen', 'haare waschen', 'accessoires anlegen', 'aufwischen (staub)', 'haare kämmen', 'dokumentation', 'mundpflege']
# subjects_to_remove = ['christine']

# rare = filter_activities(recordings, rare_activities)
# idx = [i for i in range(len(rare)) if rare[i].subject in subjects_to_remove]
# rare = [rare[i] for i in range(len(rare)) if i not in idx]

# save_recordings(rare, "/dhc/groups/bp2021ba1/data/rare_data")


# Create plot
# recordings = load_recordings("/dhc/groups/bp2021ba1/data/filtered_dataset_without_null")

# values = count_recordings_per_person(recordings)

# values.to_csv("recordings_per_person.csv")
# values.plot.bar(figsize=(22,16))
# plt.title("Recordings per person")
# plt.xlabel("Person")
# plt.ylabel("Number of recordings")
# plt.savefig('recordings_per_person.png')


# Pipeline
WINDOW_SIZE = 600
STRIDE_SIZE = 600

# Load data
recordings = load_recordings("D:\dataset\ML Prototype Recordings\without_null_activities", limit=10)

random.seed(1678978086101)
random.shuffle(recordings)

# Activities to number
for rec in recordings:
    rec.activities = rec.activities.map(lambda label: settings.ACTIVITIES[label])

# Preprocessing
recordings = preprocess(recordings, methods=[
    interpolate_linear,
    normalize_standardscaler
])

# Test Train Split
test_percentage = 0.2
recordings_train, recordings_test = split_list_by_percentage(recordings, test_percentage)

# Windowize
windowizer = Windowizer(WINDOW_SIZE, STRIDE_SIZE, Windowizer.windowize_jumping)
X_train, y_train = windowizer.windowize_convert(recordings_train)

# Init, Train
model = AdaptedDeepConvLSTM(
    window_size=WINDOW_SIZE,
    stride_size=STRIDE_SIZE,
    n_features=recordings[0].sensor_frame.shape[1],
    n_outputs=60,
    verbose=1,
    n_epochs=20)
model.fit(X_train=X_train, y_train=y_train)

# Test, Evaluate
X_test, y_test_true = windowizer.windowize_convert(recordings_test)
y_test_pred = model.predict(X_test)
print(accuracy(y_test_pred, y_test_true))

# Create Folder, save model export and evaluations there
experiment_folder_path = new_saved_experiment_folder('pamap_deepConv')

# Export model
create_conf_matrix(experiment_folder_path, y_test_pred, y_test_true)
create_text_metrics(experiment_folder_path, y_test_pred, y_test_true, [accuracy])
save_model_configuration(experiment_folder_path, model)
