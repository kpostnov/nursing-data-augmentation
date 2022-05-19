import os
import random

from evaluation.conf_matrix import create_conf_matrix
from evaluation.metrics import accuracy, f_score
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
from utils.convert_dataset import convert_to_relative_sensor_data
from utils.Windowizer import Windowizer
from models.AdaptedDeepConvLSTM import AdaptedDeepConvLSTM
from visualization.visualize import plot_pca
from loader.preprocessing import interpolate_linear, normalize_standardscaler, preprocess
from scripts.plot_people import count_windows_per_activity, count_person_length, count_activity_length, count_activities_per_person, count_recordings_per_person, count_windows_per_activity_per_person
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# 900 steps --> 48 %

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

# rare = list(filter(lambda x: x.sensor_frame.shape[0] > 0, rare))

# save_recordings(rare, "/dhc/groups/bp2021ba1/data/rare_data")

'''
# Create plot
WINDOW_SIZE = 900
STRIDE_SIZE = 600
recordings = load_recordings("D:\dataset\ML Prototype Recordings\without_null_activities", limit=10)

for rec in recordings:
    rec.activities = rec.activities.map(lambda label: settings.ACTIVITIES[label])

# Preprocessing
recordings = preprocess(recordings, methods=[
    interpolate_linear,
    normalize_standardscaler
])
'''

WINDOW_SIZE = 600
STRIDE_SIZE = 600

recordings = load_recordings(settings.sonar_lab_dataset_path)

values1 = count_person_length(recordings)
values2 = count_activity_length(recordings)
values3 = count_activities_per_person(recordings)
values4 = count_recordings_per_person(recordings)

windowizer = Windowizer(WINDOW_SIZE, STRIDE_SIZE, Windowizer.windowize_jumping)
windows = windowizer.windowize_windowize(recordings)

values5 = count_windows_per_activity(windows, WINDOW_SIZE)
values6 = count_windows_per_activity_per_person(windows, WINDOW_SIZE)

plt.figure()
values1.to_csv("person_length.csv")
values1.plot.bar(figsize=(22,16))
plt.title("Person length")
plt.xlabel("x")
plt.ylabel("y")
plt.savefig('person_length')
plt.close()

plt.figure()
values2.to_csv("activity_length.csv")
values2.plot.bar(figsize=(22,16))
plt.title("Activity length")
plt.xlabel("x")
plt.ylabel("y")
plt.savefig('activity_length.png')
plt.close()

plt.figure()
values3.to_csv("activities_per_person.csv")
values3.plot.bar(figsize=(22,16))
plt.title("Activities per person")
plt.xlabel("x")
plt.ylabel("y")
plt.savefig('activities_per_person.png')
plt.close()

plt.figure()
values4.to_csv("recordings_per_person.csv")
values4.plot.bar(figsize=(22,16))
plt.title("Recordings per person")
plt.xlabel("x")
plt.ylabel("y")
plt.savefig('recordings_per_person.png')
plt.close()

plt.figure()
values5.to_csv("windows_per_activity.csv")
values5.plot.bar(figsize=(22,16))
plt.title("Windows per activity")
plt.xlabel("x")
plt.ylabel("y")
plt.savefig('windows_per_activity.png')
plt.close()

plt.figure()
values6.to_csv("windows_per_activity_per_person.csv")
values6.plot.bar(figsize=(22,16))
plt.title("Windows per activity per person")
plt.xlabel("x")
plt.ylabel("y")
plt.savefig('windows_per_activity_per_person.png')
plt.close()


# Pipeline
WINDOW_SIZE = 900
STRIDE_SIZE = 900

# Load data
recordings = load_recordings(settings.sonar_dataset_path, limit=10)

# recordings = filter_activities_negative(recordings, ['accessoires anlegen', 'haare waschen', 'aufwischen (staub)', 'föhnen'])
# leere Recordings Filter

# recordings = convert_to_relative_sensor_data(recordings)

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

# Windowize
windowizer = Windowizer(WINDOW_SIZE, STRIDE_SIZE, Windowizer.windowize_jumping)

# K Fold
accuracies = []
f_scores = []
k = 5
partition = int(len(recordings) / k)
for i in range(k):
    recordings_test = recordings[i*partition:(i+1)*partition]
    recordings_train = recordings[:i*partition] + recordings[(i+1)*partition:]

    X_train, y_train = windowizer.windowize_convert(recordings_train)

    # Init, Train
    model = AdaptedDeepConvLSTM(
        window_size=WINDOW_SIZE,
        stride_size=STRIDE_SIZE,
        n_features=recordings[0].sensor_frame.shape[1],
        n_outputs=15,
        verbose=1,
        n_epochs=10)
    model.fit(X_train=X_train, y_train=y_train)

    # Test, Evaluate
    X_test, y_test_true = windowizer.windowize_convert(recordings_test)
    y_test_pred = model.predict(X_test)

    print(f"Accuracy: {accuracy(y_test_pred, y_test_true)}")
    print(f"F1-score: {f_score(y_test_pred, y_test_true)}")
    accuracies.append(accuracy(y_test_pred, y_test_true))
    f_scores.append(f_score(y_test_pred, y_test_true))

print(f"Accuracy: {accuracies}")
print(f"F1-score: {f_scores}")
# Create Folder, save model export and evaluations there
# experiment_folder_path = new_saved_experiment_folder('pamap_deepConv')

# Export model
# create_conf_matrix(experiment_folder_path, y_test_pred, y_test_true)
# create_text_metrics(experiment_folder_path, y_test_pred, y_test_true, [accuracy, f_score])
# save_model_configuration(experiment_folder_path, model)
