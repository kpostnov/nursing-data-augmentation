import random
from evaluation.conf_matrix import create_conf_matrix
from models.JensModel import JensModel
from utils.folder_operations import new_saved_experiment_folder
import utils.settings as settings
from loader.load_dataset import load_dataset
import pandas as pd
import numpy as np


settings.init('sonar')
random.seed(1678978086101)

recordings = load_dataset('/Users/franz/Projects/BP/mach_kaputt', limit=40)

sensors = recordings[0].sensor_frame.shape[1]
activities = len(settings.LABELS)


def map_recording_activities_to_id(recording):
    """
    Converts the string labels of one recording to integers"
    """

    recording.activities = pd.Series([settings.ACTIVITIES.get(activity) for activity in recording.activities])
    return recording


# Convert the string labels of all recordings to integers
recordings = [map_recording_activities_to_id(recording) for recording in recordings]

n_outputs = len(settings.ACTIVITIES_ID_TO_NAME)

model = JensModel(window_size=25, n_features=sensors, n_outputs=n_outputs)
model.windowize_convert_fit(recordings)

X_test, y_test_true = model.windowize_convert(recordings)
y_test_pred = model.predict(X_test)

# Create Folder, save model export and evaluations there
experiment_folder_path = new_saved_experiment_folder('opportunity_jens_cnn')  # create folder to store results

correct_counter = 0
for pred, true in zip(y_test_pred, y_test_true):
    pred = np.argmax(pred)
    true = np.argmax(true)
    # print(f"Predicted: {pred} - True: {true}")
    if pred == true:
        correct_counter += 1

print("-----------------------------------------------------")
print(f"Correct: {correct_counter} of {len(y_test_pred)}")

# model.export(experiment_folder_path)
# create_conf_matrix(experiment_folder_path, y_test_pred, y_test_true)
# create_text_metrics(experiment_folder_path, y_test_pred, y_test_true, [accuracy])
