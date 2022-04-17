import os
import random
from evaluation.conf_matrix import create_conf_matrix
from evaluation.metrics import accuracy
from evaluation.text_metrics import create_text_metrics
from loader.Preprocessor import Preprocessor
from loader.load_opportunity_dataset import load_opportunity_dataset
from loader.load_opportunity_dataset_ordonez import load_opportunity_dataset_ordonez
from models.DeepConvLSTM import SlidingWindowDeepConvLSTM, JumpingWindowDeepConvLSTM
from utils.array_operations import split_list_by_percentage
from utils.folder_operations import new_saved_experiment_folder
import utils.settings as settings


# Load data
(recordings_train, recordings_test) = load_opportunity_dataset_ordonez(settings.opportunity_dataset_path)

random.seed(1678978086101)

# TODO: apply recording label filter functions
# TODO: save/ load preprocessed data

# Preprocessing and Test Train Split
recordings = Preprocessor().ordonez_preproceess(recordings_train + recordings_test)
recordings_train = recordings[:len(recordings_train)]
recordings_test = recordings[len(recordings_train):]

# Init, Train
model = SlidingWindowDeepConvLSTM(window_size=20, stride_size=10, n_features=recordings[0].sensor_frame.shape[1], n_outputs=18, verbose=1, n_epochs=15)
model.windowize_convert_fit(recordings_train)

# Test, Evaluate
X_test, y_test_true = model.windowize_convert(recordings_test)
y_test_pred = model.predict(X_test)

# Create Folder, save model export and evaluations there
experiment_folder_path = new_saved_experiment_folder('pamap_deepconv')

# Export model
model.export(experiment_folder_path)
create_conf_matrix(experiment_folder_path, y_test_pred, y_test_true)
create_text_metrics(experiment_folder_path, y_test_pred, y_test_true, [accuracy])
