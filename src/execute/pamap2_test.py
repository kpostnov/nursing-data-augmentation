import os
import random
from evaluation.conf_matrix import create_conf_matrix
from evaluation.metrics import accuracy
from evaluation.text_metrics import create_text_metrics
from loader.Preprocessor import Preprocessor
from loader.load_opportunity_dataset import load_opportunity_dataset
from loader.load_pamap2_dataset import load_pamap2_dataset
from models.DeepConvLSTM import DeepConvLSTM
from utils.array_operations import split_list_by_percentage
from utils.folder_operations import new_saved_experiment_folder
import utils.settings as settings


# Load data
# recordings = load_pamap2_dataset(settings.pamap2_dataset_path)
recordings = load_opportunity_dataset(settings.opportunity_dataset_path)

random.seed(1678978086101)
random.shuffle(recordings)

# TODO: apply recording label filter functions

# Preprocessing
recordings = Preprocessor().jens_preprocess(recordings)

# TODO: save/ load preprocessed data

# Test Train Split
test_percentage = 0.3
recordings_train, recordings_test = split_list_by_percentage(recordings, test_percentage)

# Init, Train
model = DeepConvLSTM(window_size=25, n_features=recordings[0].sensor_frame.shape[1], n_outputs=6, verbose=1, n_epochs=10)
model.windowize_convert_fit(recordings_train)

# Test, Evaluate
# labels are always in vector format
X_test, y_test_true = model.windowize_convert(recordings_test)
y_test_pred = model.predict(X_test)

# Create Folder, save model export and evaluations there
experiment_folder_path = new_saved_experiment_folder('pamap_deepconv') # create folder to store results

model.export(experiment_folder_path) # opt: export model to folder
create_conf_matrix(experiment_folder_path, y_test_pred, y_test_true)
create_text_metrics(experiment_folder_path, y_test_pred, y_test_true, [accuracy])

