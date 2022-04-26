import os
import random
from evaluation.conf_matrix import create_conf_matrix
from evaluation.metrics import accuracy
from evaluation.text_metrics import create_text_metrics
from evaluation.save_configuration import save_model_configuration
from loader.preprocessing import pamap2_preprocess
from loader.load_pamap2_dataset import load_pamap2_dataset
from models.DeepConvLSTM import DeepConvLSTM
from utils.Windowizer import Windowizer
from utils.array_operations import split_list_by_percentage
from utils.folder_operations import new_saved_experiment_folder
import utils.settings as settings


WINDOW_SIZE = 100
STRIDE_SIZE = 100

# Load data
recordings = load_pamap2_dataset(settings.pamap2_dataset_path)

random.seed(1678978086101)
random.shuffle(recordings)

# Preprocessing
recordings = pamap2_preprocess(recordings)

# Test Train Split
test_percentage = 0.3
recordings_train, recordings_test = split_list_by_percentage(recordings, test_percentage)

# Windowize
windowizer = Windowizer(WINDOW_SIZE, STRIDE_SIZE, Windowizer.windowize_sliding)
X_train, y_train = windowizer.windowize_convert(recordings_train)

# Init, Train
model = DeepConvLSTM(
    window_size=WINDOW_SIZE,
    stride_size=STRIDE_SIZE,
    n_features=recordings[0].sensor_frame.shape[1],
    n_outputs=6,
    verbose=1,
    n_epochs=200)
model.fit(X_train=X_train, y_train=y_train)

# Test, Evaluate
X_test, y_test_true = windowizer.windowize_convert(recordings_test)
y_test_pred = model.predict(X_test)

# Create Folder, save model export and evaluations there
experiment_folder_path = new_saved_experiment_folder('pamap_deepConv')

# Export model
model.export(experiment_folder_path)
create_conf_matrix(experiment_folder_path, y_test_pred, y_test_true)
create_text_metrics(experiment_folder_path, y_test_pred, y_test_true, [accuracy])
save_model_configuration(experiment_folder_path, model)
