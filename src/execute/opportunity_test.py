import random
from tkinter.tix import WINDOW
from evaluation.conf_matrix import create_conf_matrix
from evaluation.metrics import accuracy
from evaluation.text_metrics import create_text_metrics
from loader.preprocessing import ordonez_preprocess
from loader.load_opportunity_dataset_ordonez import load_opportunity_dataset_ordonez
from models.DeepConvLSTM import DeepConvLSTM
from utils.Windowizer import Windowizer
from utils.folder_operations import new_saved_experiment_folder
import utils.settings as settings

# Test accuracy: 89%

WINDOW_SIZE = 20
STRIDE_SIZE = 10

# Load data
(recordings_train, recordings_test) = load_opportunity_dataset_ordonez(settings.opportunity_dataset_path)

random.seed(1678978086101)

# Preprocessing and Test Train Split
recordings = ordonez_preprocess(recordings_train + recordings_test)
recordings_train = recordings[:len(recordings_train)]
recordings_test = recordings[len(recordings_train):]

# Windowize
windowizer = Windowizer(WINDOW_SIZE, STRIDE_SIZE, Windowizer.windowize_sliding)
X_train, y_train = windowizer.windowize_convert(recordings_train)

# Init, Train
model = DeepConvLSTM(
    window_size=WINDOW_SIZE,
    stride_size=STRIDE_SIZE,
    n_features=recordings[0].sensor_frame.shape[1],
    n_outputs=18,
    verbose=1,
    n_epochs=15)
model.fit(X_train=X_train, y_train=y_train)

# Test, Evaluate
X_test, y_test_true = windowizer.windowize_convert(recordings_test)
y_test_pred = model.predict(X_test)

# Create Folder, save model export and evaluations there
experiment_folder_path = new_saved_experiment_folder('opportunity_deepConv')

# Export model
model.export(experiment_folder_path)
create_conf_matrix(experiment_folder_path, y_test_pred, y_test_true)
create_text_metrics(experiment_folder_path, y_test_pred, y_test_true, [accuracy])
