import os
import random
from evaluation.conf_matrix import create_conf_matrix
from evaluation.metrics import accuracy
from evaluation.text_metrics import create_text_metrics
from evaluation.save_configuration import save_model_configuration
from loader.preprocessing import pamap2_preprocess
from loader.load_pamap2_dataset import load_pamap2_dataset
from models.DeepConvLSTM import SlidingWindowDeepConvLSTM, JumpingWindowDeepConvLSTM
from utils.array_operations import split_list_by_percentage
from utils.folder_operations import new_saved_experiment_folder
import utils.settings as settings
import numpy as np


# Load data
recordings = load_pamap2_dataset(settings.pamap2_dataset_path)

random.seed(1678978086101)
random.shuffle(recordings)

# Preprocessing (Interpolation)
recordings = pamap2_preprocess(recordings)

# Train M1 on whole dataset (no normalization)
model_m1 = SlidingWindowDeepConvLSTM(
    window_size=100,
    stride_size=100,
    n_features=recordings[0].sensor_frame.shape[1],
    n_outputs=6,
    verbose=1,
    n_epochs=200)
model_m1.windowize_convert_fit(recordings)

# LOSO-folds (alpha-dataset)
subject_ids = range(1, 9)
for subject_id in subject_ids:
    print("LOSO-fold without subject: {}".format(subject_id))

    # Remove recordings where recording.subject_id == subject_id
    alpha_subset = [recording for recording in recordings if recording.subject != str(subject_id)]
    validation_subset = [recording for recording in recordings if recording.subject == str(subject_id)]

    # Train alpha model on alpha_subset
    print("Training alpha model on alpha_subset")
    model_alpha = SlidingWindowDeepConvLSTM(
        window_size=100,
        stride_size=100,
        n_features=recordings[0].sensor_frame.shape[1],
        n_outputs=6,
        verbose=1,
        n_epochs=200)
    X_train, y_train = model_alpha.windowize_convert(alpha_subset)   
    model_alpha.fit(X_train, y_train)

    # Test alpha model on validation_subset
    X_test, y_test = model_alpha.windowize_convert(validation_subset)
    y_test_pred_model_alpha = model_alpha.predict(X_test)

    # Split recordings data activity wise for data augmentation
    print("Begin data augmentation")
    activities_one_hot_encoded = np.eye(6, 6)
    for row in activities_one_hot_encoded:
        # Get all indices in y_train where the one-hot-encoded row is equal to row
        activity_group_indices = np.nonzero(np.all(np.isclose(y_train, row), axis=1))[0]
        activity_group_X = X_train[activity_group_indices]
        activity_group_y = y_train[activity_group_indices]
    
    # TODO: Data Augmentation

    # TODO: Merge augmented data with alpha_subset --> beta_subset

    # TODO: Train beta model on beta_subset
    model_beta = JumpingWindowDeepConvLSTM(
        window_size=100,
        stride_size=100,
        n_features=recordings[0].sensor_frame.shape[1],
        n_outputs=6,
        verbose=1,
        n_epochs=200)
