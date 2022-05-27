import os
import random
import numpy as np
import gc
import warnings
warnings.filterwarnings("ignore")

from loader.preprocessing import preprocess, normalize_standardscaler, interpolate_linear
from utils.cache_recordings import load_recordings
from utils.Windowizer import Windowizer
import utils.settings as settings

import TimeGAN.timegan as timegan


def start() -> None:
    WINDOW_SIZE = 300
    STRIDE_SIZE = 300
    
    # GAN Newtork parameters
    parameters = dict()
    parameters['module'] = 'gru'  # LSTM possible
    parameters['hidden_dim'] = 280  # Paper: 4 times the size of input features
    parameters['num_layer'] = 3
    parameters['iterations'] = 7500  # Paper: 10.000
    parameters['batch_size'] = 64

    # Load data
    recordings = load_recordings(settings.sonar_dataset_path)

    cols_to_remove = ["Quat_W_LF","Quat_W_LW","Quat_W_RF","Quat_W_RW","Quat_W_ST","Quat_X_LF","Quat_X_LW","Quat_X_RF","Quat_X_RW","Quat_X_ST","Quat_Y_LF","Quat_Y_LW","Quat_Y_RF","Quat_Y_RW","Quat_Y_ST","Quat_Z_LF","Quat_Z_LW","Quat_Z_RF","Quat_Z_RW","Quat_Z_ST"]

    # Activities to number, Remove quat columns
    for rec in recordings:
        rec.sensor_frame = rec.sensor_frame.drop(cols_to_remove, axis=1)
        rec.activities = rec.activities.map(lambda label: settings.ACTIVITIES[label])

    print(recordings[0].sensor_frame.shape)

    random.seed(1678978086101)
    random.shuffle(recordings)

    # Preprocessing
    # MinMaxScaler is applied in timegan.py
    recordings, _ = preprocess(recordings, methods=[
        interpolate_linear
    ])

    # Windowize all recordings
    windowizer = Windowizer(WINDOW_SIZE, STRIDE_SIZE, Windowizer.windowize_jumping, 60)

    # LOSO-folds
    for subject in settings.SUBJECTS:
        print(f"LOSO-fold without subject: {subject}")

        # Remove recordings where recording.subject_id == subject_id
        alpha_subset = [recording for recording in recordings if recording.subject != subject]
        X_train, y_train = windowizer.windowize_convert(alpha_subset)
        
        # Split recordings data activity-wise for data augmentation
        print("Begin data augmentation")
        activities_one_hot_encoded = np.eye(15, 15)
        for (index, row) in enumerate(activities_one_hot_encoded):
            # Get all indices in y_train where the one-hot-encoded row is equal to row
            activity_group_indices = np.nonzero(np.all(np.isclose(y_train, row), axis=1))[0]
            activity_group_X = X_train[activity_group_indices]
            activity_group_y = y_train[activity_group_indices]

            # -------------------------------------------------------------
            # Data augmentation
            # -------------------------------------------------------------
            ori_data = np.squeeze(activity_group_X, -1)

            generated_activity_data = timegan.timegan(ori_data, parameters, index)
            
            print(f'Finish Synthetic Data Generation: {generated_activity_data.shape}')

            # Convert generated data (list) to numpy array
            generated_activity_data = np.asarray(generated_activity_data)
            generated_activity_data = np.expand_dims(generated_activity_data, axis=-1)

            # Save generated data (unnormalized)
            np.save(f'data_{subject}_{index}_{WINDOW_SIZE}', generated_activity_data)

            # Garbage collection
            del generated_activity_data
            del activity_group_X
            del activity_group_y
            del ori_data
            gc.collect()


start()
