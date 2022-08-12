import os
import random
import numpy as np
import gc

from loader.preprocessing import interpolate_ffill, preprocess
from loader.load_pamap2_dataset import load_pamap2_dataset
from utils.Windowizer import Windowizer
import utils.settings as settings

import TimeGAN.timegan as timegan


def start_generation() -> None:

    # GAN Newtork parameters
    parameters = dict()
    parameters['module'] = 'gru'  # LSTM possible
    parameters['hidden_dim'] = 44  # Paper: 4 times the size of input features
    parameters['num_layer'] = 3
    parameters['iterations'] = 10000  # Paper: 10.000
    parameters['batch_size'] = 128

    # Load data
    recordings = load_pamap2_dataset(settings.dataset_path)

    random.seed(1678978086101)
    random.shuffle(recordings)

    # Preprocessing
    # MinMaxScaler is applied in timegan.py
    recordings, _ = preprocess(recordings, methods=[
        interpolate_ffill
    ])

    # Windowize all recordings
    windowizer = Windowizer(settings.WINDOW_SIZE, settings.STRIDE_SIZE, Windowizer.windowize_sliding)

    # LOSO-folds (alpha-dataset)
    subject_ids = range(1, 9)
    for subject_id in subject_ids:
        print("LOSO-fold without subject: {}".format(subject_id))

        # Remove recordings where recording.subject_id == subject_id
        alpha_subset = [recording for recording in recordings if recording.subject != str(subject_id)]
        X_train, y_train = windowizer.windowize_convert(alpha_subset)

        # Split recordings data activity-wise for data augmentation
        print("Begin data augmentation")
        activities_one_hot_encoded = np.eye(len(settings.LABELS), len(settings.LABELS))
        for (index, row) in enumerate(activities_one_hot_encoded):
            # Get all indices in y_train where the one-hot-encoded row is equal to row
            activity_group_indices = np.nonzero(np.all(np.isclose(y_train, row), axis=1))[0]
            activity_group_X = X_train[activity_group_indices]

            # -------------------------------------------------------------
            # Data augmentation
            # -------------------------------------------------------------
            ori_data = np.squeeze(activity_group_X, -1)

            generated_activity_data = timegan.timegan(ori_data, parameters, index)

            print(f'Finish Synthetic Data Generation: {generated_activity_data.shape}')

            # Convert generated data (list) to numpy array
            generated_activity_data = np.asarray(generated_activity_data)
            generated_activity_data = np.expand_dims(generated_activity_data, axis=-1)

            # Save generated data
            np.save(f'data_{subject_id}_{index}', generated_activity_data)

            # Garbage collection
            del generated_activity_data
            del activity_group_X
            del ori_data
            gc.collect()
