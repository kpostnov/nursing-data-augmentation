import os
import random
import numpy as np
import gc
from sklearn.utils import shuffle

from evaluation.conf_matrix import create_conf_matrix
from evaluation.metrics import accuracy, f_score, mmd_rbf
from evaluation.text_metrics import create_text_metrics
from evaluation.save_configuration import save_model_configuration
from loader.preprocessing import interpolate_ffill, normalize_standardscaler, preprocess
from loader.load_pamap2_dataset import load_pamap2_dataset
from models.AdaptedDeepConvLSTM import AdaptedDeepConvLSTM
from datatypes.Window import Window
from models.RainbowModel import RainbowModel
from utils.Windowizer import Windowizer
from utils.folder_operations import new_saved_experiment_folder
import utils.settings as settings
from visualization.visualize import plot_pca_distribution, plot_tsne_distribution, plot_time_series_alpha

import TimeGAN.timegan as timegan


def start() -> None:
    WINDOW_SIZE = 100
    STRIDE_SIZE = 100

    def get_averaged_dataframes(original_data: np.ndarray, generated_data: np.ndarray):
        '''
        Calculate the mean of each time step over all sensor channels and return the dataframes.
        '''

        assert(original_data.ndim == 3), 'original_data must be a 3D array'
        assert(generated_data.ndim == 3), 'generated_data must be a 3D array'

        number_samples = min(original_data.shape[0], 2000)
        idx = np.random.permutation(number_samples)

        original_data = original_data[idx]
        generated_data = generated_data[idx]

        seq_len = original_data.shape[1]

        for i in range(number_samples):
            if (i == 0):
                original_array = np.reshape(np.mean(original_data[0, :, :], 1), [1, seq_len])
                generated_array = np.reshape(np.mean(generated_data[0, :, :], 1), [1, seq_len])
            else:
                original_array = np.concatenate((original_array,
                                                 np.reshape(np.mean(original_data[i, :, :], 1), [1, seq_len])))
                generated_array = np.concatenate((generated_array,
                                                  np.reshape(np.mean(generated_data[i, :, :], 1), [1, seq_len])))

        # Returns 2d array (num_samples, window_size) --> Verify that!
        return original_array, generated_array

    def build_model(n_epochs: int, n_features: int) -> RainbowModel:
        return AdaptedDeepConvLSTM(
            window_size=WINDOW_SIZE,
            stride_size=STRIDE_SIZE,
            n_features=n_features,
            n_outputs=len(settings.ACTIVITIES),
            verbose=1,
            n_epochs=n_epochs)

    def preprocess_generated_array(generated_activity_data: np.ndarray, scaler) -> np.ndarray:
        """
        Preprocess the generated data the same way as the real training data.
        """
        # Reshape array for normalization
        generated_activity_data = np.squeeze(generated_activity_data, -1)
        generated_activity_data = generated_activity_data.reshape(-1, 11)
        # Normalize data
        generated_activity_data = scaler.transform(generated_activity_data)
        # Inverse reshape data
        generated_activity_data = generated_activity_data.reshape(-1, WINDOW_SIZE, 11)
        generated_activity_data = np.expand_dims(generated_activity_data, axis=-1)

        return generated_activity_data

    # GAN Newtork parameters
    parameters = dict()
    parameters['module'] = 'gru'  # LSTM possible
    parameters['hidden_dim'] = 44  # Paper: 4 times the size of input features
    parameters['num_layer'] = 3
    parameters['iterations'] = 10000  # Paper: 10.000
    parameters['batch_size'] = 128

    # Load data
    recordings = load_pamap2_dataset(settings.pamap2_dataset_path)

    random.seed(1678978086101)
    random.shuffle(recordings)

    # Preprocessing
    # MinMaxScaler is applied in timegan.py
    recordings, _ = preprocess(recordings, methods=[
        interpolate_ffill
    ])

    # Windowize all recordings
    windowizer = Windowizer(WINDOW_SIZE, STRIDE_SIZE, Windowizer.windowize_sliding)

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
            np.save(f'data_{subject_id}_{index}_pamap', generated_activity_data)

            # Garbage collection
            del generated_activity_data
            del activity_group_X
            del ori_data
            gc.collect()


start()
