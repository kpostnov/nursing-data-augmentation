import os
import random
import numpy as np
import gc
from sklearn.utils import shuffle

from evaluation.conf_matrix import create_conf_matrix
from evaluation.metrics import accuracy, f_score
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
from visualization.visualize import plot_pca_distribution, plot_tsne_distribution

import TimeGAN.timegan as timegan


def start(generate: bool = True) -> None:
    WINDOW_SIZE = 100
    STRIDE_SIZE = 100

    def build_model(n_epochs : int, n_features: int) -> RainbowModel:
        return AdaptedDeepConvLSTM(
            window_size=WINDOW_SIZE,
            stride_size=STRIDE_SIZE,
            n_features=n_features,
            n_outputs=len(settings.LABELS),
            verbose=1,
            n_epochs=n_epochs)


    # GAN Newtork parameters
    parameters = dict()
    parameters['module'] = 'gru' # LSTM possible
    parameters['hidden_dim'] = 44 # Paper: 4 times the size of input features
    parameters['num_layer'] = 3
    parameters['iterations'] = 10000 # Paper: 10.000
    parameters['batch_size'] = 128

    # Load data
    recordings = load_pamap2_dataset(settings.pamap2_dataset_path)

    random.seed(1678978086101)
    random.shuffle(recordings)

    # Preprocessing
    scaler = None
    if generate:
        # MinMaxScaler is applied in timegan.py
        recordings, _ = preprocess(recordings, methods=[
            interpolate_ffill
        ])
    else:
        recordings, scaler = preprocess(recordings, methods=[
            interpolate_ffill,
            normalize_standardscaler
        ])

    # Windowize all recordings
    windowizer = Windowizer(WINDOW_SIZE, STRIDE_SIZE, Windowizer.windowize_sliding)

    # LOSO-folds (alpha-dataset)
    subject_ids = range(1, 9)
    for subject_id in subject_ids:

        print("LOSO-fold without subject: {}".format(subject_id))

        # Remove recordings where recording.subject_id == subject_id
        alpha_subset = [recording for recording in recordings if recording.subject != str(subject_id)]
        validation_subset = [recording for recording in recordings if recording.subject == str(subject_id)]

        # Train alpha model on alpha_subset
        print("Training alpha model on alpha_subset")
        model_alpha = build_model(n_epochs=1, n_features=recordings[0].sensor_frame.shape[1])
        X_train, y_train = windowizer.windowize_convert(alpha_subset)
        # model_alpha.fit(X_train, y_train)

        # Test alpha model on validation_subset
        X_test, y_test = windowizer.windowize_convert(validation_subset)
        # y_test_pred_model_alpha = model_alpha.predict(X_test)

        # Split recordings data activity-wise for data augmentation
        print("Begin data augmentation")
        activities_one_hot_encoded = np.eye(6, 6)
        for (index, row) in enumerate(activities_one_hot_encoded):
            # Get all indices in y_train where the one-hot-encoded row is equal to row
            activity_group_indices = np.nonzero(np.all(np.isclose(y_train, row), axis=1))[0]
            activity_group_X = X_train[activity_group_indices]
            activity_group_y = y_train[activity_group_indices]

            generated_activity_data = None

            # -------------------------------------------------------------
            # Data augmentation
            # -------------------------------------------------------------
            if generate:
                ori_data = np.squeeze(activity_group_X, -1)

                generated_activity_data = timegan.timegan(ori_data, parameters, index)

                print(f'Finish Synthetic Data Generation: {generated_activity_data.shape}')

                # Convert generated data (list) to numpy array
                # generated_activity_data = np.asarray(generated_activity_data)
                generated_activity_data = np.expand_dims(generated_activity_data, axis=-1)

                # Save generated data
                np.save(f'data_{subject_id}_{index}_pamap', generated_activity_data)

                # Garbage collection
                del generated_activity_data
                del activity_group_X
                del activity_group_y
                del ori_data
                gc.collect()

                continue

            # -------------------------------------------------------------
            # Loading synthetic data
            # -------------------------------------------------------------
            else: 
                try:
                    generated_activity_data = np.load(f'data_{subject_id}_{index}_pamap.npy')
                except OSError:
                    continue

                print(generated_activity_data.shape)
                plot_pca_distribution(activity_group_X, generated_activity_data, str(subject_id) + "_" + str(index) + "_pamap")
                plot_tsne_distribution(activity_group_X, generated_activity_data, str(subject_id) + "_" + str(index) + "_pamap")

            # Reshape array for normalization
            generated_activity_data = np.squeeze(generated_activity_data, -1)
            generated_activity_data = generated_activity_data.reshape(-1, 70)
            # Normalize data
            generated_activity_data = scaler.transform(generated_activity_data)
            # Inverse reshape data
            generated_activity_data = generated_activity_data.reshape(-1, WINDOW_SIZE, 70)
            generated_activity_data = np.expand_dims(generated_activity_data, axis=-1)

            # Merge augmented data with alpha_subset
            generated_activity_labels = np.expand_dims(row, axis=0)
            generated_activity_labels = np.repeat(generated_activity_labels, len(generated_activity_data), axis=0)

            X_train = np.append(X_train, generated_activity_data, axis=0)
            y_train = np.append(y_train, generated_activity_labels, axis=0)

        # Shuffle X_train and y_train
        X_train, y_train = shuffle(X_train, y_train)

        # Train beta model on beta_subset
        model_beta = build_model(n_epochs=1, n_features=recordings[0].sensor_frame.shape[1])
        # model_beta.fit(X_train, y_train)
        # y_test_pred_model_beta = model_beta.predict(X_test)

start(generate=True)