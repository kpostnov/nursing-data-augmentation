import os
import random
from loader.preprocessing import pamap2_preprocess
from utils.cache_recordings import load_recordings
from models.DeepConvLSTM import DeepConvLSTM
from datatypes.Window import Window
from utils.Windowizer import Windowizer
import utils.settings as settings
import numpy as np

import TimeGAN.timegan as timegan
import gc


WINDOW_SIZE = 120
STRIDE_SIZE = 120

# GAN Newtork parameters
parameters = dict()
parameters['module'] = 'gru'  # LSTM possible
parameters['hidden_dim'] = 280  # Paper: 4 times the size of input features
parameters['num_layer'] = 3
parameters['iterations'] = 10000  # Paper: 10.000
parameters['batch_size'] = 128

# Load data
recordings = load_recordings(settings.sonar_dataset_path)

random.seed(1678978086101)
random.shuffle(recordings)

# Preprocessing (Interpolation)
recordings = pamap2_preprocess(recordings)

# Windowize all recordings
windowizer = Windowizer(WINDOW_SIZE, STRIDE_SIZE, Windowizer.windowize_sliding)

# LOSO-folds (alpha-dataset)
for subject in settings.SUBJECTS:
    print(f"LOSO-fold without subject: {subject}")

    # Remove recordings where recording.subject_id == subject_id
    alpha_subset = [recording for recording in recordings if recording.subject != subject]
    validation_subset = [recording for recording in recordings if recording.subject == subject]

    # Train alpha model on alpha_subset
    print("Training alpha model on alpha_subset")
    # model_alpha = DeepConvLSTM(
    #     window_size=WINDOW_SIZE,
    #     stride_size=STRIDE_SIZE,
    #     n_features=recordings[0].sensor_frame.shape[1],
    #     n_outputs=6,
    #     verbose=1,
    #     n_epochs=200)
    X_train, y_train = windowizer.windowize_convert(alpha_subset)
    # model_alpha.fit(X_train, y_train)

    # Test alpha model on validation_subset
    X_test, y_test = windowizer.windowize_convert(validation_subset)
    # y_test_pred_model_alpha = model_alpha.predict(X_test)

    # Split recordings data activity-wise for data augmentation
    print("Begin data augmentation")
    activities_one_hot_encoded = np.eye(15, 15)
    for (index, row) in enumerate(activities_one_hot_encoded):
        # Get all indices in y_train where the one-hot-encoded row is equal to row
        activity_group_indices = np.nonzero(np.all(np.isclose(y_train, row), axis=1))[0]
        activity_group_X = X_train[activity_group_indices]
        activity_group_y = y_train[activity_group_indices]

        # Data Augmentation
        ori_data = np.squeeze(activity_group_X, -1)
        # ori_data_list = list()
        # for matrix in ori_data:
        #     ori_data_list.append(matrix)

        generated_activity_data = timegan.timegan(ori_data, parameters)

        generated_activity_labels = np.expand_dims(row, axis=0)
        generated_activity_labels = np.repeat(generated_activity_labels, len(generated_activity_data), axis=0)

        print(f'Finish Synthetic Data Generation: {generated_activity_data.shape}')

        # Convert generated data (list) to numpy array
        generated_activity_data = np.asarray(generated_activity_data)
        generated_activity_data = np.expand_dims(generated_activity_data, axis=-1)

        # Save generated data
        np.save(f'data_{subject}_{index}', generated_activity_data)
        np.save(f'labels_{subject}_{index}', generated_activity_labels)

        # Garbage collection
        del generated_activity_data
        del generated_activity_labels
        del activity_group_X
        del activity_group_y
        del ori_data
        gc.collect()

        continue

        # Merge augmented data with alpha_subset
        # X_train = np.append(X_train, generated_activity_data, axis=0)
        # y_train = np.append(y_train, generated_activity_labels, axis=0)

    # TODO: Train beta model on beta_subset
    model_beta = DeepConvLSTM(
        window_size=WINDOW_SIZE,
        stride_size=STRIDE_SIZE,
        n_features=recordings[0].sensor_frame.shape[1],
        n_outputs=6,
        verbose=1,
        n_epochs=200)
    # model_beta.fit(X_train, y_train)
    # y_test_pred_model_beta = model_beta.predict(X_test)