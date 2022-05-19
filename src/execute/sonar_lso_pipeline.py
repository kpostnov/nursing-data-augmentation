import os
import random
from loader.preprocessing import preprocess, normalize_standardscaler, interpolate_linear
from utils.cache_recordings import load_recordings
from models.AdaptedDeepConvLSTM import AdaptedDeepConvLSTM
from datatypes.Window import Window
from utils.Windowizer import Windowizer
import utils.settings as settings
import numpy as np

import TimeGAN.timegan as timegan
# import mtss_gan.mtss_gan as mtss_gan
import gc

from visualization.visualize import plot_pca_distribution, plot_tsne_distribution


# Leave-subject-out pipeline
WINDOW_SIZE = 900
STRIDE_SIZE = 900

# GAN Newtork parameters
parameters = dict()
parameters['module'] = 'gru'  # LSTM possible
parameters['hidden_dim'] = 280  # Paper: 4 times the size of input features
parameters['num_layer'] = 3
parameters['iterations'] = 10000  # Paper: 10.000
parameters['batch_size'] = 32

# parameters['seq_len'] = WINDOW_SIZE
# parameters['n_seq'] = 70
# parameters['batch_size'] = 128
# parameters['hidden_dim'] = 280
# parameters['num_layer'] = 3
# parameters['train_steps'] = 10000

# Load data
recordings = load_recordings(settings.sonar_dataset_path)

# Activities to number
for rec in recordings:
    rec.activities = rec.activities.map(lambda label: settings.ACTIVITIES[label])

random.seed(1678978086101)
random.shuffle(recordings)

# Preprocessing (MinMaxScaler is applied in timegan.py)
recordings = preprocess(recordings, methods=[
    interpolate_linear
])

# Windowize all recordings
windowizer = Windowizer(WINDOW_SIZE, STRIDE_SIZE, Windowizer.windowize_jumping, 60)

# LOSO-folds (alpha-dataset)
for subject in settings.SUBJECTS:
    print(f"LOSO-fold without subject: {subject}")

    # Remove recordings where recording.subject_id == subject_id
    alpha_subset = [recording for recording in recordings if recording.subject != subject]
    validation_subset = [recording for recording in recordings if recording.subject == subject]

    # Train alpha model on alpha_subset
    print("Training alpha model on alpha_subset")
    # model_alpha = AdaptedDeepConvLSTM(
    #     window_size=WINDOW_SIZE,
    #     stride_size=STRIDE_SIZE,
    #     n_features=recordings[0].sensor_frame.shape[1],
    #     n_outputs=15,
    #     verbose=1,
    #     n_epochs=20)
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

        generated_activity_data = timegan.timegan(ori_data, parameters, index)
        # generated_activity_data = mtss_gan.start_training(ori_data, seq_len=WINDOW_SIZE, num_features=70)

        generated_activity_labels = np.expand_dims(row, axis=0)
        generated_activity_labels = np.repeat(generated_activity_labels, len(generated_activity_data), axis=0)

        print(f'Finish Synthetic Data Generation: {generated_activity_data.shape}')

        # Convert generated data (list) to numpy array
        generated_activity_data = np.asarray(generated_activity_data)
        generated_activity_data = np.expand_dims(generated_activity_data, axis=-1)

        # Save generated data
        np.save(f'data_{subject}_{index}_900', generated_activity_data)
        np.save(f'labels_{subject}_{index}_900', generated_activity_labels)

        # Garbage collection
        del generated_activity_data
        del generated_activity_labels
        del activity_group_X
        del activity_group_y
        del ori_data
        gc.collect()

        continue
        '''
        data_path = "D:\dataset\Augmented Data\SONAR"
        try:
            generated_activity_data = np.load(f'{data_path}\data_{subject}_{index}.npy')
        except OSError:
            continue
        
        print(generated_activity_data.shape)
        plot_pca_distribution(activity_group_X, generated_activity_data, str(subject) + "_" + str(index))
        plot_tsne_distribution(activity_group_X, generated_activity_data, str(subject) + "_" + str(index))
        '''
        # exit()
        # Merge augmented data with alpha_subset
        # X_train = np.append(X_train, generated_activity_data, axis=0)
        # y_train = np.append(y_train, generated_activity_labels, axis=0)

    # TODO: Train beta model on beta_subset
    model_beta = AdaptedDeepConvLSTM(
        window_size=WINDOW_SIZE,
        stride_size=STRIDE_SIZE,
        n_features=recordings[0].sensor_frame.shape[1],
        n_outputs=15,
        verbose=1,
        n_epochs=20)
    # model_beta.fit(X_train, y_train)
    # y_test_pred_model_beta = model_beta.predict(X_test)
