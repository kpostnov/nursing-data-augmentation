import os
import random
from evaluation.conf_matrix import create_conf_matrix
from evaluation.metrics import accuracy
from evaluation.text_metrics import create_text_metrics
from evaluation.save_configuration import save_model_configuration
from loader.preprocessing import pamap2_preprocess
from loader.load_pamap2_dataset import load_pamap2_dataset
from models.DeepConvLSTM import DeepConvLSTM
from utils.Window import Window
from utils.Windowizer import Windowizer
from utils.array_operations import split_list_by_percentage
from utils.folder_operations import new_saved_experiment_folder
import utils.settings as settings
import numpy as np
from visualization.visualize import plot_pca_distribution, plot_tsne_distribution

# import TimeGAN.timegan as timegan
import TimeGAN_gpu.timegan as timegan
import gc


WINDOW_SIZE = 100
STRIDE_SIZE = 100

# GAN Newtork parameters
parameters = dict()
# parameters['module'] = 'gru' # LSTM possible
# parameters['hidden_dim'] = 44 # Paper: 4 times the size of input features
# parameters['num_layer'] = 3
# parameters['iterations'] = 1 # Paper: 10.000
# parameters['batch_size'] = 128

parameters['seq_len'] = WINDOW_SIZE
parameters['n_seq'] = 11
parameters['batch_size'] = 128
parameters['hidden_dim'] = 44
parameters['num_layer'] = 3
parameters['train_steps'] = 10000

# Load data
recordings = load_pamap2_dataset(settings.pamap2_dataset_path)

random.seed(1678978086101)
random.shuffle(recordings)

# Preprocessing (Interpolation)
recordings = pamap2_preprocess(recordings)

# Windowize all recordings
windowizer = Windowizer(WINDOW_SIZE, STRIDE_SIZE, Windowizer.windowize_sliding)
X_train, y_train = windowizer.windowize_convert(recordings)

# Train M1 on whole dataset (no normalization)
# model_m1 = DeepConvLSTM(
#     window_size=WINDOW_SIZE,
#     stride_size=STRIDE_SIZE,
#     n_features=recordings[0].sensor_frame.shape[1],
#     n_outputs=6,
#     verbose=1,
#     n_epochs=200)
# model_m1.fit(X_train=X_train, y_train=y_train)

# LOSO-folds (alpha-dataset)
subject_ids = range(1, 9)
for subject_id in subject_ids:
    print("LOSO-fold without subject: {}".format(subject_id))

    # Remove recordings where recording.subject_id == subject_id
    alpha_subset = [recording for recording in recordings if recording.subject != str(subject_id)]
    validation_subset = [recording for recording in recordings if recording.subject == str(subject_id)]

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
    activities_one_hot_encoded = np.eye(6, 6)
    for (index, row) in enumerate(activities_one_hot_encoded):
        # Get all indices in y_train where the one-hot-encoded row is equal to row
        activity_group_indices = np.nonzero(np.all(np.isclose(y_train, row), axis=1))[0]
        activity_group_X = X_train[activity_group_indices]
        activity_group_y = y_train[activity_group_indices]

        # Data Augmentation
        # ori_data = np.squeeze(activity_group_X, -1)
        # generated_activity_data = timegan.timegan(ori_data, parameters)

        # generated_activity_labels = np.expand_dims(row, axis=0)
        # generated_activity_labels = np.repeat(generated_activity_labels, len(generated_activity_data), axis=0)

        # print('Finish Synthetic Data Generation')

        # Test generated data (>= 95%)

        # Convert generated data (list) to numpy array
        # generated_activity_data = np.asarray(generated_activity_data)
        # generated_activity_data = np.expand_dims(generated_activity_data, axis=-1)

        # Save generated data
        # np.save(f'data_{subject_id}_{index}', generated_activity_data)
        # np.save(f'labels_{subject_id}_{index}', generated_activity_labels)


        # Garbage collection
        # del generated_activity_data
        # del generated_activity_labels
        # del activity_group_X
        # del activity_group_y
        # del ori_data
        # gc.collect()



        data_path = "D:\dataset\Augmented Data\without_gpu\\"
        generated_activity_data = np.load(f'{data_path}\data_{subject_id}_{index}.npy')

        plot_pca_distribution(activity_group_X, generated_activity_data, str(index) + "_without_gpu")
        plot_tsne_distribution(activity_group_X, generated_activity_data, str(index) + "_without_gpu")

        data_path = "D:\dataset\Augmented Data\with_gpu\\"
        generated_activity_data = np.load(f'{data_path}\data_{subject_id}_{index}.npy')
        plot_pca_distribution(activity_group_X, generated_activity_data, str(index) + "_with_gpu")
        plot_tsne_distribution(activity_group_X, generated_activity_data, str(index) + "_with_gpu")

        
        # if index == 2:
        #     exit()
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
