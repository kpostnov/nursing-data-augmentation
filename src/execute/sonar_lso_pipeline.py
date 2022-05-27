import os
import psutil
import random
import numpy as np
import gc
from sklearn.utils import shuffle
import warnings
warnings.filterwarnings("ignore")

from evaluation.conf_matrix import create_conf_matrix
from evaluation.metrics import accuracy, f_score
from evaluation.save_configuration import save_model_configuration
from evaluation.text_metrics import create_text_metrics
from loader.preprocessing import preprocess, normalize_standardscaler, interpolate_linear
from models.RainbowModel import RainbowModel
from utils.cache_recordings import load_recordings
from models.AdaptedDeepConvLSTM import AdaptedDeepConvLSTM
from datatypes.Window import Window
from utils.Windowizer import Windowizer
from utils.folder_operations import new_saved_experiment_folder
import utils.settings as settings
from visualization.visualize import plot_pca_distribution, plot_tsne_distribution
from utils.chunk_generation import chunk_generator
from tensorflow.keras.utils import to_categorical

import TimeGAN.timegan as timegan


def start(generate: bool = True) -> None:
    # Leave-subject-out pipeline
    WINDOW_SIZE = 300
    STRIDE_SIZE = 300

    def build_model(n_epochs : int, n_features: int) -> RainbowModel:
        return AdaptedDeepConvLSTM(
            window_size=WINDOW_SIZE,
            stride_size=STRIDE_SIZE,
            n_features=n_features,
            n_outputs=len(settings.LABELS),
            verbose=1,
            n_epochs=n_epochs)


    def preprocess_generated_array(generated_activity_data: np.ndarray, scaler) -> np.ndarray:
        # Reshape array for normalization
        generated_activity_data = np.squeeze(generated_activity_data, -1)
        generated_activity_data = generated_activity_data.reshape(-1, 70)
        # Normalize data
        generated_activity_data = scaler.transform(generated_activity_data)
        # Inverse reshape data
        generated_activity_data = generated_activity_data.reshape(-1, WINDOW_SIZE, 70)
        generated_activity_data = np.expand_dims(generated_activity_data, axis=-1)

        return generated_activity_data

    
    def remove_quat_columns(array: np.ndarray) -> np.ndarray:
        """
        Removes the quaternion columns (15:35 in SONAR) from the array. New shape is (n_samples, 300, 50, 1).
        """
        if array.shape[2] == 50:
            return array
        X_sq = np.squeeze(array, -1)
        X_del = np.delete(X_sq, np.s_[15:35], axis=2)
        X_del = np.expand_dims(X_del, axis=-1)

        return X_del


    def model_training(model: RainbowModel, synth_files: list, X_train: np.ndarray, y_train: np.ndarray, scaler):
        process = psutil.Process(os.getpid())
        
        for epoch in range(model.n_epochs):
            print(f"Epoch {epoch} / {model.n_epochs}")

            unused_tenths = [i for i in range(10)]
            X_train_size = X_train.shape[0]
            X_train_chunk_size = X_train_size // 10

            for X_chunk, y_chunk in chunk_generator(synth_files):
                # Get the tenths that haven't been yielded yet
                random_tenth = random.choice(unused_tenths)

                X_train_chunk = X_train[random_tenth * X_train_chunk_size : (random_tenth + 1) * X_train_chunk_size]
                y_train_chunk = y_train[random_tenth * X_train_chunk_size : (random_tenth + 1) * X_train_chunk_size]
                
                # Remove used tenth
                unused_tenths.remove(random_tenth)
                
                # Append synthetic data if available
                if X_chunk is not None:
                    X_chunk = preprocess_generated_array(X_chunk, scaler)
                    X_train_chunk = np.append(X_train_chunk, X_chunk, axis=0)
                    y_train_chunk = np.append(y_train_chunk, y_chunk, axis=0)

                print(f"{process.memory_info().rss / 1000000} MB memory used")

                # Shuffle data
                X_train_chunk, y_train_chunk = shuffle(X_train_chunk, y_train_chunk)

                model.fit(X_train_chunk, y_train_chunk, ignore_epochs=True)

                # Garbage collection
                del X_chunk, y_chunk, X_train_chunk, y_train_chunk
                gc.collect()            
    

    # GAN Newtork parameters
    parameters = dict()
    parameters['module'] = 'gru'  # LSTM possible
    parameters['hidden_dim'] = 200  # Paper: 4 times the size of input features
    parameters['num_layer'] = 3
    parameters['iterations'] = 8000  # Paper: 10.000
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
    scaler = None
    if generate:
        # MinMaxScaler is applied in timegan.py
        recordings, _ = preprocess(recordings, methods=[
            interpolate_linear
        ])
    else:
        recordings, scaler = preprocess(recordings, methods=[
            interpolate_linear,
            normalize_standardscaler
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
        # model_alpha = build_model(n_epochs=10, n_features=recordings[0].sensor_frame.shape[1])
        X_train, y_train = windowizer.windowize_convert(alpha_subset)
        # model_alpha.fit(X_train, y_train)

        # Test alpha model on validation_subset
        X_test, y_test = windowizer.windowize_convert(validation_subset)
        # y_test_pred_model_alpha = model_alpha.predict(X_test)

        # experiment_folder_path = new_saved_experiment_folder(f'{subject}_mtss_pipeline_alpha')
        # create_conf_matrix(experiment_folder_path, y_test_pred_model_alpha, y_test)
        # create_text_metrics(experiment_folder_path, y_test_pred_model_alpha, y_test, [accuracy, f_score])
        # save_model_configuration(experiment_folder_path, model_alpha)

        # Split recordings data activity-wise for data augmentation
        print("Begin data augmentation")
        activities_one_hot_encoded = np.eye(15, 15)
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
                generated_activity_data = np.asarray(generated_activity_data)
                generated_activity_data = np.expand_dims(generated_activity_data, axis=-1)

                # Save generated data
                np.save(f'data_{subject}_{index}_{WINDOW_SIZE}', generated_activity_data)

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
                    generated_activity_data = np.load(f'data_{subject}_{index}.npy')
                except OSError:
                    continue
                
                print(generated_activity_data.shape)
                plot_pca_distribution(activity_group_X, generated_activity_data, str(subject) + "_" + str(index))
                plot_tsne_distribution(activity_group_X, generated_activity_data, str(subject) + "_" + str(index))
                
                # exit()

            generated_activity_data = preprocess_generated_array(generated_activity_data, scaler)

            # Create labels for generated data
            generated_activity_labels = np.expand_dims(row, axis=0)
            generated_activity_labels = np.repeat(generated_activity_labels, len(generated_activity_data), axis=0)
            
            # Merge augmented data with alpha_subset
            X_train = np.append(X_train, generated_activity_data, axis=0)
            y_train = np.append(y_train, generated_activity_labels, axis=0)

        # Shuffle X_train and y_train
        X_train, y_train = shuffle(X_train, y_train)

        # Train beta model on beta_subset
        model_beta = build_model(n_epochs=10, n_features=recordings[0].sensor_frame.shape[1])
        # model_beta.fit(X_train, y_train)
        # y_test_pred_model_beta = model_beta.predict(X_test)

start(generate=True)
