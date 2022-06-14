import os
import psutil
import random
import numpy as np
import gc
from sklearn.utils import shuffle
import warnings
warnings.filterwarnings("ignore")

from evaluation.conf_matrix import create_conf_matrix
from evaluation.metrics import accuracy, f_score, mmd_rbf
from evaluation.save_configuration import save_model_configuration
from evaluation.text_metrics import create_text_metrics
from loader.preprocessing import preprocess, normalize_standardscaler, interpolate_linear
from utils.file_functions import get_file_paths_in_folder
from models.RainbowModel import RainbowModel
from utils.cache_recordings import load_recordings
from models.AdaptedDeepConvLSTM import AdaptedDeepConvLSTM
from datatypes.Window import Window
from utils.Windowizer import Windowizer
from utils.folder_operations import new_saved_experiment_folder
import utils.settings as settings
from visualization.visualize import plot_pca_distribution, plot_tsne_distribution
from utils.chunk_generation import chunk_generator


def start(eval_one: bool = False, eval_two: bool = False, eval_three: bool = False, eval_four: bool = False) -> None:
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


    def get_mean_array(data: np.ndarray) -> np.ndarray:
        '''
        Calculate the mean of each time step over all sensor channels and return a 2D array (n_samples, window_size).
        '''
        assert(data.ndim == 3), 'data must be a 3D array'

        seq_len = data.shape[1]
        array = data

        for i in range(data.shape[0]):
            if (i == 0):
                array = np.reshape(np.mean(data[0,:,:], 1), [1, seq_len])
            else:
                array = np.concatenate((array, np.reshape(np.mean(data[i,:,:], 1), [1, seq_len])))

        return array


    def remove_quat_columns(array: np.ndarray) -> np.ndarray:
        """
        Removes the quaternion columns (15:35 in SONAR) from the array. New shape is (n_windows, WINDOW_SIZE, 50, 1).
        """
        if array.shape[2] == 50:
            return array
        X_sq = np.squeeze(array, -1)
        X_del = np.delete(X_sq, np.s_[15:35], axis=2)
        X_del = np.expand_dims(X_del, axis=-1)

        return X_del


    def preprocess_generated_array(generated_activity_data: np.ndarray, scaler) -> np.ndarray:
        """
        Preprocess the generated data the same way as the real training data.
        """
        # Reshape array for normalization
        generated_activity_data = np.squeeze(generated_activity_data, -1)
        generated_activity_data = generated_activity_data.reshape(-1, 50)
        # Normalize data
        generated_activity_data = scaler.transform(generated_activity_data)
        # Inverse reshape data
        generated_activity_data = generated_activity_data.reshape(-1, WINDOW_SIZE, 50)
        generated_activity_data = np.expand_dims(generated_activity_data, axis=-1)

        return generated_activity_data


    def model_training(model: RainbowModel, synth_files: list, X_train: np.ndarray, y_train: np.ndarray, scaler) -> None:
        """
        Train the model on real and synthetic data chunk-wise. This is done due to memory constraints.
        """
        process = psutil.Process(os.getpid())
        
        for epoch in range(model.n_epochs):
            print(f"Epoch {epoch} / {model.n_epochs}")

            unused_tenths = [i for i in range(10)]
            X_train_size = X_train.shape[0]
            X_train_chunk_size = X_train_size // 10

            for X_chunk, y_chunk in chunk_generator(synth_files):
                # Get the tenths from original data that haven't been used yet
                random_tenth = random.choice(unused_tenths)
                X_train_chunk = X_train[random_tenth * X_train_chunk_size : (random_tenth + 1) * X_train_chunk_size]
                y_train_chunk = y_train[random_tenth * X_train_chunk_size : (random_tenth + 1) * X_train_chunk_size]
                
                # Remove used tenth
                unused_tenths.remove(random_tenth)
                
                # Append synthetic data if available
                if isinstance(X_chunk, np.ndarray):
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
    

    synth_data_path = "/dhc/groups/bp2021ba1/kirill/nursing-data-augmentation/src"
    files = get_file_paths_in_folder(synth_data_path)

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
    recordings, scaler = preprocess(recordings, methods=[
        interpolate_linear,
        normalize_standardscaler
    ])

    # Windowize all recordings
    windowizer = Windowizer(WINDOW_SIZE, STRIDE_SIZE, Windowizer.windowize_jumping, 60)

    # LOSO-folds
    for subject in settings.SUBJECTS:
        print(f"LOSO-fold without subject: {subject}")

        # Remove recordings where recording.subject_id == subject_id
        alpha_subset = [recording for recording in recordings if recording.subject != subject]
        validation_subset = [recording for recording in recordings if recording.subject == subject]
        X_train, y_train = windowizer.windowize_convert(alpha_subset)
        X_test, y_test = windowizer.windowize_convert(validation_subset)

        # -------------------------------------------------------------
        # Evaluation 1 / Evaluation 2
        # -------------------------------------------------------------
        if eval_one or eval_two: 
            random_mmd_scores = []
            generated_mmd_scores = []

            activities_one_hot_encoded = np.eye(len(settings.LABELS), len(settings.LABELS))
            for (index, row) in enumerate(activities_one_hot_encoded):
                # Get all indices in y_train where the one-hot-encoded row is equal to row
                activity_group_indices = np.nonzero(np.all(np.isclose(y_train, row), axis=1))[0]
                activity_group_X = X_train[activity_group_indices]
                
                try:
                    generated_activity_data = np.load(f'{synth_data_path}/data_{subject}_{index}_{WINDOW_SIZE}.npy')
                except OSError:
                    continue
                
                generated_activity_data = remove_quat_columns(generated_activity_data)
                generated_activity_data = preprocess_generated_array(generated_activity_data, scaler)
                print(generated_activity_data.shape)
                
                # -------------------------------------------------------------
                # Evaluation 1: Plotting PCA / tSNE distribution
                # -------------------------------------------------------------
                if eval_one: 
                    print(f"Evaluation 1: Plotting PCA / tSNE distribution for activity {index}")
                    plot_pca_distribution(activity_group_X, generated_activity_data, f"{subject}_{index}")
                    plot_tsne_distribution(activity_group_X, generated_activity_data, f"{subject}_{index}")

                # -------------------------------------------------------------
                # Evaluation 2: Maximum Mean Discrepancy
                # -------------------------------------------------------------
                if eval_two:
                    print(f"Evaluation 2: Calculating MMD for activity {index}")

                    # Random distribution
                    random_distribution = np.load(f'{synth_data_path}/../random_data/random_data_sonar.npy')
                    random_distribution = remove_quat_columns(random_distribution)
                    random_distribution = preprocess_generated_array(random_distribution, scaler)

                    # Take random samples from all datasets
                    n_samples = min(activity_group_X.shape[0], generated_activity_data.shape[0], random_distribution.shape[0])
                    activity_group_X = activity_group_X[np.random.choice(activity_group_X.shape[0], n_samples, replace=False)]
                    generated_activity_data = generated_activity_data[np.random.choice(generated_activity_data.shape[0], n_samples, replace=False)]
                    random_distribution = random_distribution[np.random.choice(random_distribution.shape[0], n_samples, replace=False)]

                    activity_group_X = np.squeeze(activity_group_X, -1)
                    generated_activity_data = np.squeeze(generated_activity_data, -1)
                    random_distribution = np.squeeze(random_distribution, -1)

                    activity_group_X = get_mean_array(activity_group_X)
                    generated_activity_data = get_mean_array(generated_activity_data)
                    random_distribution = get_mean_array(random_distribution)

                    # Calculate MMD
                    mmd_random = mmd_rbf(activity_group_X, random_distribution)
                    mmd_generated = mmd_rbf(activity_group_X, generated_activity_data)

                    print(f"MMD random: {mmd_random}")
                    print(f"MMD generated: {mmd_generated}")

                    random_mmd_scores.append(mmd_random)
                    generated_mmd_scores.append(mmd_generated)
            
            print(f"Random MMD scores: {random_mmd_scores}")
            print(f"Generated MMD scores: {generated_mmd_scores}")
        
        # -------------------------------------------------------------
        # Evaluation 3: TSTR / TRTS
        # -------------------------------------------------------------
        if eval_three:
            # Train on synthetic data, test on real data
            print("Evaluation 2: TSTR / TRTS")
            print("Training on synthetic data, testing on real data")
            model_tstr = build_model(n_epochs=10, n_features=recordings[0].sensor_frame.shape[1])

            # Train model
            for epoch in range(model_tstr.n_epochs):
                print(f"Epoch {epoch} / {model_tstr.n_epochs}")
                for X_chunk, y_chunk in chunk_generator(files):
                    X_chunk = preprocess_generated_array(X_chunk, scaler)
                    model_tstr.fit(X_chunk, y_chunk, ignore_epochs=True)

            # Choose one thousand random values from X_train for testing
            random_indices = random.sample(range(X_train.shape[0]), 1000)
            X_subset = X_train[random_indices]
            y_subset = y_train[random_indices]

            # Original implementation
            # random_indices = random.sample(range(X_test.shape[0]), 1000)
            # X_subset = X_test[random_indices]
            # y_subset = y_test[random_indices]

            y_predict_tstr = model_tstr.predict(X_subset)
            print(f"TSTR f_score: {f_score(y_subset, y_predict_tstr)}")
            print(f"TSTR accuracy: {accuracy(y_subset, y_predict_tstr)}")

            del X_subset, y_subset, y_predict_tstr
            gc.collect()


            # Train on real data, test on synthetic data
            print("Training on real data, testing on synthetic data")
            model_trts = build_model(n_epochs=10, n_features=recordings[0].sensor_frame.shape[1])
            model_trts.fit(X_train, y_train)

            # Build test set
            X_test_trts = None
            y_test_trts = None
            for X_chunk, y_chunk in chunk_generator(files):
                # Choose one hundred random values from chunks
                random_indices = random.sample(range(X_chunk.shape[0]), 100)
                X_chunk = X_chunk[random_indices]
                y_chunk = y_chunk[random_indices]

                X_chunk = preprocess_generated_array(X_chunk, scaler)

                if isinstance(X_test_trts, np.ndarray):
                    X_test_trts = np.append(X_test_trts, X_chunk, axis=0)
                    y_test_trts = np.append(y_test_trts, y_chunk, axis=0)
                else:
                    X_test_trts = X_chunk
                    y_test_trts = y_chunk

            y_predict_trts = model_trts.predict(X_test_trts)
            print(f"TRTS f_score: {f_score(y_test_trts, y_predict_trts)}")
            print(f"TRTS accuracy: {accuracy(y_test_trts, y_predict_trts)}")

            del X_test_trts, y_test_trts, model_trts
            gc.collect()

        # -------------------------------------------------------------
        # Evaluation 4: Train model with additional synthetic data
        # -------------------------------------------------------------
        if eval_four:
            # Train alpha model on alpha_subset
            print("Evaluation 3: Train model with additional synthetic data")
            model_alpha = build_model(n_epochs=10, n_features=recordings[0].sensor_frame.shape[1])
            model_alpha.fit(X_train, y_train)
            y_test_pred_model_alpha = model_alpha.predict(X_test)

            print(f"alpha_model f_score: {f_score(y_test, y_test_pred_model_alpha)}")

            experiment_folder_path = new_saved_experiment_folder(f'{subject}_alpha')
            create_conf_matrix(experiment_folder_path, y_test_pred_model_alpha, y_test)
            create_text_metrics(experiment_folder_path, y_test_pred_model_alpha, y_test, [accuracy, f_score])
            save_model_configuration(experiment_folder_path, model_alpha)

            # Train beta model on beta_subset
            model_beta = build_model(n_epochs=10, n_features=recordings[0].sensor_frame.shape[1])
            # TODO: Different oversampling strategies
            model_training(model_beta, files, X_train, y_train, scaler)
            y_test_pred_model_beta = model_beta.predict(X_test)

            print(f"beta_model f_score: {f_score(y_test, y_test_pred_model_beta)}")

            experiment_folder_path = new_saved_experiment_folder(f'{subject}_beta')
            create_conf_matrix(experiment_folder_path, y_test_pred_model_beta, y_test)
            create_text_metrics(experiment_folder_path, y_test_pred_model_beta, y_test, [accuracy, f_score])
            save_model_configuration(experiment_folder_path, model_beta)

        exit()

start(eval_two = True)
