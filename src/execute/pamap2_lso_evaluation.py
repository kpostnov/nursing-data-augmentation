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
from loader.preprocessing import interpolate_ffill, preprocess, normalize_standardscaler, interpolate_linear
from utils.file_functions import get_file_paths_in_folder
from models.RainbowModel import RainbowModel
from loader.load_pamap2_dataset import load_pamap2_dataset
from models.AdaptedDeepConvLSTM import AdaptedDeepConvLSTM
from datatypes.Window import Window
from utils.Windowizer import Windowizer
from utils.folder_operations import new_saved_experiment_folder
import utils.settings as settings
from visualization.visualize import plot_pca_distribution, plot_tsne_distribution
from utils.chunk_generation import chunk_generator


def start(eval_one: bool = False, eval_two: bool = False, eval_three: bool = False, eval_four: bool = False) -> None:
    WINDOW_SIZE = 100
    STRIDE_SIZE = 100

    def build_model(n_epochs : int, n_features: int) -> RainbowModel:
        return AdaptedDeepConvLSTM(
            window_size=WINDOW_SIZE,
            stride_size=STRIDE_SIZE,
            n_features=n_features,
            n_outputs=len(settings.ACTIVITIES),
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
    

    synth_data_path = "/dhc/groups/bp2021ba1/kirill/nursing-data-augmentation/src/pamap2"
    files = get_file_paths_in_folder(synth_data_path)

    # Load data
    recordings = load_pamap2_dataset(settings.pamap2_dataset_path)

    random.seed(1678978086101)
    random.shuffle(recordings)

    # Preprocessing
    recordings, scaler = preprocess(recordings, methods=[
        interpolate_ffill,
        normalize_standardscaler
    ])

    # Windowize all recordings
    windowizer = Windowizer(WINDOW_SIZE, STRIDE_SIZE, Windowizer.windowize_sliding, 60)

    # LOSO-folds
    subject_ids = range(1, 9)
    for subject_id in subject_ids:
        print(f"LOSO-fold without subject: {subject_id}")

        # Remove recordings where recording.subject_id == subject_id
        # Remove recordings where recording.subject_id == subject_id
        alpha_subset = [recording for recording in recordings if recording.subject != str(subject_id)]
        validation_subset = [recording for recording in recordings if recording.subject == str(subject_id)]
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
                    random_distribution = np.load(f'{synth_data_path}/random_data/random_data_{subject}_0_{WINDOW_SIZE}.npy')
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
            # TODO: Train on different proportions of generated activity data (5k, 5k - most ...)
            model_training(model_beta, files, X_train, y_train, scaler)
            y_test_pred_model_beta = model_beta.predict(X_test)

            print(f"beta_model f_score: {f_score(y_test, y_test_pred_model_beta)}")

            experiment_folder_path = new_saved_experiment_folder(f'{subject}_beta')
            create_conf_matrix(experiment_folder_path, y_test_pred_model_beta, y_test)
            create_text_metrics(experiment_folder_path, y_test_pred_model_beta, y_test, [accuracy, f_score])
            save_model_configuration(experiment_folder_path, model_beta)

        exit()

start(eval_two = True)