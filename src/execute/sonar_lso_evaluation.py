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


def start(eval_one: bool = True, eval_two: bool = True, eval_three: bool = True) -> None:
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
        """
        Preprocess the generated data the same way as the real training data.
        """
        # Reshape array for normalization
        generated_activity_data = np.squeeze(generated_activity_data, -1)
        generated_activity_data = generated_activity_data.reshape(-1, 70)
        # Normalize data
        generated_activity_data = scaler.transform(generated_activity_data)
        # Inverse reshape data
        generated_activity_data = generated_activity_data.reshape(-1, WINDOW_SIZE, 70)
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
                # Get the tenths that haven't been yielded yet
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
    

    # GAN Newtork parameters
    parameters = dict()
    parameters['module'] = 'gru'  # LSTM possible
    parameters['hidden_dim'] = 280  # Paper: 4 times the size of input features
    parameters['num_layer'] = 3
    parameters['iterations'] = 7500  # Paper: 10.000
    parameters['batch_size'] = 64

    synth_data_path = "/dhc/groups/bp2021ba1/kirill/nursing-data-augmentation/src/mtss_data"
    files = get_file_paths_in_folder(synth_data_path)

    # Load data
    recordings = load_recordings(settings.sonar_dataset_path)

    # Activities to number
    for rec in recordings:
        rec.activities = rec.activities.map(lambda label: settings.ACTIVITIES[label])

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
        # Evaluation 1: Plotting PCA / tSNE distribution
        # -------------------------------------------------------------
        if eval_one:
            print("Evaluation 1: Plotting PCA / tSNE distributio")
            activities_one_hot_encoded = np.eye(15, 15)
            for (index, row) in enumerate(activities_one_hot_encoded):
                # Get all indices in y_train where the one-hot-encoded row is equal to row
                activity_group_indices = np.nonzero(np.all(np.isclose(y_train, row), axis=1))[0]
                activity_group_X = X_train[activity_group_indices]
                
                try:
                    generated_activity_data = np.load(f'{synth_data_path}/data_{subject}_{index}.npy')
                except OSError:
                    continue
                
                print(generated_activity_data.shape)
                plot_pca_distribution(activity_group_X, generated_activity_data, str(subject) + "_" + str(index))
                plot_tsne_distribution(activity_group_X, generated_activity_data, str(subject) + "_" + str(index))

        # -------------------------------------------------------------
        # Evaluation 2: TSTR / TRTS
        # -------------------------------------------------------------
        if eval_two:
            # Train on synthetic data, test on real data
            print("Evaluation 2: TSTR / TRTS")
            print("Training on synthetic data, testing on real data")
            model_tstr = build_model(n_epochs=10, n_features=recordings[0].sensor_frame.shape[1])
            X_test_trts = None
            y_test_trts = None
            for X_chunk, y_chunk in chunk_generator(files):
                X_chunk = preprocess_generated_array(X_chunk, scaler)

                if isinstance(X_test_trts, np.ndarray):
                    X_test_trts = np.append(X_test_trts, X_chunk, axis=0)
                    y_test_trts = np.append(y_test_trts, y_chunk, axis=0)
                else:
                    X_test_trts = X_chunk
                    y_test_trts = y_chunk

                X_test_trts, y_test_trts = shuffle(X_test_trts, y_test_trts)
                model_tstr.fit(X_test_trts, y_test_trts, ignore_epochs=True)

            # Choose one thousand random values from X_train for testing
            random_indices = random.sample(range(X_train.shape[0]), 1000)
            X_subset = X_train[random_indices]
            y_subset = y_train[random_indices]

            y_predict_tstr = model_tstr.predict(X_subset)
            print(f"TSTR f_score: {f_score(y_subset, y_predict_tstr)}")
            print(f"TSTR accuracy: {accuracy(y_subset, y_predict_tstr)}")

            del X_subset, y_subset, y_predict_tstr
            gc.collect()


            # Train on real data, test on synthetic data
            print("Training on real data, testing on synthetic data")
            model_trts = build_model(n_epochs=10, n_features=recordings[0].sensor_frame.shape[1])
            model_trts.fit(X_train, y_train)
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
        # Evaluation 3: Train model with additional synthetic data
        # -------------------------------------------------------------
        if eval_three:
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
            model_training(model_beta, files, X_train, y_train, scaler)
            y_test_pred_model_beta = model_beta.predict(X_test)

            print(f"alpha_beta f_score: {f_score(y_test, y_test_pred_model_beta)}")

            experiment_folder_path = new_saved_experiment_folder(f'{subject}_beta')
            create_conf_matrix(experiment_folder_path, y_test_pred_model_beta, y_test)
            create_text_metrics(experiment_folder_path, y_test_pred_model_beta, y_test, [accuracy, f_score])
            save_model_configuration(experiment_folder_path, model_beta)


start()
