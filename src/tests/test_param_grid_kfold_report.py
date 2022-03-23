
from utils.telegram import send_telegram
from utils.markdown import markdown_table_str
from utils import settings
from rainbow_test.k_fold_test import k_fold_cross_validation_test
from rainbow_test.create_report import create_rainbow_report, create_send_save_kfold_report, k_fold_report_str, send_telegram_report
from evaluation.EvaluationTestResult import EvaluationTestResult
import random
from importlib_metadata import itertools  # type: ignore
import numpy as np
from augmentation.rotate import append_recordings_randomly_rotated_n_times
from loader.Preprocessor import Preprocessor
from loader.convert_dataset import convert_quaternion_to_euler
from loader.filter_dataset import filter_acceleration
from loader.load_dataset import load_dataset
from models.LSTMModel import LSTMModel
import utils.settings as settings
import tensorflow as tf
import os
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# tf.get_logger().setLevel('WARNING')
# tf.autograph.set_verbosity(1)

def do_k_fold(model: RainbowModel, model_nickname: str, recordings_in: list[Recording], k: int) -> 'list[EvaluationTestResult]':
    """
    - fits the model k times on different splits of the data
    - returns the EvaluationTestReports for each split
    """
    evaluation_results: list[EvaluationTestResult] = []

    initial_weights = model.model.get_weights()
    recordings: np.ndarray = np.array(recordings_in)  # for comfortable index splitting
    k_fold = KFold(n_splits=k, random_state=None)
    for idx, (train_index, test_index) in enumerate(k_fold.split(recordings)):
        # test train split
        recordings_tuple: 'tuple[np.ndarray, np.ndarray]' = recordings[train_index], recordings[test_index]
        recordings_train, recordings_test = recordings_tuple

        # fit
        model.model.set_weights(initial_weights) # set weights to initial
        model.windowize_convert_fit(recordings_train.tolist())

        # test
        X_test, y_test = model.windowize_convert(recordings_test.tolist())
        prediction_vectors = model.predict(X_test)
        evaluation_results.append(Evaluation.get_evaluation_test_result(model, model_nickname, X_test, y_test, prediction_vectors))

    return evaluation_results


def param_grid_kfold_experiment(k_fold_k=4, telegram=True) -> list[list[EvaluationTestResult]]:

    # load data
    raw_recordings = load_dataset(os.path.join(settings.DATA_PATH, '5-sensor-all'))
    random.seed(1678978086101)
    random.shuffle(raw_recordings)

    # preprocess
    recordings = filter_acceleration(raw_recordings)
    preprocessor = Preprocessor()
    recordings = preprocessor.preprocess(recordings)

    # param grid
    window_sizes = [100, 180]
    hyper_param_sets = [
        {'epochs': 10},
        {'epochs': 20},
    ]

    # kfold for each combination in param grid
    model_nicknames: list[str] = []
    models_evaluation_results: list[list[EvaluationTestResult]] = []
    for window_size, hyper_param_set in itertools.product(window_sizes, hyper_param_sets):

        # define model depending on grid params
        n_features = recordings[0].sensor_frame.shape[1]
        model = LSTMModel(window_size=window_size, stride_size=180, test_percentage=0.2, n_features=n_features, n_outputs=len(settings.ACTIVITIES), **hyper_param_set)
        model_nickname = f"{augmentation_name} @ {hyper_param_set}"
        model_nicknames.append(model_nickname)

        # do kfold, evaluate
        results: list[EvaluationTestResult] = do_k_fold(model, model_nickname, recordings, k_fold_k)
        models_evaluation_results.append(results)

        # telegram update for kfold of one combination
        if telegram:
            avg_correct_classification_accuracy = str(sum([result.correct_classification_accuracy for result in results]) / len(results))
            avg_failure_rate = str(sum([result.average_failure_rate for result in results]) / len(results))

            send_telegram('Model: ' + model_nickname + ' | k_fold_cross_validation_test done (' + str(
                k) + ' models on different test_train_splits trained and tested)\navg_correct_classification_accuracy ' + avg_correct_classification_accuracy + '\navg_failure_rate ' + avg_failure_rate)
    
    return models_evaluation_results



# start --------------------------------------------------------------------------------------------------------------

settings.init()
models_evaluation_results: list[list[EvaluationTestResult]] = param_grid_kfold_experiment(k_fold_k=4, telegram=False)

KFoldMarkdownReport.create_send_save_kfold(
    title = 'test_param_grid_kfold_report', 
    description = 'test the pipeline', 
    models_evaluation_results = models_evaluation_results, 
    telegram = False
)
