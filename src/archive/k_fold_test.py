from pyclbr import Function
from evaluation.Evaluation import Evaluation
from evaluation.EvaluationTestResult import EvaluationTestResult
from loader.Preprocessor import Preprocessor
from sklearn.model_selection import KFold, train_test_split
from loader.filter_dataset import filter_acceleration
from loader.load_dataset import load_dataset
import os
from models.RainbowModel import RainbowModel
from utils.Recording import Recording
from utils.array_operations import split_list_by_percentage
import utils.settings as settings
import random
import numpy as np


def k_fold_cross_validation_test(model: RainbowModel, model_nickname: str, recordings_in: list[Recording], k: int, data_augmentation_func) -> 'list[EvaluationTestResult]':
    """
    - fits the model k times on different splits of the data
    - returns the EvaluationTestReports for each split
    """
    evaluation_results: list[EvaluationTestResult] = []

    initial_weights = model.model.get_weights()

    recordings: np.ndarray = np.array(recordings_in)  # for comfortable index splitting
    k_fold = KFold(n_splits=k, random_state=None)
    for idx, (train_index, test_index) in enumerate(k_fold.split(recordings)):
        recordings_tuple: 'tuple[np.ndarray, np.ndarray]' = recordings[train_index], recordings[test_index]
        if data_augmentation_func:
            recordings_tuple = data_augmentation_func(recordings[train_index]), recordings[test_index]
        recordings_train, recordings_test = recordings_tuple

        model.windowize_convert_fit(recordings_train.tolist())

        model.save_model(os.path.join(settings.ML_RAINBOW_PATH, f'saved_models'), name_suffix=f'{idx}')

        evaluation_results.append(Evaluation.test(model, model_nickname, recordings_test.tolist()))

        # reset weights to initial
        model.model.set_weights(initial_weights)

    return evaluation_results
