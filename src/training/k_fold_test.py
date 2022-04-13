from typing import Callable

import numpy as np
from sklearn.model_selection import KFold

from evaluation.Evaluation import Evaluation
from evaluation.EvaluationTestResult import EvaluationTestResult
from models.RainbowModel import RainbowModel
from utils.Recording import Recording


def k_fold_cross_validation_test(
    model_builder: Callable[[], RainbowModel],
    model_nickname: str,
    recordings_in: "list[Recording]",
    k: int,
    data_augmentation_func=None,
) -> "(list[EvaluationTestResult], np.ndarray)":
    """
    - fits the model k times on different splits of the data
    - returns the EvaluationTestReports for each split
    """
    evaluation_results: "list[EvaluationTestResult]" = []

    recordings: np.ndarray = np.array(recordings_in)  # for comfortable index splitting
    k_fold = KFold(n_splits=k, random_state=None)
    for idx, (train_index, test_index) in enumerate(k_fold.split(recordings)):
        print(f"Starting fold {idx}")
        recordings_train, recordings_test = (
            recordings[train_index],
            recordings[test_index],
        )

        model = model_builder()
        model.windowize_convert_fit(recordings_train)

        # Evaluate model
        evaluation_results.append(
            Evaluation.test(model, model_nickname, recordings_test)
        )

    return evaluation_results
