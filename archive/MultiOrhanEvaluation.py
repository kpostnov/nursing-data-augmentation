from copy import deepcopy
from unittest import TestResult
from evaluation.Evaluation import Evaluation
from evaluation.MarkdownTestResult import MarkdownTestResult
from models.MultiOrhan import MultiOrhan
from models.RainbowModel import RainbowModel
import numpy as np
from utils import settings

from utils.Recording import Recording
from utils.typing import assert_type
import os
from functools import reduce


class MultiOrhanEvaluator(Evaluation):
    def test(
        self,
        model: "MultiOrhan",
        model_nickname: str,
        recordings_test: "list[Recording]",
    ) -> MarkdownTestResult:
        """
        windowizes_converts the recordings_test's to fit to the models input shape
        uses this data to test the perfomance of the model
        """
        X_test, y_test = model.windowize_convert(recordings_test)
        recordings_prediction_vectors = model.predict(X_test)

        context_accuracy = self.__context_accuracy(
            recordings_prediction_vectors, y_test
        )

        plot_recording_window_accuracy(
            recordings_prediction_vectors,
            y_test,
            os.path.join(settings.ML_RAINBOW_PATH, "rainbow_test", "report"),
        )

        # destroys recording context!!
        prediction_vectors = np.array(
            reduce(lambda a, b: np.concatenate((a, b)), recordings_prediction_vectors)
        )  # type:ignore
        y_test = np.array(
            reduce(lambda a, b: np.concatenate((a, b)), y_test.tolist())
        )  # type:ignore

        test_activity_distribution = self._count_unique_activities(y_test)
        correct_classification_accuracy = self._correct_classification_accuracy(
            prediction_vectors, y_test
        )
        average_failure_rate = self._average_failure_rate(prediction_vectors, y_test)

        # Calculate F1 Score
        # f1_score = Evaluation.__f1_score(model, X_test, y_test)

        return MarkdownTestResult(
            model,
            model_nickname,
            test_activity_distribution,
            correct_classification_accuracy,
            average_failure_rate,
            context_accuracy,
        )

    def __context_accuracy(
        self, recordings_prediction_vectors: list[np.ndarray], y_test: np.ndarray
    ) -> dict:

        """
        needs recording context

        "{recording_idx} - {number_of_windows}w - {most_window_activity}, {second_most_window_activity}" : {number_of_argmax_correct_classified_windows} / {number_of_windows}

        {
            "0 - 5w - running, walking": 0.8,
            "1 - 10w - running, walking": 0.8,
        }
        """

        context_accuracy: dict[str, float] = {}
        for i in range(len(y_test)):
            recording_activity_vectors = y_test[i]
            recording_prediction_vectors = recordings_prediction_vectors[i]

            # get first and second most activity in recording
            bincount_stuff: list[int] = np.bincount(
                np.argmax(recording_activity_vectors, axis=1)
            )  # type:ignore
            first_most_index = np.argmax(bincount_stuff)
            first_most_activity = settings.ACTIVITIES_ID_TO_NAME[first_most_index]

            bincount_stuff[first_most_index] = 0
            second_most_activity = settings.ACTIVITIES_ID_TO_NAME[
                np.argmax(bincount_stuff)
            ]

            number_of_correct_classified_windows = 0
            for j in range(len(recording_prediction_vectors)):
                window_prediction_idx = np.argmax(recording_prediction_vectors[j])
                window_activity_idx = np.argmax(recording_activity_vectors[j])
                if window_prediction_idx == window_activity_idx:
                    number_of_correct_classified_windows += 1

            correct_classified_windows_percentage = round(
                number_of_correct_classified_windows
                / len(recording_prediction_vectors),
                ndigits=2,
            )

            recording_description = f"{i} - {len(recording_activity_vectors.tolist())}w - {first_most_activity}, {second_most_activity}"
            context_accuracy[
                recording_description
            ] = correct_classified_windows_percentage

        return context_accuracy
