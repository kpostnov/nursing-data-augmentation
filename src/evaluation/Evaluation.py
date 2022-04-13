import numpy as np

from evaluation.EvaluationTestResult import EvaluationTestResult
from models.RainbowModel import RainbowModel
from utils import settings
from utils.Recording import Recording


class Evaluation:
    def __init__(self):
        pass

    @staticmethod
    def test(
        model: "RainbowModel",
        model_nickname: str,
        recordings_test: "list[Recording]",
    ) -> EvaluationTestResult:
        """
        windowizes_converts the recordings_test's to fit to the models input shape
        uses this data to test the perfomance of the model
        """
        X_test, y_test = model.windowize_convert(recordings_test)
        prediction_vectors = model.predict(X_test)

        test_activity_distribution = Evaluation._count_unique_activities(y_test)
        correct_classification_accuracy = Evaluation._correct_classification_accuracy(
            prediction_vectors, y_test
        )
        average_failure_rate = Evaluation._average_failure_rate(
            prediction_vectors, y_test
        )

        # Calculate F1 Score
        # f1_score = Evaluation.__f1_score(model, X_test, y_test)

        return EvaluationTestResult(
            model,
            model_nickname,
            test_activity_distribution,
            correct_classification_accuracy,
            average_failure_rate,
        )

    @staticmethod
    def _count_unique_activities(y_test: np.ndarray) -> dict:
        """
        Counts the unique activites from this weird format
        """
        y_test_idx = np.argmax(y_test, axis=1)
        unique_values = np.unique(y_test_idx, return_counts=True)
        unique_counts = {}
        for i in range(len(unique_values[0])):
            activity_idx = unique_values[0][i]
            activity_name = settings.ACTIVITIES_ID_TO_NAME[activity_idx]
            unique_counts[activity_name] = unique_values[1][i]
        return unique_counts

    @staticmethod
    def _average_failure_rate(
        prediction_vectors: np.ndarray, y_test: np.ndarray
    ) -> float:
        """
        how much percent is missing to 100% for the correct activity - on the given test data
        """

        label_indices = []
        failure_sum = 0

        # get indices of correct labels
        for i in range(len(y_test)):
            label_indices.append(np.argmax(y_test[i]))  # [2, 1, 0, 3, ...]

        # sum up failure rate by calculating "1 - the prediction value of row i and expected column"
        for i in range(len(label_indices)):
            failure_sum += 1 - prediction_vectors[i][label_indices[i]]

        average_failure_rate = failure_sum / len(label_indices)
        return average_failure_rate

    @staticmethod
    def _f1_score(
        model: "RainbowModel", X_test: np.ndarray, y_test: np.ndarray
    ) -> float:
        """
        Calculates the f1 score of the model on the given test data
        """

        label_indices = []
        prediction_vecs = []
        f1_sum = 0

        # get indices of correct labels
        for i in range(len(y_test)):
            label_indices.append(np.argmax(y_test[i]))

        prediction_vecs = model.model.predict(X_test, batch_size=1)

        # sum up f1 score by calculating "2 * (prediction value of row i and expected column) / (prediction value of row i + expected column)"
        for i in range(len(label_indices)):
            f1_sum += (2 * prediction_vecs[i][label_indices[i]]) / (
                prediction_vecs[i][label_indices[i]] + y_test[i][label_indices[i]]
            )

        f1_score = f1_sum / len(label_indices)
        return f1_score

    @staticmethod
    def _correct_classification_accuracy(
        prediction_vectors: np.ndarray, y_test: np.ndarray, verbose: int = 0
    ) -> float:
        """
        returns the accuracy of the model on the test data
        """

        predictions = np.argmax(prediction_vectors, axis=1)
        y_test = np.argmax(y_test, axis=1)

        # calculate accuracy
        accuracy = np.sum(predictions == y_test) / len(predictions)
        if verbose:
            print(f"accuracy: {accuracy}")
        return accuracy
