from dataclasses import dataclass

from models.RainbowModel import RainbowModel


@dataclass
class EvaluationTestResult:
    """
    one model one test dataset
    """

    def __init__(
        self,
        model: RainbowModel,
        model_nickname: str,
        test_activity_distribution: dict,
        correct_classification_accuracy: float,
        average_failure_rate: float,
        context_accuracy: dict = None,
    ):
        """
        test_activity_distribution: dict with the number of occurences of each activity
        correct_classification_accuracy: float between 0 and 1
        average_failure_rate: float between 0 and 1
        """
        self.model = model
        self.model_nickname: str = model_nickname

        self.test_activity_distribution = test_activity_distribution
        self.correct_classification_accuracy = correct_classification_accuracy
        self.average_failure_rate = average_failure_rate
        self.context_accuracy = context_accuracy

    def __str__(self):
        return "TestResult(test_activity_distribution={}, correct_classification_accuracy={}, average_failure_rate={})".format(
            self.test_activity_distribution,
            self.correct_classification_accuracy,
            self.average_failure_rate,
        )
