import numpy as np
from sklearn.metrics import f1_score


def f_score(prediction_vectors: np.ndarray, y_test: np.ndarray) -> float:
    f_prediction_vectors = np.argmax(prediction_vectors, axis=1)
    f_y_test = np.argmax(y_test, axis=1)
    return f1_score(f_y_test, f_prediction_vectors)


def accuracy(prediction_vectors: np.ndarray, y_test: np.ndarray, verbose: int = 0) -> float:
    predictions = np.argmax(prediction_vectors, axis=1)
    acc_y_test = np.argmax(y_test, axis=1)
    accuracy = np.sum(predictions == acc_y_test) / len(predictions)
    if verbose:
        print(f"accuracy: {accuracy}")

    return accuracy


def average_failure_rate(prediction_vectors: np.ndarray, y_test: np.ndarray) -> float:
    """
    output y_test [0.03, 0.5, 0.3], correct label idx 2
    -> how much is missing to 1.0?
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
