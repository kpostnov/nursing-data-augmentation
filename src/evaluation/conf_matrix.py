from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
import itertools
import os


def create_conf_matrix(path: str, y_test_pred: np.ndarray, y_test_true: np.ndarray) -> None:
    """
    Creates and saves confusion matrix as .png to path 
    """

    print("Creating confusion matrix")

    classes = list(range(len(y_test_true[0]))) # [0, 1, 0, 0, 0] to [0, 1, 2, 3, 4]
    y_test_true = np.argmax(y_test_true, axis=1)
    y_test_pred = np.argmax(y_test_pred, axis=1)

    # Settings
    cm = confusion_matrix(y_test_true, y_test_pred)
    normalize = False
    title = 'Confusion matrix'
    cmap = plt.cm.Blues
    
    # Create Conf-Plot
    if normalize:
        cm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

    f3 = plt.figure(3)
    plt.imshow(cm, interpolation="nearest", cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = ".2f" if normalize else "d"

    thresh = cm.max() / 2.0
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(
            j,
            i,
            format(cm[i, j], fmt),
            horizontalalignment="center",
            color="white" if cm[i, j] > thresh else "black",
        )
        plt.tight_layout()
        plt.ylabel("True label")
        plt.xlabel("predicted label")
        f3.show()

    plt.savefig(os.path.join(path, "conf_matrix.png"))
