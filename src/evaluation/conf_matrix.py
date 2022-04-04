from sklearn.metrics import confusion_matrix
import numpy as np


def create_conf_matrix(path: str, y_pred: np.ndarray, y_true: np.ndarray) -> None:
    """
    creates and saves conf matrix as .png to path 
    """

    # TODO!
    

    #     self,
    #     cm,
    #     classes,
    #     path_to_model_folder,
    #     normalize=False,
    #     title="Confusion matrix",
    #     cmap=plt.cm.Blues,
    # ):


    # cm = confusion_matrix(self.y_test, self.y_pred)
    #     self.plot_confusion_matrix(
    #         cm, list(range(K)), model_folder_path
    #     )
    
    #     if normalize:
    #         cm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
    #         print("Normalized confusion matrix")
    #     else:
    #         print("Confusion matrix, without normalization")
    #     print(cm)
    #     f3 = plt.figure(3)
    #     plt.imshow(cm, interpolation="nearest", cmap=cmap)
    #     plt.title(title)
    #     plt.colorbar()
    #     tick_marks = np.arange(len(classes))
    #     plt.xticks(tick_marks, classes, rotation=45)
    #     plt.yticks(tick_marks, classes)