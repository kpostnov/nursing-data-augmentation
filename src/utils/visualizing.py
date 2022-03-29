import tensorflow.keras as keras  # type: ignore
import tensorflow as tf  # type: ignore
from matplotlib import pyplot as plt

from utils.Recording import Recording


def visualizeAccuracy(history: tf.keras.callbacks.History) -> None:
    plt.figure()
    plt.plot(history.history["accuracy"])
    plt.plot(history.history["val_accuracy"])
    plt.title("model accuracy")
    plt.ylabel("accuracy")
    plt.xlabel("epoch")
    plt.legend(["train", "val"], loc="upper left")
    plt.show()


def visualizeLoss(history: tf.keras.callbacks.History, name: str) -> None:
    plt.figure()
    plt.plot(history.history["loss"])
    plt.plot(history.history["val_loss"])
    plt.title("training loss")
    plt.ylabel("loss")
    plt.xlabel("epoch")
    plt.legend(["train", "val"], loc="upper left")
    plt.show()
    plt.savefig(f"{name}.png")


def show_recording_sort(recordings: "list[Recording]") -> None:
    for recording in recordings:
        print(f"activity: {recording.activity}, subject: {recording.subject}")
