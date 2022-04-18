from tensorflow import keras
from models import RainbowModel


def load_model(model_path: str) -> keras.models.Sequential:
    """
    Loads a model from a given path.
    """
    print("Loading model from: " + model_path)
    model = keras.models.load_model(model_path)

    return model
