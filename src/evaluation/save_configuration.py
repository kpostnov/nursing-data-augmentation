from models import RainbowModel
import os

def save_model_configuration(experiment_folder_path: str, model: RainbowModel):
    """
    Saves the model configuration to a given path.
    """
    print("Saving model configuration ...")

    with open(os.path.join(experiment_folder_path, 'model_config.txt'), "w") as f:
        f.write(f"Class name: {model.__class__.__name__}\n")
        f.write(f"Window size: {model.window_size}\n")
        f.write(f"Stride size: {model.stride_size}\n")
        f.write(f"Batch size: {model.batch_size}\n")
        f.write(f"Epochs: {model.n_epochs}\n")

        for key, value in model.kwargs.items():
            f.write(f"{key}: {value}\n")
