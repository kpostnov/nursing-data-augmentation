from datetime import datetime
import os
import utils.settings as settings

def new_saved_experiment_folder(experiment_name):
    """
    Creates a new folder in the saved_experiments
    :return:s the path -> you can save the model, evaluations (report, conf matrix, ...)
    """

    # folder name
    currentDT = datetime.now()
    currentDT_str = currentDT.strftime("%y-%m-%d_%H-%M-%S_%f")
    folder_name = currentDT_str + "-" + experiment_name

    path_to_experiment_folder = os.path.join(settings.saved_experiments_path, folder_name)
    os.makedirs(path_to_experiment_folder, exist_ok=True)
    # if not os.path.exists(path_to_experiment_folder):
    return path_to_experiment_folder
