import os
import random
from loader.load_opportunity_dataset import load_opportunity_dataset
from loader.Preprocessor import Preprocessor
import utils.settings as settings
from utils.array_operations import split_list_by_percentage
from models.JensModel import JensModel

settings.init()

recordings = load_opportunity_dataset(settings.opportunity_dataset_path) # Refactoring idea: load_dataset(x_sens_reader_func, path_to_dataset)

random.seed(1678978086101)
random.shuffle(recordings)

recordings = Preprocessor().jens_preprocess(recordings)

# Test Train Split
test_percentage = 0.3
recordings_train, recordings_test = split_list_by_percentage(recordings, test_percentage)

"""
    0: "null",
    1: "relaxing",
    2: "coffee time",
    3: "early morning",
    4: "cleanup",
    5: "sandwich time"
"""

"""
Jens Repo:
    self.x_train.shape
    (24471, 25, 51, 1)
    self.y_train.shape
    (24471,)
This:
    X_train.shape
    (35086, 35, 71)
    y_train.shape
    (35086, 6)

TODO (run fit):
- crashes... new convert override function needed
- delete fit override?
- why different number of windows?
- why different number of features?
- why not categorial input?

"""
model = JensModel(window_size=25, n_features=recordings[0].sensor_frame.shape[1], n_outputs=6)
model.windowize_convert_fit(recordings_train)



# TODO: Evaluate performance

# model_folder_path = oppo.save_model(current_path_in_repo, model_name)

# oppo.draw(model_folder_path)  # todo: no line visible at the moment

# oppo.evaluation(model_folder_path)  # plots acc and confusion matrix
