"""
    Jens Model from the research folder implemented in our RainbowModel architecture
    Aims to get the same accuracy as Jens (similar windowize, no kfold)

    Opportunity activities
        0: "null",
        1: "relaxing",
        2: "coffee time",
        3: "early morning",
        4: "cleanup",
        5: "sandwich time"

    TODO:
        - problem before training
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
        - crashes... new convert override function needed
        - delete fit override?
        - why different number of windows?
        - why different number of features?
        - why not categorial input?

"""

import os
import random
from loader.load_opportunity_dataset import load_opportunity_dataset
from loader.Preprocessor import Preprocessor
import utils.settings as settings
from utils.array_operations import split_list_by_percentage
from models.JensModel import JensModel
from utils.folder_operations import new_saved_experiment_folder


settings.init()

# Load data
recordings = load_opportunity_dataset(settings.opportunity_dataset_path) # Refactoring idea: load_dataset(x_sens_reader_func, path_to_dataset)
random.seed(1678978086101)
random.shuffle(recordings)

# TODO: apply recording label filter functions

# Preprocessing
recordings = Preprocessor().jens_preprocess(recordings)

# TODO: save/ load preprocessed data

# Test Train Split
test_percentage = 0.3
recordings_train, recordings_test = split_list_by_percentage(recordings, test_percentage)

# Init, Train
model = JensModel(window_size=25, n_features=recordings[0].sensor_frame.shape[1], n_outputs=6, verbose=1)
model.windowize_convert_fit(recordings_train)

# Test, Evaluate
# labels are always in vector format
X_test, y_test_true = model.windowize_convert(recordings_test)
y_test_pred = model.predict(X_test)

experiment_folder_path = new_saved_experiment_folder('opportunity_jens_cnn') # create folder to store results

model.export(experiment_folder_path) # opt: export model to folder


create_conf_matrix(experiment_folder_path, y_pred, y_true) # TODO: This is not implemented yet

# TODO: other metrics like accuracy
