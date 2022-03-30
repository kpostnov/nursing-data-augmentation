import os
import random
from loader.load_opportunity_dataset import load_opportunity_dataset
from loader.Preprocessor import Preprocessor
import utils.settings as settings

settings.init()

recordings = load_opportunity_dataset(settings.opportunity_dataset_path)

random.seed(1678978086101)
random.shuffle(recordings)

recordings = Preprocessor().jens_preprocess(recordings)

model = JensModel() # TODO: improve, clean implementation
# model_folder_path = oppo.save_model(current_path_in_repo, model_name)

# oppo.draw(model_folder_path)  # todo: no line visible at the moment

# oppo.evaluation(model_folder_path)  # plots acc and confusion matrix
