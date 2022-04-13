import os
from loader.load_nursing_dataset import load_nursing_train_dataset, load_nursing_test_dataset
import utils.settings as settings


# Load data
recordings = load_nursing_train_dataset(settings.nursing_dataset_path)
