from utils.folder_operations import new_saved_experiment_folder
import utils.settings as settings
import os

settings.init()

test_folder_name = 'test_folder_07'

# function to test
new_saved_experiment_folder(test_folder_name)

# check, if folder exists
experiment_folder_names = os.listdir(settings.saved_experiments_path)
filter_func = lambda folder_name: folder_name.ends_with(test_folder_name)
test_folder_names = list(filter(filter_func, experiment_folder_names))
assert len(test_folder_names) > 0, 'folder was not created!'

# remove created folders
for folder_name in test_folder_names:
    os.rmdir(os.path.join(settings.saved_experiments_path, folder_name))