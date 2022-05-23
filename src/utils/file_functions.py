import os


def get_subfolder_names(path):
    return [f.name for f in os.scandir(path) if f.is_dir()]

def get_file_names(path):
    return [f.name for f in os.scandir(path) if f.is_file()]

def get_file_paths_in_folder(path):
    return [os.path.join(path, f) for f in get_file_names(path)]
