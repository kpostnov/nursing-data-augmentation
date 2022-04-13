"""
test_compare_model_input.py
 - this test wants to monitor, how similar the windows are that get passed in jens pipeline vs. our rebuild of it
 - what does the model see? different windows, different preprocessing?
 - in both, the test_train_split is skipped
"""

import os
import random
import h5py
import numpy as np
from loader.load_opportunity_dataset import load_opportunity_dataset
from loader.Preprocessor import Preprocessor
import utils.settings as settings
from utils.array_operations import split_list_by_percentage
from models.JensModel import JensModel
from models.RainbowModel import RainbowModel
from utils.progress_bar import print_progress_bar


def our_X_y() -> (np.ndarray, np.ndarray):
    # Load data
    recordings = load_opportunity_dataset(
        settings.opportunity_dataset_path
    )  # Refactoring idea: load_dataset(x_sens_reader_func, path_to_dataset)
    random.seed(1678978086101)
    random.shuffle(recordings)

    # Preprocessing
    recordings = Preprocessor().jens_preprocess(recordings)

    # Init
    model = JensModel(
        window_size=25,
        n_features=recordings[0].sensor_frame.shape[1],
        n_outputs=6,
        verbose=1,
        n_epochs=10,
    )  # random init, to get windowize func
    windows = model.windowize(
        recordings
    )  # we dont use jens convert as it expands a dimension, does categorical

    # !!!! From Rainbow Model convert !!!
    X = np.array(list(map(lambda window: window.sensor_array, windows)))
    y = np.array(list(map(lambda window: window.activity, windows)))

    return X, y


def jens_X_y() -> (np.ndarray, np.ndarray):
    """
    - execute data_processing.py before, this function just loads the windows, doesnt do the preprocessing again
    """
    preprocessed_h5_data_path = "research/jensOpportunityDeepL/hl_2.h5"
    file = h5py.File(preprocessed_h5_data_path, "r")
    X = file.get("inputs")
    y = file.get("labels")

    return X, y


def n_duplicate_windows(windows: np.ndarray) -> int:
    """
    Very uneffcient!!!
    - windows.shape: (n_windows, window_size, window_size, n_features)
    """
    unique_duplicate_windows = []

    def already_counted_as_duplicate(window):
        for unique_duplicate_window in unique_duplicate_windows:
            if np.array_equal(window, unique_duplicate_window):
                return True
        return False

    n_duplicate_windows = 0
    n_windows = windows.shape[0]
    for i, current_window in enumerate(windows):
        print_progress_bar(i, n_windows, prefix="n_duplicate_windows")
        found_duplicate = False
        for j in range(i + 1, n_windows):
            if np.array_equal(
                current_window, windows[j]
            ) and not already_counted_as_duplicate(current_window):
                found_duplicate = True
                n_duplicate_windows += 1
        if found_duplicate:
            unique_duplicate_windows.append(current_window)
    print_progress_bar(n_windows, n_windows, prefix="n_duplicate_windows")

    return n_duplicate_windows


def test_n_duplicate_windows():
    example_array = np.array([[1, 2], [5, 6], [7, 8], [1, 2], [1, 2], [1, 2], [7, 8]])
    assert 4 == n_duplicate_windows(
        example_array
    ), "n_duplicate_windows is working wrong"


def intersection_np_axis_0(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """
    - a.shape: (n_windows, window_size, n_features)
    - b.shape: (n_windows, window_size, n_features)
    """
    pass
    # Version 1
    # n_windows = a.shape[0]
    # intersection = np.zeros(a.shape)
    # for i in range(n_windows):
    #     print_progress_bar(i, n_windows, prefix="intersection_np_axis_0")
    #     intersection[i] = np.logical_and(a[i], b[i])
    # print_progress_bar(n_windows, n_windows, prefix="intersection_np_axis_0")

    # return intersection

    # Version 2
    # nrows, ncols = A.shape
    # dtype={'names':['f{}'.format(i) for i in range(ncols)],
    #     'formats':ncols * [A.dtype]}

    # C = np.intersect1d(A.view(dtype), B.view(dtype))

    # # This last bit is optional if you're okay with "C" being a structured array...
    # return C.view(A.dtype).reshape(-1, ncols)


def test_intersection_np_axis_0():
    # window identifier on [0][0] => len(window_identifier) == n_windows
    to_window_identifier = lambda windows: list(
        map(lambda window: window[0][0], windows)
    )

    # example numpy array a of shape (7 (windows), 3 (window_size rows), 2 (n_features columns))
    a = np.array(
        [
            [[1, 2], [3, 4], [5, 6]],
            [[19, 20], [21, 22], [23, 24]],
            [[7, 8], [9, 10], [11, 12]],
            [[13, 14], [15, 16], [17, 18]],
            [[7, 8], [9, 10], [11, 12]],
            [[19, 20], [21, 22], [23, 24]],
            [[7, 8], [9, 10], [11, 12]],
        ]
    )
    window_identifier_a = [1, 19, 7, 13, 7, 19, 7]
    assert window_identifier_a == to_window_identifier(a)

    # example numpy array b of shape (4 (windows), 3 (window_size rows), 2 (n_features columns))
    b = np.array(
        [
            [[1, 2], [3, 4], [5, 6]],
            [[7, 8], [9, 10], [11, 12]],
            [[13, 14], [15, 16], [17, 18]],
            [[19, 20], [21, 22], [23, 24]],
        ]
    )
    window_identifier_b = [1, 7, 13, 19]
    assert window_identifier_b == to_window_identifier(b)

    # Intersection
    py_list_intersection = lambda list1, list2: list(
        filter(lambda x: x in list2, list1)
    )  # filter true: can stay
    window_identifier_intersection = py_list_intersection(
        window_identifier_a, window_identifier_b
    )
    intersection_a_b = intersection_np_axis_0(a, b)
    assert window_identifier_intersection == to_window_identifier(intersection_a_b)


# # Start
# settings.init()

# # Load data
# X_our, y_our = our_X_y() # our labels are categorical
# print('Shape of X_our:', X_our.shape) # (53320, 25, 51)
# X_jens, y_jens = jens_X_y()
# print('Shape of X_jens:', X_jens.shape) # (49484, 25, 51)

# # Unique
# X_our_unique = np.unique(X_our, axis=0)
# print('Shape of X_our_unique:', X_our_unique.shape) # (53187, 25, 51) we have 133 duplicate windows
# X_jens_set = np.unique(X_jens, axis=0)
# print('Shape of X_jens_set:', X_jens_set.shape) # (49484, 25, 51) no duplicate windows

test_intersection_np_axis_0()  # TODO: write a test_intersection_np_axis_0


"""
Refactoring idea (would make it faster):
https://stackoverflow.com/questions/8317022/get-intersecting-rows-across-two-2d-numpy-arrays

    import numpy as np

    A = np.array([[1,4],[2,5],[3,6]])
    B = np.array([[1,4],[3,6],[7,8]])

    nrows, ncols = A.shape
    dtype={'names':['f{}'.format(i) for i in range(ncols)],
        'formats':ncols * [A.dtype]}

    C = np.intersect1d(A.view(dtype), B.view(dtype))

    # This last bit is optional if you're okay with "C" being a structured array...
    C = C.view(A.dtype).reshape(-1, ncols)
"""
