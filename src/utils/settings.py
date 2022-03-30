import os


def init():
    """
    Refactoring idea:
    - pass the mapping, that we can easily switch between datasets and labels
    """

    global opportunity_dataset_path
    opportunity_dataset_path = "opportunity-dataset"

    global activity_initial_num_to_activity_str
    activity_initial_num_to_activity_str = {
        0: "null",
        101: "relaxing",
        102: "coffee time",
        103: "early morning",
        104: "cleanup",
        105: "sandwich time",
    }

    global activity_initial_num_to_activity_idx
    activity_initial_num_to_activity_idx = {
        0: 0,
        101: 1,
        102: 2,
        103: 3,
        104: 4,
        105: 5,
    }

