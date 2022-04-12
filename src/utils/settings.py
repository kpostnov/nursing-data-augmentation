import os


def init():

    global opportunity_dataset_path
    opportunity_dataset_path = "../../datasets/OpportunityUCIDataset"

    global pamap2_dataset_path
    pamap2_dataset_path = "../../datasets/PAMAP2_Dataset"

    global nursing_dataset_path
    nursing_dataset_path = "../../datasets/NURSING_2020"

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

    global pamap2_activity_map
    pamap2_activity_map = {
        1: "lying",
        2: "sitting",
        3: "standing",
        4: "walking",
        16: "vacuum cleaning",
        17: "ironing"
    }

    global saved_experiments_path
    saved_experiments_path = 'src/saved_experiments'
