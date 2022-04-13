import itertools
import json
import os


def init():

    # SONAR
    global LABELS
    with open("labels.json") as file:
        categories = json.load(file)["items"]
        LABELS = list(
            itertools.chain.from_iterable(
                category["entries"] for category in categories
            )
        )
        print(LABELS)

    global IS_WINDOWS
    IS_WINDOWS = os.name == "nt"

    global ACTIVITIES
    ACTIVITIES = {k: v for v, k in enumerate(LABELS)}

    global ACTIVITIES_ID_TO_NAME
    ACTIVITIES_ID_TO_NAME = {v: k for k, v in ACTIVITIES.items()}

    global activity_initial_num_to_activity_str
    activity_initial_num_to_activity_str = ACTIVITIES_ID_TO_NAME

    global BP_PATH
    BP_PATH = "/dhc/groups/bp2021ba1"

    global ML_RAINBOW_PATH
    ML_RAINBOW_PATH = BP_PATH + "/apps/ml-rainbow"

    global DATA_PATH
    DATA_PATH = (
        BP_PATH + "/data"
        if not IS_WINDOWS
        else os.path.dirname(os.path.abspath(__file__)) + "/../dataWindows"
    )

    global SENSOR_SUFFIX_ORDER
    SENSOR_SUFFIX_ORDER = ["LF", "LW", "ST", "RW", "RF"]

    global CSV_HEADER_SIZE
    CSV_HEADER_SIZE = 8

    # OPPORTUNITY
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

    # PAMAP2
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
