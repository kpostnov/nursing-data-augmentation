import itertools
import json
import os


def init(dataset: str):
    global saved_experiments_path
    saved_experiments_path = "../saved_experiments"

    # Model / Dataset specific configuration
    if dataset == "opportunity":
        init_opportunity()
    elif dataset == "sonar":
        init_sonar()
    elif dataset == "pamap2":
        init_pamap2()
    elif dataset == "nursing":
        init_nursing()
    else:
        raise Exception("Unknown dataset")


def init_sonar():
    global sonar_dataset_path
    sonar_dataset_path = "../../datasets/SONAR"

    global LABELS
    with open("labels.json") as file:
        categories = json.load(file)["items"]
        LABELS = list(
            itertools.chain.from_iterable(
                category["entries"] for category in categories
            )
        )

    global IS_WINDOWS
    IS_WINDOWS = os.name == "nt"

    global ACTIVITIES
    ACTIVITIES = {k: v for v, k in enumerate(LABELS)}

    global ACTIVITIES_ID_TO_NAME
    ACTIVITIES_ID_TO_NAME = {v: k for k, v in ACTIVITIES.items()}

    global activity_initial_num_to_activity_str
    activity_initial_num_to_activity_str = ACTIVITIES_ID_TO_NAME

    global SENSOR_SUFFIX_ORDER
    SENSOR_SUFFIX_ORDER = ["LF", "LW", "ST", "RW", "RF"]

    global CSV_HEADER_SIZE
    CSV_HEADER_SIZE = 8


def init_opportunity():
    global opportunity_dataset_path
    opportunity_dataset_path = "../../datasets/OpportunityUCIDataset"

    # global activity_initial_num_to_activity_str
    # activity_initial_num_to_activity_str = {
    #     0: "null",
    #     101: "relaxing",
    #     102: "coffee time",
    #     103: "early morning",
    #     104: "cleanup",
    #     105: "sandwich time",
    # }

    # global activity_initial_num_to_activity_idx
    # activity_initial_num_to_activity_idx = {
    #     0: 0,
    #     101: 1,
    #     102: 2,
    #     103: 3,
    #     104: 4,
    #     105: 5,
    # }


    # Ordonez et al. (2015)
    global activity_initial_num_to_activity_str
    activity_initial_num_to_activity_str = {
        0: 'null',
        406516: "Open Door 1",
        406517: "Open Door 2",
        404516: "Close Door 1",
        404517: "Close Door 2",
        406520: "Open Fridge",
        404520: "Close Fridge",
        406505: "Open Dishwasher",
        404505: "Close Dishwasher",
        406519: "Open Drawer 1", 
        404519: "Close Drawer 1",
        406511: "Open Drawer 2", 
        404511: "Close Drawer 2", 
        406508: "Open Drawer 3", 
        404508: "Close Drawer 3", 
        408512: "Clean Table",
        407521: "Drink from Cup", 
        405506: "Toggle Switch",
    }

    global activity_initial_num_to_activity_idx
    activity_initial_num_to_activity_idx =  {
        0: 0,
        406516: 1,
        406517: 2,
        404516: 3,
        404517: 4,
        406520: 5,
        404520: 6,
        406505: 7,
        404505: 8,
        406519: 9, 
        404519: 10,
        406511: 11, 
        404511: 12, 
        406508: 13, 
        404508: 14, 
        408512: 15,
        407521: 16, 
        405506: 17,
    }

    global ACTIVITIES
    ACTIVITIES = list(activity_initial_num_to_activity_str.values())


def init_pamap2():
    global pamap2_dataset_path
    pamap2_dataset_path = "../../datasets/PAMAP2_Dataset"

    global pamap2_initial_num_to_activity_idx
    pamap2_initial_num_to_activity_idx = {
        1: 0,
        2: 1,
        3: 2,
        4: 3,
        16: 4,
        17: 5,
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

    global ACTIVITIES
    ACTIVITIES = list(pamap2_activity_map.values())


def init_nursing():
    global nursing_dataset_path
    nursing_dataset_path = "../../datasets/NURSING_2020"
