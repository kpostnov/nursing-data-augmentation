import itertools
import json
import os


def init(args):
    global IS_WINDOWS
    IS_WINDOWS = os.name == "nt"

    global saved_experiments_path
    saved_experiments_path = os.path.dirname(os.path.abspath(__file__)) + "/../../saved_experiments"

    global dataset_path
    dataset_path = args.data_path

    global synth_data_path 
    synth_data_path = args.synth_data_path

    global random_data_path
    random_data_path = args.random_data_path

    global WINDOW_SIZE 
    WINDOW_SIZE = args.window_size

    global STRIDE_SIZE
    STRIDE_SIZE = args.stride_size

    # Dataset-specific configuration
    if args.dataset == "sonar":
        init_sonar()
    elif args.dataset == "sonar_lab":
        init_sonar_lab()
    elif args.dataset == "pamap2":
        init_pamap2()
    else:
        raise Exception("Unknown dataset")


def init_sonar():

    global LABELS
    with open("labels_reduced.json") as file:
        categories = json.load(file)["items"]
        LABELS = list(
            itertools.chain.from_iterable(
                category["entries"] for category in categories
            )
        )

    global FREQUENCY
    FREQUENCY = 60

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

    global SUBJECTS
    # reduced dataset
    SUBJECTS = ["100", "101", "102", "103", "104", "106", "107", "108", "110", "111", "112", "113"]


def init_sonar_lab():

    global LABELS
    with open("labels_lab.json") as file:
        categories = json.load(file)["items"]
        LABELS = list(
            itertools.chain.from_iterable(
                category["entries"] for category in categories
            )
        )

    global FREQUENCY
    FREQUENCY = 60

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

    global SUBJECTS
    SUBJECTS = ["200", "201", "202", "203", "204", "205", "206", "207", "208", "209"]


def init_pamap2():

    global FREQUENCY
    FREQUENCY = 100

    global pamap2_initial_num_to_activity_idx
    pamap2_initial_num_to_activity_idx = {
        1: 0,
        2: 1,
        3: 2,
        4: 3,
        16: 4,
        17: 5,
    }

    global pamap2_id_to_str
    pamap2_id_to_str = {
        0: "lying",
        1: "sitting",
        2: "standing",
        3: "walking",
        4: "vacuum cleaning",
        5: "ironing",
    }

    global pamap2_str_to_id
    pamap2_str_to_id = {
        "lying": 0,
        "sitting": 1,
        "standing": 2,
        "walking": 3,
        "vacuum cleaning": 4,
        "ironing": 5,
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

    global LABELS
    LABELS = list(pamap2_activity_map.values())
