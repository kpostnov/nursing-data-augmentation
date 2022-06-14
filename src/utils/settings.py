import itertools
import json
import os


def init(dataset: str):
    global saved_experiments_path
    saved_experiments_path = os.path.dirname(os.path.abspath(__file__)) + "/../../saved_experiments"

    # Model / Dataset specific configuration
    if dataset == "sonar":
        init_sonar()
    elif dataset == "sonar_lab":
        init_sonar_lab()
    elif dataset == "pamap2":
        init_pamap2()
    else:
        raise Exception("Unknown dataset")


def init_sonar():
    global sonar_dataset_path
    sonar_dataset_path = "/dhc/groups/bp2021ba1/data/reduced_data"

    global LABELS
    with open("labels_reduced.json") as file:
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

    # TODO: Convert to numbers
    global SUBJECTS
    # reduced
    SUBJECTS = ["aileen", "alex", "anja", "b2", "brueggemann", "connie", "florian", "kathi", "oli",	"rauche", "trapp", "yvan"]
    # rare
    # SUBJECTS = ["aileen", "alex", "anja", "b2", "brueggemann", "connie", "florian", "kathi", "mathias", "oli", "rauche", "trapp", "yvan"]

    global subj_to_numbers
    subj_to_numbers = {
        "aileen": "100",
        "alex": "101",
        "anja": "102",
        "b2": "103",
        "brueggemann": "104",
        "christine": "105",
        "connie": "106",
        "florian": "107",
        "kathi": "108",
        "mathias": "109",
        "oli": "110",
        "rauche": "111",
        "trapp": "112",
        "yvan" : "113"
    }

    global ger_to_en
    ger_to_en = {
        "aufräumen": "clean",
        "bad vorbereiten": "prepare bathroom",
        "bett machen": "make bed",
        "haare kämmen": "comb hair",
        "mundpflege": "oral care",
        "umkleiden": "dress",
        "gesamtwaschen im bett": "cleaning routine in bed",
        "waschen am waschbecken": "cleaning routine in bathroom",
        "essen auf teller geben": "put food on plate",
        "essen austragen": "bring food",
        "geschirr einsammeln": "collect dishes",
        "getränke ausschenken": "pour drinks",
        "rollstuhl schieben": "push wheelchair",
        "rollstuhl transfer": "wheelchair transfer",
        "dokumentation": "documentation",
        "essen reichen": "serve food",
        "küchenvorbereitung": "kitchen preparation",
        "medikamente stellen": "prepare medicine",
    }


def init_sonar_lab():
    global sonar_dataset_path
    sonar_dataset_path = "/dhc/groups/bp2021ba1/data/lab_data_filtered_without_null"

    global LABELS
    with open("labels_lab.json") as file:
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

    # TODO: Convert to numbers
    global SUBJECTS
    SUBJECTS = ["orhan", "daniel", "felix", "tobi", "lucas", "kirill", "marco", "valentin", "alex", "franz"]

    global subj_to_numbers
    subj_to_numbers = {
        "orhan": "200", 
        "daniel": "201", 
        "felix": "202", 
        "tobi": "203", 
        "lucas": "204", 
        "kirill": "205", 
        "marco": "206", 
        "valentin": "207", 
        "alex": "208", 
        "franz": "209",
    }

    global ger_to_en
    ger_to_en = {
        "aufräumen": "clean",
        "aufwischen (staub)": "wipe dust",
        "bett machen": "make bed",
        "dokumentation": "documentation",
        "essen reichen": "serve food",
        "gesamtwaschen im bett": "cleaning routine in bed",
        "getränke ausschenken": "pour drinks",
        "haare kämmen": "comb hair",
        "medikamente stellen": "prepare medicine",
        "rollstuhl schieben": "push wheelchair",
        "rollstuhl transfer": "wheelchair transfer",
        "umkleiden": "dress",
        "waschen am waschbecken": "skin care",
        "null - activity": "null - activity"
    }


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
        0: "lying",
        1: "sitting",
        2: "standing",
        3: "walking",
        4: "vacuum cleaning",
        5: "ironing",
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
