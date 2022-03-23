import os

def init():
    global IS_WINDOWS
    IS_WINDOWS = (os.name == 'nt')

    global SENSOR_COLUMN_PREFIXES
    SENSOR_COLUMN_PREFIXES = ('Quat_', 'Acc_', 'FreeAcc_', 'Gyro_', 'Mag_')

    global ACTIVITIES
    ACTIVITIES = {
        'running': 0,
        'squats': 1,
        'stairs_down': 2,
        'stairs_up': 3,
        'standing': 4,
        'walking': 5,
    }

    global ACTIVITIES_ID_TO_NAME
    ACTIVITIES_ID_TO_NAME = {v: k for k, v in ACTIVITIES.items()}

    global BP_PATH
    BP_PATH = '/dhc/groups/bp2021ba1'

    global ML_RAINBOW_PATH
    ML_RAINBOW_PATH = BP_PATH + '/apps/ml-rainbow'

    global DATA_PATH
    DATA_PATH = BP_PATH + '/data' if not IS_WINDOWS else os.path.dirname(os.path.abspath(__file__)) + '/../dataWindows'


    global SENSOR_SUFFIX_ORDER
    SENSOR_SUFFIX_ORDER = ["LF", "LW", "ST", "RW", "RF"]

    global SENSOR_MAC_SUFFIX_MAP
    SENSOR_MAC_SUFFIX_MAP = {
        "D4:22:CD:00:06:7B": "LF",
        "D4:22:CD:00:06:89": "LW",
        "D4:22:CD:00:06:7F": "ST",
        "D4:22:CD:00:06:7D": "RW",
        "D4:22:CD:00:06:72": "RF"
    }

    global CSV_HEADER_SIZE
    CSV_HEADER_SIZE = 8
