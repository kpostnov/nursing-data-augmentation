import os
import random

from loader.load_dataset import load_dataset
from datatypes.Recording import Recording
from utils.filter_activities import filter_activities, filter_activities_negative
from utils.save_all_recordings import save_all_recordings
from utils.cache_recordings import save_recordings, load_recordings
import utils.settings as settings
from utils.Windowizer import Windowizer
from models.DeepConvLSTM import DeepConvLSTM
from visualization.visualize import plot_pca
from loader.preprocessing import interpolate_linear
from scripts.plot_people import count_activities_per_person, count_recordings_per_person
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# Load data
recordings = load_recordings("D:\dataset\ML Prototype Recordings\without_null_activities", limit=3)


activities_to_remove = ['föhnen', 'essen reichen', 'haare waschen', 'accessoires anlegen', 'aufwischen (staub)', 'medikamente stellen', 'küchenvorbereitung']
subjects_to_remove = ['mathias', 'christine']

reduced = filter_activities_negative(recordings, activities_to_remove)
idx = [i for i in range(len(reduced)) if reduced[i].subject in subjects_to_remove]
reduced = [reduced[i] for i in range(len(reduced)) if i not in idx]

save_recordings(reduced, "/dhc/groups/bp2021ba1/data/reduced_data")


rare_activities = ['föhnen', 'haare waschen', 'accessoires anlegen', 'aufwischen (staub)', 'haare kämmen', 'dokumentation', 'mundpflege']
subjects_to_remove = ['christine']

rare = filter_activities(recordings, rare_activities)
idx = [i for i in range(len(rare)) if rare[i].subject in subjects_to_remove]
rare = [rare[i] for i in range(len(rare)) if i not in idx]

save_recordings(rare, "/dhc/groups/bp2021ba1/data/rare_data")



# recordings = load_recordings("/dhc/groups/bp2021ba1/data/filtered_dataset_without_null")

# values = count_recordings_per_person(recordings)

# values.to_csv("recordings_per_person.csv")
# values.plot.bar(figsize=(22,16))
# plt.title("Recordings per person")
# plt.xlabel("Person")
# plt.ylabel("Number of recordings")
# plt.savefig('recordings_per_person.png')


