import os.path
import pandas as pd

from datatypes.Recording import Recording
import utils.settings as settings
from utils.cache_recordings import load_recordings
from visualization.plot_distribution import plot_distribution_pie_chart, plot_distribution_bar_chart


def count_person_length(recordings: "list[Recording]"):
    values = pd.Series({recordings[0].subject: recordings[0].activities.count()})
    for rec in recordings[1:]:
        values = values.add(pd.Series({rec.subject: rec.activities.count()}), fill_value=0)

    return values


def plot_people() -> None:
    dataset_path = os.path.join(settings.sonar_dataset_path)

    recordings = load_recordings(dataset_path)
    counts = count_person_length(recordings)

    # Map the keys to names
    counts = counts.to_dict()
    counts = {
        k.split('.')[0]: v for k, v in counts.items()
    }
    
    plot_distribution_pie_chart(counts)
    plot_distribution_bar_chart(counts)
    return counts

plot_people()