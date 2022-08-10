import os.path
import pandas as pd

from datatypes.Window import Window
from datatypes.Recording import Recording
import utils.settings as settings
from utils.cache_recordings import load_recordings
from visualization.plot_distribution import plot_distribution_pie_chart, plot_distribution_bar_chart


def count_windows_per_activity_per_person(windows: "list[Window]", window_size: int) -> pd.DataFrame:
    values = pd.DataFrame({windows[0].subject: {windows[0].activity: 1}})
    for win in windows[1:]:
        values = values.add(pd.DataFrame({win.subject: {win.activity: 1}}), fill_value=0)
    
    return values


def count_windows_per_activity(windows: "list[Window]", window_size: int) -> pd.DataFrame:
    values = pd.Series({windows[0].activity: 1})
    for win in windows[1:]:
        values = values.add(pd.Series({win.activity: 1}), fill_value=0)

    values = values.to_dict()
    values = {
        settings.ACTIVITIES_ID_TO_NAME[k]: v for k, v in values.items()
    }
    values = pd.Series(values)
    values = values.to_frame()
    values['timesteps'] = values.iloc[:, 0] * window_size

    return values


def count_recordings_per_person(recordings: "list[Recording]") -> pd.Series:
    values = pd.Series({recordings[0].subject: 1})
    for rec in recordings[1:]:
        values = values.add(pd.Series({rec.subject: 1}), fill_value=0)

    return values


def count_activities_per_person(recordings: "list[Recording]") -> pd.DataFrame:
    values = pd.DataFrame({recordings[0].subject: recordings[0].activities.value_counts()})
    for rec in recordings[1:]:
        values = values.add(pd.DataFrame({rec.subject: rec.activities.value_counts()}), fill_value=0)

    return values


def count_activity_length(recordings: "list[Recording]") -> pd.Series:
    """
    Returns the total amount of recording time per activity in timesteps.
    """
    values = recordings[0].activities.value_counts()
    for rec in recordings[1:]:
        values = values.add(rec.activities.value_counts(), fill_value=0)

    return values


def count_person_length(recordings: "list[Recording]") -> pd.Series:
    """
    Returns the total amount of recording time per person in timesteps.
    """
    values = pd.Series({recordings[0].subject: recordings[0].activities.count()})
    for rec in recordings[1:]:
        values = values.add(pd.Series({rec.subject: rec.activities.count()}), fill_value=0)
    
    return values


def plot_data(recordings: "list[Recording]") -> None:

    # counts = count_activity_length(recordings)
    counts = count_person_length(recordings)

    # Map the keys to names
    counts = counts.to_dict()
    counts = {
        k.split('.')[0]: v for k, v in counts.items()
    }

    plot_distribution_pie_chart(counts)
    plot_distribution_bar_chart(counts)
