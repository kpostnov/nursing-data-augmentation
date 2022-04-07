from utils.Recording import Recording
import pandas as pd


def filter_short_activities(recordings: 'list[Recording]', threshhold: int = 3, strategy: int = 0) -> 'list[Recording]':
    """
    Replaces activities shorter than threshhold by [value]. [value] depends on strategy.
    strategy 0: replaces short activities with previous activity
    strategy 1: replaces short activities with 'null-activity'

    threshold: number of seconds
    """

    if strategy != 0 and strategy != 1:
        raise ValueError('strategy has to be 0 or 1')
    
    for recording in recordings:
        indizes = []
        length = recording.activities.shape[0]

        for i in range(length-1):
            if recording.activities.iloc[i] != recording.activities.iloc[i+1]:
                indizes.append(i+1)
        
        for i in range(len(indizes) - 1):
            if recording.time_frame[indizes[i+1]] - recording.time_frame[indizes[i]] < (threshhold * 1000000):
                if strategy == 0:
                    recording.activities.iloc[indizes[i]:indizes[i+1]] = recording.activities.iloc[indizes[i-1]]
                elif strategy == 1:
                    recording.activities.iloc[indizes[i]:indizes[i+1]] = 'null-activity'

    return recordings


def rename_activities(recordings: 'list[Recording]', rules: dict = {}) -> 'list[Recording]':
    """
    Renames / groups activities defined in rules.
    rules example structure:
    {
        'activity_name': 'new_activity_name',
    }
    """

    for recording in recordings:
        for old_activity, new_activity in rules.items():
            recording.activities[recording.activities == old_activity] = new_activity

    return recordings
