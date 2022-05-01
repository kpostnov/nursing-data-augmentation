# pylint: disable=import-error, wrong-import-order

import pandas as pd
from utils import settings
from utils.Window import Window
from sklearn.decomposition import PCA
from matplotlib import pyplot as plt
from typing import Dict
import math
import numpy as np


def plot_pca_distribution(original_data: np.ndarray, generated_data: np.ndarray, activity: str, ax = None) -> None:
    '''
    Plot distributions of the original and generated data.
    Calculate the mean of each time step over all sensor channels. 
    Reduce dimensionality of all time steps in a window to 2D using PCA and plot the result.
    '''
    
    assert(original_data.ndim == 4), 'original_data must be a 4D array'
    assert(generated_data.ndim == 4), 'generated_data must be a 4D array'

    number_samples = min(original_data.shape[0], 2000)
    idx = np.random.permutation(number_samples)

    original_data = original_data[idx]
    generated_data = generated_data[idx]

    original_data = np.squeeze(original_data, axis=-1)
    generated_data = np.squeeze(generated_data, axis=-1)

    seq_len = original_data.shape[1]

    for i in range(number_samples):
        if (i == 0):
            original_array = np.reshape(np.mean(original_data[0,:,:], 1), [1, seq_len])
            generated_array = np.reshape(np.mean(generated_data[0,:,:], 1), [1, seq_len])
        else:
            original_array = np.concatenate((original_array, 
                                        np.reshape(np.mean(original_data[i,:,:], 1), [1, seq_len])))
            generated_array = np.concatenate((generated_array, 
                                        np.reshape(np.mean(generated_data[i,:,:], 1), [1, seq_len])))


    df_original = pd.DataFrame(original_array)
    df_generated = pd.DataFrame(generated_array)
    df_original['color'] = 'red'
    df_generated['color'] = 'blue'

    pca = PCA(n_components=2)
    pca.fit(df_original.iloc[:, :-1])
    Xt_original = pca.transform(df_original.iloc[:, :-1])
    Xt_generated = pca.transform(df_generated.iloc[:, :-1])

    # If plot is drawn individually
    if ax is None:
        ax = plt.gca()
        plt.suptitle(f'PCA visualization of activity {activity}')

    ax.scatter(Xt_original[:, 0], Xt_original[:, 1], c=df_original['color'], alpha=0.2, label='original')
    plot = ax.scatter(Xt_generated[:, 0], Xt_generated[:, 1], c=df_generated['color'], alpha=0.2, label='generated')  
    ax.legend()

    plt.savefig(f'visualization/plots/pca_distribution_{activity}.png')


    print(f'Plotting PCA visualization of activity {activity} finished')

    return plot


def plot_pca(windows: 'list[Window]', ax = None) -> None:
    '''
    Plot distributions of actvities.
    Calculate the mean of each time step over all sensor channels. 
    Reduce dimensionality of all time steps in a window to 2D using PCA and plot the result.
    '''    
    complete_df = pd.DataFrame()

    for window in windows:
        df = pd.DataFrame(window.sensor_array)
        df['mean'] = df.mean(axis=1)
        df = df['mean']
        df = df.transpose()  # Every time step is a feature now
        df['activity'] = window.activity
        complete_df = complete_df.append(df)

    activities = list(map(lambda num: settings.ACTIVITIES_ID_TO_NAME[num],
                            complete_df['activity'].unique().tolist()))

    # If plot is drawn individually
    if ax is None:
        ax = plt.gca()
        plt.suptitle(f'PCA on all sensor channels, {len(activities)} activities, {len(windows)} windows')
        plt.savefig('visualization/plots/pca_activity.png')

    pca = PCA()
    Xt = pca.fit_transform(complete_df.iloc[:, :-1])
    plot = ax.scatter(Xt[:, 0], Xt[:, 1], c=complete_df['activity'], cmap='viridis')
    ax.legend(handles=plot.legend_elements()[0], labels=list(activities))

    print("Plotting PCA on all sensor channels, {len(activities)} activities, {len(windows)} windows")

    return plot


def plot_pca_column_wise(windows: 'list[Window]', columns_to_plot: Dict[int, str]) -> None:
    '''
    Reduce the dimensionality of a sensor channel to 2D using PCA and plot the result. 
    columns_to_plot is a dictionary mapping the index of the sensor channel (column) 
    to the name of the sensor channel.
    '''
    num_rows = math.ceil(math.sqrt(len(columns_to_plot)))
    num_cols = math.ceil(len(columns_to_plot) / num_rows)
    fig, axis = plt.subplots(num_rows, num_cols, squeeze=False, figsize=(num_cols * 5, num_rows * 5))

    activities = set()

    print(f'Plotting {len(columns_to_plot)} columns')

    for (index, (column, column_name)) in enumerate(columns_to_plot.items()):
        column_df = pd.DataFrame()

        for window in windows:
            df = pd.DataFrame(window.sensor_array)
            df = df[column]
            df = df.transpose()  # Every time step is a feature now
            df['activity'] = window.activity
            column_df = column_df.append(df)

        activities.update(list(map(lambda num: settings.ACTIVITIES_ID_TO_NAME[num],
                              column_df['activity'].unique().tolist())))

        pca = PCA()
        Xt = pca.fit_transform(column_df.iloc[:, :-1])

        row = index // num_cols
        col = index % num_cols

        print(f'Plotting row {row} and col {col}')

        graph = axis[row, col].scatter(Xt[:, 0], Xt[:, 1], c=column_df['activity'], cmap='viridis')
        axis[row, col].set_title(column_name)

    fig.suptitle(f'PCA on all sensor channels, {len(activities)} activities')
    plt.legend(handles=graph.legend_elements()[0], labels=list(activities))
    plt.savefig('visualization/plots/pca_columns.png')
    print('Saved pca_columns.png')


def plot_tsne(windows: 'list[Window]', columns_to_plot: Dict[int, str]) -> None:
    '''
    columns_to_plot is a dictionary mapping the index of the sensor channel (column) 
    to the name of the sensor channel.
    '''
