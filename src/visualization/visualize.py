# pylint: disable=import-error, wrong-import-order

import pandas as pd
from utils import settings
from utils.Window import Window
from sklearn.decomposition import PCA
from matplotlib import pyplot as plt
from typing import Dict
import math


def plot_pca(windows: 'list[Window]', columns_to_plot: Dict[int, str]) -> None:
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
    plt.savefig('visualization/plots/pca.png')
    print('Saved pca.png')


def plot_tsne(windows: 'list[Window]', columns_to_plot: Dict[int, str]) -> None:
    '''
    columns_to_plot is a dictionary mapping the index of the sensor channel (column) 
    to the name of the sensor channel.
    '''
