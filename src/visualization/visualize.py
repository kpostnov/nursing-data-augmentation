import pandas as pd
from sklearn.decomposition import PCA
from matplotlib import pyplot as plt
from typing import Tuple
import numpy as np
from sklearn.manifold import TSNE


def get_averaged_dataframes(original_data: np.ndarray, generated_data: np.ndarray) -> Tuple[pd.DataFrame, pd.DataFrame]:
    '''
    Calculate the mean of each time step over all sensor channels and return the dataframes.
    '''

    assert(original_data.ndim == 4), 'original_data must be a 4D array'
    assert(generated_data.ndim == 4), 'generated_data must be a 4D array'

    number_samples = min(original_data.shape[0], generated_data.shape[0], 2000)
    idx = np.random.permutation(number_samples)

    original_data = original_data[idx]
    generated_data = generated_data[idx]

    original_data = np.squeeze(original_data, axis=-1)
    generated_data = np.squeeze(generated_data, axis=-1)

    seq_len = original_data.shape[1]

    # np.mean(original_data, axis=2)?
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

    return df_original, df_generated


def plot_pca_distribution(original_data: np.ndarray, generated_data: np.ndarray, activity: str) -> None:
    '''
    Plot distributions of the original and generated data.
    Calculate the mean of each time step over all sensor channels. 
    Reduce dimensionality of all time steps in a window to 2D using PCA and plot the result.
    '''
    
    df_original, df_generated = get_averaged_dataframes(original_data, generated_data)

    pca = PCA(n_components=2)
    pca.fit(df_original.iloc[:, :-1])
    Xt_original = pca.transform(df_original.iloc[:, :-1])
    Xt_generated = pca.transform(df_generated.iloc[:, :-1])

    f, ax = plt.subplots(1)
    plt.scatter(Xt_original[:, 0], Xt_original[:, 1], c=df_original['color'], alpha=0.2, label='original')
    plt.scatter(Xt_generated[:, 0], Xt_generated[:, 1], c=df_generated['color'], alpha=0.2, label='generated')  
    ax.legend()
    plt.title('PCA visualization of activity: ' + activity)

    plt.savefig(f'visualization/plots/pca_distribution_{activity}.png', bbox_inches='tight')

    print(f'Plotting PCA visualization of activity {activity} finished')


def plot_tsne_distribution(original_data: np.ndarray, generated_data: np.ndarray, activity: str) -> None:
    '''
    Plot distributions of the original and generated data using t-SNE.
    '''
    
    number_samples = min(original_data.shape[0], 2000)
    
    df_original, df_generated = get_averaged_dataframes(original_data, generated_data)

    prep_data_final = pd.concat([df_original, df_generated], ignore_index=True)
    
    # TSNE anlaysis
    tsne = TSNE(n_components = 2, verbose = 1, perplexity = 40, n_iter = 300)
    tsne_results = tsne.fit_transform(prep_data_final.iloc[:, :-1])
      
    # Plotting  
    f, ax = plt.subplots(1)
    plt.scatter(tsne_results[:number_samples,0], tsne_results[:number_samples,1], 
                c = prep_data_final.loc[:number_samples-1, 'color'], alpha = 0.2, label = "Original")
    plt.scatter(tsne_results[number_samples:,0], tsne_results[number_samples:,1], 
                c = prep_data_final.loc[number_samples:, 'color'], alpha = 0.2, label = "Synthetic")
  
    ax.legend()
      
    plt.title('t-SNE visualization of activity: ' + activity)
    plt.savefig(f'visualization/plots/tsne_distribution_{activity}.png', bbox_inches='tight')
    
    print(f'Plotting t-SNE visualization of activity {activity} finished')
