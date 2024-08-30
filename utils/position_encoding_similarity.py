from matplotlib import pyplot as plt
import torch
import torch.nn as nn
from typing import Union, List
import numpy as np


def similarity_all_to_all_vectorized(pos_embeddings: torch.tensor, grid_h: int, grid_w: int) -> np.ndarray:
    """
    Vectorized implementation of similarity_all_to_all
    """
    seq_length = pos_embeddings.shape[0]
    x = pos_embeddings.repeat_interleave(seq_length, 0) 
    y = pos_embeddings.tile((seq_length, 1)) 
    similarities = nn.CosineSimilarity(dim=-1)(x, y)
    return similarities.view((grid_h, grid_w, grid_h, grid_w)).numpy()

def visualize_heatmaps(similarity_heatmaps: Union[np.ndarray, List[List]], grid_h: int, grid_w: int,epoch:int) -> None:
    """
    Plot grid_h x grid_w subplots, each containing a cosine similarity map in (-1, 1)
    :param similarity_heatmaps: grid_h x grid_w array of similarity heatmaps,
                                where similarity_heatmaps[i][j] stores similarity between embedding
                                for the patch at (i, j) to all the other embeddings
    :param grid_h:              number of patches along Y axis
    :param grid_w:              number of patches along X axis
    """
    fig, ax = plt.subplots(grid_h, grid_w, figsize=(15, 15))
    fontsize = 24

    for i in range(grid_h):
        for j in range(grid_w):
            im = ax[i, j].imshow(similarity_heatmaps[i][j], vmin=-1, vmax=1)
            if i == grid_h - 1:
                ax[i, j].set_xlabel(j + 1, fontsize=fontsize, rotation='horizontal')
            if j == 0:
                ax[i, j].set_ylabel(i + 1, fontsize=fontsize)
            ax[i, j].tick_params(labelcolor='none', which='both', top=False, bottom=False, left=False, right=False)

    cbar = fig.colorbar(im, ax=ax.ravel().tolist(), aspect=25, ticks=[-1, 1])
    cbar.set_label('Cosine similarity', rotation=-270, fontsize=fontsize)
    cbar.ax.tick_params(labelsize=fontsize)
    fig.text(0.45, 0.04, 'Input patch row', ha='center', fontsize=fontsize)
    fig.text(0.04, 0.5, 'Input patch column', va='center', rotation='vertical', fontsize=fontsize)
    fig.savefig("position_embedding_epoch_{}.png".format(epoch))
    plt.close()
