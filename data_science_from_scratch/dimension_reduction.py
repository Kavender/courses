import numpy as np
import matplotlib.pyplot as plt
from sklearn import decomposition
from sklearn.manifold import TSNE
# !pip install umap-learn[plot]
from .data_manipulation import first_principal_component, remove_projection
import umap


def principal_component_analysis(X, num_components):
    """
    Principal component analysis (PCA) extract one or more dimensions that capture as much of the variation in the data
    as possible.
    """
    components = []
    for _ in range(num_components):
        component = first_principal_component(X)
        components.append(component)
        X = remove_projection(X, component)
    return components


def visualize_num_components_explained(X):
    """Similar to the elbow methods, we'd like to find out the optimal K components we shall run PCA run.
    """
    pca = decomposition.PCA().fit(X)

    plt.figure(figsize=(10,7))
    plt.plot(np.cumsum(pca.explained_variance_ratio_), color='k', lw=2)
    plt.xlabel('Number of components')
    plt.ylabel('Total explained variance')

    plt.yticks(np.arange(0, 1.1, 0.1))
    # plt.xlim(0, 63)
    # plt.axvline(21, c='b')
    plt.axhline(0.9, c='r')
    plt.show()


def visualize_2D_projection(X_dr, target):
    plt.figure(figsize=(12,10))
    plt.scatter(X_dr[:, 0], X_dr[:, 1], c=target,
                edgecolor='none', alpha=0.7, s=40,
                cmap=plt.cm.get_cmap('nipy_spectral', 10))
    plt.colorbar()
    plt.title(f'Projecting {X_dr.shape[1]}-dimensional data to 2D')
