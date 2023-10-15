#PCA
from typing import Callable

import numpy as np

from si.data.dataset import Dataset

class PCA:
    """
    It reduce the dimensions of
    the dataset. The PCA to be implemented must use the Singular
    Value Decomposition (SVD) linear algebra technique.
    
    Parameters
    ----------
    n_components – number of components

    Attributes
    ----------
    - mean – mean of the samples
    - components – the principal components (the unitary matrix of eigenvectors)
    - explained_variance – explained variance (diagonal matrix of eigenvalues)
    """

    def __init__(self, n_components: int):
        
        # parameters
        self.n_components = n_components

    def fit(self, dataset: Dataset) -> 'PCA':
        X = dataset.X
        self._mean = np.mean(X, axis=0)
        self.X_center = X -  self._mean
        cov_matrix = np.cov(self.X_center.T)
        self.e_vals, self.e_vecs = np.linalg.eig(cov_matrix)
        
        return self

    def transform(self, dataset: Dataset) -> np.ndarray:
        
        X_center = self.scale(dataset) 
        self.sorted_index = np.argsort(self.e_vals)[::-1]
        self.e_vals_sorted = self.e_vals[self.sorted_index]
        self.e_vecs_sorted = self.e_vecs[:, self.sorted_index]
        # transition matrix, or change of base matrix. 
        self.e_vecs_subset = self.e_vecs_sorted[:, 0:self.n_components]
        # projects the data into a lower dimension.
        X_reduced = self.e_vecs_subset.T.dot(X_center.T).T
        return X_reduced

    def fit_transform(self, dataset: Dataset) -> np.ndarray:
        self.fit(dataset)
        return self.transform(dataset)

    def variance_explained(self):
        _sum = sum(self.e_vals_sorted)
        return [(i/_sum*100) for i in self.e_vals_sorted]

if __name__ == '__main__':
    from si.data.dataset import Dataset
    dataset_ = Dataset.from_random(100, 5)
    n_components = 10
    pca = PCA(n_components)
    res = pca.fit_transform(dataset_)
    print(res.shape)

