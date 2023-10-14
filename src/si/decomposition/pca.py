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
        """
        PCA

        Parameters
        ----------
        n_components – number of components
        """
        # parameters
        self.n_components = n_components

        # attributes
        
        self.mean = None
        self.components = None
        self.explained_variance = None

    def _init_centralize_data(self, dataset: Dataset):
        """
        It center the data.

        Parameters
        ----------
        dataset: Dataset
            Dataset object.
        """
        self.mean = dataset.get_mean(dataset)
        return self

    def _SVD_calculation(self, dataset: Dataset):
        """
        Get the SVD.

        Parameters
        ----------
        dataset: Dataset
            Dataset object.

        Returns
        -------
        SVD 
        """
        global S, V_t
        U, S, V_t = np.linalg.svd(dataset, full_matrices=False)
        SVD = U*S*V_t
        return SVD
    
    def _infer_components(self, dataset: Dataset):
        """
        Infer the Principal Components.

        Parameters
        ----------
        dataset: Dataset
            Dataset object.

        """
        self.components = V_t.iloc[:,:(self.n_components+1)]
        return self
    
    def _infer_EV(self, dataset: Dataset):
        """
        Infer the Explained Variance.

        Parameters
        ----------
        dataset: Dataset
            Dataset object.

        """
        n_samples = dataset.shape[0]
        self.explained_variance = S**2/(n_samples-1)
        return self
           

    def fit(self, dataset: Dataset) -> 'PCA':
        """
        estimates the mean, principal components, and explained variance
        
        Parameters
        ----------
        dataset: Dataset
            Dataset object.

        Returns
        -------
        PCA.
        """
        self._init_centralize_data(dataset)
        self._SVD_calculation(dataset)
        self._infer_components(dataset)
        self._infer_EV(dataset)
        
        return self

    def transform(self, dataset: Dataset) -> np.ndarray:
        """
        It transforms the dataset.

        Parameters
        ----------
        dataset: Dataset
            Dataset object.

        Returns
        -------
        np.ndarray
            Transformed dataset.
        """
        centered_data = self._init_centralize_data(dataset)
        V = np.transpose(V_t)
        X_reduced = np.dot(centered_data,V)
        return X_reduced

    def fit_transform(self, dataset: Dataset) -> np.ndarray:
        """
        It fits and transforms the dataset.

        Parameters
        ----------
        dataset: Dataset
            Dataset object.

        Returns
        -------
        np.ndarray
            Transformed dataset.
        """
        self.fit(dataset)
        return self.transform(dataset)

if __name__ == '__main__':
    from si.data.dataset import Dataset
    dataset_ = Dataset.from_random(100, 5)

    n_components = 10
    pca = PCA(n_components)
    res = pca.fit_transform(dataset_)
    print(res.shape)

