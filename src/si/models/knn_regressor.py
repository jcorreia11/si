from typing import Callable, Union

import numpy as np

from si.data.dataset import Dataset
from si.metrics.accuracy import accuracy
from si.statistics.euclidean_distance import euclidean_distance
from si.metrics.RMSE import rmse


class KNNRegressor:
    """
    KNN Regressor
    The k-Nearst Neighbors Regressor is a machine learning that estimates the average value of the k most similar examples instead of
    the most common class, based on a similarity measure (e.g., RMSE).

    Parameters
    ----------
    k: int
        The number of nearest neighbors to use
    distance: Callable
        The distance function to use

    Attributes
    ----------
    dataset: np.ndarray
        The training data
    """
    def __init__(self, k: int = 1, distance: Callable = euclidean_distance):
        """
        Initialize the KNN classifier

        Parameters
        ----------
        k: int
            The number of nearest neighbors to use
        distance: Callable
            The distance function to use
        """
        # parameters
        self.k = k
        self.distance = distance

        # attributes
        self.dataset = None

    def fit(self, dataset: Dataset) -> 'KNNRegressor':
        """
        It fits the model to the given dataset

        Parameters
        ----------
        dataset: Dataset
            The dataset to fit the model to

        Returns
        -------
        self: KNNRegressor
            The fitted model
        """
        self.dataset = dataset
        return self

    def _get_closest_values(self, sample: np.ndarray) -> Union[int, str]:
        """
        It returns the closest label of the given sample

        Parameters
        ----------
        sample: np.ndarray
            The sample to get the closest label of

        Returns
        -------
        label: str or int
            The closest label
        """
        # compute the distance between the sample and the dataset
        distances = self.distance(sample, self.dataset.X)

        # get the k nearest neighbors
        k_nearest_neighbors = np.argsort(distances)[:self.k]

        # get the values of the k nearest neighbors
        k_nearest_neighbors_values = self.dataset.y[k_nearest_neighbors]

        # get the mean of the values
        
        k_values_mean = k_nearest_neighbors_values.sum() / self.k
        return k_values_mean

    def predict(self, dataset: Dataset) -> np.ndarray:
        """
        It predicts the regressors of the given dataset

        Parameters
        ----------
        dataset: Dataset
            The dataset to predict the regressors values of

        Returns
        -------
        predictions: np.ndarray
            The predictions of the model
        """
        return np.apply_along_axis(self._get_closest_values, axis=1, arr=dataset.X)

    def score(self, dataset: Dataset) -> float:
        """
        It returns the rmse of the model on the given dataset

        Parameters
        ----------
        dataset: Dataset
            The dataset to evaluate the model on

        Returns
        -------
        rmse: float
            The rmse of the model
        """
        predictions = self.predict(dataset)
        return rmse(dataset.y, predictions)


if __name__ == '__main__':
    # import dataset
    from si.data.dataset import Dataset
    from si.model_selection.split import train_test_split

    # load and split the dataset
    dataset_ = Dataset.from_random(600, 100, 2)
    dataset_train, dataset_test = train_test_split(dataset_, test_size=0.2)

    # initialize the KNN Regressor
    knnr = KNNRegressor(k=5)

    # fit the model to the train dataset
    knnr.fit(dataset_train)

    # evaluate the model on the test dataset
    score = knnr.score(dataset_test)
    print(f'The rmse of the model is: {score}')