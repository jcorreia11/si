import numpy as np


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    It returns the float corresponding to the error between y_true and y_pred of the model on the given dataset

    Parameters
    ----------
    y_true: np.ndarray
        The true labels of the dataset
    y_pred: np.ndarray
        The predicted labels of the dataset

    Returns
    -------
    accuracy: float
        float corresponding to the error between y_true and y_pred
    """
    return np.sqrt(np.sum((y_true - y_pred)**2) / len(y_true))