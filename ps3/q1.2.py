# Inital imports and setup

import os
import numpy as np

###################
# Helper function #
###################
def load_data(filepath):
    '''
    Load in the given csv filepath as a numpy array

    Parameters
    ----------
    filepath (string) : path to csv file

    Returns
    -------
        X, y (np.ndarray, np.ndarray) : (n, num_features), (n,) numpy matrices
    '''
    *X, y = np.genfromtxt(
        filepath,
        delimiter=',',
        skip_header=True,
        unpack=True,
    ) # default dtype: float
    X = np.array(X, dtype=float).T # cast features to int type
    return X, y.reshape((-1, 1))

data_filepath = 'housing_data.csv'
X, y = load_data(data_filepath)
def mean_absolute_error(y_true, y_pred):
    '''
    Calculate mean absolute error between y_pred and y_true.

    Parameters
    ----------
    y_true (np.ndarray) : (n, 1) numpy matrix consists of true values
    y_pred (np.ndarray) : (n, 1) numpy matrix consists of predictions
    
    Returns
    -------
        The mean absolute error value.
    '''
  
    """ YOUR CODE HERE """
    n = len(y_true)
    result = np.sum(np.abs(y_true - y_pred))/n
    return result
    """ YOUR CODE END HERE """