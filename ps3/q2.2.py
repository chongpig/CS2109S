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
def get_bias_and_weight(X, y, include_bias = True):
    '''
    Calculate bias and weights that give the best fitting line.

    Parameters
    ----------
    X (np.ndarray) : (n, d) numpy matrix representing feature matrix
    y (np.ndarray) : (n, 1) numpy matrix representing target values
    include_bias (boolean) : Specify whether the model should include a bias term
    
    Returns
    -------
        bias (float):
            If include_bias = True, return the bias constant. Else,
            return 0
        weights (np.ndarray):
            A (d, 1) numpy matrix representing the weight(s).
    '''
  
    """ YOUR CODE HERE """
    if include_bias:
        X_with_bias = add_bias_column(X)
        w = np.linalg.inv(X_with_bias.T @ X_with_bias) @ X_with_bias.T @ y
        bias = w[0, 0]
        weights = w[1:]
    else:
        w = np.linalg.inv(X.T @ X) @ X.T @ y
        bias = 0.0
        weights = w
    return bias, weights
    """ YOUR CODE END HERE """