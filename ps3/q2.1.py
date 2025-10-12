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

def add_bias_column(X):
    '''
    Create a bias column and combine it with X.

    Parameters
    ----------
    X : (n, d) numpy matrix representing a feature matrix
    
    Returns
    -------
        new_X (np.ndarray):
            A (n, d + 1) numpy matrix with the first column consisting of all 1s
    '''
  
    """ YOUR CODE HERE """
    n = X.shape[0]
    bias = np.ones((n, 1))
    new_X = np.hstack((bias, X))
    return new_X
    """ YOUR CODE END HERE """