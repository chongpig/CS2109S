def feature_scaling(X):
    '''
    Standardize each feature column.

    Parameters
    ----------
    X (np.ndarray) : (n, d) numpy matrix representing feature matrix

    Returns
    -------
        A (n, d) numpy matrix where each column has been standardized.
    '''
    """ YOUR CODE HERE """
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    X_new = (X - mean) / std
    return X_new
    """ YOUR CODE END HERE """