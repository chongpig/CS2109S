def get_prediction_linear_regression(X, y, include_bias = True):
    '''
    Calculate the best fitting line.

    Parameters
    ----------
    X (np.ndarray) : (n, d) numpy matrix representing feature matrix
    y (np.ndarray) : (n, 1) numpy matrix representing target values
    include_bias (boolean) : Specify whether the model should include a bias term

    Returns
    -------
        y_pred (np.ndarray):
            A (n, 1) numpy matrix representing prediction values.
    '''
  
    """ YOUR CODE HERE """
    bias, weights = get_bias_and_weight(X, y, include_bias)
    y_pred = np.dot(X, weights) + bias
    return y_pred
    """ YOUR CODE END HERE """