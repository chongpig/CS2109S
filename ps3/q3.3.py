def weight_update(X: np.ndarray, y: np.ndarray, gamma: np.float64, weight_vector: np.ndarray) -> np.ndarray:
    '''
    Do the weight update for one step in gradient descent

    Parameters
    ----------
    X: np.ndarray
        (n, d) training dataset (features).
    y: np.ndarray
        (n,) training dataset (corresponding targets).
    gamma: np.float64
        logistic regression learning rate.
    weight_vector: np.ndarray
        (d,) weight parameters.

    Returns
    -------
    New weight vector after one round of update.
    '''

    """ YOUR CODE HERE """
    n = X.shape[0]
    z = X @ weight_vector
    y_pred = 1/(1 + np.exp(-z))
    error = y_pred - y
    grad = (1/n)*(X.T @ error)
    weight_vector = weight_vector - gamma * grad
    return weight_vector
    """ YOUR CODE END HERE """