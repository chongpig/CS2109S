def cost_function(X: np.ndarray, y: np.ndarray, weight_vector: np.ndarray):
    '''
    Bianry cross entropy error for logistic regression

    Parameters
    ----------
    X: np.ndarray
        (n, d) training dataset (features).
    y: np.ndarray
        (n,) training dataset (corresponding targets).
    weight_vector: np.ndarray
        (d,) weight parameters.

    Returns
    -------
    BCE cost
    '''
    
    # Machine epsilon for numpy `float64` type
    eps = np.finfo(np.float64).eps

    """ YOUR CODE HERE """
    n = X.shape[0]
    z = X @ weight_vector
    y_pred = 1/(1 + np.exp(-z))
    y_pred = np.clip(y_pred, eps, 1 - eps)
    bce = -(1/n)*np.sum(y*np.log(y_pred)+(1 - y)*np.log(1 - y_pred))
    return bce
    """ YOUR CODE END HERE """