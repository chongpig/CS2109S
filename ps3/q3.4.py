def logistic_regression_classification(X: np.ndarray, weight_vector: np.ndarray, prob_threshold: np.float64=0.5):
    '''
    Do classification task using logistic regression.

    Parameters
    ----------
    X: np.ndarray
        (n, d) training dataset (features).
    weight_vector: np.ndarray
        (d,) weight parameters.
    prob_threshold: np.float64
        the threshold for a prediction to be considered fraudulent.

    Returns
    -------
    Classification result as an (n,) np.ndarray
    '''

    """ YOUR CODE HERE """
    z = X @ weight_vector
    y_pred = 1 / (1 + np.exp(-z))
    return (y_pred >= prob_threshold)
    """ YOUR CODE END HERE """