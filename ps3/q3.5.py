def logistic_regression_stochastic_gradient_descent(X_train: np.ndarray, y_train: np.ndarray, max_num_iterations: int=250, threshold: np.float64=0.05, gamma: np.float64=1e-5, seed: int=43) -> np.ndarray:
    '''
    Initialize your weight to zeros. Write a terminating condition, and run the weight update for some iterations.
    Get the resulting weight vector.

    Parameters
    ----------
    X_train: np.ndarray
        (n, d) training dataset (features).
    y_train: np.ndarray
        (n,) training dataset (corresponding targets).
    max_num_iterations: int
        this should be one of the terminating conditions. 
        The gradient descent step should happen at most max_num_iterations times.
    threshold: np.float64
        terminating when error <= threshold value, or if you reach the max number of update rounds first.
    gamma: np.float64
        logistic regression learning rate.
    seed: int
        seed for random number generation.

    Returns
    -------
    The final (d,) weight parameters
    '''

    """ YOUR CODE HERE """
    np.random.seed(seed)
    n, d = X_train.shape
    w = np.zeros(d)
    current_loss = cost_function(X_train, y_train, w)
    for i in range(max_num_iterations):
        if current_loss <= threshold:
            break
        idx = np.random.choice(n, size=1)[0]
        X_i = X_train[idx:idx+1]
        y_i = y_train[idx:idx+1]
        z = X_i @ w
        y_pred = 1 / (1 + np.exp(-z))
        error = y_pred - y_i
        grad = X_i.T @ error
        w -= gamma * grad.flatten()
        current_loss = cost_function(X_train, y_train, w)
    return w
    """ YOUR CODE END HERE """