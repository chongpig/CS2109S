def gradient_descent_multi_variable(X, y, lr = 1e-5, number_of_epochs = 250):
    '''
    Bias and weight that gives the best fitting line.

    Parameters
    ----------
    X (np.ndarray) : (n, d) numpy matrix representing feature matrix
    y (np.ndarray) : (n, 1) numpy matrix representing target values
    lr (float) : Learning rate
    number_of_epochs (int) : Number of gradient descent epochs
    
    Returns
    -------
        bias (float):
            The bias constant
        weights (np.ndarray):
            A (d, 1) numpy matrix that specifies the weights.
        loss (list):
            A list where the i-th element denotes the MSE score at i-th epoch.
    '''
    # Do not change
    bias = 0
    weights = np.full((X.shape[1], 1), 0).astype(float)
    loss = []
    n = X.shape[0]
    for i in range(number_of_epochs):
        pred = X @ weights + bias
        error = pred - y
        mse = mean_squared_error(y, pred)
        loss.append(mse)
        grad_w = (1/n) * (X.T @ error)
        grad_b = (1/n) * np.sum(error)
        weights -= lr * grad_w
        bias -= lr * grad_b
    return bias, weights, loss