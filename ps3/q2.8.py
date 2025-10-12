def find_number_of_epochs(X, y, lr, delta_loss):
    '''
    Do gradient descent until convergence and return number of epochs
    required.

    Parameters
    ----------
    X (np.ndarray) : (n, d) numpy matrix representing feature matrix
    y (np.ndarray) : (n, 1) numpy matrix representing target values
    lr (float) : Learning rate
    delta_loss (float) : Termination criterion
    
    Returns
    -------
        bias (float):
            The bias constant
        weights (np.ndarray):
            A (d, 1) numpy matrix that specifies the weights.
        num_of_epochs (int):
            Number of epochs to reach convergence.
        current_loss (float):
            The loss value obtained after convergence.
    '''
    # Do not change
    bias = 0
    weights = np.full((X.shape[1], 1), 0).astype(float)
    num_of_epochs = 0
    previous_loss = 1e14
    current_loss = -1e14

    n = X.shape[0]
    pred = X @ weights + bias
    current_loss = mean_squared_error(y, pred)
    while abs(previous_loss - current_loss) >= delta_loss:
        """ YOUR CODE HERE """
        previous_loss = current_loss
        error = pred - y
        dw = (2 / n) * (X.T @ error)
        db = (2 / n) * np.sum(error)
        weights -= lr * dw
        bias -= lr * db
        pred = X @ weights + bias
        current_loss = mean_squared_error(y, pred)
        num_of_epochs += 1
        """ YOUR CODE END HERE """
    
    return bias, weights, num_of_epochs, current_loss