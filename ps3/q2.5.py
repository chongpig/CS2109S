def get_prediction_poly_regression(X, y, power = 2, include_bias = True):
    '''
    Calculate the best polynomial line.

    Parameters
    ----------
    X (np.ndarray) : (n, 1) numpy matrix representing feature matrix
    y (np.ndarray) : (n, 1) numpy matrix representing target values
    power (int) : Specify the degree of the polynomial
    include_bias (boolean) : Specify whether the model should include a bias term

    Returns
    -------
        A (n, 1) numpy matrix representing prediction values.
    '''
    """ YOUR CODE HERE """
    poly_X = create_polynomial_matrix(X, power)
    return get_prediction_linear_regression(poly_X, y, include_bias)
    """ YOUR CODE END HERE """