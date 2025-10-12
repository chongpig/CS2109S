def create_polynomial_matrix(X, power = 2):
    '''
    Create a polynomial matrix.
    
    Parameters
    ----------
    X: (n, 1) numpy matrix

    Returns
    -------
        A (n, power) numpy matrix where the i-th column denotes
            X raised to the power of i.
    '''
    """ YOUR CODE HERE """
    powers = np.arange(1, power + 1)
    poly_matrix = np.power(X, powers)
    return poly_matrix
    """ YOUR CODE END HERE """