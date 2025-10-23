def compress_image(image, n_colours, threshold, n_init=1, random_state=None):
    '''
    Compresses the given image by reducing the number of colours in the image to
    `n_colours`. The `n_colours` colours should be selected using `k_means`.

    Parameters
    ----------
    image: np.ndarray
        The image to be compressed. It should be an array with a dimension of `h * w * 3` filled with integers,
        where `h` and `w` are its height and width respectively.
    n_colours: int
        No. of colours that the compressed image should contain.
    threshold: double
        A positive numerical value that determines the termination condition of the
        K-Means algorithm. You MUST call `k_means` with this threshold.
    n_init: int
        No. of times to run the K-Means algorithm before the best solution is
        picked and used for compression.
    random_state: int or `None`
        Used to make the algorithm deterministic, if specified. You MUST call
        `k_means` with `random_state` to ensure reproducility.

    Returns
    -------
    An `ndarray` with the shape `(h, w, 3)`, representing the compressed image
    which only contains `n_colours` colours. Note that the entries should be 
    integers, not doubles or floats.
    '''
    # TODO: add your solution here and remove `raise NotImplementedError`
    # no loop allowed

    """ YOUR CODE HERE """
    h = image.shape[0]
    w = image.shape[1]
    X = image.reshape(-1, 3).astype(float)
    labels, centroids = k_means(X, n_colours, threshold, n_init, random_state)
    compressed_flat = centroids[labels]
    compressed_image = compressed_flat.reshape(h, w, 3).astype(np.int_)
    return compressed_image
    """ YOUR CODE END HERE """