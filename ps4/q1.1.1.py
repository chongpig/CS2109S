def update_centroids(X, labels, n_clusters):
    '''
    Updates the centroids based on the (new) assignment of clusters.

    Parameters
    ----------
    X: np.ndarray
        An `N * d` matrix where `N` is the number of samples and `d` is the
        number of features which each sample has. In other words, the `i`th sample
        is given by `X[i]`.
    labels: np.ndarray
        An array of `N` values, where `N` is the number of samples, that indicates
        which cluster the samples have been assigned to, i.e. the `i`th
        sample is assigned to the `labels[i]`th cluster.
    n_clusters: int
        No. of clusters.

    Returns
    -------
    The `centroids`, an `ndarray` with shape `(n_clusters, d)`, for each cluster,
    based on the current cluster assignment as specified by `labels`. In particular,
    `centroids[j]` returns the centroid for the `j`th cluster.
    '''
    # TODO: add your solution here and remove `raise NotImplementedError`  
    # at most 1 loop allowed
    
    """ YOUR CODE HERE """
    d = X.shape[1]
    centroids = np.zeros((n_clusters, d))
    for k in range(n_clusters):
        points = X[labels == k]
        centroids[k] = np.mean(points, axis=0)
    return centroids
    """ YOUR CODE END HERE """