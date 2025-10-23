def k_means_once(X, initial_centroids, threshold):
    '''
    Assigns each point in X to a cluster by running the K-Means algorithm
    once till convergence.

    Parameters
    ----------
    X: np.ndarray
        An `N * d` matrix where `N` is the number of samples and `d` is the
        number of features which each sample has. In other words, the `i`th sample
        is given by `X[i]`.
    initial_centroids: np.ndarray
        An `n_clusters * d` matrix, where `n_clusters` is the number of clusters and
        `d` is the number of features that each sample in X has. This matrix is such
        that the `i`th row represents the initial centroid of the `i`th cluster.
    threshold: double
        During the clustering process, if the difference in centroids between
        two consecutive iterations is less than `threshold`, the algorithm is
        deemed to have converged.

    Returns
    -------
    The cluster assignment for each sample, and the `n_clusters` centroids found. 
    In particular, the cluster assignment for the ith sample in `X` is given by `labels[i]`,
    where 0 <= `labels[i]` < `n_clusters`. Moreover, suppose c = `labels[i]`. Then,
    the `i`th sample belongs to the cluster c with the centroid given by `centroids[c]`.
    '''
    # TODO: add your solution here and remove `raise NotImplementedError`
    # at most 1 loop allowed

    """ YOUR CODE HERE """
    n_clusters = initial_centroids.shape[0]
    centroids = initial_centroids.copy()
    prev_centroids = centroids + 2*threshold
    while not check_convergence(prev_centroids, centroids, threshold):
        prev_centroids = centroids.copy()
        labels = assign_clusters(X, centroids)
        centroids = update_centroids(X, labels, n_clusters)
    return labels, centroids
    """ YOUR CODE END HERE """