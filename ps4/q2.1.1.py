def find_kmeans_clusters_w_pca(digits, n_categories, threshold=2,\
    n_init=5, random_state=2109, n_components=70):
    '''
    Finds the centroids of the `n_categories` clusters given `digits` when PCA
    is used to reduce the dimensionality of each image.
    
    Parameters
    ----------
    digits: np.ndarray
        An `N * d` matrix, where `N` is the number of handwritten digits and `d` is
        equal to 28*28. In particular, `digits[i]` represents the image of the `i`th
        handwritten digit.
    n_categories: int
        The number of distinct digits.
    threshold: double
        Threshold that determines when the K-means algorithm should terminate. This
        should be used with `k_means`.
    n_init: int
        The number of times to run the K-means algorithm before picking the best
        cluster. This should be used with `k_means`.
    random_state: int or `None`
        Used to make the K-means and PCA deterministic, if specified.
    n_components: int
        The dimension to which each sample point is reduced, using PCA.

    Returns
    -------
    An `n_categories * n_components` matrix `centroids`, where `centroids[j]` is 
    the centroid of the `j`th cluster, AND the PCA model that is used to reduce
    the dimension of each image.
    '''
    # TODO: add your solution here and remove `raise NotImplementedError`
    # no loop allowed

    """ YOUR CODE HERE """
    pca = PCA(n_components, random_state)
    digits_reduced = pca.fit_transform(digits)
    labels, centroids = k_means(digits_reduced, n_categories, threshold, n_init=n_init, random_state=random_state)
    return centroids, pca
    """ YOUR CODE END HERE """