import csv
import os
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
train_data = load_digits_data_train()
validation_data = load_digits_data_validation()

train_digits = train_data
validation_digits = validation_data[0]
validation_labels = validation_data[1]
def k_means(X, n_clusters, threshold, n_init=1, random_state=None):
    '''
    Clusters samples in X using the K-Means algorithm.

    Parameters
    ----------
    X: np.ndarray
        An `N * d` matrix where `N` is the number of samples and `d` is the
        number of features which each sample has. In other words, the `i`th sample
        is given by `X[i]`.
    n_clusters: int
        No. of clusters.
    threshold: float
        Threshold that determines when the algorithm should terminate. If between
        two consecutive iterations the cluster centroids' difference is less than
        `threshold`, terminate the algorithm, i.e. suppose `c_i` is the ith
        centroid in the kth iteration, and `c'_i` is the ith centroid in the
        (k + 1)th iteration, we terminate the algorithm if and only if for all  
        i, d(`c_i`, `c'_i`) < `threshold`, where d is the distance function.
    n_init: int
        No. of times to run K-means.
    random_state: int or `None`
        Used to make the algorithm deterministic, if specified.
  
    Returns
    -------
    The cluster assignment for each sample, and the `n_clusters` centroids found. 
    In particular, the cluster assignment for the ith sample in `X` is given by `labels[i]`,
    where 0 <= `labels[i]` < `n_clusters`. Moreover, suppose c = `labels[i]`. Then,
    the `i`th sample belongs to the cluster c with the centroid given by `centroids[c]`.

    If `n_init` > 1, then the labels and corresponding centroids that result in
    the lowest distortion will be returned.

    Note
    ----
    If `n_init` is greater than 1, the labels and centroids found from the run
    (out of `n_init` runs) which gives the lowest distortion will be used.
    '''
    best_centroids, best_labels = None, None
    lowest_loss = np.inf

    for i in range(n_init):
        curr_random_state = None if random_state is None else random_state + i
        initial_centroids = init_centroids(X, n_clusters, curr_random_state)
        
        # TODO: add your solution between the next two lines of comment and remove `raise NotImplementedError`
        # no loop allowed

        """ YOUR CODE HERE """
        labels, centroids = k_means_once(X, initial_centroids, threshold)
        loss = compute_loss(X, centroids, labels)
        if loss < lowest_loss:
            lowest_loss = loss
            best_labels = labels
            best_centroids = centroids
        """ YOUR CODE END HERE """
  
    return best_labels, best_centroids

    
def predict_labels_kmeans_w_pca(pca, centroids, cluster_to_digit, digits):
    '''
    Predicts the digit labels for each digit in `digits`.
    
    Parameters
    ----------
    pca: PCA
        The PCA model that is used when training the K-Means clustering model,
        which produced `centroids`.
    centroids: np.ndarray
        The centroids of the clusters. Specifically, `centroids[j]` should represent
        the `j`th cluster's centroid.
    cluster_to_digit: np.ndarray
        A 1D array such that `cluster_to_digit[j]` indicates which digit the `j`th
        cluster represents. For example, if the 5th cluster represents the digit 0,
        then `cluster_to_digit[5]` should evaluate to 0.
        digits: np.ndarray
        An `N * d` matrix, where `N` is the number of handwritten digits and `d` is
        equal to 28*28. In particular, `digits[i]` represents the image
        of the `i`th handwritten digit that is in the data set.
    digits: np.ndarray
        An `N * d` matrix, where `N` is the number of handwritten digits and `d` is
        equal to 28*28. In particular, `digits[i]` represents the image
        of the `i`th handwritten digit that is in the data set.

    Returns
    -------
    A 1D np.ndarray `pred_labels` with `N` entries such that `pred_labels[i]`
    returns the predicted digit label for the image that is represented by
    `digits[i]`.
    '''
    # TODO: add your solution here and remove `raise NotImplementedError`
    # no loop allowed

    """ YOUR CODE HERE """
    digits_reduced = pca.transform(digits)
    labels = assign_clusters(digits_reduced, centroids)
    pred_labels = cluster_to_digit[labels]
    return pred_labels
    """ YOUR CODE END HERE """
_, centroids = k_means(train_digits, 10, 2, 5, 2109)
pred_labels_kmeans = predict_labels_kmeans(centroids, cluster_to_digit, validation_digits)
accuracy_kmeans = compute_accuracy(pred_labels_kmeans, validation_labels)
print('Accuracy of K-Means (w/o PCA): {}'.format(accuracy_kmeans)) # might take some time to run