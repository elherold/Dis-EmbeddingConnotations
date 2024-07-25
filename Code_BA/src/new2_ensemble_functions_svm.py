from sklearn.svm import SVC
from sklearn.metrics.pairwise import cosine_similarity
import os
import numpy as np

# Set the environment variable to avoid the memory leak warning
os.environ["OMP_NUM_THREADS"] = "1"

def cosine_kernel(X, Y):
    """
    Compute the cosine similarity between two matrices X and Y.

    Parameters: 
    X (array-like): First input matrix of shape (n_samples_X, n_features).
    Y (array-like): Second input matrix of shape (n_samples_Y, n_features).

    Returns:
    array-like: Cosine similarity matrix of shape (n_samples_X, n_samples_Y).
    """
    return cosine_similarity(X, Y)

def compute_weighted_centroid(seed_vectors, frequencies):
    """
    Computes the weighted centroid for given seed vectors using their frequencies as weights.

    Parameters:
    seed_vectors (array-like): Input seed vectors of shape (n_samples, n_features).
    frequencies (array-like): Frequencies of the seed words of shape (n_samples,).

    Returns:
    array-like: Weighted centroid of the seed vectors of shape (n_features,).
    """
    weighted_sum = np.dot(frequencies, seed_vectors)
    total_weight = np.sum(frequencies)
    centroid = weighted_sum / total_weight
    return centroid

def train_svms_on_different_subsets(model, seed_vectors_pos, pos_seeds, seed_vectors_neg, neg_seeds):
    """
    Trains an SVM classifier on the weighted centroids of seed vectors using cosine kernel.

    Parameters:
    seed_vectors_1 (array-like): Seed vectors for the first class of shape (n_samples_1, n_features).
    frequencies_1 (array-like): Frequencies of the first class seed words of shape (n_samples_1,).
    seed_vectors_2 (array-like): Seed vectors for the second class of shape (n_samples_2, n_features).
    frequencies_2 (array-like): Frequencies of the second class seed words of shape (n_samples_2,).

    Returns:
    SVC: Trained SVM classifier
    """
    # Get frequencies for seed words
    frequencies_pos = np.array([model.wv.get_vecattr(word, "count") for word in pos_seeds])
    frequencies_neg = np.array([model.wv.get_vecattr(word, "count") for word in neg_seeds])

    # Compute weighted centroids
    centroid_pos = compute_weighted_centroid(seed_vectors_pos, frequencies_pos)
    centroid_neg = compute_weighted_centroid(seed_vectors_neg, frequencies_neg)

    # Combine centroids and labels
    X = np.vstack((centroid_pos, centroid_neg))
    y = np.array([1, -1])

    # Train SVM
    svm = SVC(kernel=cosine_kernel, probability=True)  # use cosine kernel with probability estimates to enable soft voting
    svm.fit(X, y)

    return svm

def distance_from_svms(svm, target_vector):
    """
    Compute the distance of a target vector from the decision boundary of an SVM classifier.

    Parameters:
    svm (SVC): Trained SVM classifier.
    target_vector (array-like): Target vector of shape (n_features,).

    Returns:
    float: Distance of the target vector from the decision boundary.
    """
    target_vector = np.array(target_vector).reshape(1, -1)  # reshape target vector for compatibility with sklearn
    distance = svm.decision_function(target_vector)
    return distance[0]