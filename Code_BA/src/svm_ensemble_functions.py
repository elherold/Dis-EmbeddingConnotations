import numpy as np
from sklearn.svm import SVC
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
import os

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

def compute_centroids(seed_vectors, n_clusters):
    """
    Computes centroids of clusters for given seed vectors using KMeans clustering. 

    Parameters:
    seed_vectors (array-like): Input seed vectors of shape (n_samples, n_features)
    n_clusters (int): Number of clusters to form.

    Returns:
    array-like: Centroids of clusters of shape (n_clusters, n_features).
    """
    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(seed_vectors)
    return kmeans.cluster_centers_

def train_svms_on_different_subsets(seed_vectors_1, seed_vectors_2, n_clusters=3, n_svm=5):
    """
    Trains multiple SVM classifiers on centroids of random subsets of seed vectors using cosine kernel.

    Parameters:
    seed_vectors_1 (array-like): Seed vectors for the first class of shape (n_samples_1, n_features).
    seed_vectors_2 (array-like): Seed vectors for the second class of shape (n_samples_2, n_features).
    n_clusters (int): Number of clusters to form.
    n_svm (int): Number of SVM classifiers to train.

    Returns:
    list: List of trained SVM classifiers
    """
    svm_estimators = [] # list to store trained SVMs

    for _ in range(n_svm):
        # Randomly sample subsets of seed vectors
        subset_indices_1 = np.random.choice(len(seed_vectors_1), len(seed_vectors_1) // 2, replace=False) # sample half of the seed vectors
        subset_indices_2 = np.random.choice(len(seed_vectors_2), len(seed_vectors_2) // 2, replace=False) # sample half of the seed vectors
        subset_1 = seed_vectors_1[subset_indices_1]
        subset_2 = seed_vectors_2[subset_indices_2]

        # Compute centroids for the subsets
        centroids_1 = compute_centroids(subset_1, n_clusters)
        centroids_2 = compute_centroids(subset_2, n_clusters)

        # Combine centroids and labels
        X = np.vstack((centroids_1, centroids_2)) # combine centroids into input array
        y = np.array([1] * len(centroids_1) + [-1] * len(centroids_2)) # create labels for centroids

        # Train SVM
        svm = SVC(kernel=cosine_kernel, probability=True) # use cosine kernel with probability estimates to enable soft voting
        svm.fit(X, y)
        svm_estimators.append(svm)

    return svm_estimators

def distance_from_svms(svm_estimators, target_vector):
    """
    Compute the average distance of a target vector from the decision boundaries of multiple SVM classifiers.

    Parameters:
    svm_estimators (list): List of trained SVM classifiers.
    target_vector (array-like): Target vector of shape (n_features,).

    Returns:
    float: Average distance of the target vector from the decision boundaries.
    """
    distances = [] 
    target_vector = np.array(target_vector).reshape(1, -1) # reshape target vector for compatibility with sklearn
    for svm in svm_estimators: # iterate through all previously trained SVMs (on different subsets of seed vectors)
        distance = svm.decision_function(target_vector)
        distances.append(distance[0])
    average_distance = np.mean(distances)
    return average_distance

# Example usage
seed_vectors_1 = np.random.rand(30, 300)  # Replace with actual positive sentiment seed vectors
seed_vectors_2 = np.random.rand(30, 300)  # Replace with actual negative sentiment seed vectors
target_vector = np.random.rand(300)  # Replace with actual target vector

# Train the ensemble model
#ensemble_model = train_ensemble_svm(seed_vectors_1, seed_vectors_2, n_clusters=3)

# Compute the distance for a target vector
#distance = distance_ensemble_svm(ensemble_model, target_vector)
#print(f"Distance from the decision boundary: {distance}")
