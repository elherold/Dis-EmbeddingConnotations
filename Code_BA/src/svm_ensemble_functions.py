import numpy as np
from sklearn.svm import SVC
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
import os

# Set the environment variable to avoid the memory leak warning
os.environ["OMP_NUM_THREADS"] = "1"

def cosine_kernel(X, Y):
    return cosine_similarity(X, Y)

def compute_centroids(seed_vectors, n_clusters):
    kmeans = KMeans(n_clusters=n_clusters, random_state=0)
    kmeans.fit(seed_vectors)
    return kmeans.cluster_centers_

def train_svms_on_centroids(seed_vectors_1, seed_vectors_2, n_clusters=3):
    centroids_1 = compute_centroids(seed_vectors_1, n_clusters)
    centroids_2 = compute_centroids(seed_vectors_2, n_clusters)
    
    X = np.vstack((centroids_1, centroids_2))
    y = np.array([1] * len(centroids_1) + [-1] * len(centroids_2))
    
    svm_estimators = []
    for i in range(n_clusters):
        svm = SVC(kernel=cosine_kernel, probability=True)  # probability=True for soft voting
        svm.fit(X, y)
        svm_estimators.append(svm)
    
    return svm_estimators

def distance_from_svms(svm_estimators, target_vector):
    distances = []
    target_vector = np.array(target_vector).reshape(1, -1)
    for svm in svm_estimators:
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
