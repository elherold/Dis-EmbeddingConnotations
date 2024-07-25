from sklearn.svm import SVC
from sklearn.metrics.pairwise import cosine_similarity
import os
import numpy as np
from sklearn.cluster import KMeans
from gensim.models import Word2Vec

# Set the environment variable to avoid the memory leak warning
os.environ["OMP_NUM_THREADS"] = "1"

def cosine_kernel(X, Y):
    return cosine_similarity(X, Y)

def compute_clusters(seed_vectors, n_clusters):
    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(seed_vectors)
    clusters = [np.where(kmeans.labels_ == i)[0] for i in range(n_clusters)]
    return clusters, kmeans.cluster_centers_

def pair_clusters_by_similarity(centroids_1, centroids_2):
    similarity_matrix = cosine_similarity(centroids_1, centroids_2)
    pairs = []
    for i in range(similarity_matrix.shape[0]):
        j = np.argmax(similarity_matrix[i])
        pairs.append((i, j))
        similarity_matrix[:, j] = -np.inf
    return pairs

def get_word_frequencies_from_model(model, seed_words):
    """
    Extracts word frequencies from a trained Gensim Word2Vec model for given seed words.

    Parameters:
    model (Word2Vec): The trained Gensim Word2Vec model.
    seed_words (list): List of seed words to get frequencies for.

    Returns:
    dict: Dictionary with seed words as keys and their frequencies as values.
    """
    return {word: model.wv.get_vecattr(word, 'count') for word in seed_words if word in model.wv.key_to_index}

def calculate_cluster_weights(clusters, seed_words, word_frequencies):
    weights = []
    for cluster in clusters:
        weight = sum(word_frequencies.get(seed_words[idx], 0) for idx in cluster) / len(cluster)
        weights.append(weight)
    return weights

def train_svms_on_different_subsets(seed_vectors_1, seed_words_1, seed_vectors_2, seed_words_2, model, n_clusters=3, n_svm=5):
    svm_estimators = []
    weights = []

    # Get word frequencies for seed words
    word_frequencies_1 = {word: model.wv.get_vecattr(word, 'count') for word in seed_words_1 if word in model.wv.key_to_index}
    word_frequencies_2 = {word: model.wv.get_vecattr(word, 'count') for word in seed_words_2 if word in model.wv.key_to_index}
    print(f"Word Frequencies 1: {word_frequencies_1}")
    print(f"Word Frequencies 2: {word_frequencies_2}")

    clusters_1, centroids_1 = compute_clusters(seed_vectors_1, n_clusters)
    clusters_2, centroids_2 = compute_clusters(seed_vectors_2, n_clusters)

    cluster_pairs = pair_clusters_by_similarity(centroids_1, centroids_2)

    for _ in range(n_svm):
        for idx1, idx2 in cluster_pairs:
            subset_1 = np.array(seed_vectors_1)[clusters_1[idx1]]
            subset_2 = np.array(seed_vectors_2)[clusters_2[idx2]]

            X = np.vstack((subset_1, subset_2))
            y = np.array([1] * len(subset_1) + [-1] * len(subset_2))

            svm = SVC(kernel=cosine_kernel, probability=True)
            svm.fit(X, y)
            svm_estimators.append(svm)

            weight_1 = calculate_cluster_weights([clusters_1[idx1]], seed_words_1, word_frequencies_1)[0]
            weight_2 = calculate_cluster_weights([clusters_2[idx2]], seed_words_2, word_frequencies_2)[0]
            weight = (weight_1 + weight_2) / 2
            weights.append(weight)

            #print(f"Subset 1: {subset_1.shape}, Subset 2: {subset_2.shape}")
            #print(f"Weight 1: {weight_1}, Weight 2: {weight_2}, Combined Weight: {weight}")

    return svm_estimators, weights

def distance_from_svms(svm_estimators, weights, target_vector):
    distances = []
    target_vector = np.array(target_vector).reshape(1, -1)
    for svm, weight in zip(svm_estimators, weights):
        distance = svm.decision_function(target_vector)
        distances.append(distance[0] * weight)
        #print(f"Distance: {distance[0]}, Weighted Distance: {distance[0] * weight}")
    weighted_average_distance = np.sum(distances) / np.sum(weights)
    return weighted_average_distance


