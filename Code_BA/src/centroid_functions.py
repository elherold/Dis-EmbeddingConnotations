import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def connotative_dim(seed_vectors_1, seed_vectors_2):
    """
    Computes the centroid of the seed vectors for each category.
    
    Parameters:
    seed_vectors_1 (numpy.ndarray): Seed vectors for the first category.
    seed_vectors_2 (numpy.ndarray): Seed vectors for the second category.
    
    Returns:
    numpy.ndarray: The centroid of the seed vectors for each category.
    """
    
    # Compute the centroid for each category
    centroid_1 = np.mean(seed_vectors_1, axis=0)
    centroid_2 = np.mean(seed_vectors_2, axis=0)
    
    # Compute the difference between the centroids
    dim = centroid_1 - centroid_2
    
    return dim

def distance_centroid(dim, target_vector):
    """
    Computes the cosine distance of a target vector to a dimension.
    
    Parameters:
    centroid_diff (numpy.ndarray): The difference between the centroids of the two categories.
    target_vector (numpy.ndarray): The vector to compute the distance for.
    
    Returns:
    float: The signed distance of the target vector to the dimension.
    """
        
    # Compute the cosine similarity between the target vector and the dimension
    similarity = cosine_similarity([target_vector], [dim])[0][0]
    
    return similarity
    