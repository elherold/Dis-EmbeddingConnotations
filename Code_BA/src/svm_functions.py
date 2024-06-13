import numpy as np
from sklearn.svm import SVC
from sklearn.metrics.pairwise import cosine_similarity

def cosine_kernel(X, Y):
    """
    Computes the cosine similarity between two sets of vectors.

    Args:
    X (numpy.ndarray): First set of vectors.
    Y (numpy.ndarray): Second set of vectors.

    Returns:
    numpy.ndarray: The cosine similarity matrix.

    Raises:
    ValueError: If the input arrays have incompatible shapes.
    Exception: For any other exceptions that may occur.
    """
    try:
        X = np.array(X)
        Y = np.array(Y)

        # Ensure that inputs are numpy arrays
        if not isinstance(X, np.ndarray) or not isinstance(Y, np.ndarray):
            raise ValueError("Both X and Y must be numpy arrays.")
        
        # Compute the cosine similarity
        similarity_matrix = cosine_similarity(X, Y)
        return similarity_matrix

    except ValueError as ve:
        print(f"ValueError: {ve}")
        raise ve  # Re-raise the exception after logging

    except Exception as e:
        print(f"An error occurred while computing cosine similarity: {e}")
        raise e  # Re-raise the exception after logging

def connotative_hyperplane(seed_vectors_1, seed_vectors_2):
    """
    Trains an SVM with a cosine kernel using the provided seed vectors.
    
    Parameters:
    seed_vectors_1 (numpy.ndarray): Seed vectors for the first category.
    seed_vectors_2 (numpy.ndarray): Seed vectors for the second category.
    
    Returns:
    SVC: The trained SVM model.
    """
    
    # Combine the seed vectors into a single dataset
    X = np.vstack((seed_vectors_1, seed_vectors_2))
    
    # Create some labels for differentiation: 1 for seed_vectors_1, -1 for seed_vectors_2
    y = np.array([1] * len(seed_vectors_1) + [-1] * len(seed_vectors_2))
    
    # Train the SVM with the custom cosine kernel
    svm = SVC(kernel=cosine_kernel)
    svm.fit(X, y)

    return svm 

def distance_svm(svm_model, target_vector):
    """
    Predicts the class of a target vector and returns the signed distance to the decision boundary.
    
    Parameters:
    svm_model (SVC): The trained SVM model.
    target_vector (numpy.ndarray): The vector to predict the class for.
    
    Returns:
    float: The signed distance to the decision boundary.
    """
    # Predict the class and the distance to the decision boundary
    #prediction = svm_model.predict([target_vector])
    distance = svm_model.decision_function([target_vector])
    
    return distance[0]