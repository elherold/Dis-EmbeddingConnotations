import gensim
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.covariance import EllipticEnvelope
import json
import seaborn as sns
import pandas as pd

def load_embeddings(filepath):
    """
    Loads the trained embeddings saved in the "models" folder.

    Args:
    filepath (str): The file path to the saved embeddings.

    Returns:
    gensim.models.Word2Vec: The loaded embeddings.

    Raises:
    FileNotFoundError: If the file does not exist.
    Exception: For any other exceptions that may occur.
    """
    try:
        # Get the current working directory
        current_path = os.getcwd()
        print(f"Current working directory: {current_path}")

        # Construct the full file path to the model
        model_path = os.path.join(current_path, filepath)
        print(f"Model path: {model_path}")

        # Load the model
        model = gensim.models.Word2Vec.load(model_path)

        return model

    except FileNotFoundError as fnf_error:
        print(f"File not found: {model_path}")
        print(fnf_error)
        raise fnf_error  # Re-raise the exception after logging

    except Exception as e:
        print(f"An error occurred while loading the embeddings: {e}")
        raise e  # Re-raise the exception after logging

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

def predict_class_distance(svm_model, target_vector):
    """
    Predicts the class of a target vector and returns the signed distance to the decision boundary.
    
    Parameters:
    svm_model (SVC): The trained SVM model.
    target_vector (numpy.ndarray): The vector to predict the class for.
    
    Returns:
    float: The signed distance to the decision boundary.
    """
    # Predict the class and the distance to the decision boundary
    prediction = svm_model.predict([target_vector])
    distance = svm_model.decision_function([target_vector])
    
    return prediction[0], distance[0]

def select_seed_words(word2vec_model, seeds, num_words=10):
    seed_vectors  = [word2vec_model.wv[word] for word in seeds if word in word2vec_model.wv.key_to_index]
    seeds = [word for word in seeds if word in word2vec_model.wv.key_to_index]  # Ensure valid words

    if len(seed_vectors) < len(seeds):
        print("Some words from the seed set were not found in the embedding space.")
    if num_words > len(seed_vectors):
        num_words = len(seed_vectors)
        print(f"The number of words to select must be smaller or equal to the number of seed words. The number of words to select has been set to {len(seed_vectors)}.")
    envelope = EllipticEnvelope(support_fraction=None).fit(seed_vectors)
    distances = envelope.mahalanobis(seed_vectors)
    selected_indices = np.argsort(distances)[:num_words]
    coherent_subset = [seed_vectors[i] for i in selected_indices]
    nl_subset = [seeds[i] for i in selected_indices]  # Ensure valid words
    print(f"Selected words: {nl_subset}")
    print(f"words that did not make the cut: {list(set(seeds) - set(nl_subset))}")
    return nl_subset  # Return words, not vectors

def connotation_heatmap(distances, X, name_dataset, cbar=True):
    """
    Generates a heatmap with the distances to the decision boundary for each connotation dimension for the target_word and its n-nearest neighbors. 
    The words are on the X-axis and the connotation dimensions are on the Y-axis. The color intensity represents the distance to the decision boundary.
    The original target word is the first instance of X and on the left side of the heatmap. The N-nearest neighbors are following according to their proximity to the target word in the input list.

    Args:
    distances (dict): A dictionary containing the distances for each connotation dimension. The indeces of the values correspond to the same order as the target words in X.
    X (list): A target word and its n-nearest neighbors. The target word is at index 0 and the n-nearest neighbors follow according to their proximity.
    name_dataset (str): The name of the dataset.

    Returns:
    None
    """
    
    # Convert the distances dictionary to a DataFrame
    df = pd.DataFrame(distances, index=X)
    
    # Set up the matplotlib figure
    plt.figure(figsize=(10, 8))
    
    # Generate a heatmap
    sns.heatmap(df.T, cbar=cbar, annot=True, fmt=".2f", cmap='coolwarm', vmin=-1, vmax=1, cbar_kws={'label': 'Distance to Decision Boundary'})
    
    # Set the title and labels
    plt.title(f'Heatmap of Distances to Decision Boundary for {X[0]} from {name_dataset}')
    plt.xlabel('Target Word and Neighbors')
    plt.ylabel('Connotation Dimensions')
    
    # Save the heatmap as a file
    output_dir = 'data/figures/heatmaps'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    output_path = os.path.join(output_dir, f'heatmap_{X[0]}_{name_dataset}.png')
    plt.savefig(output_path)
    
    # Show the heatmap
    plt.show()

def main():
    name = 'word2vec_processed_AskFeminists_comments.pkl'
    model = load_embeddings(f'models/{name}.model')
    target_word = "climate"
    predictions = {}
    distances = {}

    if target_word in model.wv.key_to_index:
        print(f"Word '{target_word}' found in the embedding space.")
    else:
        print(f"Word '{target_word}' not found in the embedding space. Please try another one")
        return None

    nearest_neighbors = [word for word, _ in model.wv.most_similar(target_word, topn=10)]
    target_words = [target_word] + nearest_neighbors

    try:
        with open('data/data_helper/seeds_cleaned.json', 'r') as f:
            dic_seeds = json.load(f)
            print("Seeds have been cleaned already.")
    except FileNotFoundError:
        print("Seeds have not been cleaned yet.")
        with open('data/data_helper/seeds.json', 'r') as f:
            dic_seeds = json.load(f)

        for key in dic_seeds:
            print(f"Seed set: {key}")
            pos_seeds = dic_seeds[key]['pos_pole']
            neg_seeds = dic_seeds[key]['neg_pole']
            print(f"Positive seeds: {pos_seeds}")
            print(f"Negative seeds: {neg_seeds}")

            num_words = 10
            seed_vectors_1 = select_seed_words(model, pos_seeds, num_words)
            seed_vectors_2 = select_seed_words(model, neg_seeds, num_words)

            dic_seeds[key]['pos_pole'] = seed_vectors_1
            dic_seeds[key]['neg_pole'] = seed_vectors_2

        with open('data/data_helper/seeds_cleaned.json', 'w') as f:
            json.dump(dic_seeds, f, indent=4)

    for key in dic_seeds:
        pos_seeds = dic_seeds[key]['pos_pole']
        neg_seeds = dic_seeds[key]['neg_pole']

        # Ensure seed_vectors_1 and seed_vectors_2 are converted to vectors for training
        seed_vectors_1 = [model.wv[word] for word in pos_seeds]
        seed_vectors_2 = [model.wv[word] for word in neg_seeds]

        svm_model = connotative_hyperplane(seed_vectors_1, seed_vectors_2)
        predictions[key] = []
        distances[key] = []

        for word in target_words:
            prediction, distance = predict_class_distance(svm_model, model.wv[word])
            predictions[key].append(prediction)
            distances[key].append(distance)

    connotation_heatmap(distances, target_words, name)
    return None

if __name__ == '__main__':
    main()





