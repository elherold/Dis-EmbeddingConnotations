import gensim
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.covariance import EllipticEnvelope
import json
import seaborn as sns

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
    # translate the natural language seeds into vectors
    seed_vectors = [word2vec_model.wv[word] for word in seeds if word in word2vec_model.wv.key_to_index]

    # check if all words were found in the embedding space
    if len(seed_vectors) < len(seeds):
        print("Some words from the seed set were not found in the embedding space.")
     
    if num_words > len(seed_vectors): # check if num_words is valid
        num_words = len(seeds)
        print(f"The number of words to select must be smaller or equal to the number of seed words. The number of words to select has been set to {len(seed_vectors)}.")

    # compute the subset of num_words that build the most coherent cluster in the embedding space
    envelope = EllipticEnvelope(support_fraction=None).fit(seed_vectors) # Fit EllipticEnvelope to seed vectors
    distances = envelope.mahalanobis(seed_vectors) # Compute the Mahalanobis distances of the seed vectors
    selected_indices = np.argsort(distances)[:num_words] # Select the num_words indices with the smallest distances
    coherent_subset = [seed_vectors[i] for i in selected_indices] # return the subset of vectors that form the most coherent cluster
    nl_subset = [seeds[i] for i in selected_indices if seeds[i] in word2vec_model.wv.key_to_index] # return the subset of words that form the most coherent cluster
    
    print(f"Selected words: {nl_subset}")	
    print(f"words that did not make the cut: {list(set(seeds) - set(nl_subset))}")

    return coherent_subset

def connotation_heatmap(data, row_labels, col_labels, ax=None, cbar_kw={}, cbarlabel="", **kwargs):
    pass

def main():
    # Load the trained embeddings
    name = 'comments'
    model = load_embeddings(f'models/{name}.model')

    # Load target words from the json file
    target_words = json.load(open('data/data_helper/macht.sprache_words.json'))
    # we only want the english words saved under the key lemma in the json to form the list of target words
    target_words = [word['lemma'] for word in target_words if word['lemma_lang'] == 'en']
    target_words = [word.lower() for word in target_words] # convert all words to lowercase

    final_target_words = []
    # Check if the target words are in the embedding space
    for word in target_words:
        if word in model.wv.key_to_index:
            final_target_words.append(word)
        else:
            print(f"Word '{word}' not found in the embedding space.")

    # Load the seeds from the json file
    dic_seeds = json.load(open('data/data_helper/seeds.json'))
    for key in dic_seeds:
        print(f"Seed set: {key}")
        pos_seeds = dic_seeds[key]['pos_pole'] 
        neg_seeds = dic_seeds[key]['neg_pole']
        print(f"Positive seeds: {pos_seeds}")
        print(f"Negative seeds: {neg_seeds}")

        # Select the most coherent subset of seed words
        num_words = 10
        seed_vectors_1 = select_seed_words(model, pos_seeds, num_words)
        seed_vectors_2 = select_seed_words(model, neg_seeds, num_words)
        
        # Train the SVM model
        svm_model = connotative_hyperplane(seed_vectors_1, seed_vectors_2)

        # Predict the class and distance for each target word
        predictions = []
        distances = []

        # Predict the class and distance for each final target word
        for word in final_target_words:
            prediction, distance = predict_class_distance(svm_model, model.wv[word])
            predictions.append(prediction)
            distances.append(distance)

        # Debug: Print lengths
        #print(f"Length of target_words: {len(target_words)}")
        #print(f"Length of final_target_words: {len(final_target_words)}")
        #print(f"Length of distances: {len(distances)}")
        #print(f"Length of predictions: {len(predictions)}")

        # Ensure lengths match
        if len(final_target_words) == len(distances) == len(predictions):
            # Plot the results
            plt.figure(figsize=(10, 6))
            plt.bar(final_target_words, distances, color=['red' if p == -1 else 'blue' for p in predictions])
            plt.axhline(0, color='black', linewidth=0.5)
            plt.xticks(rotation=45)
            plt.ylabel('Distance to Decision Boundary')
            plt.title(f'Distance of Target Words to Decision Boundary of dimension {key}')
            # save the figure if it doesnt exist already
            if not os.path.exists(f'data/figures/decision_distance_{key}_{name}.png'):
                plt.savefig(f'data/figures/Connotation_{key}_{name}.png')
            plt.show()
        else:
            print("Error: Lengths of final_target_words, distances, and predictions do not match.")

if __name__ == '__main__':
    main()





