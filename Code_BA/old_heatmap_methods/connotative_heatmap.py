import os
os.environ["OMP_NUM_THREADS"] = "1" # to avoid memory usage warning due to error in scitkit-learn on windows machine

import gensim
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.covariance import EllipticEnvelope
import json
import seaborn as sns
import pandas as pd
from svm_ensemble_functions import train_svms_on_centroids, distance_from_svms
from gensim.models import Word2Vec
from collections import defaultdict


def load_models(set_name, spellchecker, folder_path="models/new/individual_models"):
    """
    Loads the trained Word2Vec models saved in the specified folder. 
    Differentiates between spellchecker and non-spellchecker models and loads according to set name.

    Args:
    folder_path (str): The path to the folder containing the models.
    set_name (str): The name of the set to load the models for (Set_A or Set_B).
    Spellchecker (bool): Whether to load spellchecker models or not.

    Returns:
    list: A list  containing the loaded Word2Vec model.
    """
    models = []
    print(f"Loading models for {set_name} with spellchecker: {spellchecker}")
    # loading the correct model files depending on the set and whether we want a spellchecker model
    if spellchecker: 
        print("spellchecker detected, Loading spellchecker models")
        model_files = [f for f in os.listdir(folder_path) if f.endswith(f"{set_name}.model") and "spellchecker" in f.lower()]
    else:
        print(f"Loading non-spellchecker models for {set_name}")
        model_files = [f for f in os.listdir(folder_path) if f.endswith(f"{set_name}.model") and "spellchecker" not in f.lower()]

    print(f"Found {len(model_files)} models for {set_name} with spellchecker: {spellchecker}")
    # load the models
    for model_file in model_files:
        model_path = os.path.join(folder_path, model_file)
        model = Word2Vec.load(model_path)
        models.append(model)
    return models

"""
def load_embeddings(filepath):
    \"""
    Loads the trained embeddings saved in the "models" folder.

    Args:
    filepath (str): The file path to the saved embeddings.

    Returns:
    gensim.models.Word2Vec: The loaded embeddings.

    Raises:
    FileNotFoundError: If the file does not exist.
    Exception: For any other exceptions that may occur.
    \"""
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
"""

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
    output_path = os.path.join(output_dir, f'heatmap_{X[0]}_{name_dataset}_centroid.png')
    plt.savefig(output_path)
    
    # Show the heatmap
    plt.show()


def clean_seed_sets(model):
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
        
    return dic_seeds


def get_common_NN(models, target_word, n=10):
    """
    Get the n-nearest neighbors common to all models in the list.

    Args:
    models (list): A list of Word2Vec models.
    target_word (str): The target word to find the nearest neighbors for.
    n (int): The number of nearest neighbors to find.

    Returns:
    list: A list of 10-nearest neighbors common to all models.
    """
    
    while True:
        common_neighbors = []
        for model in models:
            nearest_neighbors = [word for word, _ in model.wv.most_similar(target_word, topn=n)]
            common_neighbors.append(nearest_neighbors)

        # calculate the intersection of NN of the models
        common_neighbors = set.intersection(*map(set, common_neighbors))

        if len(common_neighbors) >= 10:
            return list(common_neighbors)[:10]
        else:
            n += 1
            print(f"current size of common neighbors: {len(common_neighbors)}")

    return list(common_neighbors)

def main():
    set = 'set_B'
    models = load_models(set, spellchecker=False)
    target_word = "snowflake"
    distances = {}
    average_distances = {}

    # Check if models are loaded
    if not models:
        print("No models loaded. Please check the set and ensure models are correctly loaded.")
        return None

    if target_word in models[0].wv.key_to_index: # if the word is in one ES, it is in all of them as they're trained on the exact same vocabulary
        print(f"Word '{target_word}' found in the embedding space.")
    else:
        print(f"Word '{target_word}' not found in the embedding space. Please try another one")
        return None


    # get the set n-nearest neighbors common to all models in model
    common_nn = get_common_NN(models, target_word)

    # we save the target words together with its common nearest neighbors for further processing
    target_words = [target_word] + common_nn

    # clean the seed sets if necessary
    dic_seeds = clean_seed_sets(models[0]) # seed sets are the same for all models

    for key in dic_seeds:
        distances[key] = defaultdict(list)  # Initialize as defaultdict of lists

        pos_seeds = dic_seeds[key]['pos_pole']
        neg_seeds = dic_seeds[key]['neg_pole']

        for model in models:
            # Ensure seed_vectors_1 and seed_vectors_2 are converted to vectors for training
            seed_vectors_1 = [model.wv[word] for word in pos_seeds]
            seed_vectors_2 = [model.wv[word] for word in neg_seeds]

            # Train the SVM model
            #svm_model = connotative_hyperplane(seed_vectors_1, seed_vectors_2)
            #predictions[key] = []
            #distances[key] = []

            #for word in target_words:
            #    prediction, distance = distance_svm(svm_model, model.wv[word])
            #    predictions[key].append(prediction)
            #    distances[key].append(distance)

            #connotative_dimension = connotative_dim(seed_vectors_1, seed_vectors_2)
            #distances[key] = []

            #for word in target_words:
            #    distance = distance_centroid(connotative_dimension, model.wv[word])
            #    distances[key].append(distance)


            # Find the extreme words for the current dimension
            #extreme_words = find_extreme_words(svm_model, model, n=10)
            #print(f"Extreme words for {key}: {extreme_words}")

            # Train the SVMs on centroids
            svm_estimators = train_svms_on_centroids(seed_vectors_1, seed_vectors_2, n_clusters=3)

            for word in target_words:
                distance = distance_from_svms(svm_estimators, model.wv[word])
                distances[key][word].append(distance)

            
        # Calculate the average distance for each word in the current dimension
        average_distances[key] = {word: np.mean(dists) for word, dists in distances[key].items()}

        # Example usage of average_distances (if needed):
        for key, avg_dists in average_distances.items():
            print(f"Average distances for {key}: {avg_dists}")

    connotation_heatmap(average_distances, target_words, set)
    return None

if __name__ == '__main__':
    main()





