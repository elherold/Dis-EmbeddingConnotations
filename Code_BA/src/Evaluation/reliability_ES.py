import os
import sys
# Adding parent directory to the system path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import json
import numpy as np
from scipy.spatial.distance import cosine
from scipy.stats import bootstrap
from load_models import load_models
import matplotlib.pyplot as plt


def load_macht_sprache_json(file_path):
    """
    Load the macht.sprache json file.

    Parameters:
    ------------
    file_path (str): filepath to macht sprache words 

    Returns: 
    -----------
    None
    """
    if not os.path.exists(file_path):
        print(f"File {file_path} not found.")
        return None
    
    with open(file_path, 'r', encoding='utf-8') as jsonfile:
        data = json.load(jsonfile)

    return data

def calculate_second_order_similarity_vectors(model_l, model_r, target_words, k=10):
    """
    Calculate the second order similarity vectors of the target words in the left and right models.  The similarity vectors are
    calculated by comparing the cosine similarity of the target words with their nearest neighbors in the left and right-leaning ESs.

    Parameters:
    ------------
    model_l (Word2Vec): left-leaning word2vec model
    model_r (Word2Vec): right-leaning word2vec model
    target_words (list): list of target words
    k (int): number of nearest neighbors to consider

    Returns:
    -------------
    similarity_vectors (dict): dictionary containing the second-order similarity vectors of the target words
    """
    similarity_vectors = {word: [] for word in target_words}

    # Calculate NNs for all target words
    for word in target_words:
        if word in model_l.wv and word in model_r.wv:
            neighbors_l = model_l.wv.most_similar(word, topn=k)  # get the NNs of the word in the left model
            neighbors_r = model_r.wv.most_similar(word, topn=k)  # get the NNs of the word in the right model
            
            # Combine the neighbors from both models
            combined_neighbors = set([n[0] for n in neighbors_l] + [n[0] for n in neighbors_r])
            
            vector_l = []
            vector_r = []

            # Calculate the cosine similarity between the target word and its neighbors in both models to calculate the second order similarity vectors
            for neighbor in combined_neighbors:
                if neighbor in model_l.wv and neighbor in model_r.wv:
                    vector_l.append(model_l.wv.similarity(word, neighbor))
                    vector_r.append(model_r.wv.similarity(word, neighbor))
        
            similarity_vectors[word].append((vector_l, vector_r))

    return similarity_vectors

def calculate_mean_cosine_distances(similarity_vectors):
    """
    Calculates the mean cosine distances between the similarity vectors of the target words.

    Parameters:
    -------------
    similarity_vectors (dict): dictionary containing the similarity vectors of the target words

    Returns:
    -------------
    mean_cosine_distances (dict): dictionary containing the mean cosine distances of the target words
    """
    mean_cosine_distances = {word: np.mean([cosine(vector_r, vector_r1) for vector_r, vector_r1 in vectors])
                             for word, vectors in similarity_vectors.items() if vectors}
    return mean_cosine_distances

def bootstrap_mean_distances(targetwords, models, n_bootstraps=1000):
    """
    Perform bootstrapping to calculate the confidence intervals for the mean cosine distances of the target words.

    Parameters:
    -------------
    targetwords (list): list of target words
    models (dict): dictionary containing the left and right-leaning word2vec models
    n_bootstraps (int): number of bootstraps to perform

    Returns:
    -------------
    intra_bootstrap (BootstrapResults): bootstrapping results for the intra-dataset mean distances.
    inter_bootstrap (BootstrapResults): bootstrapping results for the inter-dataset mean distances.
    mean_intra_distances (dict): dictionary containing the mean intra-dataset distances of the target words.
    mean_inter_distances (dict): dictionary containing the mean inter-dataset distances of the target words.
    """

    # Calculate intra-dataset distances for the left models
    intra_distances_left = {word: [] for word in targetwords}
    for i in range(len(models['left'])):
        for j in range(i + 1, len(models['left'])):
            similarity_vectors = calculate_second_order_similarity_vectors(models['left'][i], models['left'][j], targetwords)
            for word, distances in calculate_mean_cosine_distances(similarity_vectors).items():
                intra_distances_left[word].append(distances)

    # Calculate intra-dataset distances for the right models
    intra_distances_right = {word: [] for word in targetwords}
    for i in range(len(models['right'])):
        for j in range(i + 1, len(models['right'])):
            similarity_vectors = calculate_second_order_similarity_vectors(models['right'][i], models['right'][j], targetwords)
            for word, distances in calculate_mean_cosine_distances(similarity_vectors).items():
                intra_distances_right[word].append(distances)

    # Calculate inter-dataset distances between left and right models
    inter_distances = {word: [] for word in targetwords}
    for i in range(len(models['left'])):
        for j in range(len(models['right'])):
            similarity_vectors = calculate_second_order_similarity_vectors(models['left'][i], models['right'][j], targetwords)
            for word, distances in calculate_mean_cosine_distances(similarity_vectors).items():
                inter_distances[word].append(distances)

    # Calculate mean intra-dataset distances
    mean_intra_distances = {word: (np.mean(intra_distances_left[word] + intra_distances_right[word]))
                            for word in targetwords if intra_distances_left[word] and intra_distances_right[word]}
    
    # Calculate mean inter-dataset distances
    mean_inter_distances = {word: np.mean(inter_distances[word]) for word in targetwords if inter_distances[word]}

    # Perform bootstrapping on intra and inter means
    intra_samples = np.array(list(mean_intra_distances.values()))  # Convert to numpy array for bootstrapping
    inter_samples = np.array(list(mean_inter_distances.values()))

    intra_bootstrap = bootstrap((intra_samples,), np.mean, n_resamples=n_bootstraps)  # Perform bootstrapping
    inter_bootstrap = bootstrap((inter_samples,), np.mean, n_resamples=n_bootstraps)

    return intra_bootstrap.confidence_interval, inter_bootstrap.confidence_interval, mean_intra_distances, mean_inter_distances



def filter_target_words(targetwords, models):
    """
    Filter target words to those present in all models.

    Parameters
    -------------
    targetwords (list): list of target words.
    models (dict): dictionary containing the left and right-leaning Word2Vec models

    Returns:
    --------------
    filtered_words (list): list of target words present in all models.
    """
    filtered_words = []
    for word in targetwords:
        if all(word in model.wv for model_set in models.values() for model in model_set):
            filtered_words.append(word)

    print(f"Number of target words present in all models: {len(filtered_words)}")
    return filtered_words

def plot_mean_distances_boxplot(intra_means, inter_means, intra_ci, inter_ci, title, ylabel, save_path=None):
    """
    Plot box plot diagrams for intra and inter mean distances of the target words.

    Parameters:
    ------------
    intra_means (dict): dictionary containing the intra-dataset mean distances of the target words.
    inter_means (dict): dictionary containing the inter-dataset mean distances of the target words.
    intra_ci (tuple): confidence interval for the intra-dataset mean distances.
    inter_ci (tuple): confidence interval for the inter-dataset mean distances.
    title (str): title of the plot.
    ylabel (str): label for the y-axis.

    Returns:
    ----------
    None
    """
    # Plot box plot bar-charts with confidence intervals
    plt.figure(figsize=(10, 6))
    
    # Data for box plots
    data = [list(intra_means.values()), list(inter_means.values())]
    labels = ['Intra-dataset', 'Inter-dataset']
    
    # Create box plots
    plt.boxplot(data, labels=labels, patch_artist=True, showmeans=True, showfliers=False)
    
    # Add mean and confidence intervals
    for i, (means, ci) in enumerate(zip([intra_means, inter_means], [intra_ci, inter_ci])):
        mean_value = np.mean(list(means.values()))
        plt.errorbar(i + 1, mean_value, yerr=[[mean_value - ci.low], [ci.high - mean_value]], fmt='o', color='black')

    # Labeling the axes
    plt.xlabel('Type of Shift', fontsize=14)
    plt.ylabel(ylabel, fontsize=14)
    
    # Title of the plot
    plt.title(title, fontsize=16, fontweight='bold')
    
    # Adding grid for better readability
    plt.grid(True)

    # Save plot if save_path is provided
    if save_path:
        plt.savefig(save_path)
        print(f"Plot saved to {save_path}")
    
    # Display the plot
    plt.show()

def main():
    """
    Main function to 
        - perform bootstrapping 
        - and plot the results for the macht.sprache target words.
    """

    filepath = "data/data_helper/cleaned_words_MachtSprache.json"
    
    # Load word2vec models
    models = load_models('left', 'right')
    
    # Load macht.sprache json file
    targetwords = load_macht_sprache_json(filepath)

    if not targetwords:
        print("No target words found.")
        return

    # Filter target words to those present in all models
    targetwords = filter_target_words(targetwords, models)

    # Perform bootstrapping for target words
    print("Performing bootstrapping to calculate confidence intervals for mean distances...")
    intra_ci, inter_ci, mean_intra_distances, mean_inter_distances = bootstrap_mean_distances(targetwords, models)

    # Define the save path for the plot
    save_folder = '../../data/figures/'
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    save_path = os.path.join(save_folder, 'mean_cosine_distances_plot.png')

    # Plot boxplots for intra vs inter for target words
    title = 'Comparison of Mean Cosine Distances for Intra and Inter-Dataset Variations (Target Words)'
    ylabel = 'Mean Cosine Distance'
    plot_mean_distances_boxplot(mean_intra_distances, mean_inter_distances, intra_ci, inter_ci, title, ylabel, save_path)
    

if __name__ == "__main__":
    main()
