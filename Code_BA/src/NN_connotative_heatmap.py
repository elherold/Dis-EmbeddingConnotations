import os
os.environ["OMP_NUM_THREADS"] = "1"  # to avoid memory usage warning due to error in scitkit-learn on windows machine

import numpy as np
import matplotlib.pyplot as plt
import json
import seaborn as sns
import pandas as pd
from new_ensemble_functions_svm import train_svms_on_different_subsets, distance_from_svms
from svm_functions import connotative_hyperplane, distance_svm
from collections import defaultdict
from load_models import load_models

def connotation_heatmap(distances, target_words, sets, dimensions, cbar=True):
    """
    Generates a heatmap with the distances to the decision boundary for each connotation dimension for the target words.
    The source sets are on the Y-axis and the target words are on the X-axis. The color intensity represents the distance to the decision boundary.

    Args:
    distances (dict): A dictionary containing the average distances for each connotation dimension.
                      The keys are the connotation dimensions, and the values are dictionaries with source sets as keys and distances as values.
    target_words (list): A list of target words with their nearest neighbors.
    sets (list): A list of the sets (e.g., 'set_A', 'set_B').
    dimensions (list): A list of the connotative dimensions.
    cbar (bool): Whether to include a color bar in the heatmap.

    Returns:
    None
    """
    # Convert the distances dictionary to a DataFrame with multi-level index and columns
    data = []
    for dimension in dimensions:
        for set_name in sets:
            for word in target_words:
                data.append([dimension, set_name, word, distances[dimension][set_name][word]])

    df = pd.DataFrame(data, columns=['Dimension', 'Set', 'Word', 'Distance'])
    df_pivot = df.pivot_table(index=['Dimension', 'Set'], columns='Word', values='Distance')

    # Debug: Print the DataFrame to check its content for debugging
    print("DataFrame to be plotted:")
    print(df_pivot)

    # Check if the DataFrame is empty for debugging
    if df_pivot.empty:
        print("The DataFrame is empty. No heatmap will be generated.")
        return
    
    if df_pivot.shape[0] == 1 or df_pivot.shape[1] == 1:
        print("The DataFrame has only one row or one column. This might cause the warning.")
    
    # Set up the matplotlib figure
    plt.figure(figsize=(12, 10))
    
    # Generate a heatmap
    sns.heatmap(df_pivot, cbar=cbar, annot=True, fmt=".2f", cmap='coolwarm', vmin=-1, vmax=1, cbar_kws={'label': 'Distance to Decision Boundary'})
    
    # Set the title and labels
    plt.title(f'Heatmap of Distances to Decision Boundary for {", ".join([word.split("_")[0] for word in target_words])}')
    plt.xlabel('Target Words and Nearest Neighbors')
    plt.ylabel('Connotation Dimensions and Sets')
    
    # Save the heatmap as a file
    output_dir = 'data/figures/heatmaps'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    output_path = os.path.join(output_dir, f'heatmap_weights_{target_words[0].split("_")[0]}_centroid.png')
    plt.savefig(output_path)
    
    # Show the heatmap
    plt.show()

def main():
    target_word = input("Please enter the target word (only single words): ").lower()
    models = load_models('feminism', 'antifeminism', spellchecker=False)
    print(f"Models loaded: {models.keys()}")
    # print the length of the models to check if they are loaded correctly
    print(f"Length of models in set A: {len(models['set_A'])}")
    print(f"Length of models in set B: {len(models['set_B'])}")
    distances = defaultdict(lambda: defaultdict(dict))
    neighbors = defaultdict(list)
    
    # Check if all models are loaded correctly and if the target word is in the vocabulary of all models
    for set_name, model_list in models.items():
        for model in model_list:
            if not model:
                print(f"No models loaded. Please check {set_name} and ensure model is correctly loaded.")
                return None

        if target_word in model_list[0].wv.key_to_index:  # Check in one model, assuming all models in the set have the same vocabulary
            print(f"Word '{target_word}' found in the embedding space of {set_name}.")
        else:
            print(f"Word '{target_word}' not found in the embedding space of {set_name}. Please try another one. The target word needs to be included in all embedding spaces.")
            return None

        # Find nearest neighbors of the target word
        nearest_neighbors = model_list[0].wv.most_similar(target_word, topn=3)
        neighbors[set_name] = [target_word] + [neighbor[0] for neighbor in nearest_neighbors]

    # Load seeds from file
    try:
        with open('data/data_helper/valid_seeds.json', 'r') as f:
            dic_seeds = json.load(f)  # store list of seeds in a dictionary
    except FileNotFoundError:
        print("couldn't find seeds at specified path.")
        return None

    # Iterate through connotative dimensions (keys) of interest
    for dimension in dic_seeds:
        print(f"Seed set: {dimension}")  # current connotative dimension
        pos_seeds = dic_seeds[dimension]['pos_pole']  # assign seeds to positive poles as specified in file
        neg_seeds = dic_seeds[dimension]['neg_pole']

        for set_name, model_list in models.items():  # Iterate through the model sets
            set_distances = []

            try:
                # Ensure seed_vectors_pos and seed_vectors_neg are converted to vectors for training
                seed_vectors_pos = [model_list[0].wv[word] for word in pos_seeds]  # all models in a set are trained on the same vocabulary
                seed_vectors_neg = [model_list[0].wv[word] for word in neg_seeds]
            except KeyError:
                print(f"One or more seed words not found in the embedding space of {dimension}. Please check the seeds file and make sure to use \"valid seeds\".")
                return None

            for model in model_list:  # Iterate through the list of models in each set
                svm_estimator = connotative_hyperplane(seed_vectors_neg, seed_vectors_pos)
                
                # Compute distance to decision boundary provided by the SVM for the target word and its nearest neighbors
                for word in neighbors[set_name]:
                    distance = distance_svm(svm_estimator, model.wv[word])
                    if word not in distances[dimension][set_name]:
                        distances[dimension][set_name][word] = []
                    distances[dimension][set_name][word].append(distance)
            
            # Average the distances for the current set and dimension
            print(f"These are the calculated distances for {set_name}: {set_distances}")
    
    # Update average distances after all dimensions are processed
    average_distances = defaultdict(lambda: defaultdict(dict))
    for dimension, sets in distances.items():
        for set_name, words in sets.items():
            for word, dist_list in words.items():
                average_distances[dimension][set_name][word] = np.mean(dist_list)

    target_words = list(set([word for neighbors_list in neighbors.values() for word in neighbors_list]))
    sets = ['set_A', 'set_B']
    dimensions = list(dic_seeds.keys())

    connotation_heatmap(average_distances, target_words, sets, dimensions)
    return None


if __name__ == '__main__':
    main()
