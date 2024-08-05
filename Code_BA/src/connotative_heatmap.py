import os
os.environ["OMP_NUM_THREADS"] = "1" # to avoid memory usage warning due to error in scitkit-learn on windows machine

import os
import numpy as np
import matplotlib.pyplot as plt
import json
import seaborn as sns
import pandas as pd
from new_ensemble_functions_svm import train_svms_on_different_subsets, distance_from_svms
#from new2_ensemble_functions_svm import train_svms_on_different_subsets, distance_from_svms
from svm_functions import connotative_hyperplane, distance_svm
from collections import defaultdict
from load_models import load_models
import sys
from centroid_functions import connotative_dim, distance_centroid

def connotation_heatmap(distances, target_word, cbar=True):
    """
    Generates a heatmap with the distances to the decision boundary for each connotation dimension for the target words.
    The source sets are on the X-axis and the connotation dimensions are on the Y-axis. The color intensity represents the distance to the decision boundary.

    Args:
    distances (dict): A dictionary containing the average distances for each connotation dimension.
                      The keys are the connotation dimensions, and the values are dictionaries with source sets as keys and distances as values.
    target_words (list): A list of target words with source sets specified (e.g., ['snowflake_set_A', 'snowflake_set_B']).
    cbar (bool): Whether to include a color bar in the heatmap.

    Returns:
    None
    """
    # Convert the distances dictionary to a DataFrame
    df = pd.DataFrame(distances)

    # Debug: Print the DataFrame to check its content for debugging
    print("DataFrame to be plotted:")
    print(df)

    # Check if the DataFrame is empty for debugging
    if df.empty:
        print("The DataFrame is empty. No heatmap will be generated.")
        return
    
    if df.shape[0] == 1 or df.shape[1] == 1:
        print("The DataFrame has only one row or one column. This might cause the warning.")
    
    # Set up the matplotlib figure
    plt.figure(figsize=(10, 8))
    
    # Generate a heatmap
    ax = sns.heatmap(df.T, cbar=cbar, annot=False, cmap='coolwarm', vmin=-1, vmax=1, cbar_kws={'label': 'Distance to Decision Boundary'})
    
    # Set the colorbar fontsize
    if cbar:
        cbar = ax.collections[0].colorbar
        cbar.ax.set_ylabel('Distance to Decision Boundary', fontsize=17)
        cbar.ax.tick_params(labelsize=15)
    
    # Set the title and labels
    plt.title(f'Distances to Decision Boundary for: {target_word[0].split("_")[0]}', fontsize=18)
    plt.xlabel('Source datasets', fontsize=18)
    plt.xticks(fontsize=15)
    plt.ylabel('Connotation Dimensions', fontsize=18)
    plt.yticks(fontsize=15)

    
    # Save the heatmap as a file
    output_dir = 'data/figures/heatmaps'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    output_path = os.path.join(output_dir, f'heatmap_{target_word[0].split("_")[0]}_centroid.png')
    plt.savefig(output_path)
    
    # Show the heatmap
    plt.show()

def main():
    target_word = input("Please enter the target word (only single words): ").lower()
    models = load_models('left', 'right', spellchecker=False)
    print(f"Models loaded: {models.keys()}")
    # print the length of the models to check if they are loaded correctly
    print(f"Length of models in set left: {len(models['left'])}")
    print(f"Length of models in set right: {len(models['right'])}")
    distances = defaultdict(lambda: defaultdict(list))
    average_distances = defaultdict(dict)
    

    # Before going into any calculations, check if all models are loaded correctly 
    # and if the target word is in the vocabulary of all models
    for set_name, model_list in models.items():
        for model in model_list:
            if not model:
                print(f"No models loaded. Please check {set_name} and ensure model is correctly loaded.")
                return None

        if target_word in model_list[0].wv.key_to_index: # target word check is in outer loop to avoid redundancy, as all models of a set are trained on the exact same vocabulary
            print(f"Word '{target_word}' found in the embedding space of {set_name}.")
        else:
            print(f"Word '{target_word}' not found in the embedding space of {set_name}. Please try another one. The target word needs to be included in all embedding spaces.")
            return None
    
    # Load seeds from file
    try:
        with open('data/data_helper/valid_seeds.json', 'r') as f:
                dic_seeds = json.load(f) # store list of seeds in a dictionary
    except FileNotFoundError:
        print("couldn't find seeds at specified path.")
        return None
    
    # iterate through connotative dimensions (keys) of interest
    for dimension in dic_seeds:
        print(f"Seed set: {dimension}") # current connotative dimension
        pos_seeds = dic_seeds[dimension]['pos_pole'] # assign seeds to negative and positive poles as specified in file
        neg_seeds = dic_seeds[dimension]['neg_pole']
        #print(f"seed sets assigned to pos and negative poles as specified in loaded json file")

        for set_name, model_list in models.items(): # Iterate through the model sets
            #print(f"Model set: {set_name}") # current model set
            set_distances = []

            for model in model_list: # Iterate through the list of models in each set

                try:
                # Ensure seed_vectors_1 and seed_vectors_2 are converted to vectors for training
                    seed_vectors_pos = [model.wv[word] for word in pos_seeds] # all models in a set are trained on the same vocabulary
                    seed_vectors_neg = [model.wv[word] for word in neg_seeds]
                    #print(f"Chosen seed words for in {dimension} seem to be valid vectors in {set_name}.")
                except KeyError:
                    print(f"One or more seed words not found in the embedding space of {dimension}. Please check the seeds file and make sure to use \"valid seeds\".")
                    return None
            
                #svm_estimator = connotative_hyperplane(seed_vectors_pos, seed_vectors_neg)
                connotative_dimen = connotative_dim(seed_vectors_pos, seed_vectors_neg)
                
                # Compute distance to decision boundary provided by the SVMs for the target word
                #distance = distance_svm(svm_estimator, model.wv[target_word])
                distance = distance_centroid(connotative_dimen, model.wv[target_word])
                set_distances.append(distance) # store distance for calculation of average of all models in the set
            
            # Average the distances for the current set
            print(f"These are the calculated distances for {set_name}: {set_distances}")
            if set_distances:
                avg_distance = np.mean(set_distances)
                distances[dimension][set_name].append(avg_distance)

    # Update average distances after all dimensions are processed
    for dimension, sets in distances.items():
        for set_name, avg_dists in sets.items():
            average_distances[dimension][set_name] = np.mean(avg_dists)

    connotation_heatmap(average_distances, [f'{target_word}_left', f'{target_word}_right'])
    return None

if __name__ == '__main__':
    main()





