import os
os.environ["OMP_NUM_THREADS"] = "1"  # to avoid memory usage warning due to error in scikit-learn on Windows machines
import sys
# Add the parent directory to the system path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import matplotlib.pyplot as plt
import json
import seaborn as sns
import pandas as pd
from svm_functions import connotative_hyperplane, distance_svm
from centroid_functions import connotative_dim, distance_centroid
from collections import defaultdict
from load_models import load_models

def connotation_heatmap(distances, target_word, lower_bound, upper_bound, cbar=True):
    """
    Generates a heatmap with the distances to the decision boundary for each connotation dimension for the target words.

    Parameters: 
    ------------
    distances (dict): A dictionary containing the distances to the decision boundary for each connotation dimension.
    target_word (str): The target word for which the distances were computed.
    lower_bound (float): The lower bound for the color scale of the heatmap.
    upper_bound (float): The upper bound for the color scale of the heatmap.
    cbar (bool): Whether to show the color bar on the heatmap.
    
    Returns:
    ------------
    None
    """
    # Convert the distances dictionary to a DataFrame
    df = pd.DataFrame(distances)

    # Check if the DataFrame is empty
    if df.empty:
        print("The DataFrame is empty. No heatmap will be generated.")
        return None
    
    plt.figure(figsize=(10, 8))
    
    # Generate a heatmap
    ax = sns.heatmap(df.T, cbar=cbar, annot=False, cmap='coolwarm', vmin=lower_bound, vmax=upper_bound, cbar_kws={'label': 'Distance to Decision Boundary'})
    
    # Set the labels and title
    if cbar:
        cbar = ax.collections[0].colorbar
        cbar.ax.set_ylabel('Distance to Decision Boundary', fontsize=17)
        cbar.ax.tick_params(labelsize=15)
    
    plt.title(f'Distances to Decision Boundary for: {target_word}', fontsize=18)
    plt.xlabel('Source datasets', fontsize=18)
    plt.xticks(fontsize=15)
    plt.ylabel('Connotation Dimensions', fontsize=18)
    plt.yticks(fontsize=15)
    
    # path to save the heatmap
    output_dir = 'data/figures/heatmaps'

    # Create the output directory if it doesn't exist yet
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # save the heatmap with current target word in title    
    output_path = os.path.join(output_dir, f'heatmap_{target_word}_centroid.png')
    plt.savefig(output_path)
    
    # Show the heatmap
    plt.show()

def main():
    """
    Main function to generate the heatmap for the target words.
        - Load the models
        - Load the seeds
        - Compute the distances
        - Generate the heatmap
    """
    # Define the target words
    target_words = ["nation", "spinster", "colorblindness"]

    # Define a methodology [1 = svm based, 0 = centroid based]
    method = 1
    

    # preprocess the target words for further processing
    target_words = [word.strip() for word in target_words]  # Strip whitespace
    target_words = [word.lower() for word in target_words]  # Convert to lowercase

    # Load the models
    models = load_models('left', 'right')
    print(f"Models loaded: {models.keys()}")
    
    # Check if all models are loaded correctly
    for set_name, model_list in models.items():
        for model in model_list:
            if not model:
                print(f"No models loaded. Please check {set_name} and ensure model is correctly loaded.")
                return None

    # Load seeds from file
    try:
        with open('data/data_helper/valid_seeds.json', 'r') as f:
            dic_seeds = json.load(f)
    except FileNotFoundError:
        print("Couldn't find seeds at the specified path.")
        return None
    
    # Process each target word in the list
    for target_word in target_words:
        print(f"\nProcessing target word: {target_word}")
        
        distances = defaultdict(lambda: defaultdict(list))

        # Process each connotative dimension in the seeds
        for dimension in dic_seeds:

            pos_seeds = dic_seeds[dimension]['pos_pole']
            neg_seeds = dic_seeds[dimension]['neg_pole']

            # Differentiate between models trained on left-leaning and right-leaning newspaper dataset
            for set_name, model_list in models.items():
                set_distances = []

                # Process each model in the list (there are 5 training runs for each dataset)
                for model in model_list:
                    try:
                        seed_vectors_pos = [model.wv[word] for word in pos_seeds]
                        seed_vectors_neg = [model.wv[word] for word in neg_seeds]
                    except KeyError:
                        print(f"One or more seed words not found in the embedding space of {dimension}.")
                        return None

                    # compute connotative hyperplane and distance to connotative hyperplane
                    if method:
                        # svm based method
                        svm_estimator = connotative_hyperplane(seed_vectors_pos, seed_vectors_neg)
                        distance = distance_svm(svm_estimator, model.wv[target_word])
                        set_distances.append(distance)
                    else: 
                        # centroid based method
                        conn_dim = connotative_dim(seed_vectors_pos, seed_vectors_neg)
                        distance = distance_centroid(conn_dim, model.wv[target_word])
                        set_distances.append(distance)

                # Averaging across training runs
                    avg_distance = np.mean(set_distances) 
                    distances[dimension][set_name] = avg_distance

        # Generate the heatmap for the current target word
        connotation_heatmap(distances, target_word, lower_bound=-1, upper_bound=1)

if __name__ == '__main__':
    main()
