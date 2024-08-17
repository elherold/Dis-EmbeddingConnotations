import os
os.environ["OMP_NUM_THREADS"] = "1"  # to avoid memory usage warning due to error in scikit-learn on Windows machines

import numpy as np
import matplotlib.pyplot as plt
import json
import seaborn as sns
import pandas as pd
from svm_functions import connotative_hyperplane, distance_svm
from collections import defaultdict
from load_models import load_models

def connotation_heatmap(distances, target_word, lower_bound, upper_bound, cbar=True):
    """
    Generates a heatmap with the distances to the decision boundary for each connotation dimension for the target words.
    """
    # Convert the distances dictionary to a DataFrame
    df = pd.DataFrame(distances)

    # Debug: Print the DataFrame to check its content for debugging
    print("DataFrame to be plotted:")
    print(df)

    if df.empty:
        print("The DataFrame is empty. No heatmap will be generated.")
        return
    
    plt.figure(figsize=(10, 8))
    
    # Generate a heatmap
    ax = sns.heatmap(df.T, cbar=cbar, annot=False, cmap='coolwarm', vmin=lower_bound, vmax=upper_bound, cbar_kws={'label': 'Distance to Decision Boundary'})
    
    if cbar:
        cbar = ax.collections[0].colorbar
        cbar.ax.set_ylabel('Distance to Decision Boundary', fontsize=17)
        cbar.ax.tick_params(labelsize=15)
    
    plt.title(f'Distances to Decision Boundary for: {target_word}', fontsize=18)
    plt.xlabel('Source datasets', fontsize=18)
    plt.xticks(fontsize=15)
    plt.ylabel('Connotation Dimensions', fontsize=18)
    plt.yticks(fontsize=15)
    
    # Save the heatmap as a file
    output_dir = 'data/figures/heatmaps'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    output_path = os.path.join(output_dir, f'heatmap_{target_word}_centroid.png')
    plt.savefig(output_path)
    
    # Show the heatmap
    plt.show()

def main():
    target_words = ["nation", "spinster", "colorblindness"]
    target_words = [word.strip() for word in target_words]  # Strip whitespace
    models = load_models('left', 'right', spellchecker=False)
    print(f"Models loaded: {models.keys()}")
    dist = []
    
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
        average_distances = defaultdict(dict)

        for dimension in dic_seeds:
            print(f"Seed set: {dimension}")
            pos_seeds = dic_seeds[dimension]['pos_pole']
            neg_seeds = dic_seeds[dimension]['neg_pole']

            for set_name, model_list in models.items():
                set_distances = []

                for model in model_list:
                    try:
                        seed_vectors_pos = [model.wv[word] for word in pos_seeds]
                        seed_vectors_neg = [model.wv[word] for word in neg_seeds]
                    except KeyError:
                        print(f"One or more seed words not found in the embedding space of {dimension}.")
                        return None

                    svm_estimator = connotative_hyperplane(seed_vectors_pos, seed_vectors_neg)
                    distance = distance_svm(svm_estimator, model.wv[target_word])
                    set_distances.append(distance)

                if set_distances: 
                    avg_distance = np.mean(set_distances) 
                    distances[dimension][set_name].append(avg_distance) 

        # Update average distances after all dimensions are processed
        for dimension, sets in distances.items():
            for set_name, avg_dists in sets.items():
                average_distances[dimension][set_name] = np.mean(avg_dists)


    connotation_heatmap(average_distances, target_word, lower_bound=-1.06, upper_bound=0.94)

if __name__ == '__main__':
    main()
