import json
import random
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from load_models import load_models
from svm_functions import connotative_hyperplane, distance_svm
from centroid_functions import distance_centroid, connotative_dim 


def load_seeds():
    """
    Load the seed words from a JSON file.
    """
    try:
        with open('data/data_helper/valid_seeds.json', 'r') as f:
            dic_seeds = json.load(f)
    except FileNotFoundError:
        print("Couldn't find seeds at the specified path.")
        return None
    return dic_seeds

def bootstrap_svm(seed_words_pos, seed_words_neg, model, target_word, n_iterations=1000):
    """
    Perform bootstrapping on the seed sets with resampling.
    """
    distances = []

    for _ in range(n_iterations):
        # Resample the seed words with replacement
        print(f"Resampling for iteration {_ + 1}")
        sampled_pos_words = random.choices(seed_words_pos, k=len(seed_words_pos))
        sampled_neg_words = random.choices(seed_words_neg, k=len(seed_words_neg))

        # Access the corresponding vector representations
        sampled_pos_vectors = [model.wv[word] for word in sampled_pos_words]
        sampled_neg_vectors = [model.wv[word] for word in sampled_neg_words]

        #print(f"The word labels for the sampled_pos: {sampled_pos_words}")
        #print(f"The word labels for the sampled_neg: {sampled_neg_words}")

        # Train SVM and calculate distance
        svm = connotative_dim(sampled_pos_vectors, sampled_neg_vectors)
        distance = distance_centroid(svm, model.wv[target_word])

        # Ensure the distance is a float and append to list
        try:
            distances.append(float(distance))
        except ValueError:
            print(f"Encountered non-numeric distance: {distance}")

    return distances

import matplotlib.pyplot as plt
import numpy as np

def plot_bootstrap_results(distances_svm, full_set_distances, target_word):
    """
    Plot boxplots with the mean and dispersion for each dimension, including the mean and CI, and the full set distance.
    """
    plt.figure(figsize=(14, 8))

    # Prepare data and labels
    labels = []
    data = []
    x_ticks = []
    
    for dim in distances_svm['left'].keys():
        data.append(distances_svm['left'][dim])
        data.append(distances_svm['right'][dim])
        labels.append('Left')
        labels.append('Right')
        x_ticks.append(f'{dim}\n(L)')
        x_ticks.append(f'{dim}\n(R)')

    # Create box plots
    box = plt.boxplot(data, patch_artist=True, showmeans=False, positions=range(len(data)))

    # Add means and confidence intervals
    for i, (dim, pos) in enumerate(zip(distances_svm['left'].keys(), range(0, len(data), 2))):
        # For left
        left_distances = distances_svm['left'][dim]
        right_distances = distances_svm['right'][dim]

        for j, distances in enumerate([left_distances, right_distances]):
            mean_value = np.mean(distances)
            ci_low, ci_high = np.percentile(distances, [2.5, 97.5])
            std_dev = np.std(distances)
            cv = std_dev / mean_value if mean_value != 0 else np.inf  # Coefficient of Variation

            # Print statistics
            print(f"\n{dim} ({labels[pos + j]}):")
            print(f"Mean: {mean_value}")
            print(f"95% Confidence Interval: [{ci_low}, {ci_high}]")
            print(f"Standard Deviation: {std_dev}")
            print(f"Coefficient of Variation (CV): {cv}")
            print(f"Full Set Distance: {full_set_distances[labels[pos + j].lower()][dim]}")

            # Plot mean as a black dot
            plt.plot(pos + j, mean_value, 'ko')

            # Plot confidence interval
            plt.errorbar(pos + j, mean_value, yerr=[[mean_value - ci_low], [ci_high - mean_value]], fmt='o', color='black')

            # Plot full set distance as a red dot
            plt.plot(pos + j, full_set_distances[labels[pos + j].lower()][dim], 'ro')

    # Set the x-ticks and their corresponding labels
    plt.xticks(ticks=range(len(data)), labels=x_ticks, rotation=45, ha='right')

    # Labeling the axes
    plt.xlabel('Connotative Dimensions', fontsize=14)
    plt.ylabel('Distance to Hyperplane', fontsize=14)

    # Title of the plot
    plt.title(f'Bootstrapped SVM Results with Full Set Comparison for {target_word}', fontsize=16, fontweight='bold')

    # Adding grid for better readability
    plt.grid(True)

    # Display the plot
    plt.tight_layout()
    plt.show()

def main(): 
    target_words = ["nation", "spinster", "colorblindness"]
    distances_multiple_svm = defaultdict(lambda: defaultdict(list))
    full_set_distances = defaultdict(dict)
    models_all = load_models('left', 'right', spellchecker=False)
    models = {'left': models_all['left'][0], 'right': models_all['right'][0]}  # Use only one model for demonstration purposes

    if not models:
        print("No models loaded. Please check the set and ensure models are correctly loaded.")
        return None
    
    dic_seeds = load_seeds()
    if not dic_seeds:
        return None

    for target_word in target_words:
        for set_name, model in models.items():
            
            if target_word not in model.wv:
                print(f"Word '{target_word}' not found in the embedding space. Please try another one")
                return None
            
            for dim in dic_seeds:
                print(f"Seed set: {dim}")
                pos_seeds = dic_seeds[dim]['pos_pole']
                neg_seeds = dic_seeds[dim]['neg_pole']

                # Perform bootstrapping on the seed sets using word labels
                distances = bootstrap_svm(pos_seeds, neg_seeds, model, target_word)
                distances_multiple_svm[set_name][dim].extend(distances)

                # Calculate distance using the full set (no sampling)
                seed_vectors_pos = [model.wv[word] for word in pos_seeds if word in model.wv]
                seed_vectors_neg = [model.wv[word] for word in neg_seeds if word in model.wv]
                svm_all = connotative_dim(seed_vectors_pos, seed_vectors_neg)
                full_set_distance = distance_centroid(svm_all, model.wv[target_word])
                full_set_distances[set_name][dim] = float(full_set_distance)  # Ensure this is also a float

        plot_bootstrap_results(distances_multiple_svm, full_set_distances, target_word)

if __name__ == "__main__":
    main()

