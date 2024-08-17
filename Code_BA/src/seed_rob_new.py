import json
import random
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from load_models import load_models
from svm_functions import connotative_hyperplane, distance_svm

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

def bootstrap_svm(seed_words_pos, seed_words_neg, model, target_word, n_iterations=10, ci=95):
    """
    Perform bootstrapping on the seed sets with resampling and calculate the confidence interval.
    """
    distances = []

    for _ in range(n_iterations):
        # Resample the seed words with replacement
        sampled_pos_words = random.choices(seed_words_pos, k=len(seed_words_pos))
        sampled_neg_words = random.choices(seed_words_neg, k=len(seed_words_neg))

        # Access the corresponding vector representations
        sampled_pos_vectors = [model.wv[word] for word in sampled_pos_words]
        sampled_neg_vectors = [model.wv[word] for word in sampled_neg_words]

        # Train SVM and calculate distance
        svm = connotative_hyperplane(sampled_pos_vectors, sampled_neg_vectors)
        distance = distance_svm(svm, model.wv[target_word])

        distances.append(float(distance))

    # Calculate the confidence interval
    lower_bound = np.percentile(distances, (100 - ci) / 2)
    upper_bound = np.percentile(distances, 100 - (100 - ci) / 2)
    mean_distance = np.mean(distances)
    

    return mean_distance, lower_bound, upper_bound, distances

def main():
    target_words = ["nation"]
    relative_changes = defaultdict(lambda: defaultdict(list))  # To store relative changes per dimension per set (left/right)
    models_all = load_models('left', 'right', spellchecker=False)
    models = {'left': models_all['left'][0], 'right': models_all['right'][0]}  # Use only one model for demonstration purposes

    if not models:
        print("No models loaded. Please check the set and ensure models are correctly loaded.")
        return None

    dic_seeds = load_seeds()
    if not dic_seeds:
        return None

    for target_word in target_words:
        for dim in dic_seeds:
            print(f"Processing dimension: {dim} for target word: {target_word}")
            pos_seeds = dic_seeds[dim]['pos_pole']
            neg_seeds = dic_seeds[dim]['neg_pole']

            for set_name, model in models.items():
                mean_distance, lower_bound, upper_bound, all_distances = bootstrap_svm(pos_seeds, neg_seeds, model, target_word)

                # Calculate range of all distances
                range_of_distances = max(all_distances) - min(all_distances)
                
                if range_of_distances > 0:  # Avoid division by zero
                    # Calculate the relative change of the CI
                    relative_change = (upper_bound - lower_bound) / range_of_distances
                    print(f"Confidence interval range: upper bound {upper_bound:.4f}, lower bound {lower_bound:.4f}")
                    print(f"Range of distances: {range_of_distances:.4f}")
                    relative_changes[set_name][dim].append(relative_change)
                    print(f"Relative change: {relative_change:.4f} for {dim} in {target_word} ({set_name} set)")
                else:
                    print(f"Range of distances is zero for {dim} in {target_word} ({set_name} set), skipping relative change calculation.")

    # Calculate the average relative change per dimension across all target words, separately for left and right
    avg_relative_changes = {
        set_name: {dim: np.mean(changes) for dim, changes in dims.items()}
        for set_name, dims in relative_changes.items()
    }

    # Print the results
    print("\nAverage Relative Changes per Dimension (Left/Right):")
    for set_name, dims in avg_relative_changes.items():
        print(f"\n{set_name.capitalize()}-leaning Set:")
        for dim, avg_change in dims.items():
            print(f"{dim}: {avg_change:.4f}")

if __name__ == "__main__":
    main()
