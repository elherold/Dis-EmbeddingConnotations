import json
import random
import numpy as np
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
        print("Couldn't find seeds at specified path.")
        return None

    return dic_seeds

def seed_set_combinations(pos_seeds, neg_seeds, seeds_per_set=5):
    """
    Return non-overlapping combinations of a positive seed set and a negative seed set.
    """
    assert len(pos_seeds) == 2 * seeds_per_set, "The number of positive seeds must be exactly twice the seeds per set."
    assert len(neg_seeds) == 2 * seeds_per_set, "The number of negative seeds must be exactly twice the seeds per set."

    # Shuffle to randomize the selection
    random.shuffle(pos_seeds)
    random.shuffle(neg_seeds)

    # Create two non-overlapping subsets for positive and negative seeds
    pos_subset_1 = pos_seeds[:seeds_per_set]
    pos_subset_2 = pos_seeds[seeds_per_set:]
    
    neg_subset_1 = neg_seeds[:seeds_per_set]
    neg_subset_2 = neg_seeds[seeds_per_set:]

    # Return the two unique combinations
    sampled_combinations = [
        (pos_subset_1, neg_subset_1),
        (pos_subset_2, neg_subset_2)
    ]
    
    return sampled_combinations

def calculate_average_difference(distances_multiple_svm, distances_complete_svm):
    """
    Calculate the average difference between the distances of subset and complete set for each dimension.
    """
    avg_differences = {}

    for dimension, distances_dict in distances_multiple_svm.items():
        differences = []
        complete_distance = distances_complete_svm[dimension]

        for distance in distances_dict.values():
            for subset_distance in distance:
                differences.append(abs(complete_distance - subset_distance))

        if differences:
            avg_differences[dimension] = np.mean(differences)
        else:
            avg_differences[dimension] = None

    return avg_differences

def main(): 
    target_word = "nation"
    set_name = 'left'
    distances_multiple_svm = defaultdict(lambda: defaultdict(list))
    distances_complete_svm = {}
    models = load_models('left', 'right', spellchecker=False)
    model = models['left'][0]  # use only one model for demonstration purposes

    if not models:
        print("No models loaded. Please check the set and ensure models are correctly loaded.")
        return None

    if target_word not in model.wv:
        print(f"Word '{target_word}' not found in the embedding space. Please try another one")
        return None
    
    dic_seeds = load_seeds()
    if not dic_seeds:
        return None

    for key in dic_seeds:
        print(f"Seed set: {key}")
        pos_seeds = dic_seeds[key]['pos_pole']
        neg_seeds = dic_seeds[key]['neg_pole']

        # Ensure seed vectors are converted to vectors for training
        seed_vectors_pos = [model.wv[word] for word in pos_seeds if word in model.wv]
        seed_vectors_neg = [model.wv[word] for word in neg_seeds if word in model.wv]

        # Train a single SVM on the complete seed set for comparison
        svm_all = connotative_dim(seed_vectors_pos, seed_vectors_neg)
        distance_all = distance_centroid(svm_all, model.wv[target_word])
        distances_complete_svm[key] = distance_all
        print(f"Distance to connotative hyperplane for dimension {key} on complete seed set: {distance_all}")

        # Get exactly two non-overlapping combinations
        seed_set_combinations_list = seed_set_combinations(pos_seeds, neg_seeds)

        for pos_sample, neg_sample in seed_set_combinations_list:
            try:
                seed_vectors_pos = [model.wv[word] for word in pos_sample]
                seed_vectors_neg = [model.wv[word] for word in neg_sample]
            except KeyError:
                print("One or more seed words not found in the embedding space. Please check the seeds file.")
                continue

            # Train multiple SVMs on different subsets of seed sets
            svm = connotative_dim(seed_vectors_pos, seed_vectors_neg)
            distance = distance_centroid(svm, model.wv[target_word])
            print(f"Distance to connotative hyperplane for dimension {key} on subset {pos_sample} and {neg_sample} of seed set: {distance}")	
            distances_multiple_svm[key][target_word].append(distance)

    # Calculate the average difference between the subset distances and the complete set distance for each dimension
    avg_differences = calculate_average_difference(distances_multiple_svm, distances_complete_svm)
    
    # Output the average differences
    for dimension, avg_diff in avg_differences.items():
        print(f"Average difference for dimension {dimension}: {avg_diff}")

if __name__ == "__main__":
    main()
