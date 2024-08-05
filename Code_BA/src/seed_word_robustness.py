import json
import random
import itertools
import numpy as np
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
        print("Couldn't find seeds at specified path.")
        return None

    return dic_seeds

def seed_set_combinations(pos_seeds, neg_seeds, seeds_per_set=5, sample_size=2):
    """
    Return a random sample of combinations of a positive seed set with a negative seed set.
    """
    pos_combinations = list(itertools.combinations(pos_seeds, seeds_per_set))
    neg_combinations = list(itertools.combinations(neg_seeds, seeds_per_set))

    # Shuffle the combinations to ensure randomness
    random.shuffle(pos_combinations)
    random.shuffle(neg_combinations)

    # Ensure we have enough combinations to sample from
    max_samples = min(len(pos_combinations), len(neg_combinations), sample_size)

    # Select combinations without replacement
    sampled_combinations = [
        (list(pos_combinations[i]), list(neg_combinations[i]))
        for i in range(max_samples)
    ]
    
    return sampled_combinations

def calculate_variances_and_means(distances_svm):
    """
    Calculate the variance and mean of distances for each connotative dimension.
    """
    variances = {key: np.var(list(distances.values())) for key, distances in distances_svm.items()}
    means = {key: np.mean(list(distances.values())) for key, distances in distances_svm.items()}
    return variances, means

def main(): 
    target_word = "nation"
    set_name = 'left'
    distances_multiple_svm = defaultdict(lambda: defaultdict(list))
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

        # Ensure seed_vectors_1 and seed_vectors_2 are converted to vectors for training
        seed_vectors_pos = [model.wv[word] for word in pos_seeds if word in model.wv]
        seed_vectors_neg = [model.wv[word] for word in neg_seeds if word in model.wv]

        # Train a single SVM on the complete seed set for Comparison
        svm_all = connotative_hyperplane(seed_vectors_pos, seed_vectors_neg)
        distance_all = distance_svm(svm_all, model.wv[target_word])
        print(f"Distance to connotative hyperplane for dimension {key} on complete seed set: {distance_all}")

        seed_set_combinations_list = seed_set_combinations(pos_seeds, neg_seeds)  # get combinations of seed sets

        for pos_sample, neg_sample in seed_set_combinations_list:
            try:
                seed_vectors_pos = [model.wv[word] for word in pos_sample]
                seed_vectors_neg = [model.wv[word] for word in neg_sample]
            except KeyError:
                print("One or more seed words not found in the embedding space. Please check the seeds file.")
                continue

            # Train multiple SVMs on different subsets of seed sets
            svm = connotative_hyperplane(seed_vectors_neg, seed_vectors_pos)
            distance = distance_svm(svm, model.wv[target_word])
            distances_multiple_svm[key][target_word].append(distance)

        
            

    variance_multiple_svm, mean_multiple_svm = calculate_variances_and_means(distances_multiple_svm)


    print(f"Variance of distance to connotative hyperplane for multiple SVMs: {variance_multiple_svm}")
    print(f"Mean distance to connotative hyperplane for multiple SVMs: {mean_multiple_svm}")

if __name__ == "__main__":
    main()
