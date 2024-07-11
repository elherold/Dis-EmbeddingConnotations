    
import json
import random
import itertools
from old_heatmap_methods.connotative_heatmap import load_models
from svm_ensemble_functions import train_svms_on_centroids

def load_seeds():
    """
    This function is used to check the robustness of the seed words.
    """
    try:
        with open('data/data_helper/seeds.json', 'r') as f:
                dic_seeds = json.load(f) # store list of seeds in a dictionary
    except FileNotFoundError:
        print("couldn't find seeds at specified path.")


        for key in dic_seeds:
            print(f"Seed set: {key}")
            pos_seeds = dic_seeds[key]['pos_pole'] # assign seeds to negative and positive poles
            neg_seeds = dic_seeds[key]['neg_pole']
            print(f"Positive seeds: {pos_seeds}") # print to check if it worked
            print(f"Negative seeds: {neg_seeds}")
    
    return pos_seeds, neg_seeds

def seed_set_combinations(pos_seeds, neg_seeds, seeds_per_set=10, sample_size=50):
    """
    This function returns a random sample of combinations of a positive seed set with a negative seed set,
    with a specified number of seeds per set out of an original set of 20 seeds.

    Parameters:
    pos_seeds (list): A list of positive seed vectors.
    neg_seeds (list): A list of negative seed vectors.
    seeds_per_set (int): The number of seeds per set to select (default is 10).
    sample_size (int): The number of random combinations to return (default is 50).

    Returns:
    list of tuples: A list of tuples where each tuple contains two lists: 
                    a combination of positive seeds and a combination of negative seeds.
    """
    
    pos_combinations = list(itertools.combinations(pos_seeds, seeds_per_set))
    neg_combinations = list(itertools.combinations(neg_seeds, seeds_per_set))

    # Randomly sample combinations without creating the full cartesian product
    sampled_combinations = []
    for _ in range(sample_size):
        pos_sample = random.choice(pos_combinations)
        neg_sample = random.choice(neg_combinations)
        sampled_combinations.append((list(pos_sample), list(neg_sample)))
    
    return sampled_combinations

def main(): 
    target_word = "feminist"
    set = 'set_B'
    distances_multiple_svm = {}
    distances_single_svm = {}
    models = load_models(set, spellchecker=False)
    model = models[0] # use only one model for demonstration purposes

    # Check if models are loaded
    if not models:
        print("No models loaded. Please check the set and ensure models are correctly loaded.")
        return None

    # Check if target word is in the embedding space
    if target_word in models[0].wv.key_to_index: # if the word is in one ES, it is in all of them as they're trained on the exact same vocabulary
        print(f"Word '{target_word}' found in the embedding space.")
    else:
        print(f"Word '{target_word}' not found in the embedding space. Please try another one")
        return None
    
    # Load seeds from file
    try:
        with open('data/data_helper/seeds.json', 'r') as f:
                dic_seeds = json.load(f) # store list of seeds in a dictionary
    except FileNotFoundError:
        print("couldn't find seeds at specified path.")

    # iterate through connotative dimensions (keys) of interest
    for key in dic_seeds:
            print(f"Seed set: {key}") # current connotative dimension
            pos_seeds = dic_seeds[key]['pos_pole'] # assign seeds to negative and positive poles as specified in file
            neg_seeds = dic_seeds[key]['neg_pole']

            # Ensure seed_vectors_1 and seed_vectors_2 are converted to vectors for training
            
            try:
                seed_vectors_1 = [model.wv[word] for word in pos_seeds]
                seed_vectors_2 = [model.wv[word] for word in neg_seeds]
            except KeyError:
                 print("One or more seed words not found in the embedding space. Please check the seeds file.")

            print(f"Positive seeds: {pos_seeds}") # print to check if it worked
            print(f"Negative seeds: {neg_seeds}")

            seed_set_combinations = seed_set_combinations(pos_seeds, neg_seeds) # get all possible combinations of seed sets
            
            for seed_set in seed_set_combinations:
                 
                distances_multiple_svm[key] = defaultdict(list)  # Initialize as defaultdict of lists
        
                ### APPROACH 1: Train SVMs on centroids
                svm_estimators = train_svms_on_centroids(seed_vectors_1, seed_vectors_2, n_clusters=3)

                distance = distance_from_svms(svm_estimators, model.wv[word])
                distances[key][word].append(distance)


                ### APPROACH 2: Train a single SVM
                svm = connotative_hyperplane(seed_vectors_1, seed_vectors_2)
                distance = distance_svm(svm, model.wv[word])
                distances_single_svm[key][word].append(distance)
            
            # store the variance of distance to connotative hyperplane for current key for both approaches
            variance_multiple_svm = np.var(list(distances_multiple_svm[key].values()))
            variance_single_svm = np.var(list(distances_single_svm[key].values()))

    print(f"Variance of distance to connotative hyperplane for multiple SVMs: {variance_multiple_svm}")
    print(f"Variance of distance to connotative hyperplane for single SVM: {variance_single_svm}")
    # rewrite into dataframe!

