from load_models import load_models
from collections import defaultdict
import json
import os


def check_seeds_in_set(model_list, seeds):
    """
    Checks if the seeds are in the vocabulary of the models in the given list and calculates their average frequency.

    Args:
    model_list (list): List of Word2Vec models.
    seeds (list): List of seed words.

    Returns:
    tuple: A tuple containing:
        - valid (dict): A dictionary of valid seed words and their average frequency.
        - missing (set): A set of missing seed words.
    """
    valid = {}
    missing = set()
    for seed in seeds:
        if any(seed in model.wv.key_to_index for model in model_list):
            valid[seed] = sum(model.wv.key_to_index.get(seed, 0) for model in model_list) / len(model_list)
        else:
            missing.add(seed)
    return valid, missing

def validate_and_filter_seeds(models, dic_seeds):
    """
    Validates and filters the seeds, ensuring they are present in both sets of models. Orders the seeds by their frequency.

    Args:
    models (dict): Dictionary of Word2Vec models for different sets.
    dic_seeds (dict): Dictionary containing seed words for different connotative dimensions.

    Returns:
    defaultdict: A dictionary containing the final valid seeds for each dimension, ordered by frequency.
    """
    valid_seeds = defaultdict(lambda: defaultdict(dict))
    missing_seeds = defaultdict(lambda: defaultdict(dict))
    final_valid_seeds = defaultdict(lambda: defaultdict(list))

    # Iterate through connotative dimensions (keys) of interest
    for dimension in dic_seeds:
        print(f"Seed set: {dimension}")  # current connotative dimension
        pos_seeds = dic_seeds[dimension]['pos_pole']  # assign seeds to positive poles as specified in file
        neg_seeds = dic_seeds[dimension]['neg_pole']

        for set_name in models.keys():
            valid_pos, missing_pos = check_seeds_in_set(models[set_name], pos_seeds)
            valid_neg, missing_neg = check_seeds_in_set(models[set_name], neg_seeds)
            valid_seeds[dimension][set_name]['pos_pole'] = valid_pos
            valid_seeds[dimension][set_name]['neg_pole'] = valid_neg
            missing_seeds[dimension][set_name]['pos_pole'] = missing_pos
            missing_seeds[dimension][set_name]['neg_pole'] = missing_neg

            # Print missing seed words and valid seed counts
            print(f"Dimension: {dimension}, Set {set_name}, Missing Seeds (pos_pole): {missing_pos}, Valid Seeds: {len(valid_pos)}")
            print(f"Dimension: {dimension}, Set {set_name}, Missing Seeds (neg_pole): {missing_neg}, Valid Seeds: {len(valid_neg)}")

        # Intersect the valid seeds from both sets to find common valid seeds
        common_valid_pos = valid_seeds[dimension]['set_A']['pos_pole'].keys() & valid_seeds[dimension]['set_B']['pos_pole'].keys()
        common_valid_neg = valid_seeds[dimension]['set_A']['neg_pole'].keys() & valid_seeds[dimension]['set_B']['neg_pole'].keys()

        # Order the seeds by their frequency
        ordered_pos = sorted(common_valid_pos, key=lambda x: (valid_seeds[dimension]['set_A']['pos_pole'][x] + valid_seeds[dimension]['set_B']['pos_pole'][x]) / 2, reverse=True)
        ordered_neg = sorted(common_valid_neg, key=lambda x: (valid_seeds[dimension]['set_A']['neg_pole'][x] + valid_seeds[dimension]['set_B']['neg_pole'][x]) / 2, reverse=True)

        final_valid_seeds[dimension]['pos_pole'] = ordered_pos
        final_valid_seeds[dimension]['neg_pole'] = ordered_neg

    return final_valid_seeds


def main():
    """
    Main function to load models and seeds, validate and filter the seeds, and save the final valid seeds to a JSON file.
    """
    models = load_models('set_A', 'set_B', spellchecker=False)
    seed_file_path = 'data/data_helper/seeds.json'
    output_path = 'data/data_helper/valid_seeds.json'

    # check if the json file already exists under output_path
    if os.path.exists(output_path):
        print(f"File {output_path} already exists. Please remove it if you want to generate a new one.")
        return None
    
    # Load seeds from file
    try:
        with open(seed_file_path, 'r') as f:
            dic_seeds = json.load(f)  # store list of seeds in a dictionary
    except FileNotFoundError:
        print("Couldn't find seeds at the specified path.")
        exit(1)

    final_valid_seeds = validate_and_filter_seeds(models, dic_seeds)

    # Print final valid seeds for each dimension
    for dimension, poles in final_valid_seeds.items():
        print(f"Dimension: {dimension}")
        for pole, seeds in poles.items():
            print(f"Pole: {pole}, Final Valid Seeds: {len(seeds)}, Seeds: {seeds}")
    
    # Save final valid seeds to a JSON file 
    with open(output_path, 'w') as f:
        json.dump(final_valid_seeds, f, indent=4)

    # Print frequency counts for each valid seed
    #for dimension, sets in valid_seeds.items():
    #    print(f"Dimension: {dimension}")
    #    for set_name, poles in sets.items():
    #        print(f"Set: {set_name}")
    #        for pole, seeds in poles.items():
    #            print(f"Pole: {pole}, Seed Frequencies: {seeds}")

if __name__ == '__main__':
    main()
