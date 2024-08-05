from load_models import load_models
from collections import defaultdict
import json
import os

def check_seeds_in_set(model, seeds):
    """
    Checks if the seeds are in the vocabulary of the model and retrieves their frequencies.

    Args:
    model (Word2Vec): A Word2Vec model.
    seeds (list): List of seed words.

    Returns:
    tuple: A tuple containing:
        - valid (dict): A dictionary of valid seed words and their frequencies.
        - missing (set): A set of missing seed words.
    """
    valid = {}
    missing = set()
    for seed in seeds:
        if seed in model.wv:
            valid[seed] = model.wv.get_vecattr(seed, "count")
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
            valid_pos, missing_pos = check_seeds_in_set(models[set_name][0], pos_seeds)  # Check the first model in the list for each set_name
            valid_neg, missing_neg = check_seeds_in_set(models[set_name][0], neg_seeds)
            valid_seeds[dimension][set_name]['pos_pole'] = valid_pos
            valid_seeds[dimension][set_name]['neg_pole'] = valid_neg
            missing_seeds[dimension][set_name]['pos_pole'] = missing_pos
            missing_seeds[dimension][set_name]['neg_pole'] = missing_neg

            # Print missing seed words and valid seed counts
            print(f"Dimension: {dimension}, Set {set_name}, Missing Seeds (pos_pole): {missing_pos}, Valid Seeds: {len(valid_pos)}")
            print(f"Dimension: {dimension}, Set {set_name}, Missing Seeds (neg_pole): {missing_neg}, Valid Seeds: {len(valid_neg)}")

        # Intersect the valid seeds from both sets to find common valid seeds
        common_valid_pos = valid_seeds[dimension]['left']['pos_pole'].keys() & valid_seeds[dimension]['right']['pos_pole'].keys()
        common_valid_neg = valid_seeds[dimension]['left']['neg_pole'].keys() & valid_seeds[dimension]['right']['neg_pole'].keys()

        # Order the seeds by their frequency in descending order based on the average frequency in both sets
        ordered_pos = sorted(common_valid_pos, key=lambda x: (valid_seeds[dimension]['left']['pos_pole'][x] + valid_seeds[dimension]['right']['pos_pole'][x]) / 2, reverse=True)
        ordered_neg = sorted(common_valid_neg, key=lambda x: (valid_seeds[dimension]['left']['neg_pole'][x] + valid_seeds[dimension]['right']['neg_pole'][x]) / 2, reverse=True)

        # Select the top 10 seeds that are highly frequent in both sets
        final_valid_seeds[dimension]['pos_pole'] = [seed for seed in ordered_pos if seed in valid_seeds[dimension]['left']['pos_pole'] and seed in valid_seeds[dimension]['right']['pos_pole']][:10]
        final_valid_seeds[dimension]['neg_pole'] = [seed for seed in ordered_neg if seed in valid_seeds[dimension]['left']['neg_pole'] and seed in valid_seeds[dimension]['right']['neg_pole']][:10]

        # Print the seeds with the lowest and highest frequencies with their respective frequencies
        #print(f"Dimension: {dimension}, Lowest Frequency Seed (pos_pole): {ordered_pos[10]}, Frequency: {valid_seeds[dimension]['left']['pos_pole'][ordered_pos[-1]]}")
        #print(f"Dimension: {dimension}, Highest Frequency Seed (pos_pole): {ordered_pos[0]}, Frequency: {valid_seeds[dimension]['left']['pos_pole'][ordered_pos[0]]}")
        #print(f"Dimension: {dimension}, Lowest Frequency Seed (neg_pole): {ordered_neg[10]}, Frequency: {valid_seeds[dimension]['left']['neg_pole'][ordered_neg[-1]]}")
        #print(f"Dimension: {dimension}, Highest Frequency Seed (neg_pole): {ordered_neg[0]}, Frequency: {valid_seeds[dimension]['left']['neg_pole'][ordered_neg[0]]}")

    return final_valid_seeds

def main():
    """
    Main function to load models and seeds, validate and filter the seeds, and save the final valid seeds to a JSON file.
    """
    models = load_models('left', 'right', spellchecker=False)
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
    
    # Save final valid seeds to a JSON file 
    with open(output_path, 'w') as f:
        json.dump(final_valid_seeds, f, indent=4)

if __name__ == '__main__':
    main()
