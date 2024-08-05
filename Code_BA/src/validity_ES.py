import json
import numpy as np
import random
from load_models import load_models
from reliability_ES import (
    calculate_second_order_similarity_vectors,
    calculate_mean_cosine_distances
)

def load_inter_distances(file_path):
    """
    Load the list of tuples containing words and their inter-dataset distances.

    Args:
        file_path (str): The path to the file containing the list of tuples.

    Returns:
        inter_distances (list): List of tuples (word, inter-distance).
    """
    with open(file_path, 'r', encoding='utf-8') as jsonfile:
        inter_distances = json.load(jsonfile)
    return inter_distances

def stratify_sample(inter_distances, top_pct=25, middle_pct=25, bottom_pct=25):
    """
    Stratify the words based on their inter-dataset distances and return one random word from each stratified group.

    Args:
        inter_distances (list): List of tuples (word, inter-distance).
        top_pct (int): Percentage for the top group.
        middle_pct (int): Percentage for the middle group.
        bottom_pct (int): Percentage for the bottom group.

    Returns:
        sample_words (list): List containing one word from the top, middle, and bottom groups.
    """
    inter_distances.sort(key=lambda x: x[1])  # Sort by inter-distance
    n = len(inter_distances)

    # Calculate the indices for stratification
    top_n = int(n * top_pct / 100)  # top_n words with the highest inter-distance
    middle_start = int(n * (50 - middle_pct / 2) / 100)
    middle_end = int(n * (50 + middle_pct / 2) / 100)
    bottom_n = int(n * bottom_pct / 100)

    # Using the indices for random sampling
    top_sample = random.choice(inter_distances[-top_n:])
    middle_sample = random.choice(inter_distances[middle_start:middle_end])
    bottom_sample = random.choice(inter_distances[:bottom_n])

    return [top_sample[0], middle_sample[0], bottom_sample[0]]

def compute_union_of_neighbors(models, word, k=10):
    """
    Compute the union of nearest neighbors across all training runs and average their distance scores.

    Args:
        models (list): List of word2vec models for a dataset.
        word (str): The target word.
        k (int): The number of nearest neighbors to return.

    Returns:
        average_scores (list): List of tuples (neighbor, average_score) sorted by average_score.
    """
    from collections import defaultdict
    neighbor_scores = defaultdict(list)
    all_neighbors = set()

    # First pass to collect all potential neighbors
    for model in models:
        if word in model.wv:
            neighbors = model.wv.most_similar(word, topn=k)
            for neighbor, _ in neighbors:
                all_neighbors.add(neighbor)

    # Second pass to collect scores for all potential neighbors from all models
    for model in models:
        if word in model.wv:
            for neighbor in all_neighbors:
                if neighbor in model.wv:
                    neighbor_scores[neighbor].append(model.wv.similarity(word, neighbor))

    # Calculate average scores
    average_scores = [(neighbor, np.mean(scores)) for neighbor, scores in neighbor_scores.items()]
    # Sort by average score in ascending order (smaller distances are closer neighbors)
    average_scores.sort(key=lambda x: x[1])

    return average_scores[:k]

def main():
    inter_distances_file = "data/data_helper/inter_distances.json"
    cleaned_words_file = "data/data_helper/cleaned_words_MachtSprache.json"
    k = 10

    # Load word2vec models
    models = load_models('left', 'right', spellchecker=False)

    # Load the inter-dataset distances
    inter_distances = load_inter_distances(inter_distances_file)

    # Stratify and sample one word from the top 10%, middle 10%, and bottom 10%
    sample_words = stratify_sample(inter_distances)

    # Print the k=10 nearest neighbors for each sampled word
    for word in sample_words:
        for set_name, models_list in models.items():
            top_nn = compute_union_of_neighbors(models_list, word, k)
            print(f"Target word: {word}")
            print(f"Nearest neighbors: {top_nn} for {set_name} dataset")


if __name__ == "__main__":
    main()
