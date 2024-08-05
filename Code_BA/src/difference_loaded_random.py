import json
import os
import numpy as np
import random
from scipy.spatial.distance import cosine
from scipy.stats import bootstrap
from load_models import load_models
from reliability_ES import calculate_second_order_similarity_vectors, calculate_mean_cosine_distances, filter_target_words
import matplotlib.pyplot as plt

def load_macht_sprache_json(file_path):
    """
    Loads the cleaned macht.sprache JSON data from a specified file path.
    """
    if not os.path.exists(file_path):
        print(f"File {file_path} not found.")
        return None

    with open(file_path, 'r', encoding='utf-8') as jsonfile:
        data = json.load(jsonfile)
    return data

def load_word_frequencies(file_path):
    """
    Load word frequencies from a JSON file.
    """
    with open(file_path, 'r', encoding='utf-8') as jsonfile:
        return json.load(jsonfile)

def get_random_words(frequencies, target_words, n):
    """
    Get a sample of random words with frequencies similar to the target words.
    """
    target_freqs = [frequencies[word] for word in target_words if word in frequencies]
    mean_freq = np.mean(target_freqs)
    std_freq = np.std(target_freqs)

    similar_words = [word for word, freq in frequencies.items() if mean_freq - std_freq <= freq <= mean_freq + std_freq and word not in target_words]
    sampled_words = random.sample(similar_words, n)
    return sampled_words

def calculate_inter_dataset_distances(models, target_words):
    """
    Calculate the mean inter-dataset distances for the given target words.
    """
    inter_distances = []

    for word in target_words:
        similarity_vectors = calculate_second_order_similarity_vectors(models['left'][0], models['right'][0], [word])
        mean_inter_distances = calculate_mean_cosine_distances(similarity_vectors)
        inter_distances.append(mean_inter_distances[word])

    return inter_distances

def bootstrap_mean(data, n_bootstraps=1000):
    """
    Perform bootstrapping to estimate the mean of the data.
    """
    boot_means = []

    for _ in range(n_bootstraps):
        sample = np.random.choice(data, size=len(data), replace=True)
        boot_means.append(np.mean(sample))

    return boot_means

def main():
    macht_sprache_filepath = "data/data_helper/cleaned_words_MachtSprache.json"
    frequencies_filepath = "data/data_helper/top_words_by_frequency.json"
    n_samples = 20
    n_bootstraps = 1000
    
    # Load word2vec models
    models = load_models('left', 'right', spellchecker=False)
    
    # Load macht.sprache json file
    target_words = load_macht_sprache_json(macht_sprache_filepath)
    if not target_words:
        print("No target words found.")
        return

    # Filter target words to those present in all models
    target_words = filter_target_words(target_words, models)
    if len(target_words) < n_samples:
        print("Not enough target words present in all models.")
        return
    
    # Load word frequencies
    word_frequencies = load_word_frequencies(frequencies_filepath)
    
    # Get random words with similar frequencies
    random_words = get_random_words(word_frequencies, target_words, n_samples)

    # Calculate inter-dataset distances for target words and random words
    target_inter_distances = calculate_inter_dataset_distances(models, target_words[:n_samples])
    random_inter_distances = calculate_inter_dataset_distances(models, random_words)

    # Perform bootstrapping
    target_boot_means = bootstrap_mean(target_inter_distances, n_bootstraps)
    random_boot_means = bootstrap_mean(random_inter_distances, n_bootstraps)

    # Calculate confidence intervals
    target_ci = np.percentile(target_boot_means, [2.5, 97.5])
    random_ci = np.percentile(random_boot_means, [2.5, 97.5])

    # Print results
    print("Target words mean inter-dataset distance:")
    print(f"Mean: {np.mean(target_inter_distances)}")
    print(f"95% CI: {target_ci}")

    print("Random words mean inter-dataset distance:")
    print(f"Mean: {np.mean(random_inter_distances)}")
    print(f"95% CI: {random_ci}")

    # Plot the bootstrap distributions
    plt.figure(figsize=(10, 6))
    plt.hist(target_boot_means, bins=30, alpha=0.5, label='Target Words')
    plt.hist(random_boot_means, bins=30, alpha=0.5, label='Random Words')
    plt.xlabel('Mean Inter-Dataset Distance', fontsize=14)
    plt.ylabel('Frequency', fontsize=14)
    plt.legend(loc='upper right')
    plt.show()

if __name__ == "__main__":
    main()
