import json
import os
import numpy as np
import pandas as pd
import random
from scipy.spatial.distance import cosine
from scipy.stats import bootstrap
from load_models import load_models
from reliability_ES import calculate_second_order_similarity_vectors, calculate_mean_cosine_distances, filter_target_words
import matplotlib.pyplot as plt

def load_csv_words(file_path):
    """
    Loads words from a CSV file.
    """
    if not os.path.exists(file_path):
        print(f"File {file_path} not found.")
        return set()

    data = pd.read_csv(file_path, header=None)
    return set(data[0].tolist())

def load_macht_sprache_json(file_path):
    """
    Loads the cleaned MachtSprache JSON data from a specified file path.
    """
    if not os.path.exists(file_path):
        print(f"File {file_path} not found.")
        return []

    with open(file_path, 'r', encoding='utf-8') as jsonfile:
        data = json.load(jsonfile)
    return data

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

    for i in range(n_bootstraps):
        print(f"Bootstrapping {i+1}/{n_bootstraps}")
        sample = np.random.choice(data, size=len(data), replace=True)
        boot_means.append(np.mean(sample))

    return boot_means

def main():
    left_filepath = "data/data_processed/left/left-leaning.csv"
    right_filepath = "data/data_processed/right/right-leaning.csv"
    macht_sprache_filepath = "data/data_helper/cleaned_words_MachtSprache.json"
    n_samples = 98
    n_bootstraps = 1000
    
    # Load word2vec models
    models = load_models('left', 'right', spellchecker=False)
    
    # Load words from CSV files
    left_words = load_csv_words(left_filepath)
    right_words = load_csv_words(right_filepath)
    
    # Load MachtSprache target words
    macht_sprache_words = load_macht_sprache_json(macht_sprache_filepath)
    
    # Find intersection of words between the two datasets
    common_words = left_words.intersection(right_words)
    
    if len(common_words) < n_samples:
        print(f"Not enough common words found. Found {len(common_words)} common words.")
        return
    
    # Randomly sample 98 common words
    sampled_words = random.sample(list(common_words), n_samples)

    # Calculate inter-dataset distances for sampled words
    print("Calculating inter-dataset distances for sampled words...")
    inter_distances_sampled = calculate_inter_dataset_distances(models, sampled_words)

    # Calculate inter-dataset distances for MachtSprache words
    print("Calculating inter-dataset distances for MachtSprache words...")
    inter_distances_macht_sprache = calculate_inter_dataset_distances(models, macht_sprache_words)

    # Perform bootstrapping for sampled words
    print("Calculating bootstrap means for sampled words...")
    boot_means_sampled = bootstrap_mean(inter_distances_sampled, n_bootstraps)
    ci_sampled = np.percentile(boot_means_sampled, [2.5, 97.5])

    # Perform bootstrapping for MachtSprache words
    print("Calculating bootstrap means for MachtSprache words...")
    boot_means_macht_sprache = bootstrap_mean(inter_distances_macht_sprache, n_bootstraps)
    ci_macht_sprache = np.percentile(boot_means_macht_sprache, [2.5, 97.5])

    # Print results
    print("Sampled words mean inter-dataset distance:")
    print(f"Mean: {np.mean(inter_distances_sampled)}")
    print(f"95% CI: {ci_sampled}")

    print("Politically Loaded words mean inter-dataset distance:")
    print(f"Mean: {np.mean(inter_distances_macht_sprache)}")
    print(f"95% CI: {ci_macht_sprache}")

    # Plot box plot bar-charts with confidence intervals
    plt.figure(figsize=(10, 6))
    
    # Data for box plots
    data = [boot_means_sampled, boot_means_macht_sprache]
    labels = ['Randomly Sampled Words', 'Politically Loaded Words']
    
    # Create box plots
    plt.boxplot(data, labels=labels, patch_artist=True, showmeans=True, showfliers=False)
    
    # Add mean and confidence intervals
    plt.errorbar(1, np.mean(boot_means_sampled), yerr=[[np.mean(boot_means_sampled) - ci_sampled[0]], [ci_sampled[1] - np.mean(boot_means_sampled)]], fmt='o', color='black')
    plt.errorbar(2, np.mean(boot_means_macht_sprache), yerr=[[np.mean(boot_means_macht_sprache) - ci_macht_sprache[0]], [ci_macht_sprache[1] - np.mean(boot_means_macht_sprache)]], fmt='o', color='black')

    plt.xlabel('Word Categories', fontsize=14)
    plt.ylabel('Mean Inter-Dataset Distance', fontsize=14)
    plt.title('Inter-Dataset Distance Comparison', fontsize=16, fontweight='bold')
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    main()