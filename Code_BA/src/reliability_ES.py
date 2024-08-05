import json
import os
import numpy as np
import random
from scipy.spatial.distance import cosine
from scipy.stats import ttest_rel, bootstrap
from load_models import load_models
import matplotlib.pyplot as plt


def load_macht_sprache_json(file_path):
    """
    Loads the cleaned acht.sprache JSON data from a specified file path. 
    If the file is not found, it calls the clean_macht_sprache function to clean the data on the original Macht.Sprache file.

    Args:
        file_path (str): the path to the cleaned macht.sprache json file
        input_file_path (str): the path to the input macht.sprache json file

    Returns:
        data (list): a list of target words extracted from the cleaned macht.sprache json file
    """
    if not os.path.exists(file_path):
        print(f"File {file_path} not found.")
        return None
    
    with open(file_path, 'r', encoding='utf-8') as jsonfile:
        data = json.load(jsonfile)
    return data

def get_nearest_neighbors(model, word, k=10):
    """
    Get the k nearest neighbors of a word in a word2vec model.

    Args:
        model (Word2Vec): a word2vec model
        word (str): the target word
        k (int): the number of neighbors to retrieve

    Returns:
        neighbors (list): a list of tuples containing the nearest neighbors and their cosine similarity scores
    """
    if word in model.wv:
        return model.wv.most_similar(word, topn=k)
    return []

def calculate_second_order_similarity_vectors(model_r, model_r1, target_words, k=10):
    """
    Calculates second-order similarity vectors for pairs of models.

    Args:
        model_r (Word2Vec): The first word2vec model.
        model_r1 (Word2Vec): The second word2vec model.
        target_words (list): The list of target words.
        k (int): The number of nearest neighbors to consider.

    Returns:
        similarity_vectors (dict): A dictionary containing second-order similarity vectors for each target word.
    """
    similarity_vectors = {word: [] for word in target_words}

    for word in target_words:
        if word in model_r.wv and word in model_r1.wv:
            neighbors_r = get_nearest_neighbors(model_r, word, k)
            neighbors_r1 = get_nearest_neighbors(model_r1, word, k)
            
            combined_neighbors = set([n[0] for n in neighbors_r] + [n[0] for n in neighbors_r1]) # extract union of NN sets from both models
            
            vector_r = []
            vector_r1 = []
            for neighbor in combined_neighbors:
                if neighbor in model_r.wv and neighbor in model_r1.wv:
                    vector_r.append(model_r.wv.similarity(word, neighbor))
                    vector_r1.append(model_r1.wv.similarity(word, neighbor))
                # Skip the neighbor if it is not present in both models

            similarity_vectors[word].append((vector_r, vector_r1))

    return similarity_vectors


def bootstrap_t_tests(targetwords, models, n_samples=20, n_bootstraps=20):
    """
    Perform bootstrapping to calculate confidence intervals for t-statistics.

    Args:
        targetwords (list): the list of target words
        models (dict): dictionary containing the word2vec models
        n_samples (int): the number of samples for each bootstrap
        n_bootstraps (int): the number of bootstrap iterations

    Returns:
        t_stats (list): list of t-statistics from each bootstrap iteration
        p_values (list): list of p-values from each bootstrap iteration
    """
    t_stats = []
    p_values = []
    word_inter_distances = []

    for _ in range(n_bootstraps):
        targetwords_sample = random.sample(targetwords, n_samples)
        print(f"Bootstrapping iteration: {len(t_stats) + 1}")
        print(f"Sampled target words: {targetwords_sample}")

        # Calculate intra-dataset distances for the left models
        intra_distances_left = {word: [] for word in targetwords_sample}
        print("Calculating intra-dataset distances for the left models...")
        for i in range(len(models['left'])):
            for j in range(i + 1, len(models['left'])):
                similarity_vectors = calculate_second_order_similarity_vectors(models['left'][i], models['left'][j], targetwords_sample)
                for word, distances in calculate_mean_cosine_distances(similarity_vectors).items():
                    intra_distances_left[word].append(distances)

        # Calculate intra-dataset distances for the right models
        print("Calculating intra-dataset distances for the right models...")
        intra_distances_right = {word: [] for word in targetwords_sample}
        for i in range(len(models['right'])):
            for j in range(i + 1, len(models['right'])):
                similarity_vectors = calculate_second_order_similarity_vectors(models['right'][i], models['right'][j], targetwords_sample)
                for word, distances in calculate_mean_cosine_distances(similarity_vectors).items():
                    intra_distances_right[word].append(distances)

        # Calculate inter-dataset distances between left and right models
        print("Calculating inter-dataset distances between left and right models...")
        inter_distances = {word: [] for word in targetwords_sample}
        for i in range(len(models['left'])):
            for j in range(len(models['right'])):
                similarity_vectors = calculate_second_order_similarity_vectors(models['left'][i], models['right'][j], targetwords_sample)
                for word, distances in calculate_mean_cosine_distances(similarity_vectors).items():
                    inter_distances[word].append(distances)

        # Calculate mean intra-dataset distances
        print("Calculating mean intra-dataset distances...")
        mean_intra_distances = {word: (np.mean(intra_distances_left[word] + intra_distances_right[word]))
                                for word in targetwords_sample if intra_distances_left[word] and intra_distances_right[word]}
        
        # Calculate mean inter-dataset distances
        print("Calculating mean inter-dataset distances...")
        mean_inter_distances = {word: np.mean(inter_distances[word]) for word in targetwords_sample if inter_distances[word]}

        # Collect mean inter-dataset distances for each word
        for word, inter_distance in mean_inter_distances.items():
            word_inter_distances.append((word, inter_distance))

        # Prepare paired data for t-test
        print("Preparing paired data for t-test...")
        paired_intra_distances = [mean_intra_distances[word] for word in targetwords_sample if word in mean_inter_distances]
        paired_inter_distances = [mean_inter_distances[word] for word in targetwords_sample if word in mean_inter_distances]

        # Perform paired samples t-test
        print("Performing paired samples t-test...")
        if len(paired_intra_distances) == len(paired_inter_distances) and len(paired_intra_distances) > 1:
            t_stat, p_value = ttest_rel(paired_intra_distances, paired_inter_distances)
            t_stats.append(t_stat)
            p_values.append(p_value)

    return t_stats, p_values, mean_intra_distances, mean_inter_distances

def calculate_confidence_interval(data, ci=95):
    """
    Calculate confidence interval for the given data.

    Args:
        data (list): list of data points
        ci (int): confidence interval percentage

    Returns:
        confidence_interval (tuple): the lower and upper bounds of the confidence interval
    """
    lower_bound = np.percentile(data, (100 - ci) / 2)
    upper_bound = np.percentile(data, 100 - (100 - ci) / 2)
    return lower_bound, upper_bound

def calculate_mean_cosine_distances(similarity_vectors):
    """
    Calculates the mean cosine distances from second-order similarity vectors.
    """
    mean_cosine_distances = {word: np.mean([cosine(vector_r, vector_r1) for vector_r, vector_r1 in vectors])
                             for word, vectors in similarity_vectors.items() if vectors}
    return mean_cosine_distances

def filter_target_words(targetwords, models):
    """
    Filters the target words to include only those present in all models.

    Args:
        targetwords (list): the original list of target words
        models (dict): dictionary containing the word2vec models

    Returns:
        filtered_words (list): list of words present in all models
    """
    filtered_words = []
    for word in targetwords:
        if all(word in model.wv for model_set in models.values() for model in model_set):
            filtered_words.append(word)

    print(f"Number of target words present in all models: {len(filtered_words)}")
    return filtered_words


def calculate_statistics(data):
    """
    Calculate descriptive statistics for the given data.

    Args:
        data (list): list of data points

    Returns:
        stats (dict): dictionary containing mean, median, standard deviation, and confidence interval
    """
    mean_val = np.mean(data)
    median_val = np.median(data)
    std_dev = np.std(data)
    ci = calculate_confidence_interval(data)
    
    stats = {
        "mean": mean_val,
        "median": median_val,
        "std_dev": std_dev,
        "confidence_interval": ci
    }
    
    return stats

def plot_mean_distances_with_error(intra_means, inter_means, title, ylabel):
    """
    Plot a bar plot with error bars for intra and inter-dataset means.

    Args:
        intra_means (list): list of intra-dataset mean distances
        inter_means (list): list of inter-dataset mean distances
        title (str): title of the plot
        ylabel (str): label for the y-axis

    Returns:
        None
    """
    intra_mean = np.mean(list(intra_means.values()))
    inter_mean = np.mean(list(inter_means.values()))
    intra_std = np.std(list(intra_means.values()))
    inter_std = np.std(list(inter_means.values()))
    intra_ci = calculate_confidence_interval(list(intra_means.values()))
    inter_ci = calculate_confidence_interval(list(inter_means.values()))
    
    labels = ['Intra-dataset', 'Inter-dataset']
    means = [intra_mean, inter_mean]
    std_devs = [intra_std, inter_std]
    ci_lower = [intra_mean - intra_ci[0], inter_mean - inter_ci[0]]
    ci_upper = [intra_ci[1] - intra_mean, inter_ci[1] - inter_mean]

    plt.figure(figsize=(10, 6))
    plt.bar(labels, means, yerr=std_devs, capsize=5, color='lightgrey')
    plt.errorbar(labels, means, yerr=[ci_lower, ci_upper], fmt='o', color='black')
    plt.title(title, fontweight='bold')
    plt.ylabel(ylabel, fontweight='bold')
    plt.xticks(fontweight='bold')
    #plt.yticks(fontweight='bold')
    plt.grid(True)
    plt.show()


def main():
    filepath = "data/data_helper/cleaned_words_MachtSprache.json"
    
    # Load word2vec models
    models = load_models('left', 'right', spellchecker=False)
    
    # Load macht.sprache json file
    targetwords = load_macht_sprache_json(filepath)

    if not targetwords:
        print("No target words found.")
        return

    # Filter target words to those present in all models
    targetwords = filter_target_words(targetwords, models)
    
    if len(targetwords) < 20:
        print("Not enough target words present in all models.")
        return

    # Perform bootstrapping
    print("Performing bootstrapping to calculate confidence intervals for t-statistics...")
    t_stats, p_values, mean_intra_distances, mean_inter_distances = bootstrap_t_tests(targetwords, models)

    # Calculate statistics for t-statistics and p-values
    print("Calculating statistics for t-statistics and p-values...")
    t_stat_stats = calculate_statistics(t_stats)
    p_value_stats = calculate_statistics(p_values)

    print("T-statistics statistics:")
    for key, value in t_stat_stats.items():
        print(f"{key}: {value}")

    print("P-values statistics:")
    for key, value in p_value_stats.items():
        print(f"{key}: {value}")

    # Plot histograms for t-statistics and p-values
    title = 'Comparison of Mean Cosine Distances for Intra and Inter-Dataset Variations'
    ylabel = 'Mean Cosine Distance'
    plot_mean_distances_with_error(mean_intra_distances, mean_inter_distances, title, ylabel)

    # Output the median t-statistic and p-value for reference
    median_t_stat = t_stat_stats['median']
    median_p_value = p_value_stats['median']
    print(f"Median T-statistic: {median_t_stat}")
    print(f"Median P-value: {median_p_value}")

    if median_p_value < 0.05:
        print("The inter-dataset shifts are significantly higher than the intra-dataset variations.")
    else:
        print("No significant difference between inter-dataset and intra-dataset variations.")



if __name__ == "__main__":
    main()

