import json
import os
import numpy as np
import random
from scipy.spatial.distance import cosine
from scipy.stats import bootstrap
from load_models import load_models
import matplotlib.pyplot as plt


def load_macht_sprache_json(file_path):
    if not os.path.exists(file_path):
        print(f"File {file_path} not found.")
        return None
    
    with open(file_path, 'r', encoding='utf-8') as jsonfile:
        data = json.load(jsonfile)
    return data

def calculate_second_order_similarity_vectors(model_l, model_r, target_words, k=10):
    similarity_vectors = {word: [] for word in target_words}

    for word in target_words:
        if word in model_l.wv and word in model_r.wv:
            neighbors_l = model_l.wv.most_similar(word, topn=k)  # get the NNs of the word in the left model
            neighbors_r = model_r.wv.most_similar(word, topn=k)  # get the NNs of the word in the right model
            
            combined_neighbors = set([n[0] for n in neighbors_l] + [n[0] for n in neighbors_r])
            
            vector_l = []
            vector_r = []
            for neighbor in combined_neighbors:
                if neighbor in model_l.wv and neighbor in model_r.wv:
                    vector_l.append(model_l.wv.similarity(word, neighbor))
                    vector_r.append(model_r.wv.similarity(word, neighbor))

            similarity_vectors[word].append((vector_l, vector_r))

    return similarity_vectors

def bootstrap_mean_distances(targetwords, models, n_bootstraps=10):
    word_inter_distances = []

    # Calculate intra-dataset distances for the left models
    intra_distances_left = {word: [] for word in targetwords}
    for i in range(len(models['left'])):
        for j in range(i + 1, len(models['left'])):
            similarity_vectors = calculate_second_order_similarity_vectors(models['left'][i], models['left'][j], targetwords)
            for word, distances in calculate_mean_cosine_distances(similarity_vectors).items():
                intra_distances_left[word].append(distances)

    # Calculate intra-dataset distances for the right models
    intra_distances_right = {word: [] for word in targetwords}
    for i in range(len(models['right'])):
        for j in range(i + 1, len(models['right'])):
            similarity_vectors = calculate_second_order_similarity_vectors(models['right'][i], models['right'][j], targetwords)
            for word, distances in calculate_mean_cosine_distances(similarity_vectors).items():
                intra_distances_right[word].append(distances)

    # Calculate inter-dataset distances between left and right models
    inter_distances = {word: [] for word in targetwords}
    for i in range(len(models['left'])):
        for j in range(len(models['right'])):
            similarity_vectors = calculate_second_order_similarity_vectors(models['left'][i], models['right'][j], targetwords)
            for word, distances in calculate_mean_cosine_distances(similarity_vectors).items():
                inter_distances[word].append(distances)

    # Calculate mean intra-dataset distances
    mean_intra_distances = {word: (np.mean(intra_distances_left[word] + intra_distances_right[word]))
                            for word in targetwords if intra_distances_left[word] and intra_distances_right[word]}
    
    # Calculate mean inter-dataset distances
    mean_inter_distances = {word: np.mean(inter_distances[word]) for word in targetwords if inter_distances[word]}

    # Perform bootstrapping on intra and inter means
    intra_samples = np.array(list(mean_intra_distances.values()))  # Convert to numpy array for bootstrapping
    inter_samples = np.array(list(mean_inter_distances.values()))

    intra_bootstrap = bootstrap((intra_samples,), np.mean, n_resamples=n_bootstraps)  # Perform bootstrapping
    inter_bootstrap = bootstrap((inter_samples,), np.mean, n_resamples=n_bootstraps)

    return intra_bootstrap.confidence_interval, inter_bootstrap.confidence_interval, mean_intra_distances, mean_inter_distances

def calculate_mean_cosine_distances(similarity_vectors):
    mean_cosine_distances = {word: np.mean([cosine(vector_r, vector_r1) for vector_r, vector_r1 in vectors])
                             for word, vectors in similarity_vectors.items() if vectors}
    return mean_cosine_distances

def filter_target_words(targetwords, models):
    filtered_words = []
    for word in targetwords:
        if all(word in model.wv for model_set in models.values() for model in model_set):
            filtered_words.append(word)

    print(f"Number of target words present in all models: {len(filtered_words)}")
    return filtered_words

def find_similar_frequency_word(word, models, vocab_intersection):
    # Get the frequency of the target word in both models
    freq_left = models['left'][0].wv.get_vecattr(word, "count")
    freq_right = models['right'][0].wv.get_vecattr(word, "count")
    
    # Define a tolerance for frequency similarity
    tolerance = 0.1  # 10% tolerance
    
    similar_words = []
    for candidate_word in vocab_intersection:
        candidate_freq_left = models['left'][0].wv.get_vecattr(candidate_word, "count")
        candidate_freq_right = models['right'][0].wv.get_vecattr(candidate_word, "count")
        
        # Check if candidate word has similar frequency in both models
        if (abs(freq_left - candidate_freq_left) / freq_left <= tolerance) and (abs(freq_right - candidate_freq_right) / freq_right <= tolerance):
            similar_words.append(candidate_word)
    
    if similar_words:
        return random.choice(similar_words)
    else:
        return None

def plot_mean_distances_boxplot(intra_means, inter_means, intra_ci, inter_ci, title, ylabel):
    # Plot box plot bar-charts with confidence intervals
    plt.figure(figsize=(10, 6))
    
    # Data for box plots
    data = [list(intra_means.values()), list(inter_means.values())]
    labels = ['Intra-dataset', 'Inter-dataset']
    
    # Create box plots
    plt.boxplot(data, labels=labels, patch_artist=True, showmeans=True, showfliers=False)
    
    # Add mean and confidence intervals
    for i, (means, ci) in enumerate(zip([intra_means, inter_means], [intra_ci, inter_ci])):
        mean_value = np.mean(list(means.values()))
        plt.errorbar(i + 1, mean_value, yerr=[[mean_value - ci.low], [ci.high - mean_value]], fmt='o', color='black')

    # Labeling the axes
    plt.xlabel('Word Categories', fontsize=14)
    plt.ylabel(ylabel, fontsize=14)
    
    # Title of the plot
    plt.title(title, fontsize=16, fontweight='bold')
    
    # Adding grid for better readability
    plt.grid(True)
    
    # Display the plot
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

    # Perform bootstrapping for target words
    print("Performing bootstrapping to calculate confidence intervals for mean distances...")
    intra_ci, inter_ci, mean_intra_distances, mean_inter_distances = bootstrap_mean_distances(targetwords, models)

    # Sample 98 random words with similar frequency
    vocab_intersection = set.intersection(*(set(model.wv.index_to_key) for model_set in models.values() for model in model_set))
    random_sample_words = []
    for word in targetwords:
        similar_word = find_similar_frequency_word(word, models, vocab_intersection)
        if similar_word:
            random_sample_words.append(similar_word)
    
    # Perform bootstrapping for random sampled words
    print("Performing bootstrapping for frequency-matched sampled words...")
    _, inter_ci_random, _, mean_inter_distances_random = bootstrap_mean_distances(random_sample_words, models)

    # Plot boxplots for intra vs inter for target words
    title = 'Comparison of Mean Cosine Distances for Intra and Inter-Dataset Variations (Target Words)'
    ylabel = 'Mean Cosine Distance'
    plot_mean_distances_boxplot(mean_intra_distances, mean_inter_distances, intra_ci, inter_ci, title, ylabel)

    # Plot boxplots for inter vs inter (target words vs random words)
    plt.figure(figsize=(10, 6))
    
    # Data for box plots
    data = [list(mean_inter_distances.values()), list(mean_inter_distances_random.values())]
    labels = ['Target Words', 'Frequency-Matched Random Words']
    
    # Create box plots
    plt.boxplot(data, labels=labels, patch_artist=True, showmeans=True, showfliers=False)
    
    # Add mean and confidence intervals
    for i, (means, ci) in enumerate(zip([mean_inter_distances, mean_inter_distances_random], [inter_ci, inter_ci_random])):
        mean_value = np.mean(list(means.values()))
        plt.errorbar(i + 1, mean_value, yerr=[[mean_value - ci.low], [ci.high - mean_value]], fmt='o', color='black')

    # Labeling the axes
    plt.xlabel('Word Categories', fontsize=14)
    plt.ylabel('Mean Cosine Distance', fontsize=14)
    
    # Title of the plot
    plt.title('Comparison of Inter-Dataset Mean Cosine Distances (Target Words vs Frequency-Matched Random Words)', fontsize=16, fontweight='bold')
    
    # Adding grid for better readability
    plt.grid(True)
    
    # Display the plot
    plt.show()

if __name__ == "__main__":
    main()
