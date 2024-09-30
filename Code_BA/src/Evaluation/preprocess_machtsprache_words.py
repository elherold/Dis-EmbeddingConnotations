import json
import os
import sys
# Add the parent directory to the system path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from collections import defaultdict
from reliability_ES import (
    calculate_second_order_similarity_vectors,
    calculate_mean_cosine_distances
)

from load_models import load_models

def clean_macht_sprache(input_file_path, output_file_path, models):
    """
    Cleans the Macht.Sprache JSON data by extracting and processing words into a list of target words 
    from "lemma" and "relatedterms" fields in case "lemma_lang==en" to make sure only English words are included. 
    It converts the extracted words to lowercase, removes spaces, hyphens, and parentheses, and filters out words containing asterisks.
    Only words present in both embedding spaces are included, and duplicates are removed.

    Parameters: 
                input_file_path (str): the path to the input macht.sprache json file
                output_file_path (str): the path to save the cleaned words
                models (dict): Dictionary containing the word2vec models

    Returns:
                None
    """
    # Load the macht.sprache json file
    try:
        with open(input_file_path, 'r', encoding='utf-8') as jsonfile:
            data = json.load(jsonfile)
    except FileNotFoundError:
        print(f"File {input_file_path} not found.")
        return None
    
    words_set = set()
    word_freqs = defaultdict(dict)

    # Extract and process the words of interest
    for entry in data:
        # only include english words
        if entry['lemma_lang'] == 'en':
            # process the fields lemma and relatedterms to extract the relevant words
            lemma = entry['lemma'].lower().replace(" ", "").replace("-", "").replace("(", "").replace(")", "")
            if '*' not in lemma:
                words_set.add(lemma)
            related_terms = [term.lower().replace(" ", "").replace("-", "").replace("(", "").replace(")", "")
                             for term in entry['relatedterms'] if '*' not in term]
            words_set.update(related_terms)
    
    words = list(words_set)
    
    # Filter words to include only those present in all models
    filtered_words = [word for word in words if all(word in model.wv for model_set in models.values() for model in model_set)]
    
    # Calculate frequencies
    for word in filtered_words:
        for set_name, model_set in models.items():
            word_freqs[word][set_name] = int(sum(model.wv.get_vecattr(word, 'count') for model in model_set if word in model.wv))
    
    # remove duplicates
    with open(output_file_path, 'w', encoding='utf-8') as jsonfile:
        json.dump(filtered_words, jsonfile, indent=4)
    
    # save word frequencies
    with open(output_file_path.replace("cleaned_words_MachtSprache.json", "word_frequencies.json"), 'w', encoding='utf-8') as jsonfile:
        json.dump(word_freqs, jsonfile, indent=4)
    
    print(f"Cleaned words saved to {output_file_path}")
    print(f"Word frequencies saved to {output_file_path.replace('cleaned_words_MachtSprache.json', 'word_frequencies.json')}")

def calculate_and_save_inter_distances(models, target_words, output_file):
    """
    Calculate the inter-dataset differences for all words and save them to a file.

    Parameters:
                models (dict): Dictionary containing the Word2Vec models.
                target_words (list): List of target words.
                output_file (str): The file path to save the inter-dataset differences.

    Returns:
                None
    """
    if not os.path.exists(output_file):
        inter_distances = []

        # Calculate inter-dataset differences via the target vectors' second order similarity vectors
        for word in target_words:
            similarity_vectors = calculate_second_order_similarity_vectors(models['left'][0], models['right'][0], [word])
            mean_inter_distances = calculate_mean_cosine_distances(similarity_vectors)
            inter_distances.append((word, mean_inter_distances[word]))

        # Save inter-dataset differences to a file
        with open(output_file, 'w', encoding='utf-8') as jsonfile:
            json.dump(inter_distances, jsonfile, indent=4)
        print(f"Inter-dataset distances saved to {output_file}")
    else:
        print(f"File {output_file} already exists. Skipping calculation.")

def main():
    """
    Clean the macht.sprache json file, 
    load the cleaned words, and calculate and save inter-dataset differences
    """
    input_file_path = "data/data_helper/macht.sprache_words.json"
    cleaned_words_file = "data/data_helper/cleaned_words_MachtSprache.json"
    inter_distances_file = "data/data_helper/inter_distances.json"

    # Load word2vec models
    models = load_models('left', 'right')

    # Clean macht.sprache json file
    clean_macht_sprache(input_file_path, cleaned_words_file, models)

    # Load cleaned macht.sprache json file
    with open(cleaned_words_file, 'r', encoding='utf-8') as jsonfile:
        target_words = json.load(jsonfile)

    # Calculate and save inter-dataset differences
    calculate_and_save_inter_distances(models, target_words, inter_distances_file)

if __name__ == "__main__":
    main()
