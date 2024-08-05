import json
import os
from collections import defaultdict
from gensim.models import Word2Vec
from load_models import load_models

def get_word_frequencies(models):
    """
    Get the word frequencies from gensim models.

    Args:
        models (dict): Dictionary containing word2vec models.

    Returns:
        word_freqs (dict): Dictionary containing word frequencies.
    """
    word_freqs = defaultdict(int)
    
    for model_set in models.values():
        for model in model_set:
            for word, vocab_obj in model.wv.key_to_index.items():
                word_freqs[word] += model.wv.get_vecattr(word, 'count')
                
    return word_freqs

def save_top_words_by_frequency(word_freqs, n=120, output_file='top_words_by_frequency.json'):
    """
    Save the top N words by frequency to a JSON file.

    Args:
        word_freqs (dict): Dictionary containing word frequencies.
        n (int): Number of top words to save.
        output_file (str): Path to the output JSON file.

    Returns:
        None
    """
    sorted_word_freqs = sorted(word_freqs.items(), key=lambda item: item[1], reverse=True)
    top_words = dict(sorted_word_freqs[:n])
    
    with open(output_file, 'w', encoding='utf-8') as jsonfile:
        json.dump(top_words, jsonfile, indent=4)
        
    print(f"Top {n} words by frequency saved to {output_file}")

def main():
    # Load word2vec models
    models = load_models('left', 'right', spellchecker=False)

    # Get word frequencies from models
    word_freqs = get_word_frequencies(models)

    # Save the top 120 words by frequency to a JSON file
    save_top_words_by_frequency(word_freqs, n=120, output_file='top_words_by_frequency.json')

if __name__ == "__main__":
    main()
