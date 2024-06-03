import read_datasets as rd
import pandas as pd
import contractions
import re
import spacy
from symspellpy import SymSpell, Verbosity
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
import os
import logging
import pickle

# Configure logging
logging.basicConfig(format="%(levelname)s - %(asctime)s: %(message)s", datefmt='%H:%M:%S', level=logging.INFO)

# Load SpaCy model
logging.info("Loading SpaCy model...")
nlp = spacy.load('en_core_web_sm')


# Initialize SymSpell
def initialize_symspell():
    """
    This is setting up and initializing the symspell  spell checker,
    by loading the necessary dictionary files that SymSpell uses.

    Returns: the initialized SymSpell object.

    """
    logging.info("Initializing SymSpell...")
    sym_spell = SymSpell(max_dictionary_edit_distance=2, prefix_length=7) # words which are at most 2 char changes away from known word can be corrected

    # Determine the base directory dynamically
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Construct the relative paths
    dictionary_path = os.path.join(base_dir, "../data/data_helper/frequency_dictionary_en_82_765.txt")
    bigram_path = os.path.join(base_dir, "../data/data_helper/frequency_bigramdictionary_en_243_342.txt")

    if not sym_spell.load_dictionary(dictionary_path, term_index=0, count_index=1):
        logging.error(f"Dictionary file {dictionary_path} not found")
        return None
    if not sym_spell.load_bigram_dictionary(bigram_path, term_index=0, count_index=2):
        logging.error(f"Bigram dictionary file {bigram_path} not found")
        return None
    logging.info("SymSpell initialized successfully.")
    return sym_spell

def correct_spelling(text, sym_spell, cache, batch_size=100):
    """
    Corrects misspelled words in the text leveraging SymSpell.
    """
def correct_spelling(text, sym_spell, cache, batch_size=100):
    corrected_text = []
    words = re.findall(r'\w+|[^\w\s]', text)
    for i in range(0, len(words), batch_size):
        batch = words[i:i + batch_size]
        for token in batch:
            if token.isalpha():
                corrected_word = cache.get(token)
                if corrected_word is None:
                    suggestions = sym_spell.lookup(token, Verbosity.CLOSEST, max_edit_distance=2)
                    if suggestions:
                        corrected_word = suggestions[0].term
                        cache[token] = corrected_word
                    else:
                        corrected_word = token
                        cache[token] = corrected_word
                corrected_text.append(corrected_word)
            else:
                corrected_text.append(token)
    return ' '.join(corrected_text)

def preprocessing(text, use_spellchecker, sym_spell, cache):
    """
    Preprocesses the text data.
    
    Parameters:
    text (str): The text data to preprocess.
    
    Returns:
    str: The preprocessed text data.
    """
    text = text.lower()  # Convert text to lowercase
    text = re.sub(r'\s+', ' ', text)  # Replace multiple whitespaces with a single whitespace
    text = re.sub(r'http\S+|www\S+|\S+\.\S+', '', text)  # Better handling of URLs
    text = re.sub(r'@\w+|#\w+', '', text)  # Handle mentions and hashtags
    text = contractions.fix(text)  # Expand contractions    
    text = re.sub(r'[^a-zA-Z\s\.\?\!]', '', text) # Remove non-text characters but keep sentence-ending punctuation
    text = re.sub(r'[\*^_~]', '', text)  # Remove Markdown Formatting
    text = text.strip()  # Remove leading and trailing whitespaces
    
    if use_spellchecker:
        text = correct_spelling(text, sym_spell, cache)  # Correct the spelling
    
    doc = nlp(text)  # Process text with SpaCy
    
    sentences = []
    for sent in doc.sents:
        lemmas = [token.lemma_ for token in sent if not token.is_punct and not token.is_space]  # Lemmatize and filter tokens
        sentences.append(lemmas)

    return sentences

def process_batch(batch, use_spellchecker, sym_spell, cache):

    processed_sentences = []
    for comment in batch:
        processed_sentences.extend(preprocessing(comment, use_spellchecker, sym_spell, cache))
    return processed_sentences

def process_large_dataset(df, use_spellchecker, batch_size=1000, num_workers=None):
    comments = df['body'].tolist()
    processed_comments = []
    sym_spell = initialize_symspell()
    cache = {} # initializing cache

    if num_workers is None:
        num_workers = os.cpu_count() // 2  # Use half of the logical processors

    logging.info(f"Processing dataset with {num_workers} workers and batch size of {batch_size}, number of total batches to be processed are approximately: {len(df)/batch_size}...")

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = []
        for i in range(0, len(comments), batch_size):
            batch = comments[i:i + batch_size]
            logging.info(f"Submitting batch {i // batch_size + 1} for processing...")
            futures.append(executor.submit(process_batch, batch, use_spellchecker, sym_spell, cache))

        for future in as_completed(futures):
            processed_comments.extend(future.result())
            logging.info(f"Batch {futures.index(future) + 1} completed.")

    logging.info("Dataset processing complete.")
    return processed_comments

def main():
    source_folder = "data/data_raw"
    target_folder = "data/data_processed"
    os.makedirs(target_folder, exist_ok=True) # Create the target folder if it doesn't exist
    
    for dataset in os.listdir(source_folder):
        try:
            file_path = os.path.join(source_folder, dataset)
            if os.path.isfile(file_path):
                # Create DataFrame for the second file
                logging.info(f"Reading dataset from {file_path}...")
                df2 = rd.create_dataframe(file_path)
                if df2 is not None:  # Check if the DataFrame was created successfully ("not None" to avoid ambiguous truth value error of dataframe)
                    # Process the dataset
                    logging.info("Starting preprocessing of comments with spell checker...")
                    # check if the dataset was already processed with a spellchecker (if the file exists)
                    if os.path.exists(f'spellchecker_processed_{dataset}.pkl'):
                        logging.info(f"File 'spellchecker_processed_{dataset}.pkl' already exists. Skipping processing...")
                        continue
                    processed_comments_spellchecker = process_large_dataset(df2, batch_size=1000, use_spellchecker=True)
                    if processed_comments_spellchecker:
                        # check if the dataset was already processed without a spellchecker (if the file exists)
                        if os.path.exists(f'processed_{dataset}.pkl'):
                            logging.info(f"File 'processed_{dataset}.pkl' already exists. Skipping processing...")
                            continue
                        processed_comments = process_large_dataset(df2, batch_size=1000, use_spellchecker=False)
                        if processed_comments:
                            print(f"Shape of processed_comments with spellchecker: {len(processed_comments_spellchecker)} rows and {len(processed_comments_spellchecker[0])} columns")
                            print(f"Shape of processed_comments without spellchecker: {len(processed_comments)} rows and {len(processed_comments[0])} columns")
                            print(f"First processed comments with spellchecker: {processed_comments_spellchecker[:5]}")
                            print(f"First processed comments without spellchecker: {processed_comments[:5]}")

                            # Save the list of lists to a pickle file
                            with open(f'spellchecker_processed_{dataset}.pkl', 'wb') as file: # writing mode + handling binary data
                                pickle.dump(processed_comments_spellchecker, file) # version with spellchecker

                            with open(f'processed_{dataset}.pkl', 'wb') as file: # versoion without spellchecker
                                pickle.dump(processed_comments, file)

                            logging.info("Preprocessing complete. Processed comments saved to 'processed_comments.csv'.")

        except Exception as e:
            print(f"Error processing the dataset {dataset}: {e}")
            continue

if __name__ == "__main__":
    main()