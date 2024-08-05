import os
import logging
import csv
from gensim.models import Word2Vec
from concurrent.futures import ProcessPoolExecutor

# Increase the CSV field size limit to handle large fields
csv.field_size_limit(10**6)

def read_processed_comments(file_path, chunk_size=10000):
    """
    Reads processed comments from a large CSV file in chunks.

    Args:
        file_path (str): The path to the file containing the processed comments.
        chunk_size (int): The number of lines to read at a time.

    Yields:
        list: A list of processed comments (10 words per line).
    """
    chunk_count = 0  # Initialize chunk counter
    try:
        with open(file_path, 'r', encoding='utf-8') as csvfile:
            reader = csv.reader(csvfile)
            chunk = []
            for line in reader:
                if len(line) == 10:
                    chunk.append(line)
                if len(chunk) >= chunk_size:
                    chunk_count += 1
                    if chunk_count % 10 == 0:  # Print progress every 10 chunks
                        print(f"Processed {chunk_count} chunks from {file_path}")
                    yield chunk
                    chunk = []
            if chunk:
                yield chunk
            print("Finished reading the CSV file.")
    except (IOError, FileNotFoundError) as e:
        print(f"Error reading the CSV file {file_path}: {e}")

def build_vocab_from_datasets(datasets, model):
    """
    Build the vocabulary for the Word2Vec model using the provided datasets.

    Args:
        datasets (list): List of file paths to the datasets.
        model (Word2Vec): The Word2Vec model to build the vocabulary for.
    """
    first_chunk = True

    for dataset in datasets:
        for chunk in read_processed_comments(dataset):
            if first_chunk:
                model.build_vocab(chunk)
                first_chunk = False
            else:
                model.build_vocab(chunk, update=True)
    
def train_word2vec_model_iteratively(datasets, model_folder, num_iterations, leaning, vector_size=300, window=10, min_count=15, workers=4, epochs=5, sg=1, negative=5, hs=0):
    """
    Trains a Word2Vec model iteratively on the provided sentences.

    Args:
        datasets (list): List of file paths to the datasets.
        model_folder (str): Path to the folder where models will be saved.
        num_iterations (int): Number of iterations for training.
        leaning (str): Label indicating the political leaning of the data.
        vector_size (int): Dimensionality of the word vectors in the resulting Embedding Space. Defaults to 300.
        window (int): Maximum distance between the current and predicted word within a sentence. Defaults to 10.
        min_count (int): Ignores all words with total frequency lower than this. Defaults to 15.
        workers (int): Number of worker threads to train the model. Defaults to 4.
        epochs (int): Number of iterations (epochs) over the corpus. Defaults to 5.
        sg (int): Training algorithm: 1 for skip-gram; otherwise CBOW. Defaults to 1 (skip-gram).
        negative (int): Number of negative samples. Defaults to 5.
        hs (int): If 1, hierarchical softmax will be used for model training. If 0, negative sampling will be used. Defaults to 0 (negative sampling).
    """
    # Initialize the model
    model = Word2Vec(vector_size=vector_size, window=window, min_count=min_count, workers=workers, sg=sg, negative=negative, hs=hs)

    # Build the vocabulary once
    build_vocab_from_datasets(datasets, model)

    total_sentences = sum(len(chunk) for dataset in datasets for chunk in read_processed_comments(dataset))
    print(f"Total number of sentences processed: {total_sentences}")

    # Train the model multiple times
    for i in range(num_iterations):  
        print(f"Iteration {i+1} of {num_iterations}")
        for dataset in datasets:
            for chunk in read_processed_comments(dataset):
                model.train(chunk, total_examples=len(chunk), epochs=epochs)

        model.save(os.path.join(model_folder, f"SG_{i}_{leaning}.model"))
        print(f"Model {i} of {leaning} saved")

def main():
    """
    Main function to load processed comments from data folders and train Word2Vec models iteratively.

    This function:
    - Defines the data and model folders.
    - Loads processed comments from specified data folders.
    - Processes the comments according to their political leaning.
    - Trains a Word2Vec model iteratively on the processed comments.
    - Saves the trained model to the specified model folder.

    Raises:
        Exception: If there is an error processing any dataset or training the model.
    """
    # Define the data and model folders
    leanings = ["left", "right"]
    data_folder = "data/data_processed/"
    model_folder = "models/new/"
    num_iterations = 5
    os.makedirs(model_folder, exist_ok=True)

    # Load the processed comments and train the Word2Vec models
    for leaning in leanings:
        datasets = [os.path.join(data_folder + leaning, dataset) for dataset in os.listdir(data_folder + leaning) if os.path.isfile(os.path.join(data_folder + leaning, dataset))]
        # Train the model iteratively
        train_word2vec_model_iteratively(datasets, model_folder, num_iterations, leaning)

if __name__ == "__main__":
    main()
