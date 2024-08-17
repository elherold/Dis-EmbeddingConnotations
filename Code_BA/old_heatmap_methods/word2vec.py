import os
import logging
import csv
import sys
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
                    if chunk_count % 10 == 0: # Print progress every 10 chunks
                        print(f"Processed {chunk_count} chunks from {file_path}")
                    yield chunk
                    chunk = []
            if chunk:
                yield chunk
            print("Finished reading the CSV file.")
    except (IOError, FileNotFoundError) as e:
        print(f"Error reading the CSV file {file_path}: {e}")

def read_and_process_file(file_path):
    """
    Reads and processes a file using read_processed_comments and returns all sentences.
    
    Args:
        file_path (str): The path to the file containing the processed comments.
    
    Returns:
        list: A list of sentences read from the file.
    """
    sentences = []
    for chunk in read_processed_comments(file_path):
        sentences.extend(chunk)
    return sentences

def word2vec_model(sentences, vector_size=300, window=10, min_count=15, workers=4, epochs=5, sg=1, negative=5, hs=0):
    """
    Trains a Word2Vec model on the provided sentences using skip-gram with negative sampling.

    Args:
        sentences (list of list of str): The training corpus, a list of sentences where each sentence is a list of words.
        vector_size (int): Dimensionality of the word vectors in the resulting Embedding Space. Defaults to 300.
        window (int): Maximum distance between the current and predicted word within a sentence. Defaults to 6.
        min_count (int): Ignores all words with total frequency lower than this. Defaults to 15.
        workers (int): Number of worker threads to train the model. Defaults to 4.
        epochs (int): Number of iterations (epochs) over the corpus. Defaults to 10.
        sg (int): Training algorithm: 1 for skip-gram; otherwise CBOW. Defaults to 1 (skip-gram).
        negative (int): Number of negative samples. Defaults to 5.

    Returns:
        Word2Vec: Trained Word2Vec model if training is successful.
        None: If there is an error during vocabulary building or model training.
    """
    # Set up logging to get information about the training process
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

    print(f"Training Word2Vec model with the following parameters:")
    print(f"Vector size: {vector_size}")
    print(f"Window size: {window}")
    print(f"Minimum count: {min_count}")
    print(f"Number of workers: {workers}")
    print(f"Number of epochs: {epochs}")
    print(f"Skip-gram: {sg == 1}")
    print(f"hierarchical softmax: {hs == 0}")
    print(f"Negative sampling: {negative}")

    try:
        # Initialize the model
        model = Word2Vec(vector_size=vector_size, window=window, min_count=min_count, workers=workers, sg=sg, hs=hs)
        
        # Build the vocabulary from the sentences
        print("Building vocabulary...")
        model.build_vocab(sentences)
        print(f"Vocabulary size: {len(model.wv.index_to_key)} words")
    except Exception as e:
        print(f"Error during vocabulary building: {e}")
        return None

    try:
        # Train the model
        print("Training the model...")
        model.train(sentences, total_examples=model.corpus_count, epochs=epochs)
        print("Training complete.")
    except Exception as e:
        print(f"Error during model training: {e}")
        return None

    # Print some information about the model
    print("Model information:")
    print(f"Vocabulary size: {len(model.wv.index_to_key)} words")
    print(f"Training loss: {model.get_latest_training_loss()}")

    return model

def main():
    """
    Main function to load processed comments from data folders and train Word2Vec models multiple times.

    This function:
    - Defines the data and model folders.
    - Loads processed comments from specified data folders.
    - Processes the comments according to set-belonging.
    - Trains Word2Vec models multiple times on each summarized set of processed comments,
      so we can later on calculate average and variance of each vector.
    - Saves the trained models to the specified model folder.

    Raises:
        Exception: If there is an error processing any dataset or training the model.
    """

    # Define the data and model folders
    data_folder_1 = "data/data_processed/right"
    data_folder_2 = "data/data_processed/left"
    model_folder = "models/new/"
    os.makedirs(model_folder, exist_ok=True)
    # specify number of times the same model is trained
    num_trainings = 5

    # Load the processed comments and train the Word2Vec models
    for data_folder in [data_folder_1]: # removed datafolder 2 for now
        datasets = [os.path.join(data_folder, dataset) for dataset in os.listdir(data_folder) if os.path.isfile(os.path.join(data_folder, dataset))]
        
        # Process files in parallel
        with ProcessPoolExecutor() as executor:
            results = executor.map(read_and_process_file, datasets)
            
            sentences = []
            chunk_count = 0
            for result in results:
                sentences.extend(result)
                chunk_count += 1
                if chunk_count % 10 == 0: # Print progress every 10 chunks
                    print(f"Processed {chunk_count} chunks in total")

        if sentences:
            print(f"Total number of sentences for training in {data_folder}: {len(sentences)}")
            # train the model num_trainings times
            for i in range(num_trainings):
                model = word2vec_model(sentences) # train model
                print(f"Training model {i+1}/{num_trainings}")
                if model:
                    model.save(os.path.join(model_folder, f"skipgram_{i+1}_{os.path.basename(data_folder)}.model"))
                    print(f"Model saved as skipgram_{i+1}_{os.path.basename(data_folder)}.model")
        else:
            print(f"No data found in {data_folder}")

if __name__ == "__main__":
    main()
