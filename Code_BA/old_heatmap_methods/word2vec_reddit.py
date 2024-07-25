from gensim.models import Word2Vec
import os
import logging
import pickle

def read_processed_comments(file_path):
    """
    Reads processed comments from a pickle file.

    Args:
        file_path (str): The path to the file containing the processed comments.

    Returns:
        list: A list of processed comments if the file is read successfully and the data is in the expected format.
              Returns an empty list if the file cannot be read or if the data is not in the expected format.

    Raises:
        IOError: If an I/O error occurs.
        FileNotFoundError: If the file does not exist.
        pickle.UnpicklingError: If there is an error unpickling the file.
    """
    try:
        with open(file_path, 'rb') as file:
            processed_comments = pickle.load(file)
        # Ensure the data is in the expected format
        if isinstance(processed_comments, list) and all(isinstance(sent, list) for sent in processed_comments):
            print(f"Great, the data seems to be in the expected format in {file_path}")
            return processed_comments
        else:
            print(f"Unexpected data format in {file_path}")
            return []
    except (IOError, FileNotFoundError, pickle.UnpicklingError) as e:
        print(f"Error reading the pickle file {file_path}: {e}")
        return []

def word2vec_model(sentences, vector_size=300, window=6, min_count=15, workers=4, epochs=5, sg=1, negative=5):
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
    print(f"Negative samples: {negative}")
    print(f"Number of sentences: {len(sentences)}")

    try:
        # Initialize the model
        model = Word2Vec(vector_size=vector_size, window=window, min_count=min_count, workers=workers, sg=sg, negative=negative)
        
        # Build the vocabulary
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
    - processes the comments according to set-belonging
    - Separates data processed with and without a spellchecker.
    - Trains Word2Vec models multiple times on each summarized set of processed comments, 
        so we can later on calculate average and variance of each vector.
    - Saves the trained models to the specified model folder.

    Raises:
        Exception: If there is an error processing any dataset or training the model.
    """

    # Define the data and model folders
    data_folder_1 = "data/data_processed/feminism"
    data_folder_2 = "data/data_processed/antifeminism"
    model_folder = "models/new/"
    os.makedirs(model_folder, exist_ok=True)
    # specify number of times the same model is trained
    num_trainings = 10

    # Load the processed comments and train the Word2Vec models
    for data_folder in [data_folder_1, data_folder_2]:
        sentences = []
        sentences_spellchecker = []
        for dataset in os.listdir(data_folder): # summarizing data of the same set
            if dataset.startswith("spellchecker"): # separate the spellchecker data from the rest
                try:
                    file_path = os.path.join(data_folder, dataset)
                    if os.path.isfile(file_path):
                        sentences_spellchecker.extend(read_processed_comments(file_path)) # load processed comments
                except Exception as e:
                    print(f"Error processing the dataset {dataset}: {e}")
                    continue
            else:
                try:
                    file_path = os.path.join(data_folder, dataset)
                    if os.path.isfile(file_path):
                        sentences.extend(read_processed_comments(file_path)) # load processed comments
                except Exception as e:
                    print(f"Error processing the dataset {dataset}: {e}")
                    continue
        if sentences:
            print(f"Total number of sentences for training without spellchecker in {data_folder}: {len(sentences)}")
            # train the model num_trainings times without spellchecker
            for i in range(num_trainings):
                model = word2vec_model(sentences) # train model without spellchecker
                print(f"Training model {i+1}/{num_trainings} without spellchecker") 
                if model:
                    model.save(os.path.join(model_folder, f"skipgram_{i+1}_{os.path.basename(data_folder)}.model")) # name should indicate set-belonging and current num of iteration
                    print(f"Model saved as word2vec_{i+1}_{data_folder[-5:]}.model")
        else: 
            print(f"No data found in {data_folder} without spellchecker")
        if sentences_spellchecker:
            continue # skip training as for now I want to know the number of sentences
            # train the model num_trainings times with spellchecker
            for i in range(num_trainings):
                model = word2vec_model(sentences_spellchecker) # train model with spellchecker
                print(f"Training model {i+1}/{num_trainings} with spellchecker")
                if model:
                    #model.save(os.path.join(model_folder, f"word2vec_spellchecker_{i+1}_{data_folder[-5:]}.model"))
                    print(f"Model saved as word2vec_spellchecker_{i+1}_{data_folder[-5:]}.model")
        else:
            print(f"No data found in {data_folder} with spellchecker")
        

if __name__ == "__main__":
    main()