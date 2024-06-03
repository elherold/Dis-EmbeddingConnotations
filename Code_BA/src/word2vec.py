from gensim.models import Word2Vec
import ast
import os
import logging
import pickle

def read_processed_comments(file_path):
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

def word2vec_model(sentences, vector_size=300, window=6, min_count=10, workers=4, epochs=10):
    # Set up logging to get information about the training process
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

    # Print information about the training setup
    print(f"Training Word2Vec model with the following parameters:")
    print(f"Vector size: {vector_size}")
    print(f"Window size: {window}")
    print(f"Minimum count: {min_count}")
    print(f"Number of workers: {workers}")
    print(f"Number of epochs: {epochs}")
    print(f"Number of sentences: {len(sentences)}")

    try:
        # Initialize the model
        model = Word2Vec(vector_size=vector_size, window=window, min_count=min_count, workers=workers)
        
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
    data_folder = "data/data_processed"
    model_folder = "models/"
    os.makedirs(model_folder, exist_ok=True)
    
    for dataset in os.listdir(data_folder):
        try:
            file_path = os.path.join(data_folder, dataset)
            if os.path.isfile(file_path):
                sentences = read_processed_comments(file_path)
                if sentences:
                    model = word2vec_model(sentences)
                    if model:
                        model.save(os.path.join(model_folder, f"word2vec_{dataset}.model"))
        except Exception as e:
            print(f"Error processing the dataset {dataset}: {e}")
            continue

if __name__ == "__main__":
    main()