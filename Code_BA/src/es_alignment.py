import os
import numpy as np
from scipy.linalg import orthogonal_procrustes
from gensim.models import Word2Vec

def load_models(set_name, spellchecker, folder_path="models/new/individual_models"):
    """
    Loads the trained Word2Vec models saved in the specified folder. 
    Differentiates between spellchecker and non-spellchecker models and loads according to set name.

    Args:
    folder_path (str): The path to the folder containing the models.
    set_name (str): The name of the set to load the models for (Set_A or Set_B).
    Spellchecker (bool): Whether to load spellchecker models or not.

    Returns:
    list: A list  containing the loaded Word2Vec model.
    """
    models = []
    print(f"Loading models for {set_name} with spellchecker: {spellchecker}")
    # loading the correct model files depending on the set and whether we want a spellchecker model
    if spellchecker: 
        print("spellchecker detected, Loading spellchecker models")
        model_files = [f for f in os.listdir(folder_path) if f.endswith(f"{set_name}.model") and "spellchecker" in f.lower()]
    else:
        print(f"Loading non-spellchecker models for {set_name}")
        model_files = [f for f in os.listdir(folder_path) if f.endswith(f"{set_name}.model") and "spellchecker" not in f.lower()]

    print(f"Found {len(model_files)} models for {set_name} with spellchecker: {spellchecker}")
    # load the models
    for model_file in model_files:
        model_path = os.path.join(folder_path, model_file)
        model = Word2Vec.load(model_path)
        models.append(model)
    return models

def align_embeddings(reference_model, models):
    """
    Aligns the embeddings of the given models to the reference model using orthogonal Procrustes.

    Args:
    reference_model (gensim.models.Word2Vec): The reference Word2Vec model.
    models (list): A list of tuples containing the model file name and the Word2Vec model to align.

    Returns:
    list: A list of tuples containing the model file name and the aligned Word2Vec model.
    """
    reference_matrix = reference_model.wv.vectors # get the embedding matrix of the reference model
    aligned_matrices = []

    # align the embeddings of the models to the reference model
    for model in models:
        embedding_matrix = model.wv.vectors
        R, _ = orthogonal_procrustes(embedding_matrix, reference_matrix) # calculate the orthogonal Procrustes transformation
        aligned_embeddings = np.dot(embedding_matrix, R) # apply the transformation to the embeddings
        aligned_matrices.append(aligned_embeddings)

    return aligned_matrices

def compute_average_and_variance(aligned_matrices):
    """
    Computes the average and variance of each vector across the aligned embeddings.

    Args:
    aligned_matrices (list): A list of aligned embedding matrices.

    Returns:
    numpy.ndarray: The mean embeddings.
    numpy.ndarray: The variance embeddings.
    """
    
    aligned_matrices = np.array(aligned_matrices) # convert the list to a numpy array so we can perform operations on all vectors at once 
    mean_embeddings = np.mean(aligned_matrices, axis=0) 
    variance_embeddings = np.var(aligned_matrices, axis=0)

    return mean_embeddings, variance_embeddings

def mean_embedding_model(vocab, mean_embeddings):
    """
    Creates a new Word2Vec model with the mean embeddings as the vectors.

    Args:
    vocab (dict): The vocabulary of the model.
    mean_embeddings (numpy.ndarray): The mean embeddings.

    Returns:
    gensim.models.Word2Vec: The Word2Vec model with the mean embeddings.
    """
    mean_model = Word2Vec(vector_size=mean_embeddings.shape[1]) # create a new Word2Vec model with the same dimensionality as the mean embeddings
    mean_model.build_vocab_from_freq(vocab) # build the vocabulary of the model
    mean_model.wv.vectors = mean_embeddings # set the mean embeddings as the vectors of the model

    return mean_model

def main():
    """
    Main function to align and average Word2Vec embeddings from multiple models as well as compute the variance of the embeddings.

    This function:
    - Loads the trained Word2Vec models.
    - Aligns the embeddings of the models to a reference model.
    - Computes the average and variance of the aligned embeddings.
    - Creates a new Word2Vec model with the mean embeddings.
    - Saves the mean model and variance embeddings to the specified folder.
    """
    
    # Differentiate between sets and spellchecker usage
    for set_name in ["set_A", "set_B"]:
        for spellchecker in [True, False]:
            models = load_models(set_name, spellchecker)
            if not models:
                print(f"No eligible models found for {set_name} with spellchecker: {spellchecker}")
                continue

            reference_model = models[0] # Use the first model as the reference model
            aligned_models = align_embeddings(reference_model, models)
            mean_embeddings, variance_embeddings = compute_average_and_variance(aligned_models)

            vocab = {word: reference_model.wv.get_vecattr(word, "count") for word in reference_model.wv.index_to_key}
            mean_model = mean_embedding_model(vocab, mean_embeddings)

            # save the mean model
            os.makedirs("models/new/summarized", exist_ok=True)
            # only save the model if the exact same name doesnt exist already
            if not os.path.exists(os.path.join("models/new/summarized", f"mean_{set_name}_{'spellchecker' if spellchecker else 'no_spellchecker'}.model")):
                mean_model.save(os.path.join("models/new/summarized", f"mean_{set_name}_{'spellchecker' if spellchecker else 'no_spellchecker'}.model"))
                print(f"Mean model has been saved for {set_name} with spellchecker: {spellchecker}")
            else:
                print(f"Model already exists for {set_name} with spellchecker: {spellchecker}")
            # save the variance embeddings only if they dont exist already
            if not os.path.exists(os.path.join("models/new/summarized", f"variance_{set_name}_{'spellchecker' if spellchecker else 'no_spellchecker'}.npy")):
                np.save(os.path.join("models/new/summarized", f"variance_{set_name}_{'spellchecker' if spellchecker else 'no_spellchecker'}.npy"), variance_embeddings)
                print(f"Variance embeddings have been saved for {set_name} with spellchecker: {spellchecker}")
            else:
                print(f"Variance embeddings already exist for {set_name} with spellchecker: {spellchecker}")

if __name__ == "__main__":
    main()

