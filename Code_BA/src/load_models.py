from gensim.models import Word2Vec
import os

def load_models(set_A, set_B, folder_path="models/new"):
    """
    Loads the trained Word2Vec models saved in the specified folder.
    Differentiates into two sets based on the provided substrings for Set A and Set B

    Parameters:
    ------------
    folder_path (str): The path to the folder containing the models.
    set_A (str): The substring to identify models belonging to Set A.
    set_B (str): The substring to identify models belonging to Set B.

    Returns:
    -----------
    dict: Two lists of loaded Word2Vec models, one for Set A and one for Set B.
    """
    models_A = []
    models_B = []

    print(f"Loading models from {folder_path}")

    # Get all model files from the specified folder
    model_files = [f for f in os.listdir(folder_path) if f.endswith(".model") ]

    print(f"Found {len(model_files)} models")

    # Load the models into respective lists based on set_A and set_B
    for model_file in model_files:
        
        model_path = os.path.join(folder_path, model_file)
        model = Word2Vec.load(model_path)
        
        if set_B in model_file:
            models_B.append(model)

        elif set_A in model_file:
            models_A.append(model)


    print(f"Loaded {len(models_A)} models for Set left and {len(models_B)} models for Set right")

    return {"left": models_A, "right": models_B}