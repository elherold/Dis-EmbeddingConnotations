from gensim.models import Word2Vec
import os

def load_models(set_A, set_B, folder_path="models/test", spellchecker=False):
    """
    Loads the trained Word2Vec models saved in the specified folder.
    Differentiates into two sets based on the provided substrings for Set A and Set B
    and excludes spellchecker models, as they are not needed for the final pipeline due to 
    their issue of changing crucial words of the vocabulary.

    Args:
    folder_path (str): The path to the folder containing the models.
    set_A (str): The substring to identify models belonging to Set A.
    set_B (str): The substring to identify models belonging to Set B.

    Returns:
    dict: Two lists of loaded Word2Vec models, one for Set A and one for Set B.
    """
    models_A = []
    models_B = []

    print(f"Loading models from {folder_path}")

    # Get all model files that are not spellchecker models
    model_files = [f for f in os.listdir(folder_path) if f.endswith(".model") and (not spellchecker or "spellchecker" not in f.lower())]

    print(f"Found {len(model_files)} models excluding spellchecker models")

    # Load the models into respective lists based on set_A and set_B
    for model_file in model_files:
        #print(f"Loading model {model_file}")
        model_path = os.path.join(folder_path, model_file)
        model = Word2Vec.load(model_path)
        if set_B in model_file:
            models_B.append(model)
            #print(f"Model {model_file} added to Set B")
        elif set_A in model_file:
            models_A.append(model)
            #print(f"Model {model_file} added to Set A")

    print(f"Loaded {len(models_A)} models for Set A and {len(models_B)} models for Set B")

    return {"set_A": models_A, "set_B": models_B}