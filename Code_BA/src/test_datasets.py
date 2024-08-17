from load_models import load_models

# Load models
models = load_models("left", "right")
# Models are stored as a dict separated into set A (feminism) and set B (anti-feminism)
# Print the nearest neighbors for a target word in each set
# For this we have to iterate through all models in each set and print the NN that are also most common across all models
# We will print the top 5 NN for each set
target_word = "spinster"

for set_name, model_list in models.items():
    print(f"\nSet: {set_name}")
    for model in model_list:
        if target_word in model.wv.key_to_index:
            # Print frequency of the target word
            word_freq = model.wv.get_vecattr(target_word, "count")
            print(f"Word '{target_word}' found in the embedding space of {set_name}. Frequency: {word_freq}")
            
            # Get and print the nearest neighbors
            nearest_neighbors = model.wv.most_similar(target_word, topn=25)
            print(f"Nearest neighbors for '{target_word}': {[neighbor[0] for neighbor in nearest_neighbors]}")
        else:
            print(f"Word '{target_word}' not found in the embedding space of {set_name}. Please try another one. The target word needs to be included in all embedding spaces.")
