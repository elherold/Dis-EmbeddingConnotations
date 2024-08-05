from load_models import load_models

# load models
models = load_models("left", "right")
# models are stored as a dict separated into set A (feminism) and set B (anti-feminism)
# print the nearest neighbors for a target word in each set
# for this we have to iterate through all models in each set and print the NN that are also most common acrosss all models
# we will print the top 5 NN for each set
target_word = "mansplain"
for set_name, model_list in models.items():
    print(f"Set: {set_name}")
    for model in model_list:
        if target_word in model.wv.key_to_index:
            print(f"Word '{target_word}' found in the embedding space of {set_name}.")
            nearest_neighbors = model.wv.most_similar(target_word, topn=25)
            print(f"Nearest neighbors for '{target_word}': {[neighbor[0] for neighbor in nearest_neighbors]}")
        else:
            print(f"Word '{target_word}' not found in the embedding space of {set_name}. Please try another one. The target word needs to be included in all embedding spaces.")