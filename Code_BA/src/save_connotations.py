import json
import pandas as pd
from connotative_heatmap import load_embeddings
from svm_ensemble_functions import train_svms_on_centroids, distance_from_svms
import os


def save_connotations(embeddings, seeds):
    connotative_features = []
    for set_name, embedding in embeddings.items():
         for seed in seeds:
            pos_seeds = seeds[seed]['pos_pole']
            neg_seeds = seeds[seed]['neg_pole']

            pos_vector = [embedding.wv[word] for word in pos_seeds if word in embedding.wv]
            neg_vector = [embedding.wv[word] for word in neg_seeds if word in embedding.wv]

            # Train SVMs on centroids
            svm_estimators = train_svms_on_centroids(pos_vector, neg_vector, n_clusters=3)

            for word in embedding.wv.index_to_key:
                word_vector = embedding.wv[word]  # Get the vector representation of the word
                distance = distance_from_svms(svm_estimators, word_vector)
                connotative_features.append({
                    'word': word,
                    'set': set_name,
                    'dimension': seed,
                    'distance': distance
                })
    
    # Convert to DataFrame and save as CSV
    df = pd.DataFrame(connotative_features)
    # create folder if it doesnt exist already
    if not os.path.exists('data/data_helper/connotations'):
        os.makedirs('data/data_helper/connotations')

    df.to_csv(f'data/data_helper/connotations/connotations_try.csv', index=False)     
            

def main():
    
    embedding_paths = {
        'set_A' : 'models/new/individual_models/word2vec_1_set_A.model',
        'set_B' : 'models/new/individual_models/word2vec_1_set_B.model'
    }

    # Load embeddings
    embeddings = {set_name: load_embeddings(path) for set_name, path in embedding_paths.items()}
    print('Embeddings loaded successfully')

    # Load seeds
    with open('data/data_helper/seeds_cleaned.json', 'r') as f:
            dic_seeds = json.load(f)
    print('Seeds loaded successfully')
    
    # Save connotations
    save_connotations(embeddings, dic_seeds)
    print(f'Connotations saved successfully to data/data_helper/connotations/connotations_try.csv')
    
    return None

if __name__ == '__main__':
    main()

