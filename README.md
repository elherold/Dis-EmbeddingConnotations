# (Dis)EmbeddingConnotations

**BA Project** *Navigating Politically Loaded Language - Static Word Embeddings as a Tool for Exploring Connotations* 

by Elena Herold

## Technical Setup
This project was implemented with Python version Python 3.12.3
Steps to run the code: 

1. To install necessary packages run:
`pip install -r requirements.txt`

2. To load necessary datasets and trained models...

## Content
### Objectives
Write short statement about what I wish to achieve with this project
### Folder Structure
```
Code_BA/
├── data/
│   ├── data_helper/
│   │   ├── cleaned_words_machtsprache.json
│   │   ├── inter_distances.json
│   │   ├── macht.sprache_words.json
│   │   ├── seeds.json
│   │   ├── valid_seeds.json
│   │   └── word_frequencies.json
│   ├── data_processed/
|   |   ├── left MISSING
|   |   └── right MISSING
│   ├── data_raw/
|   |   ├── left MISSING
|   |   └── right MISSING
|   └── figures/
|       ├── heatmaps
|       |   ├── heatmap_colorblindness_centroid.png
|       |   ├── heatmap_colorblindness_svm.png
|       |   ├── heatmap_nation_centroid.png
|       |   ├── heatmap_nation_svm.png
|       |   ├── heatmap_spinster_centroid.png
|       |   └── heatmap_spinster_svm.png
|       └── comp_inter_intra.png
├── models/
│   ├── SG_*_left.model  (Total of 5 training runs)
│   └── SG_*_right.model (Total of 5 training runs) 
├── src/
│   ├── Connotative Heatmap/
│   │   ├── centroid_functions.py
│   │   ├── connotative_heatmap.py
│   │   ├── svm_functions.py
│   │   └── test_seedwords.py
│   ├── Creating the ES/
│   │   ├── processing.py
│   │   └── word2vec_training.py
│   ├── Evaluation/
│   |   ├── preprocess_machtsprache_words.py
│   |   ├── reliability_ES.py
│   |   └── validity_ES.py
│   └──  load_models.py
├── README.md
└── requirements.txt
```

### Description of Python Files
*Note: Folders and files are ordered according to logical comprehension, not according to repository structure, for this, see "Folder Structure" above*

**1. Folder: Creating the ES**
  - **processing.py:** This file is processing the left- and right-leaning datasets from data/data_raw/ in such a way that it is in the right format to be fed to the Word2Vec model during training. Specifically, it processes the contents of the respective csv files in chunks, removes any remaining URLs and groups the contents into lines of exactly 10 words each. The results are saved incrementally to avoid high memory usage. The resulting csv files are saved under data/data_processed
  - **word2vec_training.py:** This file trains the Word2Vec models iteratively on the datasets categorized as left-leaning and right-leaning under data/data_processed. It handles large datasets by reading and processing the data in manageable chunks to avoid high memory usage. The script logs progress during Word2Vec training and saves each trained model to the specified directory under models/, naming the model files according to the training run and political leaning of the dataset. The training process is repeated five times for each leaning with the hyperparameters specified as: vector_size=30, window_size=10, min_count=15, workers=4, epochs=5, sg=1, negative=5, hs=0.
    
**2. Folder: Connotative heatmap**
  - **test_seedwords.py:** This file performs the validation and filtering process on the original selection of seed words under data/data_helper/seeds.json across the two sets of Word2Vec models. It loads the models and seeds, checks if the seed words are present in the models' vocabularies, and filters out those that are missing. Next, it orders the valid seeds according to their average frequency across both models, and selects the top 10 seeds for each connotative dimension. Final seed sets can be found under data/data_helper/valid_seeds.json
  - **centroid_functions.py:** This file defines the functions necessary to compute the connotative dimensions in the embedding spaces based on centroids of seed word vectors and calculates the cosine similarity between a target vector and the computed dimension. To compute the connotative dimension, the difference between the average vectors for the two opposing sets of seed words is calculated. 
  - **svm_functions.py:** This file defines the functions necessary to train a support vector machine (svm) model with a custom cosine similarity kernel and measure how close a target word is to the decision boundary of the svm. For this it first assigns labels to the two sets of seed words, and fits an SVM with a custom cosine kernel to generate a decision function. Finally, the decision function value for a given target vector is calculated. It indicates the distance between the target vector and the decision boundary (or hyperplane) learned by the SVM, based on cosine similarity. 
  - **connotative_heatmap.py:** this file visualizes the quantified connotatations in form of heatmaps from the trained embedding spaces using either centroid_functions.py or svm_functions.py. The heatmaps are meant to represent the relationshipds between a target word and the different connotative dimensions across both embedding spaces. For this, a list of relevant target words is defined and processed by stripping any excess whitespace and lowercasing for consistency. Next, it loads the trained word2vec models under models/ and continues either with the svm- (method=1) or centroid-based (method=0) methodology. defined. After processing the target words the heatmaps are generated with the distance values ranging from -1 to 1, representing the proximity to the respective connotative dimension. 
    
**3. Folder: Evaluation**
  - **preprocess_machtsprache_words.py:**
  - **reliability_ES.py:**
  - **validity_ES.py:**
    
**Single File:**
  - **load_models.py:**


