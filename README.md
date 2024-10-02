# (Dis)EmbeddingConnotations

**BA Project** *Navigating Politically Loaded Language - Static Word Embeddings as a Tool for Exploring Connotations* 

by Elena Herold

## Technical Setup
This project was implemented with Python version Python 3.12.3

Steps to run the code: 

1. To install necessary packages run:
`pip install -r requirements.txt`

2. To load missing datasets and trained models please download them using this link: https://1drv.ms/f/s!As6v1rinodqPl-MAEMUWQSc2PKcMOA?e=Ml7LnB
The zip files need to be extracted and integrated into the existing folder structure (see below) into the folder with the respective name. 

## Content
### Objectives

My goal is to investigate the question how word embeddings can be leveraged to capture connotations of politically loaded language for the users of the website Macht.Sprache. The scripts of this repository are meant to provide a possible avenue to tackle this question. Starting with the raw datasets (left- and right-leaning newspaper articles), it provides the necessary code to train the word embedding models and perform further operations on the embedding spaces. Furthermore, it includes files for reliability and validity checks of the proposed method. The end-result are heatmaps, in which the connotative scores for the different dimensions of interest are displayed on a color scale from blue to red for both the left-leaning and right-leaning dataset. 
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
|   |   ├── left 
|   |   └── right
│   ├── data_raw/
|   |   ├── left 
|   |   └── right
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
  - **word2vec_training.py:** This file trains the Word2Vec models iteratively on the datasets categorized as left-leaning and right-leaning under data/data_processed. It handles large datasets by reading and processing the data in manageable chunks to avoid high memory usage. The script logs progress during Word2Vec training and saves each trained model to the specified directory under models/, naming the model files according to the training run and political leaning of the dataset. The training process is repeated five times for each leaning with the hyperparameters specified as: vector_size=300, window_size=10, min_count=15, workers=4, epochs=5, sg=1, negative=5, hs=0.
    
**2. Folder: Connotative heatmap**
  - **test_seedwords.py:** This file performs the validation and filtering process on the original selection of seed words under data/data_helper/seeds.json across the two sets of Word2Vec models. It loads the models and seeds, checks if the seed words are present in the models' vocabularies, and filters out those that are missing. Next, it orders the valid seeds according to their average frequency across both models, and selects the top 10 seeds for each connotative dimension. Final seed sets can be found under data/data_helper/valid_seeds.json
  - **centroid_functions.py:** This file defines the functions necessary to compute the connotative dimensions in the embedding spaces based on centroids of seed word vectors and calculates the cosine similarity between a target vector and the computed dimension. To compute the connotative dimension, the difference between the average vectors for the two opposing sets of seed words is calculated. 
  - **svm_functions.py:** This file defines the functions necessary to train a support vector machine (svm) model with a custom cosine similarity kernel and measures how close a target word is to the decision boundary of the svm. For this it first assigns labels to the two sets of seed words, and fits an SVM with a custom cosine kernel to generate a decision function. Finally, the decision function value for a given target vector is calculated. It indicates the distance between the target vector and the decision boundary (or hyperplane) learned by the SVM, based on cosine similarity. 
  - **connotative_heatmap.py:** This file visualizes the quantified connotatations in form of heatmaps from the trained embedding spaces using either centroid_functions.py or svm_functions.py. The heatmaps are meant to represent the relationshipds between a target word and the different connotative dimensions across both embedding spaces. For this, a list of relevant target words is defined and processed by stripping any excess whitespace and lowercasing for consistency. Next, it loads the trained word2vec models under models/ and continues either with the svm- (method=1) or centroid-based (method=0) methodology. After processing the target words the heatmaps are generated with the distance values ranging from -1 to 1, representing the proximity to the respective connotative dimension. 
    
**3. Folder: Evaluation**
  - **preprocess_machtsprache_words.py:** This file processes data from the Macht.Sprache projet by cleaning and filtering the words present under data/data_helper/macht_sprache_words.json. The script is structured to ensure that only relevant English words are retained and that they appear in both left-leaning and right-leaning datasets. More specifically, it extracts words from the "lemma" and "relatedterms" fields for all English words from the original Macht.Sprache file. It cleans the words by converting them to lowercase, and removing spaces, hyphens, and parentheses, while also filtering out any words containing asterisks. This is done to match the preprocessing of the datasets in the left-leaning and right-leaning embedding spaces. Additionally, with help of the necessary functions from the reliability_ES.py, the inter-dataset distances for the list of Macht.Sprache target words are calculated and saved, which are used in the validity_ES.py. 
  - **reliability_ES.py:** This file processes a set of target words from macht.sprache under data/data_helper/cleaned_words_machtsprache.json and calculates their second-order similarity vectors using the trained embedding spaces to quantify shifts in local neighborhood embeddings. The shifts are calculated both within and between datasets and then visualized using boxplots (the figure is saved under data/figures/comp_inter_intra.png). The second order similarity vectors are created by calculating the average cosine distance between the k=10 nearest neighbor sets of the target word across both embedding spaces. The second-order similarity vectors are then used to assess how similarly a word is represented across the two different datasets. Next, bootstrapping is performed to calculate confidence intervals for the mean cosine distances, both for the intra- and inter-dataset distanes. 
  - **validity_ES.py:** This file identifies and compares nearest neighbors to stratify the words based on how differently they behave across the two datasets and then computes the nearest neighbor sets of randomly selected words from different levels of stratification. For this it laods the pre-calculated inter-dataset distances under data/data_helper/inter_distances.json. These distances reflect how much the neighborhoods of a target word differ across the left-leaning and right-leaning embedding spaces. Based on these distances, it stratifies words into hgih-shift, middle-shift, and a low-shift category. From each group a random word is sampled for further analysis. For these random words, the union of top 10 nearest neighbors across multiple training runs is computed, and the neighbors are ranked by their average similarity score. 
    
**Single File:**
  - **load_models.py:** This file provides the necessary function to load the trained word2vec mdels from models/ and organize them into two sets based on the identifiers left and right. The function is used multiple times througout the project and saved externally, so other files can import it, reducing redundancy. 


