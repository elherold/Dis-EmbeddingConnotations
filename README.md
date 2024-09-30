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
  - **processing.py:**
  - **word2vec_training.py:**
**2. Folder: Connotative heatmap**
  - **test_seedwords.py:** Des
  - **centroid_functions.py:** Description of what it does
  - **svm_functions.py:** Description of what it does
  - **connotative_heatmap.py:** des
**3. Folder: Evaluation**
  - **preprocess_machtsprache_words.py:**
  - **reliability_ES.py:**
  - **validity_ES.py:**
**Single File:**
  - **load_models.py:**


