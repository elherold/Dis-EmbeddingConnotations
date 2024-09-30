# (Dis)EmbeddingConnotations

 **BA Project** *Navigating Politically Loaded Language - Static Word Embeddings as a Tool for Exploring Connotations* 
 
 by Elena Herold
-------------------------
## Technical Setup
This project was implemented with Python version Python 3.12.3
Steps to run the code: 

1. To install necessary packages run:
`pip install -r requirements.txt`

2. To load necessary datasets and trained models...

## Content
### Folder Structure
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
│   └── data_raw/
├── models/
│   ├── model1.pkl
│   └── model2.pkl
├── src/
│   ├── utils/
│   │   └── helper.py
│   └── main.py
├── README.md
└── requirements.txt




**Preprocessing** 
- process the newspaper dataset into right format - Docstrings check - run test check
- preprocessing of Macht.Sprache words - Docstring check - run test check

**Word2vec** - Docstrings check - run test check

**Connotative Heatmap**
- load models - Docstrings check - run test check 
- test seed words - Docstring check - run test check
- Vector offset method functions - Docstrings check - run test check 
- SVM method functions - Docstrings check - run test check
- Calculate the connotative heatmap - Docstrings check - run test check 

**Evaluation**
- reliability_ES.py - Docstring check - run test check 
- validity_ES.py - Docstring check - run test check 
