# mlops6_project_source

This repository, mlops6_project_source, is dedicated to an MLOps project at DTU focused on implementing a Transformer model for the classification of tweets.

Group 6:  

| Name | Student ID |
|:-|-:|
| Andreas Millarch | s135313 |
| Enrique Vidal | s151988 |
| Spyridon Pikoulas | s230284 |  
| Frederik Erichs | s183558 |


## Overall goal of the project

The primary objective of this project is to analyze a collection of tweets and employ a Natural Language Model (NLM) for binary classification. The aim is to determine whether a given tweet pertains to a real disaster or not.

## What framework are you going to use and you do you intend to include the framework into your project?

Our implementation leverages the Hugging Face transformer encoder for Natural Language Processing (NLP).

## What data are you going to run on (initially, may change)
We have utilized data from a Kaggle competition available at: https://www.kaggle.com/competitions/nlp-getting-started/data?select=test.csv. The dataset consist of 5 columns. An "id" and the "text" of the tweet, the "location" from where the tweet was send, a "keyword" from the tweet, aswell as the "target" wich denotes if the tweet is about a real disaster (1) or not (0). As our model aims to be as simple as possible, we will only train on the "text" and "target" part of the data. Due to restrictions on accessing the labels of the provided test set, we have partitioned the training set for both training, validation and test purposes.

## What models do you expect to use

We expect to utilize one or more pretrained models based on transformer architecture to fine-tune for our purposes. Our initial experiments are based on a pretrained base BERT classifier (available at https://huggingface.co/). 

## Project structure

The directory structure of the project looks like this:

```txt

├── Makefile             <- Makefile with convenience commands like `make data` or `make train`
├── README.md            <- The top-level README for developers using this project.
├── data
│   ├── processed        <- The final, canonical data sets for modeling.
│   └── raw              <- The original, immutable data dump.
│
├── docs                 <- Documentation folder
│   │
│   ├── index.md         <- Homepage for your documentation
│   │
│   ├── mkdocs.yml       <- Configuration file for mkdocs
│   │
│   └── source/          <- Source directory for documentation files
│
├── models               <- Trained and serialized models, model predictions, or model summaries
│
├── notebooks            <- Jupyter notebooks.
│
├── pyproject.toml       <- Project configuration file
│
├── reports              <- Generated analysis as HTML, PDF, LaTeX, etc.
│   └── figures          <- Generated graphics and figures to be used in reporting
│
├── requirements.txt     <- The requirements file for reproducing the analysis environment
|
├── requirements_dev.txt <- The requirements file for reproducing the analysis environment
│
├── tests                <- Test files
│
├── src                  <- Source code for use in this project.
│   │
│   ├── __init__.py      <- Makes folder a Python module
│   │
│   ├── data             <- Scripts to download or generate data
│   │   ├── __init__.py
│   │   └── make_dataset.py
│   │
│   ├── models           <- model implementations, training script and prediction script
│   │   ├── __init__.py
│   │   ├── model.py
│   │
│   ├── visualization    <- Scripts to create exploratory and results oriented visualizations
│   │   ├── __init__.py
│   │   └── visualize.py
│   ├── train_model.py   <- script for training the model
│   └── predict_model.py <- script for predicting from a model
│
└── LICENSE              <- Open-source license if one is chosen
```

Created using [mlops_template](https://github.com/SkafteNicki/mlops_template),
a [cookiecutter template](https://github.com/cookiecutter/cookiecutter) for getting
started with Machine Learning Operations (MLOps).
