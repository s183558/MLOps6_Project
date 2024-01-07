import os
import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer
import logging
import src.common.log_config 
import torch

logger=logging.getLogger(__name__)


def find_project_dir() -> str:
    return os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def processing(df: pd.DataFrame) -> pd.DataFrame:
    # Drop stuff
    df = df.dropna()
    df = df.drop(columns=['keyword','location','id'], axis=1)
    df.drop_duplicates(inplace=True)

    # Process
    df['text'] = df['text'].str.lower()
    df = df.rename(columns={"target": "labels"})
    df = df.reset_index(drop=True)
    logger.debug(df.head(5))

    return df

def preprocess_data(file, store=False) -> pd.DataFrame:
    # Get Data
    project_directory = find_project_dir()
    data_path  = f"{project_directory}/data/raw/"
    train_df = pd.read_csv(f'{data_path}{file}.csv')

    # Process it
    logger.info(f'Preprocessing dataframe of shape: {train_df.shape}')
    train_df = processing(train_df)

    # Store it
    if store:
        logger.info(f"Stored datafile at {project_directory}/data/processed")
        train_df.to_pickle(f"{project_directory}/data/processed/{file}.pkl")

    return train_df

def load_train_df(file: str) -> pd.DataFrame:
    logger.info("Loaded processed datafile")
    project_directory = find_project_dir()
    train_df = pd.read_pickle(f"{project_directory}/data/processed/{file}.pkl")

    return train_df

def preprocess_data2():
    project_directory = find_project_dir()
    data_path  = f"{project_directory}/data/raw/"
    data_df = pd.read_csv(f'{data_path}train.csv')

    # Process it
    logger.info(f'Preprocessing dataframe of shape: {data_df.shape}')
    data_df = processing(data_df)

    train_df, val_df = train_test_split(data_df, test_size=0.2, random_state=42)

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    def tokenize_data(df):
        tokenized = tokenizer.batch_encode_plus(
            df['text'].tolist(),
            max_length=128,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        return tokenized

    train_encodings = tokenize_data(train_df)
    train_labels = torch.tensor(train_df["labels"].tolist())

    val_encodings = tokenize_data(val_df)
    val_labels = torch.tensor(val_df["labels"].tolist())

    return train_encodings, train_labels, val_encodings, val_labels

if __name__=="__main__":  
    preprocess_data("train", store=True)
