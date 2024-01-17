import os
import pandas as pd
import logging
import src.common.log_config 
import torch
from google.cloud import storage

logger=logging.getLogger(__name__)

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

def get_project_dir() -> str:
    return os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def load_bucket():
    storage_client = storage.Client()
    bucket = storage_client.bucket("mlops6_tweets")
    blobs = bucket.list_blobs(prefix="data")
    project_dir = get_project_dir()
    for blob in blobs:
        local_path = os.path.join(project_dir, blob.name)
        blob.download_to_filename(local_path)

def preprocess_data(file, store=False) -> pd.DataFrame:
    project_directory = get_project_dir()

    # Create raw directory(for git)
    raw_dir = os.path.join(project_directory, "data/raw")
    if not (os.path.exists(raw_dir) and os.path.isdir(raw_dir)):
        os.makedirs(raw_dir, exist_ok=True)

    # Load Bucket
    data_content = os.listdir(raw_dir)
    if data_content == []:
        load_bucket()

    # Get Data
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
    project_directory = get_project_dir()
    train_df = pd.read_pickle(f"{project_directory}/data/processed/{file}.pkl")

    return train_df


if __name__=="__main__":  
    preprocess_data("data_raw_train", store=True)
