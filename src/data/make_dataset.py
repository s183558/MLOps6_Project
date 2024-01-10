import os
import pandas as pd
import logging
import src.common.log_config 
import torch

logger=logging.getLogger(__name__)


############ -- train_model_lit.py -- ############

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

def preprocess_data2(file_path: str, store: bool = True):
    # Load File
    data_df = pd.read_csv(file_path)

    # Process it
    logger.info(f'Preprocessing dataframe of shape: {data_df.shape}')
    data_df = processing(data_df)

    return data_df


############ -- train_model.py -- ############
def get_project_dir() -> str:
    return os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def preprocess_data(file, store=False) -> pd.DataFrame:
    # Get Data
    project_directory = get_project_dir()
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
    preprocess_data("train", store=True)
