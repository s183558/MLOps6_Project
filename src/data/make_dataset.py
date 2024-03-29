import pandas as pd
from src.utils.utils import load_bucket
from src.utils.utils import get_project_dir

import logging
import src.common.log_config 
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

def preprocess_data(file, store=False) -> pd.DataFrame:
    project_directory = get_project_dir()

    # Load Bucket
    load_bucket("data")

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
    project_directory = get_project_dir()
    train_df = pd.read_pickle(f"{project_directory}/data/processed/{file}.pkl")
    logger.info("Loaded processed datafile")

    return train_df

if __name__=="__main__":
    preprocess_data("data_raw_train", store=True)