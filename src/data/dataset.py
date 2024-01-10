from typing import Optional
from datasets import Dataset, DatasetDict
from data.make_dataset import load_train_df, preprocess_data
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import TensorDataset, DataLoader
import pytorch_lightning as lit
import os
from typing import Optional
from transformers import BertTokenizer, AutoTokenizer

def get_datasets(cfg):


    # Split training data
    train_df, test_df = train_test_split(train_df, test_size=cfg.data.test_size)
    train_df, valid_df = train_test_split(train_df, test_size=cfg.data.valid_size)

    return train_df, valid_df, test_df


def get_dataset_dict(cfg):
    # Preprocess(or load) and get data
    try:
        train_df = load_train_df("train")
    except:
        train_df = preprocess_data("train", store=True)

    # Split training data
    train_df, test_df = train_test_split(train_df, test_size=cfg.data.test_size)

    # Build Datasets
    train_dataset = Dataset.from_pandas(train_df[:100])
    test_dataset = Dataset.from_pandas(test_df[:100])
    dataset_dict = DatasetDict({
        'train': train_dataset,
        'test': test_dataset
    })
    return dataset_dict


class LitDM(lit.LightningDataModule):
    """
    https://lightning.ai/docs/pytorch/stable/data/datamodule.html

    Load, proces and tokenize data
    Collect dataloaders
    """
    def __init__(self, cfg:dict, data_dir: str = "path/to/dir", tokenizer = None):
        super().__init__()
        self.cfg = cfg
        self.data_dir = data_dir
        self.tokenizer = tokenizer

        self.cpu_cnt = os.cpu_count() or 2

        self.prepare()

    def collect_dataset(self):
        """
        Load or proces datasets
        """
        try:
            self.data_df = load_train_df("train")
        except:
            self.data_df = preprocess_data("train", store=True)

    def split_datasets(self):
        """
        Split into train, valid and test dataframes from config file
        """
        # Split training data
        train_df, self.test_df = train_test_split(self.data_df, test_size=self.cfg.data.test_size)
        # TODO: log stuff here
        self.train_df, self.valid_df = train_test_split(train_df, test_size=self.cfg.data.valid_size)

    def __tokenize_data(self, df, tokenizer) -> torch.tensor:
        """
        Helper function to tokenize dataframes. assumes "text" column with input
        Return tokenized, padded tensor
    
        """
        tokenized = tokenizer.batch_encode_plus(
                                                df['text'].tolist(),
                                                max_length=self.cfg.data["token_max_length"],
                                                padding='max_length',
                                                truncation=True,
                                                return_tensors='pt'
                                            )
        return tokenized
    
    def tokenize_data(self):    
        """
        Tokenize input, set labels as tensors
        """
        ## Train
        self.train_encodings = self.__tokenize_data(self.train_df, self.tokenizer)
        self.train_labels = torch.tensor(self.train_df["labels"].tolist())

        ## Validation
        self.valid_encodings = self.__tokenize_data(self.valid_df, self.tokenizer)
        self.valid_labels = torch.tensor(self.valid_df["labels"].tolist())

        ## Test
        self.test_encodings = self.__tokenize_data(self.test_df, self.tokenizer)
        self.test_labels = torch.tensor(self.test_df["labels"].tolist())

    def get_tensor_datasets(self):
        self.train_dataset = TensorDataset(self.train_encodings["input_ids"], 
                                      self.train_encodings["attention_mask"], 
                                      self.train_labels)
        
        self.valid_dataset = TensorDataset(self.valid_encodings["input_ids"], 
                                    self.valid_encodings["attention_mask"], 
                                    self.valid_labels)
        
        self.test_dataset = TensorDataset(self.test_encodings["input_ids"], 
                                    self.test_encodings["attention_mask"], 
                                    self.test_labels)
    def prepare(self):
        self.collect_dataset()
        self.split_datasets()
        self.tokenize_data()
        self.get_tensor_datasets()

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.train_dataset, batch_size=self.cfg.data["batch_size"], num_workers=self.cpu_cnt)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.valid_dataset, batch_size=self.cfg.data["batch_size"], num_workers=self.cpu_cnt)
    
    def test_dataloader(self) -> DataLoader:
        return DataLoader(self.test_dataset, batch_size=self.cfg.data["batch_size"], num_workers=self.cpu_cnt)
    