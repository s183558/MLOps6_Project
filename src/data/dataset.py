import logging
import pandas as pd
import numpy as np
import os
from typing import Optional
import numpy as np

import pytorch_lightning as lit
import torch
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler


logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
                        datefmt='%m-%d %H:%M')

logger=logging.getLogger(__name__)

def processing(df):
    df = df.fillna('No info', axis=1)
    df['input'] = 'KEYWORD: ' + df.keyword + '; LOCATION: ' + df.location + '; TEXT: ' + df.text
    df['input'] = df['input'].str.lower()
    df = df.drop(columns=['keyword','location','text'], axis=1)
    df.fillna("no info", inplace=True)
    df.drop_duplicates(inplace=True)
    return df

class DisasterTweets():
    def __init__(self):
        project_directory = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        self.data_path  = f"{project_directory}/data/raw/"
        self.load_raw()
        self.process_datasets()

    def load_raw(self):
        self.train = pd.read_csv(f'{self.data_path}train.csv')       
        logger.info(f'train columns: {self.train.columns.tolist()}')
        logger.info(f'train dataset length: {len(self.train)}')
        
    def process_datasets(self):
        self.train = processing(self.train)
        self.train['target'] = np.float64(self.train['target'])

class LitDM(lit.LightningDataModule):
    def __init__(self, data_path: str='data/processed', batch_size: int = 32):
        super().__init__()
       
        self.data_path = data_path
        self.batch_size = batch_size
        self.cpu_cnt = os.cpu_count() or 2

    def setup(self, stage: Optional[str] = None) -> None:
        train_inputs = torch.load('data/processed/train_inputs.pt')
        validation_inputs= torch.load('data/processed/validation_inputs.pt')

        train_labels = torch.load('data/processed/train_labels.pt')
        validation_labels = torch.load('data/processed/validation_labels.pt')

        train_masks =torch.load('data/processed/train_masks.pt')
        validation_masks = torch.load('data/processed/validation_masks.pt')
     
        self.train_data = TensorDataset(train_inputs, train_masks, train_labels)
        self.validation_data = TensorDataset(validation_inputs, validation_masks, validation_labels)
        self.validation_data = TensorDataset(validation_inputs, validation_masks, validation_labels)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.train_data, batch_size=self.batch_size, num_workers=self.cpu_cnt)
        
    def test_dataloader(self) -> DataLoader:
        pass

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.validation_data, batch_size=self.batch_size, num_workers=self.cpu_cnt)