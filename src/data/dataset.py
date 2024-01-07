from typing import Optional
from datasets import Dataset, DatasetDict
from data.make_dataset import load_train_df, preprocess_data
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import TensorDataset, DataLoader
import pytorch_lightning as lit
import os

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