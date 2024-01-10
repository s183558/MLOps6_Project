from src.data.make_dataset import preprocess_data2
from src.models.model import BertClassifier
from torch.utils.data import DataLoader, TensorDataset
from pytorch_lightning import Trainer
from src.project_manager import ProjectManager
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer, AutoTokenizer
import torch
from hydra import compose, initialize
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

if __name__ == '__main__':
    # Create Project Manager    
    project_manager = ProjectManager()

    # Initialize hydra hypermaters
    with initialize(config_path="../conf", version_base="1.1"):
        cfg: dict = compose(config_name='config.yaml')

    # Preprocess data
    file_path = os.path.join(project_manager.get_data_folder(), "raw/train.csv")
    data_df = preprocess_data2(file_path)
    # Load processed data
    # file_path = os.path.join(project_manager.get_data_folder(), "processed/train.pkl")
    # data_df = pd.read_pickle(file_path)

    # Split
    train_df, val_df = train_test_split(data_df, test_size=cfg.data["test_size"], random_state=cfg.data["seed"])

    # Tokenize
    tokenizer =  AutoTokenizer.from_pretrained('albert-base-v1')
    def tokenize_data(df):
        tokenized = tokenizer.batch_encode_plus(
            df['text'].tolist(),
            max_length=cfg.data["token_max_length"],
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        return tokenized
    ## Train
    train_encodings = tokenize_data(train_df)
    train_labels = torch.tensor(train_df["labels"].tolist())
    ## Validation
    val_encodings = tokenize_data(val_df)
    val_labels = torch.tensor(val_df["labels"].tolist())

    # Create Dataloaders
    train_dataset = TensorDataset(train_encodings["input_ids"], train_encodings["attention_mask"], train_labels)
    test_dataset = TensorDataset(val_encodings["input_ids"], val_encodings["attention_mask"], val_labels)
    train_loader = DataLoader(train_dataset, batch_size=cfg.data["batch_size"], shuffle=True, num_workers=15)
    test_loader = DataLoader(test_dataset, batch_size=cfg.data["batch_size"], num_workers=15)
    logger.info(f"Train Batches: {len(train_loader)}")
    logger.info(f"Test Batches: {len(test_loader)}")

    # Model
    learning_rate = cfg.model["lr"]
    optimizer = cfg.model["optimizer"]
    model = BertClassifier(optimizer=optimizer, learning_rate=learning_rate)

    # Training
    logger.info(f"--- Training Starts ---")
    epochs = cfg.model["epochs"]
    trainer = Trainer(
        max_epochs=10,
        check_val_every_n_epoch=1,
        enable_checkpointing = False,
        limit_train_batches=0.2, # Train at only 20% of the data
        #limit_val_batches=0.2,
        num_sanity_val_steps=0,
        #profiler="simple",
        precision="16-true", # Drop from floar 32 to float 16 for memory efficiency
    )
    trainer.fit(model, train_loader, test_loader)

