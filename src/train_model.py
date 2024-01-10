import pandas as pd
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

from transformers import BertTokenizer, AutoTokenizer

from src.data.dataset import LitDM
from src.models.model import BertClassifier
from src.project_manager import ProjectManager

from hydra import compose, initialize

if __name__ == '__main__':
    # Create Project Manager    
    project_manager = ProjectManager()

    # Initialize hydra hypermaters
    with initialize(config_path="../conf", version_base="1.1"):
        cfg: dict = compose(config_name='config.yaml')

    # Specify tokenizerma
    tokenizer =  AutoTokenizer.from_pretrained('albert-base-v1')

    # Get Lit Data Module
    dm = LitDM(cfg, tokenizer =tokenizer)

    # Model
    learning_rate = cfg.model["lr"]
    optimizer = cfg.model["optimizer"]
    model = BertClassifier(optimizer=optimizer, learning_rate=learning_rate)

    # Training
    epochs = cfg.model["epochs"]
    trainer = Trainer(max_epochs=cfg.model["epochs"],
                      callbacks=[EarlyStopping(monitor="val_loss", mode="min")])
    
    trainer.fit(model, dm)

    # Evaluation
    trainer.test(model,dm)

