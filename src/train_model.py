import pandas as pd
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

from transformers import BertTokenizer, AutoTokenizer

from src.data.dataset import LitDM
from src.models.model import AlbertClassifier
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
    model = AlbertClassifier(optimizer=optimizer, learning_rate=learning_rate)

    # Training
    epochs = cfg.model["epochs"]
    trainer = Trainer(
        max_epochs=cfg.model["epochs"],
        callbacks=[EarlyStopping(monitor="val_loss", mode="min"),
        check_val_every_n_epoch=1, # Evaluate after every epoch
        enable_checkpointing = False, # Model checkpoints
        limit_train_batches=0.2, # Train at only 20% of the data
        #limit_val_batches=0.2,
        num_sanity_val_steps=0, # Do not perform sanity check
        #profiler="simple",
        precision="16-true", # Drop from float 32 to float 16 precision for memory efficiency
    ])
    
    trainer.fit(model, dm)

    # Evaluation
    trainer.test(model,dm)

