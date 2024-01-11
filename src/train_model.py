import pandas as pd
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

from transformers import BertTokenizer, AutoTokenizer

from src.data.dataset import LitDM
from src.models.model import AlbertClassifier
from src.project_manager import ProjectManager

import hydra
from omegaconf import DictConfig, OmegaConf

@hydra.main(version_base=None, config_path="../conf", config_name="config.yaml")
def train_main(cfg:DictConfig):
    # Specify tokenizer
    tokenizer =  AutoTokenizer.from_pretrained('albert-base-v1')

    # Get Lit Data Module
    dm = LitDM(cfg, tokenizer =tokenizer)

    # Model
    learning_rate = cfg.model["lr"]
    optimizer = cfg.model["optimizer"]
    model = AlbertClassifier(optimizer=optimizer, learning_rate=learning_rate)

    # Training
    trainer = Trainer(
        default_root_dir='models/',
        max_epochs=cfg.model["epochs"],

        callbacks=[EarlyStopping(monitor="val_loss", mode="min")],

        check_val_every_n_epoch=1, # Evaluate after every epoch
        enable_checkpointing = True, # Model checkpoints

        limit_train_batches=cfg.model["limit_train_batches"], # Train at only 20% of the data
        limit_val_batches=cfg.model["limit_val_batches"],

        num_sanity_val_steps=0, # Do not perform sanity check
        #profiler="simple",
        precision=cfg.model["mixed_precision"], # Drop from float 32 to float 16 precision for memory efficiency
                    )
    # Fit model
    trainer.fit(model, dm)

    # Evaluation
    trainer.test(model,dm)

if __name__ == '__main__':
    train_main()
