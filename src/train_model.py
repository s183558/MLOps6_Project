import os

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint

from pytorch_lightning.loggers import WandbLogger

from transformers import AutoTokenizer

from src.data.dataset import LitDM
from src.models.model import AlbertClassifier

import hydra
from omegaconf import DictConfig

@hydra.main(version_base=None, config_path="../conf", config_name="config.yaml")
def train_main(cfg:DictConfig):
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    # Specify tokenizer
    tokenizer =  AutoTokenizer.from_pretrained('albert-base-v1')

    # Get Lit Data Module
    dm = LitDM(cfg, tokenizer =tokenizer)

    # Model
    learning_rate = cfg.model["lr"]
    optimizer = cfg.model["optimizer"]
    model = AlbertClassifier(optimizer=optimizer, learning_rate=learning_rate)

    # Setup Wandb logging
    if cfg.model.wandb_logging == "enabled": # Oh I know. This was a bool, and a good one. But Hydra in Docker doesnt only allow str, float or int.
        wandb_logger = WandbLogger(log_model="all",
                                project="mlops_for_the_win",
                                entity='mlops_for_the_win',
                                )
        wandb_logger.log_hyperparams(cfg)
        wandb_logger.watch(model, log='gradients', log_freq=1)
        wandb_logger.log_metrics({"lr": learning_rate})
    else:
        wandb_logger = None

    checkpoint_callback = ModelCheckpoint(monitor="val_loss", 
                                          dirpath='models/', 
                                          filename='best_model', 
                                          save_top_k=1)
    # Training
    trainer = Trainer(
        fast_dev_run = cfg.model["fast_dev_run"], # Run 1 batch of train, val and test data if true
        default_root_dir=cfg.model["output_dir"],
        max_epochs=cfg.model["epochs"],

        callbacks=[EarlyStopping(monitor="val_loss", mode="min"), checkpoint_callback],

        check_val_every_n_epoch=1, # Evaluate after every epoch
        enable_checkpointing = cfg.model['checkpoints'], # Model checkpoints

        limit_train_batches=cfg.model["limit_train_batches"], # Train at only 20% of the data
        limit_val_batches=cfg.model["limit_val_batches"],
        limit_test_batches = cfg.model["limit_test_batches"],

        num_sanity_val_steps=0, # Do not perform sanity check
        #profiler="simple",
        precision=cfg.model["mixed_precision"], # Drop from float 32 to float 16 precision for memory efficiency
        
        # Gpu if avaialable
        devices=1,
        accelerator="auto",
        logger=wandb_logger,
                    )
    # Fit model
    trainer.fit(model, dm)

    # Evaluation
    trainer.test(model,dm)

# TODO: Move this to testing or..?
@hydra.main(version_base=None, config_path="../conf", config_name="config.yaml")
def data_testing(cfg:DictConfig):
    # Specify tokenizer
    tokenizer =  AutoTokenizer.from_pretrained('albert-base-v1')

    # Get Lit Data Module
    dm = LitDM(cfg, tokenizer =tokenizer)

if __name__ == '__main__':
    train_main()