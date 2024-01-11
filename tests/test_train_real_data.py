from unittest.mock import patch, MagicMock
from src.train_model import train_main
from transformers import AutoTokenizer
from pytorch_lightning import Trainer
from omegaconf import DictConfig
import os.path
import pytest
from tests import _PATH_DATA

@pytest.mark.skipif(not os.path.exists(_PATH_DATA), reason="Data files not found")
#@pytest.mark.parametrize("precision", ["32-true", "16-true"])
def test_train_main():
    # The configurations for the tests
    cfg = DictConfig({
        'data': {
            'test_size': 0.1,
            'valid_size': 0.2,
            'seed': 42,
            'token_max_length': 128,
            'batch_size': 32,
        },
        'model': {
            'output_dir': 'models/',
            'optimizer': 'Adam',
            'lr': 2e-5,
            'epochs': 2,

            'per_device_train_batch_size': 16,
            'per_device_eval_batch_size': 16,
            'weight_decay': 0.01,

            'limit_train_batches': 0.05,
            'limit_val_batches': 0.05,

            'mixed_precision': "32-true",
        }})

    # Call the function
    train_main(cfg)

    #
