from unittest.mock import patch, MagicMock
from src.train_model import train_main
from transformers import AutoTokenizer
from pytorch_lightning import Trainer
from omegaconf import DictConfig
from tests import _PATH_DATA
import pytest
import os

@pytest.mark.skipif(not os.path.exists(_PATH_DATA), reason="Data files not found")
@pytest.mark.parametrize("precision", ["32-true", "16-true"])
def test_train_main():
    # The configurations for the tests
    cfg = DictConfig({
        'model': {
            'lr': 0.001,
            'optimizer': 'Adam',
            'epochs': 2,
            'limit_train_batches': 0.2,
            'limit_val_batches': 0.2,
            'mixed_precision': 32
        }
    })

    # Call the function
    train_main(cfg)

    #
    