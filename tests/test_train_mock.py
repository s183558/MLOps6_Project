# File made with GitHub Copilot
from unittest.mock import patch, MagicMock
from src.train_model import train_main
from transformers import AutoTokenizer
from pytorch_lightning import Trainer
from omegaconf import DictConfig

def test_train_main():
    # Mock the configuration
    cfg = DictConfig({
        'model': {
            'lr': 0.001,
            'optimizer': 'Adam',
            'epochs': 100,
            'limit_train_batches': 1,
            'limit_val_batches': 1,
            'mixed_precision': 32
        }
    })

    # Mock the tokenizer, data module, model, and trainer
    with patch.object(AutoTokenizer, 'from_pretrained') as mock_tokenizer, \
         patch('src.train_model.LitDM') as mock_dm, \
         patch('src.train_model.AlbertClassifier') as mock_model, \
         patch.object(Trainer, '__init__') as mock_trainer_init, \
         patch.object(Trainer, 'fit') as mock_trainer_fit, \
         patch.object(Trainer, 'test') as mock_trainer_test:

        # Set the return values of the mocks
        mock_tokenizer.return_value = MagicMock()
        mock_dm.return_value = MagicMock()
        mock_model.return_value = MagicMock()
        mock_trainer_init.return_value = None

        # Call the function
        train_main(cfg)

        # Verify that the mocks were called with the correct arguments
        mock_tokenizer.assert_called_once_with('albert-base-v1')
        mock_dm.assert_called_once_with(cfg, tokenizer=mock_tokenizer.return_value)
        mock_model.assert_called_once_with(optimizer=cfg.model["optimizer"], learning_rate=cfg.model["lr"])
        mock_trainer_init.assert_called_once()
        mock_trainer_fit.assert_called_once_with(mock_model.return_value, mock_dm.return_value)
        mock_trainer_test.assert_called_once_with(mock_model.return_value, mock_dm.return_value)