from src.train_model import train_main
from omegaconf import DictConfig

#@pytest.mark.parametrize("precision", ["32-true", "16-true"])
def test_train_main():
    # The configurations for the tests
    cfg = DictConfig({
        'data': {
            'test_size': 0.1,
            'valid_size': 0.2,
            'seed': 42,
            'token_max_length': 128,
            'batch_size': 2,
        },
        'model': {
            'output_dir': 'models/',
            'optimizer': 'Adam',
            'lr': 2e-5,
            'epochs': 1,
            'max_steps': 1,

            'limit_train_batches': 0.0006,
            'limit_val_batches': 0.003,
            'limit_test_batches': 0.004,

            'mixed_precision': "32-true",
        }})

    # Call the function
    train_main(cfg)

