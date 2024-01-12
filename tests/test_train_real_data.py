from src.train_model import train_main
from hydra import initialize, compose

#@pytest.mark.parametrize("precision", ["32-true", "16-true"])
def test_train_main():
    # Call the function with test config
        with initialize(version_base=None, config_path="../conf"):
            cfg = compose(config_name="config_tests.yaml")
            train_main(cfg)

