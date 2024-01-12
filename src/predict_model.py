import torch
from src.models.model import AlbertClassifier
from src.data.dataset import LitDM
import glob
import os
from hydra import initialize, compose
from omegaconf import DictConfig
from transformers import AutoTokenizer
from pytorch_lightning import Trainer


def predict(data, config_fname="config.yaml"):
    # Call the function with test config
    with initialize(version_base=None, config_path="../conf"):
        cfg = compose(config_name=config_fname)

    if cfg.model["fast_dev_run"]:
        # Find the latest file in the folder
        model_dir = 'models/lightning_logs/'
        latest_subdir = max(glob.glob(os.path.join(model_dir, '*/')), key=os.path.getmtime)

        # Find the model name in the checkpoint folder and load in into the model
        model_names = [f for f in os.listdir(f"{latest_subdir}/checkpoints/") if os.path.isfile(os.path.join(f"{latest_subdir}/checkpoints/", f))][-1]
        model = AlbertClassifier.load_from_checkpoint(f"{latest_subdir}/checkpoints/{model_names}")
    else:
        model = AlbertClassifier(optimizer=cfg.model["optimizer"], learning_rate=cfg.model["lr"])

    # Specify tokenizer
    tokenizer = AutoTokenizer.from_pretrained('albert-base-v1')

    # Get Lit Data Module
    dm = LitDM(cfg, tokenizer = tokenizer)
    dm.prepare_predict(data)

    # Predict
    trainer = Trainer(inference_mode = True)
    output = trainer.predict(model, dm, return_predictions=True)
    
    # Unpack the tensors from their batches, and argmax to get the predicted class
    prediction_list = []
    for batch in output:
        prediction_list.append(torch.argmax(batch["logits"], dim=1))
    predictions = torch.cat(prediction_list)
    print(f'predictions: {predictions}')
    print(f'yhat: {["catastrophe" if value == 1 else "not catastrophe" for value in predictions ]}')

    return predictions
    
if __name__ == '__main__':
    # Create dummy data
    dummy_data = ["movie on my house car","my car is on fire", "elephant was on chicken",
            "keyboard mouse string", "sandwich are cool lol", "as my own"]
    print("Started predicting 1 2 3...")
    model_data = predict(dummy_data)
    print("Finished predicting.")