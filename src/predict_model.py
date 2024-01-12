import torch
from src.models.model import AlbertClassifier
from src.data.dataset import LitDM
import glob
import os
import hydra
from omegaconf import DictConfig
from transformers import AutoTokenizer
from pytorch_lightning import Trainer


@hydra.main(version_base=None, config_path="../conf", config_name="config.yaml")
def predict(cfg:DictConfig):
    # Find the latest file in the folder
    model_dir = 'models/lightning_logs/'
    latest_subdir = max(glob.glob(os.path.join(model_dir, '*/')), key=os.path.getmtime)

    # Find the model name in the checkpoint folder and load in into the model
    model_names = [f for f in os.listdir(f"{latest_subdir}/checkpoints/") if os.path.isfile(os.path.join(f"{latest_subdir}/checkpoints/", f))][-1]
    model = AlbertClassifier.load_from_checkpoint(f"{latest_subdir}/checkpoints/{model_names}")

    # Create dummy data
    data = ["movie on my house car","my car is on fire", "elephant was on chicken", "keyboard mouse string", "sandwich are cool lol", "as my own "]

    # Specify tokenizer
    tokenizer = AutoTokenizer.from_pretrained('albert-base-v1')

    # Get Lit Data Module
    dm = LitDM(cfg, tokenizer = tokenizer)
    dm.prepare_predict(data)

    # Predict
    trainer = Trainer()
    output = trainer.predict(model, dm, return_predictions=True)
    predictions = torch.argmax(output[0]["logits"], dim=1)
    print(f'yhat: {["catastrophe" if value == 1 else "not catastrophe" for value in predictions ]}')

    return predictions
    
if __name__ == '__main__':
    print("Started predicting 1 2 3...")
    model_data = predict()
    print("Finished predicting.")