import torch
from src.models.model import AlbertClassifier
from src.data.dataset import PredictLitDM
import glob
import os
from pathlib import Path
from transformers import AutoTokenizer
from pytorch_lightning import Trainer

def predict(data : [str], which_model : str = "best") -> torch.Tensor:
    
    # get model
    if which_model == "random":
        # Only specify a model with random weights and biases. We don't need a trained model for unit testing.
        model = AlbertClassifier(optimizer='Adam', learning_rate=1e-4)
    
    elif which_model == "best":
        # Find the best_model file in the folder
        model_dir = '/models/best_model/model.ckpt'
        current_dir = Path(os.getcwd())

        if current_dir.name == 'app':
            model_dir = '../'+model_dir
            

        # # Find the file names in the folder (there should always be only 1 in this folder)
        # file_list = []
        # for _, _, files in os.walk(model_dir):
        #     file_list = files

        # Create the model from the best model file
        model = AlbertClassifier.load_from_checkpoint(model_dir)

    elif which_model == "latest":
        # Find the latest file in the folder
        model_dir = 'models/lightning_logs/'

        # Note: UGLY workaround to allow predict() find the model.
        # main_fastapi.py is called from /app 
        current_dir = Path(os.getcwd())
        if current_dir.name == 'app':
            model_dir = '../models/lightning_logs'

        subdir_list = glob.glob(os.path.join(model_dir, '*/'))

        if not subdir_list:
            raise ValueError(f'No model directories found in: {model_dir}.')

        latest_subdir = max(subdir_list, key=os.path.getmtime)
            
        # Find the model name in the checkpoint folder and load in into the model
        model_names = [f for f in os.listdir(f"{latest_subdir}/checkpoints/") if os.path.isfile(os.path.join(f"{latest_subdir}/checkpoints/", f))][-1]        
        model = AlbertClassifier.load_from_checkpoint(f"{latest_subdir}/checkpoints/{model_names}")

    else:
        raise ValueError(f'which_model should either be "best", "latest" or "random". "{which_model}" is not one of those.')
       

    # Specify tokenizer
    tokenizer = AutoTokenizer.from_pretrained('albert-base-v1')

    # Get Lit Data Module
    dm = PredictLitDM(tokenizer = tokenizer)
    dm.load(data)

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
            "keyboard mouse string", "sandwich are cool lol", "as my own", "Nuclear War", "sleeping", "Shot and killed"]
    
    print("Started predicting 1 2 3...")
    model_data = predict(dummy_data)
    print("Finished predicting.")