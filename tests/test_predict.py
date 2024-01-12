from src.predict_model import predict
import torch

def test_predict():
    # Create dummy data
    dummy_data = ["movie on my house car","my car is on fire", "elephant was on chicken",
            "keyboard mouse string", "sandwich are cool lol", "as my own"]
    print("Started predicting 1 2 3...")
    model_data = predict(dummy_data, config_fname="config_tests.yaml")

    assert len(model_data) == len(dummy_data), "Length of input and output should be the same"
    assert type(model_data) == torch.Tensor, "Output should be a tensor"