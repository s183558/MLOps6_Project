import torch

def load_model():
    with open('/models/dummy.txt', 'r') as f:
        model_data = f.read()
    # Process model_data as needed
    return model_data

def predict(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader
) -> None:
    """Run prediction for a given model and dataloader.
    
    Args:
        model: model to use for prediction
        dataloader: dataloader with batches
    
    Returns
        Tensor of shape [N, d] where N is the number of samples and d is the output dimension of the model

    """
    return torch.cat([model(batch) for batch in dataloader], 0)

    
if __name__ == '__main__':
    print("Started predicting 1 2 3...")
    model_data = load_model()
    print(model_data)
    print("Finished predicting.")