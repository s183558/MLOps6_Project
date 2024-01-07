from src.data.make_dataset import preprocess_data2
from src.models.model import BertClassifier
from torch.utils.data import DataLoader, TensorDataset
from pytorch_lightning import Trainer

if __name__ == '__main__':

    # Get data
    train_encodings, train_labels, val_encodings, val_labels = preprocess_data2()

    train_dataset = TensorDataset(train_encodings["input_ids"], train_encodings["attention_mask"], train_labels)
    test_dataset = TensorDataset(val_encodings["input_ids"], val_encodings["attention_mask"], val_labels)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=15)
    test_loader = DataLoader(test_dataset, batch_size=32, num_workers=15)

    model = BertClassifier()
    trainer = Trainer(max_epochs=1, log_every_n_steps=None)
    trainer.fit(model, train_loader, test_loader)

