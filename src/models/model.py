import torch
from transformers import BertForSequenceClassification, AlbertForSequenceClassification
import pytorch_lightning as pl
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AlbertClassifier(pl.LightningModule):
    def __init__(self, optimizer="Adam", learning_rate=2e-5):
        super().__init__()
        self.model = AlbertForSequenceClassification.from_pretrained("albert-base-v1", num_labels=2)
        self.learning_rate = learning_rate
        self.optimizer = optimizer
        self.save_hyperparameters()

    def forward(self, batch):
        ids, masks = batch[0], batch[1]
        return self.model(ids, attention_mask=masks)

    def training_step(self, batch, batch_idx):
        ids, masks, labels = batch[0], batch[1], batch[2]
        output = self.model(
            ids,
            attention_mask=masks,
            labels=labels,
        )
        
        # Log loss and metric
        logger.info(f"Training Step: {batch_idx}, Loss: {output.loss}")
        self.log("train_loss", output.loss)
        return output.loss  

    def validation_step(self, batch, batch_idx):
        ids, masks, labels = batch[0], batch[1], batch[2]
        output = self.model(
            ids,
            attention_mask=masks,
            labels=labels,
        )

        preds = torch.argmax(output.logits, dim=1)
        correct = (preds == labels).sum()
        accuracy = correct / len(labels)
        logger.info(f"Validation Step: {batch_idx}, Loss: {output.loss}, Accuracy: {accuracy}")

        self.log("val_loss", output.loss)
        self.log("val_accuracy", accuracy)


    def test_step(self, batch, batch_idx):
        ids, masks, labels = batch[0], batch[1], batch[2]
        output = self.model(
            ids,
            attention_mask=masks,
            labels=labels,
        )
        preds = torch.argmax(output.logits, dim=1)
        correct = (preds == labels).sum()
        accuracy = correct / len(labels)

        self.log("test_acc", accuracy)

    def predict_step(self, batch, batch_idx):
        return self(batch)

    def configure_optimizers(self):
        if self.optimizer == "Adam":
            return torch.optim.Adam(self.parameters(), lr=self.learning_rate)