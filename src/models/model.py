import torch
from transformers import BertForSequenceClassification
import pytorch_lightning as pl

class BertClassifier(pl.LightningModule):
    def __init__(self, learning_rate=2e-5):
        super().__init__()
        self.model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)
        self.learning_rate = learning_rate

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
        
        return output.loss

    def validation_step(self, batch, batch_idx):
        ids, masks, labels = batch[0], batch[1], batch[2]
        output = self.model(
            ids,
            attention_mask=masks,
            labels=labels,
        )

        return output.loss


    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)