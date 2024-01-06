import numpy as np
import torch
from data.dataset import LitDM
from transformers import BertForSequenceClassification, AdamW, BertConfig, get_linear_schedule_with_warmup
import time
import datetime
import random

if __name__ == '__main__':
    dm = LitDM()
    dm.setup()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_dataloader = dm.train_dataloader()
    validation_dataloader = dm.val_dataloader()

    print(type(train_dataloader))
