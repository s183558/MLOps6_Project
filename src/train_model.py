
from transformers import DataCollatorWithPadding, AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from sklearn.metrics import accuracy_score
import numpy as np
from hydra import compose, initialize
from src.data.dataset import get_dataset_dict
import logging
import src.common.log_config  
logger=logging.getLogger(__name__)

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return {"accuracy": accuracy_score(labels, predictions)}

if __name__ == '__main__':
    # load hydra configs
    with initialize(config_path='../conf', version_base="1.1"):
        cfg: dict = compose(config_name='config.yaml')
    
    dataset_dict = get_dataset_dict(cfg)

    # Tokenize
    tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
    def preprocess_function(examples):
        return tokenizer(examples["text"], truncation=True)
    tokenized_data = dataset_dict.map(preprocess_function)

    # Padding by batch
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # Model
    id2label = {0: "NO DISASTER", 1: "DISASTER"}
    label2id = {"NO DISASTER": 0, "DISASTER": 1}
    model = AutoModelForSequenceClassification.from_pretrained(
        "bert-base-uncased", num_labels=2, id2label=id2label, label2id=label2id
    )

    # Training arguments
    training_args = TrainingArguments(
                    output_dir=cfg.model["output_dir"],
                    learning_rate=cfg.model["lr"],
                    per_device_train_batch_size=cfg.model["per_device_train_batch_size"],
                    per_device_eval_batch_size=cfg.model["per_device_eval_batch_size"],
                    num_train_epochs=cfg.model["epochs"],
                    weight_decay=cfg.model["weight_decay"],

                    evaluation_strategy="epoch",
                    save_strategy="epoch",
                    logging_strategy="steps",  # Log every X steps
                    logging_steps=1,
                    load_best_model_at_end=True,
                    push_to_hub=False,
                                    )

    # Trainer
    trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=tokenized_data["train"],
                eval_dataset=tokenized_data["test"],
                tokenizer=tokenizer, 
                data_collator=data_collator,
                compute_metrics=compute_metrics,
                        )

    trainer.train()









