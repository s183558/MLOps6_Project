from data.dataset import preprocess_data
from sklearn.model_selection import train_test_split
from datasets import Dataset, DatasetDict
from transformers import DataCollatorWithPadding, AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from sklearn.metrics import accuracy_score
import numpy as np

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return {"accuracy": accuracy_score(labels, predictions)}

if __name__ == '__main__':
    # Preprocess(or load) and get data
    train_df = preprocess_data("train.csv", store=True)

    # Split training data
    train_df, test_df = train_test_split(train_df, test_size=0.2)

    # Build Datasets
    train_dataset = Dataset.from_pandas(train_df[:100])
    test_dataset = Dataset.from_pandas(test_df[:100])
    dataset_dict = DatasetDict({
        'train': train_dataset,
        'test': test_dataset
    })

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
    output_dir="../models/",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=2,
    weight_decay=0.01,
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









