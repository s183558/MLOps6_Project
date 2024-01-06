
from sklearn.model_selection import train_test_split

from src.data.dataset import preprocess_data, load_train_df
from datasets import Dataset, DatasetDict

def get_dataset_dict(cfg):
    # Preprocess(or load) and get data
    try:
        train_df = load_train_df("train")
    except:
        train_df = preprocess_data("train", store=True)

    # Split training data
    train_df, test_df = train_test_split(train_df, test_size=cfg.data.test_size)

    # Build Datasets
    train_dataset = Dataset.from_pandas(train_df[:100])
    test_dataset = Dataset.from_pandas(test_df[:100])
    dataset_dict = DatasetDict({
        'train': train_dataset,
        'test': test_dataset
    })
    return dataset_dict

if __name__=="__main__":  
    preprocess_data("train", store=True)
