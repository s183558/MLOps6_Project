from src import preprocess_data
import pandas as pd


def test_create_dataset():
    # Test case 1: Verify that the dataset is created successfully
    dataset = preprocess_data("train", store = False)
    assert len(dataset) > 0, "The dataset is empty"
    assert isinstance(dataset, pd.DataFrame), "The dataset is not a pandas DataFrame"

    # Test case 2: Verify that the dataset contains the expected columns
    expected_columns = ['text', 'labels']
    assert all(col in dataset.keys().to_list() for col in expected_columns), "The dataset does not contain the expected columns"

    # Test case 3: Verify that the dataset has the correct number of rows
    expected_num_rows = 5036
    assert len(dataset) == expected_num_rows, "The dataset does not have the expected number of rows"
