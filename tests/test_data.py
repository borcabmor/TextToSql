import pandas as pd
import pytest

from src.utils import load_and_split_spider, load_config

config = load_config("config.yaml")

EXPECTED_COLUMNS = ["db_id", "question", "query"]


@pytest.fixture(scope="session")
def dataframe():
    _, _, test_split = load_and_split_spider(
        config["train_path"],
        config["validation_path"],
        config["train_size"],
        config["random_state"],
    )

    return test_split


def test_dataset_not_empty(dataframe):
    assert len(dataframe) > 0


def test_expected_columns(dataframe):
    assert list(dataframe.columns) == EXPECTED_COLUMNS


def test_no_nulls(dataframe):
    assert dataframe.isnull().sum().sum() == 0


def test_all_columns_string(dataframe):
    for column in EXPECTED_COLUMNS:
        assert pd.api.types.is_string_dtype(dataframe[column])
