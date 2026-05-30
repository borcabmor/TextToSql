import logging
from pathlib import Path

import pandas as pd
import torch
import yaml
from sklearn.model_selection import GroupShuffleSplit


def get_device():
    """
    GPU if available, else CPU
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    return device


def get_project_folder() -> Path:
    """
    Root folder of project
    """
    return Path(__file__).resolve().parents[1]


def get_config_folder() -> Path:
    """
    Config folder
    """
    return get_project_folder() / "config"


def load_config(filename: str) -> dict:
    """
    Load yaml config
    """
    logger = logging.getLogger(__name__)

    logger.info("Loading configuration")

    config_path = get_config_folder() / filename

    with open(config_path, "r") as file:
        return yaml.safe_load(file)


def data_preprocess(df: pd.DataFrame) -> pd.DataFrame:
    return_df: pd.DataFrame

    # Remove unused columns
    return_df = df[["db_id", "question", "query"]]

    # Long spaces to 1 space
    return_df["question"] = (
        return_df["question"].replace(r"\s+", " ", regex=True).str.strip()
    )
    return_df["query"] = return_df["query"].replace(r"\s+", " ", regex=True).str.strip()

    # remove rows with none values
    return_df = return_df.dropna()

    return return_df


def load_and_split_spider(
    train_path: str,
    validation_path: str,
    train_size=0.70,
    random_state=42,
):
    """
    Load Spider parquet files and split by db_id.

    Same database cannot appear in different groups.
    """

    logger = logging.getLogger(__name__)

    logger.info("Loading parquet files")

    train_df = pd.read_parquet(train_path)
    validation_df = pd.read_parquet(validation_path)

    # Merge both files
    df = pd.concat([train_df, validation_df], ignore_index=True)

    logger.info(f"Total rows: {len(df)}")

    # Dataset preprocess
    df = data_preprocess(df)

    # --------
    # TRAIN 70%
    # TEST+VAL 30%
    # --------
    splitter1 = GroupShuffleSplit(
        n_splits=1, train_size=train_size, random_state=random_state
    )

    # split between train and rest, grouping by db_id
    train_idx, remaining_idx = next(splitter1.split(df, groups=df["db_id"]))

    train_split = df.iloc[train_idx].reset_index(drop=True)
    remaining = df.iloc[remaining_idx].reset_index(drop=True)

    # --------
    # VAL 15%
    # TEST 15%
    # --------
    splitter2 = GroupShuffleSplit(n_splits=1, train_size=0.5, random_state=random_state)

    # split between validation and test, grouping by db_id
    val_idx, test_idx = next(splitter2.split(remaining, groups=remaining["db_id"]))

    validation_split = remaining.iloc[val_idx].reset_index(drop=True)
    test_split = remaining.iloc[test_idx].reset_index(drop=True)

    logger.info(f"Train: {len(train_split)}")
    logger.info(f"Validation: {len(validation_split)}")
    logger.info(f"Test: {len(test_split)}")

    return (train_split, validation_split, test_split)


def save_model(model, path: str):
    """
    Save torch model
    """
    Path(path).parent.mkdir(parents=True, exist_ok=True)

    torch.save(model.state_dict(), path)


def load_model_weights(model, path: str):
    """
    Load model weights
    """
    model.load_state_dict(torch.load(path, map_location="cpu"))

    return model


def save_data_to_config(param: str, value, config_file: str):
    config = load_config(config_file)

    config[param] = value
    config_path = get_config_folder() / config_file

    with open(config_path, "w") as file:
        yaml.safe_dump(config, file, sort_keys=False)

    logger = logging.getLogger(__name__)

    logger.info(f'Saved param "{param}" with value: {value} in config file.')


def set_max_tokens(
    config_file: str,
    train_df: pd.DataFrame,
    validation_df: pd.DataFrame,
    test_df: pd.DataFrame,
    tokenizer,
):
    """
    Save max_length based on CodeBERT tokenization
    """
    df = pd.concat([train_df, validation_df, test_df], ignore_index=True)

    max_q = max(
        len(tokenizer.encode(q, add_special_tokens=True)) for q in df["question"]
    )

    max_sql = max(
        len(tokenizer.encode(sql, add_special_tokens=True)) for sql in df["query"]
    )

    max_tokens = max(max_q, max_sql)

    save_data_to_config("max_length", int(max_tokens), config_file)
