from unittest.mock import MagicMock, patch

import pandas as pd
import torch

from src.dataset import SpiderDataset
from src.model_codebert import CodeBertBiEncoder
from src.train import evaluate_model, train_model
from src.utils import data_preprocess, get_device, save_data_to_config


class DummyTokenizer:
    def __call__(
        self,
        text,
        max_length,
        padding,
        truncation,
        return_tensors,
    ):
        return {
            "input_ids": torch.ones((1, max_length), dtype=torch.long),
            "attention_mask": torch.ones((1, max_length), dtype=torch.long),
        }


class DummyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(8, 8)

    def forward(self, q_ids, q_mask, sql_ids, sql_mask):
        batch_size = q_ids.shape[0]

        q = torch.randn(batch_size, 8)
        s = torch.randn(batch_size, 8)

        q_emb = self.linear(q)
        sql_emb = self.linear(s)

        return q_emb, sql_emb


def build_dataset():
    df = pd.DataFrame(
        {
            "question": ["q1", "q2", "q3", "q4"],
            "query": ["sql1", "sql2", "sql3", "sql4"],
        }
    )

    return SpiderDataset(df, DummyTokenizer(), max_length=8)


# --------------------
# DATASET TESTS
# --------------------
def test_dataset_len():
    dataset = build_dataset()

    assert len(dataset) == 4


def test_dataset_getitem():
    dataset = build_dataset()

    item = dataset[0]

    assert isinstance(item["question_input_ids"], torch.Tensor)
    assert isinstance(item["sql_input_ids"], torch.Tensor)

    assert item["question_input_ids"].shape == (8,)
    assert item["sql_input_ids"].shape == (8,)


# --------------------
# UTILS TESTS
# --------------------
def test_get_device():
    device = get_device()

    assert str(device) in ["cpu", "cuda"]


def test_data_preprocess():
    df = pd.DataFrame(
        {
            "db_id": ["db1"],
            "question": ["   how    many users   "],
            "query": [" SELECT   *   FROM users "],
            "unused": [123],
        }
    )

    result = data_preprocess(df)

    assert list(result.columns) == ["db_id", "question", "query"]
    assert result.iloc[0]["question"] == "how many users"
    assert result.iloc[0]["query"] == "SELECT * FROM users"


@patch("src.utils.load_config")
@patch("yaml.safe_dump")
def test_save_data_to_config(mock_dump, mock_load):
    mock_load.return_value = {"a": 1}

    with patch("builtins.open"):
        save_data_to_config(
            "max_length",
            128,
            "config.yaml",
        )

    mock_dump.assert_called_once()


# --------------------
# MODEL TESTS
# --------------------
@patch("src.model_codebert.AutoModel.from_pretrained")
def test_model_forward_shapes(mock_pretrained):
    dummy_encoder = MagicMock()

    dummy_output = MagicMock()
    dummy_output.last_hidden_state = torch.rand(2, 10, 768)

    dummy_encoder.return_value = dummy_output
    dummy_encoder.embeddings.parameters.return_value = []
    dummy_encoder.encoder.layer = []

    mock_pretrained.return_value = dummy_encoder

    model = CodeBertBiEncoder()

    q_ids = torch.randint(0, 10, (2, 10))
    q_mask = torch.ones((2, 10))
    sql_ids = torch.randint(0, 10, (2, 10))
    sql_mask = torch.ones((2, 10))

    q_emb, sql_emb = model(
        q_ids,
        q_mask,
        sql_ids,
        sql_mask,
    )

    assert q_emb.shape == (2, 768)
    assert sql_emb.shape == (2, 768)


@patch("src.model_codebert.AutoModel.from_pretrained")
def test_model_output_normalized(mock_pretrained):
    dummy_encoder = MagicMock()

    dummy_output = MagicMock()
    dummy_output.last_hidden_state = torch.rand(1, 5, 768)

    dummy_encoder.return_value = dummy_output
    dummy_encoder.embeddings.parameters.return_value = []
    dummy_encoder.encoder.layer = []

    mock_pretrained.return_value = dummy_encoder

    model = CodeBertBiEncoder()

    emb = model.encode(
        torch.randint(0, 10, (1, 5)),
        torch.ones((1, 5)),
    )

    norm = torch.norm(emb, dim=1)

    assert torch.allclose(
        norm,
        torch.tensor([1.0]),
        atol=1e-5,
    )


# --------------------
# TRAIN TESTS
# --------------------
@patch("src.train.wandb.log")
def test_evaluate_model_returns_metrics(mock_wandb):
    dataset = build_dataset()
    model = DummyModel()

    loss, acc = evaluate_model(
        model,
        dataset,
        device="cpu",
        batch_size=2,
    )

    assert isinstance(loss, float)
    assert isinstance(acc, float)


@patch("src.train.wandb.log")
def test_train_model_returns_model(mock_wandb):
    dataset = build_dataset()
    model = DummyModel()

    trained_model = train_model(
        model=model,
        train_dataset=dataset,
        val_dataset=dataset,
        epochs=1,
        batch_size=2,
        device="cpu",
    )

    assert trained_model is not None


@patch("src.train.wandb.log")
def test_train_model_logs_metrics(mock_wandb):
    dataset = build_dataset()
    model = DummyModel()

    train_model(
        model=model,
        train_dataset=dataset,
        val_dataset=dataset,
        epochs=1,
        batch_size=2,
        device="cpu",
    )

    mock_wandb.assert_called()
