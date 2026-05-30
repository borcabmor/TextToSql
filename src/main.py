import argparse
import logging

from dotenv import load_dotenv

load_dotenv()

import pandas as pd
from transformers import AutoTokenizer

import wandb
from src.dataset import SpiderDataset
from src.logging_config import setup_logging
from src.model_codebert import CodeBertBiEncoder
from src.retrieve import SQLRetriever
from src.train import evaluate_model, train_model
from src.utils import (
    get_device,
    get_project_folder,
    load_and_split_spider,
    load_config,
    save_model,
    set_max_tokens,
)


def main():
    setup_logging("debug")
    logger = logging.getLogger(__name__)

    # Loa dconfig file
    parser = argparse.ArgumentParser()
    parser.add_argument("config_file", help="Yaml config file")
    args = parser.parse_args()
    config = load_config(args.config_file)

    # Initialize W&B with config file params

    logger.info("Init execution")

    device = get_device()
    logger.info(f"Device: {device}")

    logger.info("Loading Spider dataset")

    # Split datasets for train, validation and test
    train_df, validation_df, test_df = load_and_split_spider(
        train_path=config["train_path"],
        validation_path=config["validation_path"],
        train_size=config["train_size"],
        random_state=int(config["random_state"]),
    )

    logger.info(
        f"Train: {len(train_df)} | Val: {len(validation_df)} | Test: {len(test_df)}"
    )

    # Load tokenizewr from CodeBERT model
    logger.info("Loading tokenizer")

    tokenizer = AutoTokenizer.from_pretrained(config["codebert_model_name"])

    # Calculate max_length from max length between token columns of the datasets
    set_max_tokens(args.config_file, train_df, validation_df, test_df, tokenizer)

    # reload config with data changed
    config = load_config(args.config_file)

    # Init WandB
    wandb.init(
        project="text2sql_codebert",
        config=config,
        job_type="training",
    )

    train_dataset = SpiderDataset(train_df, tokenizer, max_length=config["max_length"])

    val_dataset = SpiderDataset(
        validation_df, tokenizer, max_length=config["max_length"]
    )

    test_dataset = SpiderDataset(test_df, tokenizer, max_length=config["max_length"])

    # Load CodeBERT model
    logger.info("Loading CodeBERT model")

    model = CodeBertBiEncoder(
        model_name=config["codebert_model_name"],
        freeze_layers=int(config["freeze_layers"]),
    )

    model.to(device)

    # Train model
    logger.info("Training model")

    model = train_model(
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        lr=float(config["lr"]),
        epochs=int(config["epochs"]),
        batch_size=int(config["batch_size"]),
        weight_decay=float(config["weight_decay"]),
        device=device,
        patience=int(config["patience"]),
    )

    # Evaluate on test dataset
    logger.info("Evaluating on test dataset")

    test_loss, test_acc = evaluate_model(
        model, test_dataset, device, int(config["batch_size"])
    )

    wandb.log({"test_loss": test_loss, "test_acc": test_acc})

    logger.info(f"Test | Loss: {test_loss:.4f} | Accuracy: {test_acc:.4f}")

    # Save model
    save_model(model, config["codebert_model_path"])
    logger.info("CodeBERT model saved")

    model_artifact = wandb.Artifact(
        name="trained_model", type="model", description="Trained model"
    )

    model_file = get_project_folder() / config["codebert_model_path"]
    model_artifact.add_file(model_file)
    wandb.log_artifact(model_artifact)

    # Build embedding index with full dataset because train and test are already finished
    retriever = SQLRetriever(
        model=model,
        tokenizer=tokenizer,
        device=device,
        max_length=config["max_length"],
    )

    full_df = pd.concat([train_df, validation_df, test_df], ignore_index=True)
    retriever.build_index(full_df, int(config["batch_size"]))
    retriever.save_index(config["sql_index_path"])
    logger.info("SQL index saved")

    logger.info("Execution finished")


if __name__ == "__main__":
    main()
