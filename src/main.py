import argparse
import logging

import torch
import wandb
from transformers import AutoTokenizer

from src.logging_config import setup_logging
from model_codebert import CodeBertBiEncoder
from src.train import train_model
from src.utils import load_config, load_and_split_spider, save_model, get_device


def main():
    setup_logging("debug")

    logger = logging.getLogger(__name__)

    # Load config file
    parser = argparse.ArgumentParser()
    parser.add_argument("config_file", help="Yaml config file")
    args = parser.parse_args()
    config = load_config(args.config_file)

    wandb.init(
        project="text2sql_codebert",
        config=config,
        job_type="training",
    )

    logger.info("Init execution")

    # Device: GPU or CPU
    device = get_device()
    logger.info(f"Device: {device}")

    # Load parquet
    logger.info("Loading Spider dataset")

    train_df, validation_df, test_df = load_and_split_spider(
        train_path=config["train_path"],
        validation_path=config["validation_path"],
        train_size=config["train_size"],
        validation_size=config["validation_size"],
        test_size=config["test_size"],
        random_state=int(config["random_state"]),
    )

    logger.info(f"Train size: {len(train_df)}")
    logger.info(f"Validation size: {len(validation_df)}")
    logger.info(f"Test size: {len(test_df)}")

    # Tokenizer
    logger.info("Loading tokenizer")

    tokenizer = AutoTokenizer.from_pretrained(config["codebert_model_name"])

    # Model
    logger.info("Loading CodeBERT")

    codebert_model = CodeBertBiEncoder(model_name=config["codebert_model_name"])

    codebert_model.to(device)

    # Train
    logger.info("Training model")

    codebert_model = train_model(
        model=codebert_model,
        tokenizer=tokenizer,
        train_df=train_df,
        val_df=validation_df,
        lr=float(config["lr"]),
        epochs=int(config["epochs"]),
        batch_size=int(config["batch_size"]),
        weight_decay=float(config["weight_decay"]),
        device=device,
    )

    # Save codebert model
    save_model(codebert_model, config["codebert_model_path"])

    logger.info("CodeBERT Model saved")

    wandb.save(config["codebert_model_path"])

    logger.info("Execution finished")


if __name__ == "__main__":
    main()
