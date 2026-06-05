import logging

import torch
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoTokenizer

load_dotenv()

from src.langchain.agent import TextToSQLAgent
from src.logging_config import setup_logging
from src.model_codebert import CodeBertBiEncoder
from src.retrieve import SQLRetriever
from src.utils import (
    ensure_index_exists,
    ensure_model_exists,
    get_device,
    load_config,
    load_model_weights,
)

setup_logging("info", "app_inference.log")
logger = logging.getLogger(__name__)

_agent: TextToSQLAgent | None = None


def get_agent() -> TextToSQLAgent:
    global _agent

    if _agent is None:
        try:
            logger.info("Loading model...")

            # Load config and set if using CPU or GPU
            config = load_config("config.yaml")
            device = get_device()

            model_path = config["codebert_model_path"]
            index_path = config["sql_index_path"]

            # Download artifacts (model and sql index) from W&B if doesnt exist in local
            ensure_model_exists(model_path)
            ensure_index_exists(index_path)

            tokenizer = AutoTokenizer.from_pretrained(config["codebert_model_name"])

            model = CodeBertBiEncoder(model_name=config["codebert_model_name"])

            model = load_model_weights(
                model,
                model_path,
            )

            model.to(device)
            model.eval()

            torch.set_grad_enabled(False)

            # Initialize retriever
            retriever = SQLRetriever(
                model=model,
                tokenizer=tokenizer,
                device=device,
                max_length=int(config["max_length"]),
            )

            # Load embeddings index
            retriever.load_index(index_path)

            # Initialize LangChain agent
            _agent = TextToSQLAgent(
                retriever=retriever,
                llm_model=config["llm_model"],
            )

            logger.info("Model loaded")

        except Exception:
            logger.exception("Model loading failed")

            raise HTTPException(
                status_code=503,
                detail="Model loading failed",
            )

    return _agent


app = FastAPI(title="Text-to-SQL API")


class GenerateRequest(BaseModel):
    question: str
    connection_string: str


class QueryRequest(BaseModel):
    question: str
    connection_string: str


@app.get("/running")
def health():
    """
    Check API is running
    """

    return {"status": "ok"}


@app.post("/generate")
def generate(req: GenerateRequest):
    """
    Generate SQL query from NQL
    """
    agent = get_agent()

    state = {
        "question": req.question,
        "connection_string": req.connection_string,
        "schema": None,
        "sql": None,
        "result": None,
    }

    state["schema"] = agent.load_schema_node(state)
    sql_query = agent.generate_sql(state)

    return {"sql": " ".join(sql_query.split())}


@app.post("/query")
def query(req: QueryRequest):
    """
    Generate SQL query from NQL and execute SQL query over database
    """

    agent = get_agent()

    state = {
        "question": req.question,
        "connection_string": req.connection_string,
        "schema": None,
        "sql": None,
        "result": None,
    }

    state["schema"] = agent.load_schema_node(state)

    result = agent.query(state)

    return {
        "sql": " ".join(result["sql"].split()),
        "result": result["result"],
    }
