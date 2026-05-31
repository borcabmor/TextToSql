import logging

import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoTokenizer

from src.langchain.agent import TextToSQLAgent
from src.logging_config import setup_logging
from src.model_codebert import CodeBertBiEncoder
from src.retrieve import SQLRetriever
from src.utils import get_device, load_config, load_model_weights

setup_logging("info")
logger = logging.getLogger(__name__)

_agent: TextToSQLAgent | None = None


def get_agent() -> TextToSQLAgent:
    global _agent

    if _agent is None:
        try:
            logger.info("Loading model...")

            config = load_config("config.yaml")
            device = get_device()

            tokenizer = AutoTokenizer.from_pretrained(config["codebert_model_name"])

            model = CodeBertBiEncoder(model_name=config["codebert_model_name"])

            model = load_model_weights(model, config["codebert_model_path"])

            model.to(device)
            model.eval()

            torch.set_grad_enabled(False)

            retriever = SQLRetriever(
                model=model,
                tokenizer=tokenizer,
                device=device,
                max_length=int(config["max_length"]),
            )

            retriever.load_index(config["sql_index_path"])

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

    sql = agent.generate_sql(
        req.question,
        req.connection_string,
    )

    return {"sql": sql}


@app.post("/query")
def query(req: QueryRequest):
    """
    Generate SQL query from NQL and execute SQL query over database
    """
    agent = get_agent()

    return agent.query(
        req.question,
        req.connection_string,
    )
