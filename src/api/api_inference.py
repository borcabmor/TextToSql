import logging
from contextlib import asynccontextmanager

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


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _agent

    config = load_config("config.yaml")
    device = get_device()

    # Load weights
    tokenizer = AutoTokenizer.from_pretrained(config["codebert_model_name"])
    model = CodeBertBiEncoder(model_name=config["codebert_model_name"])
    model = load_model_weights(model, config["codebert_model_path"])
    model.to(device)
    model.eval()

    # Load index dictionary
    retriever = SQLRetriever(
        model=model,
        tokenizer=tokenizer,
        device=device,
        max_length=int(config["max_length"]),
    )

    retriever.load_index(config["sql_index_path"])

    # Create langchain agent
    _agent = TextToSQLAgent(retriever=retriever, llm_model=config["llm_model"])

    # API ready
    logger.info("API ready")

    yield


app = FastAPI(title="Text-to-SQL API", lifespan=lifespan)


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


_MODEL_NOT_LOADED = "Model not loaded"
_503 = {503: {"description": _MODEL_NOT_LOADED}}


@app.post("/generate", responses=_503)
def generate(req: GenerateRequest):
    """
    Generate SQL query from NQL
    """

    if _agent is None:
        raise HTTPException(status_code=503, detail=_MODEL_NOT_LOADED)

    sql = _agent.generate_sql(req.question, req.connection_string)

    return {"sql": sql}


@app.post("/query", responses=_503)
def query(req: QueryRequest):
    """
    Generate SQL query from NQL and execute SQL query over database
    """

    if _agent is None:
        raise HTTPException(status_code=503, detail=_MODEL_NOT_LOADED)

    return _agent.query(req.question, req.connection_string)
