import json
import logging
from pathlib import Path

import torch
import torch.nn.functional as F


class SQLRetriever:
    def __init__(self, model, tokenizer, device, max_length=128):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.max_length = max_length
        self.sql_embeddings = None
        self.sql_queries = None
        self.questions = None

    def build_index(self, df, batch_size=32):
        logger = logging.getLogger(__name__)
        logger.info(f"Building SQL index for {len(df)} queries")

        self.sql_queries = df["query"].tolist()
        self.model.eval()

        embeddings = []

        with torch.no_grad():
            for i in range(0, len(self.sql_queries), batch_size):
                batch_queries = self.sql_queries[i : i + batch_size]

                enc = self.tokenizer(
                    batch_queries,
                    max_length=self.max_length,
                    padding=True,
                    truncation=True,
                    return_tensors="pt",
                )

                batch_emb = self.model.encode(
                    enc["input_ids"].to(self.device),
                    enc["attention_mask"].to(self.device),
                )

                embeddings.append(batch_emb.cpu())

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        self.sql_embeddings = torch.cat(embeddings, dim=0)

        logger.info(f"SQL index built with shape {self.sql_embeddings.shape}")

    def retrieve(self, question: str) -> list[dict]:
        """
        Retrieve SQL query from question
        """
        if self.sql_embeddings is None:
            raise RuntimeError(
                "Index not built. Call build_index() or load_index() first."
            )

        self.model.eval()

        with torch.no_grad():
            enc = self.tokenizer(
                question,
                max_length=self.max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )

            q_emb = self.model.encode(
                enc["input_ids"].to(self.device),
                enc["attention_mask"].to(self.device),
            ).cpu()  # [1, D]

        # Calculatte scores with cosine similarity and select only the first
        scores = F.cosine_similarity(q_emb, self.sql_embeddings, dim=1)
        top_indices = scores.topk(1).indices.tolist()

        return [
            {
                "sql": self.sql_queries[top_indices[0]],
                "score": round(scores[top_indices[0]].item(), 4),
            }
        ]

    def save_index(self, path: str):
        """
        Save index dictionary to json
        """
        # Create models folder if it doesn´t exist
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)

        # Tensor saved separately for safe loading (weights_only=True)
        torch.save(self.sql_embeddings, p)

        with open(p.with_suffix(".json"), "w") as f:
            json.dump({"queries": self.sql_queries, "questions": self.questions}, f)

    def load_index(self, path: str):
        """
        Load index dictionary from json
        """
        p = Path(path)
        self.sql_embeddings = torch.load(p, map_location="cpu", weights_only=True)

        with open(p.with_suffix(".json")) as f:
            data = json.load(f)

        self.sql_queries = data["queries"]
        self.questions = data.get("questions")
