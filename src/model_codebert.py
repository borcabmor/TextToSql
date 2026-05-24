import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import (
    AutoModel,
    AutoTokenizer,
)


class CodeBertBiEncoder(nn.Module):

    def __init__(self, model_name="microsoft/codebert-base"):
        super().__init__()

        self.encoder = AutoModel.from_pretrained(model_name)

    def encode(self, input_ids, attention_mask):
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        cls = outputs.last_hidden_state[
            :, 0
        ]  # CodeBERT CLS token (full query embedding vector)

        return F.normalize(
            cls,
            p=2,
            dim=1,
        )

    def forward(self, q_ids, q_mask, sql_ids, sql_mask):
        q_emb = self.encode(
            q_ids,
            q_mask,
        )

        sql_emb = self.encode(
            sql_ids,
            sql_mask,
        )

        return q_emb, sql_emb
