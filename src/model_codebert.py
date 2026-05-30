import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import AutoModel


class CodeBertBiEncoder(nn.Module):
    def __init__(self, model_name="microsoft/codebert-base", freeze_layers: int = 6):
        super().__init__()

        self.encoder = AutoModel.from_pretrained(model_name)
        self._freeze_layers(freeze_layers)

    def _freeze_layers(self, n: int):
        for param in self.encoder.embeddings.parameters():
            param.requires_grad = False

        for layer in self.encoder.encoder.layer[:n]:
            for param in layer.parameters():
                param.requires_grad = False

    def encode(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls = outputs.last_hidden_state[
            :, 0
        ]  # CodeBERT CLS token (full query embedding vector)

        return F.normalize(cls, p=2, dim=1)

    def forward(self, q_ids, q_mask, sql_ids, sql_mask):
        q_emb = self.encode(q_ids, q_mask)
        sql_emb = self.encode(sql_ids, sql_mask)

        return q_emb, sql_emb
