import torch.nn as nn


class CodeBertModel(nn.Module):
    """
    Modelo CodeBERT biencoder para selección de consultas SQL más adecuadas a la consulta NQL aportada
    """

    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()

        latent_dim = hidden_dim // 2

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim),
            nn.ReLU(),
        )

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
        )

    def forward(self, x):
        z = self.encoder(x)
        out = self.decoder(z)

        return out
