import torch
from torch import nn


class WrenchPredictorModel(nn.Module):
    MATERIAL_PROPERTY_DIM = 9

    def __init__(
        self,
        state_dim,
        wrench_dim,
        chunk_size,
        hidden_dim,
        dim_feedforward,
        dropout=0.1,
    ):
        super().__init__()
        self.wrench_dim = wrench_dim
        self.chunk_size = chunk_size

        input_dim = state_dim + self.MATERIAL_PROPERTY_DIM
        self.output_mlp = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim_feedforward),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, chunk_size * self.wrench_dim),
        )

    def forward(self, state, material_property):
        """
        state: batch, state_dim
        material_property: batch, 9
        """
        batch_size = state.shape[0]
        context = torch.cat([state, material_property], dim=1)
        wrench_hat = self.output_mlp(context)
        wrench_hat = wrench_hat.reshape(batch_size, self.chunk_size, self.wrench_dim)
        return wrench_hat
