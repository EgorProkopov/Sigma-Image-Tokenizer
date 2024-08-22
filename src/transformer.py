import torch
import torch.nn as nn


class TransformerEncoderModel(nn.Module):
    def __init__(self, input_dim, model_dim, num_heads, num_encoder_layers, dim_feedforward, dropout=0.1):
        super(TransformerEncoderModel, self).__init__()

        self.positional_encoding = nn.Parameter(torch.randn(1, input_dim + 1, model_dim))
        self.embedding = nn.Embedding(input_dim, model_dim)

        encoder_layer = nn.TransformerEncoderLayer(
                d_model=model_dim,
                nhead=num_heads,
                dim_feedforward=dim_feedforward,
                dropout=dropout
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)
        self.fc_out = nn.Linear(model_dim, input_dim)

    def forward(self, x):
        device = x.device
        x = x + self.positional_encoding.to(device)
        out = self.transformer_encoder(x)
        out = self.fc_out(out)
        return out

