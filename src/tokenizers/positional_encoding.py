import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    def __init__(self, embedding_dim):
        super().__init__()
        self.embedding_dim = embedding_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.shape[1]

        # Create a position tensor [0, 1, 2, ..., seq_len-1]
        position = torch.arange(seq_len, dtype=torch.float, device=x.device).unsqueeze(1)  # Shape: [seq_len, 1]

        # Compute sinusoidal frequency values: (1 / 10000^(2i/d_model))
        div_term = torch.exp(torch.arange(0, self.embedding_dim, 2, device=x.device).float() *
                             (-torch.log(torch.tensor(10000.0, device=x.device)) / self.embedding_dim))

        # Apply sine to even indices, cosine to odd indices
        pe = torch.zeros(seq_len, self.embedding_dim, device=x.device)
        pe[:, 0::2] = torch.sin(position * div_term)  # Even indices
        pe[:, 1::2] = torch.cos(position * div_term)  # Odd indices

        # Add positional encoding to input
        return x + pe.unsqueeze(0)  # Shape: [batch_size, seq_len, embedding_dim]