import math
import torch.nn as nn
from src.embeddings.token_embedding import TokenEmbedding
from src.embeddings.positional_encoding import PositionalEncoding
class InputEmbeddings(nn.Module):
    def __init__(self, vocab_size, embedding_dim, max_sequence_length, dropout=0.1):
        super().__init__()

        self.embedding_dim = embedding_dim

        # token embedding
        self.token_embedding = TokenEmbedding(
            vocab_size=vocab_size,
            embedding_dim=embedding_dim
        )

        # positional Encoding
        self.positional_encoding = PositionalEncoding(
            embedding_dim=embedding_dim,
            max_sequence_length=max_sequence_length
        )

        # scale
        self.scale = math.sqrt(embedding_dim)

        self.dropout = nn.Dropout(p=dropout)

    
    def forward(self, token_ids, position_offset=0):

        token_embedding = self.token_embedding(token_ids)

        token_embedding = token_embedding * self.scale

        embeddings = self.positional_encoding(
            token_embedding,
            position_offset=position_offset
        )

        embeddings = self.dropout(embeddings)

        return embeddings




            