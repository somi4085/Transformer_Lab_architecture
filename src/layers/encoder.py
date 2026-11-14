import torch
import torch.nn as nn
from src.layers.encoder_stack import EncoderStack
from src.embeddings.input_embedding import InputEmbeddings
class TransformerEncoder(nn.Module):
    def __init__(self, vocab_size,embedding_dim,max_sequence_length,num_layers, num_heads,hidden_dim,dropout=0.1):
        super().__init__()

        self.input_embedding = InputEmbeddings(
            vocab_size=vocab_size,
            embedding_dim=embedding_dim,
            max_sequence_length=max_sequence_length,
            dropout=0.1
        )

        self.encoder_stack = EncoderStack(
            num_layers=num_layers,
            embedding_dim=embedding_dim,
            num_heads=num_heads,
            hidden_dim=hidden_dim,
            dropout=0.1
        )

    def forward(self,token_ids,mask=None):
         # Step 1: Convert token IDs into embeddings
        
        embeddings = self.input_embedding(token_ids)

        # step 2 - process embedding through encoder stack'

        encoder_output = self.encoder_stack(
            embeddings,
            mask = mask
        )
        return encoder_output


