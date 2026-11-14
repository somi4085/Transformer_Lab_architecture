import torch
import torch.nn as nn

class PositionalEncoding(nn.Module):
    def __init__(self,embedding_dim,max_sequence_length):
        super().__init__()

        self.embedding_dim = embedding_dim
        if embedding_dim <= 0:
            raise ValueError("embeddings dimension should not be 0")
        self.max_sequence_length = max_sequence_length
        if max_sequence_length <= 0:
            raise ValueError("max sequence length should not be 0")
        pe  = torch.zeros(
            max_sequence_length,
            embedding_dim
        )
        self.position = torch.arange(0,max_sequence_length)
        self.dimension = torch.arange(0,embedding_dim)
        even_dimension = self.dimension[::2]
        self.div_term = torch.exp(
            -even_dimension * torch.log(torch.tensor(10000.0)) 
            / self.embedding_dim
        )
        self.angle = self.position.unsqueeze(1)* self.div_term
        pe[:,0::2] = torch.sin(self.angle)
        pe[:,1::2] = torch.cos(self.angle[:,:pe[:,1::2].shape[1]])
        self.register_buffer("pe", pe)



    


        






        print("Position encoding shape: ",self.pe.shape)

        print("Dimension: ",self.dimension)
        print("position: ",self.position)
        print("PE Shape:", self.pe.shape)
        print("PE Matrix:")
        print(self.pe)
        print("even_dimension: ",even_dimension)
        print("frequency: ",self.div_term)
        print("Angle: ", self.angle) 
        print(self.pe)
        print("Registered Buffer: ", self.pe.shape)
        print("Named Buffer: ",list(self.named_buffers()))
        print("State dict keys: ",self.state_dict().keys())
        print("Trainable parameters: ",list(self.named_parameters())) 


    
    
    def forward(self, token_embedding, position_offset=0):

        if not isinstance(token_embedding, torch.Tensor):
            raise TypeError("token_embedding must be a PyTorch tensor")

        if token_embedding.dim() != 3:
            raise ValueError(
                "token_embedding must be a 3D tensor: "
                "(batch_size, sequence_length, embedding_dim)"
            )

        if token_embedding.size(-1) != self.embedding_dim:
            raise ValueError(
                f"Expected embedding dimension {self.embedding_dim}, "
                f"but received {token_embedding.size(-1)}"
            )

        if not isinstance(position_offset, int) or position_offset < 0:
            raise ValueError("position_offset must be a non-negative integer")

        sequence_length = token_embedding.size(1)
        end_position = position_offset + sequence_length

        if end_position > self.max_sequence_length:
            raise ValueError(
                f"Position range ending at {end_position} exceeds "
                f"maximum allowed length {self.max_sequence_length}"
            )

        positional_encoding = self.pe[position_offset:end_position, :].to(
            device=token_embedding.device,
            dtype=token_embedding.dtype
        )

        return token_embedding + positional_encoding

     



def main():
    embedding_dim = 4
    max_sequence_length = 10
    batch_size = 2
    sequence_length = 5

    positional_encoding = PositionalEncoding(
        embedding_dim=embedding_dim,
        max_sequence_length=max_sequence_length
    )

    # Dummy token embeddings
    token_embeddings = torch.randn(
        batch_size,
        sequence_length,
        embedding_dim
    )

    # Call forward method
    output = positional_encoding(token_embeddings)

    print("Token Embeddings Shape:", token_embeddings.shape)
    print("Positional Encoding Shape:", positional_encoding.pe[:sequence_length].shape)
    print("Final Output Shape:", output.shape)
    print("Final Output:")
    print(output)


if __name__ == "__main__":
    main()