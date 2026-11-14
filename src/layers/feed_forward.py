import torch.nn as nn



class FeedForwardNetwork(nn.Module):
    def __init__(self,embedding_dim,hidden_dim,dropout=0.1):
        super().__init__()

        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim

        # first layer  : Expand dimension
        self.linear1 = nn.Linear(
            embedding_dim,
            hidden_dim
        )


        # activation function 
        self.activation = nn.ReLU()

        # second layer : reduce dimension
        self.linear2 = nn.Linear(
            hidden_dim,
            embedding_dim
        )

        self.dropout = nn.Dropout(p=dropout)

    def forward(self,x):

        # expand dimensions
        output = self.linear1(x)


        # apply non-linearity 
        output = self.activation(output)

        # project back to embedding dimensions
        output = self.linear2(output)


        # apply dropout
        output = self.dropout(output)

        return output

    
