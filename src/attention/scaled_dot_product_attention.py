import torch
import torch.nn as nn
import math


class ScaledDotProductAttention(nn.Module):
    def __init__(self, dropout=0.1):
        super().__init__()

        self.dropout = nn.Dropout(p=dropout)


    def forward(self,query,key,value, mask=None, causal=False):
        key_transposed = key.transpose(-2,-1)

        # attenstion score 
        attention_score = torch.matmul(
            query,
            key_transposed
        )

        # scaled attention score
        scaled_attention_score = attention_score / math.sqrt(query.size(-1))

        # applying padding mask
        if mask is not None:
            if mask.dim() == 2:
                mask = mask.unsqueeze(1)
            scaled_attention_score = scaled_attention_score.masked_fill(
                mask == 0,
                float("-inf")
            )

        if causal:
            sequence_length = query.size(-2)

            causal_mask = torch.tril(
                torch.ones(
                    sequence_length,
                    sequence_length,
                    device=query.device,
                    dtype=torch.bool
                )
            )

            scaled_attention_score = scaled_attention_score.masked_fill(
                ~causal_mask,
                float("-inf")
            )

        # converting scaled raw score into attention weights

        attention_weight = torch.softmax(
            scaled_attention_score, # normalized scaled score
            dim=-1  
        )

        attention_weight = self.dropout(attention_weight)

        attention_output = torch.matmul(
            attention_weight,
            value
        )

        return attention_output




def main():

    # 1. Create attention layer
    attention_layer = ScaledDotProductAttention()

    # 2. Create Query, Key and Value tensors
    query = torch.randn(1, 4, 2)
    key = torch.randn(1, 4, 2)
    value = torch.randn(1, 4, 2)

    # 3. Create padding mask
    padding_mask = torch.tensor([[1, 1, 0, 0]])

    # 4. Pass tensors through attention layer
    output1 = attention_layer(
        query,
        key,
        value,
        mask=padding_mask
    )
    output2 = attention_layer(
    query,
    key,
    value,
    causal=True
)

    print("Output 1 shape:", output1.shape)
    print("output 2 shape: ", output2.shape)


if __name__ == "__main__":
    main()