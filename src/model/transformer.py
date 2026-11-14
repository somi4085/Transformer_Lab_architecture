
import torch
import torch.nn as nn

from src.layers.decoder import TransformerDecoder
from src.layers.encoder import TransformerEncoder


class Transformer(nn.Module):

    def __init__(
        self,
        vocab_size,
        embedding_dim,
        max_sequence_length,
        num_encoder_layers,
        num_decoder_layers,
        num_heads,
        hidden_dim,
        dropout=0.1
    ):
        super().__init__()

        # 1. Transformer Encoder
        self.encoder = TransformerEncoder(
            vocab_size=vocab_size,
            embedding_dim=embedding_dim,
            max_sequence_length=max_sequence_length,
            num_layers=num_encoder_layers,
            num_heads=num_heads,
            hidden_dim=hidden_dim,
            dropout=dropout
        )

        # 2. Transformer Decoder
        self.decoder = TransformerDecoder(
            vocab_size=vocab_size,
            embedding_dim=embedding_dim,
            max_sequence_length=max_sequence_length,
            num_layers=num_decoder_layers,
            num_heads=num_heads,
            hidden_dim=hidden_dim,
            dropout=dropout
        )

        # 3. Output projection: embedding dimension -> vocabulary size
        self.output_projection = nn.Linear(
            embedding_dim,
            vocab_size
        )

    # --------------------------------------------------
    # 1. NORMAL FORWARD
    # --------------------------------------------------

    def forward(
        self,
        source_token_ids,
        target_token_ids,
        source_mask=None,
        target_mask=None
    ):

        encoder_output = self.encoder(
            source_token_ids,
            mask=source_mask
        )

        decoder_output = self.decoder(
            token_ids=target_token_ids,
            encoder_output=encoder_output,
            self_attention_mask=target_mask,
            encoder_mask=source_mask
        )

        logits = self.output_projection(decoder_output)

        return logits

    # --------------------------------------------------
    # 2. ENCODE SOURCE ONLY ONCE
    # --------------------------------------------------

    def encode(
        self,
        source_token_ids,
        source_mask=None
    ):

        encoder_output = self.encoder(
            source_token_ids,
            mask=source_mask
        )

        return encoder_output

    # --------------------------------------------------
    # 3. INCREMENTAL DECODING
    # --------------------------------------------------

    def decode_step(
        self,
        target_token_ids,
        encoder_output,
        past_key_values=None,
        source_mask=None,
        position_offset=0
    ):

        decoder_output, present_key_values = self.decoder(
            token_ids=target_token_ids,
            encoder_output=encoder_output,
            self_attention_mask=None,
            encoder_mask=source_mask,
            past_key_values=past_key_values,
            use_cache=True,
            position_offset=position_offset
        )

        logits = self.output_projection(decoder_output)

        return logits, present_key_values

    # --------------------------------------------------
    # 4. KV CACHE AUTOREGRESSIVE GENERATION
    # --------------------------------------------------

    @torch.inference_mode()
    def generate(
        self,
        source_token_ids,
        source_mask=None,
        bos_token_id=2,
        eos_token_id=3,
        max_new_tokens=50
    ):

        if max_new_tokens < 0:
            raise ValueError(
                "max_new_tokens must be non-negative"
            )

        if max_new_tokens > self.decoder.input_embeddings.positional_encoding.max_sequence_length:
            raise ValueError(
                "max_new_tokens exceeds decoder positional encoding capacity"
            )

        was_training = self.training

        self.eval()

        try:

            # Step 1: Encode source only once
            encoder_output = self.encode(
                source_token_ids,
                source_mask=source_mask
            )

            # Step 2: Initialize decoder with BOS
            generated_ids = torch.full(
                (source_token_ids.size(0), 1),
                bos_token_id,
                dtype=torch.long,
                device=source_token_ids.device
            )

            # Step 3: Initialize empty KV Cache
            past_key_values = None

            # Step 4: Generate tokens incrementally
            for step in range(max_new_tokens):

                current_token = generated_ids[:, -1:]

                # Decode only the latest token
                logits, past_key_values = self.decode_step(
                    target_token_ids=current_token,
                    encoder_output=encoder_output,
                    past_key_values=past_key_values,
                    source_mask=source_mask,
                    position_offset=step
                )

                # Step 5: Greedy token selection
                next_token = logits[:, -1, :].argmax(dim=-1)

                # Step 6: Append predicted token
                generated_ids = torch.cat(
                    [
                        generated_ids,
                        next_token.unsqueeze(1)
                    ],
                    dim=1
                )

                # Step 7: Stop when all sequences generate EOS
                if torch.all(next_token == eos_token_id):
                    break

            return generated_ids

        finally:

            # Restore original model mode
            self.train(was_training)
