
import pickle
from pathlib import Path

import torch

from src.model.transformer import Transformer

class InferenceService:

    def __init__(
        self,
        checkpoint_path="checkpoints/tinystories_smoke/best.pt",
        tokenizer_path="artifacts/tokenizer.pkl",
        device="cpu"
    ):

        self.checkpoint_path = Path(checkpoint_path)
        self.tokenizer_path = Path(tokenizer_path)

        self.device = torch.device(device)

        self.model = None
        self.tokenizer = None

        self.initialized = False

        # Model configuration used during smoke training.
        self.model_config = {
            "vocab_size": 266,
            "embedding_dim": 64,
            "max_sequence_length": 64,
            "num_encoder_layers": 2,
            "num_decoder_layers": 2,
            "num_heads": 4,
            "hidden_dim": 256,
            "dropout": 0.0
        }


    
    # Load tokenizer
    

    def load_tokenizer(self):

        if not self.tokenizer_path.exists():
            raise FileNotFoundError(
                f"Tokenizer not found: {self.tokenizer_path}"
            )

        with open(self.tokenizer_path, "rb") as file:
            self.tokenizer = pickle.load(file)

        return self.tokenizer

    # Load model

    def load_model(self):

        if not self.checkpoint_path.exists():
            raise FileNotFoundError(
                f"Checkpoint not found: {self.checkpoint_path}"
            )

        model = Transformer(**self.model_config)

        checkpoint = torch.load(
            self.checkpoint_path,
            map_location=self.device,
            weights_only=False
        )

        if isinstance(checkpoint, dict):

            if "model_state_dict" in checkpoint:
                state_dict = checkpoint["model_state_dict"]

            elif "state_dict" in checkpoint:
                state_dict = checkpoint["state_dict"]

            else:
                state_dict = checkpoint

        else:
            state_dict = checkpoint

        model.load_state_dict(state_dict)

        model.to(self.device)

        model.eval()

        self.model = model

        return self.model

    # Initialize service
    

    def initialize(self):

        if self.initialized:
            return self

        self.load_tokenizer()
        self.load_model()

        self.initialized = True

        print("Inference service initialized.")
        print("Device:", self.device)
        print("Model:", type(self.model).__name__)
        print("Tokenizer:", type(self.tokenizer).__name__)

        return self

    # Generate text

    @torch.inference_mode()
    def generate(
        self,
        text,
        max_new_tokens=50
    ):

        if not self.initialized:
            raise RuntimeError(
                "InferenceService is not initialized. "
                "Call initialize() first."
            )

        if not isinstance(text, str) or not text.strip():
            raise ValueError(
                "Input text must be a non-empty string."
            )

        if not isinstance(max_new_tokens, int) or isinstance(
            max_new_tokens, bool
        ):
            raise ValueError(
                "max_new_tokens must be an integer."
            )

        if max_new_tokens < 1:
            raise ValueError(
                "max_new_tokens must be at least 1."
            )

        if max_new_tokens > self.model_config["max_sequence_length"] - 1:
            raise ValueError(
                "max_new_tokens exceeds the decoder positional capacity."
            )

        # Tokenize source text.
        source_ids = self.tokenizer.encode_to_ids(
            text,
            add_special_tokens=True
        )

        if not source_ids:
            raise ValueError("Tokenizer returned an empty sequence.")

        if len(source_ids) > self.model_config["max_sequence_length"]:
            raise ValueError(
                "Input exceeds the encoder sequence length limit."
            )

        source_token_ids = torch.tensor(
            [source_ids],
            dtype=torch.long,
            device=self.device
        )

        source_mask = torch.ones_like(source_token_ids)

        # Encode source once.
        encoder_output = self.model.encode(
            source_token_ids,
            source_mask=source_mask
        )

        # Start decoder with BOS.
        bos_token_id = self.tokenizer.token_to_id["[BOS]"]
        eos_token_id = self.tokenizer.token_to_id["[EOS]"]

        generated_ids = torch.tensor(
            [[bos_token_id]],
            dtype=torch.long,
            device=self.device
        )

        past_key_values = None

        # Incremental decoding with KV Cache.
        for step in range(max_new_tokens):

            current_token = generated_ids[:, -1:]

            logits, past_key_values = self.model.decode_step(
                target_token_ids=current_token,
                encoder_output=encoder_output,
                past_key_values=past_key_values,
                source_mask=source_mask,
                position_offset=step
            )

            next_token = logits[:, -1, :].argmax(dim=-1)

            next_token_id = next_token.item()

            generated_ids = torch.cat(
                [
                    generated_ids,
                    next_token.unsqueeze(1)
                ],
                dim=1
            )

            if next_token_id == eos_token_id:
                break

        # Remove decoder BOS before decoding.
        generated_token_ids = generated_ids[0, 1:].tolist()

        generated_text = self.tokenizer.decode(
            generated_token_ids,
            skip_special_tokens=True
        )

        return {
            "input_text": text,
            "generated_text": generated_text,
            "generated_token_count": len(generated_token_ids)
        }