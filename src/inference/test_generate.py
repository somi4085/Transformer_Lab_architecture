
import pytest
import torch
import torch.nn as nn

from src.inference.generate import generate_text


# --------------------------------------------------
# Dummy positional encoding and encoder structure
# --------------------------------------------------

class DummyPositionalEncoding(nn.Module):

    def __init__(self, max_sequence_length=64):
        super().__init__()
        self.max_sequence_length = max_sequence_length


class DummyInputEmbedding(nn.Module):

    def __init__(self):
        super().__init__()

        self.positional_encoding = DummyPositionalEncoding()


class DummyEncoder(nn.Module):

    def __init__(self):
        super().__init__()

        self.input_embedding = DummyInputEmbedding()


# --------------------------------------------------
# Dummy Transformer
# --------------------------------------------------

class DummyTransformer(nn.Module):

    def __init__(self, predicted_tokens=None):
        super().__init__()

        self.encoder = DummyEncoder()

        self.dummy_parameter = nn.Parameter(
            torch.tensor(1.0)
        )

        self.predicted_tokens = (
            predicted_tokens or [10, 11, 3]
        )

        self.forward_calls = 0

    def forward(
        self,
        source_token_ids,
        target_token_ids,
        source_mask=None,
        target_mask=None
    ):

        batch_size, target_length = target_token_ids.shape

        vocab_size = 100

        logits = torch.zeros(
            batch_size,
            target_length,
            vocab_size,
            device=target_token_ids.device
        )

        step = self.forward_calls

        if step < len(self.predicted_tokens):
            next_token = self.predicted_tokens[step]
        else:
            next_token = 5

        logits[:, -1, next_token] = 10.0

        self.forward_calls += 1

        return logits


# --------------------------------------------------
# Dummy BPE Tokenizer
# --------------------------------------------------

class DummyTokenizer:

    def __init__(self):

        self.token_to_id = {
            "[PAD]": 0,
            "[UNK]": 1,
            "[BOS]": 2,
            "[EOS]": 3
        }

        self.id_to_token = {
            0: "[PAD]",
            1: "[UNK]",
            2: "[BOS]",
            3: "[EOS]",
            10: "Once",
            11: "upon",
            12: "time"
        }

    def encode_to_ids(self, text):

        return [10, 11]

    def decode(
        self,
        token_ids,
        skip_special_tokens=True
    ):

        tokens = []

        for token_id in token_ids:

            token = self.id_to_token.get(
                token_id,
                "[UNK]"
            )

            if skip_special_tokens and token in {
                "[PAD]",
                "[UNK]",
                "[BOS]",
                "[EOS]"
            }:
                continue

            tokens.append(token)

        return " ".join(tokens)


# --------------------------------------------------
# Test fixtures
# --------------------------------------------------

@pytest.fixture
def model():

    return DummyTransformer(
        predicted_tokens=[10, 11, 3]
    )


@pytest.fixture
def tokenizer():

    return DummyTokenizer()


# --------------------------------------------------
# 1. Valid input
# --------------------------------------------------

def test_valid_input(model, tokenizer):

    output = generate_text(
        model=model,
        tokenizer=tokenizer,
        input_text="Once upon a time",
        max_new_tokens=5
    )

    assert isinstance(output, str)
    assert output == "Once upon"


# --------------------------------------------------
# 2. Empty input
# --------------------------------------------------

@pytest.mark.parametrize(
    "input_text",
    ["", "   ", "\n\t"]
)
def test_empty_input(model, tokenizer, input_text):

    with pytest.raises(ValueError):

        generate_text(
            model=model,
            tokenizer=tokenizer,
            input_text=input_text
        )


# --------------------------------------------------
# 3. Invalid input type
# --------------------------------------------------

@pytest.mark.parametrize(
    "input_text",
    [123, None, ["hello"], {"text": "hello"}]
)
def test_invalid_input_type(model, tokenizer, input_text):

    with pytest.raises(TypeError):

        generate_text(
            model=model,
            tokenizer=tokenizer,
            input_text=input_text
        )


# --------------------------------------------------
# 4. Output type
# --------------------------------------------------

def test_output_type(model, tokenizer):

    output = generate_text(
        model=model,
        tokenizer=tokenizer,
        input_text="Hello world"
    )

    assert isinstance(output, str)


# --------------------------------------------------
# 5. Special tokens are removed
# --------------------------------------------------

def test_special_tokens_removed(model, tokenizer):

    model.predicted_tokens = [2, 10, 11, 3, 12]

    output = generate_text(
        model=model,
        tokenizer=tokenizer,
        input_text="Hello world",
        max_new_tokens=5
    )

    assert "[BOS]" not in output
    assert "[EOS]" not in output
    assert "[PAD]" not in output

    assert output == "Once upon"


# --------------------------------------------------
# 6. Maximum generation length
# --------------------------------------------------

def test_maximum_generation_length(model, tokenizer):

    max_new_tokens = 2

    generate_text(
        model=model,
        tokenizer=tokenizer,
        input_text="Hello world",
        max_new_tokens=max_new_tokens
    )

    # Dummy model should be called at most
    # max_new_tokens times.
    assert model.forward_calls <= max_new_tokens


# --------------------------------------------------
# 7. CPU device compatibility
# --------------------------------------------------

def test_cpu_device(model, tokenizer):

    output = generate_text(
        model=model,
        tokenizer=tokenizer,
        input_text="Hello world",
        device="cpu"
    )

    assert isinstance(output, str)

    for parameter in model.parameters():
        assert parameter.device.type == "cpu"


# --------------------------------------------------
# 8. Original model mode is restored
# --------------------------------------------------

@pytest.mark.parametrize(
    "initial_training_mode",
    [True, False]
)
def test_model_mode_restoration(
    tokenizer,
    initial_training_mode
):

    model = DummyTransformer(
        predicted_tokens=[10, 11, 3]
    )

    model.train(initial_training_mode)

    generate_text(
        model=model,
        tokenizer=tokenizer,
        input_text="Hello world",
        max_new_tokens=5
    )

    assert model.training == initial_training_mode


# --------------------------------------------------
# 9. Maximum positional sequence length
# --------------------------------------------------

def test_input_exceeding_max_sequence_length(
    model,
    tokenizer
):

    model.encoder.input_embedding.positional_encoding.max_sequence_length = 3

    with pytest.raises(ValueError):

        generate_text(
            model=model,
            tokenizer=tokenizer,
            input_text="Hello world"
        )


if __name__ == "__main__":

    pytest.main([
        __file__,
        "-v"
    ])
