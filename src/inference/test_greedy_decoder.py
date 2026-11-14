
import pytest
import torch
import torch.nn as nn

from src.inference.greedy_decoder import greedy_decode


class DummyModel(nn.Module):
    """
    Deterministic model for testing greedy decoding.
    Returns predefined next-token predictions.
    """

    def __init__(self, token_sequences, vocab_size=100):
        super().__init__()

        self.token_sequences = token_sequences
        self.vocab_size = vocab_size
        self.forward_calls = 0
        self.grad_modes = []

    def forward(
        self,
        source_token_ids,
        target_token_ids,
        source_mask=None,
        target_mask=None
    ):
        self.grad_modes.append(torch.is_grad_enabled())

        batch_size, target_length = target_token_ids.shape

        logits = torch.zeros(
            batch_size,
            target_length,
            self.vocab_size,
            device=target_token_ids.device
        )

        step = self.forward_calls

        for batch_index in range(batch_size):

            sequence = self.token_sequences[batch_index]

            if step < len(sequence):
                next_token = sequence[step]
            else:
                next_token = 5

            logits[
                batch_index,
                -1,
                next_token
            ] = 10.0

        self.forward_calls += 1

        return logits


def create_inputs(batch_size=1):

    source_token_ids = torch.tensor(
        [[10, 20, 30]] * batch_size
    )

    source_mask = torch.ones(
        batch_size,
        3,
        dtype=torch.long
    )

    return source_token_ids, source_mask


# 1. Output shape
def test_output_shape():

    model = DummyModel([
        [10, 11, 12, 13, 14]
    ])

    source_token_ids, source_mask = create_inputs()

    output = greedy_decode(
        model=model,
        source_token_ids=source_token_ids,
        bos_token_id=2,
        eos_token_id=3,
        max_new_tokens=5,
        source_mask=source_mask
    )

    assert output.shape == (1, 6)


# 2. BOS token
def test_bos_token():

    model = DummyModel([
        [10, 11, 3]
    ])

    source_token_ids, source_mask = create_inputs()

    output = greedy_decode(
        model=model,
        source_token_ids=source_token_ids,
        bos_token_id=2,
        eos_token_id=3,
        max_new_tokens=5,
        source_mask=source_mask
    )

    assert output[0, 0].item() == 2


# 3. EOS stopping
def test_eos_stopping():

    model = DummyModel([
        [15, 21, 3, 40, 50]
    ])

    source_token_ids, source_mask = create_inputs()

    output = greedy_decode(
        model=model,
        source_token_ids=source_token_ids,
        bos_token_id=2,
        eos_token_id=3,
        max_new_tokens=5,
        source_mask=source_mask
    )

    expected = torch.tensor([
        [2, 15, 21, 3]
    ])

    assert torch.equal(output, expected)
    assert model.forward_calls == 3


# 4. Maximum generation length
def test_maximum_generation_length():

    model = DummyModel([
        [10, 11, 12, 13, 14]
    ])

    source_token_ids, source_mask = create_inputs()

    output = greedy_decode(
        model=model,
        source_token_ids=source_token_ids,
        bos_token_id=2,
        eos_token_id=3,
        max_new_tokens=3,
        source_mask=source_mask
    )

    assert output.shape == (1, 4)
    assert model.forward_calls == 3


# 5. Batch generation
def test_batch_generation():

    model = DummyModel([
        [10, 11, 3],
        [20, 21, 22, 3]
    ])

    source_token_ids, source_mask = create_inputs(
        batch_size=2
    )

    output = greedy_decode(
        model=model,
        source_token_ids=source_token_ids,
        bos_token_id=2,
        eos_token_id=3,
        max_new_tokens=5,
        source_mask=source_mask
    )

    assert output.shape == (2, 5)

    assert torch.equal(
        output[0],
        torch.tensor([2, 10, 11, 3, 0])
    )

    assert torch.equal(
        output[1],
        torch.tensor([2, 20, 21, 22, 3])
    )


# 6. Original training mode is restored
@pytest.mark.parametrize(
    "initial_training_mode",
    [True, False]
)
def test_model_mode_restoration(initial_training_mode):

    model = DummyModel([
        [10, 11, 3]
    ])

    model.train(initial_training_mode)

    source_token_ids, source_mask = create_inputs()

    greedy_decode(
        model=model,
        source_token_ids=source_token_ids,
        bos_token_id=2,
        eos_token_id=3,
        max_new_tokens=5,
        source_mask=source_mask
    )

    assert model.training == initial_training_mode


# 7. Gradients are disabled during inference
def test_no_gradients_during_inference():

    model = DummyModel([
        [10, 11, 3]
    ])

    source_token_ids, source_mask = create_inputs()

    greedy_decode(
        model=model,
        source_token_ids=source_token_ids,
        bos_token_id=2,
        eos_token_id=3,
        max_new_tokens=5,
        source_mask=source_mask
    )

    assert len(model.grad_modes) == 3

    assert all(
        grad_enabled is False
        for grad_enabled in model.grad_modes
    )


if __name__ == "__main__":

    pytest.main([
        __file__,
        "-v"
    ])

