
import torch

from src.data.dataset import TextDataset
from src.data.seq2seq_data_loader import (
    create_seq2seq_data_loader
)
from src.model.transformer import Transformer
from src.data.trainer import Trainer
from src.data.validate import validate_model


def create_examples():

    return [
        {
            "encoder_input_ids": [2, 10, 11, 12, 13, 14, 15, 3],
            "decoder_input_ids": [2, 20, 21, 22, 23, 3],
            "labels": [20, 21, 22, 23, 3, -100]
        },
        {
            "encoder_input_ids": [2, 16, 17, 18, 3],
            "decoder_input_ids": [2, 24, 25, 3],
            "labels": [24, 25, 3, -100]
        }
    ]


def create_model(vocab_size=50):

    return Transformer(
        vocab_size=vocab_size,
        embedding_dim=64,
        max_sequence_length=32,
        num_encoder_layers=2,
        num_decoder_layers=2,
        num_heads=4,
        hidden_dim=256,
        dropout=0.1
    )


def test_end_to_end_training_integration():

    torch.manual_seed(42)

    # 1. Dataset
    examples = create_examples()
    dataset = TextDataset(examples)

    # 2. DataLoader
    train_loader = create_seq2seq_data_loader(
        dataset=dataset,
        pad_token_id=0,
        batch_size=2,
        shuffle=False,
        drop_last=False,
        num_workers=0
    )

    validation_loader = create_seq2seq_data_loader(
        dataset=dataset,
        pad_token_id=0,
        batch_size=2,
        shuffle=False,
        drop_last=False,
        num_workers=0
    )

    batch = next(iter(train_loader))

    # 3. Verify batch structure
    expected_keys = {
        "encoder_input_ids",
        "encoder_attention_mask",
        "decoder_input_ids",
        "decoder_attention_mask",
        "labels"
    }

    assert set(batch.keys()) == expected_keys

    assert batch["encoder_input_ids"].shape == (2, 8)
    assert batch["decoder_input_ids"].shape == (2, 6)
    assert batch["labels"].shape == (2, 6)

    assert batch["encoder_attention_mask"].shape == (2, 8)
    assert batch["decoder_attention_mask"].shape == (2, 6)

    # 4. Create Transformer
    model = create_model(vocab_size=50)

    # 5. Verify Transformer output
    model.eval()

    with torch.no_grad():

        logits = model(
            source_token_ids=batch["encoder_input_ids"],
            target_token_ids=batch["decoder_input_ids"],
            source_mask=batch["encoder_attention_mask"],
            target_mask=batch["decoder_attention_mask"]
        )

    assert logits.shape == (2, 6, 50)
    assert torch.isfinite(logits).all()

    # 6. Create optimizer and Trainer
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=1e-3
    )

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        max_grad_norm=1.0,
        device="cpu"
    )

    # Save parameter before training
    parameter_before = next(
        model.parameters()
    ).detach().clone()

    # 7. Training step
    train_result = trainer.train_step(batch)

    parameter_after = next(
        model.parameters()
    ).detach().clone()

    assert torch.isfinite(
        torch.tensor(train_result["loss"])
    )

    assert train_result["loss"] >= 0

    assert torch.isfinite(
        torch.tensor(train_result["grad_norm"])
    )

    assert not torch.equal(
        parameter_before,
        parameter_after
    )

    # Verify gradients were generated
    gradients = [
        parameter.grad
        for parameter in model.parameters()
        if parameter.grad is not None
    ]

    assert len(gradients) > 0

    assert all(
        torch.isfinite(gradient).all()
        for gradient in gradients
    )

    # 8. Validation
    parameters_before_validation = [
        parameter.detach().clone()
        for parameter in model.parameters()
    ]

    validation_result = validate_model(
        trainer,
        validation_loader
    )

    parameters_after_validation = [
        parameter.detach().clone()
        for parameter in model.parameters()
    ]

    assert torch.isfinite(
        torch.tensor(
            validation_result["validation_loss"]
        )
    )

    assert validation_result["validation_loss"] >= 0

    assert validation_result["valid_tokens"] > 0

    # Validation must not update model parameters
    for before, after in zip(
        parameters_before_validation,
        parameters_after_validation
    ):
        assert torch.equal(before, after)

    print("\nTraining integration test completed.")
    print("Training loss:", train_result["loss"])
    print("Gradient norm:", train_result["grad_norm"])
    print(
        "Validation loss:",
        validation_result["validation_loss"]
    )
    print(
        "Valid tokens:",
        validation_result["valid_tokens"]
    )


if __name__ == "__main__":
    import pytest

    pytest.main([__file__, "-v"])
