
import pytest
import torch
import torch.nn as nn

from src.model.transformer import Transformer


def create_transformer():
    return Transformer(
        vocab_size=1000,
        embedding_dim=256,
        max_sequence_length=128,
        num_encoder_layers=3,
        num_decoder_layers=3,
        num_heads=4,
        hidden_dim=1024,
        dropout=0.1
    )


def create_inputs():
    source_ids = torch.randint(0, 1000, (2, 7))
    target_ids = torch.randint(0, 1000, (2, 5))

    return source_ids, target_ids


# 1. Model produces raw logits
def test_model_produces_raw_logits():
    model = create_transformer()
    source_ids, target_ids = create_inputs()

    logits = model(source_ids, target_ids)

    assert logits.shape == (2, 5, 1000)
    assert torch.isfinite(logits).all()


# 2. Cross entropy loss calculation
def test_cross_entropy_loss_calculation():
    model = create_transformer()
    source_ids, target_ids = create_inputs()

    logits = model(source_ids, target_ids)

    loss_function = nn.CrossEntropyLoss()

    loss = loss_function(
        logits.reshape(-1, 1000),
        target_ids.reshape(-1)
    )

    assert loss.ndim == 0
    assert torch.isfinite(loss)
    assert loss.item() > 0


# 3. Loss ignores padding tokens
def test_loss_ignores_padding_tokens():
    logits = torch.randn(2, 5, 1000)

    targets = torch.tensor([
        [10, 20, 30, 0, 0],
        [15, 25, 35, 40, 0]
    ])

    loss_function = nn.CrossEntropyLoss(
        ignore_index=0
    )

    loss_with_padding = loss_function(
        logits.reshape(-1, 1000),
        targets.reshape(-1)
    )

    valid_positions = targets != 0

    loss_without_padding = loss_function(
        logits[valid_positions],
        targets[valid_positions]
    )

    assert torch.allclose(
        loss_with_padding,
        loss_without_padding
    )


# 4. Loss supports backpropagation
def test_loss_backpropagation():
    model = create_transformer()
    source_ids, target_ids = create_inputs()

    logits = model(source_ids, target_ids)

    loss_function = nn.CrossEntropyLoss()

    loss = loss_function(
        logits.reshape(-1, 1000),
        target_ids.reshape(-1)
    )

    loss.backward()

    assert model.output_projection.weight.grad is not None
    assert torch.isfinite(
        model.output_projection.weight.grad
    ).all()


# 5. Gradients reach encoder and decoder
def test_loss_gradient_flow():
    model = create_transformer()
    source_ids, target_ids = create_inputs()

    logits = model(source_ids, target_ids)

    loss_function = nn.CrossEntropyLoss()

    loss = loss_function(
        logits.reshape(-1, 1000),
        target_ids.reshape(-1)
    )

    loss.backward()

    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            assert parameter.grad is not None, (
                f"Missing gradient: {name}"
            )
            assert torch.isfinite(parameter.grad).all(), (
                f"Non-finite gradient: {name}"
            )



# 6. Full Transformer logits remain finite
def test_numerical_stability_of_logits():
    model = create_transformer()
    source_ids, target_ids = create_inputs()

    logits = model(source_ids, target_ids)

    assert torch.isfinite(logits).all()


# 7. Cross entropy loss remains finite
def test_numerical_stability_of_loss():
    model = create_transformer()
    source_ids, target_ids = create_inputs()

    logits = model(source_ids, target_ids)

    loss_function = nn.CrossEntropyLoss()

    loss = loss_function(
        logits.reshape(-1, 1000),
        target_ids.reshape(-1)
    )

    assert torch.isfinite(loss)
    assert loss.item() >= 0


# 8. Gradients remain finite after backpropagation
def test_numerical_stability_of_gradients():
    model = create_transformer()
    source_ids, target_ids = create_inputs()

    logits = model(source_ids, target_ids)

    loss = nn.CrossEntropyLoss()(
        logits.reshape(-1, 1000),
        target_ids.reshape(-1)
    )

    loss.backward()

    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            assert parameter.grad is not None, (
                f"Missing gradient: {name}"
            )

            assert torch.isfinite(parameter.grad).all(), (
                f"Non-finite gradient: {name}"
            )


# 9. Cross entropy handles large-magnitude logits
def test_large_logits_numerical_stability():
    logits = torch.tensor([
        [10000.0, 9999.0, -10000.0],
        [-10000.0, 10000.0, 9999.0]
    ], requires_grad=True)

    targets = torch.tensor([0, 1])

    loss_function = nn.CrossEntropyLoss()
    loss = loss_function(logits, targets)

    assert torch.isfinite(loss)
    assert loss.item() >= 0

    loss.backward()

    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()


# 10. Zero-valued loss produces zero gradients
def test_zero_loss_gradient_behavior():
    logits = torch.randn(
        2, 5,
        requires_grad=True
    )

    loss = logits.sum() * 0.0

    loss.backward()

    assert logits.grad is not None
    assert torch.equal(
        logits.grad,
        torch.zeros_like(logits)
    )


# 11. Tiny dataset overfitting test
def test_transformer_can_overfit_tiny_dataset():

    torch.manual_seed(42)

    model = Transformer(
        vocab_size=100,
        embedding_dim=64,
        max_sequence_length=16,
        num_encoder_layers=2,
        num_decoder_layers=2,
        num_heads=4,
        hidden_dim=128,
        dropout=0.0
    )

    source_ids = torch.tensor([
        [10, 20, 30, 40, 50],
        [15, 25, 35, 45, 55]
    ])

    target_ids = torch.tensor([
        [2, 11, 12, 13],
        [2, 21, 22, 23]
    ])

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=0.001
    )

    loss_function = nn.CrossEntropyLoss()

    model.train()

    # Initial loss
    with torch.no_grad():
        initial_logits = model(
            source_ids,
            target_ids
        )

        initial_loss = loss_function(
            initial_logits.reshape(-1, 100),
            target_ids.reshape(-1)
        ).item()

    # Training loop
    training_steps = 200

    for step in range(training_steps):

        optimizer.zero_grad()

        logits = model(
            source_ids,
            target_ids
        )

        loss = loss_function(
            logits.reshape(-1, 100),
            target_ids.reshape(-1)
        )

        loss.backward()

        optimizer.step()

    # Final loss
    model.eval()

    with torch.no_grad():
        final_logits = model(
            source_ids,
            target_ids
        )

        final_loss = loss_function(
            final_logits.reshape(-1, 100),
            target_ids.reshape(-1)
        ).item()

    print("Initial loss:", initial_loss)
    print("Final loss:", final_loss)

    assert final_loss < initial_loss * 0.5, (
        f"Loss did not decrease sufficiently: "
        f"initial={initial_loss}, final={final_loss}"
    )


# 12. Adam optimizer updates model parameters
def test_adam_optimizer_updates_parameters():
    torch.manual_seed(42)

    model = create_transformer()
    source_ids, target_ids = create_inputs()

    original_weights = (
        model.output_projection.weight.detach().clone()
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=0.001
    )

    logits = model(source_ids, target_ids)

    loss = nn.CrossEntropyLoss()(
        logits.reshape(-1, 1000),
        target_ids.reshape(-1)
    )

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    updated_weights = model.output_projection.weight.detach()

    assert not torch.equal(
        original_weights,
        updated_weights
    )


# 13. Optimizer zero_grad resets gradients
def test_optimizer_zero_grad_resets_gradients():
    model = create_transformer()
    source_ids, target_ids = create_inputs()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=0.001
    )

    logits = model(source_ids, target_ids)

    loss = nn.CrossEntropyLoss()(
        logits.reshape(-1, 1000),
        target_ids.reshape(-1)
    )

    loss.backward()

    parameter = model.output_projection.weight

    assert parameter.grad is not None
    assert parameter.grad.abs().sum() > 0

    optimizer.zero_grad(set_to_none=False)

    assert torch.count_nonzero(parameter.grad) == 0


# 14. Parameters remain finite after training
def test_parameters_remain_finite_after_training():
    torch.manual_seed(42)

    model = create_transformer()
    source_ids, target_ids = create_inputs()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=0.001
    )

    loss_function = nn.CrossEntropyLoss()

    model.train()

    for _ in range(5):
        optimizer.zero_grad()

        logits = model(source_ids, target_ids)

        loss = loss_function(
            logits.reshape(-1, 1000),
            target_ids.reshape(-1)
        )

        loss.backward()
        optimizer.step()

    for name, parameter in model.named_parameters():
        assert torch.isfinite(parameter).all(), (
            f"Non-finite parameter: {name}"
        )


# 15. Loss history is recorded across training steps
def test_training_loss_history():
    torch.manual_seed(42)

    model = create_transformer()
    source_ids, target_ids = create_inputs()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=0.001
    )

    loss_function = nn.CrossEntropyLoss()
    loss_history = []

    model.train()

    for _ in range(5):
        optimizer.zero_grad()

        logits = model(source_ids, target_ids)

        loss = loss_function(
            logits.reshape(-1, 1000),
            target_ids.reshape(-1)
        )

        loss.backward()
        optimizer.step()

        loss_history.append(loss.item())

    assert len(loss_history) == 5
    assert all(
        isinstance(value, float)
        for value in loss_history
    )
    assert all(
        torch.isfinite(torch.tensor(value))
        for value in loss_history
    )


# 16. Evaluation mode does not update parameters
def test_evaluation_does_not_update_parameters():
    model = create_transformer()
    source_ids, target_ids = create_inputs()

    model.eval()

    original_weights = {
        name: parameter.detach().clone()
        for name, parameter in model.named_parameters()
    }

    with torch.no_grad():
        logits = model(source_ids, target_ids)

        loss = nn.CrossEntropyLoss()(
            logits.reshape(-1, 1000),
            target_ids.reshape(-1)
        )

    for name, parameter in model.named_parameters():
        assert torch.equal(
            original_weights[name],
            parameter
        ), f"Parameter changed during evaluation: {name}"

    assert torch.isfinite(loss)

