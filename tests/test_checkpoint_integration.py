
import torch
import torch.nn as nn

from src.data.trainer import Trainer
from src.data.fit import fit
from src.data.checkpoint import (
    save_checkpoint,
    load_checkpoint
)
from src.data.scheduler import create_scheduler

from src.data.early_stopping import EarlyStopping



class TinySeq2SeqModel(nn.Module):

    def __init__(self, vocab_size=20, embedding_dim=8):
        super().__init__()

        self.embedding = nn.Embedding(
            vocab_size,
            embedding_dim
        )

        self.output_projection = nn.Linear(
            embedding_dim,
            vocab_size
        )

    def forward(
        self,
        encoder_input_ids,
        decoder_input_ids,
        encoder_mask=None
    ):
        encoder_embeddings = self.embedding(
            encoder_input_ids
        )

        decoder_embeddings = self.embedding(
            decoder_input_ids
        )

        if encoder_mask is not None:
            mask = encoder_mask.unsqueeze(-1).to(
                encoder_embeddings.dtype
            )

            encoder_context = (
                (encoder_embeddings * mask).sum(dim=1)
                / mask.sum(dim=1).clamp(min=1)
            )
        else:
            encoder_context = encoder_embeddings.mean(dim=1)

        decoder_output = (
            decoder_embeddings
            + encoder_context.unsqueeze(1)
        )

        return self.output_projection(decoder_output)


def create_trainer():

    model = TinySeq2SeqModel()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=0.001
    )

    return Trainer(
        model=model,
        optimizer=optimizer,
        device="cpu"
    )


def create_batch():

    return {
        "encoder_input_ids": torch.tensor([
            [2, 5, 3],
            [2, 7, 3]
        ]),
        "encoder_attention_mask": torch.tensor([
            [1, 1, 1],
            [1, 1, 1]
        ]),
        "decoder_input_ids": torch.tensor([
            [2, 8, 9],
            [2, 6, 4]
        ]),
        "labels": torch.tensor([
            [8, 9, 3],
            [6, 4, 3]
        ])
    }


def test_checkpoint_save_load_and_resume(tmp_path):

    torch.manual_seed(42)

    checkpoint_dir = tmp_path / "checkpoints"

    train_loader = [create_batch()]
    validation_loader = [create_batch()]

    # 1. Train for two epochs.
    trainer = create_trainer()

    history = fit(
        trainer=trainer,
        train_loader=train_loader,
        validation_loader=validation_loader,
        epochs=2,
        checkpoint_dir=checkpoint_dir
    )

    latest_checkpoint = checkpoint_dir / "latest.pt"
    best_checkpoint = checkpoint_dir / "best.pt"

    # 2. Verify checkpoint files.
    assert latest_checkpoint.exists()
    assert best_checkpoint.exists()

    assert len(history) == 2
    assert history[-1]["epoch"] == 2

    # 3. Verify scheduler state exists in latest checkpoint.
    saved_checkpoint = torch.load(
        latest_checkpoint,
        map_location="cpu",
        weights_only=True
    )

    assert saved_checkpoint["scheduler_state_dict"] is not None

    # 4. Save trained model parameters.
    trained_parameters = {
        name: parameter.detach().clone()
        for name, parameter in trainer.model.state_dict().items()
    }

    # 5. Restore into a fresh trainer.
    restored_trainer = create_trainer()

    restored_scheduler = create_scheduler(
        optimizer=restored_trainer.optimizer
    )

    progress = load_checkpoint(
        model=restored_trainer.model,
        optimizer=restored_trainer.optimizer,
        checkpoint_path=latest_checkpoint,
        device="cpu",
        scheduler=restored_scheduler
    )

    assert progress["epoch"] == 1
    assert len(progress["history"]) == 2

    # 6. Verify model parameters.
    restored_parameters = restored_trainer.model.state_dict()

    for name, parameter in trained_parameters.items():
        assert torch.equal(
            parameter,
            restored_parameters[name]
        )

    # 7. Verify optimizer state.
    assert len(restored_trainer.optimizer.state) > 0

    # 8. Resume training up to three total epochs.
    resumed_history = fit(
        trainer=restored_trainer,
        train_loader=train_loader,
        validation_loader=validation_loader,
        epochs=3,
        checkpoint_dir=checkpoint_dir,
        resume_from=latest_checkpoint
    )

    assert len(resumed_history) == 3

    assert [
        item["epoch"] for item in resumed_history
    ] == [1, 2, 3]

    # 9. Verify latest checkpoint after resume.
    final_progress = load_checkpoint(
        model=restored_trainer.model,
        optimizer=restored_trainer.optimizer,
        checkpoint_path=latest_checkpoint,
        device="cpu"
    )

    assert final_progress["epoch"] == 2
    assert len(final_progress["history"]) == 3


def test_scheduler_state_and_patience_restoration(tmp_path):

    # 1. Create model, optimizer and scheduler.
    model = TinySeq2SeqModel()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=0.001
    )

    scheduler = create_scheduler(
        optimizer=optimizer,
        factor=0.5,
        patience=1,
        min_lr=1e-6,
        threshold=0.0
    )

    # 2. First validation establishes the best loss.
    scheduler.step(1.0)

    # 3. Second validation does not improve.
    scheduler.step(1.1)

    assert scheduler.num_bad_epochs == 1
    assert optimizer.param_groups[0]["lr"] == 0.001

    # 4. Save checkpoint with scheduler state.
    checkpoint_path = tmp_path / "scheduler_checkpoint.pt"

    save_checkpoint(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        epoch=0,
        train_loss=1.2,
        validation_loss=1.1,
        history=[
            {
                "epoch": 1,
                "train_loss": 1.2,
                "validation_loss": 1.1
            }
        ],
        checkpoint_path=checkpoint_path
    )

    # 5. Create fresh objects.
    restored_model = TinySeq2SeqModel()

    restored_optimizer = torch.optim.Adam(
        restored_model.parameters(),
        lr=0.001
    )

    restored_scheduler = create_scheduler(
        optimizer=restored_optimizer,
        factor=0.5,
        patience=1,
        min_lr=1e-6,
        threshold=0.0
    )

    # 6. Restore all states.
    load_checkpoint(
        model=restored_model,
        optimizer=restored_optimizer,
        scheduler=restored_scheduler,
        checkpoint_path=checkpoint_path,
        device="cpu"
    )

    # 7. Verify restored scheduler state.
    assert restored_scheduler.num_bad_epochs == 1

    assert (
        restored_optimizer.param_groups[0]["lr"]
        == 0.001
    )

    # 8. One more bad validation should trigger LR reduction.
    restored_scheduler.step(1.1)

    assert restored_scheduler.num_bad_epochs == 0

    assert (
        restored_optimizer.param_groups[0]["lr"]
        == 0.0005
    )


def test_early_stopping_state_restoration(tmp_path):

    model = TinySeq2SeqModel()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=0.001
    )

    early_stopping = EarlyStopping(
        patience=3,
        min_delta=0.01
    )

    # Establish best loss.
    assert early_stopping(1.0) is False

    # Two consecutive non-improving epochs.
    assert early_stopping(1.005) is False
    assert early_stopping(1.006) is False

    assert early_stopping.best_loss == 1.0
    assert early_stopping.counter == 2

    checkpoint_path = tmp_path / "early_stopping.pt"

    save_checkpoint(
        model=model,
        optimizer=optimizer,
        epoch=2,
        train_loss=1.2,
        validation_loss=1.006,
        history=[],
        checkpoint_path=checkpoint_path,
        early_stopping=early_stopping
    )

    # Create fresh objects.
    restored_model = TinySeq2SeqModel()

    restored_optimizer = torch.optim.Adam(
        restored_model.parameters(),
        lr=0.001
    )

    restored_early_stopping = EarlyStopping(
        patience=3,
        min_delta=0.01
    )

    # Restore state.
    load_checkpoint(
        model=restored_model,
        optimizer=restored_optimizer,
        checkpoint_path=checkpoint_path,
        device="cpu",
        early_stopping=restored_early_stopping
    )

    assert restored_early_stopping.best_loss == 1.0
    assert restored_early_stopping.counter == 2
    assert restored_early_stopping.patience == 3

    # Third non-improving epoch should trigger stopping.
    assert restored_early_stopping(1.007) is True
    assert restored_early_stopping.early_stop is True


if __name__ == "__main__":
    import pytest

    pytest.main([__file__, "-v"])

