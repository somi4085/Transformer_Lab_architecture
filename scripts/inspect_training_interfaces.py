
import inspect

from src.model.transformer import Transformer
from src.data.trainer import Trainer
from src.data.train import train_model
from src.data.validate import validate_model
from src.data.fit import fit
from src.data.checkpoint import (
    save_checkpoint,
    load_checkpoint
)


def print_signature(name, function):
    print(f"\n{name}")
    print("-" * 60)
    print(inspect.signature(function))


def inspect_interfaces():

    print("\n========== TRANSFORMER TRAINING INTERFACES ==========")

    print_signature(
        "Transformer constructor",
        Transformer.__init__
    )

    print_signature(
        "Transformer forward",
        Transformer.forward
    )

    print_signature(
        "Trainer constructor",
        Trainer.__init__
    )

    print_signature(
        "Trainer train_step",
        Trainer.train_step
    )

    print_signature(
        "Training function",
        train_model
    )

    print_signature(
        "Validation function",
        validate_model
    )

    print_signature(
        "Fit function",
        fit
    )

    print_signature(
        "Save checkpoint",
        save_checkpoint
    )

    print_signature(
        "Load checkpoint",
        load_checkpoint
    )

    print("\n========== INSPECTION COMPLETE ==========")


if __name__ == "__main__":
    inspect_interfaces()
