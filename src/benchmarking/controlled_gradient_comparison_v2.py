
import math
import torch
import torch.nn as nn
import csv
from pathlib import Path

from src.layers.encoder_layer import EncoderLayer
from src.layers.pre_norm_encoder_layer import PreNormEncoderLayer


# ==========================================
# 1. Experiment Configuration
# ==========================================

SEEDS = [42, 123, 456, 789, 2026]
BATCH_SIZE = 2
SEQUENCE_LENGTH = 32
EMBEDDING_DIM = 256

NUM_HEADS = 8
HIDDEN_DIM = 1024
DROPOUT = 0.0

LAYER_COUNTS = [2, 4, 8, 12]


# ==========================================
# 2. Fixed Input and Probe Generation
# ==========================================

def create_fixed_input_and_probe(seed):

    torch.manual_seed(seed)

    fixed_input = torch.randn(
        BATCH_SIZE,
        SEQUENCE_LENGTH,
        EMBEDDING_DIM
    )

    fixed_probe = torch.randn(
        BATCH_SIZE,
        SEQUENCE_LENGTH,
        EMBEDDING_DIM
    )

    return fixed_input, fixed_probe


# ==========================================
# 3. Post-LN Model Creation
# ==========================================
def create_post_norm_model(num_layers, seed):

    torch.manual_seed(seed)

    model = nn.ModuleList([
        EncoderLayer(
            embedding_dim=EMBEDDING_DIM,
            num_heads=NUM_HEADS,
            hidden_dim=HIDDEN_DIM,
            dropout=DROPOUT
        )
        for _ in range(num_layers)
    ])

    return model


def create_pre_norm_model(num_layers, seed):

    torch.manual_seed(seed)

    model = nn.ModuleList([
        PreNormEncoderLayer(
            embedding_dim=EMBEDDING_DIM,
            num_heads=NUM_HEADS,
            hidden_dim=HIDDEN_DIM,
            dropout=DROPOUT
        )
        for _ in range(num_layers)
    ])

    return model



# ==========================================
# 5. Forward Pass
# ==========================================

def run_model(model, fixed_input):

    output = fixed_input.clone()

    for layer in model:
        output = layer(output)

    return output


# ==========================================
# 6. Fixed Probe Loss
# ==========================================

def calculate_probe_loss(output, fixed_probe):

    total_elements = fixed_probe.numel()

    loss = (
        output * fixed_probe
    ).sum() / math.sqrt(total_elements)

    return loss


# ==========================================
# 7. Gradient Norm Calculation
# ==========================================

def calculate_gradient_norm(module):

    squared_norm = 0.0

    for parameter in module.parameters():

        if parameter.grad is not None:

            gradient_norm = (
                parameter.grad.detach().norm(2).item()
            )

            squared_norm += gradient_norm ** 2

    return math.sqrt(squared_norm)


# ==========================================
# 8. Backward Experiment
# ==========================================


def run_backward_experiment(model, fixed_input, fixed_probe):

    model.zero_grad(set_to_none=True)

    input_tensor = (
        fixed_input.clone()
        .detach()
        .requires_grad_(True)
    )

    # Store activation metrics for every layer
    activation_norms = []
    normalized_activation_norms = []

    output = input_tensor

    # Forward pass layer by layer
    for layer in model:

        output = layer(output)

        # Raw activation norm
        activation_norm = (
            output.detach().norm(2).item()
        )

        activation_norms.append(activation_norm)

        # Normalized activation norm
        total_elements = output.numel()

        normalized_norm = (
            activation_norm / math.sqrt(total_elements)
        )

        normalized_activation_norms.append(
            normalized_norm
        )

    # Calculate fixed probe loss
    loss = calculate_probe_loss(
        output,
        fixed_probe
    )

    # Backpropagation
    loss.backward()

    # Input gradient norm
    input_gradient_norm = (
        input_tensor.grad.detach().norm(2).item()
    )

    # Gradient norm for every layer
    layer_gradient_norms = [
        calculate_gradient_norm(layer)
        for layer in model
    ]

    # First-to-last layer gradient ratio
    first_layer_gradient = layer_gradient_norms[0]
    last_layer_gradient = layer_gradient_norms[-1]

    if last_layer_gradient > 0:

        gradient_ratio = (
            first_layer_gradient / last_layer_gradient
        )

    else:
        gradient_ratio = float("nan")

    return {
        "loss": loss.item(),
        "input_gradient_norm": input_gradient_norm,
        "layer_gradient_norms": layer_gradient_norms,
        "activation_norms": activation_norms,
        "normalized_activation_norms": normalized_activation_norms,
        "gradient_ratio": gradient_ratio
    }


# ==========================================
# 9. Run Complete Experiment
# ==========================================

def run_experiment():

    detailed_rows = []
    summary_rows = []

    architectures = {
        "Post-LN": create_post_norm_model,
        "Pre-LN": create_pre_norm_model
    }

    # Outer loop: different random seeds
    for seed in SEEDS:

        print(f"\n{'=' * 60}")
        print(f"Running experiments for seed: {seed}")
        print(f"{'=' * 60}")

        fixed_input, fixed_probe = create_fixed_input_and_probe(seed)

        print("Input shape:", fixed_input.shape)
        print("Probe shape:", fixed_probe.shape)

        # Middle loop: model depths
        for num_layers in LAYER_COUNTS:

            # Inner loop: normalization architectures
            for architecture_name, model_factory in architectures.items():

                model = model_factory(
                    num_layers=num_layers,
                    seed=seed
                )

                results = run_backward_experiment(
                    model,
                    fixed_input,
                    fixed_probe
                )

                layer_gradient_norms = results["layer_gradient_norms"]
                activation_norms = results["activation_norms"]

                normalized_activation_norms = results[
                    "normalized_activation_norms"
                ]

                # Collect individual layer records
                for layer_index in range(num_layers):

                    detailed_rows.append({
                        "seed": seed,
                        "architecture": architecture_name,
                        "num_layers": num_layers,
                        "layer_index": layer_index + 1,
                        "activation_norm": activation_norms[layer_index],
                        "normalized_activation_norm": (
                            normalized_activation_norms[layer_index]
                        ),
                        "parameter_gradient_norm": (
                            layer_gradient_norms[layer_index]
                        ),
                        "loss": results["loss"],
                        "input_gradient_norm": (
                            results["input_gradient_norm"]
                        ),
                        "gradient_ratio": results["gradient_ratio"]
                    })

                # Collect complete model summary
                summary_rows.append({
                    "seed": seed,
                    "architecture": architecture_name,
                    "num_layers": num_layers,
                    "loss": results["loss"],
                    "input_gradient_norm": (
                        results["input_gradient_norm"]
                    ),
                    "first_layer_gradient_norm": (
                        layer_gradient_norms[0]
                    ),
                    "last_layer_gradient_norm": (
                        layer_gradient_norms[-1]
                    ),
                    "gradient_ratio": results["gradient_ratio"],
                    "first_activation_norm": activation_norms[0],
                    "last_activation_norm": activation_norms[-1],
                    "first_normalized_activation_norm": (
                        normalized_activation_norms[0]
                    ),
                    "last_normalized_activation_norm": (
                        normalized_activation_norms[-1]
                    )
                })

                print(
                    f"{architecture_name} | "
                    f"Layers: {num_layers} | "
                    f"Loss: {results['loss']:.6f} | "
                    f"Gradient ratio: {results['gradient_ratio']:.6f}"
                )

    # -------------------------------------------------
    # Aggregate results across seeds
    # -------------------------------------------------

    metric_columns = [
        "loss",
        "input_gradient_norm",
        "first_layer_gradient_norm",
        "last_layer_gradient_norm",
        "gradient_ratio",
        "first_activation_norm",
        "last_activation_norm",
        "first_normalized_activation_norm",
        "last_normalized_activation_norm"
    ]

    statistics_rows = []

    for num_layers in LAYER_COUNTS:

        for architecture_name in architectures:

            matching_rows = [
                row for row in summary_rows
                if row["num_layers"] == num_layers
                and row["architecture"] == architecture_name
            ]

            statistics_row = {
                "architecture": architecture_name,
                "num_layers": num_layers,
                "number_of_seeds": len(matching_rows)
            }

            for metric in metric_columns:

                values = torch.tensor(
                    [row[metric] for row in matching_rows],
                    dtype=torch.float64
                )

                statistics_row[f"{metric}_mean"] = (
                    values.mean().item()
                )

                statistics_row[f"{metric}_std"] = (
                    values.std(unbiased=True).item()
                )

            statistics_rows.append(statistics_row)

    # -------------------------------------------------
    # Save CSV reports
    # -------------------------------------------------

    project_root = Path(__file__).resolve().parents[2]

    reports_directory = project_root / "reports"

    reports_directory.mkdir(
        parents=True,
        exist_ok=True
    )

    # Detailed layer-level report
    detailed_file = (
        reports_directory / "gradient_comparison_v2.csv"
    )

    with open(
        detailed_file,
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=detailed_rows[0].keys()
        )

        writer.writeheader()
        writer.writerows(detailed_rows)

    # Per-seed model-level report
    summary_file = (
        reports_directory / "gradient_summary_v2.csv"
    )

    with open(
        summary_file,
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=summary_rows[0].keys()
        )

        writer.writeheader()
        writer.writerows(summary_rows)

    # Mean and standard deviation report
    statistics_file = (
        reports_directory / "gradient_statistics_v2.csv"
    )

    with open(
        statistics_file,
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=statistics_rows[0].keys()
        )

        writer.writeheader()
        writer.writerows(statistics_rows)

    print("\n" + "=" * 60)
    print("MULTI-SEED EXPERIMENT COMPLETED")
    print("=" * 60)

    print("Seeds:", SEEDS)
    print("Detailed rows:", len(detailed_rows))
    print("Summary rows:", len(summary_rows))
    print("Statistics rows:", len(statistics_rows))

    print("\nReports saved:")
    print(detailed_file)
    print(summary_file)
    print(statistics_file)



# ==========================================
# 10. Main Entry Point
# ==========================================

if __name__ == "__main__":

    run_experiment()
