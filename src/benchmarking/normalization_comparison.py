
import csv
import os
import torch
import torch.nn as nn

from src.layers.encoder_layer import EncoderLayer
from src.layers.pre_norm_encoder_layer import PreNormEncoderLayer
from src.layers.feed_forward import FeedForwardNetwork
from src.attention.multi_head_attention import MultiHeadAttention

# Experiment configuration
EMBEDDING_DIM = 256
NUM_HEADS = 8
HIDDEN_DIM = 1024
BATCH_SIZE = 2
SEQUENCE_LENGTH = 32
DROPOUT = 0.0

LAYER_COUNTS = [2, 4, 8, 12]
NUM_RUNS = 3


def create_post_norm_model(num_layers):

    return nn.ModuleList([
        EncoderLayer(
            embedding_dim=EMBEDDING_DIM,
            num_heads=NUM_HEADS,
            hidden_dim=HIDDEN_DIM,
            dropout=DROPOUT
        )
        for _ in range(num_layers)
    ])


def create_pre_norm_model(num_layers):

    return nn.ModuleList([
        PreNormEncoderLayer(
            embedding_dim=EMBEDDING_DIM,
            num_heads=NUM_HEADS,
            hidden_dim=HIDDEN_DIM,
            dropout=DROPOUT
        )
        for _ in range(num_layers)
    ])


def calculate_gradient_norm(module):

    squared_norms = []

    for parameter in module.parameters():

        if parameter.grad is not None:

            squared_norms.append(
                parameter.grad.detach().norm(2).item() ** 2
            )

    if not squared_norms:
        return 0.0

    return sum(squared_norms) ** 0.5


def collect_layer_gradients(model_type, num_layers):

    if model_type == "post_norm":
        model = create_post_norm_model(num_layers)

    elif model_type == "pre_norm":
        model = create_pre_norm_model(num_layers)

    else:
        raise ValueError("Unknown model type")

    model.train()

    input_tensor = torch.randn(
        BATCH_SIZE,
        SEQUENCE_LENGTH,
        EMBEDDING_DIM
    )

    # Forward pass
    output = input_tensor

    for layer in model:
        output = layer(output)

    # Same loss for both architectures
    loss = output.square().mean()

    # Backward pass
    model.zero_grad()
    loss.backward()

    results = []

    for layer_index, layer in enumerate(model):

        attention_module = next(
        module for module in layer.modules()
        if isinstance(module, MultiHeadAttention)
    )

        feed_forward_module = layer.feed_forward

        attention_gradient_norm = calculate_gradient_norm(
            attention_module
        )

        feed_forward_gradient_norm = calculate_gradient_norm(
            feed_forward_module
        )

        combined_gradient_norm = calculate_gradient_norm(layer)

        results.append({
            "model_type": model_type,
            "num_layers": num_layers,
            "layer_index": layer_index + 1,
            "attention_gradient_norm": attention_gradient_norm,
            "feed_forward_gradient_norm": feed_forward_gradient_norm,
            "combined_gradient_norm": combined_gradient_norm,
            "loss": loss.item()
        })

    return results


def run_experiment():

    all_results = []

    for num_layers in LAYER_COUNTS:

        for model_type in ["post_norm", "pre_norm"]:

            run_results = []

            for run in range(NUM_RUNS):

                results = collect_layer_gradients(
                    model_type=model_type,
                    num_layers=num_layers
                )

                run_results.append(results)

            # Average gradient norms across runs
            for layer_index in range(num_layers):

                averaged_result = {
                    "model_type": model_type,
                    "num_layers": num_layers,
                    "layer_index": layer_index + 1,
                    "attention_gradient_norm": sum(
                        result[layer_index]["attention_gradient_norm"]
                        for result in run_results
                    ) / NUM_RUNS,
                    "feed_forward_gradient_norm": sum(
                        result[layer_index]["feed_forward_gradient_norm"]
                        for result in run_results
                    ) / NUM_RUNS,
                    "combined_gradient_norm": sum(
                        result[layer_index]["combined_gradient_norm"]
                        for result in run_results
                    ) / NUM_RUNS,
                    "loss": sum(
                        result[layer_index]["loss"]
                        for result in run_results
                    ) / NUM_RUNS
                }

                all_results.append(averaged_result)

    return all_results


def save_results(results):

    os.makedirs("reports", exist_ok=True)

    output_path = "reports/normalization_comparison.csv"

    fieldnames = [
        "model_type",
        "num_layers",
        "layer_index",
        "attention_gradient_norm",
        "feed_forward_gradient_norm",
        "combined_gradient_norm",
        "loss"
    ]

    with open(output_path, "w", newline="") as csv_file:

        writer = csv.DictWriter(
            csv_file,
            fieldnames=fieldnames
        )

        writer.writeheader()
        writer.writerows(results)

    print(f"Results saved to: {output_path}")


if __name__ == "__main__":

    results = run_experiment()

    save_results(results)

    print("\nGradient Flow Comparison")
    print("-" * 60)

    for result in results:

        print(
            f"{result['model_type']:10} | "
            f"Layers: {result['num_layers']:2} | "
            f"Layer: {result['layer_index']:2} | "
            f"Gradient: {result['combined_gradient_norm']:.8f}"
        )
