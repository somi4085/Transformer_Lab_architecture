
import csv
import os
import random

import numpy as np
import torch
import torch.nn as nn

from src.layers.encoder_layer import EncoderLayer
from src.layers.pre_norm_encoder_layer import PreNormEncoderLayer
from src.attention.multi_head_attention import MultiHeadAttention
from src.layers.feed_forward import FeedForwardNetwork


# --------------------------------------------------
# Experiment configuration
# --------------------------------------------------

SEED = 42

EMBEDDING_DIM = 256
NUM_HEADS = 8
HIDDEN_DIM = 1024

BATCH_SIZE = 2
SEQUENCE_LENGTH = 32

DROPOUT = 0.0

LAYER_COUNTS = [2, 4, 8, 12]

EPSILON = 1e-12


# --------------------------------------------------
# Reproducibility
# --------------------------------------------------

def set_seed(seed=SEED):

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


# --------------------------------------------------
# Model creation
# --------------------------------------------------

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


# --------------------------------------------------
# Fixed input creation
# --------------------------------------------------

def create_fixed_input():

    generator = torch.Generator(device="cpu")
    generator.manual_seed(SEED)

    return torch.randn(
        BATCH_SIZE,
        SEQUENCE_LENGTH,
        EMBEDDING_DIM,
        generator=generator
    )


# --------------------------------------------------
# Gradient norm calculation
# --------------------------------------------------

def calculate_gradient_norm(module):

    squared_norms = []

    for parameter in module.parameters():

        if parameter.grad is not None:

            gradient_norm = parameter.grad.detach().norm(2).item()

            squared_norms.append(gradient_norm ** 2)

    if not squared_norms:
        return 0.0

    return sum(squared_norms) ** 0.5


# --------------------------------------------------
# Activation norm calculation
# --------------------------------------------------

def calculate_activation_norm(activation):

    return activation.detach().norm(2).item()


# --------------------------------------------------
# Find modules by type
# --------------------------------------------------

def find_module(layer, module_type):

    matching_modules = [
        module
        for module in layer.modules()
        if isinstance(module, module_type)
    ]

    if len(matching_modules) != 1:

        raise RuntimeError(
            f"Expected exactly one {module_type.__name__}, "
            f"found {len(matching_modules)}"
        )

    return matching_modules[0]


# --------------------------------------------------
# Collect gradient and activation metrics
# --------------------------------------------------

def collect_layer_metrics(model_type, num_layers, fixed_input):

    # Reset seed before every model construction.
    set_seed(SEED)

    if model_type == "post_norm":

        model = create_post_norm_model(num_layers)

    elif model_type == "pre_norm":

        model = create_pre_norm_model(num_layers)

    else:

        raise ValueError(
            f"Unknown model type: {model_type}"
        )

    model.train()

    # Store layer outputs during forward propagation.
    activations = {}

    handles = []

    def create_activation_hook(layer_index):

        def hook(module, inputs, output):

            activations[layer_index] = calculate_activation_norm(
                output
            )

        return hook

    for index, layer in enumerate(model):

        handle = layer.register_forward_hook(
            create_activation_hook(index + 1)
        )

        handles.append(handle)

    # Forward pass
    output = fixed_input.clone()

    for layer in model:

        output = layer(output)

    # Same probe loss for both architectures.
    loss = output.square().mean()

    # Backward pass
    model.zero_grad(set_to_none=True)

    loss.backward()

    # Remove hooks after forward/backward.
    for handle in handles:

        handle.remove()

    # Collect metrics layer by layer.
    layer_metrics = []

    for index, layer in enumerate(model):

        attention_module = find_module(
            layer,
            MultiHeadAttention
        )

        feed_forward_module = find_module(
            layer,
            FeedForwardNetwork
        )

        attention_gradient_norm = calculate_gradient_norm(
            attention_module
        )

        feed_forward_gradient_norm = calculate_gradient_norm(
            feed_forward_module
        )

        combined_gradient_norm = calculate_gradient_norm(
            layer
        )

        activation_norm = activations[index + 1]

        layer_metrics.append({
            "model_type": model_type,
            "num_layers": num_layers,
            "layer_index": index + 1,
            "attention_gradient_norm": attention_gradient_norm,
            "feed_forward_gradient_norm": feed_forward_gradient_norm,
            "combined_gradient_norm": combined_gradient_norm,
            "activation_norm": activation_norm,
            "loss": loss.item()
        })

    # Use the final layer's gradient norm as reference.
    last_layer_gradient = layer_metrics[-1][
        "combined_gradient_norm"
    ]

    for metric in layer_metrics:

        metric["relative_gradient_ratio"] = (
            metric["combined_gradient_norm"]
            / (last_layer_gradient + EPSILON)
        )

    return layer_metrics


# --------------------------------------------------
# Run controlled experiment
# --------------------------------------------------

def run_experiment():

    all_results = []

    # Same fixed input for every architecture and depth.
    fixed_input = create_fixed_input()

    for num_layers in LAYER_COUNTS:

        print(f"\nTesting {num_layers} layers")

        for model_type in ["post_norm", "pre_norm"]:

            print(f"Running {model_type}...")

            results = collect_layer_metrics(
                model_type=model_type,
                num_layers=num_layers,
                fixed_input=fixed_input
            )

            all_results.extend(results)

    return all_results


# --------------------------------------------------
# Save detailed results
# --------------------------------------------------

def save_detailed_results(results):

    os.makedirs("reports", exist_ok=True)

    output_path = (
        "reports/controlled_gradient_comparison.csv"
    )

    fieldnames = [
        "model_type",
        "num_layers",
        "layer_index",
        "attention_gradient_norm",
        "feed_forward_gradient_norm",
        "combined_gradient_norm",
        "activation_norm",
        "relative_gradient_ratio",
        "loss"
    ]

    with open(
        output_path,
        "w",
        newline=""
    ) as csv_file:

        writer = csv.DictWriter(
            csv_file,
            fieldnames=fieldnames
        )

        writer.writeheader()
        writer.writerows(results)

    print(f"\nDetailed results saved: {output_path}")


# --------------------------------------------------
# Save summary results
# --------------------------------------------------

def save_summary_results(results):

    os.makedirs("reports", exist_ok=True)

    summary = []

    configurations = sorted({
        (result["model_type"], result["num_layers"])
        for result in results
    })

    for model_type, num_layers in configurations:

        current_results = [
            result
            for result in results
            if result["model_type"] == model_type
            and result["num_layers"] == num_layers
        ]

        first_layer = current_results[0]
        last_layer = current_results[-1]

        summary.append({
            "model_type": model_type,
            "num_layers": num_layers,
            "first_layer_gradient": (
                first_layer["combined_gradient_norm"]
            ),
            "last_layer_gradient": (
                last_layer["combined_gradient_norm"]
            ),
            "first_layer_relative_gradient": (
                first_layer["relative_gradient_ratio"]
            ),
            "first_layer_activation_norm": (
                first_layer["activation_norm"]
            ),
            "last_layer_activation_norm": (
                last_layer["activation_norm"]
            ),
            "loss": first_layer["loss"]
        })

    output_path = (
        "reports/controlled_gradient_summary.csv"
    )

    fieldnames = list(summary[0].keys())

    with open(
        output_path,
        "w",
        newline=""
    ) as csv_file:

        writer = csv.DictWriter(
            csv_file,
            fieldnames=fieldnames
        )

        writer.writeheader()
        writer.writerows(summary)

    print(f"Summary results saved: {output_path}")


# --------------------------------------------------
# Main
# --------------------------------------------------

if __name__ == "__main__":

    results = run_experiment()

    save_detailed_results(results)

    save_summary_results(results)

    print("\nControlled Gradient Flow Comparison")
    print("=" * 85)

    for result in results:

        print(
            f"{result['model_type']:10} | "
            f"Layers: {result['num_layers']:2} | "
            f"Layer: {result['layer_index']:2} | "
            f"Gradient: {result['combined_gradient_norm']:.8f} | "
            f"Relative: {result['relative_gradient_ratio']:.8f} | "
            f"Activation: {result['activation_norm']:.8f}"
        )
