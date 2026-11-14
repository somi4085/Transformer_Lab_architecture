# Transformer Architecture Laboratory

An educational and research-oriented implementation of the **Transformer architecture from scratch using PyTorch**. This project explores the internal components of an encoder-decoder Transformer, custom tokenization, model training, inference, and performance evaluation.

## Project Overview

The goal of this laboratory is to understand and implement the core components of the Transformer architecture while evaluating their behavior during training and inference.

The implementation includes:

- Custom Byte Pair Encoding (BPE) tokenizer
- Token embeddings and sinusoidal positional encoding
- Multi-head self-attention
- Scaled dot-product attention
- Encoder and decoder layers
- Cross-attention
- Feed-forward networks
- Residual connections and layer normalization
- Encoder-decoder Transformer architecture
- Training and checkpoint management
- Text generation through a FastAPI inference service
- KV-cache benchmarking

## Architecture

The project follows the encoder-decoder Transformer architecture.

### Encoder

Each encoder layer contains:

1. Multi-head self-attention
2. Residual connection and layer normalization
3. Position-wise feed-forward network
4. Residual connection and layer normalization

### Decoder

Each decoder layer contains:

1. Masked multi-head self-attention
2. Residual connection and layer normalization
3. Encoder-decoder cross-attention
4. Residual connection and layer normalization
5. Feed-forward network
6. Residual connection and layer normalization

### Input Embeddings

The input representation combines scaled token embeddings with sinusoidal positional encodings. Dropout is applied before the representations enter the Transformer layers.

## Tokenization

A custom BPE tokenizer is implemented to explore subword tokenization.

It supports:

- Vocabulary construction
- Subword merge learning
- Token-to-ID and ID-to-token mappings
- Encoding and decoding
- Special tokens: PAD, UNK, BOS, EOS
- Padding and truncation
- Attention-mask generation
- Batch encoding

## Training

The project uses the TinyStories dataset for experimentation.

A small smoke-training run was performed to validate the training pipeline, model execution, checkpoint saving, and inference workflow.

**Note:** The smoke-training run is an implementation validation experiment, not a claim of production-quality language generation.

## Inference API

A FastAPI service provides an interface for text generation.

Available endpoints:

| Method | Endpoint | Purpose |
|---|---|---|
| GET | `/health` | Check service and model status |
| POST | `/generate` | Generate text from an input prompt |
| GET | `/docs` | Explore the API using Swagger UI |

Run locally using Uvicorn:

```bash
uvicorn src.api.main:app --host 127.0.0.1 --port 8000
```

Open the interactive API documentation:

```text
http://127.0.0.1:8000/docs
```

## KV-Cache Benchmark

The project includes an experiment comparing autoregressive generation with and without KV caching.

| Metric | Result |
|---|---:|
| Non-cached latency | 0.16584 seconds |
| Cached latency | 0.12162 seconds |
| Observed speedup | Approximately 1.36x |
| KV-cache memory | Approximately 350 KB |

The benchmark showed reduced generation latency with KV caching in the tested configuration. Results are specific to that experiment and hardware environment.

## Repository Structure

```text
Transformer_Lab_architecture/
│
├── artifacts/
│   ├── tokenizer.pkl
│   └── benchmark and evaluation results
│
├── checkpoints/
│   └── tinystories_smoke/
│       ├── best.pt
│       └── latest.pt
│
├── evalaution/
│
├── reports/
│
├── scripts/
│
├── src/
│   ├── api/
│   ├── attention/
│   ├── embeddings/
│   ├── layers/
│   ├── tokenization/
│   └── ...
│
├── tests/
│
├── requirements.txt
└── README.md
```

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/somi4085/Transformer_Lab_tokenizer.git
cd Transformer_Lab_tokenizer
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

Activate it on Windows:

```powershell
.venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run tests

```bash
pytest tests -v
```

### 5. Start the inference API

```bash
uvicorn src.api.main:app --host 127.0.0.1 --port 8000
```

## Current Experimental Scope

This repository currently demonstrates:

- Transformer component implementation
- Unit and integration testing
- Small-scale training validation
- Checkpoint saving and loading
- Local text-generation API
- KV-cache performance comparison

The current smoke-trained model is experimental. Its generation quality has not been established as production-ready.

## Technology Stack

- Python
- PyTorch
- NumPy
- FastAPI
- Uvicorn
- Pytest
- Custom BPE tokenizer
- TinyStories dataset

## Project Objective

The long-term objective is to build a deeper understanding of Transformer internals, evaluate architectural and inference optimizations, and progressively extend the implementation toward scalable and reliable language-model systems.

---

