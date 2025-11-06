# RevDEQ-PyTorch

PyTorch implementation of Reversible Deep Equilibrium Models (RevDEQ).

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

**Note**: This is a PyTorch reimplementation inspired by the original JAX/Equinox implementation. See [Credits](#credits) for details.

## Overview

Reversible Deep Equilibrium Models (RevDEQ) are a type of Deep Equilibrium Models (DEQ) that define model outputs as fixed points of learned functions. RevDEQ enables exact gradient computation, requires no regularization, and requires fewer function evaluations than DEQ.

## Features

- **Memory Efficient**: Reversible gradient computation enables memory-efficient training
- **Exact Gradients**: Exact gradient computation through reversible fixed-point iteration
- **Fewer Function Evaluations**: Converges with fewer function evaluations than DEQ
- **PyTorch Implementation**: Pure PyTorch implementation integrated with transformers library

## Requirements

- Python 3.10+
- PyTorch 2.0+
- CUDA (recommended for GPU, works on CPU)

## Installation

### Using Docker (Recommended)

```bash
# Build and start container
docker-compose up -d

# Enter container
docker-compose exec revdeq bash

# Run training
python train.py --config configs/default.yaml
```

### Local Installation

```bash
# Using uv (recommended)
uv sync

# Using pip
pip install -r requirements.txt
pip install -e .
```

## Quick Start

### Training

```bash
# Train with default configuration
python train.py --config configs/default.yaml

# Train with custom dataset
python train.py \
    --config configs/default.yaml \
    --dataset wikitext \
    --dataset_config wikitext-2-raw-v1 \
    --output_dir checkpoints
```

### Inference

```bash
# Generate text with trained model
python inference.py \
    --model_path checkpoints/model.pt \
    --text "The quick brown fox" \
    --max_length 100
```

### Testing

```bash
# Run all tests
python -m pytest tests/ -v

# Or using the test script
./run_tests.sh
```

The test suite verifies:
- Model initialization and forward pass
- Gradient computation (reversible function)
- Training step execution
- Model save/load functionality
- No NaN/Inf values in gradients

## Project Structure

```
RevDEQ/
├── revdeq/              # Main module
│   ├── __init__.py
│   └── model.py         # RevDEQ model implementation
├── notebooks/           # Jupyter notebooks
│   └── revdeq_colab.ipynb
├── configs/             # Configuration files
│   ├── default.yaml
│   └── small_test.yaml
├── tests/               # Test suite
│   ├── test_model.py
│   └── test_training.py
├── train.py            # Training script
├── inference.py        # Inference script
├── Dockerfile          # Docker image definition
├── docker-compose.yml  # Docker Compose configuration
├── requirements.txt    # Python dependencies
└── pyproject.toml     # Project configuration
```

## Configuration

Edit `configs/default.yaml` to customize model and training settings:

- `hidden_size`: Hidden layer size
- `num_layers`: Number of layers
- `num_fixed_point_iterations`: Maximum fixed point iterations
- `fixed_point_tol`: Convergence tolerance
- `beta`: Relaxation parameter for reversible updates (0 < beta <= 1)
- `learning_rate`: Learning rate
- `batch_size`: Batch size

## Google Colab

This repository works on Google Colab. Open `notebooks/revdeq_colab.ipynb` in Colab to run experiments and verify that loss reaches around 25.

The Colab notebook includes:
- Repository cloning from GitHub
- Loss tracking and visualization
- Training progress monitoring
- Target loss achievement verification

## Model Architecture

RevDEQ is a Transformer-based language model that uses **fixed-point iteration** instead of stacking multiple layers:

```
RevDEQ Model:
├── Token Embedding (vocab_size → hidden_size)
├── Position Embedding (max_position → hidden_size)
├── Reversible Fixed Point Iteration
│   └── RevDEQLayer (single layer applied repeatedly)
│       ├── Self-Attention (Multi-head)
│       └── Feed-Forward Network
├── Layer Normalization
└── Language Model Head (hidden_size → vocab_size)
```

**Key Features:**
- **Fixed Point Iteration**: One layer is applied repeatedly until convergence, instead of stacking multiple layers
- **Reversible Gradient Computation**: Memory-efficient gradient computation using `ReversibleFunction`
- **Parameter Efficiency**: Sharing one layer reduces the number of parameters

## Training Data

**Default Dataset**: `wikitext/wikitext-2-raw-v1`
- Wikipedia articles dataset (~36,000 training examples)
- Tokenized with GPT-2 tokenizer
- Task: Next Token Prediction (language modeling)

## Implementation Details

This implementation follows the RevDEQ paper's algorithm:

1. **Reversible Updates**: Uses two states (y, z) with relaxation parameter β:
   - `y_{n+1} = (1 - β) * y_n + β * f(z_n)`
   - `z_{n+1} = (1 - β) * z_n + β * f(y_{n+1})`

2. **Reversible Gradient Computation**: Implements the reversible backpropagation algorithm using `torch.autograd.Function` to compute exact gradients without storing all intermediate states.

3. **Memory Efficiency**: The reversible implementation allows training with constant memory regardless of the number of fixed-point iterations.

## Comparison with Original Implementation

The original implementation ([sammccallum/reversible-deq](https://github.com/sammccallum/reversible-deq)) uses JAX/Equinox and provides a general-purpose solver API. This PyTorch implementation:

- **Minimal Changes**: Only adapts the algorithm to PyTorch's autograd system
- **Model Integration**: Directly integrates with PyTorch models and transformers library
- **Same Algorithm**: Follows the same reversible update equations and gradient computation from the paper

## Troubleshooting

### Memory Issues
- Reduce `batch_size`
- Increase `gradient_accumulation_steps`
- Reduce `num_fixed_point_iterations`

### Slow Training
- Use GPU (set `CUDA_VISIBLE_DEVICES=0`)
- Enable `fp16` in config
- Adjust `dataloader_num_workers`

## License

MIT License - see [LICENSE](LICENSE) file for details.

## Credits

This implementation is inspired by:

- **Paper**: "Reversible Deep Equilibrium Models" ([arXiv:2509.12917](https://arxiv.org/abs/2509.12917))
- **Original Implementation**: [sammccallum/reversible-deq](https://github.com/sammccallum/reversible-deq) (JAX/Equinox, Apache-2.0 License)

This implementation follows the paper's algorithm and is independently implemented in PyTorch. The original JAX/Equinox implementation's design principles were referenced, but the code is written from scratch.

## Contributing

Pull requests and issues are welcome!
