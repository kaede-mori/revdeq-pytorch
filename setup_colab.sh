#!/bin/bash
# Colab環境セットアップスクリプト

echo "Installing dependencies for Colab..."
pip install -q torch transformers datasets accelerate tqdm wandb pyyaml tensorboard

echo "Checking GPU availability..."
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"

echo "Setup complete!"

