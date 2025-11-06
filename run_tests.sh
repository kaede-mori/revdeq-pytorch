#!/bin/bash
# Test execution script

echo "=== RevDEQ Implementation Tests ==="
echo ""

# Check if torch is installed
python -c "import torch" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "Error: PyTorch is not installed."
    echo "Please install dependencies: pip install -r requirements.txt"
    exit 1
fi

# Run tests
echo "Running model tests..."
python -m pytest tests/test_model.py -v

echo ""
echo "Running training tests..."
python -m pytest tests/test_training.py -v

echo ""
echo "=== All tests completed ==="

