"""
Inference script for RevDEQ

Reference:
- Paper: "Reversible Deep Equilibrium Models" (arXiv:2509.12917)
"""

import argparse
import torch
from transformers import AutoTokenizer
from revdeq import RevDEQ, RevDEQConfig


def load_model(model_path: str, device: str = "cuda"):
    """Load RevDEQ model from checkpoint"""
    checkpoint = torch.load(model_path, map_location=device)
    config = checkpoint.get("config", RevDEQConfig())
    model = RevDEQ(config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model, config


def main():
    parser = argparse.ArgumentParser(description="Run inference with RevDEQ")
    parser.add_argument("--model_path", type=str, required=True,
                       help="Path to model checkpoint")
    parser.add_argument("--tokenizer", type=str, default="gpt2",
                       help="Tokenizer name")
    parser.add_argument("--text", type=str, default="The quick brown fox",
                       help="Input text for generation")
    parser.add_argument("--max_length", type=int, default=100,
                       help="Maximum generation length")
    parser.add_argument("--temperature", type=float, default=1.0,
                       help="Sampling temperature")
    parser.add_argument("--top_k", type=int, default=50,
                       help="Top-k sampling")
    parser.add_argument("--top_p", type=float, default=0.9,
                       help="Top-p (nucleus) sampling")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                       help="Device to use")
    
    args = parser.parse_args()
    
    # Load model
    print(f"Loading model from {args.model_path}")
    model, config = load_model(args.model_path, args.device)
    
    # Load tokenizer
    print(f"Loading tokenizer: {args.tokenizer}")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Encode input
    input_ids = tokenizer.encode(args.text, return_tensors="pt").to(args.device)
    
    # Generate
    print(f"Generating text...")
    with torch.no_grad():
        generated = model.generate(
            input_ids,
            max_length=args.max_length,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p
        )
    
    # Decode
    generated_text = tokenizer.decode(generated[0], skip_special_tokens=True)
    print(f"\nGenerated text:\n{generated_text}")


if __name__ == "__main__":
    main()

