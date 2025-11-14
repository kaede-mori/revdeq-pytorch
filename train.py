"""
Training script for RevDEQ using transformers' SFT trainer

Reference:
- Paper: "Reversible Deep Equilibrium Models" (arXiv:2509.12917)
"""

import os
import argparse
import yaml
import torch
from torch.utils.data import Dataset
from transformers import (
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
    AutoTokenizer,
    AutoConfig
)
from datasets import load_dataset
from revdeq import RevDEQ, RevDEQConfig


class RevDEQDataset(Dataset):
    """Dataset wrapper for RevDEQ training"""
    
    def __init__(self, texts, tokenizer, max_length=512):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze(),
            "labels": encoding["input_ids"].squeeze()
        }


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file"""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    
    # Convert string scientific notation to float
    float_keys = ["learning_rate", "fixed_point_tol"]
    for key in float_keys:
        if key in config and isinstance(config[key], str):
            try:
                config[key] = float(config[key])
            except ValueError:
                pass
    
    return config


def prepare_dataset(dataset_name: str = "Salesforce/wikitext", dataset_config: str = "wikitext-103-v1", max_texts: int = None):
    """Prepare dataset for training"""
    print(f"Loading dataset: {dataset_name}/{dataset_config}")
    dataset = load_dataset(dataset_name, dataset_config, split="train")
    
    # Extract texts
    texts = []
    for example in dataset:
        if "text" in example and len(example["text"].strip()) > 0:
            texts.append(example["text"])
            if max_texts is not None and len(texts) >= max_texts:
                break
    
    return texts


def main():
    parser = argparse.ArgumentParser(description="Train RevDEQ model")
    parser.add_argument("--config", type=str, default="configs/default.yaml",
                       help="Path to configuration file")
    parser.add_argument("--dataset", type=str, default="Salesforce/wikitext",
                       help="Dataset name")
    parser.add_argument("--dataset_config", type=str, default="wikitext-103-v1",
                       help="Dataset configuration")
    parser.add_argument("--output_dir", type=str, default="./checkpoints",
                       help="Output directory for checkpoints")
    parser.add_argument("--resume_from_checkpoint", type=str, default=None,
                       help="Path to checkpoint to resume from")
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config) if os.path.exists(args.config) else {}
    
    # Model configuration
    model_config = RevDEQConfig(
        hidden_size=config.get("hidden_size", 768),
        num_heads=config.get("num_heads", 12),
        intermediate_size=config.get("intermediate_size", 3072),
        max_position_embeddings=config.get("max_position_embeddings", 512),
        vocab_size=config.get("vocab_size", 50257),
        num_fixed_point_iterations=config.get("num_fixed_point_iterations", 10),
        fixed_point_tol=config.get("fixed_point_tol", 1e-5),
        use_reversible=config.get("use_reversible", True),
    )
    
    # Initialize tokenizer
    tokenizer_name = config.get("tokenizer", "gpt2")
    print(f"Loading tokenizer: {tokenizer_name}")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Update vocab size if needed
    model_config.vocab_size = len(tokenizer)
    
    # Initialize model
    print("Initializing RevDEQ model...")
    model = RevDEQ(model_config)
    
    # Prepare dataset
    max_texts = config.get("max_texts", None)  # Limit dataset size for testing
    texts = prepare_dataset(args.dataset, args.dataset_config, max_texts=max_texts)
    train_dataset = RevDEQDataset(texts, tokenizer, max_length=model_config.max_position_embeddings)
    print(f"Dataset size: {len(train_dataset)} examples")
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=config.get("num_epochs", 3),
        per_device_train_batch_size=config.get("batch_size", 4),
        gradient_accumulation_steps=config.get("gradient_accumulation_steps", 4),
        learning_rate=config.get("learning_rate", 5e-5),
        weight_decay=config.get("weight_decay", 0.01),
        warmup_steps=config.get("warmup_steps", 1000),
        logging_dir=os.path.join(args.output_dir, "logs"),
        logging_steps=config.get("logging_steps", 100),
        save_steps=config.get("save_steps", 1000),
        save_total_limit=config.get("save_total_limit", 3),
        eval_strategy="no",
        save_strategy="no",  # We'll save manually to avoid safetensors issue
        load_best_model_at_end=False,
        fp16=config.get("fp16", torch.cuda.is_available()),
        bf16=config.get("bf16", False),
        dataloader_num_workers=config.get("dataloader_num_workers", 4),
        report_to=[] if config.get("report_to") == "none" else config.get("report_to", "tensorboard"),
        remove_unused_columns=False,
        push_to_hub=False,
    )
    
    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )
    
    # Custom Trainer class to handle our model's output format
    # Note: Our model returns dict with 'loss' and 'logits' when labels are provided
    class RevDEQTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
            """Custom loss computation for RevDEQ"""
            labels = inputs.get("labels")
            outputs = model(**inputs)
            
            if isinstance(outputs, dict):
                loss = outputs.get("loss")
                if return_outputs:
                    return loss, outputs
                return loss
            elif isinstance(outputs, tuple):
                logits, loss = outputs
                if return_outputs:
                    return loss, {"logits": logits}
                return loss
            else:
                # Fallback - should not happen
                if return_outputs:
                    return None, outputs
                return None
    
    # Use custom trainer
    trainer = RevDEQTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
    )
    
    # Train
    print("Starting training...")
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    
    # Save final model
    print(f"Saving final model to {args.output_dir}")
    
    # Save using torch.save
    os.makedirs(args.output_dir, exist_ok=True)
    torch.save({
        "model_state_dict": model.state_dict(),
        "config": model_config,
    }, os.path.join(args.output_dir, "model.pt"))
    
    tokenizer.save_pretrained(args.output_dir)
    
    print("Model saved successfully!")


if __name__ == "__main__":
    main()

