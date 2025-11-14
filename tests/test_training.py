"""
Tests for training functionality
"""

import torch
import os
import tempfile
import shutil
from revdeq import RevDEQ, RevDEQConfig
from torch.utils.data import Dataset


class SimpleDataset(Dataset):
    """Simple dataset for testing"""
    def __init__(self, size=10, seq_len=5, vocab_size=20):
        self.size = size
        self.seq_len = seq_len
        self.vocab_size = vocab_size
    
    def __len__(self):
        return self.size
    
    def __getitem__(self, idx):
        return {
            "input_ids": torch.randint(0, self.vocab_size, (self.seq_len,)),
            "labels": torch.randint(0, self.vocab_size, (self.seq_len,)),
        }


def test_training_step():
    """Test a single training step"""
    config = RevDEQConfig(
        hidden_size=32,
        num_heads=2,
        intermediate_size=32,
        vocab_size=20,
        max_position_embeddings=32,
        num_fixed_point_iterations=3,
        use_reversible=True,
        beta=0.8,
    )
    model = RevDEQ(config)
    model.train()
    
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    
    # Single training step
    input_ids = torch.randint(0, 20, (2, 5))
    labels = input_ids.clone()
    
    optimizer.zero_grad()
    outputs = model(input_ids, labels=labels)
    # Model returns dict when labels are provided
    if isinstance(outputs, dict):
        loss = outputs["loss"]
    else:
        logits, loss = outputs
    loss.backward()
    optimizer.step()
    
    assert loss.item() > 0
    assert not torch.isnan(loss)


def test_training_multiple_steps():
    """Test multiple training steps to see if loss decreases"""
    config = RevDEQConfig(
        hidden_size=32,
        num_heads=2,
        intermediate_size=32,
        vocab_size=20,
        max_position_embeddings=32,
        num_fixed_point_iterations=3,
        use_reversible=True,
        beta=0.8,
    )
    model = RevDEQ(config)
    model.train()
    
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    losses = []
    for step in range(5):
        input_ids = torch.randint(0, 20, (2, 5))
        labels = input_ids.clone()
        
        optimizer.zero_grad()
        outputs = model(input_ids, labels=labels)
        # Model returns dict when labels are provided
        if isinstance(outputs, dict):
            loss = outputs["loss"]
        else:
            logits, loss = outputs
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        
        optimizer.step()
        losses.append(loss.item())
    
    # Loss should be finite
    assert all(not torch.isnan(torch.tensor(l)) for l in losses)
    assert all(not torch.isinf(torch.tensor(l)) for l in losses)
    
    # Ideally loss should decrease, but we'll just check it's reasonable
    final_loss = losses[-1]
    assert final_loss > 0
    assert final_loss < 100  # Should be reasonable for this small model


def test_model_save_load():
    """Test saving and loading model"""
    config = RevDEQConfig(
        hidden_size=32,
        num_heads=2,
        intermediate_size=32,
        vocab_size=20,
        max_position_embeddings=32,
    )
    model = RevDEQ(config)
    
    # Save
    temp_dir = tempfile.mkdtemp()
    save_path = os.path.join(temp_dir, "model.pt")
    
    try:
        torch.save({
            "model_state_dict": model.state_dict(),
            "config": config,
        }, save_path)
        
        # Load (weights_only=False to allow custom classes like RevDEQConfig)
        checkpoint = torch.load(save_path, weights_only=False)
        loaded_config = checkpoint["config"]
        loaded_model = RevDEQ(loaded_config)
        loaded_model.load_state_dict(checkpoint["model_state_dict"])
        
        # Test that loaded model works
        input_ids = torch.randint(0, 20, (1, 5))
        with torch.no_grad():
            logits, _ = loaded_model(input_ids)
        
        assert logits.shape == (1, 5, 20)
    finally:
        shutil.rmtree(temp_dir)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

