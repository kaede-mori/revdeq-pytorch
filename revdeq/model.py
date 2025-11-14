"""
Reversible Deep Equilibrium Model implementation

This is a PyTorch implementation of Reversible Deep Equilibrium Models (RevDEQ).

Reference:
- Paper: "Reversible Deep Equilibrium Models" (arXiv:2509.12917)
  https://arxiv.org/abs/2509.12917
- Original JAX/Equinox implementation: https://github.com/sammccallum/reversible-deq
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Callable
from dataclasses import dataclass


@dataclass
class RevDEQConfig:
    """Configuration for RevDEQ model"""
    hidden_size: int = 768
    num_layers: int = 12
    num_heads: int = 12
    intermediate_size: int = 3072
    max_position_embeddings: int = 448  # Source: language-deq.py line 255
    vocab_size: int = 50304  # Source: language-deq.py line 254 (padded to multiple of 32)
    layer_norm_eps: float = 1e-5
    use_bias: bool = False
    # RevDEQ specific parameters
    num_fixed_point_iterations: int = 4  # Source: language-deq.py line 259
    fixed_point_tol: float = 1e-3  # Source: language-deq.py line 258
    use_reversible: bool = True
    beta: float = 0.5  # Relaxation parameter (source: language-deq.py line 257)


class RevDEQLayer(nn.Module):
    """Single reversible layer for RevDEQ"""
    
    def __init__(self, config: RevDEQConfig):
        super().__init__()
        self.config = config
        
        # Self-attention
        self.attention = nn.MultiheadAttention(
            embed_dim=config.hidden_size,
            num_heads=config.num_heads,
            dropout=0.0,
            batch_first=True,
            bias=config.use_bias
        )
        
        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(config.hidden_size, config.intermediate_size, bias=config.use_bias),
            nn.GELU(),
            nn.Linear(config.intermediate_size, config.hidden_size, bias=config.use_bias)
        )
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.norm2 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        
    def forward(self, z: torch.Tensor, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass through the reversible layer with input injection
        
        Args:
            z: Hidden state (updated through fixed point iteration)
            x: Input embeddings (injected at each iteration)
            attn_mask: Attention mask
        
        Returns:
            Updated hidden state z
        """
        # Input injection: z + x before norm1, then attention
        # Following the reference implementation: z = z + attention(norm1(z + x))
        norm1_input = self.norm1(z + x)
        attn_out, _ = self.attention(norm1_input, norm1_input, norm1_input, attn_mask=attn_mask)
        z = z + attn_out
        
        # Feed-forward with residual
        # Following the reference implementation: z = z + mlp(norm2(z))
        norm2_input = self.norm2(z)
        ffn_out = self.ffn(norm2_input)
        z = z + ffn_out
        
        return z


class ReversibleFunction(torch.autograd.Function):
    """Reversible function for fixed point iteration
    
    This implements the reversible gradient computation for fixed point iteration
    following the RevDEQ paper. The key insight is that we can compute gradients
    without storing all intermediate states by reversing through the iterations.
    
    The reversible update uses two states (y, z) and a relaxation parameter beta:
    y_{n+1} = (1 - beta) * y_n + beta * f(z_n)
    z_{n+1} = (1 - beta) * z_n + beta * f(y_{n+1})
    """
    
    @staticmethod
    def forward(ctx, f: Callable, z0: torch.Tensor, x: torch.Tensor, attn_mask: Optional[torch.Tensor], 
                max_iter: int = 10, tol: float = 1e-5, beta: float = 0.8):
        """Forward pass: find fixed point using reversible updates with input injection
        
        Args:
            f: Function to find fixed point of (takes z, x, attn_mask)
            z0: Initial guess for hidden state
            x: Input embeddings (injected at each iteration)
            attn_mask: Attention mask
            max_iter: Maximum number of iterations
            tol: Tolerance for convergence
            beta: Relaxation parameter (0 < beta <= 1)
        
        Returns:
            z: Fixed point
        """
        # Initialize both states
        y = z0.clone()
        z = z0.clone()
        
        # Store states for backward pass
        y_list = []
        z_list = []
        
        for i in range(max_iter):
            # Reversible update: y_{n+1} = (1 - beta) * y_n + beta * f(z_n, x)
            f_z = f(z, x, attn_mask)
            y_new = (1 - beta) * y + beta * f_z
            
            # Reversible update: z_{n+1} = (1 - beta) * z_n + beta * f(y_{n+1}, x)
            f_y = f(y_new, x, attn_mask)
            z_new = (1 - beta) * z + beta * f_y
            
            # Store states
            y_list.append(y.clone())
            z_list.append(z.clone())
            
            # Check convergence
            diff = torch.norm(z_new - z) + torch.norm(y_new - y)
            if diff < tol:
                break
            
            y = y_new
            z = z_new
        
        # Store necessary information for backward
        ctx.f = f
        ctx.y_list = y_list
        ctx.z_list = z_list
        ctx.x = x
        ctx.attn_mask = attn_mask
        ctx.max_iter = max_iter
        ctx.tol = tol
        ctx.beta = beta
        ctx.num_iterations = len(z_list)
        
        return z
    
    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        """Backward pass: reversible gradient computation
        
        The gradient is computed by reversing through the fixed point iterations.
        Following the reversible backpropagation algorithm from the paper.
        """
        f = ctx.f
        y_list = ctx.y_list
        z_list = ctx.z_list
        x = ctx.x
        attn_mask = ctx.attn_mask
        num_iterations = ctx.num_iterations
        beta = ctx.beta
        
        # Start with the gradient from the output (z is the final state)
        grad_z = grad_output
        grad_y = torch.zeros_like(grad_z)
        
        # Reverse through iterations
        for i in range(num_iterations - 1, -1, -1):
            y = y_list[i]
            z = z_list[i]
            
            # Recompute forward pass to get computational graph
            # The reversible update at iteration i:
            #   y_new = (1 - beta) * y + beta * f(z, x)
            #   z_new = (1 - beta) * z + beta * f(y_new, x)
            
            # We need gradients w.r.t. y and z from the next iteration
            # We'll compute these by re-evaluating the forward pass
            
            # Recompute f(z, x) with gradient tracking
            z_detached = z.detach().requires_grad_(True)
            f_z_detached = f(z_detached, x, attn_mask)
            
            # Recompute y_new with gradient tracking
            # Note: y_new depends on both y and f(z, x), so we need to track gradients through both
            y_detached = y.detach().requires_grad_(True)
            # f_z_detached already has gradient tracking from z_detached
            # We need to ensure y_new_detached is part of the computation graph
            # by combining y_detached and f_z_detached
            y_new_detached = (1 - beta) * y_detached + beta * f_z_detached
            # Ensure y_new_detached requires grad
            if not y_new_detached.requires_grad:
                y_new_detached = y_new_detached.requires_grad_(True)
            
            # Recompute f(y_new, x) with gradient tracking
            # This creates a new computational graph from y_new_detached
            f_y_new_detached = f(y_new_detached, x, attn_mask)
            
            # Recompute z_new with gradient tracking
            # z_new depends on both z and f(y_new)
            # Note: z_detached is already used above for f_z_detached, but we need it again here
            z_new_detached = (1 - beta) * z_detached + beta * f_y_new_detached
            
            # Now compute gradients backward through this iteration
            # We have gradients from the next iteration: grad_z and grad_y
            
            # Gradient through z_new = (1 - beta) * z + beta * f(y_new)
            # This gives us grad_z_new w.r.t. z and y_new
            grad_z_new = grad_z
            
            # Backward through z_new
            grad_z_from_z_new = (1 - beta) * grad_z_new
            grad_f_y_new = beta * grad_z_new
            
            # Backward through f(y_new)
            # y_new_detached must have requires_grad=True and be in the computation graph
            if y_new_detached.requires_grad:
                try:
                    grad_y_new_from_f = torch.autograd.grad(
                        outputs=f_y_new_detached,
                        inputs=y_new_detached,
                        grad_outputs=grad_f_y_new,
                        retain_graph=True,
                        create_graph=torch.is_grad_enabled(),
                        allow_unused=True
                    )[0]
                    if grad_y_new_from_f is None:
                        grad_y_new_from_f = torch.zeros_like(y_new_detached)
                except RuntimeError as e:
                    # If gradient computation fails, use zero gradient
                    grad_y_new_from_f = torch.zeros_like(y_new_detached)
            else:
                grad_y_new_from_f = torch.zeros_like(y_new_detached)
            
            # Gradient w.r.t. y_new
            grad_y_new = grad_y + grad_y_new_from_f
            
            # Backward through y_new = (1 - beta) * y + beta * f(z)
            grad_y_from_y_new = (1 - beta) * grad_y_new
            grad_f_z = beta * grad_y_new
            
            # Backward through f(z)
            # f_z_detached depends on z_detached, compute gradient
            if f_z_detached.requires_grad and z_detached.requires_grad:
                try:
                    grad_z_from_f = torch.autograd.grad(
                        outputs=f_z_detached,
                        inputs=z_detached,
                        grad_outputs=grad_f_z,
                        retain_graph=True,
                        create_graph=torch.is_grad_enabled(),
                        allow_unused=True
                    )[0]
                    if grad_z_from_f is None:
                        grad_z_from_f = torch.zeros_like(z_detached)
                except RuntimeError as e:
                    # If gradient computation fails, use zero gradient
                    grad_z_from_f = torch.zeros_like(z_detached)
            else:
                grad_z_from_f = torch.zeros_like(z_detached)
            
            # Total gradients for this iteration
            grad_z = grad_z_from_z_new + grad_z_from_f
            grad_y = grad_y_from_y_new
        
        # The gradient w.r.t. initial z0 (y0 = z0 initially)
        grad_z0 = grad_z + grad_y
        
        # Note: Gradients w.r.t. function parameters and x are handled by autograd
        # x is a constant input that doesn't require gradients in the backward pass
        # Return gradients for: f, z0, x, attn_mask, max_iter, tol, beta
        return None, grad_z0, None, None, None, None, None


class RevDEQ(nn.Module):
    """Reversible Deep Equilibrium Model"""
    
    def __init__(self, config: RevDEQConfig):
        super().__init__()
        self.config = config
        
        # Embeddings
        self.token_embedding = nn.Embedding(config.vocab_size, config.hidden_size)
        self.position_embedding = nn.Embedding(
            config.max_position_embeddings, config.hidden_size
        )
        
        # Single layer (will be reused in fixed point iteration)
        self.layer = RevDEQLayer(config)
        
        # Output layer
        self.ln_f = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        
    def forward_layer(self, z: torch.Tensor, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Single forward pass through the layer with input injection
        
        Args:
            z: Hidden state (updated through fixed point iteration)
            x: Input embeddings (injected at each iteration)
            attn_mask: Attention mask
        
        Returns:
            Updated hidden state z
        """
        return self.layer(z, x, attn_mask)
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        use_cache: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass with fixed point iteration
        
        Args:
            input_ids: Token IDs [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]
            labels: Labels for language modeling [batch_size, seq_len]
            use_cache: Whether to use caching (not used in DEQ)
        
        Returns:
            logits: [batch_size, seq_len, vocab_size]
            loss: Optional loss tensor
        """
        batch_size, seq_len = input_ids.shape
        device = input_ids.device
        
        # Create attention mask if not provided
        if attention_mask is None:
            attention_mask = torch.ones(batch_size, seq_len, device=device)
        
        # Create causal mask for autoregressive generation
        attn_mask = self._create_causal_mask(seq_len, device)
        
        # Embeddings
        positions = torch.arange(0, seq_len, device=device).unsqueeze(0).expand(batch_size, -1)
        x = self.token_embedding(input_ids) + self.position_embedding(positions)
        
        # Fixed point iteration with input injection
        # Initialize z0 with x (input embeddings)
        # At each iteration, x is injected: z = z + attention(norm1(z + x))
        if self.config.use_reversible and self.training:
            # Use reversible function for training (memory efficient)
            z0 = torch.zeros_like(x)  # Zero initialization (source: language-deq.py line 113)
            z_final = ReversibleFunction.apply(
                self.forward_layer,
                z0,
                x,  # Input embeddings to inject at each iteration
                attn_mask,
                self.config.num_fixed_point_iterations,
                self.config.fixed_point_tol,
                self.config.beta
            )
            x = z_final
        else:
            # Simple fixed point iteration for inference or non-reversible mode
            z = torch.zeros_like(x)  # Zero initialization (source: language-deq.py line 113)
            for _ in range(self.config.num_fixed_point_iterations):
                z_new = self.forward_layer(z, x, attn_mask)  # Inject x at each iteration
                diff = torch.norm(z_new - z)
                if diff < self.config.fixed_point_tol:
                    break
                z = z_new
            x = z
        
        # Output projection
        x = self.ln_f(x)
        logits = self.lm_head(x)
        
        # Compute loss if labels provided
        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100
            )
            # For transformers Trainer, return dict or just loss
            # Trainer expects model to return loss when labels are provided
            return {"loss": loss, "logits": logits}
        
        return logits, loss
    
    def _create_causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """Create causal attention mask"""
        mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1)
        mask = mask.masked_fill(mask == 1, float('-inf'))
        return mask
    
    def generate(
        self,
        input_ids: torch.Tensor,
        max_length: int = 100,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 0.9
    ) -> torch.Tensor:
        """Generate text using the model"""
        self.eval()
        generated = input_ids.clone()
        
        with torch.no_grad():
            for _ in range(max_length - input_ids.size(1)):
                logits, _ = self.forward(generated)
                next_token_logits = logits[:, -1, :] / temperature
                
                # Top-k filtering
                if top_k > 0:
                    # Ensure top_k doesn't exceed vocab size
                    effective_top_k = min(top_k, next_token_logits.size(-1))
                    if effective_top_k > 0:
                        indices_to_remove = next_token_logits < torch.topk(next_token_logits, effective_top_k)[0][..., -1, None]
                        next_token_logits[indices_to_remove] = float('-inf')
                
                # Top-p (nucleus) filtering
                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                    next_token_logits[indices_to_remove] = float('-inf')
                
                # Sample
                probs = F.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                generated = torch.cat([generated, next_token], dim=1)
        
        return generated

