"""
Fixed point solver for RevDEQ

Reference:
- Paper: "Reversible Deep Equilibrium Models" (arXiv:2509.12917)
"""

import torch
import torch.nn as nn
from typing import Optional, Callable, Tuple


class FixedPointSolver:
    """Solver for fixed point iterations in RevDEQ"""
    
    def __init__(
        self,
        max_iter: int = 10,
        tol: float = 1e-5,
        method: str = "anderson"
    ):
        self.max_iter = max_iter
        self.tol = tol
        self.method = method
    
    def solve(
        self,
        f: Callable,
        z0: torch.Tensor,
        *args,
        **kwargs
    ) -> Tuple[torch.Tensor, dict]:
        """
        Solve for fixed point z* such that z* = f(z*)
        
        Args:
            f: Function to find fixed point of
            z0: Initial guess
            *args: Additional arguments for f
            **kwargs: Additional keyword arguments for f
        
        Returns:
            z_star: Fixed point
            info: Dictionary with convergence information
        """
        if self.method == "anderson":
            return self._anderson_solve(f, z0, *args, **kwargs)
        else:
            return self._simple_solve(f, z0, *args, **kwargs)
    
    def _simple_solve(
        self,
        f: Callable,
        z0: torch.Tensor,
        *args,
        **kwargs
    ) -> Tuple[torch.Tensor, dict]:
        """Simple fixed point iteration"""
        z = z0
        converged = False
        
        for i in range(self.max_iter):
            z_new = f(z, *args, **kwargs)
            diff = torch.norm(z_new - z)
            
            if diff < self.tol:
                converged = True
                break
            
            z = z_new
        
        info = {
            "converged": converged,
            "iterations": i + 1,
            "final_diff": diff.item() if converged else None
        }
        
        return z, info
    
    def _anderson_solve(
        self,
        f: Callable,
        z0: torch.Tensor,
        *args,
        **kwargs
    ) -> Tuple[torch.Tensor, dict]:
        """Anderson acceleration for faster convergence"""
        z = z0
        m = 5  # Number of previous iterates to use
        residuals = []
        
        converged = False
        
        for i in range(self.max_iter):
            z_new = f(z, *args, **kwargs)
            residual = z_new - z
            diff = torch.norm(residual)
            
            if diff < self.tol:
                converged = True
                break
            
            residuals.append(residual)
            
            # Anderson acceleration
            if len(residuals) > m:
                residuals.pop(0)
            
            if len(residuals) > 1:
                # Solve for Anderson coefficients
                R = torch.stack(residuals[:-1])
                r = residuals[-1]
                
                # Compute coefficients
                try:
                    coeffs = torch.linalg.solve(R @ R.T, R @ r)
                    # Compute next iterate
                    Z = torch.stack([z] + [z - r for r in residuals[:-1]])
                    z = torch.sum(coeffs.unsqueeze(-1).unsqueeze(-1) * Z, dim=0)
                except:
                    # If solve fails, just use simple iteration
                    z = z_new
            else:
                z = z_new
        
        info = {
            "converged": converged,
            "iterations": i + 1,
            "final_diff": diff.item() if converged else None
        }
        
        return z, info

