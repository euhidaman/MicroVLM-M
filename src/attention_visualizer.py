"""
Attention Visualization Module
Implements SlicingUnivariateTest and FastEppsPulley for analyzing cross-attention
between image patches and text tokens
"""

import torch
import torch.nn as nn
import json
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

class FastEppsPulley(nn.Module):
    """
    Fast Epps-Pulley test statistic for univariate normality testing
    """
    
    def __init__(self, t_max=3.0, n_points=17, integration='trapezoid'):
        super().__init__()
        
        assert n_points % 2 == 1, "n_points must be odd"
        
        self.integration = integration
        self.n_points = n_points
        
        # Linearly spaced points [0, t_max]
        t = torch.linspace(0, t_max, n_points, dtype=torch.float32)
        self.register_buffer('t', t)
        
        # Integration weights (trapezoidal rule)
        dt = t_max / (n_points - 1)
        weights = torch.full((n_points,), 2 * dt, dtype=torch.float32)
        weights[[0, -1]] = dt  # Half weight at endpoints
        
        # Precompute phi(t) = exp(-t^2/2) for standard normal
        phi = torch.exp(-0.5 * t ** 2)
        self.register_buffer('phi', phi)
        self.register_buffer('weights', weights * phi)
    
    def forward(self, x):
        """
        Compute Epps-Pulley statistic
        
        Args:
            x: (*, N, K) samples where N is number of samples, K is number of slices
        
        Returns:
            statistic: (*, K) test statistics
        """
        N = x.size(-2)
        
        # Compute empirical characteristic function
        x_t = x.unsqueeze(-1) * self.t  # (*, N, K, n_points)
        cos_vals = torch.cos(x_t)
        sin_vals = torch.sin(x_t)
        
        # Mean across samples
        cos_mean = cos_vals.mean(-3)  # (*, K, n_points)
        sin_mean = sin_vals.mean(-3)  # (*, K, n_points)
        
        # Compute error vs standard normal
        err = (cos_mean - self.phi) ** 2 + sin_mean ** 2
        
        # Weighted integration
        statistic = (err @ self.weights) * N
        
        return statistic.squeeze(-1)  # (*, K)


class SlicingUnivariateTest(nn.Module):
    """
    Multivariate test using random slicing and univariate statistics
    """
    
    def __init__(self, univariate_test, num_slices=256, reduction='mean',
                 sampler='gaussian', clip_value=None):
        super().__init__()
        
        self.univariate_test = univariate_test
        self.num_slices = num_slices
        self.reduction = reduction
        self.sampler = sampler
        self.clip_value = clip_value
        
        self.register_buffer('global_step', torch.zeros((), dtype=torch.long))
        self._generator = None
        self._generator_device = None
    
    def _get_generator(self, device, seed):
        """Get or create generator"""
        if self._generator is None or self._generator_device != device:
            self._generator = torch.Generator(device=device)
            self._generator_device = device
        self._generator.manual_seed(seed)
        return self._generator
    
    def forward(self, x):
        """
        Apply sliced univariate test
        
        Args:
            x: (*, N, D) samples
        
        Returns:
            aggregated statistic
        """
        with torch.no_grad():
            seed = self.global_step.item()
            device = x.device
            
            # Generate random projection directions
            g = self._get_generator(device, seed)
            proj_shape = (x.size(-1), self.num_slices)
            A = torch.randn(proj_shape, device=device, generator=g)
            A = A / A.norm(p=2, dim=0, keepdim=True)  # Normalize
            
            self.global_step.add_(1)
        
        # Project onto random directions
        x_projected = x @ A  # (*, N, num_slices)
        
        # Apply univariate test
        stats = self.univariate_test(x_projected)
        
        # Clip small values
        if self.clip_value is not None:
            stats = torch.where(stats < self.clip_value, torch.zeros_like(stats), stats)
        
        # Aggregate
        if self.reduction == 'mean':
            return stats.mean()
        elif self.reduction == 'sum':
            return stats.sum()
        else:
            return stats


class AttentionVisualizer(nn.Module):
    """
    Attention visualization module with statistical analysis
    """
    
    def __init__(self, config_path=None, config_dict=None):
        super().__init__()
        
        # Load configuration
        if config_dict is not None:
            config = config_dict
        elif config_path is not None:
            with open(config_path, 'r') as f:
                master_config = json.load(f)
            config = master_config['attention_viz']
        else:
            raise ValueError("Must provide either config_path or config_dict")
        
        self.num_slices = config['num_slices']
        self.reduction = config['reduction']
        self.sampler = config['sampler']
        self.clip_value = config['clip_value']
        self.t_max = config['t_max']
        self.n_points = config['n_points']
        self.integration = config['integration']
        
        # Create univariate test
        univariate_test = FastEppsPulley(
            t_max=self.t_max,
            n_points=self.n_points,
            integration=self.integration
        )
        
        # Create slicing test
        self.slicing_test = SlicingUnivariateTest(
            univariate_test=univariate_test,
            num_slices=self.num_slices,
            reduction=self.reduction,
            sampler=self.sampler,
            clip_value=self.clip_value
        )
    
    def extract_cross_attention(self, attention_weights, image_token_range, text_token_range):
        """
        Extract cross-attention between image and text tokens
        
        Args:
            attention_weights: (batch, num_heads, seq_len, seq_len)
            image_token_range: (start_idx, end_idx) for image tokens
            text_token_range: (start_idx, end_idx) for text tokens
        
        Returns:
            cross_attn: (batch, num_heads, num_image_tokens, num_text_tokens)
        """
        img_start, img_end = image_token_range
        text_start, text_end = text_token_range
        
        # Extract cross-attention submatrix
        cross_attn = attention_weights[:, :, img_start:img_end, text_start:text_end]
        
        return cross_attn
    
    def compute_attention_divergence(self, cross_attn):
        """
        Compute statistical divergence of attention patterns
        
        Args:
            cross_attn: (batch, num_heads, num_image_tokens, num_text_tokens)
        
        Returns:
            divergence: scalar divergence measure
        """
        batch, num_heads, num_img, num_text = cross_attn.shape
        
        # Flatten for analysis
        attn_flat = cross_attn.reshape(-1, num_img * num_text)  # (batch * num_heads, num_img * num_text)
        
        # Apply slicing test
        divergence = self.slicing_test(attn_flat.unsqueeze(1))  # Add sample dimension
        
        return divergence
    
    def generate_heatmap(self, cross_attn, save_path=None, title="Cross-Attention Heatmap"):
        """
        Generate attention heatmap visualization
        
        Args:
            cross_attn: (num_image_tokens, num_text_tokens) attention matrix
            save_path: optional path to save figure
            title: plot title
        
        Returns:
            fig: matplotlib figure
        """
        # Convert to numpy
        if isinstance(cross_attn, torch.Tensor):
            cross_attn = cross_attn.detach().cpu().numpy()
        
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Plot heatmap
        im = ax.imshow(cross_attn, cmap='viridis', aspect='auto', interpolation='nearest')
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Attention Weight', rotation=270, labelpad=20)
        
        # Labels
        ax.set_xlabel('Text Token Index')
        ax.set_ylabel('Image Patch Index')
        ax.set_title(title)
        
        # Grid
        ax.grid(False)
        
        # Save if requested
        if save_path is not None:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        return fig
    
    def generate_memory_heatmap(self, memory_addressing, save_path=None, title="Memory Addressing Heatmap"):
        """
        Generate memory addressing heatmap
        
        Args:
            memory_addressing: (batch, k_mem) or (k_mem,) addressing weights
            save_path: optional path to save figure
            title: plot title
        
        Returns:
            fig: matplotlib figure
        """
        # Convert to numpy
        if isinstance(memory_addressing, torch.Tensor):
            memory_addressing = memory_addressing.detach().cpu().numpy()
        
        # Ensure 2D
        if memory_addressing.ndim == 1:
            memory_addressing = memory_addressing.reshape(1, -1)
        
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 4))
        
        # Plot heatmap
        im = ax.imshow(memory_addressing, cmap='plasma', aspect='auto', interpolation='nearest')
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Addressing Weight', rotation=270, labelpad=20)
        
        # Labels
        ax.set_xlabel('Memory Slot Index')
        ax.set_ylabel('Batch Index')
        ax.set_title(title)
        
        # Grid
        ax.grid(False)
        
        # Save if requested
        if save_path is not None:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        return fig


if __name__ == "__main__":
    # Test attention visualizer
    config_path = os.path.join(os.path.dirname(__file__), "..", "configs", "model_config.json")
    
    visualizer = AttentionVisualizer(config_path=config_path)
    
    # Test cross-attention extraction
    batch_size = 2
    num_heads = 4
    seq_len = 100
    
    attention_weights = torch.rand(batch_size, num_heads, seq_len, seq_len)
    attention_weights = attention_weights / attention_weights.sum(dim=-1, keepdim=True)
    
    cross_attn = visualizer.extract_cross_attention(
        attention_weights,
        image_token_range=(0, 25),
        text_token_range=(25, 100)
    )
    
    print(f"Cross-attention shape: {cross_attn.shape}")
    
    # Test divergence computation
    divergence = visualizer.compute_attention_divergence(cross_attn)
    print(f"Attention divergence: {divergence.item():.4f}")
    
    # Test heatmap generation
    sample_attn = cross_attn[0, 0].detach()  # First batch, first head
    fig = visualizer.generate_heatmap(sample_attn, title="Test Cross-Attention")
    plt.close(fig)
    
    # Test memory heatmap
    memory_weights = torch.rand(2, 128)  # 2 batch, 128 memory slots
    memory_weights = memory_weights / memory_weights.sum(dim=-1, keepdim=True)
    fig = visualizer.generate_memory_heatmap(memory_weights, title="Test Memory Addressing")
    plt.close(fig)
    
    print("Attention visualizer test passed!")
