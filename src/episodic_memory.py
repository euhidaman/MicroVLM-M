"""
Episodic Memory Module (Larimar-style)
Implements memory matrix with read/write operations and KV injection
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import json
import os
import numpy as np

EPSILON = 1e-6


class EpisodicMemory(nn.Module):
    """
    Episodic Memory Module for injecting memory-derived representations
    into BitNet decoder KV caches

    Architecture:
        - Memory matrix M: (K_mem, C_mem) where C_mem = bitnet_hidden_dim
        - Write mechanism: incremental updates with Gaussian addressing
        - Read mechanism: compute addressing weights and retrieve memory
        - Projection W_M: maps memory readout to KV space for all layers

    Memory is designed as a separate loadable component for fast deployment
    """

    def __init__(self, config_path=None, config_dict=None, device='cuda'):
        super().__init__()

        # Load configuration
        if config_dict is not None:
            # Check if config_dict has nested structure
            if 'memory' in config_dict:
                config = config_dict['memory']
                bitnet_config = config_dict.get('bitnet', {})
            else:
                config = config_dict
                bitnet_config = {}
        elif config_path is not None:
            with open(config_path, 'r') as f:
                master_config = json.load(f)
            config = master_config['memory']
            bitnet_config = master_config['bitnet']
        else:
            raise ValueError("Must provide either config_path or config_dict")

        self.device = device
        self.k_mem = config['k_mem']
        self.c_mem = config['c_mem']
        self.num_layers = config['num_layers']
        self.num_heads = config['num_heads']
        self.head_dim = config['head_dim']
        self.observation_noise_std = config.get('observation_noise_std', 0.01)
        self.direct_writing = config.get('direct_writing', True)
        self.ordering = config.get('ordering', False)
        self.pseudoinverse_approx_step = config.get(
            'pseudoinverse_approx_step', 3)
        self.w_logvar_setting = config.get('w_logvar_setting', 0)
        self.deterministic = config.get('deterministic', False)

        # Memory matrix initialization (K_mem, C_mem)
        self.memory_mean = nn.Parameter(
            torch.randn(self.k_mem, self.c_mem) * 0.02,
            requires_grad=True
        )

        # Memory covariance (diagonal, stored as logvar for numerical stability)
        self.memory_logvar = nn.Parameter(
            torch.zeros(self.k_mem),
            requires_grad=False  # Fixed variance
        )

        # Addressing weight variance
        if self.w_logvar_setting == 0:
            self.w_logvar = nn.Parameter(torch.zeros(1), requires_grad=True)
        elif self.w_logvar_setting == 1:
            self.w_logvar = nn.Parameter(
                torch.zeros(self.k_mem), requires_grad=True)

        # Projection to KV space: C_mem -> (num_layers * num_heads * head_dim * 2)
        # Factor of 2 for both K and V
        self.kv_dim = self.num_layers * self.num_heads * self.head_dim * 2
        self.W_M = nn.Linear(self.c_mem, self.kv_dim, bias=False)

        # Ben-Cohen initialization for pseudoinverse approximation
        self.register_buffer('ben_cohen_init', torch.tensor([-5.0]))

        self._init_weights()

    def _init_weights(self):
        """Initialize projection weights"""
        nn.init.xavier_uniform_(self.W_M.weight)

    def _get_prior_params(self):
        """Get prior memory distribution parameters"""
        prior_var = torch.exp(self.memory_logvar) + EPSILON
        prior_cov = torch.diag(prior_var)
        return self.memory_mean, prior_cov

    def _approx_pseudo_inverse(self, A, iterative_step=3):
        """
        Approximate pseudoinverse using Ben-Cohen iterative method

        Args:
            A: (batch, M, N) matrix
            iterative_step: number of iterations

        Returns:
            A_pinv: (batch, N, M) pseudoinverse
        """
        alpha = torch.exp(self.ben_cohen_init).clamp(max=5e-4)
        A_pinv = alpha * A.transpose(-2, -1)

        for _ in range(iterative_step):
            A_pinv = 2 * A_pinv - torch.bmm(torch.bmm(A_pinv, A), A_pinv)

        return A_pinv

    def write(self, z_sequence, batch_size=1):
        """
        Write encoded inputs to memory

        Args:
            z_sequence: (sequence_len, batch, C_mem) encoded representations
            batch_size: batch size

        Returns:
            posterior_memory: updated (memory_mean, memory_cov)
            dkl_M: KL divergence for memory regularization
        """
        # Get prior
        prior_mean, prior_cov = self._get_prior_params()
        batch_prior_mean = prior_mean.unsqueeze(0).expand(batch_size, -1, -1)
        batch_prior_cov = prior_cov.unsqueeze(0).expand(batch_size, -1, -1)

        # Add noise for robustness
        noise = torch.randn_like(z_sequence) * self.observation_noise_std
        z_noise = z_sequence + noise

        if self.direct_writing:
            # Direct write using pseudoinverse
            # z_noise: (seq_len, batch, C_mem)
            # Solve for addressing weights w: (seq_len, batch, K_mem)
            z_t = z_noise.transpose(0, 1)  # (batch, seq_len, C_mem)
            M_pinv = self._approx_pseudo_inverse(
                batch_prior_mean,
                iterative_step=self.pseudoinverse_approx_step
            )  # (batch, C_mem, K_mem)

            w = torch.bmm(z_t, M_pinv)  # (batch, seq_len, K_mem)
            w_pinv = self._approx_pseudo_inverse(
                w,
                iterative_step=self.pseudoinverse_approx_step
            )  # (batch, K_mem, seq_len)

            # Update memory mean
            new_memory_mean = torch.bmm(w_pinv, z_t)  # (batch, K_mem, C_mem)
            posterior_memory = (new_memory_mean, batch_prior_cov)
        else:
            # Incremental Kalman-style update (not implemented for efficiency)
            posterior_memory = (batch_prior_mean, batch_prior_cov)

        # Compute KL divergence for regularization
        dkl_M = self._compute_kl_divergence(
            (batch_prior_mean, batch_prior_cov),
            posterior_memory
        )

        return posterior_memory, dkl_M

    def read(self, z_query, memory_state, deterministic=None):
        """
        Read from memory given query

        Args:
            z_query: (batch, C_mem) or (sequence_len, batch, C_mem) query
            memory_state: (memory_mean, memory_cov)
            deterministic: whether to use deterministic addressing

        Returns:
            z_retrieved: (batch, C_mem) retrieved representation
            Z_r_kv: (batch, kv_dim) memory-derived KV entries
            dkl_w: KL divergence for addressing weights
        """
        memory_mean, memory_cov = memory_state
        batch_size = memory_mean.size(0)

        if z_query.dim() == 3:
            z_query = z_query[-1]  # Take last timestep

        # Compute addressing weights using pseudoinverse
        M_pinv = self._approx_pseudo_inverse(
            memory_mean,
            iterative_step=self.pseudoinverse_approx_step
        )  # (batch, C_mem, K_mem)

        z_query_exp = z_query.unsqueeze(1)  # (batch, 1, C_mem)
        w_mean = torch.bmm(z_query_exp, M_pinv).squeeze(1)  # (batch, K_mem)

        # Add noise if not deterministic
        use_deterministic = deterministic if deterministic is not None else self.deterministic
        if not use_deterministic:
            w_std = torch.exp(0.5 * self.w_logvar)
            if self.w_logvar_setting == 0:
                w_std = w_std.expand_as(w_mean)
            w = w_mean + w_std * torch.randn_like(w_mean)
        else:
            w = w_mean

        # Retrieve from memory
        w_exp = w.unsqueeze(1)  # (batch, 1, K_mem)
        z_retrieved = torch.bmm(
            w_exp, memory_mean).squeeze(1)  # (batch, C_mem)

        # Project to KV space
        Z_r_kv = self.W_M(z_retrieved)  # (batch, kv_dim)

        # Compute KL divergence for addressing
        dkl_w = self._compute_w_kl(w_mean, self.w_logvar)

        return z_retrieved, Z_r_kv, dkl_w

    def inject_to_kv_cache(self, Z_r_kv, layer_idx):
        """
        Extract K, V for specific layer from memory projection

        Args:
            Z_r_kv: (batch, kv_dim) memory-derived KV
            layer_idx: which layer to extract for

        Returns:
            mem_k: (batch, num_heads, 1, head_dim)
            mem_v: (batch, num_heads, 1, head_dim)
        """
        batch_size = Z_r_kv.size(0)

        # Reshape: (batch, num_layers, 2, num_heads, head_dim)
        kv_reshaped = Z_r_kv.view(
            batch_size,
            self.num_layers,
            2,  # K and V
            self.num_heads,
            self.head_dim
        )

        # Extract for specific layer
        mem_k = kv_reshaped[:, layer_idx, 0, :, :].unsqueeze(
            2)  # (batch, num_heads, 1, head_dim)
        mem_v = kv_reshaped[:, layer_idx, 1, :, :].unsqueeze(
            2)  # (batch, num_heads, 1, head_dim)

        return mem_k, mem_v

    def _compute_kl_divergence(self, prior_memory, posterior_memory):
        """Compute KL divergence between memory distributions"""
        prior_mean, prior_cov = prior_memory
        post_mean, post_cov = posterior_memory

        # Diagonal covariances
        prior_var = torch.diagonal(prior_cov, dim1=-2, dim2=-1)
        post_var = torch.diagonal(post_cov, dim1=-2, dim2=-1)

        # KL divergence for multivariate Gaussians
        t1 = self.c_mem * torch.sum(post_var / prior_var, dim=-1)
        t2 = torch.sum((post_mean - prior_mean) ** 2 /
                       prior_var.unsqueeze(-1), dim=[-2, -1])
        t3 = -self.c_mem * self.k_mem
        t4 = self.c_mem * \
            torch.sum(torch.log(prior_var) - torch.log(post_var), dim=-1)

        dkl = t1 + t2 + t3 + t4
        return dkl.mean()

    def _compute_w_kl(self, w_mean, w_logvar):
        """Compute KL divergence for addressing weights"""
        w_logvar_exp = w_logvar
        if self.w_logvar_setting == 0:
            w_logvar_exp = w_logvar.expand_as(w_mean)

        dkl = 0.5 * (torch.exp(w_logvar_exp) + w_mean ** 2 - 1 - w_logvar_exp)
        return dkl.sum()

    def save_memory(self, path):
        """Save memory state for deployment"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            'memory_mean': self.memory_mean,
            'memory_logvar': self.memory_logvar,
            'W_M': self.W_M.state_dict(),
            'config': {
                'k_mem': self.k_mem,
                'c_mem': self.c_mem,
                'num_layers': self.num_layers,
                'num_heads': self.num_heads,
                'head_dim': self.head_dim
            }
        }, path)
        print(f"Memory saved to {path}")

    def load_memory(self, path):
        """Load memory state from file"""
        checkpoint = torch.load(path, map_location=self.device)
        self.memory_mean = nn.Parameter(checkpoint['memory_mean'])
        self.memory_logvar = nn.Parameter(checkpoint['memory_logvar'])
        self.W_M.load_state_dict(checkpoint['W_M'])
        print(f"Memory loaded from {path}")



