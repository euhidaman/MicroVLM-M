"""
Multimodal Adapter Module
Projects DeiT-Tiny visual embeddings to BitNet hidden dimension
and performs prefix-token pooling
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import json
import os

class MultimodalAdapter(nn.Module):
    """
    Adapter that converts DeiT-Tiny patch embeddings to BitNet-compatible prefix tokens
    
    Architecture:
        1. Linear projection: deit_embed_dim -> bitnet_hidden_dim
        2. Optional MLP refinement
        3. Group-pooling to reduce num_patches -> K_prefix tokens
        4. Learned positional embeddings for prefix tokens
        5. Layer normalization
    
    Shape Flow:
        Input:  (batch, num_patches, deit_embed_dim)
        Output: (batch, K_prefix, bitnet_hidden_dim)
    """
    
    def __init__(self, config_path=None, config_dict=None):
        super().__init__()
        
        # Load configuration
        if config_dict is not None:
            config = config_dict
        elif config_path is not None:
            with open(config_path, 'r') as f:
                master_config = json.load(f)
            config = master_config['adapter']
            self.deit_config = master_config['deit_tiny']
            self.bitnet_config = master_config['bitnet']
        else:
            raise ValueError("Must provide either config_path or config_dict")
        
        self.input_dim = config['input_dim']
        self.output_dim = config['output_dim']
        self.k_prefix = config['k_prefix']
        self.num_patches = config['num_patches']
        self.hidden_dim = config.get('hidden_dim', 512)
        self.use_mlp = config.get('use_mlp', True)
        self.use_layer_norm = config.get('use_layer_norm', True)
        self.dropout = config.get('dropout', 0.1)
        
        # Main projection layer
        self.projection = nn.Linear(self.input_dim, self.output_dim)
        
        # Optional MLP refinement
        if self.use_mlp:
            self.mlp = nn.Sequential(
                nn.Linear(self.output_dim, self.hidden_dim),
                nn.GELU(),
                nn.Dropout(self.dropout),
                nn.Linear(self.hidden_dim, self.output_dim)
            )
        
        # Group pooling parameters
        self.pool_size = self.num_patches // self.k_prefix
        if self.num_patches % self.k_prefix != 0:
            # Use learned attention pooling if not evenly divisible
            self.use_attention_pool = True
            self.pool_query = nn.Parameter(torch.randn(1, self.k_prefix, self.output_dim))
            self.pool_attn = nn.MultiheadAttention(
                embed_dim=self.output_dim,
                num_heads=4,
                dropout=self.dropout,
                batch_first=True
            )
        else:
            self.use_attention_pool = False
        
        # Learned positional embeddings for prefix tokens
        self.prefix_pos_embed = nn.Parameter(torch.randn(1, self.k_prefix, self.output_dim))
        
        # Layer normalization
        if self.use_layer_norm:
            self.layer_norm = nn.LayerNorm(self.output_dim)
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights"""
        nn.init.xavier_uniform_(self.projection.weight)
        nn.init.zeros_(self.projection.bias)
        
        if self.use_mlp:
            for layer in self.mlp:
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_uniform_(layer.weight)
                    nn.init.zeros_(layer.bias)
        
        nn.init.normal_(self.prefix_pos_embed, std=0.02)
        
        if self.use_attention_pool:
            nn.init.normal_(self.pool_query, std=0.02)
    
    def forward(self, patch_embeddings):
        """
        Forward pass
        
        Args:
            patch_embeddings: (batch, num_patches, deit_embed_dim)
        
        Returns:
            prefix_tokens: (batch, K_prefix, bitnet_hidden_dim)
        """
        batch_size = patch_embeddings.size(0)
        
        # Project to BitNet hidden dimension
        x = self.projection(patch_embeddings)  # (batch, num_patches, output_dim)
        
        # Optional MLP refinement
        if self.use_mlp:
            x = x + self.mlp(x)  # Residual connection
        
        # Group pooling to reduce tokens
        if self.use_attention_pool:
            # Attention-based pooling
            query = self.pool_query.expand(batch_size, -1, -1)
            pooled, _ = self.pool_attn(query, x, x)  # (batch, k_prefix, output_dim)
        else:
            # Simple average pooling over groups
            x_reshaped = x.view(batch_size, self.k_prefix, self.pool_size, self.output_dim)
            pooled = x_reshaped.mean(dim=2)  # (batch, k_prefix, output_dim)
        
        # Add positional embeddings
        prefix_tokens = pooled + self.prefix_pos_embed
        
        # Layer normalization
        if self.use_layer_norm:
            prefix_tokens = self.layer_norm(prefix_tokens)
        
        return prefix_tokens
    
    def get_output_shape(self, batch_size):
        """Return expected output shape"""
        return (batch_size, self.k_prefix, self.output_dim)


if __name__ == "__main__":
    # Test the adapter
    import sys
    sys.path.append(os.path.dirname(os.path.dirname(__file__)))
    
    config_path = os.path.join(os.path.dirname(__file__), "..", "configs", "model_config.json")
    
    adapter = MultimodalAdapter(config_path=config_path)
    
    # Test with dummy input
    batch_size = 4
    num_patches = 196
    deit_dim = 192
    
    dummy_patches = torch.randn(batch_size, num_patches, deit_dim)
    output = adapter(dummy_patches)
    
    print(f"Input shape: {dummy_patches.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Expected shape: {adapter.get_output_shape(batch_size)}")
    
    assert output.shape == adapter.get_output_shape(batch_size), "Shape mismatch!"
    print("Adapter test passed!")
