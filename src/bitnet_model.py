"""
BitNet Integration Module
Direct implementation of BitNet architecture with weight loading
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import json
import os
from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass
class BitNetConfig:
    """BitNet model configuration"""
    hidden_size: int = 2560
    num_layers: int = 30
    num_heads: int = 20
    num_kv_heads: int = 5
    vocab_size: int = 128256
    ffn_dim: int = 6912
    norm_eps: float = 1e-5
    rope_theta: float = 500000.0
    max_seq_length: int = 2048
    use_kernel: bool = False  # False for fp16 training


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization"""

    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        norm_x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return self.weight * norm_x


class BitLinear(nn.Linear):
    """
    BitNet linear layer with quantized forward pass
    For training: uses fp16
    For inference: can be quantized to 1.58-bit
    """

    def __init__(self, in_features, out_features, bias=False):
        super().__init__(in_features, out_features, bias=bias)

    def quant_input(self, x):
        """Quantize input to 8-bit during forward (simulation)"""
        s = 127 / x.abs().max(dim=-1, keepdim=True).values.clamp_(min=1e-5)
        return (x * s).round().clamp(-128, 127) / s

    def forward(self, x):
        """Forward pass with input quantization"""
        x_quant = self.quant_input(x)
        return F.linear(x_quant, self.weight, self.bias)


class RotaryEmbedding(nn.Module):
    """Rotary Position Embedding (RoPE)"""

    def __init__(self, dim, max_seq_len=2048, theta=500000.0):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.theta = theta

        # Precompute frequency bands
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)

        # Cache for efficiency
        self._cos_cached = None
        self._sin_cached = None
        self._seq_len_cached = 0

    def forward(self, x, seq_len):
        """Apply rotary embeddings"""
        if seq_len > self._seq_len_cached:
            self._seq_len_cached = seq_len
            t = torch.arange(seq_len, device=x.device).type_as(self.inv_freq)
            freqs = torch.einsum('i,j->ij', t, self.inv_freq)
            emb = torch.cat((freqs, freqs), dim=-1)
            self._cos_cached = emb.cos()[None, None, :, :]
            self._sin_cached = emb.sin()[None, None, :, :]

        return self._cos_cached[:, :, :seq_len, :], self._sin_cached[:, :, :seq_len, :]


def apply_rotary_pos_emb(q, k, cos, sin):
    """Apply rotary position embeddings to queries and keys"""
    def rotate_half(x):
        x1, x2 = x[..., :x.shape[-1]//2], x[..., x.shape[-1]//2:]
        return torch.cat((-x2, x1), dim=-1)

    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class BitNetAttention(nn.Module):
    """BitNet attention mechanism"""

    def __init__(self, config: BitNetConfig):
        super().__init__()

        self.num_heads = config.num_heads
        self.num_kv_heads = config.num_kv_heads
        self.head_dim = config.hidden_size // config.num_heads
        self.hidden_size = config.hidden_size

        # QKV projection
        self.wqkv = BitLinear(
            config.hidden_size,
            (self.num_heads + 2 * self.num_kv_heads) * self.head_dim,
            bias=False
        )

        # Output projection
        self.wo = BitLinear(
            self.num_heads * self.head_dim,
            config.hidden_size,
            bias=False
        )

        # RoPE
        self.rope = RotaryEmbedding(
            self.head_dim,
            max_seq_len=config.max_seq_length,
            theta=config.rope_theta
        )

        # Sub-normalization
        self.attn_sub_norm = RMSNorm(config.hidden_size, config.norm_eps)

    def forward(self, x, attention_mask=None, kv_cache=None, memory_kv=None):
        """
        Forward pass with optional memory KV injection

        Args:
            x: (batch, seq_len, hidden_size)
            attention_mask: optional attention mask
            kv_cache: optional (k_cache, v_cache) for autoregressive generation
            memory_kv: optional (mem_k, mem_v) from episodic memory

        Returns:
            output: (batch, seq_len, hidden_size)
            new_kv_cache: updated (k_cache, v_cache)
        """
        batch_size, seq_len, _ = x.shape

        # QKV projection
        qkv = self.wqkv(x)
        q = qkv[:, :, :self.num_heads * self.head_dim]
        kv = qkv[:, :, self.num_heads * self.head_dim:]
        k, v = kv.chunk(2, dim=-1)

        # Reshape for multi-head attention
        q = q.view(batch_size, seq_len, self.num_heads,
                   self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_kv_heads,
                   self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_kv_heads,
                   self.head_dim).transpose(1, 2)

        # Apply RoPE
        cos, sin = self.rope(x, seq_len)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        # Inject memory KV if provided
        if memory_kv is not None:
            mem_k, mem_v = memory_kv
            k = torch.cat([mem_k, k], dim=2)  # Concat along seq dimension
            v = torch.cat([mem_v, v], dim=2)

        # Update KV cache
        if kv_cache is not None:
            k_cache, v_cache = kv_cache
            k = torch.cat([k_cache, k], dim=2)
            v = torch.cat([v_cache, v], dim=2)
        new_kv_cache = (k, v)

        # Repeat KV heads if needed (GQA)
        if self.num_kv_heads != self.num_heads:
            k = k.repeat_interleave(self.num_heads // self.num_kv_heads, dim=1)
            v = v.repeat_interleave(self.num_heads // self.num_kv_heads, dim=1)

        # Scaled dot-product attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)

        if attention_mask is not None:
            scores = scores + attention_mask

        attn_weights = F.softmax(scores, dim=-1)
        output = torch.matmul(attn_weights, v)

        # Reshape and project
        output = output.transpose(1, 2).contiguous().view(
            batch_size, seq_len, -1)
        output = self.attn_sub_norm(output)
        output = self.wo(output)

        return output, new_kv_cache, attn_weights


class BitNetFeedForward(nn.Module):
    """BitNet feed-forward network with squared ReLU"""

    def __init__(self, config: BitNetConfig):
        super().__init__()

        self.w13 = BitLinear(config.hidden_size, 2 *
                             config.ffn_dim, bias=False)
        self.w2 = BitLinear(config.ffn_dim, config.hidden_size, bias=False)
        self.ffn_sub_norm = RMSNorm(config.ffn_dim, config.norm_eps)

    def forward(self, x):
        """Forward pass with squared ReLU activation"""
        x13 = self.w13(x)
        x1, x3 = x13.chunk(2, dim=-1)
        inner = self.ffn_sub_norm(F.relu(x1) ** 2 * x3)
        return self.w2(inner)


class BitNetBlock(nn.Module):
    """Single BitNet transformer block"""

    def __init__(self, config: BitNetConfig):
        super().__init__()

        self.attention = BitNetAttention(config)
        self.feed_forward = BitNetFeedForward(config)
        self.attention_norm = RMSNorm(config.hidden_size, config.norm_eps)
        self.ffn_norm = RMSNorm(config.hidden_size, config.norm_eps)

    def forward(self, x, attention_mask=None, kv_cache=None, memory_kv=None):
        """Forward pass with residual connections"""
        # Attention with residual
        attn_out, new_kv_cache, attn_weights = self.attention(
            self.attention_norm(x),
            attention_mask=attention_mask,
            kv_cache=kv_cache,
            memory_kv=memory_kv
        )
        x = x + attn_out

        # FFN with residual
        x = x + self.feed_forward(self.ffn_norm(x))

        return x, new_kv_cache, attn_weights


class BitNetModel(nn.Module):
    """Complete BitNet language model"""

    def __init__(self, config: BitNetConfig):
        super().__init__()

        self.config = config

        # Token embeddings
        self.tok_embeddings = nn.Embedding(
            config.vocab_size, config.hidden_size)

        # Transformer blocks
        self.layers = nn.ModuleList([
            BitNetBlock(config) for _ in range(config.num_layers)
        ])

        # Final norm
        self.norm = RMSNorm(config.hidden_size, config.norm_eps)

        # Output projection
        self.output = nn.Linear(
            config.hidden_size, config.vocab_size, bias=False)

        # Tie weights
        self.output.weight = self.tok_embeddings.weight

    def forward(self, input_ids=None, inputs_embeds=None, attention_mask=None,
                kv_caches=None, memory_kvs=None, return_attention=False):
        """
        Forward pass

        Args:
            input_ids: (batch, seq_len) token IDs
            inputs_embeds: (batch, seq_len, hidden_size) or None
            attention_mask: attention mask
            kv_caches: list of (k_cache, v_cache) per layer
            memory_kvs: list of (mem_k, mem_v) per layer for memory injection
            return_attention: whether to return attention weights

        Returns:
            logits: (batch, seq_len, vocab_size)
            new_kv_caches: updated KV caches
            attention_weights: optional list of attention weights
        """
        # Get embeddings
        if inputs_embeds is None:
            x = self.tok_embeddings(input_ids)
        else:
            x = inputs_embeds

        # Initialize caches
        if kv_caches is None:
            kv_caches = [None] * len(self.layers)

        if memory_kvs is None:
            memory_kvs = [None] * len(self.layers)

        new_kv_caches = []
        all_attention_weights = [] if return_attention else None

        # Apply transformer blocks
        for i, (layer, kv_cache, memory_kv) in enumerate(zip(self.layers, kv_caches, memory_kvs)):
            x, new_kv_cache, attn_weights = layer(
                x,
                attention_mask=attention_mask,
                kv_cache=kv_cache,
                memory_kv=memory_kv
            )
            new_kv_caches.append(new_kv_cache)

            if return_attention:
                all_attention_weights.append(attn_weights)

        # Final norm and projection
        x = self.norm(x)
        logits = self.output(x)

        return logits, new_kv_caches, all_attention_weights

    def load_pretrained_weights(self, checkpoint_path):
        """Load pretrained BitNet weights from checkpoint"""
        print(f"Loading BitNet weights from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location='cpu')

        # Handle different checkpoint formats
        if 'model' in checkpoint:
            state_dict = checkpoint['model']
        elif 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint

        # Load state dict
        incompatible = self.load_state_dict(state_dict, strict=False)

        if incompatible.missing_keys:
            print(f"Missing keys: {incompatible.missing_keys}")
        if incompatible.unexpected_keys:
            print(f"Unexpected keys: {incompatible.unexpected_keys}")

        print("BitNet weights loaded successfully")


if __name__ == "__main__":
    # Test BitNet model
    config = BitNetConfig(
        hidden_size=2560,
        num_layers=30,
        num_heads=20,
        num_kv_heads=5,
        vocab_size=128256,
        ffn_dim=6912
    )

    model = BitNetModel(config)

    # Test forward pass
    batch_size = 2
    seq_len = 10
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))

    logits, kv_caches, attn_weights = model(input_ids, return_attention=True)

    print(f"Input shape: {input_ids.shape}")
    print(f"Logits shape: {logits.shape}")
    print(f"Number of KV caches: {len(kv_caches)}")
    print(f"Number of attention weight tensors: {len(attn_weights)}")
    print(f"Attention weight shape: {attn_weights[0].shape}")

    print("BitNet model test passed!")
