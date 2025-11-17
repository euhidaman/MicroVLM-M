"""
Model Configuration Extraction Script
Extracts BitNet and DeiT-Tiny configurations and saves to JSON
"""

import json
import os
import sys


def extract_bitnet_config():
    """Extract standard BitNet configuration parameters"""
    bitnet_config = {
        "model_name": "BitNet-3B-1.58bit",
        "hidden_size": 2560,
        "num_layers": 30,
        "num_heads": 20,
        "num_kv_heads": 5,
        "vocab_size": 128256,
        "ffn_dim": 6912,
        "norm_eps": 1e-5,
        "rope_theta": 500000.0,
        "max_seq_length": 2048,
        "head_dim": 128,  # hidden_size / num_heads
        "use_kernel": False  # Set to False for fp16 training
    }
    return bitnet_config


def extract_deit_tiny_config():
    """Extract DeiT-Tiny configuration parameters"""
    deit_config = {
        "model_name": "deit_tiny_patch16_224",
        "image_size": 224,
        "patch_size": 16,
        "num_patches": 196,  # (224/16)^2
        "embed_dim": 192,
        "num_layers": 12,
        "num_heads": 3,
        "mlp_ratio": 4.0,
        "num_classes": 1000,
        "drop_rate": 0.0,
        "attn_drop_rate": 0.0,
        "drop_path_rate": 0.0,
        "pretrained": True
    }
    return deit_config


def compute_adapter_config(bitnet_config, deit_config):
    """Compute multimodal adapter configuration"""
    num_patches = deit_config["num_patches"]

    # Dynamic K_prefix calculation: clamp(ceil(num_patches / 8), 8, 64)
    import math
    k_prefix = max(8, min(64, math.ceil(num_patches / 8)))

    adapter_config = {
        "input_dim": deit_config["embed_dim"],
        "output_dim": bitnet_config["hidden_size"],
        "k_prefix": k_prefix,
        "num_patches": num_patches,
        "hidden_dim": 512,  # Intermediate MLP dimension
        "dropout": 0.1,
        "use_mlp": True,
        "use_layer_norm": True
    }
    return adapter_config


def compute_memory_config(bitnet_config):
    """Compute episodic memory configuration"""
    memory_config = {
        "k_mem": 128,  # Number of memory slots
        "c_mem": bitnet_config["hidden_size"],  # Memory dimension
        "num_layers": bitnet_config["num_layers"],
        "num_heads": bitnet_config["num_heads"],
        "head_dim": bitnet_config["head_dim"],
        "observation_noise_std": 0.01,
        "direct_writing": True,
        "ordering": False,
        "pseudoinverse_approx_step": 3,
        "w_logvar_setting": 0,
        "deterministic": False
    }
    return memory_config


def compute_scope_config(bitnet_config):
    """Compute ScopeNet configuration"""
    scope_config = {
        "input_dim": bitnet_config["hidden_size"],
        "hidden_dim": 256,
        "num_layers": 2,
        "dropout": 0.1,
        "activation": "relu"
    }
    return scope_config


def compute_attention_viz_config():
    """Compute attention visualization configuration"""
    viz_config = {
        "num_slices": 256,
        "reduction": "mean",
        "sampler": "gaussian",
        "clip_value": 0.01,
        "t_max": 3.0,
        "n_points": 17,
        "integration": "trapezoid"
    }
    return viz_config


def main():
    """Main extraction function"""
    print("Extracting model configurations...")

    # Extract individual configs
    bitnet_config = extract_bitnet_config()
    deit_config = extract_deit_tiny_config()
    adapter_config = compute_adapter_config(bitnet_config, deit_config)
    memory_config = compute_memory_config(bitnet_config)
    scope_config = compute_scope_config(bitnet_config)
    viz_config = compute_attention_viz_config()

    # Combine into master config
    master_config = {
        "bitnet": bitnet_config,
        "deit_tiny": deit_config,
        "adapter": adapter_config,
        "memory": memory_config,
        "scope": scope_config,
        "attention_viz": viz_config,
        "model_size_target_mb": 500
    }

    # Save to configs directory
    config_dir = os.path.join(os.path.dirname(__file__), "..", "configs")
    os.makedirs(config_dir, exist_ok=True)

    config_path = os.path.join(config_dir, "model_config.json")
    with open(config_path, "w") as f:
        json.dump(master_config, f, indent=2)

    print(f"Configuration saved to: {config_path}")
    print("\nConfiguration Summary:")
    print(f"  BitNet hidden size: {bitnet_config['hidden_size']}")
    print(f"  BitNet layers: {bitnet_config['num_layers']}")
    print(f"  DeiT-Tiny embed dim: {deit_config['embed_dim']}")
    print(f"  DeiT-Tiny num patches: {deit_config['num_patches']}")
    print(f"  Adapter K_prefix: {adapter_config['k_prefix']}")
    print(f"  Memory slots: {memory_config['k_mem']}")
    print(f"  Attention viz slices: {viz_config['num_slices']}")

    return master_config


if __name__ == "__main__":
    config = main()
