import os
import sys
import argparse
from pathlib import Path
import torch

sys.path.append(str(Path(__file__).parent.parent))

from src.tiny_vlm import TinyVLM
from src.episodic_memory import EpisodicMemory


def export_memory_module(checkpoint_path, output_path, config_path):
    """
    Export episodic memory module as separate loadable component
    """
    print(f"Loading checkpoint from {checkpoint_path}")
    
    device = torch.device('cpu')
    model = TinyVLM(config_path=config_path, device=device)
    model.load_checkpoint(checkpoint_path)
    
    memory_state = {
        'memory_matrix': model.episodic_memory.memory.cpu(),
        'w_m': model.episodic_memory.W_M.state_dict(),
        'config': {
            'num_slots': model.episodic_memory.num_slots,
            'slot_dim': model.episodic_memory.slot_dim,
            'kv_dim': model.episodic_memory.kv_dim,
            'num_iterations': model.episodic_memory.num_iterations,
            'noise_std': model.episodic_memory.noise_std
        }
    }
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    torch.save(memory_state, output_path)
    
    memory_size = os.path.getsize(output_path) / (1024 ** 2)
    print(f"Memory module exported to: {output_path}")
    print(f"Memory module size: {memory_size:.2f} MB")
    
    return memory_state


def verify_memory_loading(memory_path, config_path):
    """
    Verify that exported memory can be loaded correctly
    """
    print(f"\nVerifying memory loading from {memory_path}")
    
    import json
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    memory_config = config['memory']
    memory = EpisodicMemory(
        num_slots=memory_config['num_slots'],
        slot_dim=memory_config['slot_dim'],
        kv_dim=memory_config['kv_dim'],
        num_iterations=memory_config['num_iterations'],
        noise_std=memory_config['noise_std']
    )
    
    memory.load_memory(memory_path)
    
    print("Memory loading successful!")
    print(f"Memory shape: {memory.memory.shape}")
    print(f"W_M parameters: {sum(p.numel() for p in memory.W_M.parameters()):,}")
    
    test_query = torch.randn(1, memory.slot_dim)
    Z_r, addressing_weights = memory.read(test_query)
    
    print(f"Test read - Z_r shape: {Z_r.shape}")
    print(f"Test read - Addressing weights shape: {addressing_weights.shape}")
    
    return True


def main():
    parser = argparse.ArgumentParser(description='Export episodic memory module')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to full model checkpoint')
    parser.add_argument('--output', type=str, default='checkpoints/memory.pt',
                        help='Output path for memory module')
    parser.add_argument('--config', type=str, default='configs/model_config.json',
                        help='Path to model config')
    parser.add_argument('--verify', action='store_true',
                        help='Verify memory loading after export')
    
    args = parser.parse_args()
    
    memory_state = export_memory_module(args.checkpoint, args.output, args.config)
    
    if args.verify:
        verify_memory_loading(args.output, args.config)
    
    print("\nMemory export complete!")
    print(f"Memory module saved to: {args.output}")
    print("\nUsage:")
    print("  from src.episodic_memory import EpisodicMemory")
    print(f"  memory = EpisodicMemory(...)")
    print(f"  memory.load_memory('{args.output}')")


if __name__ == '__main__':
    main()
