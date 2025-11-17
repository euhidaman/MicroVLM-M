import sys
import os
from pathlib import Path
import torch
import json

sys.path.append(str(Path(__file__).parent.parent))

from src.multimodal_adapter import MultimodalAdapter


def test_adapter_shape_validation():
    """
    Test multimodal adapter shape transformations
    """
    print("Testing MultimodalAdapter shape validation...")
    
    config = {
        'input_dim': 192,
        'output_dim': 2560,
        'num_patches': 196,
        'k_prefix': 25,
        'use_mlp': True,
        'mlp_hidden_dim': 512,
        'dropout': 0.1
    }
    
    adapter = MultimodalAdapter(config)
    
    batch_size = 4
    num_patches = 196
    embed_dim = 192
    
    x = torch.randn(batch_size, num_patches, embed_dim)
    
    output = adapter(x)
    
    expected_shape = (batch_size, config['k_prefix'], config['output_dim'])
    assert output.shape == expected_shape, f"Expected {expected_shape}, got {output.shape}"
    
    print(f"  Input shape: {x.shape}")
    print(f"  Output shape: {output.shape}")
    print(f"  Expected prefix tokens: {config['k_prefix']}")
    print(f"  Expected output dim: {config['output_dim']}")
    print("  ✓ Shape validation passed")
    
    return True


def test_adapter_gradient_flow():
    """
    Test that gradients flow through adapter
    """
    print("\nTesting MultimodalAdapter gradient flow...")
    
    config = {
        'input_dim': 192,
        'output_dim': 2560,
        'num_patches': 196,
        'k_prefix': 25,
        'use_mlp': True,
        'mlp_hidden_dim': 512,
        'dropout': 0.0
    }
    
    adapter = MultimodalAdapter(config)
    
    x = torch.randn(2, 196, 192, requires_grad=True)
    output = adapter(x)
    loss = output.sum()
    loss.backward()
    
    assert x.grad is not None, "No gradient for input"
    
    for name, param in adapter.named_parameters():
        assert param.grad is not None, f"No gradient for {name}"
    
    print("  ✓ Gradients flow correctly")
    
    return True


def test_adapter_pooling():
    """
    Test that pooling reduces patch count correctly
    """
    print("\nTesting MultimodalAdapter pooling mechanism...")
    
    config = {
        'input_dim': 192,
        'output_dim': 2560,
        'num_patches': 196,
        'k_prefix': 25,
        'use_mlp': False,
        'dropout': 0.0
    }
    
    adapter = MultimodalAdapter(config)
    
    x = torch.randn(1, 196, 192)
    output = adapter(x)
    
    assert output.shape[1] == config['k_prefix'], f"Expected {config['k_prefix']} tokens, got {output.shape[1]}"
    
    print(f"  Patches: 196 → {config['k_prefix']} prefix tokens")
    print("  ✓ Pooling works correctly")
    
    return True


def run_all_tests():
    """
    Run all adapter tests
    """
    print("=" * 60)
    print("MULTIMODAL ADAPTER TESTS")
    print("=" * 60)
    
    tests = [
        test_adapter_shape_validation,
        test_adapter_gradient_flow,
        test_adapter_pooling
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"  ✗ Test failed: {e}")
            failed += 1
    
    print("\n" + "=" * 60)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 60)
    
    return failed == 0


if __name__ == '__main__':
    success = run_all_tests()
    sys.exit(0 if success else 1)
