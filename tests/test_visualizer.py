import sys
import os
from pathlib import Path
import torch

sys.path.append(str(Path(__file__).parent.parent))

from src.attention_visualizer import AttentionVisualizer


def test_visualizer_initialization():
    """
    Test AttentionVisualizer initialization
    """
    print("Testing AttentionVisualizer initialization...")
    
    config = {
        'num_slices': 256,
        'clip_threshold': 0.01,
        'num_integration_points': 17,
        'use_gaussian_sampling': True
    }
    
    visualizer = AttentionVisualizer(config)
    
    assert visualizer.num_slices == 256
    assert visualizer.clip_threshold == 0.01
    
    print(f"  Num slices: {visualizer.num_slices}")
    print(f"  Clip threshold: {visualizer.clip_threshold}")
    print("  ✓ Initialization passed")
    
    return True


def test_slicing_univariate():
    """
    Test SlicingUnivariateTest
    """
    print("\nTesting SlicingUnivariateTest...")
    
    config = {
        'num_slices': 256,
        'clip_threshold': 0.01
    }
    
    visualizer = AttentionVisualizer(config)
    
    X = torch.randn(100, 10)
    Y = torch.randn(100, 10)
    
    statistic = visualizer.slicing_univariate_test(X, Y)
    
    assert isinstance(statistic, torch.Tensor)
    assert statistic.dim() == 0
    assert statistic.item() >= 0
    
    print(f"  Test statistic: {statistic.item():.6f}")
    print("  ✓ SlicingUnivariateTest passed")
    
    return True


def test_fast_epps_pulley():
    """
    Test FastEppsPulley
    """
    print("\nTesting FastEppsPulley...")
    
    config = {
        'num_slices': 256,
        'clip_threshold': 0.01,
        'num_integration_points': 17
    }
    
    visualizer = AttentionVisualizer(config)
    
    X = torch.randn(100)
    Y = torch.randn(100)
    
    statistic = visualizer.fast_epps_pulley(X, Y)
    
    assert isinstance(statistic, torch.Tensor)
    assert statistic.dim() == 0
    
    print(f"  Test statistic: {statistic.item():.6f}")
    print("  ✓ FastEppsPulley passed")
    
    return True


def test_extract_cross_attention():
    """
    Test cross-attention extraction
    """
    print("\nTesting cross-attention extraction...")
    
    config = {
        'num_slices': 256,
        'clip_threshold': 0.01
    }
    
    visualizer = AttentionVisualizer(config)
    
    batch_size = 2
    num_heads = 20
    seq_len = 50
    
    attn_weights = {
        'layer_0': torch.randn(batch_size, num_heads, seq_len, seq_len).softmax(dim=-1),
        'layer_1': torch.randn(batch_size, num_heads, seq_len, seq_len).softmax(dim=-1)
    }
    
    cross_attn = visualizer.extract_cross_attention(attn_weights, prefix_len=25)
    
    assert cross_attn.shape[0] == batch_size
    assert cross_attn.shape[1] == num_heads
    
    print(f"  Cross-attention shape: {cross_attn.shape}")
    print("  ✓ Cross-attention extraction passed")
    
    return True


def test_generate_heatmap():
    """
    Test heatmap generation
    """
    print("\nTesting heatmap generation...")
    
    config = {
        'num_slices': 256,
        'clip_threshold': 0.01
    }
    
    visualizer = AttentionVisualizer(config)
    
    attention_map = torch.randn(2, 20, 196, 30).softmax(dim=-1)
    
    import tempfile
    with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as f:
        temp_path = f.name
    
    try:
        visualizer.generate_heatmap(attention_map, temp_path)
        
        assert os.path.exists(temp_path), "Heatmap file not created"
        
        file_size = os.path.getsize(temp_path)
        assert file_size > 0, "Heatmap file is empty"
        
        print(f"  Heatmap saved to: {temp_path}")
        print(f"  File size: {file_size} bytes")
        print("  ✓ Heatmap generation passed")
        
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)
    
    return True


def run_all_tests():
    """
    Run all visualizer tests
    """
    print("=" * 60)
    print("ATTENTION VISUALIZER TESTS")
    print("=" * 60)
    
    tests = [
        test_visualizer_initialization,
        test_slicing_univariate,
        test_fast_epps_pulley,
        test_extract_cross_attention,
        test_generate_heatmap
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"  ✗ Test failed: {e}")
            import traceback
            traceback.print_exc()
            failed += 1
    
    print("\n" + "=" * 60)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 60)
    
    return failed == 0


if __name__ == '__main__':
    success = run_all_tests()
    sys.exit(0 if success else 1)
