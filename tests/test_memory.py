from src.episodic_memory import EpisodicMemory
import sys
import os
from pathlib import Path
import torch

sys.path.append(str(Path(__file__).parent.parent))


def test_memory_initialization():
    """
    Test episodic memory initialization
    """
    print("Testing EpisodicMemory initialization...")

    num_slots = 128
    slot_dim = 2560
    kv_dim = 153600

    memory = EpisodicMemory(
        num_slots=num_slots,
        slot_dim=slot_dim,
        kv_dim=kv_dim
    )

    assert memory.memory.shape == (
        num_slots, slot_dim), f"Wrong memory shape: {memory.memory.shape}"
    assert memory.W_M is not None, "W_M not initialized"

    print(f"  Memory shape: {memory.memory.shape}")
    print(f"  W_M output dim: {kv_dim}")
    print("  ✓ Initialization passed")

    return True


def test_memory_write():
    """
    Test memory write operation
    """
    print("\nTesting EpisodicMemory write...")

    memory = EpisodicMemory(num_slots=128, slot_dim=2560, kv_dim=153600)

    batch_size = 4
    z_t = torch.randn(batch_size, 2560)

    memory_before = memory.memory.clone()
    memory.write(z_t)
    memory_after = memory.memory

    assert not torch.allclose(
        memory_before, memory_after), "Memory not updated after write"

    print(f"  Input shape: {z_t.shape}")
    print(
        f"  Memory updated: {not torch.allclose(memory_before, memory_after)}")
    print("  ✓ Write operation passed")

    return True


def test_memory_read():
    """
    Test memory read operation
    """
    print("\nTesting EpisodicMemory read...")

    memory = EpisodicMemory(num_slots=128, slot_dim=2560, kv_dim=153600)

    memory.memory = torch.randn(128, 2560)

    query = torch.randn(2, 2560)

    Z_r, addressing_weights = memory.read(query)

    assert Z_r.shape == (2, 2560), f"Wrong Z_r shape: {Z_r.shape}"
    assert addressing_weights.shape == (
        2, 128), f"Wrong addressing shape: {addressing_weights.shape}"

    weights_sum = addressing_weights.sum(dim=1)
    assert torch.allclose(weights_sum, torch.ones(
        2), atol=1e-5), f"Addressing weights don't sum to 1: {weights_sum}"

    print(f"  Query shape: {query.shape}")
    print(f"  Z_r shape: {Z_r.shape}")
    print(f"  Addressing weights shape: {addressing_weights.shape}")
    print(
        f"  Weights sum to 1: {torch.allclose(weights_sum, torch.ones(2), atol=1e-5)}")
    print("  ✓ Read operation passed")

    return True


def test_memory_kv_injection():
    """
    Test KV cache injection
    """
    print("\nTesting EpisodicMemory KV injection...")

    memory = EpisodicMemory(num_slots=128, slot_dim=2560, kv_dim=153600)

    Z_r = torch.randn(2, 2560)

    kv_cache = memory.inject_to_kv_cache(Z_r)

    assert kv_cache.shape == (
        2, 153600), f"Wrong KV cache shape: {kv_cache.shape}"

    print(f"  Z_r shape: {Z_r.shape}")
    print(f"  KV cache shape: {kv_cache.shape}")
    print("  ✓ KV injection passed")

    return True


def test_memory_save_load():
    """
    Test memory serialization and loading
    """
    print("\nTesting EpisodicMemory save/load...")

    memory1 = EpisodicMemory(num_slots=128, slot_dim=2560, kv_dim=153600)

    z_t = torch.randn(4, 2560)
    memory1.write(z_t)

    import tempfile
    with tempfile.NamedTemporaryFile(delete=False, suffix='.pt') as f:
        temp_path = f.name

    try:
        memory1.save_memory(temp_path)

        memory2 = EpisodicMemory(num_slots=128, slot_dim=2560, kv_dim=153600)
        memory2.load_memory(temp_path)

        assert torch.allclose(
            memory1.memory, memory2.memory), "Memory not loaded correctly"

        print(f"  Saved to: {temp_path}")
        print(
            f"  Memory matches after load: {torch.allclose(memory1.memory, memory2.memory)}")
        print("  ✓ Save/load passed")

    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

    return True


def run_all_tests():
    """
    Run all memory tests
    """
    print("=" * 60)
    print("EPISODIC MEMORY TESTS")
    print("=" * 60)

    tests = [
        test_memory_initialization,
        test_memory_write,
        test_memory_read,
        test_memory_kv_injection,
        test_memory_save_load
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
