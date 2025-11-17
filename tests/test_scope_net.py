from src.scope_net import ScopeNet
import sys
import os
from pathlib import Path
import torch

sys.path.append(str(Path(__file__).parent.parent))


def test_scope_net_output():
    """
    Test ScopeNet output shape and range
    """
    print("Testing ScopeNet output...")

    config = {
        'input_dim': 2560,
        'hidden_dim': 256,
        'num_layers': 2,
        'dropout': 0.1
    }

    scope_net = ScopeNet(config)

    batch_size = 4
    x = torch.randn(batch_size, 2560)

    decision, probability = scope_net(x)

    assert decision.shape == (
        batch_size,), f"Wrong decision shape: {decision.shape}"
    assert probability.shape == (
        batch_size,), f"Wrong probability shape: {probability.shape}"

    assert (probability >= 0).all() and (probability <=
                                         1).all(), "Probabilities out of [0, 1] range"

    assert ((decision == 0) | (decision == 1)).all(), "Decisions not binary"

    print(f"  Input shape: {x.shape}")
    print(f"  Decision shape: {decision.shape}")
    print(f"  Probability shape: {probability.shape}")
    print(
        f"  Probability range: [{probability.min():.4f}, {probability.max():.4f}]")
    print(
        f"  Decisions are binary: {((decision == 0) | (decision == 1)).all()}")
    print("  ✓ Output test passed")

    return True


def test_scope_net_loss():
    """
    Test ScopeNet loss computation
    """
    print("\nTesting ScopeNet loss...")

    config = {
        'input_dim': 2560,
        'hidden_dim': 256,
        'num_layers': 2,
        'dropout': 0.0
    }

    scope_net = ScopeNet(config)

    x = torch.randn(4, 2560)
    target = torch.tensor([1.0, 0.0, 1.0, 0.0])

    loss = scope_net.get_loss(x, target)

    assert loss.dim() == 0, "Loss should be scalar"
    assert loss.item() >= 0, "Loss should be non-negative"

    print(f"  Loss value: {loss.item():.6f}")
    print(f"  Loss is scalar: {loss.dim() == 0}")
    print("  ✓ Loss computation passed")

    return True


def test_scope_net_gradient_flow():
    """
    Test gradient flow through ScopeNet
    """
    print("\nTesting ScopeNet gradient flow...")

    config = {
        'input_dim': 2560,
        'hidden_dim': 256,
        'num_layers': 2,
        'dropout': 0.0
    }

    scope_net = ScopeNet(config)

    x = torch.randn(2, 2560, requires_grad=True)
    target = torch.tensor([1.0, 0.0])

    loss = scope_net.get_loss(x, target)
    loss.backward()

    assert x.grad is not None, "No gradient for input"

    for name, param in scope_net.named_parameters():
        assert param.grad is not None, f"No gradient for {name}"

    print("  ✓ Gradients flow correctly")

    return True


def test_scope_net_decision_threshold():
    """
    Test that decision changes with probability threshold
    """
    print("\nTesting ScopeNet decision threshold...")

    config = {
        'input_dim': 2560,
        'hidden_dim': 256,
        'num_layers': 2,
        'dropout': 0.0
    }

    scope_net = ScopeNet(config)
    scope_net.eval()

    x = torch.randn(100, 2560)

    with torch.no_grad():
        decision, probability = scope_net(x)

    positive_rate = decision.float().mean()

    assert 0 <= positive_rate <= 1, "Positive rate out of range"

    print(f"  Positive decision rate: {positive_rate:.2%}")
    print(f"  Mean probability: {probability.mean():.4f}")
    print("  ✓ Decision threshold test passed")

    return True


def run_all_tests():
    """
    Run all ScopeNet tests
    """
    print("=" * 60)
    print("SCOPENET TESTS")
    print("=" * 60)

    tests = [
        test_scope_net_output,
        test_scope_net_loss,
        test_scope_net_gradient_flow,
        test_scope_net_decision_threshold
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
