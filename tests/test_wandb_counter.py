import sys
import os
from pathlib import Path
import torch

sys.path.append(str(Path(__file__).parent.parent))

from src.wandb_counter import WandBRunCounter


def test_counter_initialization():
    """
    Test WandBRunCounter initialization
    """
    print("Testing WandBRunCounter initialization...")
    
    import tempfile
    temp_dir = tempfile.mkdtemp()
    counter_path = os.path.join(temp_dir, 'test_counter.json')
    
    try:
        counter = WandBRunCounter(counter_path)
        
        assert os.path.exists(counter_path), "Counter file not created"
        
        print(f"  Counter file: {counter_path}")
        print("  ✓ Initialization passed")
        
        return True
        
    finally:
        if os.path.exists(counter_path):
            os.remove(counter_path)
        if os.path.exists(temp_dir):
            os.rmdir(temp_dir)


def test_counter_increment():
    """
    Test run counter increments correctly
    """
    print("\nTesting WandBRunCounter increment...")
    
    import tempfile
    temp_dir = tempfile.mkdtemp()
    counter_path = os.path.join(temp_dir, 'test_counter.json')
    
    try:
        counter = WandBRunCounter(counter_path)
        
        run_name1, run_number1 = counter.get_next_run_name('test')
        run_name2, run_number2 = counter.get_next_run_name('test')
        run_name3, run_number3 = counter.get_next_run_name('other')
        
        assert run_number2 == run_number1 + 1, f"Counter not incrementing: {run_number1} -> {run_number2}"
        assert run_number3 == run_number2 + 1, f"Counter not incrementing across configs: {run_number2} -> {run_number3}"
        
        assert 'run_1' in run_name1, f"Wrong run name format: {run_name1}"
        assert 'run_2' in run_name2, f"Wrong run name format: {run_name2}"
        
        print(f"  Run 1: {run_name1} (#{run_number1})")
        print(f"  Run 2: {run_name2} (#{run_number2})")
        print(f"  Run 3: {run_name3} (#{run_number3})")
        print("  ✓ Increment test passed")
        
        return True
        
    finally:
        if os.path.exists(counter_path):
            os.remove(counter_path)
        if os.path.exists(temp_dir):
            os.rmdir(temp_dir)


def test_counter_persistence():
    """
    Test that counter persists across instances
    """
    print("\nTesting WandBRunCounter persistence...")
    
    import tempfile
    temp_dir = tempfile.mkdtemp()
    counter_path = os.path.join(temp_dir, 'test_counter.json')
    
    try:
        counter1 = WandBRunCounter(counter_path)
        run_name1, run_number1 = counter1.get_next_run_name('test')
        
        counter2 = WandBRunCounter(counter_path)
        run_name2, run_number2 = counter2.get_next_run_name('test')
        
        assert run_number2 == run_number1 + 1, f"Counter not persisting: {run_number1} -> {run_number2}"
        
        print(f"  Instance 1: {run_name1} (#{run_number1})")
        print(f"  Instance 2: {run_name2} (#{run_number2})")
        print("  ✓ Persistence test passed")
        
        return True
        
    finally:
        if os.path.exists(counter_path):
            os.remove(counter_path)
        if os.path.exists(temp_dir):
            os.rmdir(temp_dir)


def test_counter_format():
    """
    Test run name formatting
    """
    print("\nTesting WandBRunCounter name format...")
    
    import tempfile
    temp_dir = tempfile.mkdtemp()
    counter_path = os.path.join(temp_dir, 'test_counter.json')
    
    try:
        counter = WandBRunCounter(counter_path)
        
        run_name, run_number = counter.get_next_run_name('stage1')
        
        assert 'run_' in run_name, "Run name should contain 'run_'"
        assert 'stage1' in run_name, "Run name should contain config name"
        assert str(run_number) in run_name, "Run name should contain run number"
        
        print(f"  Generated name: {run_name}")
        print(f"  Run number: {run_number}")
        print("  ✓ Format test passed")
        
        return True
        
    finally:
        if os.path.exists(counter_path):
            os.remove(counter_path)
        if os.path.exists(temp_dir):
            os.rmdir(temp_dir)


def run_all_tests():
    """
    Run all WandB counter tests
    """
    print("=" * 60)
    print("WANDB RUN COUNTER TESTS")
    print("=" * 60)
    
    tests = [
        test_counter_initialization,
        test_counter_increment,
        test_counter_persistence,
        test_counter_format
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
