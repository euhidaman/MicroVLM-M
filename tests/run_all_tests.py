import sys
import os
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

print("=" * 60)
print("RUNNING ALL MICROVLM-M TESTS")
print("=" * 60)

test_modules = [
    'test_adapter',
    'test_memory',
    'test_scope_net',
    'test_visualizer',
    'test_wandb_counter'
]

passed_modules = 0
failed_modules = 0

for module_name in test_modules:
    print(f"\n{'=' * 60}")
    print(f"Running {module_name}")
    print(f"{'=' * 60}\n")
    
    try:
        module = __import__(module_name)
        success = module.run_all_tests()
        
        if success:
            passed_modules += 1
            print(f"\n✓ {module_name} PASSED")
        else:
            failed_modules += 1
            print(f"\n✗ {module_name} FAILED")
            
    except Exception as e:
        failed_modules += 1
        print(f"\n✗ {module_name} FAILED with exception: {e}")
        import traceback
        traceback.print_exc()

print("\n" + "=" * 60)
print("FINAL RESULTS")
print("=" * 60)
print(f"Modules passed: {passed_modules}/{len(test_modules)}")
print(f"Modules failed: {failed_modules}/{len(test_modules)}")
print("=" * 60)

sys.exit(0 if failed_modules == 0 else 1)
