"""
Test script to verify imports work correctly
"""

import os
import sys

# Add paths
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
PERFORMANCE_TESTING_DIR = os.path.join(PROJECT_ROOT, 'ml_models', 'performance_testing')
ML_PERFORMANCE_TESTING_DIR = os.path.dirname(__file__)

print(f"Project root: {PROJECT_ROOT}")
print(f"Performance testing dir: {PERFORMANCE_TESTING_DIR}")
print(f"ML performance testing dir: {ML_PERFORMANCE_TESTING_DIR}")

# Add to sys.path
if PERFORMANCE_TESTING_DIR not in sys.path:
    sys.path.insert(0, PERFORMANCE_TESTING_DIR)
if ML_PERFORMANCE_TESTING_DIR not in sys.path:
    sys.path.insert(0, ML_PERFORMANCE_TESTING_DIR)

print("\nChecking file existence:")
perf_tester_path = os.path.join(PERFORMANCE_TESTING_DIR, 'performance_tester.py')
ml_perf_tester_path = os.path.join(ML_PERFORMANCE_TESTING_DIR, 'ml_performance_tester.py')

print(f"performance_tester.py exists: {os.path.exists(perf_tester_path)}")
print(f"ml_performance_tester.py exists: {os.path.exists(ml_perf_tester_path)}")

print("\nTrying imports:")

try:
    from performance_tester import PerformanceTester
    print("✅ performance_tester imported successfully")
except Exception as e:
    print(f"❌ performance_tester import failed: {e}")
    import traceback
    traceback.print_exc()

try:
    from ml_performance_tester import MLPerformanceTester
    print("✅ ml_performance_tester imported successfully")
except Exception as e:
    print(f"❌ ml_performance_tester import failed: {e}")
    import traceback
    traceback.print_exc()

print("\nTest completed!")
