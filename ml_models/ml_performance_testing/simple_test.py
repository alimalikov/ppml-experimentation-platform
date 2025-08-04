"""
Simple test for platform comparator imports
"""

print("Starting import test...")

try:
    import os
    import sys
    print("✅ Basic imports work")
    
    # Add paths
    PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    PERFORMANCE_TESTING_DIR = os.path.join(PROJECT_ROOT, 'ml_models', 'performance_testing')
    ML_PERFORMANCE_TESTING_DIR = os.path.dirname(__file__)
    
    print(f"Paths configured:")
    print(f"  PROJECT_ROOT: {PROJECT_ROOT}")
    print(f"  PERFORMANCE_TESTING_DIR: {PERFORMANCE_TESTING_DIR}")
    print(f"  ML_PERFORMANCE_TESTING_DIR: {ML_PERFORMANCE_TESTING_DIR}")
    
    # Test platform comparator import
    from platform_comparator import PlatformComparator
    print("✅ Platform comparator imported successfully!")
    
    # Test creating instance
    comparator = PlatformComparator(test_level='simple')
    print("✅ Platform comparator instance created successfully!")
    
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()

print("Test completed.")
