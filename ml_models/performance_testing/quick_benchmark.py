"""
Quick Performance Benchmark for Anonymization Platform
=====================================================

A lightweight script for quick performance testing of anonymization techniques.
Ideal for regular performance monitoring and quick comparisons.

Author: Bachelor Thesis Project  
Date: July 2025
"""

import os
import sys
import time
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Tuple

# Add project root to path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

try:
    from core.app import load_anonymizer_plugins, ANONYMIZER_PLUGINS
except ImportError as e:
    print(f"Error importing: {e}")
    print("Make sure you're running this from the ml_models directory")
    sys.exit(1)

def create_test_dataset(size: str = "medium") -> pd.DataFrame:
    """Create a test dataset for performance testing."""
    np.random.seed(42)
    
    sizes = {
        "tiny": 500,
        "small": 1000,
        "medium": 10000, 
        "large": 50000,
        "huge": 100000
    }
    
    n_rows = sizes.get(size, 10000)
    
    # Create realistic dataset
    data = {
        'id': range(n_rows),
        'age': np.random.randint(18, 80, n_rows),
        'income': np.random.normal(50000, 15000, n_rows),
        'education': np.random.choice(['High School', 'Bachelor', 'Master', 'PhD'], n_rows),
        'city': np.random.choice([f'City_{i}' for i in range(20)], n_rows),
        'sensitive_group': np.random.choice(['Group_A', 'Group_B', 'Group_C'], n_rows),
        'score1': np.random.normal(100, 15, n_rows),
        'score2': np.random.normal(80, 20, n_rows),
        'category': np.random.choice(['Cat_A', 'Cat_B', 'Cat_C', 'Cat_D'], n_rows),
        'is_member': np.random.choice([True, False], n_rows)
    }
    
    return pd.DataFrame(data)

def quick_benchmark(technique_name: str, plugin, df: pd.DataFrame) -> Dict:
    """Run a quick performance benchmark for a technique."""
    try:
        start_time = time.time()
        
        # Use empty parameters for quick testing
        result_df = plugin.anonymize(df.copy(), {}, 'sensitive_group')
        
        end_time = time.time()
        execution_time = end_time - start_time
        
        return {
            'technique': technique_name,
            'success': True,
            'execution_time': execution_time,
            'rows_processed': len(df),
            'throughput': len(df) / execution_time if execution_time > 0 else 0,
            'result_shape': result_df.shape,
            'data_preserved': len(result_df) == len(df)
        }
    
    except Exception as e:
        return {
            'technique': technique_name,
            'success': False,
            'error': str(e),
            'execution_time': None,
            'throughput': 0
        }

def run_quick_benchmark():
    """Run quick performance benchmark."""
    print("🚀 Quick Performance Benchmark")
    print("=" * 40)
    
    # Get user preferences
    print("\nSelect dataset sizes to test:")
    print("1. 🐣 Tiny (500 rows) - Ultra fast")
    print("2. 🚀 Small (1K rows) - Fast")
    print("3. 📊 Medium (10K rows) - Standard")
    print("4. 📈 Large (50K rows) - Comprehensive")
    print("5. 🏭 Huge (100K rows) - Stress test")
    print("6. 🌈 All sizes - Complete benchmark")
    
    while True:
        try:
            choice = input("\nEnter choice (1-6): ").strip()
            if choice == '1':
                sizes = ["tiny"]
                break
            elif choice == '2':
                sizes = ["small"]
                break
            elif choice == '3':
                sizes = ["medium"]
                break
            elif choice == '4':
                sizes = ["large"]
                break
            elif choice == '5':
                sizes = ["huge"]
                break
            elif choice == '6':
                sizes = ["tiny", "small", "medium", "large", "huge"]
                break
            else:
                print("❌ Invalid choice. Please enter 1-6.")
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            return
    
    # Ask about terminal output logging
    save_output = input("Save terminal output to file? (y/N): ").strip().lower()
    
    # Set up logging if requested
    terminal_logger = None
    if save_output == 'y':
        from datetime import datetime
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        size_str = "_".join(sizes) if len(sizes) <= 3 else "all_sizes"
        log_filename = f"quick_benchmark_{size_str}_{timestamp}.txt"
        
        class TerminalLogger:
            def __init__(self, filename):
                self.terminal = sys.stdout
                self.log = open(filename, 'w', encoding='utf-8')
            
            def write(self, message):
                self.terminal.write(message)
                self.log.write(message)
                self.log.flush()
            
            def flush(self):
                self.terminal.flush()
                self.log.flush()
            
            def close(self):
                self.log.close()
        
        terminal_logger = TerminalLogger(log_filename)
        sys.stdout = terminal_logger
        print(f"📝 Terminal output will be saved to: {log_filename}")
    
    try:
        # Load plugins
        load_anonymizer_plugins(include_test_plugin=False)
        
        # Skip homomorphic encryption techniques
        techniques_to_skip = [
            'homomorphic',
            'encryption',
            'Homomorphic Encryption',
            'Homomorphic Encryption Core',
            'homomorphic_encryption',
            'Homomorphic'
        ]
        
        original_count = len(ANONYMIZER_PLUGINS)
        plugins_to_test = ANONYMIZER_PLUGINS.copy()
        
        for technique_name in list(plugins_to_test.keys()):
            if any(skip_name.lower() in technique_name.lower() for skip_name in techniques_to_skip):
                print(f"⚠️  Skipping {technique_name} (homomorphic encryption)")
                del plugins_to_test[technique_name]
        
        skipped_count = original_count - len(plugins_to_test)
        print(f"📊 Loaded {len(plugins_to_test)} techniques")
        if skipped_count > 0:
            print(f"⏭️  Skipped {skipped_count} homomorphic encryption technique(s)")
        
        # Test with selected dataset sizes
        for size in sizes:
            print(f"\n📈 Testing with {size} dataset...")
            df = create_test_dataset(size)
            print(f"   Dataset shape: {df.shape}")
            
            results = []
            
            # Test each technique
            for technique_name, plugin in plugins_to_test.items():
                print(f"   ⚙️  Testing {technique_name}...", end=" ")
                
                result = quick_benchmark(technique_name, plugin, df)
                results.append(result)
                
                if result['success']:
                    print(f"✅ {result['execution_time']:.3f}s ({result['throughput']:.0f} rows/s)")
                else:
                    print(f"❌ Failed: {result.get('error', 'Unknown')[:50]}...")
            
            # Show top performers for this size
            successful_results = [r for r in results if r['success']]
            if successful_results:
                print(f"\n   🏆 Top 3 fastest for {size} dataset:")
                top_performers = sorted(successful_results, key=lambda x: x['execution_time'])[:3]
                
                for i, result in enumerate(top_performers, 1):
                    print(f"      {i}. {result['technique']}: {result['execution_time']:.3f}s")
            
            print(f"\n   📊 Success rate: {len(successful_results)}/{len(results)} techniques")
    
    finally:
        # Restore stdout and close log file if logging was enabled
        if terminal_logger:
            sys.stdout = terminal_logger.terminal
            terminal_logger.close()
            print(f"📄 Terminal output saved to: {log_filename}")
            
def run_single_technique_test(technique_name: str):
    """Test a single technique with detailed output."""
    load_anonymizer_plugins(include_test_plugin=False)
    
    # Skip homomorphic encryption techniques
    techniques_to_skip = [
        'homomorphic',
        'encryption',
        'Homomorphic Encryption',
        'Homomorphic Encryption Core',
        'homomorphic_encryption',
        'Homomorphic'
    ]
    
    plugins_to_test = ANONYMIZER_PLUGINS.copy()
    for tech_name in list(plugins_to_test.keys()):
        if any(skip_name.lower() in tech_name.lower() for skip_name in techniques_to_skip):
            del plugins_to_test[tech_name]
    
    if technique_name not in plugins_to_test:
        print(f"❌ Technique '{technique_name}' not found!")
        print(f"Available techniques: {list(plugins_to_test.keys())}")
        return
    
    plugin = plugins_to_test[technique_name]
    
    print(f"🔬 Detailed test for: {technique_name}")
    print("=" * 50)
    
    sizes = ["small", "medium", "large"]
    
    for size in sizes:
        df = create_test_dataset(size)
        print(f"\n📊 Testing {size} dataset (Shape: {df.shape})")
        
        # Run multiple iterations for average
        times = []
        for i in range(3):
            result = quick_benchmark(technique_name, plugin, df)
            if result['success']:
                times.append(result['execution_time'])
        
        if times:
            avg_time = sum(times) / len(times)
            min_time = min(times)
            max_time = max(times)
            
            print(f"   ⏱️  Average time: {avg_time:.3f}s")
            print(f"   📈 Min time: {min_time:.3f}s")
            print(f"   📉 Max time: {max_time:.3f}s")
            print(f"   🚀 Throughput: {len(df) / avg_time:.0f} rows/s")
        else:
            print(f"   ❌ All tests failed")

def compare_techniques(technique_names: List[str]):
    """Compare specific techniques side by side."""
    load_anonymizer_plugins(include_test_plugin=False)
    
    # Skip homomorphic encryption techniques
    techniques_to_skip = [
        'homomorphic',
        'encryption',
        'Homomorphic Encryption',
        'Homomorphic Encryption Core',
        'homomorphic_encryption',
        'Homomorphic'
    ]
    
    plugins_to_test = ANONYMIZER_PLUGINS.copy()
    for tech_name in list(plugins_to_test.keys()):
        if any(skip_name.lower() in tech_name.lower() for skip_name in techniques_to_skip):
            del plugins_to_test[tech_name]
    
    # Validate technique names
    valid_techniques = []
    for name in technique_names:
        if name in plugins_to_test:
            valid_techniques.append(name)
        else:
            print(f"⚠️  Technique '{name}' not found, skipping...")
    
    if not valid_techniques:
        print("❌ No valid techniques to compare!")
        return
    
    print(f"⚖️  Comparing {len(valid_techniques)} techniques")
    print("=" * 50)
    
    df = create_test_dataset("medium")
    print(f"📊 Test dataset shape: {df.shape}")
    
    results = []
    
    for technique_name in valid_techniques:
        plugin = plugins_to_test[technique_name]
        result = quick_benchmark(technique_name, plugin, df)
        results.append(result)
    
    # Display comparison table
    print(f"\n{'Technique':<25} {'Status':<8} {'Time (s)':<10} {'Throughput':<15}")
    print("-" * 65)
    
    for result in results:
        status = "✅ Pass" if result['success'] else "❌ Fail"
        time_str = f"{result['execution_time']:.3f}" if result['success'] else "N/A"
        throughput_str = f"{result['throughput']:.0f} rows/s" if result['success'] else "N/A"
        
        print(f"{result['technique']:<25} {status:<8} {time_str:<10} {throughput_str:<15}")
    
    # Show winner
    successful = [r for r in results if r['success']]
    if successful:
        winner = min(successful, key=lambda x: x['execution_time'])
        print(f"\n🏆 Fastest: {winner['technique']} ({winner['execution_time']:.3f}s)")

def main():
    """Main function with command line interface."""
    if len(sys.argv) == 1:
        # Run full benchmark
        run_quick_benchmark()
    
    elif len(sys.argv) == 2:
        command = sys.argv[1]
        
        if command == "--help" or command == "-h":
            print("Quick Performance Benchmark Tool")
            print("Usage:")
            print("  python quick_benchmark.py                    # Run full benchmark")
            print("  python quick_benchmark.py [technique_name]   # Test single technique")
            print("  python quick_benchmark.py --list             # List available techniques")
            
        elif command == "--list":
            load_anonymizer_plugins(include_test_plugin=False)
            
            # Skip homomorphic encryption techniques
            techniques_to_skip = [
                'homomorphic',
                'encryption',
                'Homomorphic Encryption',
                'Homomorphic Encryption Core',
                'homomorphic_encryption',
                'Homomorphic'
            ]
            
            plugins_to_test = ANONYMIZER_PLUGINS.copy()
            skipped_techniques = []
            
            for tech_name in list(plugins_to_test.keys()):
                if any(skip_name.lower() in tech_name.lower() for skip_name in techniques_to_skip):
                    skipped_techniques.append(tech_name)
                    del plugins_to_test[tech_name]
            
            print("Available techniques:")
            for i, name in enumerate(plugins_to_test.keys(), 1):
                print(f"  {i}. {name}")
            
            if skipped_techniques:
                print(f"\n⚠️  Skipped {len(skipped_techniques)} homomorphic encryption technique(s):")
                for tech in skipped_techniques:
                    print(f"  - {tech}")
        
        else:
            # Test single technique
            run_single_technique_test(command)
    
    elif len(sys.argv) > 2:
        # Compare multiple techniques
        technique_names = sys.argv[1:]
        compare_techniques(technique_names)

if __name__ == "__main__":
    main()
