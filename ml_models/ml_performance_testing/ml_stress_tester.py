"""
ML Stress Testing Suite
======================

Tests edge cases, robustness, and failure scenarios for ML algorithms.
Helps identify potential issues and limitations of the ML platform.

Author: Bachelor Thesis Project
Date: July 2025
"""

import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Any
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

SRC_DIR = os.path.join(PROJECT_ROOT, 'src')
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

try:
    from src.ml_plugins.plugin_manager import get_plugin_manager
    from sklearn.model_selection import train_test_split
    from sklearn.datasets import make_classification, make_regression
    from sklearn.preprocessing import StandardScaler
except ImportError as e:
    print(f"Error importing: {e}")
    sys.exit(1)

class MLStressTester:
    """
    Stress testing framework for ML algorithms.
    """
    
    def __init__(self, test_level='full', output_file=None):
        self.test_level = test_level  # 'simple', 'medium', 'full'
        self.output_file = output_file
        self.load_algorithms()
        self.results = {
            'timestamp': datetime.now().isoformat(),
            'test_level': test_level,
            'edge_case_tests': {},
            'robustness_tests': {},
            'data_quality_tests': {},
            'failure_analysis': {}
        }
    
    def load_algorithms(self):
        """Load ML algorithms."""
        try:
            plugin_manager = get_plugin_manager()
            self.classification_algorithms = plugin_manager.get_available_plugins("classification")
            self.regression_algorithms = plugin_manager.get_available_plugins("regression")
            
            self.all_algorithms = {}
            self.all_algorithms.update(self.classification_algorithms)
            self.all_algorithms.update(self.regression_algorithms)
            
            print(f"🔧 Loaded {len(self.all_algorithms)} ML algorithms for stress testing")
            print(f"   📊 Classification: {len(self.classification_algorithms)}")
            print(f"   📈 Regression: {len(self.regression_algorithms)}")
            
        except Exception as e:
            print(f"⚠️  Error loading ML algorithms: {e}")
            self.all_algorithms = {}
    
    def create_edge_case_datasets(self) -> Dict[str, Dict]:
        """Create datasets with edge cases and challenging characteristics."""
        datasets = {}
        
        # Adjust dataset sizes based on test level
        if self.test_level == 'simple':
            small_size, medium_size, large_size = 100, 500, 1000
            n_features = 10
            print(f"📊 Using simple dataset sizes: {small_size}-{large_size} samples")
        elif self.test_level == 'medium':
            small_size, medium_size, large_size = 500, 2000, 5000
            n_features = 20
            print(f"📊 Using medium dataset sizes: {small_size}-{large_size} samples")
        else:  # full
            small_size, medium_size, large_size = 1000, 5000, 10000
            n_features = 30
            print(f"📊 Using full dataset sizes: {small_size}-{large_size} samples")
        
        # 1. Minimal dataset
        X_min, y_min = make_classification(n_samples=50, n_features=5, n_classes=2, random_state=42)
        datasets['minimal_classification'] = {
            'data': pd.DataFrame(X_min, columns=[f'f_{i}' for i in range(5)]).assign(target=y_min),
            'task_type': 'classification',
            'description': 'Minimal dataset (50 samples, 5 features)'
        }
        
        # 2. Imbalanced classification dataset
        X_imb, y_imb = make_classification(
            n_samples=medium_size, n_features=n_features, n_classes=3,
            weights=[0.8, 0.15, 0.05], random_state=42
        )
        datasets['imbalanced_classification'] = {
            'data': pd.DataFrame(X_imb, columns=[f'f_{i}' for i in range(n_features)]).assign(target=y_imb),
            'task_type': 'classification',
            'description': 'Highly imbalanced classes (80%, 15%, 5%)'
        }
        
        # 3. High-dimensional dataset (more features than samples)
        if self.test_level != 'simple':
            n_samples_hd = small_size
            n_features_hd = small_size + 50  # More features than samples
            X_hd, y_hd = make_classification(
                n_samples=n_samples_hd, n_features=n_features_hd, n_informative=10,
                n_redundant=5, random_state=42
            )
            datasets['high_dimensional_classification'] = {
                'data': pd.DataFrame(X_hd, columns=[f'f_{i}' for i in range(n_features_hd)]).assign(target=y_hd),
                'task_type': 'classification',
                'description': f'High-dimensional ({n_samples_hd} samples, {n_features_hd} features)'
            }
        
        # 4. Noisy regression dataset
        X_noisy, y_noisy = make_regression(
            n_samples=medium_size, n_features=n_features, noise=5.0, random_state=42
        )
        datasets['noisy_regression'] = {
            'data': pd.DataFrame(X_noisy, columns=[f'f_{i}' for i in range(n_features)]).assign(target=y_noisy),
            'task_type': 'regression',
            'description': 'High-noise regression dataset'
        }
        
        # 5. Perfect linear relationship (for regression)
        X_perfect = np.random.randn(medium_size, n_features)
        y_perfect = np.sum(X_perfect, axis=1)  # Perfect linear relationship
        datasets['perfect_linear_regression'] = {
            'data': pd.DataFrame(X_perfect, columns=[f'f_{i}' for i in range(n_features)]).assign(target=y_perfect),
            'task_type': 'regression',
            'description': 'Perfect linear relationship'
        }
        
        # 6. Constant target (regression)
        X_const, _ = make_regression(n_samples=medium_size, n_features=n_features, random_state=42)
        y_const = np.full(medium_size, 5.0)  # Constant target
        datasets['constant_target_regression'] = {
            'data': pd.DataFrame(X_const, columns=[f'f_{i}' for i in range(n_features)]).assign(target=y_const),
            'task_type': 'regression',
            'description': 'Constant target values'
        }
        
        # 7. Binary classification with perfect separation
        if self.test_level == 'full':
            X_sep = np.random.randn(medium_size, n_features)
            y_sep = (X_sep[:, 0] > 0).astype(int)  # Perfect separation based on first feature
            datasets['perfect_separation_classification'] = {
                'data': pd.DataFrame(X_sep, columns=[f'f_{i}' for i in range(n_features)]).assign(target=y_sep),
                'task_type': 'classification',
                'description': 'Perfect class separation'
            }
        
        # 8. Large dataset (only in full mode)
        if self.test_level == 'full':
            X_large, y_large = make_classification(
                n_samples=large_size, n_features=n_features, n_classes=5, random_state=42
            )
            datasets['large_classification'] = {
                'data': pd.DataFrame(X_large, columns=[f'f_{i}' for i in range(n_features)]).assign(target=y_large),
                'task_type': 'classification',
                'description': f'Large dataset ({large_size} samples, 5 classes)'
            }
        
        return datasets
    
    def test_edge_cases(self):
        """Test how algorithms handle edge cases."""
        print("\n🧪 Testing edge cases...")
        
        edge_datasets = self.create_edge_case_datasets()
        
        for dataset_name, dataset_info in edge_datasets.items():
            print(f"  📊 Testing {dataset_name}...")
            
            df = dataset_info['data']
            task_type = dataset_info['task_type']
            
            self.results['edge_case_tests'][dataset_name] = {
                'dataset_info': {
                    'shape': df.shape,
                    'task_type': task_type,
                    'description': dataset_info['description'],
                    'memory_usage': df.memory_usage(deep=True).sum() if not df.empty else 0
                },
                'algorithm_results': {}
            }
            
            # Get compatible algorithms
            algorithms = self.classification_algorithms if task_type == 'classification' else self.regression_algorithms
            
            for algorithm_name, algorithm_plugin in algorithms.items():
                try:
                    # Prepare data
                    X = df.drop('target', axis=1)
                    y = df['target']
                    
                    # Check if dataset is too small for train/test split
                    if len(df) < 10:
                        X_train, X_test, y_train, y_test = X, X, y, y
                    else:
                        X_train, X_test, y_train, y_test = train_test_split(
                            X, y, test_size=0.2, random_state=42,
                            stratify=y if task_type == 'classification' and len(np.unique(y)) <= len(y) // 2 else None
                        )
                    
                    # Scale features
                    scaler = StandardScaler()
                    X_train_scaled = scaler.fit_transform(X_train)
                    X_test_scaled = scaler.transform(X_test)
                    
                    # Create and train model
                    if hasattr(algorithm_plugin, 'create_model_instance'):
                        model = algorithm_plugin.create_model_instance({})
                    else:
                        model = algorithm_plugin
                    
                    model.fit(X_train_scaled, y_train)
                    y_pred = model.predict(X_test_scaled)
                    
                    # Calculate basic metrics
                    if task_type == 'classification':
                        from sklearn.metrics import accuracy_score
                        accuracy = accuracy_score(y_test, y_pred) if len(y_test) > 0 else 0.0
                        metric_value = accuracy
                        metric_name = 'accuracy'
                    else:
                        from sklearn.metrics import r2_score
                        r2 = r2_score(y_test, y_pred) if len(y_test) > 0 else 0.0
                        metric_value = r2
                        metric_name = 'r2_score'
                    
                    test_result = {
                        'success': True,
                        'train_samples': len(X_train),
                        'test_samples': len(X_test),
                        'features': X.shape[1],
                        metric_name: metric_value,
                        'predictions_made': len(y_pred),
                        'error_message': None
                    }
                    
                except Exception as e:
                    test_result = {
                        'success': False,
                        'error_message': str(e),
                        'error_type': type(e).__name__
                    }
                
                self.results['edge_case_tests'][dataset_name]['algorithm_results'][algorithm_name] = test_result
        
        print("✅ Edge case testing completed")
    
    def test_robustness(self):
        """Test robustness with various configurations."""
        print("\n💪 Testing robustness...")
        
        # Adjust dataset size based on test level
        if self.test_level == 'simple':
            n_samples = 500
            n_features = 10
        elif self.test_level == 'medium':
            n_samples = 1500
            n_features = 15
        else:  # full
            n_samples = 3000
            n_features = 20
        
        print(f"📊 Using {n_samples} samples for robustness testing")
        
        # Create standard test datasets
        X_class, y_class = make_classification(
            n_samples=n_samples, n_features=n_features, n_classes=3, random_state=42
        )
        df_class = pd.DataFrame(X_class, columns=[f'f_{i}' for i in range(n_features)])
        df_class['target'] = y_class
        
        X_reg, y_reg = make_regression(
            n_samples=n_samples, n_features=n_features, random_state=42
        )
        df_reg = pd.DataFrame(X_reg, columns=[f'f_{i}' for i in range(n_features)])
        df_reg['target'] = y_reg
        
        test_datasets = {
            'classification': {'data': df_class, 'task_type': 'classification'},
            'regression': {'data': df_reg, 'task_type': 'regression'}
        }
        
        for dataset_name, dataset_info in test_datasets.items():
            algorithms = (self.classification_algorithms if dataset_info['task_type'] == 'classification' 
                         else self.regression_algorithms)
            
            for algorithm_name, algorithm_plugin in algorithms.items():
                print(f"  🔧 Testing {algorithm_name} robustness on {dataset_name}...")
                
                algorithm_results = []
                
                # Test with different train/test splits
                test_sizes = [0.1, 0.2, 0.3, 0.4] if self.test_level == 'full' else [0.2, 0.3]
                
                for test_size in test_sizes:
                    try:
                        df = dataset_info['data']
                        X = df.drop('target', axis=1)
                        y = df['target']
                        
                        X_train, X_test, y_train, y_test = train_test_split(
                            X, y, test_size=test_size, random_state=42,
                            stratify=y if dataset_info['task_type'] == 'classification' else None
                        )
                        
                        scaler = StandardScaler()
                        X_train_scaled = scaler.fit_transform(X_train)
                        X_test_scaled = scaler.transform(X_test)
                        
                        if hasattr(algorithm_plugin, 'create_model_instance'):
                            model = algorithm_plugin.create_model_instance({})
                        else:
                            model = algorithm_plugin
                        
                        model.fit(X_train_scaled, y_train)
                        y_pred = model.predict(X_test_scaled)
                        
                        test_result = {
                            'test_size': test_size,
                            'train_samples': len(X_train),
                            'test_samples': len(X_test),
                            'success': True,
                            'predictions_made': len(y_pred)
                        }
                        
                    except Exception as e:
                        test_result = {
                            'test_size': test_size,
                            'success': False,
                            'error': str(e),
                            'error_type': type(e).__name__
                        }
                    
                    algorithm_results.append(test_result)
                
                self.results['robustness_tests'][f"{algorithm_name}_{dataset_name}"] = algorithm_results
        
        print("✅ Robustness testing completed")
    
    def analyze_failures(self):
        """Analyze failure patterns across all tests."""
        print("\n📊 Analyzing failure patterns...")
        
        failure_analysis = {
            'total_tests': 0,
            'total_failures': 0,
            'failure_by_algorithm': {},
            'failure_by_dataset_type': {},
            'common_error_types': {}
        }
        
        # Analyze edge case failures
        for dataset_name, dataset_results in self.results['edge_case_tests'].items():
            for algorithm_name, result in dataset_results['algorithm_results'].items():
                failure_analysis['total_tests'] += 1
                
                if not result.get('success', False):
                    failure_analysis['total_failures'] += 1
                    
                    # Track by algorithm
                    if algorithm_name not in failure_analysis['failure_by_algorithm']:
                        failure_analysis['failure_by_algorithm'][algorithm_name] = 0
                    failure_analysis['failure_by_algorithm'][algorithm_name] += 1
                    
                    # Track by dataset type
                    if dataset_name not in failure_analysis['failure_by_dataset_type']:
                        failure_analysis['failure_by_dataset_type'][dataset_name] = 0
                    failure_analysis['failure_by_dataset_type'][dataset_name] += 1
                    
                    # Track error types
                    error_type = result.get('error_type', 'Unknown')
                    if error_type not in failure_analysis['common_error_types']:
                        failure_analysis['common_error_types'][error_type] = 0
                    failure_analysis['common_error_types'][error_type] += 1
        
        # Calculate failure rates
        if failure_analysis['total_tests'] > 0:
            failure_analysis['overall_failure_rate'] = (
                failure_analysis['total_failures'] / failure_analysis['total_tests']
            )
        
        self.results['failure_analysis'] = failure_analysis
        
        print(f"📈 Overall failure rate: {failure_analysis.get('overall_failure_rate', 0)*100:.1f}%")
        print("✅ Failure analysis completed")
    
    def generate_stress_test_report(self):
        """Generate comprehensive stress test report."""
        print("\n📋 Generating ML stress test report...")
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        report_file = f"ml_stress_test_report_{timestamp}.txt"
        
        with open(report_file, 'w') as f:
            f.write("ML PLATFORM STRESS TEST REPORT\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # Summary
            f.write("SUMMARY\n")
            f.write("-" * 20 + "\n")
            failure_analysis = self.results.get('failure_analysis', {})
            f.write(f"Total Tests: {failure_analysis.get('total_tests', 0)}\n")
            f.write(f"Total Failures: {failure_analysis.get('total_failures', 0)}\n")
            f.write(f"Overall Success Rate: {(1 - failure_analysis.get('overall_failure_rate', 0))*100:.1f}%\n\n")
            
            # Edge case results
            f.write("EDGE CASE TEST RESULTS\n")
            f.write("-" * 30 + "\n")
            for dataset_name, results in self.results['edge_case_tests'].items():
                f.write(f"\n{dataset_name}:\n")
                f.write(f"  Description: {results['dataset_info']['description']}\n")
                success_count = sum(1 for r in results['algorithm_results'].values() if r.get('success'))
                total_count = len(results['algorithm_results'])
                f.write(f"  Success Rate: {success_count}/{total_count} ({success_count/total_count*100:.1f}%)\n")
            
            # Most problematic datasets
            f.write("\nMOST CHALLENGING DATASETS\n")
            f.write("-" * 30 + "\n")
            if 'failure_by_dataset_type' in failure_analysis:
                sorted_failures = sorted(
                    failure_analysis['failure_by_dataset_type'].items(),
                    key=lambda x: x[1], reverse=True
                )
                for dataset, failures in sorted_failures[:5]:
                    f.write(f"  {dataset}: {failures} failures\n")
            
            # Most problematic algorithms
            f.write("\nALGORITHMS WITH MOST FAILURES\n")
            f.write("-" * 35 + "\n")
            if 'failure_by_algorithm' in failure_analysis:
                sorted_alg_failures = sorted(
                    failure_analysis['failure_by_algorithm'].items(),
                    key=lambda x: x[1], reverse=True
                )
                for algorithm, failures in sorted_alg_failures[:5]:
                    f.write(f"  {algorithm}: {failures} failures\n")
            
            # Common error types
            f.write("\nCOMMON ERROR TYPES\n")
            f.write("-" * 20 + "\n")
            if 'common_error_types' in failure_analysis:
                sorted_errors = sorted(
                    failure_analysis['common_error_types'].items(),
                    key=lambda x: x[1], reverse=True
                )
                for error_type, count in sorted_errors:
                    f.write(f"  {error_type}: {count} occurrences\n")
        
        print(f"📄 ML stress test report saved to: {report_file}")
    
    def run_all_tests(self):
        """Run all stress tests."""
        print("🔬 Starting Comprehensive ML Stress Testing")
        print("=" * 50)
        print(f"📊 Test Level: {self.test_level.upper()}")
        
        self.test_edge_cases()
        self.test_robustness()
        self.analyze_failures()
        self.generate_stress_test_report()
        
        print("\n🎉 ML stress testing completed successfully!")


class TerminalLogger:
    """Class to capture and save terminal output to file."""
    
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


def get_user_preferences():
    """Get user preferences for test level and dataset size."""
    print("🔬 ML Platform Stress Testing")
    print("=" * 40)
    
    # Get test level
    print("\nSelect test level:")
    print("1. 🚀 Simple (Fast, smaller datasets)")
    print("2. 📊 Medium (Balanced, medium datasets)")
    print("3. 🔬 Full (Comprehensive, larger datasets)")
    
    while True:
        try:
            choice = input("\nEnter choice (1-3): ").strip()
            if choice == '1':
                test_level = 'simple'
                break
            elif choice == '2':
                test_level = 'medium'
                break
            elif choice == '3':
                test_level = 'full'
                break
            else:
                print("❌ Invalid choice. Please enter 1, 2, or 3.")
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            sys.exit(0)
    
    # Ask about saving terminal output
    print(f"\nSelected test level: {test_level.upper()}")
    save_output = input("Save terminal output to file? (y/N): ").strip().lower()
    
    return test_level, save_output == 'y'


def main():
    """Main function to run ML stress tests."""
    test_level, save_output = get_user_preferences()
    
    # Set up terminal logging if requested
    terminal_logger = None
    if save_output:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_filename = f"ml_stress_test_terminal_output_{test_level}_{timestamp}.txt"
        terminal_logger = TerminalLogger(log_filename)
        sys.stdout = terminal_logger
        print(f"📝 Terminal output will be saved to: {log_filename}")
    
    try:
        tester = MLStressTester(test_level=test_level)
        tester.run_all_tests()
    
    finally:
        # Restore original stdout and close log file
        if terminal_logger:
            sys.stdout = terminal_logger.terminal
            terminal_logger.close()
            print(f"📄 Terminal output saved to: {log_filename}")


if __name__ == "__main__":
    main()
