"""
Quick ML Benchmark Suite
========================

Fast performance benchmarking for ML algorithms with configurable dataset sizes.

Author: Bachelor Thesis Project
Date: July 2025
"""

import os
import sys
import time
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List
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

def create_ml_test_dataset(size: str) -> Dict[str, Dict]:
    """Create test datasets for ML benchmarking."""
    size_configs = {
        'tiny': {'n_samples': 500, 'n_features': 10},
        'small': {'n_samples': 1000, 'n_features': 15},
        'medium': {'n_samples': 10000, 'n_features': 20},
        'large': {'n_samples': 50000, 'n_features': 30},
        'huge': {'n_samples': 100000, 'n_features': 40}
    }
    
    config = size_configs[size]
    datasets = {}
    
    # Classification dataset
    X_class, y_class = make_classification(
        n_samples=config['n_samples'],
        n_features=config['n_features'],
        n_informative=max(5, config['n_features'] // 2),
        n_redundant=max(2, config['n_features'] // 4),
        n_classes=3,
        random_state=42
    )
    
    feature_names = [f'feature_{i}' for i in range(config['n_features'])]
    df_class = pd.DataFrame(X_class, columns=feature_names)
    df_class['target'] = y_class
    
    datasets['classification'] = {
        'data': df_class,
        'task_type': 'classification',
        'n_samples': config['n_samples'],
        'n_features': config['n_features']
    }
    
    # Regression dataset
    X_reg, y_reg = make_regression(
        n_samples=config['n_samples'],
        n_features=config['n_features'],
        n_informative=max(5, config['n_features'] // 2),
        noise=0.1,
        random_state=42
    )
    
    df_reg = pd.DataFrame(X_reg, columns=feature_names)
    df_reg['target'] = y_reg
    
    datasets['regression'] = {
        'data': df_reg,
        'task_type': 'regression',
        'n_samples': config['n_samples'],
        'n_features': config['n_features']
    }
    
    return datasets

def quick_ml_benchmark(algorithm_name: str, algorithm_plugin, dataset_info: Dict) -> Dict:
    """Run a quick performance benchmark for an ML algorithm."""
    try:
        df = dataset_info['data']
        task_type = dataset_info['task_type']
        
        # Prepare data
        X = df.drop('target', axis=1)
        y = df['target']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, 
            stratify=y if task_type == 'classification' else None
        )
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Check compatibility
        if task_type == 'classification':
            if not getattr(algorithm_plugin, '_supports_classification', True):
                raise ValueError(f"Algorithm doesn't support classification")
        else:
            if not getattr(algorithm_plugin, '_supports_regression', True):
                raise ValueError(f"Algorithm doesn't support regression")
        
        # Training
        start_time = time.time()
        
        if hasattr(algorithm_plugin, 'create_model_instance'):
            model = algorithm_plugin.create_model_instance({})
        else:
            model = algorithm_plugin
        
        model.fit(X_train_scaled, y_train)
        training_time = time.time() - start_time
        
        # Prediction
        pred_start = time.time()
        y_pred = model.predict(X_test_scaled)
        prediction_time = time.time() - pred_start
        
        total_time = training_time + prediction_time
        
        # Calculate ML metrics
        ml_metrics = {}
        if task_type == 'classification':
            try:
                from sklearn.metrics import accuracy_score, f1_score
                ml_metrics = {
                    'accuracy': accuracy_score(y_test, y_pred),
                    'f1_score': f1_score(y_test, y_pred, average='weighted', zero_division=0)
                }
            except:
                ml_metrics = {'accuracy': 0.0, 'f1_score': 0.0}
        else:
            try:
                from sklearn.metrics import mean_squared_error, r2_score
                ml_metrics = {
                    'mse': mean_squared_error(y_test, y_pred),
                    'r2_score': r2_score(y_test, y_pred)
                }
            except:
                ml_metrics = {'mse': float('inf'), 'r2_score': 0.0}
        
        return {
            'algorithm': algorithm_name,
            'task_type': task_type,
            'success': True,
            'training_time': training_time,
            'prediction_time': prediction_time,
            'total_time': total_time,
            'samples_processed': len(X_train),
            'training_throughput': len(X_train) / training_time if training_time > 0 else 0,
            'prediction_throughput': len(X_test) / prediction_time if prediction_time > 0 else 0,
            'ml_metrics': ml_metrics,
            'dataset_shape': df.shape
        }
    
    except Exception as e:
        return {
            'algorithm': algorithm_name,
            'task_type': dataset_info.get('task_type', 'unknown'),
            'success': False,
            'error': str(e),
            'training_time': None,
            'prediction_time': None,
            'total_time': None,
            'training_throughput': 0,
            'prediction_throughput': 0,
            'ml_metrics': {}
        }

def run_quick_ml_benchmark():
    """Run quick ML performance benchmark."""
    print("🚀 Quick ML Performance Benchmark")
    print("=" * 40)
    
    # Get user preferences
    print("\nSelect dataset sizes to test:")
    print("1. 🐣 Tiny (500 samples) - Ultra fast")
    print("2. 🚀 Small (1K samples) - Fast")
    print("3. 📊 Medium (10K samples) - Standard")
    print("4. 📈 Large (50K samples) - Comprehensive")
    print("5. 🏭 Huge (100K samples) - Stress test")
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
        log_filename = f"quick_ml_benchmark_{size_str}_{timestamp}.txt"
        
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
        plugin_manager = get_plugin_manager()
        
        # Get algorithms by category
        classification_algorithms = plugin_manager.get_available_plugins("classification")
        regression_algorithms = plugin_manager.get_available_plugins("regression")
        
        print(f"📊 Loaded ML algorithms:")
        print(f"   📈 Classification: {len(classification_algorithms)}")
        print(f"   📉 Regression: {len(regression_algorithms)}")
        
        # Test with selected dataset sizes
        for size in sizes:
            print(f"\n📈 Testing with {size} datasets...")
            datasets = create_ml_test_dataset(size)
            
            for task_type, dataset_info in datasets.items():
                print(f"\n   📊 {task_type.upper()} Task - Dataset shape: {dataset_info['data'].shape}")
                
                # Get compatible algorithms
                algorithms = classification_algorithms if task_type == 'classification' else regression_algorithms
                
                if not algorithms:
                    print(f"   ⚠️  No {task_type} algorithms available")
                    continue
                
                results = []
                
                # Test each algorithm
                for algorithm_name, algorithm_plugin in algorithms.items():
                    print(f"     ⚙️  Testing {algorithm_name}...", end=" ")
                    
                    result = quick_ml_benchmark(algorithm_name, algorithm_plugin, dataset_info)
                    results.append(result)
                    
                    if result['success']:
                        print(f"✅ {result['training_time']:.3f}s training, {result['prediction_time']:.3f}s prediction")
                        
                        # Display ML metrics
                        metrics = result['ml_metrics']
                        if task_type == 'classification':
                            print(f"         Accuracy: {metrics.get('accuracy', 0):.3f}, F1: {metrics.get('f1_score', 0):.3f}")
                        else:
                            print(f"         R²: {metrics.get('r2_score', 0):.3f}, MSE: {metrics.get('mse', float('inf')):.3f}")
                    else:
                        print(f"❌ Failed: {result.get('error', 'Unknown')[:50]}...")
                
                # Show top performers for this task and size
                successful_results = [r for r in results if r['success']]
                if successful_results:
                    print(f"\n     🏆 Top 3 fastest for {task_type} ({size}):")
                    top_performers = sorted(successful_results, key=lambda x: x['training_time'])[:3]
                    
                    for i, result in enumerate(top_performers, 1):
                        ml_metric_str = ""
                        if task_type == 'classification':
                            acc = result['ml_metrics'].get('accuracy', 0)
                            ml_metric_str = f"Acc: {acc:.3f}"
                        else:
                            r2 = result['ml_metrics'].get('r2_score', 0)
                            ml_metric_str = f"R²: {r2:.3f}"
                        
                        print(f"        {i}. {result['algorithm']}: {result['training_time']:.3f}s ({ml_metric_str})")
                
                print(f"\n     📊 Success rate: {len(successful_results)}/{len(results)} algorithms")
    
    finally:
        # Restore stdout and close log file if logging was enabled
        if terminal_logger:
            sys.stdout = terminal_logger.terminal
            terminal_logger.close()
            print(f"📄 Terminal output saved to: {log_filename}")

if __name__ == "__main__":
    run_quick_ml_benchmark()
