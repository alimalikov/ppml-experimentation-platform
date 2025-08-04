"""
ML Platform Performance Testing Suite
====================================

This module provides comprehensive performance testing for the ML experimentation platform,
including algorithm performance benchmarking, scalability analysis, and detailed reporting.

Author: Bachelor Thesis Project
Date: July 2025
"""

import os
import sys
import time
import psutil
import pandas as pd
import numpy as np
import json
import tracemalloc
from datetime import datetime
from typing import Dict, List, Tuple, Any, Optional
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Add project root to path for imports
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Add src directory
SRC_DIR = os.path.join(PROJECT_ROOT, 'src')
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

# Import the ML plugin system
try:
    from src.ml_plugins.plugin_manager import get_plugin_manager
    from src.ml_plugins.metric_manager import get_metric_manager
    from sklearn.model_selection import train_test_split
    from sklearn.datasets import make_classification, make_regression
    from sklearn.preprocessing import StandardScaler
except ImportError as e:
    print(f"Error importing ML modules: {e}")
    print("Make sure you're running this from the correct directory structure")
    sys.exit(1)

class MLPerformanceTester:
    """
    Comprehensive performance testing framework for ML algorithms.
    """
    
    def __init__(self, output_dir: str = "ml_performance_results"):
        """
        Initialize the ML performance tester.
        
        Args:
            output_dir: Directory to save performance results
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Initialize results storage
        self.results = {
            'test_metadata': {
                'timestamp': datetime.now().isoformat(),
                'system_info': self._get_system_info()
            },
            'dataset_performance': {},
            'algorithm_performance': {},
            'scalability_tests': {},
            'training_metrics': {},
            'comparative_analysis': {}
        }
        
        # Load ML plugins
        try:
            self.plugin_manager = get_plugin_manager()
            self.metric_manager = get_metric_manager()
            
            # Get available algorithms by task type (this returns Dict[str, MLPlugin])
            self.classification_algorithms = self.plugin_manager.get_available_plugins("classification")
            self.regression_algorithms = self.plugin_manager.get_available_plugins("regression")
            
            # Combine all algorithms
            self.all_algorithms = {}
            self.all_algorithms.update(self.classification_algorithms)
            self.all_algorithms.update(self.regression_algorithms)
            
            print(f"🔧 Loaded {len(self.all_algorithms)} ML algorithms for testing")
            print(f"   📊 Classification: {len(self.classification_algorithms)}")
            print(f"   📈 Regression: {len(self.regression_algorithms)}")
            
        except Exception as e:
            print(f"⚠️  Error loading ML plugins: {e}")
            self.all_algorithms = {}
        
        print(f"📁 Results will be saved to: {self.output_dir.absolute()}")
    
    def _get_system_info(self) -> Dict[str, Any]:
        """Get system information for performance context."""
        return {
            'cpu_count': psutil.cpu_count(),
            'cpu_freq': psutil.cpu_freq().current if psutil.cpu_freq() else 'Unknown',
            'memory_total_gb': round(psutil.virtual_memory().total / (1024**3), 2),
            'python_version': sys.version,
            'platform': sys.platform
        }
    
    def generate_synthetic_datasets(self, selected_sizes: List[str] = None) -> Dict[str, Dict[str, Any]]:
        """
        Generate synthetic datasets with varying characteristics for ML testing.
        
        Args:
            selected_sizes: List of dataset sizes to generate ('tiny', 'small', 'medium', 'large', 'huge')
        """
        if selected_sizes is None:
            selected_sizes = ['small', 'medium', 'large']
        
        print(f"📊 Generating synthetic ML datasets for sizes: {', '.join(selected_sizes)}...")
        
        datasets = {}
        
        # Define size configurations
        size_configs = {
            'tiny': {'n_samples': 500, 'n_features': 10},
            'small': {'n_samples': 1000, 'n_features': 15},
            'medium': {'n_samples': 10000, 'n_features': 20},
            'large': {'n_samples': 50000, 'n_features': 30},
            'huge': {'n_samples': 100000, 'n_features': 40}
        }
        
        # Generate datasets for selected sizes
        for size in selected_sizes:
            if size in size_configs:
                config = size_configs[size]
                
                # Generate classification dataset
                X_class, y_class = make_classification(
                    n_samples=config['n_samples'],
                    n_features=config['n_features'],
                    n_informative=max(5, config['n_features'] // 2),
                    n_redundant=max(2, config['n_features'] // 4),
                    n_clusters_per_class=1,
                    n_classes=3,
                    random_state=42
                )
                
                # Generate regression dataset
                X_reg, y_reg = make_regression(
                    n_samples=config['n_samples'],
                    n_features=config['n_features'],
                    n_informative=max(5, config['n_features'] // 2),
                    noise=0.1,
                    random_state=42
                )
                
                # Create DataFrames
                feature_names = [f'feature_{i}' for i in range(config['n_features'])]
                
                df_class = pd.DataFrame(X_class, columns=feature_names)
                df_class['target'] = y_class
                
                df_reg = pd.DataFrame(X_reg, columns=feature_names)
                df_reg['target'] = y_reg
                
                datasets[f'{size}_classification_{config["n_samples"]}'] = {
                    'data': df_class,
                    'task_type': 'classification',
                    'n_samples': config['n_samples'],
                    'n_features': config['n_features']
                }
                
                datasets[f'{size}_regression_{config["n_samples"]}'] = {
                    'data': df_reg,
                    'task_type': 'regression',
                    'n_samples': config['n_samples'],
                    'n_features': config['n_features']
                }
                
                print(f"   ✓ {size}: {config['n_samples']} samples, {config['n_features']} features (classification + regression)")
        
        print(f"✅ Generated {len(datasets)} ML test datasets")
        return datasets
    
    def run_ml_performance_test(self, algorithm_name: str, algorithm_plugin, 
                                dataset_info: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run performance test for a specific ML algorithm on a dataset.
        """
        # Start memory tracking
        tracemalloc.start()
        process = psutil.Process()
        
        # Get initial memory
        mem_before = process.memory_info().rss / 1024 / 1024  # MB
        
        try:
            df = dataset_info['data']
            task_type = dataset_info['task_type']
            
            # Prepare data
            X = df.drop('target', axis=1)
            y = df['target']
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y if task_type == 'classification' else None
            )
            
            # Scale features if needed
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # Check algorithm compatibility
            if task_type == 'classification':
                if not getattr(algorithm_plugin, '_supports_classification', True):
                    raise ValueError(f"Algorithm {algorithm_name} doesn't support classification")
            else:
                if not getattr(algorithm_plugin, '_supports_regression', True):
                    raise ValueError(f"Algorithm {algorithm_name} doesn't support regression")
            
            # Start timing
            start_time = time.time()
            start_cpu = time.process_time()
            
            # Training phase
            training_start = time.time()
            
            # Create model instance
            if hasattr(algorithm_plugin, 'create_model_instance'):
                model = algorithm_plugin.create_model_instance({})
            else:
                model = algorithm_plugin
            
            # Train model
            model.fit(X_train_scaled, y_train)
            training_time = time.time() - training_start
            
            # Prediction phase
            prediction_start = time.time()
            y_pred = model.predict(X_test_scaled)
            prediction_time = time.time() - prediction_start
            
            # End timing
            end_time = time.time()
            end_cpu = time.process_time()
            
            # Get final memory
            current, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            mem_after = process.memory_info().rss / 1024 / 1024  # MB
            
            # Calculate metrics
            wall_time = end_time - start_time
            cpu_time = end_cpu - start_cpu
            memory_used = mem_after - mem_before
            peak_memory = peak / 1024 / 1024  # MB
            
            # Calculate ML metrics
            ml_metrics = {}
            if task_type == 'classification':
                try:
                    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
                    ml_metrics = {
                        'accuracy': accuracy_score(y_test, y_pred),
                        'precision': precision_score(y_test, y_pred, average='weighted', zero_division=0),
                        'recall': recall_score(y_test, y_pred, average='weighted', zero_division=0),
                        'f1_score': f1_score(y_test, y_pred, average='weighted', zero_division=0)
                    }
                except Exception as e:
                    ml_metrics = {'error': str(e)}
            else:
                try:
                    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
                    ml_metrics = {
                        'mse': mean_squared_error(y_test, y_pred),
                        'mae': mean_absolute_error(y_test, y_pred),
                        'r2_score': r2_score(y_test, y_pred)
                    }
                except Exception as e:
                    ml_metrics = {'error': str(e)}
            
            # Performance metrics
            samples_per_second = len(X_train) / training_time if training_time > 0 else 0
            predictions_per_second = len(X_test) / prediction_time if prediction_time > 0 else 0
            
            return {
                'success': True,
                'wall_time_seconds': wall_time,
                'cpu_time_seconds': cpu_time,
                'training_time_seconds': training_time,
                'prediction_time_seconds': prediction_time,
                'memory_used_mb': memory_used,
                'peak_memory_mb': peak_memory,
                'samples_processed': len(X_train),
                'features_processed': X_train.shape[1],
                'training_throughput_samples_per_sec': samples_per_second,
                'prediction_throughput_samples_per_sec': predictions_per_second,
                'ml_metrics': ml_metrics,
                'task_type': task_type,
                'data_shape': df.shape
            }
            
        except Exception as e:
            tracemalloc.stop()
            return {
                'success': False,
                'error': str(e),
                'error_type': type(e).__name__,
                'wall_time_seconds': None,
                'cpu_time_seconds': None,
                'training_time_seconds': None,
                'prediction_time_seconds': None,
                'memory_used_mb': None,
                'peak_memory_mb': None,
                'training_throughput_samples_per_sec': 0,
                'prediction_throughput_samples_per_sec': 0,
                'ml_metrics': {},
                'task_type': dataset_info['task_type']
            }
    
    def run_comprehensive_tests(self, selected_sizes: List[str] = None) -> None:
        """
        Run comprehensive performance tests across all algorithms and datasets.
        
        Args:
            selected_sizes: List of dataset sizes to test ('tiny', 'small', 'medium', 'large', 'huge')
        """
        print("🚀 Starting comprehensive ML performance testing...")
        
        # Generate test datasets
        datasets = self.generate_synthetic_datasets(selected_sizes)
        
        # Test each algorithm on each compatible dataset
        total_tests = 0
        for dataset_name, dataset_info in datasets.items():
            task_type = dataset_info['task_type']
            compatible_algorithms = self.classification_algorithms if task_type == 'classification' else self.regression_algorithms
            # Count only non-polynomial algorithms
            non_poly_algorithms = {name: plugin for name, plugin in compatible_algorithms.items() 
                                 if not ('polynomial' in name.lower() or 'poly' in name.lower())}
            total_tests += len(non_poly_algorithms)
        
        current_test = 0
        
        for dataset_name, dataset_info in datasets.items():
            task_type = dataset_info['task_type']
            df = dataset_info['data']
            
            print(f"\n📊 Testing dataset: {dataset_name} (Shape: {df.shape}, Task: {task_type})")
            
            self.results['dataset_performance'][dataset_name] = {
                'shape': df.shape,
                'task_type': task_type,
                'n_samples': dataset_info['n_samples'],
                'n_features': dataset_info['n_features'],
                'memory_size_mb': df.memory_usage(deep=True).sum() / 1024 / 1024,
                'algorithm_results': {}
            }
            
            # Get compatible algorithms
            compatible_algorithms = self.classification_algorithms if task_type == 'classification' else self.regression_algorithms
            
            for algorithm_name, algorithm_plugin in compatible_algorithms.items():
                # Skip polynomial regressor as it can be problematic with large datasets
                if 'polynomial' in algorithm_name.lower() or 'poly' in algorithm_name.lower():
                    print(f"  ⏭️  Skipping {algorithm_name} (polynomial regressor excluded)")
                    continue
                
                current_test += 1
                progress = (current_test / total_tests) * 100
                
                print(f"  ⚙️  [{progress:.1f}%] Testing {algorithm_name}...")
                
                # Run the performance test
                result = self.run_ml_performance_test(algorithm_name, algorithm_plugin, dataset_info)
                
                # Store results
                self.results['dataset_performance'][dataset_name]['algorithm_results'][algorithm_name] = result
                
                if not result['success']:
                    print(f"    ❌ Failed: {result.get('error', 'Unknown error')}")
                else:
                    print(f"    ✅ Success: {result['training_time_seconds']:.3f}s training, "
                          f"{result['training_throughput_samples_per_sec']:.0f} samples/sec")
        
        print("\n✅ Comprehensive ML testing completed!")
    
    def run_scalability_tests(self) -> None:
        """
        Run scalability tests to see how algorithms perform with increasing data sizes.
        """
        print("\n🔬 Running ML scalability analysis...")
        
        # Test with increasing dataset sizes
        test_sizes = [1000, 5000, 10000, 25000, 50000]
        
        # Select a few representative algorithms for scalability testing
        test_algorithms = {}
        if self.classification_algorithms:
            # Get first 3 classification algorithms (excluding polynomial)
            non_poly_class = {name: plugin for name, plugin in self.classification_algorithms.items() 
                            if not ('polynomial' in name.lower() or 'poly' in name.lower())}
            test_algorithms.update(dict(list(non_poly_class.items())[:3]))
        if self.regression_algorithms:
            # Get first 3 regression algorithms (excluding polynomial)
            non_poly_reg = {name: plugin for name, plugin in self.regression_algorithms.items() 
                          if not ('polynomial' in name.lower() or 'poly' in name.lower())}
            test_algorithms.update(dict(list(non_poly_reg.items())[:3]))
        
        for algorithm_name, algorithm_plugin in test_algorithms.items():
            # Skip polynomial regressor as it can be problematic with large datasets
            if 'polynomial' in algorithm_name.lower() or 'poly' in algorithm_name.lower():
                print(f"  ⏭️  Skipping {algorithm_name} scalability test (polynomial regressor excluded)")
                continue
                
            print(f"  📈 Testing scalability for: {algorithm_name}")
            
            self.results['scalability_tests'][algorithm_name] = []
            
            # Determine if this is classification or regression algorithm
            is_classification = algorithm_name in self.classification_algorithms
            task_type = 'classification' if is_classification else 'regression'
            
            for size in test_sizes:
                try:
                    # Generate dataset of specific size
                    if is_classification:
                        X, y = make_classification(
                            n_samples=size, n_features=20, n_informative=10,
                            n_redundant=5, n_classes=3, random_state=42
                        )
                    else:
                        X, y = make_regression(
                            n_samples=size, n_features=20, n_informative=10,
                            noise=0.1, random_state=42
                        )
                    
                    # Create dataset info
                    feature_names = [f'feature_{i}' for i in range(20)]
                    df = pd.DataFrame(X, columns=feature_names)
                    df['target'] = y
                    
                    dataset_info = {
                        'data': df,
                        'task_type': task_type,
                        'n_samples': size,
                        'n_features': 20
                    }
                    
                    # Run test
                    result = self.run_ml_performance_test(algorithm_name, algorithm_plugin, dataset_info)
                    
                    if result['success']:
                        self.results['scalability_tests'][algorithm_name].append({
                            'dataset_size': size,
                            'training_time': result['training_time_seconds'],
                            'prediction_time': result['prediction_time_seconds'],
                            'memory_used': result['memory_used_mb'],
                            'training_throughput': result['training_throughput_samples_per_sec'],
                            'prediction_throughput': result['prediction_throughput_samples_per_sec']
                        })
                        
                        print(f"    ✓ Size {size}: {result['training_time_seconds']:.3f}s training")
                    else:
                        print(f"    ✗ Size {size}: Failed - {result.get('error', 'Unknown')}")
                        
                except Exception as e:
                    print(f"    ✗ Size {size}: Error - {str(e)}")
        
        print("✅ Scalability testing completed!")
    
    def generate_performance_report(self) -> None:
        """Generate comprehensive performance report with JSON and readable formats."""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Create summary statistics
        self._create_summary_statistics()
        
        # Save raw results as JSON
        json_file = self.output_dir / f"ml_performance_results_{timestamp}.json"
        with open(json_file, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        print(f"📄 Raw results saved to: {json_file}")
        
        # Create visualizations
        self._create_ml_performance_visualizations()
        
        # Generate readable report
        self._generate_readable_report()
    
    def _create_summary_statistics(self) -> None:
        """Create summary statistics for the test results."""
        summary = {
            'total_algorithms_tested': 0,
            'total_datasets_tested': len(self.results['dataset_performance']),
            'successful_tests': 0,
            'failed_tests': 0,
            'classification_tests': 0,
            'regression_tests': 0,
            'fastest_training': None,
            'fastest_prediction': None,
            'most_memory_efficient': None,
            'highest_training_throughput': None,
            'highest_prediction_throughput': None
        }
        
        fastest_training = None
        fastest_prediction = None
        most_memory_efficient = None
        highest_training_throughput = None
        highest_prediction_throughput = None
        
        for dataset_name, dataset_results in self.results['dataset_performance'].items():
            task_type = dataset_results['task_type']
            if task_type == 'classification':
                summary['classification_tests'] += len(dataset_results['algorithm_results'])
            else:
                summary['regression_tests'] += len(dataset_results['algorithm_results'])
            
            for algorithm_name, result in dataset_results['algorithm_results'].items():
                summary['total_algorithms_tested'] += 1
                
                if result['success']:
                    summary['successful_tests'] += 1
                    
                    # Track fastest training
                    if (fastest_training is None or 
                        result['training_time_seconds'] < fastest_training['time']):
                        fastest_training = {
                            'algorithm': algorithm_name,
                            'dataset': dataset_name,
                            'time': result['training_time_seconds']
                        }
                    
                    # Track fastest prediction
                    if (fastest_prediction is None or 
                        result['prediction_time_seconds'] < fastest_prediction['time']):
                        fastest_prediction = {
                            'algorithm': algorithm_name,
                            'dataset': dataset_name,
                            'time': result['prediction_time_seconds']
                        }
                    
                    # Track most memory efficient
                    if (most_memory_efficient is None or 
                        result['memory_used_mb'] < most_memory_efficient['memory']):
                        most_memory_efficient = {
                            'algorithm': algorithm_name,
                            'dataset': dataset_name,
                            'memory': result['memory_used_mb']
                        }
                    
                    # Track highest training throughput
                    if (highest_training_throughput is None or 
                        result['training_throughput_samples_per_sec'] > highest_training_throughput['throughput']):
                        highest_training_throughput = {
                            'algorithm': algorithm_name,
                            'dataset': dataset_name,
                            'throughput': result['training_throughput_samples_per_sec']
                        }
                    
                    # Track highest prediction throughput
                    if (highest_prediction_throughput is None or 
                        result['prediction_throughput_samples_per_sec'] > highest_prediction_throughput['throughput']):
                        highest_prediction_throughput = {
                            'algorithm': algorithm_name,
                            'dataset': dataset_name,
                            'throughput': result['prediction_throughput_samples_per_sec']
                        }
                        
                else:
                    summary['failed_tests'] += 1
        
        # Store summary results
        if fastest_training:
            summary['fastest_training'] = {
                'algorithm': fastest_training['algorithm'],
                'dataset': fastest_training['dataset'],
                'time': fastest_training['time']
            }
        
        if fastest_prediction:
            summary['fastest_prediction'] = {
                'algorithm': fastest_prediction['algorithm'],
                'dataset': fastest_prediction['dataset'],
                'time': fastest_prediction['time']
            }
        
        if most_memory_efficient:
            summary['most_memory_efficient'] = {
                'algorithm': most_memory_efficient['algorithm'],
                'dataset': most_memory_efficient['dataset'],
                'memory': most_memory_efficient['memory']
            }
        
        if highest_training_throughput:
            summary['highest_training_throughput'] = {
                'algorithm': highest_training_throughput['algorithm'],
                'dataset': highest_training_throughput['dataset'],
                'throughput': highest_training_throughput['throughput']
            }
        
        if highest_prediction_throughput:
            summary['highest_prediction_throughput'] = {
                'algorithm': highest_prediction_throughput['algorithm'],
                'dataset': highest_prediction_throughput['dataset'],
                'throughput': highest_prediction_throughput['throughput']
            }
        
        self.results['summary'] = summary
    
    def _create_ml_performance_visualizations(self) -> None:
        """Create separate ML performance visualization charts."""
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
            
            # Set style
            plt.style.use('seaborn-v0_8')
            sns.set_palette("husl")
            
            # Collect data for visualization
            performance_data = []
            
            for dataset_name, dataset_results in self.results['dataset_performance'].items():
                for algorithm_name, result in dataset_results['algorithm_results'].items():
                    if result['success']:
                        performance_data.append({
                            'Dataset': dataset_name,
                            'Algorithm': algorithm_name,
                            'Task Type': result['task_type'],
                            'Training Time (s)': result['training_time_seconds'],
                            'Prediction Time (s)': result['prediction_time_seconds'],
                            'CPU Time (s)': result['cpu_time_seconds'],
                            'Memory Used (MB)': result['memory_used_mb'],
                            'Training Throughput (samples/s)': result['training_throughput_samples_per_sec'],
                            'Prediction Throughput (samples/s)': result['prediction_throughput_samples_per_sec'],
                            'Samples': result['samples_processed']
                        })
            
            if not performance_data:
                print("⚠️  No successful results to visualize")
                return
            
            df_viz = pd.DataFrame(performance_data)
            print(f"📊 Creating 6 separate ML performance visualization charts...")
            
            # 1. Training Time Distribution by Algorithm
            plt.figure(figsize=(14, 8))
            sns.boxplot(data=df_viz, x='Algorithm', y='Training Time (s)', hue='Task Type')
            plt.xticks(rotation=45, ha='right')
            plt.title('ML Training Time Distribution by Algorithm', fontsize=14, fontweight='bold')
            plt.xlabel('ML Algorithm', fontsize=12)
            plt.ylabel('Training Time (seconds)', fontsize=12)
            plt.legend(title='Task Type')
            plt.tight_layout()
            
            # Save chart 1
            chart1_file = self.output_dir / 'ml_training_time_distribution_by_algorithm.png'
            plt.savefig(chart1_file, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"   ✅ Saved: {chart1_file.name}")
            
            # 2. Memory Usage vs Dataset Size
            plt.figure(figsize=(12, 8))
            sns.scatterplot(data=df_viz, x='Samples', y='Memory Used (MB)', 
                           hue='Algorithm', style='Task Type', s=100, alpha=0.7)
            plt.title('Memory Usage vs Dataset Size (ML Algorithms)', fontsize=14, fontweight='bold')
            plt.xlabel('Number of Training Samples', fontsize=12)
            plt.ylabel('Memory Used (MB)', fontsize=12)
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.tight_layout()
            
            # Save chart 2
            chart2_file = self.output_dir / 'ml_memory_usage_vs_dataset_size.png'
            plt.savefig(chart2_file, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"   ✅ Saved: {chart2_file.name}")
            
            # 3. Training Throughput Comparison
            plt.figure(figsize=(12, 8))
            avg_training_throughput = df_viz.groupby('Algorithm')['Training Throughput (samples/s)'].mean().sort_values(ascending=True)
            colors = plt.cm.viridis(range(len(avg_training_throughput)))
            bars = avg_training_throughput.plot(kind='barh', color=colors, figsize=(12, 8))
            plt.title('Average Training Throughput by ML Algorithm', fontsize=14, fontweight='bold')
            plt.xlabel('Training Throughput (samples per second)', fontsize=12)
            plt.ylabel('ML Algorithm', fontsize=12)
            
            # Add value labels on bars
            for i, v in enumerate(avg_training_throughput.values):
                plt.text(v + max(avg_training_throughput.values) * 0.01, i, f'{v:.0f}', 
                        va='center', fontsize=10)
            
            plt.tight_layout()
            
            # Save chart 3
            chart3_file = self.output_dir / 'ml_average_training_throughput_by_algorithm.png'
            plt.savefig(chart3_file, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"   ✅ Saved: {chart3_file.name}")
            
            # 4. Training vs Prediction Time Heatmap
            plt.figure(figsize=(14, 8))
            pivot_data = df_viz.pivot_table(values='Training Time (s)', index='Dataset', columns='Algorithm', aggfunc='mean')
            sns.heatmap(pivot_data, annot=True, fmt='.3f', cmap='YlOrRd', 
                       cbar_kws={'label': 'Training Time (seconds)'})
            plt.title('ML Training Time Heatmap (seconds)', fontsize=14, fontweight='bold')
            plt.xlabel('ML Algorithm', fontsize=12)
            plt.ylabel('Dataset', fontsize=12)
            plt.xticks(rotation=45, ha='right')
            plt.yticks(rotation=0)
            plt.tight_layout()
            
            # Save chart 4
            chart4_file = self.output_dir / 'ml_training_time_heatmap_seconds.png'
            plt.savefig(chart4_file, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"   ✅ Saved: {chart4_file.name}")
            
            # 5. CPU Usage Analysis for ML
            plt.figure(figsize=(12, 10))
            
            # Create subplot for CPU vs Training Time comparison
            plt.subplot(2, 1, 1)
            algorithms = df_viz['Algorithm'].unique()
            x_pos = range(len(algorithms))
            
            avg_training_time = [df_viz[df_viz['Algorithm'] == alg]['Training Time (s)'].mean() for alg in algorithms]
            avg_cpu_time = [df_viz[df_viz['Algorithm'] == alg]['CPU Time (s)'].mean() for alg in algorithms]
            
            width = 0.35
            plt.bar([x - width/2 for x in x_pos], avg_training_time, width, label='Training Time', alpha=0.8, color='skyblue')
            plt.bar([x + width/2 for x in x_pos], avg_cpu_time, width, label='CPU Time', alpha=0.8, color='lightcoral')
            
            plt.xlabel('ML Algorithm')
            plt.ylabel('Time (seconds)')
            plt.title('CPU vs Training Time Comparison by ML Algorithm', fontweight='bold')
            plt.xticks(x_pos, algorithms, rotation=45, ha='right')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            # Create subplot for Training vs Prediction efficiency
            plt.subplot(2, 1, 2)
            avg_prediction_time = [df_viz[df_viz['Algorithm'] == alg]['Prediction Time (s)'].mean() for alg in algorithms]
            
            training_pred_ratio = []
            for i, alg in enumerate(algorithms):
                ratio = avg_prediction_time[i] / avg_training_time[i] * 100 if avg_training_time[i] > 0 else 0
                training_pred_ratio.append(ratio)
            
            colors = ['green' if ratio < 10 else 'orange' if ratio < 50 else 'red' for ratio in training_pred_ratio]
            bars = plt.bar(algorithms, training_pred_ratio, color=colors, alpha=0.7)
            plt.xlabel('ML Algorithm')
            plt.ylabel('Prediction/Training Time Ratio (%)')
            plt.title('Prediction Efficiency by ML Algorithm (Prediction Time / Training Time * 100)', fontweight='bold')
            plt.xticks(rotation=45, ha='right')
            plt.grid(True, alpha=0.3)
            
            # Add percentage labels on bars
            for bar, ratio in zip(bars, training_pred_ratio):
                plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(training_pred_ratio) * 0.01, 
                        f'{ratio:.1f}%', ha='center', va='bottom', fontsize=9)
            
            plt.tight_layout()
            
            # Save chart 5
            chart5_file = self.output_dir / 'ml_cpu_usage_and_efficiency_analysis.png'
            plt.savefig(chart5_file, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"   ✅ Saved: {chart5_file.name}")
            
            # 6. Task-Specific Performance Comparison
            plt.figure(figsize=(14, 10))
            
            # Split by task type
            classification_data = df_viz[df_viz['Task Type'] == 'classification']
            regression_data = df_viz[df_viz['Task Type'] == 'regression']
            
            plt.subplot(2, 1, 1)
            if not classification_data.empty:
                class_throughput = classification_data.groupby('Algorithm')['Training Throughput (samples/s)'].mean().sort_values(ascending=False)
                class_throughput.plot(kind='bar', color='lightblue', alpha=0.8)
                plt.title('Classification Algorithms - Training Throughput', fontweight='bold')
                plt.ylabel('Samples per Second')
                plt.xticks(rotation=45, ha='right')
                plt.grid(True, alpha=0.3)
            
            plt.subplot(2, 1, 2)
            if not regression_data.empty:
                reg_throughput = regression_data.groupby('Algorithm')['Training Throughput (samples/s)'].mean().sort_values(ascending=False)
                reg_throughput.plot(kind='bar', color='lightgreen', alpha=0.8)
                plt.title('Regression Algorithms - Training Throughput', fontweight='bold')
                plt.ylabel('Samples per Second')
                plt.xticks(rotation=45, ha='right')
                plt.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            # Save chart 6
            chart6_file = self.output_dir / 'ml_task_specific_performance_comparison.png'
            plt.savefig(chart6_file, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"   ✅ Saved: {chart6_file.name}")
            
            # Create scalability chart if data exists
            if self.results['scalability_tests']:
                self._create_ml_scalability_chart()
                print(f"📈 All 7 ML visualization charts saved to: {self.output_dir}")
                print("   📊 Charts created:")
                print("   1. ml_training_time_distribution_by_algorithm.png")
                print("   2. ml_memory_usage_vs_dataset_size.png") 
                print("   3. ml_average_training_throughput_by_algorithm.png")
                print("   4. ml_training_time_heatmap_seconds.png")
                print("   5. ml_cpu_usage_and_efficiency_analysis.png")
                print("   6. ml_task_specific_performance_comparison.png")
                print("   7. ml_scalability_analysis_by_dataset_size.png")
            else:
                print(f"📈 All 6 ML visualization charts saved to: {self.output_dir}")
                print("   📊 Charts created:")
                print("   1. ml_training_time_distribution_by_algorithm.png")
                print("   2. ml_memory_usage_vs_dataset_size.png") 
                print("   3. ml_average_training_throughput_by_algorithm.png")
                print("   4. ml_training_time_heatmap_seconds.png")
                print("   5. ml_cpu_usage_and_efficiency_analysis.png")
                print("   6. ml_task_specific_performance_comparison.png")
            
        except ImportError:
            print("⚠️  Matplotlib/Seaborn not available for visualizations")
        except Exception as e:
            print(f"⚠️  Error creating ML visualizations: {e}")
    
    def _create_ml_scalability_chart(self) -> None:
        """Create ML scalability analysis chart."""
        try:
            import matplotlib.pyplot as plt
            
            fig, axes = plt.subplots(2, 2, figsize=(16, 12))
            fig.suptitle('ML Algorithm Scalability Analysis', fontsize=16, fontweight='bold')
            
            for algorithm_name, results in self.results['scalability_tests'].items():
                if not results:
                    continue
                
                sizes = [r['dataset_size'] for r in results]
                training_times = [r['training_time'] for r in results]
                prediction_times = [r['prediction_time'] for r in results]
                memory = [r['memory_used'] for r in results]
                training_throughput = [r['training_throughput'] for r in results]
                
                # Training time scaling
                axes[0, 0].plot(sizes, training_times, marker='o', label=algorithm_name, linewidth=2)
                
                # Prediction time scaling
                axes[0, 1].plot(sizes, prediction_times, marker='s', label=algorithm_name, linewidth=2)
                
                # Memory scaling
                axes[1, 0].plot(sizes, memory, marker='^', label=algorithm_name, linewidth=2)
                
                # Training throughput scaling
                axes[1, 1].plot(sizes, training_throughput, marker='d', label=algorithm_name, linewidth=2)
            
            axes[0, 0].set_title('Training Time vs Dataset Size', fontweight='bold')
            axes[0, 0].set_xlabel('Dataset Size (samples)')
            axes[0, 0].set_ylabel('Training Time (seconds)')
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)
            
            axes[0, 1].set_title('Prediction Time vs Dataset Size', fontweight='bold')
            axes[0, 1].set_xlabel('Dataset Size (samples)')
            axes[0, 1].set_ylabel('Prediction Time (seconds)')
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)
            
            axes[1, 0].set_title('Memory Usage vs Dataset Size', fontweight='bold')
            axes[1, 0].set_xlabel('Dataset Size (samples)')
            axes[1, 0].set_ylabel('Memory Used (MB)')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)
            
            axes[1, 1].set_title('Training Throughput vs Dataset Size', fontweight='bold')
            axes[1, 1].set_xlabel('Dataset Size (samples)')
            axes[1, 1].set_ylabel('Training Throughput (samples/sec)')
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            scalability_file = self.output_dir / 'ml_scalability_analysis_by_dataset_size.png'
            plt.savefig(scalability_file, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"   ✅ Saved: {scalability_file.name}")
            
        except Exception as e:
            print(f"⚠️  Error creating ML scalability chart: {e}")
    
    def _generate_readable_report(self) -> None:
        """Generate a human-readable ML performance report."""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        report_file = self.output_dir / f"ml_performance_report_{timestamp}.md"
        
        with open(report_file, 'w') as f:
            f.write("# ML Platform Performance Report\n\n")
            f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # System Information
            f.write("## System Information\n\n")
            system_info = self.results['test_metadata']['system_info']
            f.write(f"- **CPU Count:** {system_info['cpu_count']}\n")
            f.write(f"- **CPU Frequency:** {system_info['cpu_freq']} MHz\n")
            f.write(f"- **Total Memory:** {system_info['memory_total_gb']} GB\n")
            f.write(f"- **Python Version:** {system_info['python_version']}\n")
            f.write(f"- **Platform:** {system_info['platform']}\n\n")
            
            # Test Summary
            f.write("## Test Summary\n\n")
            summary = self.results.get('summary', {})
            f.write(f"- **Total Algorithm Tests:** {summary.get('total_algorithms_tested', 0)}\n")
            f.write(f"- **Total Datasets:** {summary.get('total_datasets_tested', 0)}\n")
            f.write(f"- **Successful Tests:** {summary.get('successful_tests', 0)}\n")
            f.write(f"- **Failed Tests:** {summary.get('failed_tests', 0)}\n")
            f.write(f"- **Classification Tests:** {summary.get('classification_tests', 0)}\n")
            f.write(f"- **Regression Tests:** {summary.get('regression_tests', 0)}\n\n")
            
            # Performance Highlights
            f.write("## Performance Highlights\n\n")
            if summary.get('fastest_training'):
                fastest = summary['fastest_training']
                f.write(f"- **Fastest Training:** {fastest['algorithm']} ")
                f.write(f"({fastest['time']:.3f}s on {fastest['dataset']})\n")
            
            if summary.get('fastest_prediction'):
                fastest = summary['fastest_prediction']
                f.write(f"- **Fastest Prediction:** {fastest['algorithm']} ")
                f.write(f"({fastest['time']:.3f}s on {fastest['dataset']})\n")
            
            if summary.get('most_memory_efficient'):
                efficient = summary['most_memory_efficient']
                f.write(f"- **Most Memory Efficient:** {efficient['algorithm']} ")
                f.write(f"({efficient['memory']:.2f} MB on {efficient['dataset']})\n")
            
            if summary.get('highest_training_throughput'):
                throughput = summary['highest_training_throughput']
                f.write(f"- **Highest Training Throughput:** {throughput['algorithm']} ")
                f.write(f"({throughput['throughput']:.0f} samples/sec on {throughput['dataset']})\n")
            
            if summary.get('highest_prediction_throughput'):
                throughput = summary['highest_prediction_throughput']
                f.write(f"- **Highest Prediction Throughput:** {throughput['algorithm']} ")
                f.write(f"({throughput['throughput']:.0f} samples/sec on {throughput['dataset']})\n\n")
            
            # Dataset results
            f.write("## Dataset Performance Results\n\n")
            for dataset_name, dataset_results in self.results['dataset_performance'].items():
                f.write(f"### {dataset_name}\n\n")
                f.write(f"- **Shape:** {dataset_results['shape']}\n")
                f.write(f"- **Task Type:** {dataset_results['task_type']}\n")
                f.write(f"- **Samples:** {dataset_results['n_samples']}\n")
                f.write(f"- **Features:** {dataset_results['n_features']}\n")
                f.write(f"- **Memory Size:** {dataset_results['memory_size_mb']:.2f} MB\n\n")
                
                f.write("| Algorithm | Status | Training Time (s) | Prediction Time (s) | Memory (MB) | Training Throughput (samples/s) |\n")
                f.write("|-----------|--------|-------------------|---------------------|-------------|--------------------------------|\n")
                
                for algorithm_name, result in dataset_results['algorithm_results'].items():
                    if result['success']:
                        f.write(f"| {algorithm_name} | ✅ | {result['training_time_seconds']:.3f} | ")
                        f.write(f"{result['prediction_time_seconds']:.3f} | {result['memory_used_mb']:.1f} | ")
                        f.write(f"{result['training_throughput_samples_per_sec']:.0f} |\n")
                    else:
                        f.write(f"| {algorithm_name} | ❌ | - | - | - | - |\n")
                
                f.write("\n")
            
            # Recommendations
            f.write("## ML Algorithm Recommendations\n\n")
            f.write("Based on the test results:\n\n")
            f.write("1. **For Small Datasets (< 5K samples):** Most algorithms should perform well\n")
            f.write("2. **For Large Datasets (> 50K samples):** Consider algorithms with high training throughput\n")
            f.write("3. **For Real-time Prediction:** Focus on algorithms with fastest prediction times\n")
            f.write("4. **For Memory-constrained Environments:** Use algorithms with lowest memory usage\n")
            f.write("5. **For Classification Tasks:** Compare classification-specific performance metrics\n")
            f.write("6. **For Regression Tasks:** Compare regression-specific performance metrics\n\n")
            
            f.write("## Notes\n\n")
            f.write("- Performance may vary based on data characteristics and system configuration\n")
            f.write("- Results are based on synthetic datasets for consistent comparison\n")
            f.write("- Failed tests may indicate missing dependencies or configuration issues\n")
            f.write("- Training and prediction times include data preprocessing overhead\n")
        
        print(f"📋 ML performance report saved to: {report_file}")


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
    """Get user preferences for dataset sizes and terminal output logging."""
    print("🔬 ML Platform Comprehensive Performance Testing")
    print("=" * 55)
    
    # Get dataset sizes
    print("\nSelect dataset sizes to test:")
    print("1. 🐣 Tiny (500 samples) - Ultra fast")
    print("2. 🚀 Small (1K samples) - Fast")
    print("3. 📊 Medium (10K samples) - Standard")
    print("4. 📈 Large (50K samples) - Comprehensive")
    print("5. 🏭 Huge (100K samples) - Stress test")
    print("6. 🌈 All sizes - Complete benchmark")
    print("7. 🎯 Custom selection - Choose multiple sizes")
    
    while True:
        try:
            choice = input("\nEnter choice (1-7): ").strip()
            if choice == '1':
                selected_sizes = ["tiny"]
                break
            elif choice == '2':
                selected_sizes = ["small"]
                break
            elif choice == '3':
                selected_sizes = ["medium"]
                break
            elif choice == '4':
                selected_sizes = ["large"]
                break
            elif choice == '5':
                selected_sizes = ["huge"]
                break
            elif choice == '6':
                selected_sizes = ["tiny", "small", "medium", "large", "huge"]
                break
            elif choice == '7':
                print("\nCustom selection - Enter size numbers separated by commas (e.g., 1,3,5):")
                print("1=tiny, 2=small, 3=medium, 4=large, 5=huge")
                custom_input = input("Your selection: ").strip()
                try:
                    size_map = {1: "tiny", 2: "small", 3: "medium", 4: "large", 5: "huge"}
                    custom_numbers = [int(x.strip()) for x in custom_input.split(',')]
                    selected_sizes = [size_map[num] for num in custom_numbers if num in size_map]
                    if selected_sizes:
                        break
                    else:
                        print("❌ No valid sizes selected. Please try again.")
                except ValueError:
                    print("❌ Invalid input format. Please use numbers separated by commas.")
            else:
                print("❌ Invalid choice. Please enter 1-7.")
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            sys.exit(0)
    
    # Ask about saving terminal output
    print(f"\nSelected dataset sizes: {', '.join(selected_sizes)}")
    save_output = input("Save terminal output to file? (y/N): ").strip().lower()
    
    return selected_sizes, save_output == 'y'


def main():
    """
    Main function to run ML performance testing.
    """
    selected_sizes, save_output = get_user_preferences()
    
    # Set up terminal logging if requested
    terminal_logger = None
    if save_output:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        size_str = "_".join(selected_sizes) if len(selected_sizes) <= 3 else "all_sizes"
        log_filename = f"ml_comprehensive_performance_test_{size_str}_{timestamp}.txt"
        terminal_logger = TerminalLogger(log_filename)
        sys.stdout = terminal_logger
        print(f"📝 Terminal output will be saved to: {log_filename}")
    
    try:
        print("🔬 ML Platform Performance Testing Suite")
        print("=" * 50)
        
        # Create performance tester
        tester = MLPerformanceTester()
        
        # Run comprehensive tests with selected sizes
        tester.run_comprehensive_tests(selected_sizes)
        
        # Run scalability tests
        print("\n🔬 Running additional scalability analysis...")
        tester.run_scalability_tests()
        
        # Generate report
        tester.generate_performance_report()
        
        print("\n🎉 ML performance testing completed successfully!")
        print(f"📁 Check the '{tester.output_dir}' directory for detailed results")
    
    finally:
        # Restore original stdout and close log file
        if terminal_logger:
            sys.stdout = terminal_logger.terminal
            terminal_logger.close()
            print(f"📄 Terminal output saved to: {log_filename}")


if __name__ == "__main__":
    main()
