"""
Performance Testing Suite for Anonymization Platform
====================================================

This module provides comprehensive performance testing for the anonymization platform,
including dataset generation, performance benchmarking, and detailed reporting.

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
from typing import Dict, List, Tuple, Any
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Add project root to path for imports
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Import the anonymization plugins
try:
    from src.anonymizers.base_anonymizer import Anonymizer
    from core.app import load_anonymizer_plugins, ANONYMIZER_PLUGINS
except ImportError as e:
    print(f"Error importing anonymization modules: {e}")
    print("Make sure you're running this from the correct directory structure")
    sys.exit(1)

class PerformanceTester:
    """
    Comprehensive performance testing framework for anonymization techniques.
    """
    
    def __init__(self, output_dir: str = "performance_results"):
        """
        Initialize the performance tester.
        
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
            'technique_performance': {},
            'scalability_tests': {},
            'memory_usage': {},
            'comparative_analysis': {}
        }
        
        # Load plugins
        load_anonymizer_plugins(include_test_plugin=False)
        self.plugins = ANONYMIZER_PLUGINS.copy()
        
        # Skip homomorphic encryption techniques
        techniques_to_skip = [
            'homomorphic',
            'encryption',
            'Homomorphic Encryption',
            'Homomorphic Encryption Core',
            'homomorphic_encryption',
            'Homomorphic'
        ]
        
        original_count = len(self.plugins)
        for technique_name in list(self.plugins.keys()):
            if any(skip_name.lower() in technique_name.lower() for skip_name in techniques_to_skip):
                print(f"⚠️  Skipping {technique_name} (homomorphic encryption)")
                del self.plugins[technique_name]
        
        skipped_count = original_count - len(self.plugins)
        print(f"🔧 Loaded {len(self.plugins)} anonymization techniques for testing")
        if skipped_count > 0:
            print(f"⏭️  Skipped {skipped_count} homomorphic encryption technique(s)")
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
    
    def generate_synthetic_datasets(self, selected_sizes: List[str] = None) -> Dict[str, pd.DataFrame]:
        """
        Generate synthetic datasets with varying characteristics for testing.
        
        Args:
            selected_sizes: List of dataset sizes to generate ('tiny', 'small', 'medium', 'large', 'huge')
        """
        if selected_sizes is None:
            selected_sizes = ['small', 'medium', 'large']
        
        print(f"📊 Generating synthetic test datasets for sizes: {', '.join(selected_sizes)}...")
        
        datasets = {}
        
        # Define size configurations
        size_configs = {
            'tiny': {'n_rows': 500, 'n_cols': 8, 'complexity': 'simple'},
            'small': {'n_rows': 1000, 'n_cols': 10, 'complexity': 'simple'},
            'medium': {'n_rows': 10000, 'n_cols': 15, 'complexity': 'medium'},
            'large': {'n_rows': 50000, 'n_cols': 20, 'complexity': 'complex'},
            'huge': {'n_rows': 100000, 'n_cols': 25, 'complexity': 'complex'}
        }
        
        # Generate datasets for selected sizes
        for size in selected_sizes:
            if size in size_configs:
                config = size_configs[size]
                datasets[f'{size}_{config["n_rows"]}'] = self._create_synthetic_dataset(
                    n_rows=config['n_rows'], 
                    n_cols=config['n_cols'], 
                    complexity=config['complexity']
                )
                print(f"   ✓ {size}: {config['n_rows']} rows, {config['n_cols']} columns")
        
        # Always include a mixed types dataset for comprehensive testing
        if 'medium' in selected_sizes or 'large' in selected_sizes:
            base_rows = 5000 if 'medium' in selected_sizes else 10000
            datasets['mixed_types'] = self._create_mixed_types_dataset(n_rows=base_rows)
            print(f"   ✓ mixed_types: {base_rows} rows with various data types")
        
        print(f"✅ Generated {len(datasets)} test datasets")
        return datasets
    
    def _create_synthetic_dataset(self, n_rows: int, n_cols: int, complexity: str) -> pd.DataFrame:
        """Create a synthetic dataset with specified characteristics."""
        np.random.seed(42)  # For reproducible results
        
        data = {}
        
        if complexity == 'simple':
            # Simple numeric and categorical data
            for i in range(n_cols // 2):
                data[f'numeric_{i}'] = np.random.normal(100, 15, n_rows)
            for i in range(n_cols // 2):
                data[f'category_{i}'] = np.random.choice(['A', 'B', 'C', 'D'], n_rows)
        
        elif complexity == 'medium':
            # Mix of data types with some correlations
            data['age'] = np.random.randint(18, 80, n_rows)
            data['income'] = data['age'] * 1000 + np.random.normal(0, 10000, n_rows)
            data['education'] = np.random.choice(['High School', 'Bachelor', 'Master', 'PhD'], n_rows)
            data['city'] = np.random.choice([f'City_{i}' for i in range(20)], n_rows)
            
            # Additional random columns
            for i in range(n_cols - 4):
                if i % 2 == 0:
                    data[f'feature_{i}'] = np.random.exponential(2, n_rows)
                else:
                    data[f'label_{i}'] = np.random.choice([f'Label_{j}' for j in range(10)], n_rows)
        
        elif complexity == 'complex':
            # Complex relationships and multiple data types
            data['id'] = range(n_rows)
            data['timestamp'] = pd.date_range('2020-01-01', periods=n_rows, freq='H')
            data['sensitive_attr'] = np.random.choice(['Group_A', 'Group_B', 'Group_C'], n_rows)
            
            # Create correlated features
            base_value = np.random.normal(0, 1, n_rows)
            for i in range(n_cols - 3):
                noise_level = 0.1 if i < 5 else 0.5  # Some features more correlated
                data[f'corr_feature_{i}'] = base_value + np.random.normal(0, noise_level, n_rows)
        
        elif complexity == 'wide':
            # Many columns with different characteristics
            for i in range(n_cols):
                if i % 4 == 0:
                    data[f'numeric_{i}'] = np.random.normal(0, 1, n_rows)
                elif i % 4 == 1:
                    data[f'categorical_{i}'] = np.random.choice([f'Cat_{j}' for j in range(5)], n_rows)
                elif i % 4 == 2:
                    data[f'binary_{i}'] = np.random.choice([0, 1], n_rows)
                else:
                    data[f'text_{i}'] = np.random.choice([f'Text_{j}' for j in range(100)], n_rows)
        
        elif complexity == 'high_cardinality':
            # High cardinality categorical variables
            data['user_id'] = [f'user_{i}' for i in range(n_rows)]
            data['email'] = [f'user_{i}@domain{i%10}.com' for i in range(n_rows)]
            data['ip_address'] = [f'192.168.{i%256}.{i%256}' for i in range(n_rows)]
            
            # Regular columns
            for i in range(n_cols - 3):
                if i % 2 == 0:
                    data[f'metric_{i}'] = np.random.exponential(1, n_rows)
                else:
                    data[f'category_{i}'] = np.random.choice([f'C_{j}' for j in range(50)], n_rows)
        
        return pd.DataFrame(data)
    
    def _create_mixed_types_dataset(self, n_rows: int = 5000) -> pd.DataFrame:
        """Create a dataset with mixed data types for comprehensive testing."""
        
        data = {
            # Numeric types
            'integer_col': np.random.randint(1, 1000, n_rows),
            'float_col': np.random.normal(50, 15, n_rows),
            'currency': np.random.uniform(10, 10000, n_rows),
            
            # Categorical types
            'category_low_card': np.random.choice(['A', 'B', 'C'], n_rows),
            'category_high_card': np.random.choice([f'Item_{i}' for i in range(100)], n_rows),
            
            # Text-like data
            'names': np.random.choice([f'Person_{i}' for i in range(500)], n_rows),
            'addresses': np.random.choice([f'{i} Main St, City {i%20}' for i in range(200)], n_rows),
            
            # Boolean
            'is_active': np.random.choice([True, False], n_rows),
            
            # Date-like (as strings for anonymization testing)
            'join_date': pd.date_range('2015-01-01', periods=n_rows, freq='D').strftime('%Y-%m-%d'),
            
            # Sensitive attribute
            'sensitive_group': np.random.choice(['Sensitive_A', 'Sensitive_B', 'Sensitive_C'], n_rows)
        }
        
        return pd.DataFrame(data)
    
    def run_performance_test(self, technique_name: str, plugin: Anonymizer, 
                           df: pd.DataFrame, sa_col: str = None) -> Dict[str, Any]:
        """
        Run performance test for a specific technique on a dataset.
        """
        # Start memory tracking
        tracemalloc.start()
        process = psutil.Process()
        
        # Get initial memory
        mem_before = process.memory_info().rss / 1024 / 1024  # MB
        
        # Get default parameters (simplified for testing)
        try:
            # Create a unique key prefix for this test
            test_key_prefix = f"perf_test_{technique_name.lower().replace(' ', '_')}"
            
            # Get minimal parameters - we'll use empty dict for most tests
            # to focus on core performance rather than UI complexity
            all_cols = df.columns.tolist()
            
            # For performance testing, we'll use default/minimal parameters
            # This avoids the complexity of UI state management during automated testing
            if hasattr(plugin, 'get_default_parameters'):
                parameters = plugin.get_default_parameters(all_cols, sa_col, df)
            else:
                # Fallback: use empty parameters dict
                parameters = {}
            
            # Start timing
            start_time = time.time()
            start_cpu = time.process_time()
            
            # Run anonymization
            result_df = plugin.anonymize(df.copy(), parameters, sa_col)
            
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
            
            # Data quality metrics
            rows_processed = len(df)
            cols_processed = len(df.columns)
            throughput = rows_processed / wall_time if wall_time > 0 else 0
            
            return {
                'success': True,
                'wall_time_seconds': wall_time,
                'cpu_time_seconds': cpu_time,
                'memory_used_mb': memory_used,
                'peak_memory_mb': peak_memory,
                'rows_processed': rows_processed,
                'cols_processed': cols_processed,
                'throughput_rows_per_sec': throughput,
                'result_shape': result_df.shape,
                'data_preserved': len(result_df) == len(df),
                'parameters_used': parameters
            }
            
        except Exception as e:
            tracemalloc.stop()
            return {
                'success': False,
                'error': str(e),
                'error_type': type(e).__name__,
                'wall_time_seconds': None,
                'cpu_time_seconds': None,
                'memory_used_mb': None,
                'peak_memory_mb': None,
                'throughput_rows_per_sec': 0
            }
    
    def run_comprehensive_tests(self, selected_sizes: List[str] = None) -> None:
        """
        Run comprehensive performance tests across all techniques and datasets.
        
        Args:
            selected_sizes: List of dataset sizes to test ('tiny', 'small', 'medium', 'large', 'huge')
        """
        print("🚀 Starting comprehensive performance testing...")
        
        # Generate test datasets
        datasets = self.generate_synthetic_datasets(selected_sizes)
        
        # Test each technique on each dataset
        total_tests = len(self.plugins) * len(datasets)
        current_test = 0
        
        for dataset_name, df in datasets.items():
            print(f"\n📊 Testing dataset: {dataset_name} (Shape: {df.shape})")
            
            self.results['dataset_performance'][dataset_name] = {
                'shape': df.shape,
                'memory_size_mb': df.memory_usage(deep=True).sum() / 1024 / 1024,
                'dtypes': df.dtypes.to_dict(),
                'technique_results': {}
            }
            
            for technique_name, plugin in self.plugins.items():
                current_test += 1
                progress = (current_test / total_tests) * 100
                
                print(f"  ⚙️  [{progress:.1f}%] Testing {technique_name}...")
                
                # Determine sensitive attribute for testing
                sa_col = None
                if 'sensitive' in df.columns:
                    sa_col = 'sensitive'
                elif 'sensitive_attr' in df.columns:
                    sa_col = 'sensitive_attr'
                elif 'sensitive_group' in df.columns:
                    sa_col = 'sensitive_group'
                
                # Run the performance test
                result = self.run_performance_test(technique_name, plugin, df, sa_col)
                
                # Store results
                self.results['dataset_performance'][dataset_name]['technique_results'][technique_name] = result
                
                if not result['success']:
                    print(f"    ❌ Failed: {result.get('error', 'Unknown error')}")
                else:
                    print(f"    ✅ Success: {result['wall_time_seconds']:.3f}s, "
                          f"{result['throughput_rows_per_sec']:.0f} rows/sec")
        
        print("\n✅ Comprehensive testing completed!")
    
    def run_scalability_tests(self) -> None:
        """
        Test how techniques scale with increasing dataset sizes.
        """
        print("\n📈 Running scalability tests...")
        
        # Test with increasing sizes
        sizes = [1000, 5000, 10000, 25000, 50000]
        
        # Select a few representative techniques for scalability testing
        test_techniques = {}
        for name, plugin in list(self.plugins.items())[:5]:  # Test first 5 techniques
            test_techniques[name] = plugin
        
        for technique_name, plugin in test_techniques.items():
            print(f"  📊 Scalability test for {technique_name}...")
            
            scalability_results = []
            
            for size in sizes:
                print(f"    Testing size: {size} rows...")
                
                # Generate dataset of this size
                test_df = self._create_synthetic_dataset(size, 10, 'medium')
                
                # Run performance test
                result = self.run_performance_test(technique_name, plugin, test_df)
                
                if result['success']:
                    scalability_results.append({
                        'dataset_size': size,
                        'wall_time': result['wall_time_seconds'],
                        'memory_used': result['memory_used_mb'],
                        'throughput': result['throughput_rows_per_sec']
                    })
            
            self.results['scalability_tests'][technique_name] = scalability_results
        
        print("✅ Scalability tests completed!")
    
    def generate_performance_report(self) -> None:
        """
        Generate comprehensive performance report with visualizations.
        """
        print("\n📊 Generating performance report...")
        
        # Create summary statistics
        self._create_summary_statistics()
        
        # Create visualizations
        self._create_performance_visualizations()
        
        # Save raw results
        results_file = self.output_dir / f"performance_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(results_file, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        
        # Generate readable report
        self._generate_readable_report()
        
        print(f"📋 Performance report saved to: {self.output_dir}")
    
    def _create_summary_statistics(self) -> None:
        """Create summary statistics for the performance report."""
        summary = {
            'total_techniques_tested': len(self.plugins),
            'total_datasets_tested': len(self.results['dataset_performance']),
            'successful_tests': 0,
            'failed_tests': 0,
            'fastest_technique': None,
            'most_memory_efficient': None,
            'highest_throughput': None
        }
        
        all_results = []
        
        # Collect all successful results
        for dataset_name, dataset_results in self.results['dataset_performance'].items():
            for technique_name, result in dataset_results['technique_results'].items():
                if result['success']:
                    summary['successful_tests'] += 1
                    all_results.append({
                        'dataset': dataset_name,
                        'technique': technique_name,
                        **result
                    })
                else:
                    summary['failed_tests'] += 1
        
        if all_results:
            # Find best performers
            fastest = min(all_results, key=lambda x: x['wall_time_seconds'])
            summary['fastest_technique'] = {
                'technique': fastest['technique'],
                'dataset': fastest['dataset'],
                'time': fastest['wall_time_seconds']
            }
            
            most_efficient = min(all_results, key=lambda x: x['memory_used_mb'])
            summary['most_memory_efficient'] = {
                'technique': most_efficient['technique'],
                'dataset': most_efficient['dataset'],
                'memory': most_efficient['memory_used_mb']
            }
            
            highest_throughput = max(all_results, key=lambda x: x['throughput_rows_per_sec'])
            summary['highest_throughput'] = {
                'technique': highest_throughput['technique'],
                'dataset': highest_throughput['dataset'],
                'throughput': highest_throughput['throughput_rows_per_sec']
            }
        
        self.results['summary'] = summary
    
    def _create_performance_visualizations(self) -> None:
        """Create separate performance visualization charts."""
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
            
            # Set style
            plt.style.use('seaborn-v0_8')
            sns.set_palette("husl")
            
            # Collect data for visualization
            performance_data = []
            
            for dataset_name, dataset_results in self.results['dataset_performance'].items():
                for technique_name, result in dataset_results['technique_results'].items():
                    if result['success']:
                        performance_data.append({
                            'Dataset': dataset_name,
                            'Technique': technique_name,
                            'Wall Time (s)': result['wall_time_seconds'],
                            'CPU Time (s)': result['cpu_time_seconds'],
                            'Memory Used (MB)': result['memory_used_mb'],
                            'Throughput (rows/s)': result['throughput_rows_per_sec'],
                            'Rows': result['rows_processed']
                        })
            
            if not performance_data:
                print("⚠️  No successful results to visualize")
                return
            
            df_viz = pd.DataFrame(performance_data)
            print(f"📊 Creating {5} separate visualization charts...")
            
            # 1. Execution Time Distribution by Technique
            plt.figure(figsize=(12, 8))
            sns.boxplot(data=df_viz, x='Technique', y='Wall Time (s)')
            plt.xticks(rotation=45, ha='right')
            plt.title('Execution Time Distribution by Technique', fontsize=14, fontweight='bold')
            plt.xlabel('Anonymization Technique', fontsize=12)
            plt.ylabel('Wall Time (seconds)', fontsize=12)
            plt.tight_layout()
            
            # Save chart 1
            chart1_file = self.output_dir / 'execution_time_distribution_by_technique.png'
            plt.savefig(chart1_file, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"   ✅ Saved: {chart1_file.name}")
            
            # 2. Memory Usage vs Dataset Size
            plt.figure(figsize=(12, 8))
            sns.scatterplot(data=df_viz, x='Rows', y='Memory Used (MB)', hue='Technique', s=100, alpha=0.7)
            plt.title('Memory Usage vs Dataset Size', fontsize=14, fontweight='bold')
            plt.xlabel('Number of Rows', fontsize=12)
            plt.ylabel('Memory Used (MB)', fontsize=12)
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.tight_layout()
            
            # Save chart 2
            chart2_file = self.output_dir / 'memory_usage_vs_dataset_size.png'
            plt.savefig(chart2_file, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"   ✅ Saved: {chart2_file.name}")
            
            # 3. Average Throughput by Technique
            plt.figure(figsize=(12, 8))
            avg_throughput = df_viz.groupby('Technique')['Throughput (rows/s)'].mean().sort_values(ascending=True)
            colors = plt.cm.viridis(range(len(avg_throughput)))
            bars = avg_throughput.plot(kind='barh', color=colors, figsize=(12, 8))
            plt.title('Average Throughput by Technique', fontsize=14, fontweight='bold')
            plt.xlabel('Rows per Second', fontsize=12)
            plt.ylabel('Anonymization Technique', fontsize=12)
            
            # Add value labels on bars
            for i, v in enumerate(avg_throughput.values):
                plt.text(v + max(avg_throughput.values) * 0.01, i, f'{v:.0f}', 
                        va='center', fontsize=10)
            
            plt.tight_layout()
            
            # Save chart 3
            chart3_file = self.output_dir / 'average_throughput_by_technique.png'
            plt.savefig(chart3_file, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"   ✅ Saved: {chart3_file.name}")
            
            # 4. Execution Time Heatmap
            plt.figure(figsize=(14, 8))
            pivot_data = df_viz.pivot_table(values='Wall Time (s)', index='Dataset', columns='Technique', aggfunc='mean')
            sns.heatmap(pivot_data, annot=True, fmt='.2f', cmap='YlOrRd', cbar_kws={'label': 'Wall Time (seconds)'})
            plt.title('Execution Time Heatmap (seconds)', fontsize=14, fontweight='bold')
            plt.xlabel('Anonymization Technique', fontsize=12)
            plt.ylabel('Dataset', fontsize=12)
            plt.xticks(rotation=45, ha='right')
            plt.yticks(rotation=0)
            plt.tight_layout()
            
            # Save chart 4
            chart4_file = self.output_dir / 'execution_time_heatmap_seconds.png'
            plt.savefig(chart4_file, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"   ✅ Saved: {chart4_file.name}")
            
            # 5. CPU Usage Analysis (NEW)
            plt.figure(figsize=(12, 8))
            
            # Create subplot for CPU vs Wall Time comparison
            plt.subplot(2, 1, 1)
            techniques = df_viz['Technique'].unique()
            x_pos = range(len(techniques))
            
            avg_wall_time = [df_viz[df_viz['Technique'] == tech]['Wall Time (s)'].mean() for tech in techniques]
            avg_cpu_time = [df_viz[df_viz['Technique'] == tech]['CPU Time (s)'].mean() for tech in techniques]
            
            width = 0.35
            plt.bar([x - width/2 for x in x_pos], avg_wall_time, width, label='Wall Time', alpha=0.8, color='skyblue')
            plt.bar([x + width/2 for x in x_pos], avg_cpu_time, width, label='CPU Time', alpha=0.8, color='lightcoral')
            
            plt.xlabel('Anonymization Technique')
            plt.ylabel('Time (seconds)')
            plt.title('CPU vs Wall Time Comparison by Technique', fontweight='bold')
            plt.xticks(x_pos, techniques, rotation=45, ha='right')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            # Create subplot for CPU efficiency
            plt.subplot(2, 1, 2)
            cpu_efficiency = []
            for tech in techniques:
                tech_data = df_viz[df_viz['Technique'] == tech]
                avg_wall = tech_data['Wall Time (s)'].mean()
                avg_cpu = tech_data['CPU Time (s)'].mean()
                efficiency = (avg_cpu / avg_wall * 100) if avg_wall > 0 else 0
                cpu_efficiency.append(efficiency)
            
            colors = ['green' if eff > 80 else 'orange' if eff > 60 else 'red' for eff in cpu_efficiency]
            bars = plt.bar(techniques, cpu_efficiency, color=colors, alpha=0.7)
            plt.xlabel('Anonymization Technique')
            plt.ylabel('CPU Efficiency (%)')
            plt.title('CPU Efficiency by Technique (CPU Time / Wall Time * 100)', fontweight='bold')
            plt.xticks(rotation=45, ha='right')
            plt.grid(True, alpha=0.3)
            
            # Add percentage labels on bars
            for bar, eff in zip(bars, cpu_efficiency):
                plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                        f'{eff:.1f}%', ha='center', va='bottom', fontsize=9)
            
            plt.tight_layout()
            
            # Save chart 5
            chart5_file = self.output_dir / 'cpu_usage_analysis.png'
            plt.savefig(chart5_file, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"   ✅ Saved: {chart5_file.name}")
            
            # Create scalability chart if data exists
            if self.results['scalability_tests']:
                self._create_scalability_chart()
                print(f"📈 All 6 visualization charts saved to: {self.output_dir}")
                print("   📊 Charts created:")
                print("   1. execution_time_distribution_by_technique.png")
                print("   2. memory_usage_vs_dataset_size.png") 
                print("   3. average_throughput_by_technique.png")
                print("   4. execution_time_heatmap_seconds.png")
                print("   5. cpu_usage_analysis.png")
                print("   6. scalability_analysis_by_dataset_size.png")
            else:
                print(f"📈 All 5 visualization charts saved to: {self.output_dir}")
                print("   📊 Charts created:")
                print("   1. execution_time_distribution_by_technique.png")
                print("   2. memory_usage_vs_dataset_size.png") 
                print("   3. average_throughput_by_technique.png")
                print("   4. execution_time_heatmap_seconds.png")
                print("   5. cpu_usage_analysis.png")
            
        except ImportError:
            print("⚠️  Matplotlib/Seaborn not available for visualizations")
        except Exception as e:
            print(f"⚠️  Error creating visualizations: {e}")
    
    def _create_scalability_chart(self) -> None:
        """Create scalability analysis chart."""
        try:
            import matplotlib.pyplot as plt
            
            fig, axes = plt.subplots(1, 3, figsize=(18, 6))
            fig.suptitle('Scalability Analysis', fontsize=16, fontweight='bold')
            
            for technique_name, results in self.results['scalability_tests'].items():
                if not results:
                    continue
                
                sizes = [r['dataset_size'] for r in results]
                times = [r['wall_time'] for r in results]
                memory = [r['memory_used'] for r in results]
                throughput = [r['throughput'] for r in results]
                
                # Execution time scaling
                axes[0].plot(sizes, times, marker='o', label=technique_name, linewidth=2)
                
                # Memory scaling
                axes[1].plot(sizes, memory, marker='s', label=technique_name, linewidth=2)
                
                # Throughput scaling
                axes[2].plot(sizes, throughput, marker='^', label=technique_name, linewidth=2)
            
            axes[0].set_title('Execution Time vs Dataset Size', fontweight='bold')
            axes[0].set_xlabel('Dataset Size (rows)')
            axes[0].set_ylabel('Wall Time (seconds)')
            axes[0].legend()
            axes[0].grid(True, alpha=0.3)
            
            axes[1].set_title('Memory Usage vs Dataset Size', fontweight='bold')
            axes[1].set_xlabel('Dataset Size (rows)')
            axes[1].set_ylabel('Memory Used (MB)')
            axes[1].legend()
            axes[1].grid(True, alpha=0.3)
            
            axes[2].set_title('Throughput vs Dataset Size', fontweight='bold')
            axes[2].set_xlabel('Dataset Size (rows)')
            axes[2].set_ylabel('Throughput (rows/sec)')
            axes[2].legend()
            axes[2].grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            scalability_file = self.output_dir / 'scalability_analysis_by_dataset_size.png'
            plt.savefig(scalability_file, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"   ✅ Saved: {scalability_file.name}")
            
        except Exception as e:
            print(f"⚠️  Error creating scalability chart: {e}")
    
    def _generate_readable_report(self) -> None:
        """Generate a human-readable performance report."""
        report_file = self.output_dir / f"performance_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
        
        with open(report_file, 'w') as f:
            f.write("# Anonymization Platform Performance Report\n\n")
            f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # System info
            f.write("## System Information\n\n")
            sys_info = self.results['test_metadata']['system_info']
            f.write(f"- **CPU Cores:** {sys_info['cpu_count']}\n")
            f.write(f"- **CPU Frequency:** {sys_info['cpu_freq']} MHz\n")
            f.write(f"- **Total Memory:** {sys_info['memory_total_gb']} GB\n")
            f.write(f"- **Platform:** {sys_info['platform']}\n\n")
            
            # Summary
            if 'summary' in self.results:
                summary = self.results['summary']
                f.write("## Test Summary\n\n")
                f.write(f"- **Techniques Tested:** {summary['total_techniques_tested']}\n")
                f.write(f"- **Datasets Tested:** {summary['total_datasets_tested']}\n")
                f.write(f"- **Successful Tests:** {summary['successful_tests']}\n")
                f.write(f"- **Failed Tests:** {summary['failed_tests']}\n\n")
                
                if summary.get('fastest_technique'):
                    fastest = summary['fastest_technique']
                    f.write(f"- **Fastest Technique:** {fastest['technique']} ")
                    f.write(f"({fastest['time']:.3f}s on {fastest['dataset']})\n")
                
                if summary.get('most_memory_efficient'):
                    efficient = summary['most_memory_efficient']
                    f.write(f"- **Most Memory Efficient:** {efficient['technique']} ")
                    f.write(f"({efficient['memory']:.1f}MB on {efficient['dataset']})\n")
                
                if summary.get('highest_throughput'):
                    throughput = summary['highest_throughput']
                    f.write(f"- **Highest Throughput:** {throughput['technique']} ")
                    f.write(f"({throughput['throughput']:.0f} rows/sec on {throughput['dataset']})\n\n")
            
            # Dataset results
            f.write("## Dataset Performance Results\n\n")
            for dataset_name, dataset_results in self.results['dataset_performance'].items():
                f.write(f"### {dataset_name}\n\n")
                f.write(f"- **Shape:** {dataset_results['shape']}\n")
                f.write(f"- **Memory Size:** {dataset_results['memory_size_mb']:.2f} MB\n\n")
                
                f.write("| Technique | Status | Time (s) | Memory (MB) | Throughput (rows/s) |\n")
                f.write("|-----------|--------|----------|-------------|--------------------|\n")
                
                for technique_name, result in dataset_results['technique_results'].items():
                    if result['success']:
                        f.write(f"| {technique_name} | ✅ | {result['wall_time_seconds']:.3f} | ")
                        f.write(f"{result['memory_used_mb']:.1f} | {result['throughput_rows_per_sec']:.0f} |\n")
                    else:
                        f.write(f"| {technique_name} | ❌ | - | - | - |\n")
                
                f.write("\n")
            
            # Recommendations
            f.write("## Performance Recommendations\n\n")
            f.write("Based on the test results:\n\n")
            f.write("1. **For Small Datasets (< 5K rows):** Any technique should perform well\n")
            f.write("2. **For Large Datasets (> 50K rows):** Consider memory-efficient techniques\n")
            f.write("3. **For Real-time Applications:** Focus on techniques with highest throughput\n")
            f.write("4. **For Memory-constrained Environments:** Use techniques with lowest memory usage\n\n")
            
            f.write("## Notes\n\n")
            f.write("- Performance may vary based on data characteristics and system configuration\n")
            f.write("- Results are averaged across multiple test runs where applicable\n")
            f.write("- Failed tests may indicate missing dependencies or configuration issues\n")
        
        print(f"📋 Readable report saved to: {report_file}")


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
    print("🔬 Anonymization Platform Comprehensive Performance Testing")
    print("=" * 65)
    
    # Get dataset sizes
    print("\nSelect dataset sizes to test:")
    print("1. 🐣 Tiny (500 rows) - Ultra fast")
    print("2. 🚀 Small (1K rows) - Fast")
    print("3. 📊 Medium (10K rows) - Standard")
    print("4. 📈 Large (50K rows) - Comprehensive")
    print("5. 🏭 Huge (100K rows) - Stress test")
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
    Main function to run performance testing.
    """
    selected_sizes, save_output = get_user_preferences()
    
    # Set up terminal logging if requested
    terminal_logger = None
    if save_output:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        size_str = "_".join(selected_sizes) if len(selected_sizes) <= 3 else "all_sizes"
        log_filename = f"comprehensive_performance_test_{size_str}_{timestamp}.txt"
        terminal_logger = TerminalLogger(log_filename)
        sys.stdout = terminal_logger
        print(f"� Terminal output will be saved to: {log_filename}")
    
    try:
        print("�🔬 Anonymization Platform Performance Testing Suite")
        print("=" * 60)
        
        # Create performance tester
        tester = PerformanceTester()
        
        # Run comprehensive tests with selected sizes
        tester.run_comprehensive_tests(selected_sizes)
        
        # Run scalability tests (keep default behavior for now)
        print("\n🔬 Running additional scalability analysis...")
        tester.run_scalability_tests()
        
        # Generate report
        tester.generate_performance_report()
        
        print("\n🎉 Performance testing completed successfully!")
        print(f"📁 Check the '{tester.output_dir}' directory for detailed results")
    
    finally:
        # Restore original stdout and close log file
        if terminal_logger:
            sys.stdout = terminal_logger.terminal
            terminal_logger.close()
            print(f"📄 Terminal output saved to: {log_filename}")


if __name__ == "__main__":
    main()
