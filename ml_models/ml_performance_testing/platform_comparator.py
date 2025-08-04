"""
Platform Performance Comparison Tool
===================================

Compares performance between Anonymization and ML platforms side-by-side.
Generates comparative visualizations and analysis reports.

Author: Bachelor Thesis Project
Date: July 2025
"""

import os
import sys
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from typing import Dict, List, Any, Tuple
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Add performance testing directories to path
PERFORMANCE_TESTING_DIR = os.path.join(PROJECT_ROOT, 'ml_models', 'performance_testing')
ML_PERFORMANCE_TESTING_DIR = os.path.dirname(__file__)  # Current directory

if PERFORMANCE_TESTING_DIR not in sys.path:
    sys.path.insert(0, PERFORMANCE_TESTING_DIR)
if ML_PERFORMANCE_TESTING_DIR not in sys.path:
    sys.path.insert(0, ML_PERFORMANCE_TESTING_DIR)

SRC_DIR = os.path.join(PROJECT_ROOT, 'ml_models', 'src')
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

# Import performance testers
PerformanceTester = None
MLPerformanceTester = None

# Try to import performance_tester
try:
    sys.path.insert(0, PERFORMANCE_TESTING_DIR)
    from performance_tester import PerformanceTester
    print("✅ Successfully imported PerformanceTester")
except ImportError as e:
    print(f"⚠️  Warning: Could not import PerformanceTester: {e}")
    print("   Anonymization platform testing will be skipped")

# Try to import ml_performance_tester
try:
    sys.path.insert(0, ML_PERFORMANCE_TESTING_DIR)
    from ml_performance_tester import MLPerformanceTester
    print("✅ Successfully imported MLPerformanceTester")
except ImportError as e:
    print(f"⚠️  Warning: Could not import MLPerformanceTester: {e}")
    print("   ML platform testing will be skipped")

# Check if at least one platform is available
if PerformanceTester is None and MLPerformanceTester is None:
    print("❌ Error: Neither performance tester could be imported!")
    print("Make sure the required dependencies are installed.")
    sys.exit(1)
elif PerformanceTester is None:
    print("ℹ️  Running in ML-only mode (anonymization platform unavailable)")
elif MLPerformanceTester is None:
    print("ℹ️  Running in Anonymization-only mode (ML platform unavailable)")


class PlatformComparator:
    """
    Compare performance between Anonymization and ML platforms.
    """
    
    def __init__(self, test_level='medium'):
        self.test_level = test_level
        self.results = {
            'timestamp': datetime.now().isoformat(),
            'test_level': test_level,
            'anonymization_results': None,
            'ml_results': None,
            'comparison_metrics': {}
        }
        
    def run_anonymization_tests(self):
        """Run anonymization platform performance tests."""
        if PerformanceTester is None:
            print("⚠️  Skipping anonymization tests - PerformanceTester not available")
            return
            
        print("🔐 Running Anonymization Platform Tests...")
        
        # Create test configurations for different dataset sizes
        test_configs = self._get_test_configurations()
        
        anon_results = {}
        
        for size_name, config in test_configs.items():
            print(f"  📊 Testing with {size_name} dataset...")
            
            try:
                tester = PerformanceTester(
                    anonymizer_count=config['algorithm_count'],
                    dataset_size=config['dataset_size'],
                    save_plots=False  # We'll create our own comparison plots
                )
                
                # Run tests and collect results
                results = tester.run_performance_test()
                anon_results[size_name] = results
            except Exception as e:
                print(f"   ❌ Error testing {size_name}: {e}")
                anon_results[size_name] = {'error': str(e)}
        
        self.results['anonymization_results'] = anon_results
        print("✅ Anonymization tests completed")
        
    def run_ml_tests(self):
        """Run ML platform performance tests."""
        if MLPerformanceTester is None:
            print("⚠️  Skipping ML tests - MLPerformanceTester not available")
            return
            
        print("🤖 Running ML Platform Tests...")
        
        test_configs = self._get_test_configurations()
        
        ml_results = {}
        
        for size_name, config in test_configs.items():
            print(f"  📊 Testing with {size_name} dataset...")
            
            try:
                tester = MLPerformanceTester(
                    algorithm_count=config['algorithm_count'],
                    dataset_size=config['dataset_size'],
                    save_plots=False  # We'll create our own comparison plots
                )
                
                # Run tests and collect results
                results = tester.run_ml_performance_test()
                ml_results[size_name] = results
            except Exception as e:
                print(f"   ❌ Error testing {size_name}: {e}")
                ml_results[size_name] = {'error': str(e)}
        
        self.results['ml_results'] = ml_results
        print("✅ ML tests completed")
    
    def _get_test_configurations(self) -> Dict[str, Dict]:
        """Get test configurations based on test level."""
        if self.test_level == 'simple':
            return {
                'small': {'dataset_size': 1000, 'algorithm_count': 3},
                'medium': {'dataset_size': 5000, 'algorithm_count': 5}
            }
        elif self.test_level == 'medium':
            return {
                'small': {'dataset_size': 1000, 'algorithm_count': 5},
                'medium': {'dataset_size': 5000, 'algorithm_count': 7},
                'large': {'dataset_size': 10000, 'algorithm_count': 7}
            }
        else:  # full
            return {
                'small': {'dataset_size': 1000, 'algorithm_count': 7},
                'medium': {'dataset_size': 5000, 'algorithm_count': 10},
                'large': {'dataset_size': 10000, 'algorithm_count': 10},
                'huge': {'dataset_size': 50000, 'algorithm_count': 10}
            }
    
    def extract_metrics_for_comparison(self):
        """Extract comparable metrics from both platforms."""
        print("📊 Extracting metrics for comparison...")
        
        comparison_data = {
            'execution_times': {'anonymization': [], 'ml': []},
            'memory_usage': {'anonymization': [], 'ml': []},
            'throughput': {'anonymization': [], 'ml': []},
            'cpu_efficiency': {'anonymization': [], 'ml': []},
            'dataset_sizes': [],
            'algorithm_counts': {'anonymization': [], 'ml': []}
        }
        
        # Extract anonymization metrics
        if self.results['anonymization_results']:
            for size_name, results in self.results['anonymization_results'].items():
                if 'performance_metrics' in results:
                    metrics = results['performance_metrics']
                    comparison_data['execution_times']['anonymization'].extend(
                        metrics.get('execution_times', [])
                    )
                    comparison_data['memory_usage']['anonymization'].extend(
                        metrics.get('memory_usage', [])
                    )
                    comparison_data['throughput']['anonymization'].extend(
                        metrics.get('throughput', [])
                    )
                    comparison_data['cpu_efficiency']['anonymization'].extend(
                        metrics.get('cpu_efficiency', [])
                    )
                    
                    # Add dataset info
                    if size_name not in comparison_data['dataset_sizes']:
                        comparison_data['dataset_sizes'].append(size_name)
                    
                    comparison_data['algorithm_counts']['anonymization'].append(
                        len(metrics.get('execution_times', []))
                    )
        
        # Extract ML metrics
        if self.results['ml_results']:
            for size_name, results in self.results['ml_results'].items():
                if 'performance_metrics' in results:
                    metrics = results['performance_metrics']
                    comparison_data['execution_times']['ml'].extend(
                        metrics.get('execution_times', [])
                    )
                    comparison_data['memory_usage']['ml'].extend(
                        metrics.get('memory_usage', [])
                    )
                    comparison_data['throughput']['ml'].extend(
                        metrics.get('throughput', [])
                    )
                    comparison_data['cpu_efficiency']['ml'].extend(
                        metrics.get('cpu_efficiency', [])
                    )
                    
                    comparison_data['algorithm_counts']['ml'].append(
                        len(metrics.get('execution_times', []))
                    )
        
        self.results['comparison_metrics'] = comparison_data
        print("✅ Metrics extraction completed")
    
    def create_comparison_visualizations(self):
        """Create side-by-side comparison visualizations."""
        print("📈 Creating comparison visualizations...")
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Set up the plotting style
        plt.style.use('default')
        sns.set_palette("husl")
        
        comparison_data = self.results['comparison_metrics']
        
        # 1. Execution Time Comparison
        self._create_execution_time_comparison(comparison_data, timestamp)
        
        # 2. Memory Usage Comparison
        self._create_memory_usage_comparison(comparison_data, timestamp)
        
        # 3. Throughput Comparison
        self._create_throughput_comparison(comparison_data, timestamp)
        
        # 4. CPU Efficiency Comparison
        self._create_cpu_efficiency_comparison(comparison_data, timestamp)
        
        # 5. Overall Performance Dashboard
        self._create_performance_dashboard(comparison_data, timestamp)
        
        print("✅ Comparison visualizations created")
    
    def _create_execution_time_comparison(self, data: Dict, timestamp: str):
        """Create execution time comparison chart."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Anonymization execution times
        anon_times = data['execution_times']['anonymization']
        if anon_times:
            ax1.hist(anon_times, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
            ax1.set_title('Anonymization Platform\nExecution Time Distribution', fontsize=12, fontweight='bold')
            ax1.set_xlabel('Execution Time (seconds)')
            ax1.set_ylabel('Frequency')
            ax1.grid(True, alpha=0.3)
            
            # Add statistics
            mean_time = np.mean(anon_times)
            ax1.axvline(mean_time, color='red', linestyle='--', alpha=0.8, label=f'Mean: {mean_time:.2f}s')
            ax1.legend()
        
        # ML execution times
        ml_times = data['execution_times']['ml']
        if ml_times:
            ax2.hist(ml_times, bins=20, alpha=0.7, color='lightcoral', edgecolor='black')
            ax2.set_title('ML Platform\nExecution Time Distribution', fontsize=12, fontweight='bold')
            ax2.set_xlabel('Execution Time (seconds)')
            ax2.set_ylabel('Frequency')
            ax2.grid(True, alpha=0.3)
            
            # Add statistics
            mean_time = np.mean(ml_times)
            ax2.axvline(mean_time, color='red', linestyle='--', alpha=0.8, label=f'Mean: {mean_time:.2f}s')
            ax2.legend()
        
        plt.tight_layout()
        filename = f'platform_execution_time_comparison_{timestamp}.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"📊 Execution time comparison saved: {filename}")
    
    def _create_memory_usage_comparison(self, data: Dict, timestamp: str):
        """Create memory usage comparison chart."""
        fig, ax = plt.subplots(1, 1, figsize=(12, 6))
        
        platforms = []
        memory_values = []
        
        if data['memory_usage']['anonymization']:
            platforms.extend(['Anonymization'] * len(data['memory_usage']['anonymization']))
            memory_values.extend(data['memory_usage']['anonymization'])
        
        if data['memory_usage']['ml']:
            platforms.extend(['ML'] * len(data['memory_usage']['ml']))
            memory_values.extend(data['memory_usage']['ml'])
        
        if platforms and memory_values:
            # Convert to MB for better readability
            memory_mb = [m / (1024**2) for m in memory_values]
            
            df_memory = pd.DataFrame({
                'Platform': platforms,
                'Memory Usage (MB)': memory_mb
            })
            
            sns.boxplot(data=df_memory, x='Platform', y='Memory Usage (MB)', ax=ax)
            ax.set_title('Platform Memory Usage Comparison', fontsize=14, fontweight='bold')
            ax.grid(True, alpha=0.3)
            
            # Add mean lines
            for platform in df_memory['Platform'].unique():
                platform_data = df_memory[df_memory['Platform'] == platform]['Memory Usage (MB)']
                mean_val = platform_data.mean()
                ax.text(platform, mean_val, f'μ={mean_val:.1f}MB', 
                       ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        filename = f'platform_memory_usage_comparison_{timestamp}.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"📊 Memory usage comparison saved: {filename}")
    
    def _create_throughput_comparison(self, data: Dict, timestamp: str):
        """Create throughput comparison chart."""
        fig, ax = plt.subplots(1, 1, figsize=(12, 6))
        
        platforms = []
        throughput_values = []
        
        if data['throughput']['anonymization']:
            platforms.extend(['Anonymization'] * len(data['throughput']['anonymization']))
            throughput_values.extend(data['throughput']['anonymization'])
        
        if data['throughput']['ml']:
            platforms.extend(['ML'] * len(data['throughput']['ml']))
            throughput_values.extend(data['throughput']['ml'])
        
        if platforms and throughput_values:
            df_throughput = pd.DataFrame({
                'Platform': platforms,
                'Throughput (records/sec)': throughput_values
            })
            
            sns.violinplot(data=df_throughput, x='Platform', y='Throughput (records/sec)', ax=ax)
            ax.set_title('Platform Throughput Comparison', fontsize=14, fontweight='bold')
            ax.grid(True, alpha=0.3)
            
            # Add median lines
            for i, platform in enumerate(df_throughput['Platform'].unique()):
                platform_data = df_throughput[df_throughput['Platform'] == platform]['Throughput (records/sec)']
                median_val = platform_data.median()
                ax.text(i, median_val, f'Median: {median_val:.1f}', 
                       ha='center', va='bottom', fontweight='bold', 
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        filename = f'platform_throughput_comparison_{timestamp}.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"📊 Throughput comparison saved: {filename}")
    
    def _create_cpu_efficiency_comparison(self, data: Dict, timestamp: str):
        """Create CPU efficiency comparison chart."""
        fig, ax = plt.subplots(1, 1, figsize=(12, 6))
        
        platforms = []
        cpu_efficiency_values = []
        
        if data['cpu_efficiency']['anonymization']:
            platforms.extend(['Anonymization'] * len(data['cpu_efficiency']['anonymization']))
            cpu_efficiency_values.extend(data['cpu_efficiency']['anonymization'])
        
        if data['cpu_efficiency']['ml']:
            platforms.extend(['ML'] * len(data['cpu_efficiency']['ml']))
            cpu_efficiency_values.extend(data['cpu_efficiency']['ml'])
        
        if platforms and cpu_efficiency_values:
            df_cpu = pd.DataFrame({
                'Platform': platforms,
                'CPU Efficiency (%)': [x * 100 for x in cpu_efficiency_values]
            })
            
            # Create bar plot with error bars
            platform_stats = df_cpu.groupby('Platform')['CPU Efficiency (%)'].agg(['mean', 'std']).reset_index()
            
            bars = ax.bar(platform_stats['Platform'], platform_stats['mean'], 
                         yerr=platform_stats['std'], capsize=5, alpha=0.7,
                         color=['skyblue', 'lightcoral'])
            
            ax.set_title('Platform CPU Efficiency Comparison', fontsize=14, fontweight='bold')
            ax.set_ylabel('CPU Efficiency (%)')
            ax.grid(True, alpha=0.3)
            
            # Add value labels on bars
            for bar, mean_val in zip(bars, platform_stats['mean']):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                       f'{mean_val:.1f}%', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        filename = f'platform_cpu_efficiency_comparison_{timestamp}.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"📊 CPU efficiency comparison saved: {filename}")
    
    def _create_performance_dashboard(self, data: Dict, timestamp: str):
        """Create comprehensive performance dashboard."""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # Prepare data for radar chart comparison
        metrics = ['Avg Execution Time', 'Avg Memory (MB)', 'Avg Throughput', 'Avg CPU Efficiency']
        
        anon_metrics = []
        ml_metrics = []
        
        # Calculate averages (normalize for comparison)
        if data['execution_times']['anonymization']:
            anon_exec = np.mean(data['execution_times']['anonymization'])
            anon_metrics.append(1 / max(anon_exec, 0.001))  # Inverse for "higher is better"
        else:
            anon_metrics.append(0)
        
        if data['execution_times']['ml']:
            ml_exec = np.mean(data['execution_times']['ml'])
            ml_metrics.append(1 / max(ml_exec, 0.001))  # Inverse for "higher is better"
        else:
            ml_metrics.append(0)
        
        if data['memory_usage']['anonymization']:
            anon_mem = np.mean(data['memory_usage']['anonymization']) / (1024**2)
            anon_metrics.append(1 / max(anon_mem, 0.001))  # Inverse for "lower is better"
        else:
            anon_metrics.append(0)
        
        if data['memory_usage']['ml']:
            ml_mem = np.mean(data['memory_usage']['ml']) / (1024**2)
            ml_metrics.append(1 / max(ml_mem, 0.001))  # Inverse for "lower is better"
        else:
            ml_metrics.append(0)
        
        if data['throughput']['anonymization']:
            anon_metrics.append(np.mean(data['throughput']['anonymization']))
        else:
            anon_metrics.append(0)
        
        if data['throughput']['ml']:
            ml_metrics.append(np.mean(data['throughput']['ml']))
        else:
            ml_metrics.append(0)
        
        if data['cpu_efficiency']['anonymization']:
            anon_metrics.append(np.mean(data['cpu_efficiency']['anonymization']) * 100)
        else:
            anon_metrics.append(0)
        
        if data['cpu_efficiency']['ml']:
            ml_metrics.append(np.mean(data['cpu_efficiency']['ml']) * 100)
        else:
            ml_metrics.append(0)
        
        # Summary statistics table
        ax1.axis('off')
        summary_data = {
            'Metric': ['Algorithms Tested', 'Avg Execution Time (s)', 'Avg Memory (MB)', 'Avg Throughput (rec/s)', 'Avg CPU Efficiency (%)'],
            'Anonymization': [
                len(data['execution_times']['anonymization']),
                f"{np.mean(data['execution_times']['anonymization']) if data['execution_times']['anonymization'] else 0:.2f}",
                f"{np.mean(data['memory_usage']['anonymization']) / (1024**2) if data['memory_usage']['anonymization'] else 0:.1f}",
                f"{np.mean(data['throughput']['anonymization']) if data['throughput']['anonymization'] else 0:.1f}",
                f"{np.mean(data['cpu_efficiency']['anonymization']) * 100 if data['cpu_efficiency']['anonymization'] else 0:.1f}"
            ],
            'ML Platform': [
                len(data['execution_times']['ml']),
                f"{np.mean(data['execution_times']['ml']) if data['execution_times']['ml'] else 0:.2f}",
                f"{np.mean(data['memory_usage']['ml']) / (1024**2) if data['memory_usage']['ml'] else 0:.1f}",
                f"{np.mean(data['throughput']['ml']) if data['throughput']['ml'] else 0:.1f}",
                f"{np.mean(data['cpu_efficiency']['ml']) * 100 if data['cpu_efficiency']['ml'] else 0:.1f}"
            ]
        }
        
        table = ax1.table(cellText=[[summary_data['Metric'][i], summary_data['Anonymization'][i], summary_data['ML Platform'][i]] 
                                   for i in range(len(summary_data['Metric']))],
                         colLabels=['Metric', 'Anonymization', 'ML Platform'],
                         cellLoc='center', loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2)
        ax1.set_title('Performance Summary Comparison', fontsize=12, fontweight='bold')
        
        # Performance trend comparison
        if len(data['dataset_sizes']) > 1:
            dataset_indices = range(len(data['dataset_sizes']))
            
            # Simplified trend (using algorithm counts as proxy)
            anon_trend = data['algorithm_counts']['anonymization'][:len(dataset_indices)]
            ml_trend = data['algorithm_counts']['ml'][:len(dataset_indices)]
            
            ax2.plot(dataset_indices, anon_trend, 'o-', label='Anonymization', color='skyblue', linewidth=2)
            ax2.plot(dataset_indices, ml_trend, 's-', label='ML Platform', color='lightcoral', linewidth=2)
            ax2.set_title('Algorithm Count by Dataset Size', fontsize=12, fontweight='bold')
            ax2.set_xlabel('Dataset Size Category')
            ax2.set_ylabel('Number of Algorithms Tested')
            ax2.set_xticks(dataset_indices)
            ax2.set_xticklabels(data['dataset_sizes'])
            ax2.legend()
            ax2.grid(True, alpha=0.3)
        
        # Platform efficiency comparison
        efficiency_platforms = []
        efficiency_values = []
        
        if data['cpu_efficiency']['anonymization']:
            efficiency_platforms.extend(['Anonymization'] * len(data['cpu_efficiency']['anonymization']))
            efficiency_values.extend([x * 100 for x in data['cpu_efficiency']['anonymization']])
        
        if data['cpu_efficiency']['ml']:
            efficiency_platforms.extend(['ML'] * len(data['cpu_efficiency']['ml']))
            efficiency_values.extend([x * 100 for x in data['cpu_efficiency']['ml']])
        
        if efficiency_platforms:
            df_eff = pd.DataFrame({'Platform': efficiency_platforms, 'Efficiency (%)': efficiency_values})
            sns.boxplot(data=df_eff, x='Platform', y='Efficiency (%)', ax=ax3)
            ax3.set_title('CPU Efficiency Distribution', fontsize=12, fontweight='bold')
            ax3.grid(True, alpha=0.3)
        
        # Performance score calculation
        platforms = ['Anonymization', 'ML Platform']
        if anon_metrics and ml_metrics:
            # Normalize metrics for fair comparison
            max_values = [max(anon_metrics[i], ml_metrics[i]) for i in range(min(len(anon_metrics), len(ml_metrics)))]
            
            anon_normalized = [anon_metrics[i] / max(max_values[i], 0.001) for i in range(len(anon_metrics))]
            ml_normalized = [ml_metrics[i] / max(max_values[i], 0.001) for i in range(len(ml_metrics))]
            
            anon_score = np.mean(anon_normalized) * 100
            ml_score = np.mean(ml_normalized) * 100
            
            bars = ax4.bar(platforms, [anon_score, ml_score], color=['skyblue', 'lightcoral'], alpha=0.7)
            ax4.set_title('Overall Performance Score', fontsize=12, fontweight='bold')
            ax4.set_ylabel('Performance Score (%)')
            ax4.grid(True, alpha=0.3)
            
            # Add value labels
            for bar, score in zip(bars, [anon_score, ml_score]):
                ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                        f'{score:.1f}%', ha='center', va='bottom', fontweight='bold')
        
        plt.suptitle('Platform Performance Dashboard', fontsize=16, fontweight='bold')
        plt.tight_layout()
        filename = f'platform_performance_dashboard_{timestamp}.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"📊 Performance dashboard saved: {filename}")
    
    def generate_comparison_report(self):
        """Generate comprehensive comparison report."""
        print("📋 Generating platform comparison report...")
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        report_file = f"platform_comparison_report_{timestamp}.txt"
        
        with open(report_file, 'w') as f:
            f.write("PLATFORM PERFORMANCE COMPARISON REPORT\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Test Level: {self.test_level.upper()}\n\n")
            
            # Anonymization Platform Summary
            f.write("ANONYMIZATION PLATFORM SUMMARY\n")
            f.write("-" * 35 + "\n")
            if self.results['anonymization_results']:
                anon_data = self.results['comparison_metrics']
                f.write(f"Algorithms tested: {len(anon_data['execution_times']['anonymization'])}\n")
                if anon_data['execution_times']['anonymization']:
                    f.write(f"Average execution time: {np.mean(anon_data['execution_times']['anonymization']):.2f}s\n")
                if anon_data['memory_usage']['anonymization']:
                    f.write(f"Average memory usage: {np.mean(anon_data['memory_usage']['anonymization']) / (1024**2):.1f} MB\n")
                if anon_data['throughput']['anonymization']:
                    f.write(f"Average throughput: {np.mean(anon_data['throughput']['anonymization']):.1f} records/sec\n")
                if anon_data['cpu_efficiency']['anonymization']:
                    f.write(f"Average CPU efficiency: {np.mean(anon_data['cpu_efficiency']['anonymization']) * 100:.1f}%\n")
            
            # ML Platform Summary
            f.write("\nML PLATFORM SUMMARY\n")
            f.write("-" * 20 + "\n")
            if self.results['ml_results']:
                ml_data = self.results['comparison_metrics']
                f.write(f"Algorithms tested: {len(ml_data['execution_times']['ml'])}\n")
                if ml_data['execution_times']['ml']:
                    f.write(f"Average execution time: {np.mean(ml_data['execution_times']['ml']):.2f}s\n")
                if ml_data['memory_usage']['ml']:
                    f.write(f"Average memory usage: {np.mean(ml_data['memory_usage']['ml']) / (1024**2):.1f} MB\n")
                if ml_data['throughput']['ml']:
                    f.write(f"Average throughput: {np.mean(ml_data['throughput']['ml']):.1f} records/sec\n")
                if ml_data['cpu_efficiency']['ml']:
                    f.write(f"Average CPU efficiency: {np.mean(ml_data['cpu_efficiency']['ml']) * 100:.1f}%\n")
            
            # Performance Comparison
            f.write("\nPERFORMANCE COMPARISON\n")
            f.write("-" * 25 + "\n")
            if (self.results['anonymization_results'] and self.results['ml_results'] and 
                self.results['comparison_metrics']):
                
                comp_data = self.results['comparison_metrics']
                
                # Execution time comparison
                if (comp_data['execution_times']['anonymization'] and 
                    comp_data['execution_times']['ml']):
                    anon_exec = np.mean(comp_data['execution_times']['anonymization'])
                    ml_exec = np.mean(comp_data['execution_times']['ml'])
                    if anon_exec < ml_exec:
                        f.write(f"Execution Time Winner: Anonymization ({anon_exec:.2f}s vs {ml_exec:.2f}s)\n")
                    else:
                        f.write(f"Execution Time Winner: ML Platform ({ml_exec:.2f}s vs {anon_exec:.2f}s)\n")
                
                # Memory comparison
                if (comp_data['memory_usage']['anonymization'] and 
                    comp_data['memory_usage']['ml']):
                    anon_mem = np.mean(comp_data['memory_usage']['anonymization']) / (1024**2)
                    ml_mem = np.mean(comp_data['memory_usage']['ml']) / (1024**2)
                    if anon_mem < ml_mem:
                        f.write(f"Memory Efficiency Winner: Anonymization ({anon_mem:.1f}MB vs {ml_mem:.1f}MB)\n")
                    else:
                        f.write(f"Memory Efficiency Winner: ML Platform ({ml_mem:.1f}MB vs {anon_mem:.1f}MB)\n")
                
                # Throughput comparison
                if (comp_data['throughput']['anonymization'] and 
                    comp_data['throughput']['ml']):
                    anon_through = np.mean(comp_data['throughput']['anonymization'])
                    ml_through = np.mean(comp_data['throughput']['ml'])
                    if anon_through > ml_through:
                        f.write(f"Throughput Winner: Anonymization ({anon_through:.1f} vs {ml_through:.1f} rec/s)\n")
                    else:
                        f.write(f"Throughput Winner: ML Platform ({ml_through:.1f} vs {anon_through:.1f} rec/s)\n")
            
            # Recommendations
            f.write("\nRECOMMENDATIONS\n")
            f.write("-" * 15 + "\n")
            f.write("• Both platforms show distinct performance characteristics\n")
            f.write("• Consider workload-specific platform selection\n")
            f.write("• Monitor resource usage for optimal deployment\n")
            f.write("• Regular performance testing recommended\n")
        
        print(f"📄 Platform comparison report saved to: {report_file}")
    
    def run_comparison(self):
        """Run complete platform comparison."""
        print("🔄 Starting Platform Performance Comparison")
        print("=" * 50)
        print(f"📊 Test Level: {self.test_level.upper()}")
        
        self.run_anonymization_tests()
        self.run_ml_tests()
        self.extract_metrics_for_comparison()
        self.create_comparison_visualizations()
        self.generate_comparison_report()
        
        print("\n🎉 Platform comparison completed successfully!")


def get_user_preferences():
    """Get user preferences for comparison level."""
    print("⚖️ Platform Performance Comparison Tool")
    print("=" * 40)
    
    print("\nSelect comparison level:")
    print("1. 🚀 Simple (Fast, basic comparison)")
    print("2. 📊 Medium (Balanced, comprehensive)")
    print("3. 🔬 Full (Detailed, extensive analysis)")
    
    while True:
        try:
            choice = input("\nEnter choice (1-3): ").strip()
            if choice == '1':
                return 'simple'
            elif choice == '2':
                return 'medium'
            elif choice == '3':
                return 'full'
            else:
                print("❌ Invalid choice. Please enter 1, 2, or 3.")
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            sys.exit(0)


def main():
    """Main function to run platform comparison."""
    test_level = get_user_preferences()
    
    comparator = PlatformComparator(test_level=test_level)
    comparator.run_comparison()


if __name__ == "__main__":
    main()
