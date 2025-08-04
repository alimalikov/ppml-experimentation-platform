"""
Stress Testing Suite for Anonymization Platform
==============================================

Tests edge cases, robustness, and failure scenarios for anonymization techniques.
Helps identify potential issues and limitations of the platform.

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

try:
    from core.app import load_anonymizer_plugins, ANONYMIZER_PLUGINS
except ImportError as e:
    print(f"Error importing: {e}")
    sys.exit(1)

class StressTester:
    """
    Stress testing framework for anonymization techniques.
    """
    
    def __init__(self, test_level='full', output_file=None):
        self.test_level = test_level  # 'simple', 'medium', 'full'
        self.output_file = output_file
        self.load_plugins()
        self.results = {
            'timestamp': datetime.now().isoformat(),
            'test_level': test_level,
            'edge_case_tests': {},
            'robustness_tests': {},
            'data_quality_tests': {},
            'failure_analysis': {}
        }
    
    def load_plugins(self):
        """Load anonymization plugins."""
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
        print(f"🔧 Loaded {len(self.plugins)} techniques for stress testing")
        if skipped_count > 0:
            print(f"⏭️  Skipped {skipped_count} homomorphic encryption technique(s)")
    
    def create_edge_case_datasets(self) -> Dict[str, pd.DataFrame]:
        """Create datasets with edge cases and challenging characteristics."""
        datasets = {}
        
        # Adjust dataset sizes based on test level
        if self.test_level == 'simple':
            small_size, medium_size, large_size = 100, 500, 1000
            wide_cols = 20
            print(f"📊 Using simple dataset sizes: {small_size}-{large_size} rows")
        elif self.test_level == 'medium':
            small_size, medium_size, large_size = 500, 2000, 5000
            wide_cols = 50
            print(f"📊 Using medium dataset sizes: {small_size}-{large_size} rows")
        else:  # full
            small_size, medium_size, large_size = 1000, 5000, 10000
            wide_cols = 200
            print(f"📊 Using full dataset sizes: {small_size}-{large_size} rows")
        
        # 1. Empty dataset
        datasets['empty'] = pd.DataFrame()
        
        # 2. Single row dataset
        datasets['single_row'] = pd.DataFrame({
            'col1': [1],
            'col2': ['A'],
            'sensitive': ['Group_A']
        })
        
        # 3. Single column dataset
        datasets['single_column'] = pd.DataFrame({
            'only_column': range(small_size)
        })
        
        # 4. All missing values
        datasets['all_missing'] = pd.DataFrame({
            'col1': [np.nan] * small_size,
            'col2': [None] * small_size,
            'col3': [np.nan] * small_size
        })
        
        # 5. Mixed missing values
        datasets['mixed_missing'] = pd.DataFrame({
            'complete': range(medium_size),
            'half_missing': [i if i % 2 == 0 else np.nan for i in range(medium_size)],
            'mostly_missing': [i if i % 10 == 0 else np.nan for i in range(medium_size)],
            'sensitive': np.random.choice(['A', 'B', np.nan], medium_size)
        })
        
        # 6. Extreme values
        datasets['extreme_values'] = pd.DataFrame({
            'tiny_numbers': np.random.uniform(-1e-10, 1e-10, medium_size),
            'huge_numbers': np.random.uniform(1e10, 1e15, medium_size),
            'inf_values': [float('inf'), float('-inf')] * (medium_size // 2),
            'zero_values': [0] * medium_size,
            'negative_values': np.random.uniform(-1000, -1, medium_size)
        })
        
        # 7. High cardinality (unique values)
        datasets['high_cardinality'] = pd.DataFrame({
            'unique_ids': [f'id_{i}' for i in range(large_size)],
            'unique_emails': [f'user{i}@domain{i}.com' for i in range(large_size)],
            'timestamps': pd.date_range('2020-01-01', periods=large_size, freq='1min'),
            'sensitive': np.random.choice(['Group_A', 'Group_B'], large_size)
        })
        
        # 8. Low cardinality (repeated values)
        datasets['low_cardinality'] = pd.DataFrame({
            'same_value': ['SAME'] * medium_size,
            'binary': np.random.choice([0, 1], medium_size),
            'three_values': np.random.choice(['A', 'B', 'C'], medium_size),
            'sensitive': ['OnlyGroup'] * medium_size
        })
        
        # 9. Special characters and encoding (only in medium and full)
        if self.test_level != 'simple':
            datasets['special_chars'] = pd.DataFrame({
                'unicode_text': ['ñáéíóú', '中文测试', 'العربية', 'русский'] * (medium_size // 4),
                'special_symbols': ['@#$%^&*()', '{}[]|\\', '<>/?~`'] * (medium_size // 3),
                'quotes_and_spaces': ['"quoted"', "'single'", '  spaces  ', '\t\n'] * (medium_size // 4),
                'sensitive': np.random.choice(['Group_Ñ', 'Group_中'], medium_size)
            })
        
        # 10. Very wide dataset (only in full)
        if self.test_level == 'full':
            wide_data = {}
            for i in range(wide_cols):  # Variable number of columns
                wide_data[f'col_{i}'] = np.random.normal(0, 1, medium_size)
            wide_data['sensitive'] = np.random.choice(['A', 'B'], medium_size)
            datasets['very_wide'] = pd.DataFrame(wide_data)
        
        return datasets
    
    def test_edge_cases(self):
        """Test how techniques handle edge cases."""
        print("\n🧪 Testing edge cases...")
        
        edge_datasets = self.create_edge_case_datasets()
        
        for dataset_name, df in edge_datasets.items():
            print(f"  📊 Testing {dataset_name} dataset...")
            
            self.results['edge_case_tests'][dataset_name] = {
                'dataset_info': {
                    'shape': df.shape if not df.empty else (0, 0),
                    'memory_usage': df.memory_usage(deep=True).sum() if not df.empty else 0,
                    'has_missing': df.isnull().any().any() if not df.empty else False
                },
                'technique_results': {}
            }
            
            for technique_name, plugin in self.plugins.items():
                try:
                    # Determine sensitive attribute
                    sa_col = 'sensitive' if 'sensitive' in df.columns else None
                    
                    # Run anonymization
                    result_df = plugin.anonymize(df.copy(), {}, sa_col)
                    
                    # Analyze result
                    test_result = {
                        'success': True,
                        'result_shape': result_df.shape,
                        'preserves_rows': len(result_df) == len(df) if not df.empty else True,
                        'result_empty': result_df.empty,
                        'has_errors': False,
                        'error_message': None
                    }
                    
                except Exception as e:
                    test_result = {
                        'success': False,
                        'has_errors': True,
                        'error_message': str(e),
                        'error_type': type(e).__name__
                    }
                
                self.results['edge_case_tests'][dataset_name]['technique_results'][technique_name] = test_result
        
        print("✅ Edge case testing completed")
    
    def test_data_quality_preservation(self):
        """Test how well techniques preserve data quality and relationships."""
        print("\n🔍 Testing data quality preservation...")
        
        # Create test dataset with known relationships
        np.random.seed(42)
        
        # Adjust dataset size based on test level
        if self.test_level == 'simple':
            n_rows = 1000
        elif self.test_level == 'medium':
            n_rows = 3000
        else:  # full
            n_rows = 5000
        
        print(f"📊 Using {n_rows} rows for data quality testing")
        
        # Create correlated data
        base_income = np.random.normal(50000, 15000, n_rows)
        df = pd.DataFrame({
            'age': np.random.randint(22, 65, n_rows),
            'income': base_income,
            'education_years': np.random.randint(12, 20, n_rows),
            'experience_years': np.maximum(0, np.random.normal(10, 5, n_rows)),
            'city_size': np.random.choice(['Small', 'Medium', 'Large'], n_rows),
            'sensitive_group': np.random.choice(['Group_A', 'Group_B', 'Group_C'], n_rows)
        })
        
        # Add some correlations
        df.loc[df['education_years'] >= 16, 'income'] *= 1.3  # Higher education = higher income
        df.loc[df['city_size'] == 'Large', 'income'] *= 1.2   # Large city = higher income
        
        # Calculate original statistics
        original_stats = {
            'correlations': df.corr(numeric_only=True).to_dict(),
            'means': df.mean(numeric_only=True).to_dict(),
            'stds': df.std(numeric_only=True).to_dict(),
            'value_counts': {col: df[col].value_counts().to_dict() 
                           for col in df.select_dtypes(include=['object']).columns}
        }
        
        for technique_name, plugin in self.plugins.items():
            print(f"  ⚙️  Testing {technique_name}...")
            
            try:
                result_df = plugin.anonymize(df.copy(), {}, 'sensitive_group')
                
                # Calculate anonymized statistics
                if not result_df.empty and len(result_df) > 0:
                    anon_stats = {
                        'correlations': result_df.corr(numeric_only=True).to_dict(),
                        'means': result_df.mean(numeric_only=True).to_dict(),
                        'stds': result_df.std(numeric_only=True).to_dict()
                    }
                    
                    # Calculate preservation metrics
                    correlation_preservation = self._calculate_correlation_preservation(
                        original_stats['correlations'], anon_stats['correlations']
                    )
                    
                    mean_preservation = self._calculate_statistical_preservation(
                        original_stats['means'], anon_stats['means']
                    )
                    
                    std_preservation = self._calculate_statistical_preservation(
                        original_stats['stds'], anon_stats['stds']
                    )
                    
                    quality_result = {
                        'success': True,
                        'correlation_preservation': correlation_preservation,
                        'mean_preservation': mean_preservation,
                        'std_preservation': std_preservation,
                        'data_completeness': len(result_df) / len(df),
                        'column_preservation': len(result_df.columns) / len(df.columns)
                    }
                else:
                    quality_result = {
                        'success': False,
                        'error': 'Empty result dataset'
                    }
                
            except Exception as e:
                quality_result = {
                    'success': False,
                    'error': str(e)
                }
            
            self.results['data_quality_tests'][technique_name] = quality_result
        
        print("✅ Data quality testing completed")
    
    def test_robustness(self):
        """Test robustness with various parameter configurations."""
        print("\n💪 Testing robustness...")
        
        # Adjust dataset size based on test level
        if self.test_level == 'simple':
            n_rows = 500
        elif self.test_level == 'medium':
            n_rows = 1500
        else:  # full
            n_rows = 1000
        
        print(f"📊 Using {n_rows} rows for robustness testing")
        
        # Create standard test dataset
        df = pd.DataFrame({
            'numeric1': np.random.normal(100, 15, n_rows),
            'numeric2': np.random.uniform(0, 1000, n_rows),
            'category1': np.random.choice(['A', 'B', 'C', 'D'], n_rows),
            'category2': np.random.choice([f'Item_{i}' for i in range(20)], n_rows),
            'sensitive': np.random.choice(['Group_A', 'Group_B'], n_rows)
        })
        
        # Test with different parameter combinations
        parameter_tests = [
            {},  # Empty parameters
            {'k': 5} if 'k' in str(self.plugins) else {},  # Common k-anonymity parameter
            {'epsilon': 1.0} if 'epsilon' in str(self.plugins) else {},  # DP parameter
        ]
        
        for technique_name, plugin in self.plugins.items():
            print(f"  🔧 Testing {technique_name} robustness...")
            
            technique_results = []
            
            for i, params in enumerate(parameter_tests):
                try:
                    result_df = plugin.anonymize(df.copy(), params, 'sensitive')
                    
                    test_result = {
                        'parameter_set': i,
                        'parameters': params,
                        'success': True,
                        'result_shape': result_df.shape,
                        'execution_successful': True
                    }
                    
                except Exception as e:
                    test_result = {
                        'parameter_set': i,
                        'parameters': params,
                        'success': False,
                        'error': str(e),
                        'error_type': type(e).__name__
                    }
                
                technique_results.append(test_result)
            
            self.results['robustness_tests'][technique_name] = technique_results
        
        print("✅ Robustness testing completed")
    
    def _calculate_correlation_preservation(self, orig_corr: dict, anon_corr: dict) -> float:
        """Calculate how well correlations are preserved."""
        if not orig_corr or not anon_corr:
            return 0.0
        
        total_diff = 0
        count = 0
        
        for col1 in orig_corr:
            if col1 in anon_corr:
                for col2 in orig_corr[col1]:
                    if col2 in anon_corr[col1]:
                        if not (pd.isna(orig_corr[col1][col2]) or pd.isna(anon_corr[col1][col2])):
                            diff = abs(orig_corr[col1][col2] - anon_corr[col1][col2])
                            total_diff += diff
                            count += 1
        
        if count == 0:
            return 0.0
        
        avg_diff = total_diff / count
        return max(0, 1 - avg_diff)  # Convert to preservation score
    
    def _calculate_statistical_preservation(self, orig_stats: dict, anon_stats: dict) -> float:
        """Calculate statistical preservation score."""
        if not orig_stats or not anon_stats:
            return 0.0
        
        total_diff = 0
        count = 0
        
        for col in orig_stats:
            if col in anon_stats:
                orig_val = orig_stats[col]
                anon_val = anon_stats[col]
                
                if not (pd.isna(orig_val) or pd.isna(anon_val)):
                    if orig_val != 0:
                        relative_diff = abs(orig_val - anon_val) / abs(orig_val)
                        total_diff += relative_diff
                        count += 1
        
        if count == 0:
            return 0.0
        
        avg_relative_diff = total_diff / count
        return max(0, 1 - avg_relative_diff)
    
    def analyze_failures(self):
        """Analyze failure patterns across all tests."""
        print("\n📊 Analyzing failure patterns...")
        
        failure_analysis = {
            'total_tests': 0,
            'total_failures': 0,
            'failure_by_technique': {},
            'failure_by_dataset_type': {},
            'common_error_types': {}
        }
        
        # Analyze edge case failures
        for dataset_name, dataset_results in self.results['edge_case_tests'].items():
            for technique_name, result in dataset_results['technique_results'].items():
                failure_analysis['total_tests'] += 1
                
                if not result.get('success', False):
                    failure_analysis['total_failures'] += 1
                    
                    # Track by technique
                    if technique_name not in failure_analysis['failure_by_technique']:
                        failure_analysis['failure_by_technique'][technique_name] = 0
                    failure_analysis['failure_by_technique'][technique_name] += 1
                    
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
        print("\n📋 Generating stress test report...")
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        report_file = f"stress_test_report_{timestamp}.txt"
        
        with open(report_file, 'w') as f:
            f.write("ANONYMIZATION PLATFORM STRESS TEST REPORT\n")
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
                success_count = sum(1 for r in results['technique_results'].values() if r.get('success'))
                total_count = len(results['technique_results'])
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
            
            # Most problematic techniques
            f.write("\nTECHNIQUES WITH MOST FAILURES\n")
            f.write("-" * 35 + "\n")
            if 'failure_by_technique' in failure_analysis:
                sorted_tech_failures = sorted(
                    failure_analysis['failure_by_technique'].items(),
                    key=lambda x: x[1], reverse=True
                )
                for technique, failures in sorted_tech_failures[:5]:
                    f.write(f"  {technique}: {failures} failures\n")
            
            # Data quality results
            f.write("\nDATA QUALITY PRESERVATION\n")
            f.write("-" * 30 + "\n")
            for technique, quality in self.results['data_quality_tests'].items():
                if quality.get('success'):
                    f.write(f"{technique}:\n")
                    f.write(f"  Correlation Preservation: {quality.get('correlation_preservation', 0)*100:.1f}%\n")
                    f.write(f"  Mean Preservation: {quality.get('mean_preservation', 0)*100:.1f}%\n")
                    f.write(f"  Data Completeness: {quality.get('data_completeness', 0)*100:.1f}%\n")
        
        print(f"📄 Stress test report saved to: {report_file}")
    
    def run_all_tests(self):
        """Run all stress tests."""
        print("🔬 Starting Comprehensive Stress Testing")
        print("=" * 50)
        print(f"📊 Test Level: {self.test_level.upper()}")
        
        self.test_edge_cases()
        self.test_data_quality_preservation()
        self.test_robustness()
        self.analyze_failures()
        self.generate_stress_test_report()
        
        print("\n🎉 Stress testing completed successfully!")


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
    print("🔬 Anonymization Platform Stress Testing")
    print("=" * 50)
    
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
    """Main function to run stress tests."""
    test_level, save_output = get_user_preferences()
    
    # Set up terminal logging if requested
    terminal_logger = None
    if save_output:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_filename = f"stress_test_terminal_output_{test_level}_{timestamp}.txt"
        terminal_logger = TerminalLogger(log_filename)
        sys.stdout = terminal_logger
        print(f"📝 Terminal output will be saved to: {log_filename}")
    
    try:
        tester = StressTester(test_level=test_level)
        tester.run_all_tests()
    
    finally:
        # Restore original stdout and close log file
        if terminal_logger:
            sys.stdout = terminal_logger.terminal
            terminal_logger.close()
            print(f"📄 Terminal output saved to: {log_filename}")


if __name__ == "__main__":
    main()
