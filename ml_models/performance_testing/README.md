# Performance Testing Suite for Anonymization Platform

This directory contains a comprehensive performance testing framework for your anonymization platform. The suite evaluates execution time, memory usage, scalability, robustness, and data quality preservation across all anonymization techniques.

## 📁 Files Overview

### Core Testing Scripts

- **`run_tests.py`** - Main entry point with interactive menu
- **`quick_benchmark.py`** - Fast performance benchmarking (2-5 minutes)
- **`performance_tester.py`** - Comprehensive performance analysis (10-30 minutes)
- **`stress_tester.py`** - Edge case and robustness testing (5-15 minutes)

### Generated Outputs

- **`performance_results/`** - Detailed performance reports and charts
- **`stress_test_report_*.txt`** - Stress testing analysis reports

## 🚀 Quick Start

### Option 1: Interactive Menu (Recommended)
```bash
cd ml_models/performance_testing
python run_tests.py
```

### Option 2: Direct Command Line
```bash
# Quick benchmark (fastest)
python run_tests.py --quick

# Comprehensive analysis (most detailed)
python run_tests.py --comprehensive

# Stress testing (robustness)
python run_tests.py --stress

# Compare specific techniques
python run_tests.py --compare

# List available techniques
python run_tests.py --list
```

## 📊 Testing Types

### 1. Quick Benchmark (`quick_benchmark.py`)
**Duration:** 2-5 minutes  
**Purpose:** Fast performance overview

**What it tests:**
- Execution time for small (1K), medium (10K), and large (50K) datasets
- Throughput (rows processed per second)
- Success/failure rates
- Top performing techniques

**Output:**
- Console display with performance rankings
- Immediate results for quick comparison

**Example:**
```bash
python quick_benchmark.py                    # Test all techniques
python quick_benchmark.py "K-Anonymity Simple"  # Test single technique
python quick_benchmark.py "K-Anonymity Simple" "Differential Privacy Core"  # Compare two
```

### 2. Comprehensive Performance Test (`performance_tester.py`)
**Duration:** 10-30 minutes  
**Purpose:** Detailed performance analysis with visualizations

**What it tests:**
- Multiple dataset types (numeric, categorical, mixed, high-cardinality)
- Various dataset sizes (1K to 100K rows)
- Memory usage tracking
- Scalability analysis
- CPU time vs wall time

**Generated outputs:**
- `performance_results_YYYYMMDD_HHMMSS.json` - Raw performance data
- `performance_report_YYYYMMDD_HHMMSS.md` - Human-readable report
- `performance_visualization.png` - Performance charts
- `scalability_analysis.png` - Scalability trends

**Synthetic datasets tested:**
- **Small (1K):** Quick testing dataset
- **Medium (10K):** Standard performance baseline
- **Large (100K):** Scalability testing
- **Wide (50 cols):** Column-heavy dataset
- **High cardinality:** Unique identifier heavy
- **Mixed types:** Various data types together

### 3. Stress Testing (`stress_tester.py`)
**Duration:** 5-15 minutes  
**Purpose:** Test robustness and edge case handling

**Edge cases tested:**
- Empty datasets
- Single row/column datasets
- All missing values
- Extreme numeric values (infinity, very large/small)
- High cardinality data (unique values)
- Low cardinality data (repeated values)
- Special characters and Unicode
- Very wide datasets (200+ columns)

**Analysis provided:**
- Success/failure rates by dataset type
- Data quality preservation metrics
- Correlation preservation
- Statistical preservation (means, standard deviations)
- Common failure patterns
- Error type analysis

## 📈 Understanding Results

### Performance Metrics

**Execution Time:**
- **Wall Time:** Real-world time elapsed
- **CPU Time:** Actual processor time used
- **Throughput:** Rows processed per second

**Memory Metrics:**
- **Memory Used:** RAM increase during processing
- **Peak Memory:** Maximum memory usage

**Data Quality Metrics:**
- **Correlation Preservation:** How well statistical relationships are maintained (0-100%)
- **Mean Preservation:** How well averages are preserved (0-100%)
- **Data Completeness:** Percentage of original data retained

### Performance Interpretation

**Good Performance Indicators:**
- ✅ **High throughput** (>1000 rows/sec for simple techniques)
- ✅ **Low memory usage** (<100MB for 10K rows)
- ✅ **Consistent performance** across dataset sizes
- ✅ **High data quality preservation** (>80%)

**Performance Concerns:**
- ⚠️ **Very slow execution** (<10 rows/sec)
- ⚠️ **High memory usage** (>1GB for small datasets)
- ⚠️ **Poor scalability** (exponential time increase)
- ⚠️ **Low data quality** (<50% preservation)

## 🛠️ Customization

### Adding Custom Test Datasets

Edit the dataset generation functions in the testing scripts:

```python
# In performance_tester.py
def _create_synthetic_dataset(self, n_rows: int, n_cols: int, complexity: str):
    # Add your custom dataset generation logic
    pass

# In stress_tester.py  
def create_edge_case_datasets(self):
    # Add your custom edge cases
    datasets['my_custom_case'] = your_dataframe
    return datasets
```

### Testing Specific Scenarios

Create focused test scripts for your specific use cases:

```python
from performance_testing.quick_benchmark import quick_benchmark, create_test_dataset
from core.app import load_anonymizer_plugins, ANONYMIZER_PLUGINS

# Load plugins
load_anonymizer_plugins()

# Create your specific test scenario
test_df = create_test_dataset("large")  # or your custom dataset

# Test specific technique
technique_name = "Your Technique"
plugin = ANONYMIZER_PLUGINS[technique_name]
result = quick_benchmark(technique_name, plugin, test_df)

print(f"Result: {result}")
```

## 📋 Dependencies

### Required
- `pandas` - Data manipulation
- `numpy` - Numerical operations

### Optional (for full features)
- `matplotlib` - Chart generation
- `seaborn` - Advanced visualizations  
- `psutil` - System resource monitoring

Install all dependencies:
```bash
pip install pandas numpy matplotlib seaborn psutil
```

## 🔧 Troubleshooting

### Common Issues

**"ImportError: Could not import Anonymizer base class"**
- Ensure you're running from the `ml_models` directory
- Check that `src/anonymizers/base_anonymizer.py` exists

**"No anonymization plugins loaded"**
- Verify plugins exist in `src/anonymizers/plugins/`
- Check plugin files have correct structure and `get_plugin()` function

**"Performance test takes too long"**
- Use `--quick` option for faster results
- Reduce dataset sizes in the scripts
- Test fewer techniques by modifying plugin selection

**"Charts not generated"**
- Install matplotlib and seaborn: `pip install matplotlib seaborn`
- Check write permissions in the performance_testing directory

**"Memory errors during testing"**
- Reduce dataset sizes in the configuration
- Close other applications to free memory
- Test techniques individually rather than all at once

### Performance Optimization

**For faster testing:**
1. Use smaller dataset sizes
2. Test fewer techniques at once
3. Skip comprehensive tests and use quick benchmark
4. Disable chart generation if not needed

**For more detailed analysis:**
1. Increase dataset sizes and variety
2. Add more iterations for statistical significance
3. Enable all visualizations
4. Run tests on dedicated hardware

## 📊 Example Results Interpretation

### Quick Benchmark Output
```
📈 Testing with medium dataset...
   Dataset shape: (10000, 10)
   ⚙️  Testing K-Anonymity Simple... ✅ 0.234s (42735 rows/s)
   ⚙️  Testing Differential Privacy Core... ✅ 0.156s (64103 rows/s)
   ⚙️  Testing L-Diversity Core... ✅ 0.445s (22472 rows/s)

   🏆 Top 3 fastest for medium dataset:
      1. Differential Privacy Core: 0.156s
      2. K-Anonymity Simple: 0.234s  
      3. L-Diversity Core: 0.445s
```

**Interpretation:** Differential Privacy Core is fastest, processing ~64K rows/second, while L-Diversity takes twice as long.

### Stress Test Insights
```
EDGE CASE TEST RESULTS
empty:
  Success Rate: 12/15 techniques (80.0%)
high_cardinality:
  Success Rate: 8/15 techniques (53.3%)
```

**Interpretation:** Most techniques handle empty datasets well, but high cardinality data is challenging for many techniques.

## 💡 Best Practices

1. **Run quick benchmark first** to identify obvious issues
2. **Use comprehensive testing** before production deployment
3. **Regular stress testing** to catch regressions
4. **Compare techniques** for specific use cases
5. **Monitor trends** over time as you add features
6. **Test with realistic data** that matches your actual use cases

## 📚 Academic Usage

For thesis documentation, these tests provide:

- **Quantitative performance comparison** between techniques
- **Scalability analysis** showing computational complexity
- **Robustness evaluation** for production readiness  
- **Data utility preservation** metrics for privacy-utility tradeoffs
- **Comprehensive benchmarking** across multiple dimensions

The generated reports and charts can be directly used in academic papers to demonstrate the platform's performance characteristics and comparative advantages of different anonymization approaches.
