# Dataset Size Selection Guide
## Performance Testing Framework - Updated Features

### Overview
Both the **Quick Benchmark** and **Comprehensive Performance Test** now support the same dataset size selection options, giving you full control over testing scope and execution time.

### Available Dataset Sizes

| Size Option | Rows | Description | Execution Time | Use Case |
|-------------|------|-------------|----------------|----------|
| **🐣 Tiny** | 500 | Ultra fast testing | 30 seconds - 2 minutes | Quick validation, debugging |
| **🚀 Small** | 1,000 | Fast testing | 1-3 minutes | Development testing |
| **📊 Medium** | 10,000 | Standard testing | 5-10 minutes | Regular performance checks |
| **📈 Large** | 50,000 | Comprehensive testing | 15-25 minutes | Thesis benchmarking |
| **🏭 Huge** | 100,000 | Stress testing | 30-45 minutes | Maximum performance analysis |
| **🌈 All Sizes** | All of above | Complete benchmark | 45-60 minutes | Full academic analysis |
| **🎯 Custom** | User choice | Select multiple sizes | Variable | Flexible testing |

### Quick Benchmark vs Comprehensive Performance Test

#### Quick Benchmark (`quick_benchmark.py`)
- **Purpose**: Fast performance overview
- **Features**: 
  - Dataset size selection (1-6 options)
  - Basic execution time and throughput
  - Terminal output logging
  - Top 3 fastest techniques per size
- **Best for**: Quick checks, development, time-constrained testing

#### Comprehensive Performance Test (`performance_tester.py`)
- **Purpose**: Detailed academic-quality analysis
- **Features**:
  - Same dataset size selection (1-7 options including custom)
  - Memory usage tracking
  - Scalability analysis
  - Data quality preservation metrics
  - Visualization generation
  - Detailed reports (JSON + Markdown)
  - Terminal output logging
- **Best for**: Thesis documentation, research, academic papers

### How to Use

#### Option 1: Interactive Menu
```bash
python run_tests.py
```
Choose option 1 (Quick) or 2 (Comprehensive), then select your dataset sizes.

#### Option 2: Direct Execution
```bash
# Quick Benchmark
python quick_benchmark.py

# Comprehensive Performance Test  
python performance_tester.py
```

### Dataset Size Selection Interface

Both tools now provide the same user-friendly interface:

```
Select dataset sizes to test:
1. 🐣 Tiny (500 rows) - Ultra fast
2. 🚀 Small (1K rows) - Fast  
3. 📊 Medium (10K rows) - Standard
4. 📈 Large (50K rows) - Comprehensive
5. 🏭 Huge (100K rows) - Stress test
6. 🌈 All sizes - Complete benchmark
7. 🎯 Custom selection - Choose multiple sizes (Comprehensive only)

Enter choice (1-6): 
```

### Terminal Output Logging

Both tools ask if you want to save terminal output:
```
Save terminal output to file? (y/N): 
```

If you choose 'y', output is saved to timestamped files:
- Quick: `quick_benchmark_{sizes}_{timestamp}.txt`
- Comprehensive: `comprehensive_performance_test_{sizes}_{timestamp}.txt`

### Recommendations by Use Case

#### For Thesis Writing
- Use **Comprehensive Performance Test** with **Large** or **All sizes**
- Enable terminal output logging for documentation
- Results include academic-quality visualizations and reports

#### For Development & Debugging
- Use **Quick Benchmark** with **Tiny** or **Small**
- Fast feedback for code changes

#### For Regular Performance Monitoring
- Use **Quick Benchmark** with **Medium**
- Quick overview of system performance

#### For Academic Research
- Use **Comprehensive Performance Test** with **All sizes**
- Complete analysis with memory profiling and scalability metrics

### Output Files Generated

#### Quick Benchmark
- Terminal output log (optional)
- Console summary of results

#### Comprehensive Performance Test  
- Terminal output log (optional)
- `performance_results_{timestamp}.json` - Raw results
- `performance_report_{timestamp}.md` - Readable report
- Visualization charts (PNG files)
- Memory usage graphs

### Technical Details

#### Dataset Characteristics by Size
- **Tiny/Small**: Simple data types, basic correlations
- **Medium**: Mixed data types, moderate complexity
- **Large/Huge**: Complex relationships, high cardinality data
- **Mixed Types**: Special dataset with comprehensive data type coverage

#### Memory Requirements
- **Tiny**: ~5 MB RAM
- **Small**: ~10 MB RAM  
- **Medium**: ~100 MB RAM
- **Large**: ~500 MB RAM
- **Huge**: ~1 GB RAM

Choose dataset sizes based on your available time, system resources, and analysis needs.
