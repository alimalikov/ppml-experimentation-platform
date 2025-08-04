# Performance Visualization Charts - Updated Features
## Anonymization Platform Performance Analysis

### Overview
The comprehensive performance test now generates **6 separate PNG files** instead of one combined visualization, making it easier to use individual charts in your thesis or presentations.

### Generated Charts

#### 1. **execution_time_distribution_by_technique.png**
- **Type**: Box plot
- **Shows**: Distribution of execution times for each anonymization technique
- **Purpose**: Compare technique consistency and identify outliers
- **Use in thesis**: Technique comparison sections

#### 2. **memory_usage_vs_dataset_size.png**
- **Type**: Scatter plot with color-coded techniques
- **Shows**: Memory consumption patterns as dataset size increases
- **Purpose**: Understand memory scalability of different techniques
- **Use in thesis**: Resource usage analysis

#### 3. **average_throughput_by_technique.png**
- **Type**: Horizontal bar chart
- **Shows**: Average processing speed (rows per second) by technique
- **Purpose**: Identify fastest techniques for real-time applications
- **Use in thesis**: Performance comparison tables

#### 4. **execution_time_heatmap_seconds.png**
- **Type**: Heatmap
- **Shows**: Execution time patterns across datasets and techniques
- **Purpose**: Quick visual comparison of technique performance
- **Use in thesis**: Performance overview sections

#### 5. **cpu_usage_analysis.png** ⭐ **NEW**
- **Type**: Dual subplot chart
- **Shows**: 
  - Top: CPU time vs Wall time comparison
  - Bottom: CPU efficiency percentage by technique
- **Purpose**: Analyze computational efficiency and parallelization
- **Use in thesis**: Technical performance analysis

#### 6. **scalability_analysis_by_dataset_size.png**
- **Type**: Multi-line plot (3 subplots)
- **Shows**: How techniques scale with dataset size
  - Execution time scaling
  - Memory usage scaling  
  - Throughput scaling
- **Purpose**: Understand scalability characteristics
- **Use in thesis**: Scalability analysis sections

### Chart Quality & Features

#### Technical Specifications
- **Resolution**: 300 DPI (publication quality)
- **Format**: PNG with transparent backgrounds
- **Size**: Optimized for both digital and print use
- **Color scheme**: Professional, colorblind-friendly palette

#### Enhanced Visual Features
- **Clear titles** with bold formatting
- **Axis labels** with proper units
- **Legends** positioned to avoid overlap
- **Grid lines** for easier reading
- **Value annotations** on bar charts
- **Color coding** for categorical data

### CPU Usage Analysis Details

The new CPU usage chart provides insights into:

#### CPU vs Wall Time Comparison
- **Wall Time**: Total elapsed time (includes I/O, waiting)
- **CPU Time**: Actual processor computation time
- **Interpretation**: 
  - Similar values = CPU-bound operation
  - Wall time >> CPU time = I/O-bound or parallelizable

#### CPU Efficiency Metrics
- **Formula**: (CPU Time / Wall Time) × 100%
- **Color coding**:
  - 🟢 Green (>80%): Highly CPU-efficient
  - 🟠 Orange (60-80%): Moderately efficient
  - 🔴 Red (<60%): I/O-bound or inefficient

### File Naming Convention

All charts follow a descriptive naming pattern:
```
{description}_{metric/type}.png
```

Examples:
- `execution_time_distribution_by_technique.png`
- `memory_usage_vs_dataset_size.png`
- `cpu_usage_analysis.png`

### Usage in Academic Writing

#### For Thesis Chapters
1. **Introduction**: Use `average_throughput_by_technique.png`
2. **Methodology**: Use `scalability_analysis_by_dataset_size.png`
3. **Results**: Use `execution_time_heatmap_seconds.png`
4. **Analysis**: Use `cpu_usage_analysis.png`
5. **Discussion**: Use `execution_time_distribution_by_technique.png`

#### For Presentations
- Each chart is slide-ready at high resolution
- Clear titles make them self-explanatory
- Color schemes work well with projectors

#### For Papers/Reports
- 300 DPI ensures crisp printing
- Professional color palette
- Consistent styling across all charts

### How to Generate

#### Option 1: Complete Test
```bash
python performance_tester.py
```
Select your dataset sizes and the test will generate all charts.

#### Option 2: Via Menu
```bash
python run_tests.py
```
Choose option 2 (Comprehensive Performance Test).

### Output Location

All charts are saved in the `performance_results/` directory with timestamps:
```
performance_results/
├── execution_time_distribution_by_technique.png
├── memory_usage_vs_dataset_size.png
├── average_throughput_by_technique.png
├── execution_time_heatmap_seconds.png
├── cpu_usage_analysis.png
├── scalability_analysis_by_dataset_size.png
├── performance_results_20250720_143022.json
└── performance_report_20250720_143022.md
```

### Tips for Thesis Use

#### Best Practices
1. **Reference charts by filename** in your thesis
2. **Include chart numbers** in captions
3. **Explain the color coding** in figure descriptions
4. **Mention dataset sizes** used for context

#### Sample Figure Captions
```
Figure 3.1: Execution time distribution by anonymization technique 
(execution_time_distribution_by_technique.png). Box plots show median, 
quartiles, and outliers across all tested dataset sizes.

Figure 4.2: CPU usage analysis comparing computational efficiency 
(cpu_usage_analysis.png). Top panel shows CPU vs wall time, bottom 
panel shows efficiency percentages with color coding.
```

The separate chart files give you maximum flexibility for creating professional thesis documentation! 📊✨
