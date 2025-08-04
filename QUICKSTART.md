# Quick Start Guide

Get up and running with the Modular Dataset Anonymization Tool in minutes!

## 🚀 Installation

### Option 1: Direct Installation
```bash
# Clone the repository
git clone <repository-url>
cd bachelor-thesis-anonymization

# Install dependencies
pip install -r requirements.txt

# Run the application
streamlit run ml_models/core/app.py
```

### Option 2: Virtual Environment (Recommended)
```bash
# Clone and navigate
git clone <repository-url>
cd bachelor-thesis-anonymization

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Launch application
streamlit run ml_models/core/app.py
```

## 📊 First Steps

### 1. Upload Your Data
- Click "📁 Upload Your Own Dataset"
- Select a CSV or Excel file
- View the data preview and statistics

### 2. Choose Anonymization Technique
- Browse categories in the sidebar
- Select a technique (e.g., "Basic Redaction")
- Configure parameters as needed

### 3. Apply Anonymization
- Click the "Anonymize" button
- Review the anonymized data preview
- Download results as Excel file

### 4. Evaluate Performance (Optional)
- Switch to ML evaluation mode
- Compare model performance before/after anonymization
- View detailed metrics and visualizations

## 🎲 Try Sample Data

Don't have data ready? Use built-in samples:

1. **Select Category**: Choose from Healthcare, Financial, etc.
2. **Pick Dataset**: Select from available options
3. **Choose Size**: Sample, Medium, or Full scale
4. **Load Data**: Click "📥 Load" button

Popular samples:
- 🏥 **Medical Records**: Healthcare data with HIPAA considerations
- 💰 **Financial Transactions**: Banking data with PCI-DSS requirements
- 👥 **Customer Data**: CRM data with GDPR implications

## 🔧 Plugin Development

Want to create custom anonymization techniques?

1. **Enable Developer Mode** in sidebar
2. **Open Plugin Editor** 
3. **Use Code Snippets** for templates
4. **Test Your Plugin** in real-time
5. **Save to File System** when ready

## 📈 Performance Analysis

### Quick Evaluation
```bash
# Run performance benchmark
python ml_models/performance_testing/quick_benchmark.py
```

### Comprehensive Analysis
```bash
# Full ML impact assessment
python ml_models/ml_performance_testing/ml_performance_tester.py
```

## 🎯 Common Use Cases

### Privacy Compliance
- **GDPR**: Use generalization and suppression techniques
- **HIPAA**: Apply medical data redaction methods  
- **PCI-DSS**: Implement financial data masking

### Research & Development
- **Algorithm Testing**: Compare anonymization techniques
- **Performance Analysis**: Measure privacy-utility tradeoffs
- **Custom Methods**: Develop new anonymization approaches

### Data Sharing
- **Safe Publication**: Anonymize before sharing datasets
- **Collaboration**: Share data while preserving privacy
- **Analytics**: Enable analysis on anonymized data

## 🆘 Troubleshooting

### Common Issues

**Application won't start:**
```bash
# Check Python version (3.8+ required)
python --version

# Reinstall dependencies
pip install -r requirements.txt --force-reinstall
```

**File upload fails:**
- Ensure file is CSV or Excel format
- Check file size (recommended < 50MB)
- Verify file has proper headers

**Plugin errors:**
- Enable Developer Mode for detailed error messages
- Check plugin code syntax
- Ensure all required methods are implemented

### Getting Help

1. **Check Documentation**: Review README.md and CONTRIBUTING.md
2. **Examine Examples**: Look at existing plugins for reference
3. **Test with Samples**: Use built-in datasets to isolate issues
4. **Review Logs**: Check console output for error details

## 🎓 Learning Resources

### Understanding Anonymization
- **K-Anonymity**: Groups records to prevent identification
- **Differential Privacy**: Adds calibrated noise for privacy
- **Suppression**: Removes sensitive information entirely
- **Generalization**: Replaces specific values with ranges

### Privacy-Utility Tradeoff
- Higher privacy often means lower data utility
- Different techniques suit different use cases
- Evaluation metrics help find optimal balance
- ML performance testing quantifies impact

## 🔗 Next Steps

1. **Explore Techniques**: Try different anonymization methods
2. **Analyze Results**: Use the visualization dashboard
3. **Develop Plugins**: Create custom anonymization techniques
4. **Share Findings**: Export results and configurations
5. **Contribute**: Help improve the platform

---

**Need more help?** Check the full documentation in README.md or explore the interactive help within the application!