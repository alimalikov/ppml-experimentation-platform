# Contributing to Modular Dataset Anonymization Tool

Thank you for your interest in contributing to this project! This is an academic project developed as part of a Bachelor's thesis.

## 🎓 Academic Context

This project was developed as part of a Bachelor's thesis research on privacy-preserving data anonymization techniques and their impact on machine learning model performance.

## 🔧 Development Setup

1. **Clone the repository**:
```bash
git clone <repository-url>
cd bachelor-thesis-anonymization
```

2. **Create virtual environment**:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**:
```bash
pip install -r requirements.txt
```

4. **Run the application**:
```bash
streamlit run ml_models/core/app.py
```

## 🔌 Plugin Development

### Creating Custom Anonymization Plugins

1. **Enable Developer Mode** in the application sidebar
2. **Use the Plugin Editor** to write your custom anonymization technique
3. **Follow the Plugin Template**:

```python
from ..base_anonymizer import Anonymizer
import pandas as pd
from typing import List, Dict, Any

class MyCustomPlugin(Anonymizer):
    def __init__(self):
        self._name = "My Custom Technique"
        self._description = "Description of the technique"
    
    def get_name(self) -> str:
        return self._name
    
    def get_description(self) -> str:
        return self._description
    
    def anonymize(self, df_input: pd.DataFrame, parameters: Dict[str, Any], sa_col: str | None) -> pd.DataFrame:
        # Implement your anonymization logic here
        return df_input.copy()
    
    # Implement other required methods...

def get_plugin():
    return MyCustomPlugin()
```

### Plugin Requirements

- Must inherit from `Anonymizer` base class
- Must implement all abstract methods
- Must include `get_plugin()` factory function
- Should follow naming convention: `*_plugin.py`

## 🧪 Testing

### Running Performance Tests

```bash
# Quick benchmark
python ml_models/performance_testing/quick_benchmark.py

# Comprehensive performance test
python ml_models/performance_testing/performance_tester.py

# ML model impact analysis
python ml_models/ml_performance_testing/ml_performance_tester.py
```

### Adding New Tests

1. Create test files in appropriate testing directories
2. Follow existing test patterns
3. Include both unit tests and integration tests
4. Test with various dataset sizes and types

## 📊 Visualization Development

### Adding New Charts

1. Extend visualization modules in `src/dashboard_analyse/`
2. Follow the existing chart generation patterns
3. Ensure charts are accessible and informative
4. Include proper legends and labels

### Chart Types Available

- Performance comparison charts
- Privacy-utility tradeoff analysis
- Technique ranking visualizations
- Model impact analysis
- Degradation analysis plots

## 🗂️ Code Organization

```
ml_models/
├── core/                    # Main Streamlit application
│   └── app.py              # Primary application entry point
├── src/
│   ├── anonymizers/        # Anonymization techniques
│   │   ├── base_anonymizer.py
│   │   └── plugins/        # Plugin implementations
│   ├── ml_plugins/         # ML evaluation modules
│   ├── dashboard_analyse/  # Visualization components
│   └── utilis/            # Utility functions
├── performance_testing/    # Performance benchmarks
└── scripts/               # Utility scripts
```

## 📝 Documentation Standards

- Use clear, descriptive docstrings
- Include type hints where appropriate
- Comment complex algorithms
- Update README.md for significant changes
- Document new plugin interfaces

## 🔍 Code Quality

### Style Guidelines

- Follow PEP 8 Python style guide
- Use meaningful variable and function names
- Keep functions focused and modular
- Include error handling and validation

### Performance Considerations

- Optimize for large datasets
- Use efficient pandas operations
- Cache expensive computations
- Profile performance-critical code

## 🐛 Bug Reports

When reporting bugs, please include:

1. **Environment details** (Python version, OS, dependencies)
2. **Steps to reproduce** the issue
3. **Expected vs actual behavior**
4. **Sample data** (if applicable and non-sensitive)
5. **Error messages** and stack traces

## 💡 Feature Requests

For new features, please provide:

1. **Clear description** of the proposed feature
2. **Use case** and motivation
3. **Implementation suggestions** (if any)
4. **Potential impact** on existing functionality

## 📧 Contact

For questions related to this academic project:

- **Author**: Ali Malikov
- **Institution**: [University Name]
- **Project Type**: Bachelor's Thesis

## 📄 License

This project is developed for academic purposes. Please respect the academic nature of this work and provide appropriate attribution when referencing or building upon it.