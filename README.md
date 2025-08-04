# Modular Dataset Anonymization Tool

A comprehensive privacy-preserving data anonymization platform built with Streamlit, featuring modular plugin architecture and machine learning performance evaluation.

## 🎯 Overview

This repository contains the implementation of a modular dataset anonymization tool developed as part of a Bachelor's thesis. The platform provides a user-friendly interface for applying various anonymization techniques to datasets while evaluating their impact on machine learning model performance.

## ✨ Key Features

- **🔧 Modular Plugin Architecture**: Extensible system supporting custom anonymization techniques
- **📊 Interactive Web Interface**: Streamlit-based GUI for easy dataset upload and anonymization
- **🤖 ML Performance Evaluation**: Comprehensive analysis of anonymization impact on model accuracy
- **📈 Visualization Dashboard**: Rich charts and graphs for performance comparison
- **🔒 Privacy Techniques**: Implementation of various anonymization methods including k-anonymity, differential privacy, and perturbation methods
- **📁 Multi-format Support**: CSV and Excel file upload capabilities
- **🎲 Sample Datasets**: Built-in datasets for testing and demonstration

## 🏗️ Architecture

The system follows a modular architecture with the following components:

- **Core Application** (`ml_models/core/`): Main Streamlit application
- **Anonymization Plugins** (`ml_models/src/anonymizers/`): Modular anonymization techniques
- **ML Evaluation** (`ml_models/src/ml_plugins/`): Machine learning performance testing
- **Visualization** (`ml_models/src/dashboard_analyse/`): Performance analysis and charts
- **Utilities** (`ml_models/src/utilis/`): Helper functions and data processing

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- pip package manager

### Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd bachelor-thesis-anonymization
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the application:
```bash
streamlit run ml_models/core/app.py
```

## 📖 Usage

1. **Upload Data**: Use the file uploader to load your CSV or Excel dataset
2. **Select Technique**: Choose an anonymization technique from the sidebar
3. **Configure Parameters**: Adjust technique-specific settings
4. **Apply Anonymization**: Process your data with the selected technique
5. **Evaluate Performance**: Use the ML evaluation dashboard to assess impact
6. **Download Results**: Export anonymized data and performance reports

## 🔌 Plugin Development

The system supports custom anonymization plugins. See the developer mode in the application for:

- Plugin code editor with syntax validation
- Real-time testing capabilities
- Code snippets and templates
- Plugin management tools

## 📊 Evaluation Metrics

The platform evaluates anonymization techniques using:

- **Accuracy**: Overall model performance
- **Precision/Recall**: Classification quality metrics
- **F1-Score**: Balanced performance measure
- **ROC-AUC**: Receiver operating characteristic analysis
- **Privacy-Utility Tradeoff**: Balance between privacy and data utility

## 🗂️ Repository Structure

```
bachelor-thesis-anonymization/
├── ml_models/
│   ├── core/                    # Main application
│   ├── src/                     # Source code
│   │   ├── anonymizers/         # Anonymization plugins
│   │   ├── ml_plugins/          # ML evaluation modules
│   │   └── dashboard_analyse/   # Visualization components
│   ├── performance_testing/     # Performance benchmarks
│   └── visualizations/          # Generated charts and graphs
├── docs/                        # Documentation
├── examples/                    # Usage examples
└── tests/                       # Test suites
```

## 🤝 Contributing

This is an academic project developed for a Bachelor's thesis. For questions or suggestions, please refer to the thesis documentation.

## 📄 License

This project is developed for academic purposes as part of a Bachelor's thesis.

## 👨‍🎓 Author

**Ali Malikov**  
Bachelor's Thesis Project  
TU Berlin
2025

## 📚 Citation

If you use this work in your research, please cite:

```bibtex
@thesis{malikov2024anonymization,
  title={Modular Dataset Anonymization Tool: A Privacy-Preserving Approach with Machine Learning Performance Evaluation},
  author={Malikov, Ali},
  year={2024},
  school={[University Name]},
  type={Bachelor's Thesis}
}
```