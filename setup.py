from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="modular-dataset-anonymization-tool",
    version="1.0.0",
    author="Ali Malikov",
    author_email="ali.malikov@campus.tu-berlin.de",
    description="A modular privacy-preserving data anonymization platform with ML performance evaluation",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/bachelor-thesis-anonymization",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Security :: Cryptography",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
            "mypy>=1.0.0",
        ],
        "docs": [
            "sphinx>=6.0.0",
            "sphinx-rtd-theme>=1.2.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "anonymization-tool=ml_models.core.app:main",
        ],
    },
    include_package_data=True,
    package_data={
        "ml_models": [
            "src/anonymizers/plugins/*.py",
            "src/ml_plugins/*.py",
            "src/dashboard_analyse/*.py",
            "src/utilis/*.py",
        ],
    },
    keywords="anonymization, privacy, machine-learning, data-protection, streamlit",
    project_urls={
        "Bug Reports": "https://github.com/yourusername/bachelor-thesis-anonymization/issues",
        "Source": "https://github.com/yourusername/bachelor-thesis-anonymization",
        "Documentation": "https://github.com/yourusername/bachelor-thesis-anonymization/wiki",
    },
)