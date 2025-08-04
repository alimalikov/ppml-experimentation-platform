#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Thesis Analysis Script for PPML Benchmarking

This script loads the results CSV file from your local path, cleans the data,
and performs a series of analyses and visualizations to explore the effect of
anonymization techniques on ML model performance. It includes:
    - Exploratory Data Analysis (distribution, box, and violin plots)
    - Grouped and aggregated performance comparisons (bar charts)
    - Correlation analysis (heatmap and pairplot)
    - Scatter and regression plots for trade-off analysis
    - Statistical testing (ANOVA)
    - PCA for dimensionality reduction

Each visualization includes clear axis labels and printed descriptions to
explain its academic value.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy.stats import f_oneway
import re

# -----------------------------
# 1. Data Loading and Cleaning
# -----------------------------

# Provide the complete file path to your CSV file.
DATA_FILE = r"C:\Users\alise\OneDrive\Desktop\Bachelor Thesis\ml_models\results\final_master_summary_consolidated.csv"

# Load the CSV file into a DataFrame
df = pd.read_csv(DATA_FILE)

# Display initial data information
print("----- Data Head -----")
print(df.head())
print("\n----- Data Info -----")
print(df.info())
print("\n----- Data Description -----")
print(df.describe())

# Drop rows with missing critical values: Model, Privacy_Type, Accuracy.
df_clean = df.dropna(subset=["Model", "Privacy_Type", "Accuracy"])

# Convert columns to categorical types where applicable.
df_clean['Model'] = df_clean['Model'].astype('category')
df_clean['Privacy_Type'] = df_clean['Privacy_Type'].astype('category')

# Optional: Extract numeric value from Privacy_Params_Final (e.g., epsilon values like "ε≈5.0")
def extract_numeric(value):
    """Extracts the first numeric substring from a string like 'ε≈5.0'."""
    if isinstance(value, str):
        match = re.search(r"([\d\.]+)", value)
        if match:
            return float(match.group(1))
    return np.nan

if "Privacy_Params_Final" in df_clean.columns:
    df_clean['Privacy_Param_Num'] = df_clean['Privacy_Params_Final'].apply(extract_numeric)

# -----------------------------
# 2. Exploratory Data Analysis (EDA)
# -----------------------------
print("\nVisualization 1: Distribution of Model Accuracy Scores")
print("This histogram shows the frequency distribution of Accuracy scores, revealing central tendency, dispersion, and potential outliers.")

plt.figure(figsize=(8, 5))
sns.histplot(df_clean['Accuracy'], bins=20, kde=True)
plt.title('Distribution of Model Accuracy Scores')
plt.xlabel('Accuracy Score')
plt.ylabel('Frequency')
plt.tight_layout()
plt.show()

print("\nVisualization 2: Boxplot of Model Accuracy by ML Model")
print("This boxplot compares the accuracy distribution for each ML model, showing medians, quartiles, and potential outliers. It helps us assess the variability and robustness of each model.")

plt.figure(figsize=(10, 6))
sns.boxplot(x='Model', y='Accuracy', data=df_clean)
plt.title('Boxplot of Model Accuracy by ML Model')
plt.xlabel('ML Model')
plt.ylabel('Accuracy Score')
plt.tight_layout()
plt.show()

print("\nVisualization 3: Violin Plot of Accuracy by Privacy Type and ML Model")
print("This violin plot displays the full distribution of accuracy scores for each privacy type and model, revealing nuances in distribution shape along with summary statistics such as the median and interquartile range.")

plt.figure(figsize=(12, 6))
sns.violinplot(x='Privacy_Type', y='Accuracy', hue='Model', data=df_clean, split=True)
plt.title('Violin Plot: Accuracy Distribution by Privacy Type and ML Model')
plt.xlabel('Privacy Type')
plt.ylabel('Accuracy Score')
plt.legend(title='ML Model', bbox_to_anchor=(1.05, 1), loc=2)
plt.tight_layout()
plt.show()

# -----------------------------
# 3. Grouped and Aggregate Analysis
# -----------------------------
print("\nVisualization 4: Bar Chart of Mean Accuracy across Models and Privacy Types")
print("This bar chart aggregates the mean accuracy (with standard deviation error bars) for each combination of ML model and privacy type. It highlights group differences and reliability.")

# Group by Model and Privacy_Type; compute mean, std, and median for Accuracy.
grouped = df_clean.groupby(['Model', 'Privacy_Type'])['Accuracy'].agg(['mean', 'std', 'median']).reset_index()
print(grouped)

plt.figure(figsize=(12, 7))
sns.barplot(x='Model', y='mean', hue='Privacy_Type', data=grouped, capsize=.1, errwidth=1.5)
plt.title('Mean Accuracy across Models and Privacy Types')
plt.xlabel('ML Model')
plt.ylabel('Mean Accuracy Score')
plt.legend(title='Privacy Type')
plt.tight_layout()
plt.show()

# -----------------------------
# 4. Correlation Analysis
# -----------------------------
print("\nVisualization 5: Heatmap of Selected Metrics Correlation")
print("This heatmap shows the correlation coefficients among key performance metrics and privacy parameters. It is essential for detecting linear relationships and potential confounding factors.")

# Choose numeric columns for correlation analysis.
numeric_cols = ['Accuracy', 'Balanced_Accuracy', 'Precision_>50K', 'Recall_>50K', 'F1_>50K', 'Training Time (s)']
if "Privacy_Param_Num" in df_clean.columns:
    numeric_cols.append('Privacy_Param_Num')
if "Achieved_Epsilon" in df_clean.columns:
    numeric_cols.append('Achieved_Epsilon')

available_numeric = [col for col in numeric_cols if col in df_clean.columns]
if available_numeric:
    corr_data = df_clean[available_numeric].dropna()
    correlation_matrix = corr_data.corr()
    plt.figure(figsize=(8, 6))
    sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm")
    plt.title('Correlation Matrix of Selected Metrics')
    plt.tight_layout()
    plt.show()
else:
    print("Warning: No expected numeric columns found for correlation analysis.")

print("\nVisualization 6: Pairplot of Key Performance Metrics")
print("This pairplot provides a comprehensive view of the pairwise relationships (including distributions) among the selected performance metrics and privacy parameters, facilitating in-depth analysis of potential interactions.")

# Use pairplot for additional insights if enough numeric columns exist.
if len(available_numeric) >= 2:
    sns.pairplot(df_clean[available_numeric].dropna())
    plt.suptitle('Pairplot of Selected Performance Metrics', y=1.02)
    plt.tight_layout()
    plt.show()
else:
    print("Not enough numeric features available for a pairplot.")

# -----------------------------
# 5. Scatter and Trade-off Analysis
# -----------------------------
# Determine a trade-off parameter: first check for Achieved_Epsilon; if not, fall back to Privacy_Param_Num.
tradeoff_col = None
if "Achieved_Epsilon" in df_clean.columns and df_clean['Achieved_Epsilon'].notna().sum() > 0:
    tradeoff_col = "Achieved_Epsilon"
elif "Privacy_Param_Num" in df_clean.columns and df_clean['Privacy_Param_Num'].notna().sum() > 0:
    tradeoff_col = "Privacy_Param_Num"

if tradeoff_col:
    print(f"\nVisualization 7: Scatter Plot with Regression Line ({tradeoff_col} vs Accuracy)")
    print(f"This plot depicts the relationship between the trade-off parameter '{tradeoff_col}' and model accuracy. A regression line is fitted to observe trends, indicating how changes in privacy parameters affect performance.")

    # Use seaborn's lmplot to display a regression line along with scatter points.
    sns.lmplot(x=tradeoff_col, y='Accuracy', hue='Model', col='Privacy_Type', data=df_clean, aspect=1.2, height=4, markers="o", ci=None)
    plt.subplots_adjust(top=0.85)
    plt.suptitle(f'Relationship between {tradeoff_col} and Accuracy by Privacy Type and Model')
    plt.show()

    print("\nVisualization 8: Line Plot for Trade-off Analysis")
    print(f"This line plot shows the trend of Accuracy as the value of '{tradeoff_col}' changes. Different lines correspond to different ML models and privacy types, making it useful for comparing sensitivity to privacy settings.")
    
    plt.figure(figsize=(10, 6))
    sns.lineplot(x=tradeoff_col, y='Accuracy', hue='Model', style='Privacy_Type', data=df_clean, markers=True)
    plt.title(f'Trade-off Analysis: {tradeoff_col} vs Accuracy')
    plt.xlabel(f'{tradeoff_col} (Privacy Parameter)')
    plt.ylabel('Accuracy Score')
    plt.tight_layout()
    plt.show()
else:
    print("No suitable trade-off column found for scatter/line plots.")

# -----------------------------
# 6. Statistical Significance Testing (ANOVA Example)
# -----------------------------
print("\nStatistical Testing: ANOVA for Comparing Accuracy across Privacy Types for a Selected Model")
# Example: Perform ANOVA for a specific model (e.g., 'LGB') comparing Accuracy across different Privacy_Types.
model_name = 'LGB'
techniques_unique = df_clean[df_clean['Model'] == model_name]['Privacy_Type'].unique()

if len(techniques_unique) >= 2:
    # Compare the first two privacy techniques for the chosen model.
    group1 = df_clean[(df_clean['Model'] == model_name) & (df_clean['Privacy_Type'] == techniques_unique[0])]['Accuracy']
    group2 = df_clean[(df_clean['Model'] == model_name) & (df_clean['Privacy_Type'] == techniques_unique[1])]['Accuracy']
    
    anova_result = f_oneway(group1, group2)
    print(f"ANOVA Result for Model '{model_name}': Comparing '{techniques_unique[0]}' vs. '{techniques_unique[1]}'")
    print("F statistic =", anova_result.statistic)
    print("p-value =", anova_result.pvalue)
    print("A low p-value (< 0.05) suggests a statistically significant difference in accuracy between the groups.")
else:
    print(f"Not enough privacy groups for model '{model_name}' to perform ANOVA.")

# -----------------------------
# 7. Advanced Analysis: PCA
# -----------------------------
print("\nVisualization 9: PCA of Selected Performance Metrics")
print("This scatter plot, based on Principal Component Analysis, reduces multiple performance metrics into two principal components, allowing visualization of underlying patterns and clusters across models and privacy types.")

# Select features for PCA (adjust as needed)
pca_features = [col for col in ['Accuracy', 'Balanced_Accuracy', 'Precision_>50K', 'Recall_>50K', 'F1_>50K'] if col in df_clean.columns]
if tradeoff_col:
    pca_features.append(tradeoff_col)

if len(pca_features) >= 2:
    # Standardize the features and drop rows with missing values.
    scaler = StandardScaler()
    pca_data = df_clean[pca_features].dropna()
    scaled_features = scaler.fit_transform(pca_data)
    
    # Perform PCA to reduce dimensions to 2 components.
    pca = PCA(n_components=2)
    principal_components = pca.fit_transform(scaled_features)
    
    # Create a DataFrame for the principal components.
    pca_df = pd.DataFrame(data=principal_components, columns=['PC1', 'PC2'])
    # Align with model and privacy type data.
    pca_df = pd.concat([pca_df, df_clean.loc[pca_data.index, ['Model', 'Privacy_Type']].reset_index(drop=True)], axis=1)
    
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x='PC1', y='PC2', hue='Model', style='Privacy_Type', data=pca_df)
    plt.title('PCA of Selected Performance Metrics')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.tight_layout()
    plt.show()
else:
    print("Not enough numeric features available for PCA.")

# -----------------------------
# End of Analysis Script
# -----------------------------
print("\nAnalysis complete.")
