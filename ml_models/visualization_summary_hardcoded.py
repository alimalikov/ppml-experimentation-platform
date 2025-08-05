import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for server environments
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import os
from matplotlib import rcParams
import matplotlib.patches as mpatches

# Configure publication-quality visualization settings
plt.style.use('default')
sns.set_palette("husl")

# Academic publication typography and layout parameters
rcParams['font.family'] = 'Porsche Next TT'
rcParams['font.size'] = 16
rcParams['axes.titlesize'] = 20
rcParams['axes.labelsize'] = 18
rcParams['xtick.labelsize'] = 15
rcParams['ytick.labelsize'] = 15
rcParams['legend.fontsize'] = 16
rcParams['legend.title_fontsize'] = 17
rcParams['figure.titlesize'] = 22
rcParams['figure.dpi'] = 300
rcParams['savefig.dpi'] = 300
rcParams['savefig.bbox'] = 'tight'
rcParams['savefig.pad_inches'] = 0.2
rcParams['axes.spines.top'] = False
rcParams['axes.spines.right'] = False
rcParams['axes.grid'] = True
rcParams['grid.alpha'] = 0.3
rcParams['grid.linewidth'] = 0.8
rcParams['lines.linewidth'] = 2.5
rcParams['lines.markersize'] = 8

# Colorblind-friendly palette for academic publications
PROFESSIONAL_COLORS = ['#007ACC', '#FF6B35', '#28A745']
MODEL_COLORS = {
    'Logistic Regression': '#007ACC',
    'Random Forest': '#FF6B35',
    'XGBoost': '#28A745'
}

TECHNIQUE_COLORS = {
    'Original': '#2D3748',
    'Micro Aggregation (High)': '#38A169',
    'Micro Aggregation (Medium)': '#48BB78',
    'Micro Aggregation (Minimal)': '#68D391',
    'Differential Privacy (Minimal)': '#4299E1',
    'Differential Privacy (Medium)': '#3182CE',
    'Differential Privacy (High)': '#2B6CB0',
    'Randomized Response (Minimal)': '#ED8936',
    'Randomized Response (Medium)': '#DD6B20',
    'Randomized Response (High)': '#C05621'
}

# Experimental results: nested dictionary structure for performance metrics
results = {
    "Iris": {
        "Original": {
            "Logistic Regression": {"accuracy": 0.9767, "precision": 0.9769, "recall": 0.9767, "f1_score": 0.9767, "balanced_accuracy": 0.9767, "matthews_correlation": 0.9651, "roc_auc": 0.9992},
            "Random Forest": {"accuracy": 1.0000, "precision": 1.0000, "recall": 1.0000, "f1_score": 1.0000, "balanced_accuracy": 1.0000, "matthews_correlation": 1.0000, "roc_auc": 1.0000},
            "XGBoost": {"accuracy": 1.0000, "precision": 1.0000, "recall": 1.0000, "f1_score": 1.0000, "balanced_accuracy": 1.0000, "matthews_correlation": 1.0000, "roc_auc": 1.0000}
        },
        "Differential Privacy (High)": {
            "Logistic Regression": {"accuracy": 0.2867, "precision": 0.2803, "recall": 0.2867, "f1_score": 0.2801, "balanced_accuracy": 0.2867, "matthews_correlation": -0.0709, "roc_auc": 0.484},
            "Random Forest": {"accuracy": 0.3467, "precision": 0.3472, "recall": 0.3467, "f1_score": 0.3468, "balanced_accuracy": 0.3467, "matthews_correlation": 0.02, "roc_auc": 0.5146},
            "XGBoost": {"accuracy": 0.35, "precision": 0.3502, "recall": 0.35, "f1_score": 0.35, "balanced_accuracy": 0.35, "matthews_correlation": 0.025, "roc_auc": 0.5103}
        },
        "Differential Privacy (Medium)": {
            "Logistic Regression": {"accuracy": 0.3867, "precision": 0.379, "recall": 0.3867, "f1_score": 0.3806, "balanced_accuracy": 0.3867, "matthews_correlation": 0.0805, "roc_auc": 0.5912},
            "Random Forest": {"accuracy": 0.4167, "precision": 0.4158, "recall": 0.4167, "f1_score": 0.4161, "balanced_accuracy": 0.4167, "matthews_correlation": 0.125, "roc_auc": 0.5988},
            "XGBoost": {"accuracy": 0.4367, "precision": 0.4397, "recall": 0.4367, "f1_score": 0.4373, "balanced_accuracy": 0.4367, "matthews_correlation": 0.1553, "roc_auc": 0.6066}
        },
        "Differential Privacy (Minimal)": {
            "Logistic Regression": {"accuracy": 0.92, "precision": 0.9201, "recall": 0.92, "f1_score": 0.92, "balanced_accuracy": 0.92, "matthews_correlation": 0.8801, "roc_auc": 0.9851},
            "Random Forest": {"accuracy": 0.94, "precision": 0.9404, "recall": 0.94, "f1_score": 0.94, "balanced_accuracy": 0.94, "matthews_correlation": 0.9102, "roc_auc": 0.9893},
            "XGBoost": {"accuracy": 0.9467, "precision": 0.9471, "recall": 0.9467, "f1_score": 0.9466, "balanced_accuracy": 0.9467, "matthews_correlation": 0.9202, "roc_auc": 0.9933}
        },
        "Randomized Response (High)": {
            "Logistic Regression": {"accuracy": 0.4867, "precision": 0.4607, "recall": 0.5599, "f1_score": 0.4454, "balanced_accuracy": 0.5599, "matthews_correlation": 0.2135, "roc_auc": 0.6807},
            "Random Forest": {"accuracy": 0.4833, "precision": 0.4401, "recall": 0.5421, "f1_score": 0.4456, "balanced_accuracy": 0.5421, "matthews_correlation": 0.1674, "roc_auc": 0.6722},
            "XGBoost": {"accuracy": 0.5100, "precision": 0.3847, "recall": 0.3772, "f1_score": 0.3773, "balanced_accuracy": 0.3772, "matthews_correlation": 0.0952, "roc_auc": 0.6665}
        },
        "Randomized Response (Medium)": {
            "Logistic Regression": {"accuracy": 0.7767, "precision": 0.7773, "recall": 0.7987, "f1_score": 0.7782, "balanced_accuracy": 0.7987, "matthews_correlation": 0.6711, "roc_auc": 0.8595},
            "Random Forest": {"accuracy": 0.7900, "precision": 0.7874, "recall": 0.8105, "f1_score": 0.7917, "balanced_accuracy": 0.8105, "matthews_correlation": 0.6884, "roc_auc": 0.8590},
            "XGBoost": {"accuracy": 0.7900, "precision": 0.7874, "recall": 0.8105, "f1_score": 0.7917, "balanced_accuracy": 0.8105, "matthews_correlation": 0.6884, "roc_auc": 0.8604}
        },
        "Randomized Response (Minimal)": {
            "Logistic Regression": {"accuracy": 0.9333, "precision": 0.9334, "recall": 0.9365, "f1_score": 0.9336, "balanced_accuracy": 0.9365, "matthews_correlation": 0.9012, "roc_auc": 0.9583},
            "Random Forest": {"accuracy": 0.9433, "precision": 0.9435, "recall": 0.9464, "f1_score": 0.9439, "balanced_accuracy": 0.9464, "matthews_correlation": 0.916, "roc_auc": 0.9605},
            "XGBoost": {"accuracy": 0.9433, "precision": 0.9435, "recall": 0.9464, "f1_score": 0.9439, "balanced_accuracy": 0.9464, "matthews_correlation": 0.916, "roc_auc": 0.9661}
        },
        "Micro Aggregation (High)": {
            "Logistic Regression": {"accuracy": 0.9767, "precision": 0.9769, "recall": 0.9767, "f1_score": 0.9767, "balanced_accuracy": 0.9767, "matthews_correlation": 0.9651, "roc_auc": 0.9992},
            "Random Forest": {"accuracy": 1.0000, "precision": 1.0000, "recall": 1.0000, "f1_score": 1.0000, "balanced_accuracy": 1.0000, "matthews_correlation": 1.0000, "roc_auc": 1.0000},
            "XGBoost": {"accuracy": 1.0000, "precision": 1.0000, "recall": 1.0000, "f1_score": 1.0000, "balanced_accuracy": 1.0000, "matthews_correlation": 1.0000, "roc_auc": 1.0000}
        },
        "Micro Aggregation (Medium)": {
            "Logistic Regression": {"accuracy": 0.9767, "precision": 0.9769, "recall": 0.9767, "f1_score": 0.9767, "balanced_accuracy": 0.9767, "matthews_correlation": 0.9651, "roc_auc": 0.9992},
            "Random Forest": {"accuracy": 1.0000, "precision": 1.0000, "recall": 1.0000, "f1_score": 1.0000, "balanced_accuracy": 1.0000, "matthews_correlation": 1.0000, "roc_auc": 1.0000},
            "XGBoost": {"accuracy": 1.0000, "precision": 1.0000, "recall": 1.0000, "f1_score": 1.0000, "balanced_accuracy": 1.0000, "matthews_correlation": 1.0000, "roc_auc": 1.0000}
        },
        "Micro Aggregation (Minimal)": {
            "Logistic Regression": {"accuracy": 0.9733, "precision": 0.9738, "recall": 0.9733, "f1_score": 0.9733, "balanced_accuracy": 0.9733, "matthews_correlation": 0.9603, "roc_auc": 0.9991},
            "Random Forest": {"accuracy": 0.9967, "precision": 0.9967, "recall": 0.9967, "f1_score": 0.9967, "balanced_accuracy": 0.9967, "matthews_correlation": 0.995, "roc_auc": 1.0},
            "XGBoost": {"accuracy": 0.9967, "precision": 0.9967, "recall": 0.9967, "f1_score": 0.9967, "balanced_accuracy": 0.9967, "matthews_correlation": 0.995, "roc_auc": 1.0}
        }
    },
    "Breast Cancer": {
        "Original": {
            "Logistic Regression": {"accuracy": 0.9895, "precision": 0.9883, "recall": 0.9892, "f1_score": 0.9887, "balanced_accuracy": 0.9892, "matthews_correlation": 0.9775, "roc_auc": 0.999},
            "Random Forest": {"accuracy": 1.0000, "precision": 1.0000, "recall": 1.0000, "f1_score": 1.0000, "balanced_accuracy": 1.0000, "matthews_correlation": 1.0000, "roc_auc": 1.0000},
            "XGBoost": {"accuracy": 1.0000, "precision": 1.0000, "recall": 1.0000, "f1_score": 1.0000, "balanced_accuracy": 1.0000, "matthews_correlation": 1.0000, "roc_auc": 1.0000}
        },
        "Differential Privacy (High)": {
            "Logistic Regression": {"accuracy": 0.9868, "precision": 0.9848, "recall": 0.9871, "f1_score": 0.9859, "balanced_accuracy": 0.9871, "matthews_correlation": 0.9719, "roc_auc": 0.9984},
            "Random Forest": {"accuracy": 1.0000, "precision": 1.0000, "recall": 1.0000, "f1_score": 1.0000, "balanced_accuracy": 1.0000, "matthews_correlation": 1.0000, "roc_auc": 1.0000},
            "XGBoost": {"accuracy": 1.0000, "precision": 1.0000, "recall": 1.0000, "f1_score": 1.0000, "balanced_accuracy": 1.0000, "matthews_correlation": 1.0000, "roc_auc": 1.0000}
        },
        "Differential Privacy (Medium)": {
            "Logistic Regression": {"accuracy": 0.9815, "precision": 0.9796, "recall": 0.9810, "f1_score": 0.9803, "balanced_accuracy": 0.9810, "matthews_correlation": 0.9606, "roc_auc": 0.9979},
            "Random Forest": {"accuracy": 1.0000, "precision": 1.0000, "recall": 1.0000, "f1_score": 1.0000, "balanced_accuracy": 1.0000, "matthews_correlation": 1.0000, "roc_auc": 1.0000},
            "XGBoost": {"accuracy": 1.0000, "precision": 1.0000, "recall": 1.0000, "f1_score": 1.0000, "balanced_accuracy": 1.0000, "matthews_correlation": 1.0000, "roc_auc": 1.0000}
        },
        "Differential Privacy (Minimal)": {
            "Logistic Regression": {"accuracy": 0.9842, "precision": 0.9822, "recall": 0.9840, "f1_score": 0.9831, "balanced_accuracy": 0.9840, "matthews_correlation": 0.9663, "roc_auc": 0.9983},
            "Random Forest": {"accuracy": 1.0000, "precision": 1.0000, "recall": 1.0000, "f1_score": 1.0000, "balanced_accuracy": 1.0000, "matthews_correlation": 1.0000, "roc_auc": 1.0000},
            "XGBoost": {"accuracy": 1.0000, "precision": 1.0000, "recall": 1.0000, "f1_score": 1.0000, "balanced_accuracy": 1.0000, "matthews_correlation": 1.0000, "roc_auc": 1.0000}
        },
        "Randomized Response (High)": {
            "Logistic Regression": {"accuracy": 0.6195, "precision": 0.6183, "recall": 0.6130, "f1_score": 0.6117, "balanced_accuracy": 0.6130, "matthews_correlation": 0.2312, "roc_auc": 0.6277},
            "Random Forest": {"accuracy": 0.5518, "precision": 0.5510, "recall": 0.5512, "f1_score": 0.5509, "balanced_accuracy": 0.5512, "matthews_correlation": 0.1022, "roc_auc": 0.5744},
            "XGBoost": {"accuracy": 0.5641, "precision": 0.5617, "recall": 0.5614, "f1_score": 0.5614, "balanced_accuracy": 0.5614, "matthews_correlation": 0.1231, "roc_auc": 0.5728}
        },
        "Randomized Response (Medium)": {
            "Logistic Regression": {"accuracy": 0.8401, "precision": 0.8390, "recall": 0.8286, "f1_score": 0.8326, "balanced_accuracy": 0.8286, "matthews_correlation": 0.6675, "roc_auc": 0.8493},
            "Random Forest": {"accuracy": 0.8541, "precision": 0.8517, "recall": 0.8457, "f1_score": 0.8483, "balanced_accuracy": 0.8457, "matthews_correlation": 0.6973, "roc_auc": 0.8520},
            "XGBoost": {"accuracy": 0.8585, "precision": 0.8569, "recall": 0.8494, "f1_score": 0.8526, "balanced_accuracy": 0.8494, "matthews_correlation": 0.7063, "roc_auc": 0.8534}
        },
        "Randomized Response (Minimal)": {
            "Logistic Regression": {"accuracy": 0.9306, "precision": 0.9310, "recall": 0.9218, "f1_score": 0.9259, "balanced_accuracy": 0.9218, "matthews_correlation": 0.8527, "roc_auc": 0.9375},
            "Random Forest": {"accuracy": 0.9543, "precision": 0.9536, "recall": 0.9496, "f1_score": 0.9515, "balanced_accuracy": 0.9496, "matthews_correlation": 0.9032, "roc_auc": 0.9498},
            "XGBoost": {"accuracy": 0.9543, "precision": 0.9536, "recall": 0.9496, "f1_score": 0.9515, "balanced_accuracy": 0.9496, "matthews_correlation": 0.9032, "roc_auc": 0.9483}
        },
        "Micro Aggregation (High)": {
            "Logistic Regression": {"accuracy": 0.9895, "precision": 0.9883, "recall": 0.9892, "f1_score": 0.9887, "balanced_accuracy": 0.9892, "matthews_correlation": 0.9775, "roc_auc": 0.9990},
            "Random Forest": {"accuracy": 1.0000, "precision": 1.0000, "recall": 1.0000, "f1_score": 1.0000, "balanced_accuracy": 1.0000, "matthews_correlation": 1.0000, "roc_auc": 1.0000},
            "XGBoost": {"accuracy": 1.0000, "precision": 1.0000, "recall": 1.0000, "f1_score": 1.0000, "balanced_accuracy": 1.0000, "matthews_correlation": 1.0000, "roc_auc": 1.0000}
        },
        "Micro Aggregation (Medium)": {
            "Logistic Regression": {"accuracy": 0.9895, "precision": 0.9883, "recall": 0.9892, "f1_score": 0.9887, "balanced_accuracy": 0.9892, "matthews_correlation": 0.9775, "roc_auc": 0.9990},
            "Random Forest": {"accuracy": 1.0000, "precision": 1.0000, "recall": 1.0000, "f1_score": 1.0000, "balanced_accuracy": 1.0000, "matthews_correlation": 1.0000, "roc_auc": 1.0000},
            "XGBoost": {"accuracy": 1.0000, "precision": 1.0000, "recall": 1.0000, "f1_score": 1.0000, "balanced_accuracy": 1.0000, "matthews_correlation": 1.0000, "roc_auc": 1.0000}
        },
        "Micro Aggregation (Minimal)": {
            "Logistic Regression": {"accuracy": 0.9895, "precision": 0.9883, "recall": 0.9892, "f1_score": 0.9887, "balanced_accuracy": 0.9892, "matthews_correlation": 0.9775, "roc_auc": 0.9990},
            "Random Forest": {"accuracy": 1.0000, "precision": 1.0000, "recall": 1.0000, "f1_score": 1.0000, "balanced_accuracy": 1.0000, "matthews_correlation": 1.0000, "roc_auc": 1.0000},
            "XGBoost": {"accuracy": 1.0000, "precision": 1.0000, "recall": 1.0000, "f1_score": 1.0000, "balanced_accuracy": 1.0000, "matthews_correlation": 1.0000, "roc_auc": 1.0000}
        }
    },
    "Handwritten Digits": {
        "Original": {
            "Logistic Regression": {"accuracy": 1.0000, "precision": 1.0000, "recall": 1.0000, "f1_score": 1.0000, "balanced_accuracy": 1.0000, "matthews_correlation": 1.0000, "roc_auc": 1.0000},
            "Random Forest": {"accuracy": 1.0000, "precision": 1.0000, "recall": 1.0000, "f1_score": 1.0000, "balanced_accuracy": 1.0000, "matthews_correlation": 1.0000, "roc_auc": 1.0000},
            "XGBoost": {"accuracy": 1.0000, "precision": 1.0000, "recall": 1.0000, "f1_score": 1.0000, "balanced_accuracy": 1.0000, "matthews_correlation": 1.0000, "roc_auc": 1.0000}
        },
        "Differential Privacy (High)": {
            "Logistic Regression": {"accuracy": 0.1071, "precision": 0.1085, "recall": 0.1070, "f1_score": 0.1060, "balanced_accuracy": 0.1070, "matthews_correlation": 0.0077, "roc_auc": 0.5018},
            "Random Forest": {"accuracy": 0.0982, "precision": 0.0986, "recall": 0.0982, "f1_score": 0.0982, "balanced_accuracy": 0.0982, "matthews_correlation": -0.0020, "roc_auc": 0.5050},
            "XGBoost": {"accuracy": 0.1130, "precision": 0.1131, "recall": 0.1129, "f1_score": 0.1128, "balanced_accuracy": 0.1129, "matthews_correlation": 0.0143, "roc_auc": 0.5081}
        },
        "Differential Privacy (Medium)": {
            "Logistic Regression": {"accuracy": 0.3258, "precision": 0.3208, "recall": 0.3256, "f1_score": 0.3214, "balanced_accuracy": 0.3256, "matthews_correlation": 0.2511, "roc_auc": 0.7678},
            "Random Forest": {"accuracy": 0.3525, "precision": 0.3495, "recall": 0.3522, "f1_score": 0.3470, "balanced_accuracy": 0.3522, "matthews_correlation": 0.2811, "roc_auc": 0.7791},
            "XGBoost": {"accuracy": 0.3784, "precision": 0.3761, "recall": 0.3783, "f1_score": 0.3766, "balanced_accuracy": 0.3783, "matthews_correlation": 0.3094, "roc_auc": 0.8027}
        },
        "Differential Privacy (Minimal)": {
            "Logistic Regression": {"accuracy": 0.5899, "precision": 0.5873, "recall": 0.5897, "f1_score": 0.5879, "balanced_accuracy": 0.5897, "matthews_correlation": 0.5444, "roc_auc": 0.9167},
            "Random Forest": {"accuracy": 0.6503, "precision": 0.6481, "recall": 0.6500, "f1_score": 0.6469, "balanced_accuracy": 0.6500, "matthews_correlation": 0.6118, "roc_auc": 0.9317},
            "XGBoost": {"accuracy": 0.6797, "precision": 0.6803, "recall": 0.6796, "f1_score": 0.6796, "balanced_accuracy": 0.6796, "matthews_correlation": 0.6442, "roc_auc": 0.9494}
        },
        "Randomized Response (High)": {
            "Logistic Regression": {"accuracy": 0.4591, "precision": 0.1752, "recall": 0.1057, "f1_score": 0.1064, "balanced_accuracy": 0.1057, "matthews_correlation": 0.0024, "roc_auc": 0.8597},
            "Random Forest": {"accuracy": 0.2874, "precision": 0.1766, "recall": 0.5471, "f1_score": 0.2097, "balanced_accuracy": 0.5471, "matthews_correlation": 0.0698, "roc_auc": 0.8631},
            "XGBoost": {"accuracy": 0.4555, "precision": 0.1246, "recall": 0.1045, "f1_score": 0.1045, "balanced_accuracy": 0.1045, "matthews_correlation": 0.0047, "roc_auc": 0.8626}
        },
        "Randomized Response (Medium)": {
            "Logistic Regression": {"accuracy": 0.7001, "precision": 0.7011, "recall": 0.8299, "f1_score": 0.7431, "balanced_accuracy": 0.8299, "matthews_correlation": 0.6727, "roc_auc": 0.9240},
            "Random Forest": {"accuracy": 0.7240, "precision": 0.7235, "recall": 0.8689, "f1_score": 0.7616, "balanced_accuracy": 0.8689, "matthews_correlation": 0.7061, "roc_auc": 0.9246},
            "XGBoost": {"accuracy": 0.6989, "precision": 0.7004, "recall": 0.8300, "f1_score": 0.7437, "balanced_accuracy": 0.8300, "matthews_correlation": 0.6711, "roc_auc": 0.9243}
        },
        "Randomized Response (Minimal)": {
            "Logistic Regression": {"accuracy": 0.9530, "precision": 0.9533, "recall": 0.9598, "f1_score": 0.9553, "balanced_accuracy": 0.9598, "matthews_correlation": 0.9481, "roc_auc": 0.9823},
            "Random Forest": {"accuracy": 0.9597, "precision": 0.9597, "recall": 0.9663, "f1_score": 0.9614, "balanced_accuracy": 0.9663, "matthews_correlation": 0.9556, "roc_auc": 0.9811},
            "XGBoost": {"accuracy": 0.9597, "precision": 0.9597, "recall": 0.9663, "f1_score": 0.9614, "balanced_accuracy": 0.9663, "matthews_correlation": 0.9556, "roc_auc": 0.9816}
        },
        "Micro Aggregation (High)": {
            "Logistic Regression": {"accuracy": 1.0000, "precision": 1.0000, "recall": 1.0000, "f1_score": 1.0000, "balanced_accuracy": 1.0000, "matthews_correlation": 1.0000, "roc_auc": 1.0000},
            "Random Forest": {"accuracy": 1.0000, "precision": 1.0000, "recall": 1.0000, "f1_score": 1.0000, "balanced_accuracy": 1.0000, "matthews_correlation": 1.0000, "roc_auc": 1.0000},
            "XGBoost": {"accuracy": 1.0000, "precision": 1.0000, "recall": 1.0000, "f1_score": 1.0000, "balanced_accuracy": 1.0000, "matthews_correlation": 1.0000, "roc_auc": 1.0000}
        },
        "Micro Aggregation (Medium)": {
            "Logistic Regression": {"accuracy": 1.0000, "precision": 1.0000, "recall": 1.0000, "f1_score": 1.0000, "balanced_accuracy": 1.0000, "matthews_correlation": 1.0000, "roc_auc": 1.0000},
            "Random Forest": {"accuracy": 1.0000, "precision": 1.0000, "recall": 1.0000, "f1_score": 1.0000, "balanced_accuracy": 1.0000, "matthews_correlation": 1.0000, "roc_auc": 1.0000},
            "XGBoost": {"accuracy": 1.0000, "precision": 1.0000, "recall": 1.0000, "f1_score": 1.0000, "balanced_accuracy": 1.0000, "matthews_correlation": 1.0000, "roc_auc": 1.0000}
        },
        "Micro Aggregation (Minimal)": {
            "Logistic Regression": {"accuracy": 1.0000, "precision": 1.0000, "recall": 1.0000, "f1_score": 1.0000, "balanced_accuracy": 1.0000, "matthews_correlation": 1.0000, "roc_auc": 1.0000},
            "Random Forest": {"accuracy": 1.0000, "precision": 1.0000, "recall": 1.0000, "f1_score": 1.0000, "balanced_accuracy": 1.0000, "matthews_correlation": 1.0000, "roc_auc": 1.0000},
            "XGBoost": {"accuracy": 1.0000, "precision": 1.0000, "recall": 1.0000, "f1_score": 1.0000, "balanced_accuracy": 1.0000, "matthews_correlation": 1.0000, "roc_auc": 1.0000}
        }
    }
}

def plot_summary_bar(results, dataset, metric, save_path=None):
    """
    Create a grouped bar chart showing how anonymization affects model performance for a given dataset and metric.
    """
    # Transform nested results into DataFrame for visualization
    data = []
    for anonymization, models in results.get(dataset, {}).items():
        for model, metrics in models.items():
            value = metrics.get(metric, None)
            if value is not None:
                data.append({
                    "Anonymization": anonymization,
                    "Model": model,
                    metric: value
                })
    if not data:
        print(f"No data found for dataset '{dataset}' and metric '{metric}'.")
        return
    
    df = pd.DataFrame(data)
    
    # Initialize figure with academic formatting
    fig, ax = plt.subplots(figsize=(16, 10))
    
    # Generate grouped bar chart with consistent styling
    bar_plot = sns.barplot(
        data=df,
        x="Anonymization",
        y=metric,
        hue="Model",
        palette=PROFESSIONAL_COLORS,
        ax=ax,
        edgecolor='white',
        linewidth=1.5,
        alpha=0.9
    )
    
    # Apply academic formatting standards
    ax.set_title(f'{dataset} Dataset - {metric.replace("_", " ").title()} Performance', 
                fontsize=18, fontweight='bold', pad=20)
    ax.set_ylabel(f'{metric.replace("_", " ").title()}', fontsize=16, fontweight='semibold')
    ax.set_xlabel('Anonymization Technique', fontsize=16, fontweight='semibold')
    
    # Optimize label orientation for clarity
    plt.xticks(rotation=45, ha='right', fontsize=14)
    plt.yticks(fontsize=15)
    
    # Configure legend with academic standards
    legend = ax.legend(title='ML Algorithm', fontsize=14, title_fontsize=15,
                      loc='upper right', frameon=True, fancybox=True, shadow=True)
    legend.get_frame().set_facecolor('white')
    legend.get_frame().set_alpha(0.95)
    legend.get_frame().set_edgecolor('lightgray')
    
    # Label only highest bars to maintain visual clarity
    # Identify maximum values per x-position
    x_positions = {}
    for i, container in enumerate(bar_plot.containers):
        for j, bar in enumerate(container):
            x_pos = bar.get_x() + bar.get_width() / 2
            height = bar.get_height()
            if x_pos not in x_positions or height > x_positions[x_pos]['height']:
                x_positions[x_pos] = {'height': height, 'container_idx': i, 'bar_idx': j}
    
    # Apply selective labeling strategy
    for container_idx, container in enumerate(bar_plot.containers):
        for bar_idx, bar in enumerate(container):
            x_pos = bar.get_x() + bar.get_width() / 2
            height = bar.get_height()
            if x_positions[x_pos]['container_idx'] == container_idx and x_positions[x_pos]['bar_idx'] == bar_idx:
                ax.text(x_pos, height + 0.01, f'{height:.3f}', 
                       ha='center', va='bottom', fontsize=12, fontweight='semibold')
    
    # Optimize y-axis range for data visibility
    ax.set_ylim(0, max(df[metric]) * 1.1)
    
    # Apply consistent grid formatting
    ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.8, axis='y')
    ax.set_axisbelow(True)
    
    # Minimize chart junk for academic presentation
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches="tight", pad_inches=0.3, dpi=300, facecolor='white')
        print(f"✅ Saved professional chart to {save_path}")
    else:
        plt.show()
    plt.close()

def plot_anonymization_impact_heatmap(results, dataset, save_path=None):
    """
    Create a heatmap showing the impact of different anonymization techniques across all models and metrics.
    """
    # Structure data for heatmap visualization
    anonymization_techniques = list(results[dataset].keys())
    models = list(results[dataset][anonymization_techniques[0]].keys())
    metrics = list(results[dataset][anonymization_techniques[0]][models[0]].keys())
    
    # Build performance matrix
    heatmap_data = []
    labels = []
    
    for anon in anonymization_techniques:
        for model in models:
            row = []
            for metric in metrics:
                value = results[dataset][anon][model][metric]
                row.append(value)
            heatmap_data.append(row)
            labels.append(f"{anon}\n{model}")
    
    heatmap_df = pd.DataFrame(heatmap_data, columns=metrics, index=labels)
    
    plt.figure(figsize=(10, 12))
    ax = sns.heatmap(heatmap_df, annot=True, fmt='.3f', cmap='RdYlGn', 
                     cbar_kws={'label': 'Performance Score'}, 
                     linewidths=0.5, linecolor='white', annot_kws={'fontsize': 12})
    ax.set_title(f"{dataset} Dataset - Performance Heatmap\nAnonymization Impact Across Models and Metrics", 
                 fontsize=20, fontweight='bold', pad=20)
    ax.set_xlabel("Metrics", fontsize=18)
    ax.set_ylabel("Anonymization Technique & Model", fontsize=18)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches="tight", pad_inches=0.1, dpi=300)
        print(f"Saved heatmap to {save_path}")
    else:
        plt.show()
    plt.close()

def plot_performance_degradation(results, dataset, save_path=None):
    """
    Create a professional line plot showing performance degradation from original to anonymized versions.
    """
    # Establish baseline performance metrics
    original_data = results[dataset]["Original"]
    
    # Structure data for degradation analysis
    plot_data = []
    anonymization_order = ["Original", 
                          "Micro Aggregation (Minimal)", "Micro Aggregation (Medium)", "Micro Aggregation (High)",
                          "Differential Privacy (Minimal)", "Differential Privacy (Medium)", "Differential Privacy (High)",
                          "Randomized Response (Minimal)", "Randomized Response (Medium)", "Randomized Response (High)"]
    
    for anon in anonymization_order:
        if anon in results[dataset]:
            for model, metrics in results[dataset][anon].items():
                for metric, value in metrics.items():
                    plot_data.append({
                        "Anonymization": anon,
                        "Model": model,
                        "Metric": metric,
                        "Performance": value
                    })
    
    df = pd.DataFrame(plot_data)
    
    # Determine metric set for adaptive layout
    available_metrics = list(original_data[list(original_data.keys())[0]].keys())
    
    # Configure subplot grid based on metric count
    if len(available_metrics) == 4:
        fig, axes = plt.subplots(2, 2, figsize=(18, 14))
        axes = axes.flatten()
    elif len(available_metrics) == 7:
        fig, axes = plt.subplots(3, 3, figsize=(24, 18))
        axes = axes.flatten()
        # Hide unused subplots
        for i in range(len(available_metrics), len(axes)):
            axes[i].set_visible(False)
    else:
        # Fallback for other metric counts
        rows = (len(available_metrics) + 2) // 3
        fig, axes = plt.subplots(rows, 3, figsize=(20, 6*rows))
        axes = axes.flatten() if rows > 1 else [axes] if rows == 1 else axes
    
    for i, metric in enumerate(available_metrics):
        ax = axes[i]
        metric_data = df[df["Metric"] == metric]
        
        for j, model in enumerate(["Logistic Regression", "Random Forest", "XGBoost"]):
            model_data = metric_data[metric_data["Model"] == model]
            if not model_data.empty:
                x_pos = range(len(model_data))
                ax.plot(x_pos, model_data["Performance"], 
                       marker='o', linewidth=3.5, markersize=10, 
                       label=model, color=PROFESSIONAL_COLORS[j],
                       markerfacecolor='white', markeredgewidth=2,
                       markeredgecolor=PROFESSIONAL_COLORS[j])
        
        # Apply academic formatting
        ax.set_title(f"{metric.replace('_', ' ').title()} Performance Trends", 
                    fontsize=20, fontweight='bold', pad=15)
        ax.set_ylabel(f"{metric.replace('_', ' ').title()}", fontsize=18, fontweight='semibold')
        ax.set_xlabel("Anonymization Technique", fontsize=18, fontweight='semibold')
        ax.set_xticks(range(len(anonymization_order)))
        ax.set_xticklabels([anon.replace(" (", "\n(") for anon in anonymization_order], 
                          rotation=45, ha='right', fontsize=15)
        
        # Configure grid and visual elements
        ax.grid(True, alpha=0.4, linestyle='-', linewidth=0.8)
        ax.set_axisbelow(True)
        
        # Single legend to avoid redundancy
        if i == 0:
            legend = ax.legend(fontsize=16, title="ML Model", title_fontsize=17,
                              loc='upper right', frameon=True, fancybox=True, shadow=True)
            legend.get_frame().set_facecolor('white')
            legend.get_frame().set_alpha(0.9)
        
        ax.set_ylim(0, 1.05)
        ax.tick_params(labelsize=15)
    
    plt.suptitle(f"{dataset} Dataset\nML Performance Degradation Analysis Across Anonymization Techniques", 
                 fontsize=24, fontweight='bold', y=0.98)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches="tight", pad_inches=0.3, dpi=300, facecolor='white')
        print(f"✅ Saved professional degradation analysis to {save_path}")
    else:
        plt.show()
    plt.close()
    
    # Generate supplementary metric-specific analyses
    if len(available_metrics) >= 7:  # Generate grouped analyses for comprehensive datasets
        # Primary classification metrics
        core_metrics = ['accuracy', 'precision', 'recall', 'f1_score']
        available_core = [m for m in core_metrics if m in available_metrics]
        
        if len(available_core) >= 2:  # Ensure sufficient metrics for meaningful analysis
            _create_grouped_plot(df, available_core, dataset, "Core Metrics (Accuracy, Precision, Recall, F1)", 
                               save_path, "_core_metrics")
        
        # Advanced performance indicators
        extended_metrics = ['balanced_accuracy', 'matthews_correlation', 'roc_auc']
        available_extended = [m for m in extended_metrics if m in available_metrics]
        
        if len(available_extended) >= 2:  # Validate metric availability
            _create_grouped_plot(df, available_extended, dataset, "Extended Metrics (Balanced Accuracy, MCC, ROC AUC)", 
                               save_path, "_extended_metrics")

def _create_grouped_plot(df, selected_metrics, dataset, plot_title, base_save_path, suffix):
    """Generate metric-specific degradation plots with optimized layout"""
    
    # Extract relevant metric subset
    filtered_df = df[df['Metric'].isin(selected_metrics)]
    
    # Configure subplot arrangement
    num_metrics = len(selected_metrics)
    if num_metrics <= 2:
        fig, axes = plt.subplots(1, num_metrics, figsize=(9 * num_metrics, 7))
        if num_metrics == 1:
            axes = [axes]
    elif num_metrics <= 4:
        fig, axes = plt.subplots(2, 2, figsize=(18, 14))
        axes = axes.flatten()
    else:
        fig, axes = plt.subplots(2, 3, figsize=(24, 14))
        axes = axes.flatten()
    
    models = sorted(filtered_df['Model'].unique())
    
    # Generate individual metric visualizations
    for i, metric in enumerate(selected_metrics):
        ax = axes[i]
        metric_data = filtered_df[filtered_df['Metric'] == metric]
        
        for j, model in enumerate(models):
            model_data = metric_data[metric_data['Model'] == model]
            
            ax.plot(model_data['Anonymization'], model_data['Performance'], 
                   marker='o', linestyle='-', linewidth=3, markersize=8,
                   label=model, color=PROFESSIONAL_COLORS[j],
                   markerfacecolor='white', markeredgewidth=2,
                   markeredgecolor=PROFESSIONAL_COLORS[j])
        
        # Apply academic formatting
        ax.set_title(f"{metric.replace('_', ' ').title()} Performance Trends", 
                    fontsize=20, fontweight='bold', pad=15)
        ax.set_ylabel(f"{metric.replace('_', ' ').title()}", fontsize=18, fontweight='semibold')
        ax.set_xlabel("Anonymization Technique", fontsize=18, fontweight='semibold')
        
        # Rotate x-axis labels
        ax.tick_params(axis='x', rotation=45, labelsize=15)
        ax.tick_params(axis='y', labelsize=15)
        
        ax.grid(True, alpha=0.4, linestyle='-', linewidth=0.8)
        ax.set_axisbelow(True)
        ax.set_ylim(0, 1.05)
    
    # Remove excess subplot panels
    for i in range(len(selected_metrics), len(axes)):
        fig.delaxes(axes[i])
    
    # Position unified legend
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.95), 
               ncol=len(models), fontsize=17, title="Machine Learning Model", 
               title_fontsize=18, frameon=True, fancybox=True, shadow=True)
    
    # Eliminate redundant legends
    for ax in axes[:len(selected_metrics)]:
        ax.legend().remove() if ax.get_legend() else None
    
    plt.suptitle(f"{dataset} Dataset\n{plot_title} Degradation Analysis", 
                 fontsize=24, fontweight='bold', y=0.88)
    plt.tight_layout(rect=[0, 0, 1, 0.85])
    
    if base_save_path:
        grouped_save_path = base_save_path.replace('.png', f'{suffix}.png')
        plt.savefig(grouped_save_path, bbox_inches="tight", pad_inches=0.3, dpi=300, facecolor='white')
        print(f"✅ Saved {plot_title.lower()} degradation analysis to {grouped_save_path}")
    else:
        plt.show()
    plt.close()

def plot_anonymization_comparison(results, dataset, save_path=None):
    """
    Create a professional comprehensive comparison showing the trade-off between privacy and utility.
    """
    # Quantify privacy intensity for trade-off analysis
    privacy_levels = {
        "Original": 0,
        "Micro Aggregation (Minimal)": 1,
        "Micro Aggregation (Medium)": 2,
        "Micro Aggregation (High)": 3,
        "Differential Privacy (Minimal)": 4,
        "Differential Privacy (Medium)": 5,
        "Differential Privacy (High)": 6,
        "Randomized Response (Minimal)": 7,
        "Randomized Response (Medium)": 8,
        "Randomized Response (High)": 9
    }
    
    # Structure data for privacy-utility visualization
    plot_data = []
    for anon_tech, models in results[dataset].items():
        privacy_score = privacy_levels.get(anon_tech, 0)
        for model_name, metrics in models.items():
            # Use accuracy as utility measure
            utility = metrics.get('accuracy', 0)
            plot_data.append({
                'Privacy_Level': privacy_score,
                'Utility_Score': utility,
                'Model': model_name,
                'Anonymization': anon_tech
            })
    
    df = pd.DataFrame(plot_data)
    
    # Initialize scatter plot with academic formatting
    fig, ax = plt.subplots(figsize=(14, 10))
    
    # Generate model-specific scatter points
    markers = ['o', 's', '^']  # circle, square, triangle
    for i, model in enumerate(["Logistic Regression", "Random Forest", "XGBoost"]):
        model_data = df[df['Model'] == model]
        scatter = ax.scatter(model_data['Privacy_Level'], model_data['Utility_Score'],
                           c=PROFESSIONAL_COLORS[i], marker=markers[i], 
                           s=150, alpha=0.8, edgecolors='white', linewidth=2,
                           label=model)
    
    # Apply academic formatting standards
    ax.set_title(f"{dataset} Dataset\nPrivacy vs. Utility Trade-off Analysis", 
                fontsize=24, fontweight='bold', pad=25)
    ax.set_xlabel("Privacy Level (Higher = More Private)", fontsize=20, fontweight='semibold')
    ax.set_ylabel("Model Accuracy (Utility)", fontsize=20, fontweight='semibold')
    
    # Configure grid system
    ax.grid(True, alpha=0.4, linestyle='-', linewidth=0.8)
    ax.set_axisbelow(True)
    
    # Position legend with academic standards
    legend = ax.legend(title="Machine Learning Model", fontsize=17, title_fontsize=18,
                      loc='lower left', frameon=True, fancybox=True, shadow=True)
    legend.get_frame().set_facecolor('white')
    legend.get_frame().set_alpha(0.9)
    
    # Configure axis labeling
    ax.set_xticks(range(10))
    ax.set_xticklabels(['Original', 'MA-Min', 'MA-Med', 'MA-High', 
                       'DP-Min', 'DP-Med', 'DP-High',
                       'RR-Min', 'RR-Med', 'RR-High'], 
                      rotation=45, ha='right', fontsize=16)
    
    ax.tick_params(labelsize=16)
    ax.set_ylim(0, 1.05)
    
    # Overlay regression trend
    x_vals = df['Privacy_Level'].values
    y_vals = df['Utility_Score'].values
    z = np.polyfit(x_vals, y_vals, 1)
    p = np.poly1d(z)
    ax.plot(sorted(x_vals), p(sorted(x_vals)), "r--", alpha=0.6, linewidth=2)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches="tight", pad_inches=0.3, dpi=300, facecolor='white')
        print(f"✅ Saved professional privacy-utility analysis to {save_path}")
    else:
        plt.show()
    plt.close()

def plot_percentage_degradation(results, dataset, save_path=None):
    """
    Create a professional bar chart showing percentage degradation from original performance.
    """
    # Establish baseline performance metrics
    original_data = results[dataset]["Original"]
    
    # Compute relative performance loss
    degradation_data = []
    for anon, models in results[dataset].items():
        if anon == "Original":
            continue
            
        for model, metrics in models.items():
            for metric, value in metrics.items():
                original_value = original_data[model][metric]
                if original_value > 0:
                    degradation_pct = ((original_value - value) / original_value) * 100
                    degradation_data.append({
                        "Anonymization": anon,
                        "Model": model,
                        "Metric": metric,
                        "Degradation (%)": degradation_pct
                    })
    
    df = pd.DataFrame(degradation_data)
    
    # Determine metric configuration
    available_metrics = list(original_data[list(original_data.keys())[0]].keys())
    
    # Configure subplot grid
    if len(available_metrics) == 4:
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        axes = axes.flatten()
    elif len(available_metrics) == 7:
        fig, axes = plt.subplots(3, 3, figsize=(20, 15))
        axes = axes.flatten()
        # Hide unused subplots
        for i in range(len(available_metrics), len(axes)):
            axes[i].set_visible(False)
    else:
        # Fallback for other metric counts
        rows = (len(available_metrics) + 2) // 3
        fig, axes = plt.subplots(rows, 3, figsize=(18, 6*rows))
        axes = axes.flatten() if rows > 1 else [axes] if rows == 1 else axes
    
    for i, metric in enumerate(available_metrics):
        ax = axes[i]
        metric_data = df[df["Metric"] == metric]
        
        if not metric_data.empty:
            bar_plot = sns.barplot(data=metric_data, x="Anonymization", y="Degradation (%)", 
                                 hue="Model", ax=ax, palette=PROFESSIONAL_COLORS,
                                 edgecolor='white', linewidth=1.5, alpha=0.9)
        
        ax.set_title(f"{metric.replace('_', ' ').title()} - Performance Degradation", 
                    fontsize=18, fontweight='bold')
        ax.set_ylabel("Performance Degradation (%)", fontsize=16)
        ax.set_xlabel("Anonymization Technique", fontsize=16)
        ax.tick_params(axis='x', rotation=45, labelsize=14)
        ax.tick_params(axis='y', labelsize=15)
        
        # Configure legend formatting
        if i == 0:  # Single legend to avoid redundancy
            legend = ax.legend(fontsize=14, title="ML Algorithm", title_fontsize=15)
            legend.get_frame().set_facecolor('white')
            legend.get_frame().set_alpha(0.9)
        else:
            ax.legend().remove()
            
        ax.grid(True, alpha=0.3, axis='y')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        # Label significant degradations only
        for container in ax.containers:
            for bar in container:
                height = bar.get_height()
                if height > 10:  # Threshold for visual clarity
                    ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                           f'{height:.0f}%', ha='center', va='bottom', 
                           fontsize=8, fontweight='semibold')
    
    plt.suptitle(f"{dataset} Dataset - Performance Degradation Analysis (Percentage)", 
                 fontsize=18, fontweight='bold', y=0.98)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches="tight", pad_inches=0.1, dpi=300)
        print(f"Saved percentage degradation chart to {save_path}")
    else:
        plt.show()
    plt.close()

def plot_model_impact_analysis(results, dataset, save_path=None):
    """
    Create a visualization showing how each anonymization technique affects different models.
    """
    # Structure data for comparative analysis
    original_data = results[dataset]["Original"]
    
    impact_data = []
    for anon, models in results[dataset].items():
        if anon == "Original":
            continue
            
        for model, metrics in models.items():
            # Compute aggregate performance impact
            total_degradation = 0
            metric_count = 0
            
            for metric, value in metrics.items():
                original_value = original_data[model][metric]
                if original_value > 0:
                    degradation_pct = ((original_value - value) / original_value) * 100
                    total_degradation += degradation_pct
                    metric_count += 1
            
            avg_degradation = total_degradation / metric_count if metric_count > 0 else 0
            
            impact_data.append({
                "Anonymization": anon,
                "Model": model,
                "Average Degradation (%)": avg_degradation,
                "Performance After": sum(metrics.values()) / len(metrics)
            })
    
    df = pd.DataFrame(impact_data)
    
    # Initialize dual-panel comparison
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))
    
    # Panel 1: Degradation sensitivity heatmap
    pivot_degradation = df.pivot(index="Anonymization", columns="Model", values="Average Degradation (%)")
    sns.heatmap(pivot_degradation, annot=True, fmt='.1f', cmap='Reds', ax=ax1, 
                cbar_kws={'label': 'Average Degradation (%)'}, annot_kws={'fontsize': 12})
    ax1.set_title("Model Sensitivity to Anonymization\n(Average Degradation %)", 
                  fontsize=18, fontweight='bold')
    ax1.set_xlabel("Model", fontsize=16)
    ax1.set_ylabel("Anonymization Technique", fontsize=16)
    
    # Panel 2: Absolute performance retention
    pivot_performance = df.pivot(index="Anonymization", columns="Model", values="Performance After")
    sns.heatmap(pivot_performance, annot=True, fmt='.3f', cmap='RdYlGn', ax=ax2,
                cbar_kws={'label': 'Performance Score'}, annot_kws={'fontsize': 12})
    ax2.set_title("Remaining Performance After Anonymization\n(Absolute Scores)", 
                  fontsize=18, fontweight='bold')
    ax2.set_xlabel("Model", fontsize=16)
    ax2.set_ylabel("Anonymization Technique", fontsize=16)
    
    plt.suptitle(f"{dataset} Dataset - Model Impact Analysis", 
                 fontsize=22, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches="tight", pad_inches=0.1, dpi=300)
        print(f"Saved model impact analysis to {save_path}")
    else:
        plt.show()
    plt.close()

def plot_anonymization_technique_ranking(results, dataset, save_path=None):
    """
    Create a ranking visualization showing which anonymization techniques preserve performance best.
    """
    # Calculate average performance for each anonymization technique
    technique_performance = {}
    
    for anon, models in results[dataset].items():
        total_performance = 0
        count = 0
        
        for model, metrics in models.items():
            for metric, value in metrics.items():
                total_performance += value
                count += 1
        
        avg_performance = total_performance / count if count > 0 else 0
        technique_performance[anon] = avg_performance
    
    # Sort by performance
    sorted_techniques = sorted(technique_performance.items(), key=lambda x: x[1], reverse=True)
    
    techniques = [item[0] for item in sorted_techniques]
    performances = [item[1] for item in sorted_techniques]
    
    # Create color map (green for best, red for worst)
    colors = plt.cm.RdYlGn([p for p in performances])
    
    plt.figure(figsize=(12, 8))
    bars = plt.barh(range(len(techniques)), performances, color=colors)
    
    plt.yticks(range(len(techniques)), [t.replace(" (", "\n(") for t in techniques])
    plt.xlabel("Average Performance Score", fontsize=18)
    plt.title(f"{dataset} Dataset - Anonymization Technique Ranking\n(By Average Performance Preservation)", 
              fontsize=20, fontweight='bold')
    
    # Add value labels
    for i, (bar, perf) in enumerate(zip(bars, performances)):
        plt.text(perf + 0.01, bar.get_y() + bar.get_height()/2, 
                f'{perf:.3f}', va='center', fontsize=14, fontweight='bold')
    
    plt.grid(True, alpha=0.3, axis='x')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches="tight", pad_inches=0.1, dpi=300)
        print(f"Saved technique ranking to {save_path}")
    else:
        plt.show()
    plt.close()

# Generate all Iris visualizations
def generate_dataset_visualizations(dataset_name):
    """Generate comprehensive visualization suite for specified dataset"""
    dataset = dataset_name
    
    if dataset not in results:
        print(f"No data found for dataset: {dataset}")
        return
    
    print(f"Generating {dataset} dataset visualizations...")
    
    # Initialize output directory structure
    output_dir = f"{dataset.lower().replace(' ', '_')}_visualizations"
    os.makedirs(output_dir, exist_ok=True)
    
    # Determine dataset-specific metrics
    sample_model = list(results[dataset]["Original"].keys())[0]
    available_metrics = list(results[dataset]["Original"][sample_model].keys())
    
    # Create metric-specific comparative visualizations
    for metric in available_metrics:
        plot_summary_bar(results, dataset, metric, 
                        save_path=f"{output_dir}/{dataset.lower().replace(' ', '_')}_{metric}_comparison.png")
    
    # Create performance correlation matrix
    plot_anonymization_impact_heatmap(results, dataset, 
                                    save_path=f"{output_dir}/{dataset.lower().replace(' ', '_')}_performance_heatmap.png")
    
    # Analyze temporal performance trends
    plot_performance_degradation(results, dataset, 
                                save_path=f"{output_dir}/{dataset.lower().replace(' ', '_')}_performance_degradation.png")
    
    # Visualize privacy-utility relationship
    plot_anonymization_comparison(results, dataset, 
                                save_path=f"{output_dir}/{dataset.lower().replace(' ', '_')}_privacy_utility_tradeoff.png")
    
    # Quantify relative performance loss
    plot_percentage_degradation(results, dataset,
                               save_path=f"{output_dir}/{dataset.lower().replace(' ', '_')}_percentage_degradation.png")
    
    # Assess model-specific vulnerabilities
    plot_model_impact_analysis(results, dataset,
                              save_path=f"{output_dir}/{dataset.lower().replace(' ', '_')}_model_impact_analysis.png")
    
    # Rank techniques by effectiveness
    plot_anonymization_technique_ranking(results, dataset,
                                       save_path=f"{output_dir}/{dataset.lower().replace(' ', '_')}_technique_ranking.png")
    
    print(f"All {dataset} visualizations saved to '{output_dir}' directory!")
    print(f"Generated charts for metrics: {', '.join(available_metrics)}")

def generate_combined_dataset_comparison(save_path_prefix="combined_analysis"):
    """Generate cross-dataset comparative analysis suite"""
    
    if not results:
        print("No data available for combined analysis")
        return
    
    print("Generating combined dataset comparison visualizations...")
    
    # Initialize comparative analysis directory
    output_dir = "combined_visualizations"
    os.makedirs(output_dir, exist_ok=True)
    
    # Cross-dataset performance analysis
    plot_combined_dataset_performance(save_path=f"{output_dir}/{save_path_prefix}_performance_comparison.png")
    
    # Technique effectiveness comparison
    plot_combined_technique_effectiveness(save_path=f"{output_dir}/{save_path_prefix}_technique_effectiveness.png")
    
    # Model robustness assessment
    plot_combined_model_robustness(save_path=f"{output_dir}/{save_path_prefix}_model_robustness.png")
    
    print(f"Combined visualizations saved to '{output_dir}' directory!")

def plot_combined_dataset_performance(save_path=None):
    """Generate comprehensive cross-dataset performance analysis"""
    
    # Structure data for comparative analysis
    combined_data = []
    
    for dataset, anon_data in results.items():
        for anon, models in anon_data.items():
            for model, metrics in models.items():
                avg_performance = sum(metrics.values()) / len(metrics)
                
                # Compute relative performance degradation
                if anon != "Original":
                    original_avg = sum(anon_data["Original"][model].values()) / len(anon_data["Original"][model])
                    degradation_pct = ((original_avg - avg_performance) / original_avg) * 100 if original_avg > 0 else 0
                else:
                    degradation_pct = 0
                
                combined_data.append({
                    "Dataset": dataset,
                    "Anonymization": anon,
                    "Model": model,
                    "Average Performance": avg_performance,
                    "Degradation (%)": degradation_pct
                })
    
    df = pd.DataFrame(combined_data)
    
    # Initialize multi-panel analysis layout
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 16))
    
    # Panel 1: Cross-dataset performance matrix
    performance_pivot = df[df["Anonymization"] != "Original"].pivot_table(
        index=["Dataset", "Anonymization"], 
        columns="Model", 
        values="Average Performance"
    )
    sns.heatmap(performance_pivot, annot=True, fmt='.3f', cmap='RdYlGn', ax=ax1)
    ax1.set_title("Performance Across All Datasets\n(After Anonymization)", fontsize=18, fontweight='bold')
    
    # Panel 2: Degradation impact analysis
    degradation_pivot = df[df["Anonymization"] != "Original"].pivot_table(
        index=["Dataset", "Anonymization"],
        columns="Model",
        values="Degradation (%)"
    )
    sns.heatmap(degradation_pivot, annot=True, fmt='.1f', cmap='Reds', ax=ax2)
    ax2.set_title("Performance Degradation %\n(Across All Datasets)", fontsize=18, fontweight='bold')
    
    # Panel 3: Dataset vulnerability assessment
    dataset_sensitivity = df.groupby(["Dataset", "Anonymization"])["Degradation (%)"].mean().reset_index()
    dataset_sensitivity = dataset_sensitivity[dataset_sensitivity["Anonymization"] != "Original"]
    
    sns.boxplot(data=dataset_sensitivity, x="Dataset", y="Degradation (%)", ax=ax3)
    ax3.set_title("Dataset Sensitivity to Anonymization\n(Distribution of Degradation)", fontsize=18, fontweight='bold')
    ax3.tick_params(axis='x', rotation=45)
    
    # Panel 4: Model resilience comparison
    model_robustness = df.groupby(["Model", "Dataset"])["Degradation (%)"].mean().reset_index()
    model_robustness = model_robustness[model_robustness["Dataset"] != "Original"]
    
    sns.barplot(data=model_robustness, x="Model", y="Degradation (%)", hue="Dataset", ax=ax4)
    ax4.set_title("Model Robustness Across Datasets\n(Average Degradation)", fontsize=18, fontweight='bold')
    ax4.legend(title="Dataset", bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.suptitle("Comprehensive Cross-Dataset Analysis", fontsize=24, fontweight='bold', y=0.98)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches="tight", pad_inches=0.1, dpi=300)
        print(f"Saved combined dataset performance to {save_path}")
    else:
        plt.show()
    plt.close()

def plot_combined_technique_effectiveness(save_path=None):
    """Analyze technique effectiveness across dataset types"""
    
    # Compute cross-dataset technique performance
    technique_data = []
    
    for dataset, anon_data in results.items():
        original_performance = {}
        for model, metrics in anon_data["Original"].items():
            original_performance[model] = sum(metrics.values()) / len(metrics)
        
        for anon, models in anon_data.items():
            if anon == "Original":
                continue
                
            total_preservation = 0
            count = 0
            
            for model, metrics in models.items():
                current_performance = sum(metrics.values()) / len(metrics)
                preservation_rate = (current_performance / original_performance[model]) * 100
                total_preservation += preservation_rate
                count += 1
            
            avg_preservation = total_preservation / count if count > 0 else 0
            
            technique_data.append({
                "Dataset": dataset,
                "Anonymization": anon,
                "Performance Preservation (%)": avg_preservation
            })
    
    df = pd.DataFrame(technique_data)
    
    plt.figure(figsize=(16, 10))
    
    # Generate comparative effectiveness visualization
    sns.barplot(data=df, x="Anonymization", y="Performance Preservation (%)", hue="Dataset", palette="deep")
    
    plt.title("Anonymization Technique Effectiveness Across Datasets\n(Performance Preservation %)", 
              fontsize=22, fontweight='bold')
    plt.xlabel("Anonymization Technique", fontsize=18)
    plt.ylabel("Performance Preservation (%)", fontsize=18)
    plt.xticks(rotation=45, ha='right')
    plt.legend(title="Dataset", bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Mark acceptable performance threshold
    plt.axhline(y=90, color='red', linestyle='--', alpha=0.7, label='90% Preservation Threshold')
    
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches="tight", pad_inches=0.1, dpi=300)
        print(f"Saved technique effectiveness comparison to {save_path}")
    else:
        plt.show()
    plt.close()

def plot_combined_model_robustness(save_path=None):
    """Assess model resilience across datasets and techniques"""
    
    # Compute model vulnerability metrics
    robustness_data = []
    
    for dataset, anon_data in results.items():
        for model in anon_data["Original"].keys():
            original_performance = sum(anon_data["Original"][model].values()) / len(anon_data["Original"][model])
            
            total_degradation = 0
            anon_count = 0
            
            for anon, models in anon_data.items():
                if anon == "Original":
                    continue
                    
                current_performance = sum(models[model].values()) / len(models[model])
                degradation = ((original_performance - current_performance) / original_performance) * 100
                total_degradation += degradation
                anon_count += 1
            
            avg_degradation = total_degradation / anon_count if anon_count > 0 else 0
            
            robustness_data.append({
                "Dataset": dataset,
                "Model": model,
                "Average Degradation (%)": avg_degradation,
                "Robustness Score": max(0, 100 - avg_degradation)  # Inverse relationship: higher score indicates greater robustness
            })
    
    df = pd.DataFrame(robustness_data)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))
    
    # Panel 1: Vulnerability assessment
    sns.barplot(data=df, x="Model", y="Average Degradation (%)", hue="Dataset", ax=ax1)
    ax1.set_title("Model Vulnerability to Anonymization\n(Average Degradation Across Techniques)", 
                  fontsize=14, fontweight='bold')
    ax1.legend(title="Dataset", bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Panel 2: Resilience scoring
    sns.barplot(data=df, x="Model", y="Robustness Score", hue="Dataset", ax=ax2)
    ax2.set_title("Model Robustness Score\n(Higher = More Robust to Anonymization)", 
                  fontsize=14, fontweight='bold')
    ax2.legend(title="Dataset", bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.suptitle("Model Robustness Analysis Across All Datasets", fontsize=18, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches="tight", pad_inches=0.1, dpi=300)
        print(f"Saved model robustness analysis to {save_path}")
    else:
        plt.show()
    plt.close()

# Convenience functions for generating specific dataset visualizations
def generate_iris_visualizations():
    """Create complete Iris dataset analysis suite"""
    generate_dataset_visualizations("Iris")

def generate_breast_cancer_visualizations():
    """Create complete Breast Cancer dataset analysis suite"""
    generate_dataset_visualizations("Breast Cancer")

def generate_handwritten_digits_visualizations():
    """Create complete Handwritten Digits dataset analysis suite"""
    generate_dataset_visualizations("Handwritten Digits")

def generate_adult_income_visualizations():
    """Create complete Adult Income dataset analysis suite"""
    generate_dataset_visualizations("Adult Income")

def generate_all_visualizations():
    """Execute comprehensive analysis across all datasets"""
    available_datasets = list(results.keys())
    
    print(f"Generating visualizations for all datasets: {available_datasets}")
    
    # Process each dataset independently
    for dataset in available_datasets:
        generate_dataset_visualizations(dataset)
    
    # Execute cross-dataset comparative analysis
    if len(available_datasets) > 1:
        generate_combined_dataset_comparison()
    
    print("All visualizations generated successfully!")

# Execute dataset-specific analysis workflows
generate_iris_visualizations()                    # Botanical classification dataset
generate_breast_cancer_visualizations()           # Medical diagnostic dataset
generate_handwritten_digits_visualizations()      # Computer vision dataset

# Execute cross-dataset comparative analysis
generate_combined_dataset_comparison()            # Privacy-utility trade-off patterns across domains

# Execute comprehensive analysis pipeline
generate_all_visualizations()                     # Complete academic visualization suite
