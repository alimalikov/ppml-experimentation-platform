import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectKBest, SelectPercentile, RFE
from sklearn.preprocessing import OneHotEncoder, TargetEncoder
from sklearn.feature_selection import SelectFromModel

import numpy as np
import os
import sys
import ast
import re
import tempfile
import importlib.util
import traceback
from datetime import datetime

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

SRC_DIR = os.path.join(PROJECT_ROOT, 'src')
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

import streamlit as st
st.set_page_config(layout="wide", page_title="ML Experimentation Platform")

try:
    from src.ml_plugins.visualizations.ppml_comparison_visualization_plugin_v2 import PPMLComparisonVisualization
    PPML_VISUALIZATION_AVAILABLE = True
except ImportError as e:
    st.error(f"Professional PPML Visualization plugin not found: {e}")
    # Create a fallback class to prevent the app from crashing
    class PPMLComparisonVisualization:
        def __init__(self):
            self.name = "PPML Comparison (Not Available)"
        
        def get_config_ui(self, key_prefix):
            st.warning("Professional PPML Comparison Visualization is not available. Please check if the plugin file exists.")
            return None
        
        def render(self, data, model_results, config):
            st.error("PPML Visualization plugin is not properly installed.")
            return False
    
    PPML_VISUALIZATION_AVAILABLE = False

try:
    from src.ml_plugins.plugin_manager import get_plugin_manager
    from src.ml_plugins.metric_manager import get_metric_manager
    from src.ml_plugins.visualization_manager import get_visualization_manager 
    PLUGINS_AVAILABLE = True
    VISUALIZATION_PLUGINS_AVAILABLE = True
except ImportError as e:
    st.error(f"Failed to import plugin managers: {e}")
    PLUGINS_AVAILABLE = False
    VISUALIZATION_PLUGINS_AVAILABLE = False
    st.stop()


def _extract_percentage_value(percentage_str):
    """Helper function to extract numeric value from percentage string with arrows"""
    try:
        if isinstance(percentage_str, str) and (' ' in percentage_str):
            numeric_part = percentage_str.split()[1].rstrip('%')
            return float(numeric_part)
        return 0.0
    except (IndexError, ValueError, AttributeError):
        return 0.0

ML_PLUGIN_SNIPPETS = {
    "--- Select a Snippet ---": "",
    
    # === ML ALGORITHM SNIPPETS ===
# Replace lines 46-146 with this corrected template:

    "ML Algorithm - Basic Template": """import streamlit as st
import pandas as pd
import numpy as np
from typing import Dict, Any
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array

# Import for plugin system - will be auto-fixed during save
try:
    from ....base_ml_plugin import MLPlugin
except ImportError:
    # Fallback for testing
    import sys
    import os
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))
    sys.path.append(project_root)
    from src.ml_plugins.base_ml_plugin import MLPlugin

class MyMLAlgorithm(BaseEstimator, ClassifierMixin, MLPlugin):
    def __init__(self, learning_rate=0.01, max_iterations=100):
        super().__init__()
        self.learning_rate = learning_rate
        self.max_iterations = max_iterations
        
        # Required plugin metadata
        self._name = "My ML Algorithm"
        self._description = "Description of my custom ML algorithm"
        self._category = "Custom"
        
        # Required capability flags - THESE ARE ESSENTIAL!
        self._supports_classification = True
        self._supports_regression = False
        self._min_samples_required = 50
        
    def get_name(self) -> str:
        return self._name
        
    def get_description(self) -> str:
        return self._description
        
    def get_category(self) -> str:
        return self._category
        
    def fit(self, X, y):
        X, y = check_X_y(X, y)
        # TODO: Implement your training logic here
        self.classes_ = np.unique(y)
        self.is_fitted_ = True
        return self
        
    def predict(self, X):
        X = check_array(X)
        # TODO: Implement your prediction logic here
        return np.random.choice(self.classes_, size=X.shape[0])
        
    def predict_proba(self, X):
        X = check_array(X)
        n_classes = len(self.classes_)
        # TODO: Implement probability predictions
        return np.random.dirichlet(np.ones(n_classes), size=X.shape[0])
        
    def get_hyperparameter_config(self, key_prefix: str) -> Dict[str, Any]:
        \"\"\"Generate Streamlit UI for hyperparameters\"""
        st.sidebar.subheader(f"{self.get_name()} Configuration")
        
        learning_rate = st.sidebar.number_input(
            "Learning Rate:", 
            value=self.learning_rate, 
            min_value=0.001, 
            max_value=1.0, 
            step=0.001,
            key=f"{key_prefix}_learning_rate"
        )
        
        max_iterations = st.sidebar.number_input(
            "Max Iterations:", 
            value=self.max_iterations, 
            min_value=10, 
            max_value=1000, 
            step=10,
            key=f"{key_prefix}_max_iterations"
        )
        
        return {
            "learning_rate": learning_rate,
            "max_iterations": max_iterations
        }
        
    def create_model_instance(self, hyperparameters: Dict[str, Any]):
        \"\"\"Create model instance with given hyperparameters\"""
        return MyMLAlgorithm(
            learning_rate=hyperparameters.get("learning_rate", self.learning_rate),
            max_iterations=hyperparameters.get("max_iterations", self.max_iterations)
        )
        
    def preprocess_data(self, X, y):
        \"\"\"Optional data preprocessing\"""
        return X, y
        
    def is_compatible_with_data(self, df, target_column):
        \"\"\"Check if algorithm is compatible with the data\"""
        try:
            # TODO: Add your compatibility checks
            return True, "Compatible"
        except Exception as e:
            return False, str(e)

def get_plugin():
    return MyMLAlgorithm()
""",    # === METRICS SNIPPETS ===
# Replace the "Metric - Basic Template" in ML_PLUGIN_SNIPPETS (around lines 150-200) with this corrected version:
    "Metric - Basic Template": """import streamlit as st
import numpy as np
from typing import Any, Optional
from sklearn.metrics import accuracy_score

# Import for plugin system - will be auto-fixed during save
try:
    from src.ml_plugins.base_metric_plugin import MetricPlugin
except ImportError:
    # Fallback for testing
    import sys
    import os
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))
    sys.path.append(project_root)
    from src.ml_plugins.base_metric_plugin import MetricPlugin

class MyCustomMetric(MetricPlugin):
    def __init__(self):
        super().__init__()
        self._name = "My Custom Metric"
        self._description = "Description of my custom evaluation metric"
        self._category = "Custom"
        self._supports_classification = True
        self._supports_regression = False
        self._requires_probabilities = False
        self._higher_is_better = True
        self._range = (0.0, 1.0)
        
    def get_name(self) -> str:
        return self._name
        
    def get_description(self) -> str:
        return self._description
        
    def calculate(self, y_true: np.ndarray, y_pred: np.ndarray, y_proba: Optional[np.ndarray] = None, **kwargs) -> float:
        \"\"\"Calculate the metric value\"\"\"
        try:
            # TODO: Implement your metric calculation
            # Example: return accuracy_score(y_true, y_pred)
            return float(np.random.random())  # Placeholder - replace with actual calculation
        except Exception as e:
            raise ValueError(f"Error calculating metric: {str(e)}")
        
    def get_interpretation(self, value: float) -> str:
        \"\"\"Provide interpretation of the metric value\"\"\"
        if value >= 0.9:
            return "Excellent"
        elif value >= 0.8:
            return "Good"
        elif value >= 0.7:
            return "Fair"
        else:
            return "Poor"

def get_metric_plugin() -> MetricPlugin:
    return MyCustomMetric()
""",
# === VISUALIZATION SNIPPETS ===
    "Visualization - Basic Template": """import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, Any, List

# Import for plugin system - will be auto-fixed during save
try:
    from ....base_visualization import BaseVisualization
except ImportError:
    # Fallback for testing
    import sys
    import os
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))
    sys.path.append(project_root)
    from src.ml_plugins.base_visualization import BaseVisualization

class MyCustomVisualization(BaseVisualization):
    def __init__(self):
        self._name = "My Custom Visualization"
        self._description = "Description of my custom visualization"
        self._category = "Custom"
        
    def get_name(self) -> str:
        return self._name
        
    def get_description(self) -> str:
        return self._description
        
    def get_category(self) -> str:
        return self._category
        
    def get_supported_data_types(self) -> List[str]:
        \"\"\"Return list of supported data types\"""
        return ["classification", "regression", "binary"]
        
    def is_compatible_with_data(self, data_type: str, model_results: List[Dict], data: pd.DataFrame) -> bool:
        \"""Check compatibility with current data and results\"""
        # TODO: Add your compatibility logic
        return len(model_results) >= 1  # Example: need at least 1 model
        
    def get_config_ui(self, key_prefix: str) -> Dict[str, Any]:
        \"""Generate configuration UI\"""
        st.subheader(f"{self.get_name()} Configuration")
        
        chart_type = st.selectbox(
            "Chart Type:",
            options=["Bar", "Line", "Scatter"],
            key=f"{key_prefix}_chart_type"
        )
        
        show_legend = st.checkbox(
            "Show Legend",
            value=True,
            key=f"{key_prefix}_show_legend"
        )
        
        return {
            "chart_type": chart_type,
            "show_legend": show_legend
        }
        
    def render(self, data: pd.DataFrame, model_results: List[Dict], config: Dict[str, Any]) -> bool:
        \"""Render the visualization\"""
        try:
            st.subheader(f"{self.get_name()}")
            
            # TODO: Implement your visualization logic
            # Example: Create a simple chart
            if model_results:
                metrics_data = []
                for result in model_results:
                    metrics_data.append({
                        "Model": result.get("model_name", "Unknown"),
                        "Accuracy": result.get("accuracy", 0)
                    })
                
                df_metrics = pd.DataFrame(metrics_data)
                
                if config.get("chart_type") == "Bar":
                    fig = px.bar(df_metrics, x="Model", y="Accuracy", 
                               title="Model Accuracy Comparison")
                else:
                    fig = px.line(df_metrics, x="Model", y="Accuracy",
                                title="Model Accuracy Comparison")
                
                fig.update_layout(showlegend=config.get("show_legend", True))
                st.plotly_chart(fig, use_container_width=True)
                
            return True
            
        except Exception as e:
            st.error(f"Error rendering visualization: {e}")
            return False

def get_plugin():
    return MyCustomVisualization()
"""
}
# Replace the validate_ml_plugin_code function (around lines 250-350) with this corrected version:

def validate_ml_plugin_code(code_str: str, class_name_str: str, plugin_type: str) -> list:
    """Validate ML plugin code based on type"""
    results = []
    
    if not code_str.strip():
        results.append({
            'status': '❌ Failed',
            'check': 'Code Presence',
            'message': 'No code provided'
        })
        return results
    
    # Basic syntax check
    try:
        ast.parse(code_str)
        results.append({
            'status': '✅ Passed',
            'check': 'Python Syntax',
            'message': 'Code is syntactically valid'
        })
    except SyntaxError as e:
        results.append({
            'status': '❌ Failed',
            'check': 'Python Syntax',
            'message': f"Line {e.lineno}: {e.msg}"
        })
        return results
    
    # Check for correct factory function based on plugin type
    if plugin_type == "ML Algorithm":
        factory_function = "def get_plugin"
        required_methods = ['fit', 'predict', 'get_name', 'get_hyperparameter_config']
        base_import = "base_ml_plugin"
    elif plugin_type == "Metric":
        factory_function = "def get_metric_plugin"  # FIXED: Use correct function name for metrics
        required_methods = ['calculate', 'get_name', 'get_description']
        base_import = "base_metric"
    elif plugin_type == "Visualization":
        factory_function = "def get_plugin"
        required_methods = ['render', 'get_name', 'is_compatible_with_data']
        base_import = "base_visualization"
    else:
        factory_function = "def get_plugin"
        required_methods = []
        base_import = ""
    
    # Check for factory function
    if factory_function not in code_str:
        results.append({
            'status': '❌ Failed',
            'check': f'{factory_function}() function',
            'message': f"Missing '{factory_function}()' factory function"
        })
    else:
        results.append({
            'status': '✅ Passed',
            'check': f'{factory_function}() function',
            'message': ''
        })
    
    # Check class exists
    if f"class {class_name_str}" not in code_str:
        results.append({
            'status': '❌ Failed',
            'check': f"Class '{class_name_str}' exists",
            'message': f"Class '{class_name_str}' not found"
        })
    else:
        results.append({
            'status': '✅ Passed',
            'check': f"Class '{class_name_str}' exists",
            'message': ''
        })
    
    # Check required methods
    missing_methods = []
    for method in required_methods:
        if f"def {method}" not in code_str:
            missing_methods.append(method)
    
    if missing_methods:
        results.append({
            'status': '❌ Failed',
            'check': 'Required Methods',
            'message': f"Missing: {', '.join(missing_methods)}"
        })
    else:
        results.append({
            'status': '✅ Passed',
            'check': 'Required Methods',
            'message': ''
        })
    
    # Check base class import
    if base_import and base_import not in code_str:
        results.append({
            'status': '❌ Failed',
            'check': 'Base Class Import',
            'message': f"Missing import from {base_import}"
        })
    else:
        results.append({
            'status': '✅ Passed',
            'check': 'Base Class Import',
            'message': ''
        })
    
    return results

# Add session state initialization for plugin development
if "ml_plugin_type" not in st.session_state:
    st.session_state.ml_plugin_type = "ML Algorithm"
if "ml_plugin_raw_code" not in st.session_state:
    st.session_state.ml_plugin_raw_code = ""
if "ml_plugin_class_name" not in st.session_state:
    st.session_state.ml_plugin_class_name = ""
if "ml_plugin_display_name" not in st.session_state:
    st.session_state.ml_plugin_display_name = ""
if "ml_plugin_validation_results" not in st.session_state:
    st.session_state.ml_plugin_validation_results = []
if "ml_plugin_test_instance" not in st.session_state:
    st.session_state.ml_plugin_test_instance = None
if "ml_plugin_error" not in st.session_state:
    st.session_state.ml_plugin_error = None
if "ml_plugin_save_status" not in st.session_state:
    st.session_state.ml_plugin_save_status = ""
if "show_ml_plugin_snippets" not in st.session_state:
    st.session_state.show_ml_plugin_snippets = False
if "ml_developer_mode" not in st.session_state:
    st.session_state.ml_developer_mode = False
if "ml_plugin_category" not in st.session_state:
    st.session_state.ml_plugin_category = "Custom"
if "test_plugin_available" not in st.session_state:
    st.session_state.test_plugin_available = False
if "test_plugin_name" not in st.session_state:
    st.session_state.test_plugin_name = ""
if "test_plugin_instance" not in st.session_state:
    st.session_state.test_plugin_instance = None
if 'trained_combinations' not in st.session_state:
    st.session_state.trained_combinations = set()  # Track dataset+algorithm combinations that are already trained
if 'ppml_dashboard_visualizer' not in st.session_state:
    st.session_state.ppml_dashboard_visualizer = PPMLComparisonVisualization()
ppml_viz = st.session_state.ppml_dashboard_visualizer

# --- Helper Functions ---
@st.cache_data
def load_data(uploaded_file):
    if uploaded_file is not None:
        try:
            file_extension = uploaded_file.name.split('.')[-1].lower()
            if file_extension == 'csv':
                df = pd.read_csv(uploaded_file)
            elif file_extension in ['xls', 'xlsx']:
                df = pd.read_excel(uploaded_file)
            else:
                st.error(f"Unsupported file type: .{file_extension}. Please upload a CSV or XLSX file.")
                return None
            return df
        except Exception as e:
            st.error(f"Error loading file: {e}")
            return None
    return None

def determine_task_type(target_series):
    """Determine if the task is classification or regression based on target variable."""
    if target_series.dtype in ['object', 'category', 'bool']:
        return "classification"
    
    # For numeric data, check if it looks like discrete classes
    unique_values = target_series.nunique()
    total_values = len(target_series)
    
    # If less than 10 unique values or less than 5% unique ratio, likely classification
    if unique_values <= 10 or (unique_values / total_values) < 0.05:
        return "classification"
    else:
        return "regression"

def train_and_evaluate_model(plugin, hyperparameters, X_train, y_train, X_test, y_test, target_column=None, selected_metrics=None):
    """Train a model using a plugin and return its metrics."""
    results = {}
    model_name_for_status = plugin.get_name()
    st.write(f"ℹ️ [Evaluation] Preparing to train {model_name_for_status}...")
    
    try:
        # Create model instance using the plugin
        with st.status(f"[Evaluation] Creating model instance for {model_name_for_status}...", expanded=False) as status_create:
            model = plugin.create_model_instance(hyperparameters)
            status_create.update(label=f"[Evaluation] Model instance for {model_name_for_status} created.", state="complete")

        # Apply any plugin-specific preprocessing
        with st.status(f"[Evaluation] Preprocessing data for {model_name_for_status} via plugin...", expanded=False) as status_plugin_preprocess:
            X_train_processed, y_train_processed = plugin.preprocess_data(X_train, y_train)
            X_test_processed, y_test_processed = plugin.preprocess_data(X_test, y_test)
            status_plugin_preprocess.update(label=f"[Evaluation] Plugin preprocessing for {model_name_for_status} complete.", state="complete")
        
        # STORE FEATURE NAMES BEFORE TRAINING
        feature_names = []
        if hasattr(X_train, 'columns'):
            feature_names = list(X_train.columns)
            print(f"DEBUG TRAIN - Got feature names from X_train.columns: {feature_names}")
        elif hasattr(X_train_processed, 'columns'):
            feature_names = list(X_train_processed.columns)
            print(f"DEBUG TRAIN - Got feature names from X_train_processed.columns: {feature_names}")
        else:
            # Fixed: Determine the number of features first, then create the list
            if hasattr(X_train_processed, 'shape'):
                num_features = X_train_processed.shape[1]
            else:
                num_features = X_train.shape[1]
            
            feature_names = [f"feature_{i}" for i in range(num_features)]
            print(f"DEBUG TRAIN - Created default feature names: {feature_names}")

        # Make sure X_train_processed retains column names if it's a DataFrame
        if hasattr(X_train_processed, 'columns'):
            print(f"DEBUG TRAIN - X_train_processed has columns: {list(X_train_processed.columns)}")
        else:
            print(f"DEBUG TRAIN - X_train_processed is numpy array, shape: {X_train_processed.shape}")
            # Convert back to DataFrame to preserve column names
            if feature_names and len(feature_names) == X_train_processed.shape[1]:
                X_train_processed = pd.DataFrame(X_train_processed, columns=feature_names)
                X_test_processed = pd.DataFrame(X_test_processed, columns=feature_names)
                print(f"DEBUG TRAIN - Converted back to DataFrame with columns: {feature_names}")
        
        # Train the model - make sure X_train_processed is a DataFrame with columns
        print(f"DEBUG TRAIN - Before fit(), X_train_processed type: {type(X_train_processed)}")
        if hasattr(X_train_processed, 'columns'):
            print(f"DEBUG TRAIN - Before fit(), columns: {list(X_train_processed.columns)}")

        # Train the model
        st.write(f"⏳ [Evaluation] Starting model.fit() for {model_name_for_status}. This might take a while...")
        with st.status(f"[Evaluation] Fitting {model_name_for_status}...", expanded=True) as status_fit:
            model.fit(X_train_processed, y_train_processed)
            status_fit.update(label=f"[Evaluation] Fitting {model_name_for_status} complete.", state="complete")
        st.write(f"✅ [Evaluation] model.fit() for {model_name_for_status} finished.")

        with st.status(f"[Evaluation] Predicting with {model_name_for_status}...", expanded=False) as status_predict:
            y_pred = model.predict(X_test_processed)
            status_predict.update(label=f"[Evaluation] Prediction with {model_name_for_status} complete.", state="complete")
        
        # Get prediction probabilities if available (for classification tasks)
        y_proba = None
        if hasattr(model, 'predict_proba'):
            try:
                y_proba = model.predict_proba(X_test_processed)
            except Exception as e_proba:
                st.warning(f"Could not get prediction probabilities: {e_proba}")

        # ADD THIS SECTION TO GET ALGORITHM-SPECIFIC METRICS
        algorithm_specific_metrics = {}
        if hasattr(plugin, 'get_algorithm_specific_metrics'): # Check if the plugin implements this new method
            try:
                # Ensure to call get_algorithm_specific_metrics on the FITTED model instance
                # The 'model' variable here is the fitted plugin instance
                # The arguments y_true, y_pred, y_proba should correspond to what the method expects.
                # Based on your plugin, it currently uses y_test_processed for y_true, and y_pred for y_pred.
                # y_proba is also available from your training loop.
                if hasattr(model, 'get_algorithm_specific_metrics'): # model is the instance returned by plugin.create_model_instance()
                    algorithm_specific_metrics = model.get_algorithm_specific_metrics(y_true=y_test_processed, y_pred=y_pred, y_proba=y_proba)
                else:
                    # Fallback or error if the created model instance doesn't have the method (should not happen if create_model_instance is correct)
                    st.warning(f"⚠️ The created model instance for {plugin.get_name()} does not have 'get_algorithm_specific_metrics'.")
                    algorithm_specific_metrics['status'] = "Method not found on model instance"

            except Exception as e_alg_metrics:
                st.warning(f"⚠️ Could not retrieve algorithm-specific metrics for {plugin.get_name()}: {e_alg_metrics}")
                algorithm_specific_metrics['status'] = f"Error retrieving: {e_alg_metrics}"
        results["algorithm_specific_metrics"] = algorithm_specific_metrics
        # END OF ADDED SECTION

        with st.status(f"[Evaluation] Calculating metrics for {model_name_for_status}...", expanded=False) as status_metrics:
            # Determine task type to calculate appropriate metrics
            task_type = st.session_state.get('task_type', 'classification')
            
            if task_type == 'regression':
                # Import regression metrics
                from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error, explained_variance_score, max_error
                
                # Essential regression metrics
                results["mse"] = mean_squared_error(y_test_processed, y_pred)
                results["rmse"] = np.sqrt(results["mse"])  # More efficient
                results["mae"] = mean_absolute_error(y_test_processed, y_pred)
                results["r2_score"] = r2_score(y_test_processed, y_pred)
                
                # MAPE with error handling
                try:
                    results["mape"] = mean_absolute_percentage_error(y_test_processed, y_pred)
                except Exception:
                    # Handle division by zero in MAPE calculation
                    epsilon = 1e-10
                    results["mape"] = np.mean(np.abs((y_test_processed - y_pred) / np.maximum(np.abs(y_test_processed), epsilon))) * 100
                
                # Additional regression metrics
                results["explained_variance"] = explained_variance_score(y_test_processed, y_pred)
                results["max_error"] = max_error(y_test_processed, y_pred)
                
                # For UI compatibility, use the main regression metric as the primary score
                # This ensures existing UI components work without modification
                results["accuracy"] = results["r2_score"]  # R² is the most interpretable for regression
                results["precision"] = results["r2_score"]  # Keep for UI compatibility
                results["recall"] = results["r2_score"]     # Keep for UI compatibility
                results["f1_score"] = results["r2_score"]   # Keep for UI compatibility
                results["confusion_matrix"] = None          # Not applicable for regression
                
                # Add a flag to indicate this is a regression task
                results["task_type"] = "regression"
                results["primary_metric"] = "r2_score"
                results["primary_metric_name"] = "R² Score"
                
            else:
                # Classification metrics
                from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
                
                results["accuracy"] = accuracy_score(y_test_processed, y_pred)
                results["precision"] = precision_score(y_test_processed, y_pred, average='macro', zero_division=0)
                results["recall"] = recall_score(y_test_processed, y_pred, average='macro', zero_division=0)
                results["f1_score"] = f1_score(y_test_processed, y_pred, average='macro', zero_division=0)
                results["confusion_matrix"] = confusion_matrix(y_test_processed, y_pred)
                
                # Add classification-specific metrics
                results["task_type"] = "classification"
                results["primary_metric"] = "f1_score"
                results["primary_metric_name"] = "F1-Score"
            
            # Calculate custom metrics if selected (works for both task types)
            if selected_metrics and PLUGINS_AVAILABLE:
                custom_metrics = {}
                for metric_name in selected_metrics:
                    try:
                        metric_plugin = metric_manager.get_metric(metric_name)
                        if metric_plugin:
                            metric_value = metric_plugin.calculate(y_test_processed, y_pred, y_proba)
                            custom_metrics[metric_name] = metric_value
                    except Exception as e:
                        st.warning(f"Error calculating {metric_name}: {e}")
                        custom_metrics[metric_name] = f"Error: {str(e)}"
                
                results["custom_metrics"] = custom_metrics
            
            status_metrics.update(label=f"[Evaluation] Metrics calculation for {model_name_for_status} complete.", state="complete")
        
        results["model_name"] = plugin.get_name()
        results["model_description"] = plugin.get_description()
        results["trained_model"] = model
        results["hyperparameters"] = hyperparameters
        results["target_column"] = target_column
        
        # STORE FEATURE NAMES AND TEST DATA
        results["feature_names"] = feature_names
        results["X_test"] = X_test_processed
        results["y_test"] = y_test_processed
        
        # Store unique class labels for confusion matrix display (classification only)
        if task_type == 'classification':
            results["class_labels"] = sorted(list(set(y_train_processed) | set(y_test_processed)))
        else:
            results["class_labels"] = None
        
    except Exception as e:
        st.error(f"Error training/evaluating {plugin.get_name()}: {e}")
        results["error"] = str(e)
    return results

@st.cache_data
def load_sample_dataset(dataset_name):
    """Load sklearn sample datasets and convert to DataFrame"""
    from sklearn import datasets
    import pandas as pd
    import numpy as np
    
    try:
        # === EXISTING DATASETS ===
        if dataset_name == "iris":
            data = datasets.load_iris()
            df = pd.DataFrame(data.data, columns=data.feature_names)
            df['target'] = data.target
            df['target'] = df['target'].map({0: 'setosa', 1: 'versicolor', 2: 'virginica'})
            
        elif dataset_name == "wine":
            data = datasets.load_wine()
            df = pd.DataFrame(data.data, columns=data.feature_names)
            df['target'] = data.target
            df['target'] = df['target'].map({0: 'class_0', 1: 'class_1', 2: 'class_2'})
            
        elif dataset_name == "breast_cancer":
            data = datasets.load_breast_cancer()
            df = pd.DataFrame(data.data, columns=data.feature_names)
            df['target'] = data.target
            df['target'] = df['target'].map({0: 'malignant', 1: 'benign'})
            
        elif dataset_name == "boston":
            try:
                data = datasets.load_boston()
                df = pd.DataFrame(data.data, columns=data.feature_names)
                df['target'] = data.target
            except ImportError:
                data = datasets.fetch_california_housing()
                df = pd.DataFrame(data.data, columns=data.feature_names)
                df['target'] = data.target
                st.info("ℹ️ Loaded California Housing dataset (Boston dataset deprecated)")
            
        elif dataset_name == "diabetes":
            data = datasets.load_diabetes()
            df = pd.DataFrame(data.data, columns=data.feature_names)
            df['target'] = data.target
            
        elif dataset_name == "digits":
            data = datasets.load_digits()
            df = pd.DataFrame(data.data, columns=[f'pixel_{i}' for i in range(64)])
            df['target'] = data.target
            
        elif dataset_name == "california_housing":
            data = datasets.fetch_california_housing()
            df = pd.DataFrame(data.data, columns=data.feature_names)
            df['target'] = data.target
        
        # === NEW BINARY CLASSIFICATION DATASETS ===
        elif dataset_name == "credit_approval":
            # Synthetic credit approval dataset
            np.random.seed(42)
            n_samples = 1000
            
            # Create correlated features
            age = np.random.normal(40, 12, n_samples)
            income = np.random.exponential(50000, n_samples) + age * 1000
            credit_score = np.random.normal(650, 100, n_samples) + income * 0.002
            debt_ratio = np.random.beta(2, 5, n_samples) * 0.8
            employment_years = np.random.exponential(5, n_samples)
            
            # Create approval decision based on logical rules + noise
            approval_score = (
                (credit_score - 600) * 0.01 +
                (income - 30000) * 0.00002 +
                (employment_years - 1) * 0.1 +
                (0.4 - debt_ratio) * 2 +
                np.random.normal(0, 0.3, n_samples)
            )
            
            df = pd.DataFrame({
                'age': age.clip(18, 80),
                'annual_income': income.clip(15000, 200000),
                'credit_score': credit_score.clip(300, 850),
                'debt_to_income_ratio': debt_ratio,
                'employment_years': employment_years.clip(0, 40),
                'num_credit_cards': np.random.poisson(3, n_samples),
                'num_bank_accounts': np.random.poisson(2, n_samples),
                'target': (approval_score > 0.5).astype(int)
            })
            df['target'] = df['target'].map({0: 'rejected', 1: 'approved'})
            
        elif dataset_name == "titanic":
            # Simplified Titanic dataset
            np.random.seed(42)
            n_samples = 891
            
            # Passenger class distribution
            pclass = np.random.choice([1, 2, 3], n_samples, p=[0.24, 0.21, 0.55])
            
            # Age distribution by class
            age = np.where(pclass == 1, np.random.normal(38, 15, n_samples),
                          np.where(pclass == 2, np.random.normal(30, 14, n_samples),
                                  np.random.normal(25, 12, n_samples)))
            
            # Gender distribution
            is_male = np.random.choice([0, 1], n_samples, p=[0.35, 0.65])
            
            # Fare by class
            fare = np.where(pclass == 1, np.random.normal(84, 78, n_samples),
                           np.where(pclass == 2, np.random.normal(20, 13, n_samples),
                                   np.random.normal(13, 11, n_samples)))
            
            # Survival based on historical patterns
            survival_prob = (
                0.4 +  # Base survival rate
                (pclass == 1) * 0.25 +  # First class bonus
                (pclass == 2) * 0.1 +   # Second class bonus
                (1 - is_male) * 0.35 +  # Female bonus
                (age < 16) * 0.2        # Child bonus
            )
            survived = np.random.binomial(1, survival_prob.clip(0, 1), n_samples)
            
            df = pd.DataFrame({
                'passenger_class': pclass,
                'age': age.clip(0, 80),
                'is_male': is_male,
                'fare': fare.clip(0, 500),
                'family_size': np.random.poisson(1, n_samples),
                'target': survived
            })
            df['target'] = df['target'].map({0: 'died', 1: 'survived'})
            
        elif dataset_name == "heart_disease":
            # Heart disease prediction dataset
            np.random.seed(42)
            n_samples = 1000
            
            age = np.random.normal(54, 9, n_samples).clip(29, 77)
            is_male = np.random.choice([0, 1], n_samples, p=[0.32, 0.68])
            chest_pain_type = np.random.choice([0, 1, 2, 3], n_samples)
            resting_bp = np.random.normal(131, 17, n_samples).clip(94, 200)
            cholesterol = np.random.normal(246, 51, n_samples).clip(126, 564)
            max_heart_rate = np.random.normal(149, 22, n_samples).clip(71, 202)
            
            # Heart disease probability
            risk_score = (
                (age - 50) * 0.02 +
                is_male * 0.3 +
                (chest_pain_type == 0) * 0.4 +  # Typical angina
                (resting_bp - 120) * 0.005 +
                (cholesterol - 200) * 0.002 +
                (150 - max_heart_rate) * 0.01 +
                np.random.normal(0, 0.3, n_samples)
            )
            
            df = pd.DataFrame({
                'age': age,
                'is_male': is_male,
                'chest_pain_type': chest_pain_type,
                'resting_blood_pressure': resting_bp,
                'cholesterol': cholesterol,
                'max_heart_rate': max_heart_rate,
                'exercise_induced_angina': np.random.choice([0, 1], n_samples),
                'target': (risk_score > 0.4).astype(int)
            })
            df['target'] = df['target'].map({0: 'no_disease', 1: 'disease'})
        
        # === NEW MULTI-CLASS DATASETS ===
        elif dataset_name == "flower_species":
            # Extended flower classification with more species
            np.random.seed(42)
            n_per_class = 100
            species = ['rose', 'tulip', 'daisy', 'sunflower', 'lily']
            
            data_list = []
            for i, species_name in enumerate(species):
                # Each species has different characteristics
                petal_length = np.random.normal(2 + i * 1.5, 0.8, n_per_class)
                petal_width = np.random.normal(0.5 + i * 0.4, 0.3, n_per_class)
                sepal_length = np.random.normal(5 + i * 0.8, 1.0, n_per_class)
                sepal_width = np.random.normal(3 + i * 0.3, 0.5, n_per_class)
                stem_height = np.random.normal(20 + i * 10, 5, n_per_class)
                
                for j in range(n_per_class):
                    data_list.append({
                        'petal_length': petal_length[j],
                        'petal_width': petal_width[j],
                        'sepal_length': sepal_length[j],
                        'sepal_width': sepal_width[j],
                        'stem_height': stem_height[j],
                        'target': species_name
                    })
            
            df = pd.DataFrame(data_list)
            
        elif dataset_name == "music_genre":
            # Music genre classification
            np.random.seed(42)
            n_per_genre = 150
            genres = ['rock', 'jazz', 'classical', 'electronic', 'country']
            
            data_list = []
            for i, genre in enumerate(genres):
                # Musical features vary by genre
                tempo = np.random.normal(100 + i * 20, 15, n_per_genre)
                energy = np.random.beta(2 + i, 8 - i, n_per_genre)
                danceability = np.random.beta(1 + i * 0.5, 4, n_per_genre)
                valence = np.random.beta(2, 3, n_per_genre)
                loudness = np.random.normal(-10 + i * 2, 3, n_per_genre)
                
                for j in range(n_per_genre):
                    data_list.append({
                        'tempo': tempo[j],
                        'energy': energy[j],
                        'danceability': danceability[j],
                        'valence': valence[j],
                        'loudness': loudness[j],
                        'duration_ms': np.random.normal(240000, 60000),
                        'target': genre
                    })
            
            df = pd.DataFrame(data_list)
        
        # === NEW REGRESSION DATASETS ===
        elif dataset_name == "auto_mpg":
            # Car fuel efficiency prediction
            np.random.seed(42)
            n_samples = 800
            
            cylinders = np.random.choice([4, 6, 8], n_samples, p=[0.5, 0.3, 0.2])
            displacement = np.where(cylinders == 4, np.random.normal(120, 30, n_samples),
                                  np.where(cylinders == 6, np.random.normal(200, 40, n_samples),
                                          np.random.normal(300, 60, n_samples)))
            horsepower = displacement * 0.8 + np.random.normal(0, 20, n_samples)
            weight = displacement * 8 + np.random.normal(0, 200, n_samples)
            year = np.random.choice(range(70, 83), n_samples)
            
            # MPG calculation based on realistic factors
            mpg = (
                40 -
                cylinders * 2 -
                displacement * 0.05 -
                weight * 0.003 +
                (year - 70) * 0.5 +
                np.random.normal(0, 2, n_samples)
            ).clip(8, 50)
            
            df = pd.DataFrame({
                'cylinders': cylinders,
                'displacement': displacement,
                'horsepower': horsepower,
                'weight': weight,
                'model_year': year,
                'origin': np.random.choice([1, 2, 3], n_samples, p=[0.6, 0.2, 0.2]),
                'target': mpg
            })
            
        elif dataset_name == "stock_prices":
            # Stock price prediction
            np.random.seed(42)
            n_samples = 1000
            
            # Market indicators
            market_cap = np.random.exponential(1000, n_samples)  # Million $
            pe_ratio = np.random.gamma(2, 8, n_samples)
            debt_ratio = np.random.beta(2, 5, n_samples)
            revenue_growth = np.random.normal(0.1, 0.3, n_samples)
            volume = np.random.exponential(1000000, n_samples)
            
            # Stock price based on fundamentals
            price = (
                market_cap * 0.01 +
                pe_ratio * 2 +
                (1 - debt_ratio) * 20 +
                revenue_growth * 50 +
                np.log(volume) * 5 +
                np.random.normal(0, 10, n_samples)
            ).clip(1, 500)
            
            df = pd.DataFrame({
                'market_cap_millions': market_cap,
                'pe_ratio': pe_ratio,
                'debt_to_equity': debt_ratio,
                'revenue_growth_rate': revenue_growth,
                'trading_volume': volume,
                'sector': np.random.choice(['tech', 'finance', 'healthcare', 'energy'], n_samples),
                'target': price
            })
        
        # === LARGE DATASETS ===
        elif dataset_name == "adult_income":
            # Adult income prediction (Census data style)
            np.random.seed(42)
            n_samples = 48000
            
            age = np.random.gamma(2, 20, n_samples).clip(17, 90)
            education_num = np.random.choice(range(1, 17), n_samples, 
                                           p=np.array([0.02, 0.05, 0.08, 0.1, 0.12, 0.15, 0.15, 0.12, 0.08, 0.05, 0.03, 0.02, 0.01, 0.01, 0.005, 0.005]))
            hours_per_week = np.random.normal(40, 12, n_samples).clip(1, 99)
            
            # Income probability based on features
            income_prob = (
                (age - 25) * 0.01 +
                education_num * 0.05 +
                (hours_per_week - 20) * 0.02 +
                np.random.normal(0, 0.3, n_samples)
            )
            
            df = pd.DataFrame({
                'age': age,
                'education_num': education_num,
                'hours_per_week': hours_per_week,
                'workclass': np.random.choice(['private', 'gov', 'self_employed'], n_samples, p=[0.7, 0.2, 0.1]),
                'marital_status': np.random.choice(['married', 'single', 'divorced'], n_samples, p=[0.5, 0.3, 0.2]),
                'occupation': np.random.choice(['tech', 'sales', 'service', 'admin', 'manual'], n_samples),
                'capital_gain': np.random.exponential(100, n_samples),
                'capital_loss': np.random.exponential(50, n_samples),
                'target': (income_prob > 0.5).astype(int)
            })
            df['target'] = df['target'].map({0: '<=50K', 1: '>50K'})
        
        # === SPECIALIZED DATASETS ===
        elif dataset_name == "marketing_campaign":
            # Marketing campaign response prediction
            np.random.seed(42)
            n_samples = 2000
            
            age = np.random.normal(45, 15, n_samples).clip(18, 80)
            income = np.random.lognormal(10.5, 0.5, n_samples)
            recency = np.random.exponential(50, n_samples)
            frequency = np.random.poisson(5, n_samples)
            monetary = np.random.exponential(500, n_samples)
            
            # Response probability
            response_score = (
                (income - 30000) * 0.00001 +
                (100 - recency) * 0.01 +
                frequency * 0.1 +
                monetary * 0.001 +
                np.random.normal(0, 0.3, n_samples)
            )
            
            df = pd.DataFrame({
                'age': age,
                'annual_income': income,
                'recency_days': recency,
                'frequency_purchases': frequency,
                'monetary_value': monetary,
                'channel_preference': np.random.choice(['email', 'phone', 'mail', 'sms'], n_samples),
                'previous_campaigns': np.random.poisson(2, n_samples),
                'target': (response_score > 0.5).astype(int)
            })
            df['target'] = df['target'].map({0: 'no_response', 1: 'response'})
        
        else:
            st.error(f"Unknown dataset: {dataset_name}")
            return None
            
        return df
        
    except Exception as e:
        st.error(f"Error loading dataset {dataset_name}: {e}")
        return None

def get_dataset_info(dataset_name):
    """Get information about the dataset"""
    dataset_info = {
        # === EXISTING DATASETS ===
        "iris": {
            "description": "Famous flower classification dataset with 3 species",
            "features": 4, "samples": 150, "task_type": "classification", "target_column": "target"
        },
        "wine": {
            "description": "Wine classification based on chemical analysis", 
            "features": 13, "samples": 178, "task_type": "classification", "target_column": "target"
        },
        "breast_cancer": {
            "description": "Breast cancer diagnosis prediction",
            "features": 30, "samples": 569, "task_type": "classification", "target_column": "target"
        },
        "boston": {
            "description": "Boston housing price prediction",
            "features": 13, "samples": 506, "task_type": "regression", "target_column": "target"
        },
        "diabetes": {
            "description": "Diabetes progression prediction",
            "features": 10, "samples": 442, "task_type": "regression", "target_column": "target"
        },
        "digits": {
            "description": "Handwritten digit recognition (0-9)",
            "features": 64, "samples": 1797, "task_type": "classification", "target_column": "target"
        },
        "california_housing": {
            "description": "California housing price prediction",
            "features": 8, "samples": 20640, "task_type": "regression", "target_column": "target"
        },
        
        # === NEW BINARY CLASSIFICATION ===
        "credit_approval": {
            "description": "Predict credit approval based on financial profile",
            "features": 7, "samples": 1000, "task_type": "classification", "target_column": "target"
        },
        "titanic": {
            "description": "Predict passenger survival on the Titanic",
            "features": 5, "samples": 891, "task_type": "classification", "target_column": "target"
        },
        "heart_disease": {
            "description": "Predict heart disease risk from medical indicators",
            "features": 7, "samples": 1000, "task_type": "classification", "target_column": "target"
        },
        
        # === NEW MULTI-CLASS ===
        "flower_species": {
            "description": "Classify 5 different flower species",
            "features": 5, "samples": 500, "task_type": "classification", "target_column": "target"
        },
        "music_genre": {
            "description": "Classify music tracks into 5 genres",
            "features": 6, "samples": 750, "task_type": "classification", "target_column": "target"
        },
        
        # === NEW REGRESSION ===
        "auto_mpg": {
            "description": "Predict car fuel efficiency (MPG)",
            "features": 6, "samples": 800, "task_type": "regression", "target_column": "target"
        },
        "stock_prices": {
            "description": "Predict stock prices from financial indicators",
            "features": 6, "samples": 1000, "task_type": "regression", "target_column": "target"
        },
        
        # === LARGE DATASETS ===
        "adult_income": {
            "description": "Predict income level from census data",
            "features": 8, "samples": 48000, "task_type": "classification", "target_column": "target"
        },
        
        # === SPECIALIZED ===
        "marketing_campaign": {
            "description": "Predict customer response to marketing campaigns",
            "features": 7, "samples": 2000, "task_type": "classification", "target_column": "target"
        }
    }
    
    return dataset_info.get(dataset_name, {})

# --- Callback for Deleting a Result ---
def delete_experiment_result(result_index: int):
    """Removes a specific result from the session state by its index."""
    if 0 <= result_index < len(st.session_state.experiment_results):
        del st.session_state.experiment_results[result_index]
    else:
        st.warning("Attempted to delete an invalid result index.")

# Helper function to display metrics in a consistent table format
def _display_metrics_table(metrics_dict, title):
    """Helper function to display algorithm metrics in a consistent table format"""
    
    if not metrics_dict:
        st.info(f"No {title.lower()} available")
        return
    
    st.markdown(f"**{title}:**")
    
    # Convert metrics to display format
    display_data = []
    
    for key, value in metrics_dict.items():
        # Format the metric name for display
        display_name = key.replace('_', ' ').title()
        
        # Handle different value types
        if isinstance(value, bool):
            formatted_value = "✅ Yes" if value else "❌ No"
            value_type = "Boolean"
        elif isinstance(value, int):
            formatted_value = f"{value:,}"
            value_type = "Integer"
        elif isinstance(value, float):
            if abs(value) >= 1000:
                formatted_value = f"{value:,.2f}"
            elif abs(value) >= 1:
                formatted_value = f"{value:.4f}"
            else:
                formatted_value = f"{value:.6f}"
            value_type = "Float"
        elif isinstance(value, str):
            formatted_value = value
            value_type = "String"
        else:
            formatted_value = str(value)
            value_type = "Other"
        
        # Add interpretation for common metrics
        interpretation = ""
        if "accuracy" in key.lower() or "score" in key.lower():
            if isinstance(value, (int, float)):
                if value >= 0.9:
                    interpretation = "🟢 Excellent"
                elif value >= 0.8:
                    interpretation = "🟡 Good"
                elif value >= 0.7:
                    interpretation = "🟠 Fair"
                else:
                    interpretation = "🔴 Poor"
        elif "error" in key.lower() or "loss" in key.lower():
            if isinstance(value, (int, float)):
                if value <= 0.1:
                    interpretation = "🟢 Low"
                elif value <= 0.3:
                    interpretation = "🟡 Moderate"
                else:
                    interpretation = "🔴 High"
        elif "ratio" in key.lower() or "percentage" in key.lower():
            if isinstance(value, (int, float)):
                if value >= 0.8:
                    interpretation = "🟢 High"
                elif value >= 0.5:
                    interpretation = "🟡 Medium"
                else:
                    interpretation = "🔴 Low"
        
        display_data.append({
            "📊 Metric": display_name,
            "📈 Value": formatted_value,
            "🏷️ Type": value_type,
            "💡 Assessment": interpretation if interpretation else "—"
        })
    
    # Create and display the table
    if display_data:
        metrics_df = pd.DataFrame(display_data)
        
        # Style the table
        styled_metrics = metrics_df.style.set_properties(**{
            'padding': '8px',
            'font-size': '13px',
            'text-align': 'center',
            'border': '1px solid var(--text-color-secondary)'
        }).set_table_styles([
            {'selector': 'th', 'props': [
                ('background-color', '#17a2b8'),
                ('color', 'white'),
                ('font-weight', 'bold'),
                ('text-align', 'center'),
                ('padding', '10px'),
                ('border', '1px solid #17a2b8')
            ]},
            {'selector': 'td:first-child', 'props': [
                ('text-align', 'left'),
                ('font-weight', 'bold')
            ]},
            {'selector': 'td', 'props': [
                ('text-align', 'center'),
                ('vertical-align', 'middle'),
                ('border', '1px solid var(--text-color-secondary)')
            ]},
            {'selector': '', 'props': [
                ('border-collapse', 'collapse'),
                ('margin', '10px 0'),
                ('border-radius', '6px'),
                ('overflow', 'hidden')
            ]}
        ])
        
        st.dataframe(styled_metrics, use_container_width=True, hide_index=True)
        
        # Add summary
        st.caption(f"📊 {len(display_data)} {title.lower()} displayed")


# --- Initialize Session State ---
if 'df_uploaded' not in st.session_state:
    st.session_state.df_uploaded = None
if 'target_column' not in st.session_state:
    st.session_state.target_column = None
if 'task_type' not in st.session_state:
    st.session_state.task_type = "classification"
if 'selected_plugins_config' not in st.session_state:
    st.session_state.selected_plugins_config = {}
if 'experiment_results' not in st.session_state:
    st.session_state.experiment_results = []
if 'execute_training_flag' not in st.session_state:
    st.session_state.execute_training_flag = False
if 'selected_metrics' not in st.session_state:
    st.session_state.selected_metrics = ["ROC AUC Score", "Matthews Correlation Coefficient"]  # Default metrics
if 'sample_dataset_info' not in st.session_state:
    st.session_state.sample_dataset_info = {}
# ADD THE NEW DATA MANAGEMENT STRUCTURE HERE:
# --- Multi-Dataset Support for PPML Benchmarking ---
if 'datasets_collection' not in st.session_state:
    st.session_state.datasets_collection = {
        'original': None,
        'anonymized_datasets': {}  # {'k-anon': [df1, df2, df3], 'l-diversity': [df1, df2], etc.}
    }
# Add counters for multiple datasets per method
if 'dataset_counters' not in st.session_state:
    st.session_state.dataset_counters = {}  # {'k-anon': 3, 'l-diversity': 2, etc.}
if 'dataset_metadata' not in st.session_state:
    st.session_state.dataset_metadata = {}
if 'preprocessing_configs' not in st.session_state:
    st.session_state.preprocessing_configs = {}

# Initialize plugin and metric managers
if PLUGINS_AVAILABLE:
    plugin_manager = get_plugin_manager()
    metric_manager = get_metric_manager()
    viz_manager = get_visualization_manager()

# --- Main Application UI ---
st.title("🚀 ML Experimentation Platform")
st.markdown("Upload your data, select algorithms, configure parameters, and compare their performance.")

# --- Sidebar for Inputs ---
with st.sidebar:
    st.header("📊 Unified Data Input System")
    
    # Single tab-based interface that handles all data input scenarios
    tab1, tab2, tab3, tab4 = st.tabs(["📂 Data Source", "🔒 Anonymized Data", "⚙️ Preprocessing", "📈 Dashboard"])
    
    with tab1:
        st.subheader("Primary Data Input")
        
        # Create two input methods: File Upload and Sample Datasets
        input_method = st.radio(
            "Choose your data source:",
            ["📁 Upload File", "🎲 Sample Dataset"],
            help="Select how you want to provide data to the system"
        )
        
        # Initialize a key for the file uploader in session state if it doesn't exist
        if 'file_uploader_key' not in st.session_state:
            st.session_state.file_uploader_key = 'file_uploader_0'

        # File upload section
        uploaded_file = st.file_uploader(
            "Upload your CSV or XLSX file",
            type=["csv", "xlsx"],
            help="Upload your dataset for analysis",
            key=st.session_state.file_uploader_key # Use the key from session state
        )

        # Add a small refresh button for the file upload status
        if st.button("🔄 Clear Uploaded File", help="Clear the current file from the uploader and reset related application data.", key="clear_uploaded_file_state_button"):
            # Reset application data derived from the file
            st.session_state.df_uploaded = None
            st.session_state.datasets_collection['original'] = None
            if 'original' in st.session_state.dataset_metadata:
                del st.session_state.dataset_metadata['original']
            st.session_state.target_column = None
            st.session_state.task_type = "classification" # Reset to default
            st.session_state.sample_dataset_info = {}

            # Change the key of the file_uploader to force it to reset visually
            # This makes Streamlit treat it as a new widget
            current_key_suffix = int(st.session_state.file_uploader_key.split('_')[-1])
            st.session_state.file_uploader_key = f'file_uploader_{current_key_suffix + 1}'
            
            st.rerun()

        # Handle uploaded file with preview and load buttons
        if uploaded_file:
            st.info(f"📁 File '{uploaded_file.name}' ready for processing")
            
            # Create two buttons: Preview and Load
            col1, col2 = st.columns([1, 1])
            
            # Initialize preview trigger in session state
            if 'show_file_preview' not in st.session_state:
                st.session_state.show_file_preview = False
            
            with col1:
                if st.button("🔍 Preview Data", key="preview_uploaded_file", use_container_width=True, help="Quick preview without loading"):
                    # Set the preview flag to True
                    st.session_state.show_file_preview = True
            
            with col2:
                if st.button("📥 Load Dataset", key="load_uploaded_file", type="primary", use_container_width=True, help="Load data for analysis"):
                    with st.spinner(f"Loading {uploaded_file.name}..."):
                        loaded_df = load_data(uploaded_file)
                        if loaded_df is not None:
                            # Store as both original and main dataset
                            st.session_state.df_uploaded = loaded_df
                            st.session_state.datasets_collection['original'] = loaded_df
                            st.session_state.dataset_metadata['original'] = {
                                'name': uploaded_file.name,
                                'type': 'original',
                                'upload_time': pd.Timestamp.now()
                            }
                            
                            st.success(f"✅ File '{uploaded_file.name}' loaded successfully!")
                            
                            # Show quick summary
                            st.info(f"📊 Dataset loaded: {loaded_df.shape[0]:,} rows × {loaded_df.shape[1]:,} columns")
                            
                            # Clear preview when data is loaded
                            st.session_state.show_file_preview = False
                            
                            # Auto-refresh page to show new data in the interface
                            st.rerun()
                        else:
                            st.error(f"❌ Failed to load {uploaded_file.name}")

        # MOVED OUTSIDE COLUMNS: Full-width preview section
        if uploaded_file and st.session_state.show_file_preview:
            st.markdown("---")  # Separator
            
            with st.spinner(f"Loading preview of {uploaded_file.name}..."):
                preview_df = load_data(uploaded_file)
                if preview_df is not None:
                    st.success(f"✅ Preview loaded successfully!")
                    
                    # Show preview in full-width expander
                    with st.expander("📋 Data Preview", expanded=True):
                        st.dataframe(preview_df.head(10), use_container_width=True)
                        st.caption(f"Showing first 10 rows of {len(preview_df):,} total rows × {len(preview_df.columns):,} columns")
                        
                        # Show column info in full width
                        st.markdown("**📊 Column Information:**")
                        col_info = []
                        for col in preview_df.columns:
                            dtype_str = str(preview_df[col].dtype)
                            null_count = preview_df[col].isnull().sum()
                            unique_count = preview_df[col].nunique()
                            col_info.append({
                                "Column": col,
                                "Data Type": dtype_str,
                                "Missing": null_count,
                                "Unique Values": unique_count
                            })
                        
                        col_info_df = pd.DataFrame(col_info)
                        st.dataframe(col_info_df, use_container_width=True, hide_index=True)
                        
                        # Add a close preview button
                        if st.button("❌ Close Preview", key="close_preview_btn"):
                            st.session_state.show_file_preview = False
                            st.rerun()
                else:
                    st.error(f"❌ Failed to preview {uploaded_file.name}")
                    st.session_state.show_file_preview = False   
                                             
        elif input_method == "🎲 Sample Dataset":
            # Sample dataset section with enhanced filtering and more datasets
            st.markdown("**Filter by Category:**")
            dataset_categories = {
                "All Datasets": "all",
                "🌱 Classic ML (Small & Real)": "classic_small_real",
                "🎯 Binary Classification (Real & Synthetic)": "binary",
                "🌈 Multi-class Classification (Real & Synthetic)": "multiclass",
                "📈 Regression (Real & Synthetic)": "regression",
                "🏭 Synthetic Datasets": "synthetic",
                "🌍 Popular Real-World (Larger/Complex)": "popular_real_world",
                "📊 Large Scale (10K+ samples)": "large"
            }

            selected_category = st.selectbox(
                "Choose dataset category:",
                options=list(dataset_categories.keys()),
                key="unified_dataset_category_filter",
                help="Filter datasets by their primary use case or characteristics"
            )
            
            # Define sample_dataset_options based on selected category
            sample_dataset_options = {"None": None} # Default
            selected_cat_key = dataset_categories.get(selected_category, "all")

            if selected_cat_key == "all":
                sample_dataset_options = {
                    "None": None,
                    # Classic Small Real
                    "🌸 Iris (Classic Multi-class)": "iris",
                    "🍷 Wine (Classic Multi-class)": "wine",
                    "🎗️ Breast Cancer (Classic Binary)": "breast_cancer",
                    "🏠 Boston Housing (Classic Regression)": "boston",
                    "💉 Diabetes (Classic Regression)": "diabetes",
                    # Synthetic
                    "💳 Credit Approval (Synthetic Binary)": "credit_approval",
                    "🎮 Titanic Survival (Synthetic Binary)": "titanic",
                    "💔 Heart Disease (Synthetic Binary)": "heart_disease",
                    "🌺 Flower Species (Synthetic Multi-class)": "flower_species",
                    "🎵 Music Genre (Synthetic Multi-class)": "music_genre",
                    "🚗 Auto MPG (Synthetic Regression)": "auto_mpg",
                    "💰 Stock Prices (Synthetic Regression)": "stock_prices",
                    "📊 Adult Income (Synthetic Large Binary)": "adult_income",
                    "🎯 Marketing Campaign (Synthetic Binary)": "marketing_campaign",
                    # Popular Real-World (Larger/Complex)
                    "🔢 Digits (Handwritten Digit Recognition)": "digits",
                    "🌴 California Housing (Regression)": "california_housing",
                    "🌲 Covertype (Large Multi-class Forest)": "covertype", # New
                    "🏘️ Ames Housing (Advanced Regression)": "ames_housing", # New
                    "🏦 Bank Marketing (Campaign Outcome Binary)": "bank_marketing", # New
                    "👚 Fashion MNIST Sample (Image-like Multi-class)": "fashion_mnist_sample" # New
                }
            elif selected_cat_key == "classic_small_real":
                sample_dataset_options = {
                    "None": None,
                    "🌸 Iris (Classic Multi-class)": "iris",
                    "🍷 Wine (Classic Multi-class)": "wine",
                    "🎗️ Breast Cancer (Classic Binary)": "breast_cancer",
                    "🏠 Boston Housing (Classic Regression)": "boston",
                    "💉 Diabetes (Classic Regression)": "diabetes",
                }
            elif selected_cat_key == "binary":
                sample_dataset_options = {
                    "None": None,
                    "🎗️ Breast Cancer (Classic Binary)": "breast_cancer",
                    "💳 Credit Approval (Synthetic Binary)": "credit_approval",
                    "🎮 Titanic Survival (Synthetic Binary)": "titanic",
                    "💔 Heart Disease (Synthetic Binary)": "heart_disease",
                    "🎯 Marketing Campaign (Synthetic Binary)": "marketing_campaign",
                    "📊 Adult Income (Synthetic Large Binary)": "adult_income",
                    "🏦 Bank Marketing (Campaign Outcome Binary)": "bank_marketing", # New
                }
            elif selected_cat_key == "multiclass":
                sample_dataset_options = {
                    "None": None,
                    "🌸 Iris (Classic Multi-class)": "iris",
                    "🍷 Wine (Classic Multi-class)": "wine",
                    "🌺 Flower Species (Synthetic Multi-class)": "flower_species",
                    "🎵 Music Genre (Synthetic Multi-class)": "music_genre",
                    "🔢 Digits (Handwritten Digit Recognition)": "digits",
                    "👚 Fashion MNIST Sample (Image-like Multi-class)": "fashion_mnist_sample", # New
                    "🌲 Covertype (Large Multi-class Forest)": "covertype", # New
                }
            elif selected_cat_key == "regression":
                sample_dataset_options = {
                    "None": None,
                    "🏠 Boston Housing (Classic Regression)": "boston",
                    "💉 Diabetes (Classic Regression)": "diabetes",
                    "🚗 Auto MPG (Synthetic Regression)": "auto_mpg",
                    "💰 Stock Prices (Synthetic Regression)": "stock_prices",
                    "🌴 California Housing (Regression)": "california_housing",
                    "🏘️ Ames Housing (Advanced Regression)": "ames_housing", # New
                }
            elif selected_cat_key == "synthetic":
                sample_dataset_options = {
                    "None": None,
                    "💳 Credit Approval (Synthetic Binary)": "credit_approval",
                    "🎮 Titanic Survival (Synthetic Binary)": "titanic",
                    "💔 Heart Disease (Synthetic Binary)": "heart_disease",
                    "🌺 Flower Species (Synthetic Multi-class)": "flower_species",
                    "🎵 Music Genre (Synthetic Multi-class)": "music_genre",
                    "🚗 Auto MPG (Synthetic Regression)": "auto_mpg",
                    "💰 Stock Prices (Synthetic Regression)": "stock_prices",
                    "📊 Adult Income (Synthetic Large Binary)": "adult_income",
                    "🎯 Marketing Campaign (Synthetic Binary)": "marketing_campaign",
                }
            elif selected_cat_key == "popular_real_world":
                 sample_dataset_options = {
                    "None": None,
                    "🔢 Digits (Handwritten Digit Recognition)": "digits",
                    "🌴 California Housing (Regression)": "california_housing",
                    "🌲 Covertype (Large Multi-class Forest)": "covertype", # New
                    "🏘️ Ames Housing (Advanced Regression)": "ames_housing", # New
                    "🏦 Bank Marketing (Campaign Outcome Binary)": "bank_marketing", # New
                    "👚 Fashion MNIST Sample (Image-like Multi-class)": "fashion_mnist_sample", # New
                }
            elif selected_cat_key == "large":
                sample_dataset_options = {
                    "None": None,
                    "🌴 California Housing (Regression)": "california_housing", # ~20k
                    "📊 Adult Income (Synthetic Large Binary)": "adult_income", # ~48k
                    "🌲 Covertype (Large Multi-class Forest)": "covertype", # ~580k
                    "👚 Fashion MNIST Sample (Image-like Multi-class)": "fashion_mnist_sample", # Assuming ~10k
                    "🏦 Bank Marketing (Campaign Outcome Binary)": "bank_marketing", # UCI version is ~45k
                }
            
            # Dataset selection dropdown
            selected_dataset = st.selectbox(
                "Choose a sample dataset:",
                options=list(sample_dataset_options.keys()),
                key="unified_sample_dataset_select",
                help="Select a built-in dataset to test the platform. Descriptions provide context."
            )

            if selected_dataset != "None":
                dataset_key = sample_dataset_options[selected_dataset]
                
                # Show dataset info
                info = get_dataset_info(dataset_key)
                if info:
                    with st.expander(f"📋 About {selected_dataset}", expanded=False):
                        col1, col2 = st.columns(2)
                        with col1:
                            st.write(f"**📏 Samples:** {info.get('samples', 'N/A'):,}")
                            st.write(f"**📊 Features:** {info.get('features', 'N/A')}")
                        with col2:
                            st.write(f"**🎯 Task:** {info.get('task_type', 'N/A').title()}")
                            st.write(f"**🏷️ Target:** {info.get('target_column', 'N/A')}")
                        st.write(f"**📝 Description:** {info.get('description', 'No description available')}")
                
                # Load button
                col1, col2 = st.columns([2, 1])
                with col1:
                    if st.button("📥 Load Sample Dataset", type="secondary", use_container_width=True):
                        with st.spinner(f"Loading {selected_dataset}..."):
                            sample_df = load_sample_dataset(dataset_key)
                            
                            if sample_df is not None:
                                # Store as both original and main dataset
                                st.session_state.df_uploaded = sample_df
                                st.session_state.datasets_collection['original'] = sample_df
                                st.session_state.sample_dataset_info = info
                                
                                # Auto-set target column and task type
                                if info.get('target_column') and info['target_column'] in sample_df.columns:
                                    st.session_state.target_column = info['target_column']
                                    st.session_state.task_type = info.get('task_type', 'classification')
                                
                                st.success(f"✅ {selected_dataset} loaded successfully!")
                                st.rerun()
                            else:
                                st.error(f"❌ Failed to load {selected_dataset}")
                
                with col2:
                    if st.button("🔍 Preview", use_container_width=True, help="Quick preview without loading"):
                        with st.spinner("Loading preview..."):
                            preview_df = load_sample_dataset(dataset_key)
                            if preview_df is not None:
                                st.subheader(f"👀 Preview: {selected_dataset}")
                                st.dataframe(preview_df.head(10), use_container_width=True)
                                st.caption(f"Showing first 10 rows of {len(preview_df):,} total rows")
        
        st.markdown("---")

    with tab2:
        st.subheader("Privacy-Preserving Data")
        
        if st.session_state.datasets_collection['original'] is not None:
            st.success("✅ Original dataset loaded - Ready for PPML analysis")
            
            # Dynamic anonymized dataset upload with multiple dataset support
            # Enhanced anonymization method selection with search functionality
            st.markdown("**Select Anonymization Method:**")
            
            # Define all available methods with categories
            anonymization_methods = {
                # Classical Privacy Models
                "Classical Privacy": [
                    "k-anonymity", 
                    "l-diversity", 
                    "t-closeness",
                    "p-sensitive"
                ],
                
                # Differential Privacy Variants
                "Differential Privacy": [
                    "differential-privacy",
                    "local-differential-privacy", 
                    "federated-differential-privacy",
                    "adaptive-differential-privacy"
                ],
                
                # Synthetic Data Generation
                "Synthetic Data": [
                    "generative-adversarial-networks",
                    "variational-autoencoders", 
                    "bayesian-networks",
                    "copula-based-synthesis",
                    "marginal-synthesis",
                    "ctgan-synthesis",
                    "table-gan",
                    "privbayes"
                ],
                
                # Perturbation Methods
                "Perturbation": [
                    "additive-noise",
                    "multiplicative-noise", 
                    "laplace-mechanism",
                    "gaussian-mechanism",
                    "exponential-mechanism",
                    "randomized-response"
                ],
                
                # Cryptographic Methods
                "Cryptographic": [
                    "secure-multiparty-computation",
                    "homomorphic-encryption",
                    "functional-encryption",
                    "private-set-intersection"
                ],
                
                # Data Transformation
                "Data Transformation": [
                    "data-swapping",
                    "micro-aggregation",
                    "rank-swapping",
                    "top-bottom-coding",
                    "global-recoding",
                    "local-recoding"
                ],
                
                # Clustering & Grouping
                "Clustering & Grouping": [
                    "mondrian-anonymization",
                    "incognito-anonymization", 
                    "anatomy-anonymization",
                    "slicing-anonymization",
                    "bucketization"
                ],
                
                # Advanced Privacy Models
                "Advanced Privacy": [
                    "delta-presence",
                    "beta-likeness",
                    "personalized-privacy",
                    "crowd-blending-privacy",
                    "distributional-privacy"
                ],
                
                # Federated Learning
                "Federated Learning": [
                    "federated-averaging",
                    "secure-aggregation",
                    "private-federated-learning"
                ],
                
                # Custom & Hybrid
                "Custom & Hybrid": [
                    "hybrid-anonymization",
                    "custom-method"
                ]
            }
            
            # Flatten all methods for search
            all_methods = []
            for category, methods in anonymization_methods.items():
                all_methods.extend(methods)
            
            # Search functionality
            search_query = st.text_input(
                "🔍 Search for anonymization method:",
                placeholder="Type to search (e.g., 'differential', 'k-anon', 'encryption')...",
                help="Start typing to filter methods. Search works on method names and categories.",
                key="anon_method_search"
            )
            
            
            # Filter methods based on search
            if search_query:
                # Case-insensitive search across method names and categories
                search_lower = search_query.lower()
                filtered_methods = []
                
                for category, methods in anonymization_methods.items():
                    category_matches = search_lower in category.lower()
                    matching_methods = [m for m in methods if search_lower in m.lower()]
                    
                    if category_matches:
                        # If category matches, include all methods from that category
                        filtered_methods.extend(methods)
                    else:
                        # Include only methods that match
                        filtered_methods.extend(matching_methods)
                
                # Remove duplicates and sort
                filtered_methods = sorted(list(set(filtered_methods)))
                
                if filtered_methods:
                    st.info(f"🔍 Found {len(filtered_methods)} method(s) matching '{search_query}'")
                    
                    # Display filtered methods in a more compact selectbox
                    anon_method = st.selectbox(
                        "Select from filtered results:",
                        options=filtered_methods,
                        help=f"Showing methods matching '{search_query}'. Clear search to see all methods.",
                        key="anon_method_filtered"
                    )
                    
                    # Show which category the selected method belongs to
                    if anon_method:
                        for category, methods in anonymization_methods.items():
                            if anon_method in methods:
                                st.caption(f"📂 Category: **{category}**")
                                break
                
                else:
                    st.warning(f"❌ No methods found matching '{search_query}'. Try different keywords or clear search.")
                    anon_method = None
                    
                    # Show search suggestions
                    st.markdown("**💡 Search Suggestions:**")
                    suggestions = [
                        ("**Popular terms:** k-anon, differential, encryption, synthesis", "🔥"),
                        ("**By category:** classical, crypto, federated, perturbation", "📂"),
                        ("**By technique:** noise, swap, cluster, anonymize", "🔧")
                    ]
                    
                    for suggestion, icon in suggestions:
                        st.markdown(f"{icon} {suggestion}")
            
            else:
                # Show all methods organized by category when no search
                st.info("📋 Browse all available methods by category, or use search above")
                
                # Category-based selection for better UX
                selected_category = st.selectbox(
                    "1️⃣ First, choose a category:",
                    options=list(anonymization_methods.keys()),
                    help="Select a privacy technique category to see specific methods",
                    key="anon_category_select"
                )
                
                if selected_category:
                    category_methods = anonymization_methods[selected_category]
                    
                    # Show category description
                    category_descriptions = {
                        "Classical Privacy": "Traditional anonymization techniques like k-anonymity and l-diversity",
                        "Differential Privacy": "Mathematical privacy guarantees with controlled noise injection",
                        "Synthetic Data": "Generate artificial data that preserves statistical properties",
                        "Perturbation": "Add controlled noise or modifications to original data",
                        "Cryptographic": "Use encryption and secure computation techniques",
                        "Data Transformation": "Modify data through swapping, coding, and aggregation",
                        "Clustering & Grouping": "Group similar records for anonymization",
                        "Advanced Privacy": "Modern privacy models with enhanced guarantees",
                        "Federated Learning": "Distributed learning with privacy preservation",
                        "Custom & Hybrid": "Custom implementations and hybrid approaches"
                    }
                    
                    if selected_category in category_descriptions:
                        st.caption(f"ℹ️ {category_descriptions[selected_category]}")
                    
                    anon_method = st.selectbox(
                        "2️⃣ Then, select specific method:",
                        options=category_methods,
                        help=f"Choose a specific {selected_category.lower()} method for your dataset",
                        key="anon_method_by_category"
                    )
                    
                    # Show method count in category
                    st.caption(f"📊 {len(category_methods)} methods available in {selected_category}")
                else:
                    anon_method = None
            
            # Show quick method info if selected
            if anon_method:
                with st.expander(f"ℹ️ About {anon_method.title()}", expanded=False):
                    # Method-specific descriptions
                    method_info = {
                        "k-anonymity": "Ensures each record is identical to at least k-1 others on quasi-identifiers. Simple but effective for basic privacy.",
                        "l-diversity": "Extends k-anonymity by requiring diversity in sensitive attributes. Better protection against homogeneity attacks.",
                        "t-closeness": "Requires distribution of sensitive attributes to be close to overall distribution. Strongest among classical methods.",
                        "differential-privacy": "Adds calibrated noise to provide mathematical privacy guarantees. Gold standard for statistical privacy.",
                        "generative-adversarial-networks": "Uses GAN architecture to generate synthetic data. Good for complex data distributions.",
                        "homomorphic-encryption": "Allows computation on encrypted data. Enables secure outsourced computation.",
                        "federated-averaging": "Trains models across decentralized data. Popular for distributed machine learning.",
                        "data-swapping": "Swaps values between records to reduce disclosure risk. Preserves marginal distributions.",
                        "micro-aggregation": "Replaces individual values with group averages. Reduces granularity while preserving trends."
                    }
                    
                    # Default description for methods not in the info dict
                    description = method_info.get(
                        anon_method, 
                        f"Privacy-preserving technique: {anon_method.replace('-', ' ').title()}. Consult documentation for specific implementation details."
                    )
                    
                    st.write(description)
                    
                    # Add technical complexity indicator
                    complexity_levels = {
                        # Simple
                        "k-anonymity": "🟢 Simple",
                        "l-diversity": "🟢 Simple", 
                        "data-swapping": "🟢 Simple",
                        "additive-noise": "🟢 Simple",
                        
                        # Moderate
                        "t-closeness": "🟡 Moderate",
                        "differential-privacy": "🟡 Moderate",
                        "micro-aggregation": "🟡 Moderate",
                        "bucketization": "🟡 Moderate",
                        
                        # Complex
                        "generative-adversarial-networks": "🔴 Complex",
                        "homomorphic-encryption": "🔴 Complex",
                        "secure-multiparty-computation": "🔴 Complex",
                        "federated-differential-privacy": "🔴 Complex"
                    }
                    
                    complexity = complexity_levels.get(anon_method, "🟡 Moderate")
                    st.write(f"**Implementation Complexity:** {complexity}")
                    
                    # Add use case recommendations
                    use_cases = {
                        "k-anonymity": "📊 Good for: Basic anonymization, regulatory compliance, moderate privacy needs",
                        "differential-privacy": "🔒 Good for: Statistical releases, research data, high privacy requirements",
                        "generative-adversarial-networks": "🎯 Good for: Data sharing, ML training, complex datasets",
                        "homomorphic-encryption": "🛡️ Good for: Secure computation, cloud processing, sensitive calculations"
                    }
                    
                    if anon_method in use_cases:
                        st.write(use_cases[anon_method])
            
            # Advanced search tips
            if search_query and not filtered_methods:
                with st.expander("🔍 Advanced Search Tips", expanded=False):
                    st.markdown("""
                    **🎯 Search Tips:**
                    - **Partial matching:** Try "diff" for differential privacy, "anon" for anonymization
                    - **Category search:** "crypto" for cryptographic methods, "synth" for synthetic data
                    - **Common terms:** "noise", "encrypt", "federated", "cluster", "swap"
                    - **Technique types:** "mechanism", "aggregation", "perturbation"
                    
                    **📚 Popular Searches:**
                    - `k-anon` → k-anonymity and related methods
                    - `differential` → All differential privacy variants  
                    - `gan` → GAN-based synthetic data methods
                    - `encrypt` → Encryption-based techniques
                    - `federated` → Federated learning approaches
                    """)
            
            # Show current count for this method
            # Actual number of datasets currently loaded for this method
            actual_loaded_count = len(st.session_state.datasets_collection['anonymized_datasets'].get(anon_method, []))
            
            # Highest ID number assigned so far for this method (from counters)
            # This is what your original 'current_count' was effectively tracking for 'next_number' generation
            highest_id_assigned_so_far = st.session_state.dataset_counters.get(anon_method, 0)
            next_sequential_id = highest_id_assigned_so_far + 1 # This is your original 'next_number'
            
            st.info(f"📊 **{anon_method.title()}**: {actual_loaded_count} dataset(s) currently loaded. Next upload for this method will be identified as #{next_sequential_id}.")
            
            # ADDED: Optional custom label for the dataset
            user_label = st.text_input(
                f"Optional: Custom label for {anon_method} dataset #{next_sequential_id}",
                placeholder="e.g., k=5 adult_data, epsilon_0.1 strong_privacy",
                key=f"label_anon_{anon_method}_{next_sequential_id}",
                help="Provide a descriptive name for this dataset variant. If left blank, the default ID will be used."
            )
            
            anon_file = st.file_uploader(
                f"Upload {anon_method} dataset #{next_sequential_id}", 
                type=["csv", "xlsx"],
                key=f"unified_anon_{anon_method}_{next_sequential_id}", # Key uses the next sequential ID
                help=f"Upload dataset #{next_sequential_id} processed with {anon_method}"
            )
            
            # Enhanced add button with dataset naming
            col1, col2 = st.columns([3, 1])
            
            with col1:
                if anon_file and st.button(f"📥 Add {anon_method} Dataset #{next_sequential_id}", key=f"add_{anon_method}_{next_sequential_id}", type="primary", use_container_width=True):
                    with st.spinner(f"Loading {anon_method} dataset #{next_sequential_id}..."):
                        anon_df = load_data(anon_file)
                        if anon_df is not None:
                            # Initialize the method list if it doesn't exist
                            if anon_method not in st.session_state.datasets_collection['anonymized_datasets']:
                                st.session_state.datasets_collection['anonymized_datasets'][anon_method] = []
                            
                            # Create unique dataset identifier using sequential ID
                            dataset_id = f"{anon_method}_{next_sequential_id}"
                            
                            # Store dataset with metadata including user label
                            dataset_entry = {
                                'data': anon_df,
                                'id': dataset_id,
                                'number': next_sequential_id,
                                'filename': anon_file.name,
                                'upload_time': pd.Timestamp.now(),
                                'method': anon_method,
                                'user_label': user_label if user_label.strip() else None
                            }
                            
                            # Add to the list
                            st.session_state.datasets_collection['anonymized_datasets'][anon_method].append(dataset_entry)
                            
                            # Update counter to track highest ID used
                            st.session_state.dataset_counters[anon_method] = next_sequential_id
                            
                            # Store metadata including user label
                            st.session_state.dataset_metadata[dataset_id] = {
                                'name': anon_file.name,
                                'type': 'anonymized',
                                'method': anon_method,
                                'number': next_sequential_id,
                                'upload_time': pd.Timestamp.now(),
                                'user_label': user_label if user_label.strip() else None
                            }
                            
                            success_message = f"✅ {anon_method} dataset #{next_sequential_id} added successfully!"
                            if user_label.strip():
                                success_message += f" (Label: {user_label.strip()})"
                            st.success(success_message)
                            st.rerun()
                        else:
                            st.error(f"❌ Failed to load {anon_method} dataset #{next_sequential_id}")
            
            # with col2:
            #     # Quick info about this method
            #     if current_count > 0:
            #         st.metric(
            #             label=f"📊 {anon_method.title()}",
            #             value=f"{current_count} datasets",
            #             delta="Ready for comparison"
            #         )
            
            # Enhanced dataset management interface
            st.markdown("---")
            st.markdown("**📊 Loaded Anonymized Dataset Collection:**")
            
            if st.session_state.datasets_collection['anonymized_datasets']:
                # Create expandable sections for each method
                for method, dataset_list in st.session_state.datasets_collection['anonymized_datasets'].items():
                    method_count = len(dataset_list)
                    
                    with st.expander(f"🔒 **{method.title()}** ({method_count} dataset{'' if method_count == 1 else 's'})", expanded=method_count <= 3):
                        
                        if dataset_list:
                            # Create a table for this method's datasets
                            method_data = []
                            for dataset_entry in dataset_list:
                                df = dataset_entry['data']
                                
                                # Create display name with user label if available
                                display_name = f"#{dataset_entry['number']}"  # Default display
                                if dataset_entry.get('user_label'):
                                    display_name = f"{dataset_entry['user_label']} (#{dataset_entry['number']})"
                                
                                method_data.append({
                                    "📊 Dataset": display_name,
                                    "📄 Filename": dataset_entry['filename'],
                                    "📏 Shape": f"{df.shape[0]:,} × {df.shape[1]:,}",
                                    "💾 Size": f"{df.memory_usage(deep=True).sum() / 1024**2:.1f} MB",
                                    "⏰ Uploaded": dataset_entry['upload_time'].strftime('%H:%M:%S'),
                                    "🎯 Actions": dataset_entry['id']  # Use ID for action buttons
                                })
                            
                            method_df = pd.DataFrame(method_data)
                            
                            # Display the table without the Actions column for cleaner look
                            display_df = method_df.drop('🎯 Actions', axis=1)
                            st.dataframe(display_df, use_container_width=True, hide_index=True)
                            
                            # Action buttons for each dataset
                            st.markdown("**🎛️ Dataset Actions:**")
                            
                            # Create action buttons in rows
                            action_cols = st.columns(min(len(dataset_list), 4))
                            
                            for i, dataset_entry in enumerate(dataset_list):
                                col_idx = i % 4
                                with action_cols[col_idx]:
                                    
                                    # Remove button
                                    if st.button(f"🗑️ Remove #{dataset_entry['number']}", 
                                            key=f"remove_{dataset_entry['id']}", 
                                            type="secondary", 
                                            use_container_width=True):
                                        
                                        # Remove from datasets collection
                                        st.session_state.datasets_collection['anonymized_datasets'][method] = [
                                            d for d in dataset_list if d['id'] != dataset_entry['id']
                                        ]
                                        
                                        # Clean up empty method
                                        if not st.session_state.datasets_collection['anonymized_datasets'][method]:
                                            del st.session_state.datasets_collection['anonymized_datasets'][method]
                                            if method in st.session_state.dataset_counters:
                                                del st.session_state.dataset_counters[method]
                                        
                                        # Remove metadata
                                        if dataset_entry['id'] in st.session_state.dataset_metadata:
                                            del st.session_state.dataset_metadata[dataset_entry['id']]
                                        
                                        st.success(f"✅ {method} dataset #{dataset_entry['number']} removed!")
                                        st.rerun()
                            
                            # Method-level actions
                            st.markdown("**🔧 Method Actions:**")
                            method_action_col1, method_action_col2, method_action_col3 = st.columns(3)
                            
                            with method_action_col1:
                                if st.button(f"📊 Compare All {method.title()}", 
                                        key=f"compare_all_{method}", 
                                        use_container_width=True):
                                    st.info(f"🎯 Comparison feature for all {method} datasets will be available after training!")
                            
                            with method_action_col2:
                                if st.button(f"📈 Method Statistics", 
                                        key=f"stats_{method}", 
                                        use_container_width=True):
                                    
                                    # Show statistics for this method
                                    total_rows = sum(d['data'].shape[0] for d in dataset_list)
                                    avg_rows = total_rows / len(dataset_list)
                                    total_size = sum(d['data'].memory_usage(deep=True).sum() for d in dataset_list)
                                    
                                    stats_data = [
                                        ["📊 Total Datasets", len(dataset_list)],
                                        ["📏 Total Rows", f"{total_rows:,}"],
                                        ["📊 Average Rows", f"{avg_rows:,.0f}"],
                                        ["💾 Total Size", f"{total_size / 1024**2:.1f} MB"],
                                        ["📅 Latest Upload", max(d['upload_time'] for d in dataset_list).strftime('%H:%M:%S')]
                                    ]
                                    
                                    stats_df = pd.DataFrame(stats_data, columns=['Metric', 'Value'])
                                    st.dataframe(stats_df, use_container_width=True, hide_index=True)
                            
                            with method_action_col3:
                                # Initialize confirmation state for this method if not exists
                                confirm_key = f"confirm_remove_all_{method}"
                                if confirm_key not in st.session_state:
                                    st.session_state[confirm_key] = False
                                
                                if not st.session_state[confirm_key]:
                                    # Show the initial "Remove All" button
                                    if st.button(f"🗑️ Remove All {method.title()}", 
                                                key=f"remove_all_{method}", 
                                                type="secondary", 
                                                use_container_width=True):
                                        # Set confirmation state to True
                                        st.session_state[confirm_key] = True
                                        st.rerun()
                                else:
                                    # Show confirmation buttons WITHOUT using st.columns() in sidebar
                                    # Use sequential layout instead of side-by-side columns
                                    
                                    # Confirmation button
                                    if st.button(f"⚠️ Confirm Remove All {method.title()}", 
                                                key=f"confirm_yes_{method}", 
                                                type="secondary",
                                                use_container_width=True):
                                        
                                        # Remove all datasets for this method
                                        for dataset_entry in dataset_list:
                                            if dataset_entry['id'] in st.session_state.dataset_metadata:
                                                del st.session_state.dataset_metadata[dataset_entry['id']]
                                        
                                        del st.session_state.datasets_collection['anonymized_datasets'][method]
                                        if method in st.session_state.dataset_counters:
                                            del st.session_state.dataset_counters[method]
                                        
                                        # Reset confirmation state
                                        st.session_state[confirm_key] = False
                                        
                                        st.success(f"✅ All {method} datasets removed!")
                                        st.rerun()
                                    
                                    # Cancel button (placed below the confirm button)
                                    if st.button("❌ Cancel", 
                                                key=f"confirm_no_{method}", 
                                                use_container_width=True):
                                        # Reset confirmation state
                                        st.session_state[confirm_key] = False
                                        st.rerun()
                # Quick start guide
                with st.expander("🚀 **Quick Start Guide**", expanded=True):
                    # Professional PPML Quick Start Guide with comprehensive structured information
                    st.markdown("""
                    ### 🚀 **Professional PPML Quick Start Guide**
                    *Comprehensive guide for privacy-preserving machine learning benchmarking*
                    """)
                    
                    # Structured Information Sections
                    guide_tab1, guide_tab2, guide_tab3, guide_tab4 = st.tabs([
                        "📋 **Workflow Overview**", 
                        "🔧 **Technical Setup**", 
                        "📊 **Analysis Strategy**", 
                        "💡 **Best Practices**"
                    ])
                    
                    with guide_tab1:
                        st.markdown("""
                        ### 📋 **Complete PPML Workflow**
                        
                        **Phase 1: Data Preparation** 🗂️
                        ```
                        1. Load original dataset (baseline for comparison)
                        2. Apply privacy-preserving techniques with different parameters
                        3. Upload multiple variants per anonymization method
                        4. Verify data schema consistency across all datasets
                        ```
                        
                        **Phase 2: Privacy Method Configuration** 🔒
                        ```
                        📌 k-Anonymity Variants:
                           • k=3 (minimal privacy, high utility)
                           • k=5 (balanced privacy-utility)
                           • k=10 (high privacy, moderate utility)
                        
                        📌 l-Diversity Variants:
                           • l=2 (basic diversity requirement)
                           • l=3 (enhanced diversity)
                           • l=5 (maximum diversity)
                        
                        📌 Differential Privacy Variants:
                           • ε=1.0 (strong privacy)
                           • ε=0.5 (moderate privacy)
                           • ε=0.1 (maximum privacy)
                        ```
                        
                        **Phase 3: ML Training & Evaluation** 🤖
                        ```
                        1. Train identical models on all datasets
                        2. Apply consistent evaluation metrics
                        3. Measure privacy-utility trade-offs
                        4. Generate comprehensive comparison reports
                        ```
                        
                        **Phase 4: Analysis & Reporting** 📈
                        ```
                        1. Compare performance across privacy methods
                        2. Analyze utility degradation patterns
                        3. Identify optimal privacy-utility configurations
                        4. Generate actionable insights for deployment
                        ```
                        """)
                    
                    with guide_tab2:
                        st.markdown("""
                        ### 🔧 **Technical Implementation Guide**
                        
                        **Dataset Requirements** 📊
                        """)
                        
                        # Technical requirements table
                        tech_requirements = pd.DataFrame([
                            ["📄 File Format", "CSV, XLSX", "Supported formats for upload"],
                            ["🔗 Schema Consistency", "Identical columns", "All datasets must have same structure"],
                            ["🎯 Target Column", "Same across datasets", "Consistent target variable naming"],
                            ["📏 Sample Size", "≥ 100 samples", "Minimum for reliable ML training"],
                            ["🏷️ Data Types", "Numeric, Categorical", "Mixed data types supported"],
                            ["❌ Missing Values", "< 20% per column", "Excessive missing values affect quality"]
                        ], columns=["Requirement", "Specification", "Description"])
                        
                        st.dataframe(tech_requirements, use_container_width=True, hide_index=True)
                        
                        st.markdown("""
                        **File Naming Conventions** 📝
                        ```
                        ✅ RECOMMENDED:
                        • k_anonymity_k3_balanced.csv
                        • l_diversity_l2_sensitive_attr.csv
                        • differential_privacy_eps_1_0.csv
                        • t_closeness_t_0_2_hierarchical.csv
                        
                        ❌ AVOID:
                        • dataset1.csv, dataset2.csv
                        • anonymized_data.csv
                        • final_version.csv
                        ```
                        
                        **Quality Assurance Checklist** ✅
                        ```
                        Before Upload:
                        □ Verify column names match original dataset
                        □ Check data types are consistent
                        □ Validate target column exists and populated
                        □ Ensure no duplicate or corrupted rows
                        □ Confirm privacy technique applied correctly
                        □ Document anonymization parameters used
                        ```
                        """)
                    
                    with guide_tab3:
                        st.markdown("""
                        ### 📊 **Strategic Analysis Framework**
                        
                        **Multi-Dimensional Comparison Strategy** 🎯
                        """)
                        
                        # Analysis framework table
                        analysis_framework = pd.DataFrame([
                            ["🔒 Privacy Level", "Data Reduction %", "Measure information loss"],
                            ["📈 Utility Retention", "Performance Metrics", "ML model accuracy preservation"],
                            ["⚖️ Privacy-Utility Trade-off", "Composite Score", "Balanced evaluation metric"],
                            ["🎯 Method Effectiveness", "Cross-Method Comparison", "Best technique identification"],
                            ["📊 Parameter Sensitivity", "Within-Method Analysis", "Optimal parameter selection"],
                            ["🚀 Deployment Readiness", "Real-world Applicability", "Production suitability assessment"]
                        ], columns=["Analysis Dimension", "Measurement", "Purpose"])
                        
                        st.dataframe(analysis_framework, use_container_width=True, hide_index=True)
                        
                        st.markdown("""
                        **Evaluation Metrics Hierarchy** 📏
                        ```
                        PRIMARY METRICS (Model Performance):
                        ├── Accuracy: Overall correctness
                        ├── F1-Score: Balanced precision-recall
                        ├── Precision: False positive control
                        └── Recall: True positive coverage
                        
                        SECONDARY METRICS (Privacy Assessment):
                        ├── Data Retention: % of original data preserved
                        ├── Feature Preservation: Column-level utility
                        ├── Distribution Similarity: Statistical fidelity
                        └── Re-identification Risk: Privacy level quantification
                        
                        COMPOSITE METRICS (Trade-off Analysis):
                        ├── Utility-Privacy Ratio: Combined effectiveness
                        ├── Performance Degradation: Quality loss measurement
                        └── Cost-Benefit Score: Deployment decision support
                        ```
                        
                        **Result Interpretation Guidelines** 🔍
                        ```
                        HIGH-UTILITY SCENARIOS (≥90% performance retention):
                        • Suitable for production deployment
                        • Minimal privacy-utility trade-off
                        • Recommended for sensitive applications
                        
                        MODERATE-UTILITY SCENARIOS (70-90% retention):
                        • Acceptable for most use cases
                        • Moderate privacy gains
                        • Consider parameter tuning
                        
                        LOW-UTILITY SCENARIOS (<70% retention):
                        • High privacy protection
                        • Significant utility loss
                        • Evaluate if privacy requirements justify cost
                        ```
                        """)
                    
                    with guide_tab4:
                        st.markdown("""
                        ### 💡 **Professional Best Practices**
                        
                        **Dataset Management Excellence** 🗂️
                        """)
                        
                        # Best practices table
                        best_practices = pd.DataFrame([
                            ["📋 Documentation", "Maintain detailed parameter logs", "🟢 Critical"],
                            ["🔄 Version Control", "Track dataset modifications", "🟢 Critical"],
                            ["🧪 Validation Testing", "Cross-validate on test sets", "🟢 Critical"],
                            ["📊 Baseline Comparison", "Always include original dataset", "🟢 Critical"],
                            ["⚖️ Balance Assessment", "Evaluate class distribution changes", "🟡 Important"],
                            ["🔒 Privacy Verification", "Validate anonymization effectiveness", "🟡 Important"],
                            ["📈 Performance Monitoring", "Track metrics across iterations", "🟡 Important"],
                            ["💾 Backup Strategy", "Secure data storage protocols", "🔵 Recommended"]
                        ], columns=["Practice", "Description", "Priority"])
                        
                        st.dataframe(best_practices, use_container_width=True, hide_index=True)
                        
                        st.markdown("""
                        **Advanced Configuration Strategies** ⚙️
                        ```
                        PROGRESSIVE PRIVACY TESTING:
                        1. Start with minimal privacy (k=3, ε=1.0)
                        2. Gradually increase privacy levels
                        3. Monitor utility degradation patterns
                        4. Identify optimal privacy-utility sweet spots
                        
                        MULTI-ALGORITHM VALIDATION:
                        1. Test multiple ML algorithms per dataset
                        2. Assess algorithm sensitivity to privacy techniques
                        3. Identify robust algorithm-privacy combinations
                        4. Validate results across different model families
                        
                        PARAMETER OPTIMIZATION WORKFLOW:
                        1. Grid search across privacy parameters
                        2. Cross-validation with multiple random seeds
                        3. Statistical significance testing
                        4. Confidence interval reporting
                        ```
                        
                        **Production Deployment Checklist** 🚀
                        ```
                        PRE-DEPLOYMENT VALIDATION:
                        □ Privacy requirements compliance verified
                        □ Utility thresholds met consistently
                        □ Regulatory compliance confirmed
                        □ Stakeholder approval obtained
                        □ Monitoring infrastructure prepared
                        □ Rollback procedures established
                        
                        POST-DEPLOYMENT MONITORING:
                        □ Performance metrics tracking active
                        □ Privacy breach detection systems online
                        □ Regular re-evaluation scheduled
                        □ Incident response procedures tested
                        ```
                        
                        **Common Pitfalls & Solutions** ⚠️
                        """)
                        
                        pitfalls_solutions = pd.DataFrame([
                            ["🔴 Schema Mismatch", "Inconsistent column names/types", "Standardize preprocessing pipeline"],
                            ["🔴 Target Leakage", "Privacy technique affects target", "Validate target distribution preservation"],
                            ["🔴 Overfitting", "Model memorizes privacy artifacts", "Use proper train/validation/test splits"],
                            ["🔴 Evaluation Bias", "Inconsistent metric calculation", "Standardize evaluation protocols"],
                            ["🔴 Parameter Drift", "Inconsistent anonymization parameters", "Document and version all parameters"],
                            ["🔴 Sample Bias", "Non-representative test samples", "Ensure balanced sampling strategies"]
                        ], columns=["Issue Type", "Problem", "Solution"])
                        
                        st.dataframe(pitfalls_solutions, use_container_width=True, hide_index=True)
                        
                        st.markdown("""
                        **Success Metrics & KPIs** 📈
                        ```
                        PROJECT SUCCESS INDICATORS:
                        • Privacy compliance: 100% regulatory adherence
                        • Utility retention: ≥85% performance maintenance
                        • Cost efficiency: <20% computational overhead
                        • Deployment readiness: Production-grade quality
                        • Stakeholder satisfaction: Business requirements met
                        ```
                        """)
            
        else:
            st.info("👆 Load an original dataset first to enable anonymized dataset uploads")
            
            # Enhanced guidance for PPML beginners
            with st.expander("📚 **Privacy-Preserving ML Guide**", expanded=False):
                st.markdown("""
                **🔒 Understanding Privacy-Preserving ML:**
                
                **🎯 What are anonymization methods?**
                - **k-anonymity**: Ensures each record is identical to at least k-1 others
                - **l-diversity**: Adds diversity requirement for sensitive attributes  
                - **t-closeness**: Maintains distribution similarity for sensitive attributes
                - **Differential Privacy**: Adds mathematical noise for privacy guarantees
                
                **📊 Why multiple datasets per method?**
                - Test different parameter values (k=3 vs k=5 vs k=10)
                - Compare preprocessing approaches
                - Evaluate privacy-utility trade-offs comprehensively
                
                **🚀 Getting Started:**
                1. Load your original dataset above
                2. Apply privacy techniques with different parameters
                3. Upload multiple variants per method
                4. Train models and compare results
                """)
        
        st.markdown("---")
        
    with tab3:
        st.subheader("Data Preprocessing")
        
        # Global preprocessing settings
        enable_preprocessing = st.checkbox(
            "Enable Preprocessing", 
            value=st.session_state.preprocessing_configs.get('enabled', False),
            help="Apply preprocessing to all datasets"
        )
        
        if enable_preprocessing:
            col1, col2 = st.columns(2)
            
            with col1:
                # Scaling options
                scaling_method = st.selectbox(
                    "Scaling Method:",
                    ["None", "StandardScaler", "MinMaxScaler", "RobustScaler"],
                    index=["None", "StandardScaler", "MinMaxScaler", "RobustScaler"].index(
                        st.session_state.preprocessing_configs.get('scaling', 'None')
                    )
                )
                
                # Encoding options
                categorical_encoding = st.selectbox(
                    "Categorical Encoding:",
                    ["None", "OneHotEncoder", "LabelEncoder", "TargetEncoder"],
                    index=["None", "OneHotEncoder", "LabelEncoder", "TargetEncoder"].index(
                        st.session_state.preprocessing_configs.get('encoding', 'None')
                    )
                )
                
            with col2:
                # Feature selection
                feature_selection = st.checkbox(
                    "Apply Feature Selection",
                    value=st.session_state.preprocessing_configs.get('feature_selection', 'None') != 'None'
                )
                
                if feature_selection:
                    selection_method = st.selectbox(
                        "Selection Method:",
                        ["SelectKBest", "SelectPercentile", "RFE"]
                    )
                else:
                    selection_method = "None"
                
                # Handle missing values
                missing_strategy = st.selectbox(
                    "Missing Value Strategy:",
                    ["median", "mean", "most_frequent", "drop"],
                    index=["median", "mean", "most_frequent", "drop"].index(
                        st.session_state.preprocessing_configs.get('missing_strategy', 'median')
                    )
                )
            
            # Update session state
            st.session_state.preprocessing_configs = {
                'enabled': enable_preprocessing,
                'scaling': scaling_method,
                'encoding': categorical_encoding,
                'feature_selection': selection_method,
                'missing_strategy': missing_strategy
            }
        else:
            # Reset preprocessing configs when disabled
            st.session_state.preprocessing_configs = {
                'enabled': False,
                'scaling': "None",
                'encoding': "None",
                'feature_selection': "None",
                'missing_strategy': "median"            }
        
        st.markdown("---")
    
    with tab4:
        st.subheader("📈 External Results Dashboard")
        st.markdown("Import and analyze results from previous experiments")
        
        # Initialize session state for file-based dashboard
        if 'external_dashboard_enabled' not in st.session_state:
            st.session_state.external_dashboard_enabled = False
        if 'external_dashboard_data' not in st.session_state:
            st.session_state.external_dashboard_data = None
        
        # Toggle to enable/disable the external dashboard
        enable_external_dashboard = st.checkbox(
            "🔓 Activate External Dashboard",
            value=st.session_state.external_dashboard_enabled,
            help="Enable this to upload and analyze results from external experiments using the same PPML analysis capabilities"
        )
        
        st.session_state.external_dashboard_enabled = enable_external_dashboard
        
        if enable_external_dashboard:
            st.success("✅ External Dashboard is now active! Scroll to the top of the main area to see it.")
            st.markdown("---")
            
            # File upload section
            st.markdown("**📁 Upload Results File**")
            uploaded_file = st.file_uploader(
                "Choose a file containing experiment results",
                type=['xlsx', 'json', 'csv'],
                help="Upload Excel, JSON, or CSV files containing your ML experiment results"
            )
            
            if uploaded_file is not None:
                try:
                    # Process the uploaded file
                    if uploaded_file.name.endswith('.xlsx'):
                        df = pd.read_excel(uploaded_file)
                        st.success(f"✅ Excel file loaded: {uploaded_file.name}")
                    elif uploaded_file.name.endswith('.json'):
                        import json
                        json_data = json.load(uploaded_file)
                        df = pd.json_normalize(json_data) if isinstance(json_data, list) else pd.DataFrame([json_data])
                        st.success(f"✅ JSON file loaded: {uploaded_file.name}")
                    elif uploaded_file.name.endswith('.csv'):
                        df = pd.read_csv(uploaded_file)
                        st.success(f"✅ CSV file loaded: {uploaded_file.name}")
                    
                    # Store the data in session state
                    st.session_state.external_dashboard_data = df
                    
                    # Show data preview
                    st.markdown("**📊 Data Preview**")
                    st.dataframe(df.head(), use_container_width=True)
                    st.markdown(f"**📈 Dataset Info**: {df.shape[0]} rows × {df.shape[1]} columns")
                    
                except Exception as e:
                    st.error(f"❌ Error loading file: {str(e)}")
                    st.session_state.external_dashboard_data = None
            
            # Show expected format examples
            with st.expander("📋 Expected File Format Examples"):
                st.markdown("""
                **Excel/CSV Format Example:**
                
                | 🗂️ Dataset | 🔒 Type | 🤖 Model | 🎯 Accuracy | 🏆 F1-Score | 🎯 Accuracy % Diff |
                |-------------|---------|-----------|------------|-------------|-------------------|
                | Adult | Original | RandomForest | 0.85 | 0.82 | 0.00% |
                | Adult | K-Anonymity | RandomForest | 0.78 | 0.75 | -8.24% ↓ |
                | Adult | Differential Privacy | SVM | 0.72 | 0.70 | -15.29% ↓ |
                
                **JSON Format Example:**
                ```json
                [
                  {"🗂️ Dataset": "Adult", "🔒 Type": "Original", "🤖 Model": "RandomForest", "🎯 Accuracy": 0.85, "🏆 F1-Score": 0.82},
                  {"🗂️ Dataset": "Adult", "🔒 Type": "K-Anonymity", "🤖 Model": "RandomForest", "🎯 Accuracy": 0.78, "🏆 F1-Score": 0.75}
                ]
                ```
                
                **Column Requirements:**
                - Must have Model column (e.g., 🤖 Model)
                - Must have performance metrics (e.g., 🎯 Accuracy, 🏆 F1-Score)
                - Can include privacy type/method columns
                - Can include percentage difference columns with % Diff
                """)
        else:
            st.info("Enable the dashboard above to upload and analyze external experiment results with full PPML capabilities.")
        st.markdown("---")
    # Dataset Overview Dashboard
    if st.session_state.datasets_collection['original'] is not None:
        st.header("📋 Dataset Collection Overview")
        
        overview_data = []
        
        # Add original dataset info
        overview_data.append({
            "📊 Component": "Original Dataset",
            "✅ Status": "Loaded",
            "📏 Details": f"{st.session_state.datasets_collection['original'].shape[0]:,} rows × {st.session_state.datasets_collection['original'].shape[1]:,} cols"
        })
        
        # Calculate anonymized dataset counts correctly
        total_anon_datasets = 0
        anon_methods_count = len(st.session_state.datasets_collection['anonymized_datasets'])
        method_details_summary = []

        for method, dataset_entry_list in st.session_state.datasets_collection['anonymized_datasets'].items():
            if isinstance(dataset_entry_list, list):
                total_anon_datasets += len(dataset_entry_list)
                method_details_summary.append(f"{method.title()}: {len(dataset_entry_list)}")
            else:
                # This case should ideally not happen if data is structured correctly
                st.warning(f"Data for method '{method}' in overview has unexpected structure. Expected a list of dataset entries.")
                # Attempt to count if it's a single DataFrame (legacy or error state)
                if isinstance(dataset_entry_list, pd.DataFrame):
                    total_anon_datasets += 1
                    method_details_summary.append(f"{method.title()}: 1 (unexpected structure)")


        if total_anon_datasets > 0:
            details_text_summary = f"{total_anon_datasets} datasets across {anon_methods_count} methods ({', '.join(method_details_summary)})"
        else:
            details_text_summary = "Upload anonymized data to enable PPML comparison"

        overview_data.append({
            "📊 Component": "Anonymized Datasets", 
            "✅ Status": f"{total_anon_datasets} loaded" if total_anon_datasets > 0 else "None",
            "📏 Details": details_text_summary
        })
        
        # Add preprocessing status (from original code)
        preprocessing_status = "Enabled" if st.session_state.preprocessing_configs.get('enabled', False) else "Disabled"
        preprocessing_methods_list = [] # Renamed to avoid conflict
        if st.session_state.preprocessing_configs.get('enabled', False):
            config = st.session_state.preprocessing_configs
            if config.get('scaling') != 'None':
                preprocessing_methods_list.append(config['scaling'])
            if config.get('encoding') != 'None':
                preprocessing_methods_list.append(config['encoding'])
            if config.get('feature_selection') != 'None' and config.get('feature_selection') != False : # Check for actual method string
                preprocessing_methods_list.append(config['feature_selection'])
        
        preprocessing_details = ", ".join(preprocessing_methods_list) if preprocessing_methods_list else "No methods applied"
        
        overview_data.append({
            "📊 Component": "Preprocessing",
            "✅ Status": preprocessing_status,
            "📏 Details": preprocessing_details
        })
        
        # Add total for comparison (from original code, using corrected total_anon_datasets)
        total_datasets_for_analysis = 1 + total_anon_datasets
        
        overview_data.append({
            "📊 Component": "Total for Analysis",
            "✅ Status": f"{total_datasets_for_analysis} datasets",
            "📏 Details": f"Enhanced PPML comparison with {anon_methods_count} privacy methods" if total_anon_datasets > 0 else "Single dataset analysis"
        })
        
        # Create and display the overview table
        overview_df = pd.DataFrame(overview_data)
        
        # Professional table styling (from original code)
        styled_overview = overview_df.style.set_properties(**{
            'padding': '12px',
            'font-size': '14px',
            'text-align': 'left',
            'border': '1px solid #dee2e6', # Updated to match other tables
            'font-weight': '500'
        }).set_table_styles([
            {'selector': 'th', 'props': [
                ('background-color', '#0d6efd'),
                ('color', 'white'),
                ('font-weight', 'bold'),
                ('text-align', 'center'),
                ('padding', '15px'),
                ('border', '1px solid #0d6efd'),
                ('font-size', '15px')
            ]},
            {'selector': 'tbody tr:nth-child(even)', 'props': [
                ('background-color', 'rgba(0,0,0,0.02)')
            ]},
            {'selector': 'tbody tr:hover', 'props': [
                ('background-color', 'rgba(13,110,253,0.1)'),
                ('transition', 'background-color 0.3s ease')
            ]},
            {'selector': 'td:first-child', 'props': [
                ('font-weight', 'bold'),
                ('color', '#0d6efd')
            ]},
            {'selector': 'td:nth-child(2)', 'props': [
                ('text-align', 'center'),
                ('font-weight', 'bold')
            ]},
            {'selector': 'td', 'props': [ # General td styling
                ('border', '1px solid #dee2e6') 
            ]},
            {'selector': '', 'props': [
                ('border-collapse', 'collapse'),
                ('margin', '20px 0'),
                ('border-radius', '8px'),
                ('overflow', 'hidden'),
                ('box-shadow', '0 4px 6px rgba(0, 0, 0, 0.1)')
            ]}
        ])
        
        st.dataframe(styled_overview, use_container_width=True, hide_index=True)
        
        # Dataset details table (if there are multiple anonymized datasets/methods)
        if anon_methods_count > 0: # anon_methods_count is len(st.session_state.datasets_collection['anonymized_datasets'])
            st.markdown("#### 🔍 **Detailed Dataset Information**")
            
            dataset_details = []
            
            # Add original dataset info
            if st.session_state.datasets_collection['original'] is not None:
                orig_df = st.session_state.datasets_collection['original']
                dataset_details.append({
                    "📂 Dataset Name": "Original",
                    "🔒 Privacy Type": "None (Original Data)",
                    "📏 Rows": f"{orig_df.shape[0]:,}",
                    "📊 Columns": f"{orig_df.shape[1]:,}",
                    "💾 Memory": f"{orig_df.memory_usage(deep=True).sum() / 1024**2:.1f} MB",
                    "✅ Status": "🟢 Ready"
                })
            
            # Add anonymized datasets info, iterating through lists of entries
            for method, dataset_entry_list in st.session_state.datasets_collection['anonymized_datasets'].items():
                if not isinstance(dataset_entry_list, list):
                    st.warning(f"Skipping method '{method}' in Detailed Info due to unexpected data structure. Expected list.")
                    continue

                for dataset_entry in dataset_entry_list:
                    anon_df = dataset_entry['data']
                    
                    # --- MODIFIED ---
                    user_label = dataset_entry.get('user_label')
                    base_name = f"{method.title()} #{dataset_entry['number']}"
                    dataset_name_display = f"{user_label} (#{dataset_entry['number']})" if user_label else base_name
                    # --- END MODIFIED ---

                    if st.session_state.datasets_collection['original'] is not None:
                        orig_shape = st.session_state.datasets_collection['original'].shape
                        row_retention = (anon_df.shape[0] / orig_shape[0]) * 100 if orig_shape[0] > 0 else 0
                        col_retention = (anon_df.shape[1] / orig_shape[1]) * 100 if orig_shape[1] > 0 else 0
                        # --- MODIFIED ---
                        privacy_type_display = f"{dataset_name_display} ({row_retention:.0f}% rows, {col_retention:.0f}% cols)"
                        # --- END MODIFIED ---
                    else:
                        # --- MODIFIED ---
                        privacy_type_display = dataset_name_display # Fallback
                        # --- END MODIFIED ---
                    
                    dataset_details.append({
                        "📂 Dataset Name": dataset_name_display,
                        "🔒 Privacy Type": privacy_type_display,
                        "📏 Rows": f"{anon_df.shape[0]:,}",
                        "📊 Columns": f"{anon_df.shape[1]:,}",
                        "💾 Memory": f"{anon_df.memory_usage(deep=True).sum() / 1024**2:.1f} MB",
                        "✅ Status": "🟢 Ready"
                    })
            
            if dataset_details:
                details_df = pd.DataFrame(dataset_details)
                # Enhanced styling for details table (from original code)
                styled_details = details_df.style.set_properties(**{
                    'padding': '10px',
                    'font-size': '13px',
                    'text-align': 'center',
                    'border': '1px solid #dee2e6',
                    'font-weight': '500'
                }).set_table_styles([
                    {'selector': 'th', 'props': [
                        ('background-color', '#198754'),
                        ('color', 'white'),
                        ('font-weight', 'bold'),
                        ('text-align', 'center'),
                        ('padding', '12px'),
                        ('border', '1px solid #198754'),
                        ('font-size', '14px')
                    ]},
                    {'selector': 'tbody tr:nth-child(even)', 'props': [
                        ('background-color', 'rgba(0,0,0,0.02)')
                    ]},
                    {'selector': 'tbody tr:hover', 'props': [
                        ('background-color', 'rgba(25,135,84,0.1)'),
                        ('transition', 'background-color 0.3s ease')
                    ]},
                    {'selector': 'td:first-child', 'props': [
                        ('font-weight', 'bold'),
                        ('text-align', 'left'),
                        ('color', '#198754')
                    ]},
                    {'selector': 'td:nth-child(2)', 'props': [
                        ('text-align', 'left'),
                        ('font-style', 'italic')
                    ]},
                    {'selector': 'td', 'props': [ # General td styling
                         ('border', '1px solid #dee2e6')
                    ]},
                    {'selector': '', 'props': [
                        ('border-collapse', 'collapse'),
                        ('margin', '15px 0'),
                        ('border-radius', '6px'),
                        ('overflow', 'hidden'),
                        ('box-shadow', '0 2px 4px rgba(0, 0, 0, 0.1)')
                    ]}
                ])
                st.dataframe(styled_details, use_container_width=True, hide_index=True)
            
            # Privacy-utility summary table
            # Check if there's more than just original or if any anonymized data exists for summary
            if total_anon_datasets > 0: 
                st.markdown("#### ⚖️ **Privacy-Utility Summary**")
                
                privacy_summary = []
                
                for method, dataset_entry_list in st.session_state.datasets_collection['anonymized_datasets'].items():
                    if not isinstance(dataset_entry_list, list):
                        st.warning(f"Skipping method '{method}' in Privacy Summary due to unexpected data structure. Expected list.")
                        continue
                    
                    for dataset_entry in dataset_entry_list:
                        anon_df = dataset_entry['data']
                        # --- MODIFIED ---
                        user_label = dataset_entry.get('user_label')
                        base_name = f"{method.title()} #{dataset_entry['number']}"
                        dataset_name_display = f"{user_label} (#{dataset_entry['number']})" if user_label else base_name
                        # --- END MODIFIED ---

                        if st.session_state.datasets_collection['original'] is not None:
                            orig_df = st.session_state.datasets_collection['original']

                            row_retention = (anon_df.shape[0] / orig_df.shape[0]) * 100 if orig_df.shape[0] > 0 else 0
                            col_retention = (anon_df.shape[1] / orig_df.shape[1]) * 100 if orig_df.shape[1] > 0 else 0
                            data_reduction = 100 - row_retention # Based on row retention
                            
                            privacy_level = "⚪ Minimal"
                            if data_reduction > 50: privacy_level = "🔴 High"
                            elif data_reduction > 25: privacy_level = "🟡 Medium"
                            elif data_reduction > 10: privacy_level = "🟢 Low"
                            
                            avg_retention = (row_retention + col_retention) / 2
                            utility_grade = "🔴 Poor"
                            if avg_retention >= 95: utility_grade = "🟢 Excellent"
                            elif avg_retention >= 85: utility_grade = "🟡 Good"
                            elif avg_retention >= 70: utility_grade = "🟠 Fair"
                            
                            trade_off_display = "⚪ Minimal Impact"
                            if data_reduction > 5 : # Only consider trade-off if data reduction is somewhat significant
                                if 70 <= avg_retention <= 90:
                                    trade_off_display = "🟢 Balanced"
                                else:
                                    trade_off_display = "⚠️ Extreme"
                            
                            privacy_summary.append({
                                "🔒 Privacy Method": dataset_name_display,
                                "📉 Data Reduction": f"{data_reduction:.1f}%",
                                "🔒 Privacy Level": privacy_level,
                                "📈 Utility Retention": f"{avg_retention:.1f}%",
                                "⚖️ Utility Grade": utility_grade,
                                "🎯 Trade-off": trade_off_display
                            })
                
                if privacy_summary:
                    summary_df = pd.DataFrame(privacy_summary)
                    # Style privacy summary table (from original code)
                    styled_summary = summary_df.style.set_properties(**{
                        'padding': '10px',
                        'font-size': '13px',
                        'text-align': 'center',
                        'border': '1px solid #dee2e6',
                        'font-weight': '500'
                    }).set_table_styles([
                        {'selector': 'th', 'props': [
                            ('background-color', '#6f42c1'),
                            ('color', 'white'),
                            ('font-weight', 'bold'),
                            ('text-align', 'center'),
                            ('padding', '12px'),
                            ('border', '1px solid #6f42c1'),
                            ('font-size', '14px')
                        ]},
                        {'selector': 'tbody tr:nth-child(even)', 'props': [
                            ('background-color', 'rgba(0,0,0,0.02)')
                        ]},
                        {'selector': 'tbody tr:hover', 'props': [
                            ('background-color', 'rgba(111,66,193,0.1)'),
                            ('transition', 'background-color 0.3s ease')
                        ]},
                        {'selector': 'td:first-child', 'props': [
                            ('font-weight', 'bold'),
                            ('text-align', 'left'),
                            ('color', '#6f42c1')
                        ]},
                        {'selector': 'td', 'props': [ # General td styling
                             ('border', '1px solid #dee2e6')
                        ]},
                        {'selector': '', 'props': [
                            ('border-collapse', 'collapse'),
                            ('margin', '15px 0'),
                            ('border-radius', '6px'),
                            ('overflow', 'hidden'),
                            ('box-shadow', '0 2px 4px rgba(0, 0, 0, 0.1)')
                        ]}
                    ])
                    st.dataframe(styled_summary, use_container_width=True, hide_index=True)
        
        st.markdown("---")  # Separator
    # --- Task Configuration ---
    if st.session_state.df_uploaded is not None:
        st.header("Task Configuration")
        
# Get columns from the current dataset
        available_columns = st.session_state.df_uploaded.columns.tolist()
        
        # Reset target column if it's not in the current dataset
        if (st.session_state.target_column not in available_columns or 
            st.session_state.target_column is None):
            # CHANGED: Set a better default target column
            # Try to find a meaningful target column instead of just the first one
            potential_targets = []
            for col in available_columns:
                col_lower = col.lower()
                # Look for common target column patterns
                if any(keyword in col_lower for keyword in ['target', 'label', 'class', 'rating', 'score', 'outcome', 'result', 'prediction']):
                    potential_targets.append(col)
            
            # If we found potential targets, use the first one; otherwise use the last column
            if potential_targets:
                st.session_state.target_column = potential_targets[0]
            elif len(available_columns) > 1:
                # Use the last column as it's often the target in many datasets
                st.session_state.target_column = available_columns[-1]
            else:
                st.session_state.target_column = available_columns[0] if available_columns else None
            
        # Show dataset info
        st.info(f"📊 Dataset: {st.session_state.df_uploaded.shape[0]} rows, {st.session_state.df_uploaded.shape[1]} columns")
        
        # REMOVE OR COMMENT OUT THE DEBUG SECTION:
        # st.write("**Debug: Available Columns:**")
        # st.write(list(st.session_state.df_uploaded.columns))
        # st.write("**Debug: Current target column:**", st.session_state.target_column)
        # st.write("**Debug: Dataset shape:**", st.session_state.df_uploaded.shape)
        
        # Display available columns for debugging
        with st.expander("🔍 Target Column Selection Guide", expanded=False):
            st.write("**Columns in current dataset:**")
            for i, col in enumerate(available_columns):
                st.write(f"{i+1}. `{col}`")
            
            # ADD COMPREHENSIVE DATASET-AWARE SUGGESTIONS:
            st.markdown("**💡 Smart Target Column Recommendations:**")
            
            # Get the currently loaded dataset info if it's a sample dataset
            current_dataset_info = st.session_state.get('sample_dataset_info', {})
            dataset_key = None
            
            # Try to identify which sample dataset is loaded
            for col in available_columns:
                if col == 'target':
                    # This is likely a sample dataset with 'target' column
                    if current_dataset_info:
                        dataset_key = "sample_with_target"
                    break
            
            # Provide dataset-specific recommendations
            if "target" in available_columns:
                st.success("🎯 **Recommended:** `target` - This is the standard target column for sample datasets")
                
                # Add task-specific guidance based on sample dataset info
                if current_dataset_info:
                    task_type = current_dataset_info.get('task_type', 'unknown')
                    dataset_name = current_dataset_info.get('description', '')
                    
                    if task_type == 'classification':
                        st.write("• **Classification Task**: Predicting discrete categories/classes")
                        if 'iris' in dataset_name.lower():
                            st.write("  - Iris species (setosa, versicolor, virginica)")
                        elif 'wine' in dataset_name.lower():
                            st.write("  - Wine quality classes")
                        elif 'breast cancer' in dataset_name.lower():
                            st.write("  - Cancer diagnosis (malignant/benign)")
                        elif 'credit' in dataset_name.lower():
                            st.write("  - Credit approval (approved/rejected)")
                        elif 'titanic' in dataset_name.lower():
                            st.write("  - Passenger survival (survived/died)")
                        elif 'heart' in dataset_name.lower():
                            st.write("  - Heart disease risk (disease/no_disease)")
                        elif 'flower' in dataset_name.lower():
                            st.write("  - Flower species (rose, tulip, daisy, sunflower, lily)")
                        elif 'music' in dataset_name.lower():
                            st.write("  - Music genres (rock, jazz, classical, electronic, country)")
                        elif 'digits' in dataset_name.lower():
                            st.write("  - Handwritten digits (0-9)")
                        elif 'adult' in dataset_name.lower():
                            st.write("  - Income level (<=50K/>50K)")
                        elif 'marketing' in dataset_name.lower():
                            st.write("  - Campaign response (response/no_response)")
                        elif 'bank' in dataset_name.lower():
                            st.write("  - Marketing campaign outcome (success/failure)")
                        elif 'covertype' in dataset_name.lower():
                            st.write("  - Forest cover type (7 different forest types)")
                        elif 'fashion' in dataset_name.lower():
                            st.write("  - Fashion item categories (10 clothing types)")
                    
                    elif task_type == 'regression':
                        st.write("• **Regression Task**: Predicting continuous numerical values")
                        if 'housing' in dataset_name.lower() or 'boston' in dataset_name.lower():
                            st.write("  - Housing prices (in thousands of dollars)")
                        elif 'diabetes' in dataset_name.lower():
                            st.write("  - Disease progression score")
                        elif 'mpg' in dataset_name.lower():
                            st.write("  - Car fuel efficiency (miles per gallon)")
                        elif 'stock' in dataset_name.lower():
                            st.write("  - Stock prices (in dollars)")
                        elif 'california' in dataset_name.lower():
                            st.write("  - Median house value (in hundreds of thousands)")
                        elif 'ames' in dataset_name.lower():
                            st.write("  - House sale prices")
            
            # General column-based recommendations for any dataset
            else:
                # Classification indicators
                classification_indicators = []
                regression_indicators = []
                avoid_indicators = []
                
                for col in available_columns:
                    col_lower = col.lower()
                    
                    # Classification targets
                    if any(keyword in col_lower for keyword in [
                        'class', 'category', 'type', 'status', 'label', 'outcome', 'result',
                        'approved', 'survived', 'response', 'success', 'failure', 'disease',
                        'diagnosis', 'species', 'genre', 'rating', 'level', 'grade'
                    ]):
                        classification_indicators.append(col)
                    
                    # Regression targets
                    elif any(keyword in col_lower for keyword in [
                        'price', 'cost', 'value', 'amount', 'salary', 'income', 'revenue',
                        'sales', 'score', 'rate', 'percentage', 'age', 'weight', 'height',
                        'mpg', 'efficiency', 'progression', 'index'
                    ]):
                        regression_indicators.append(col)
                    
                    # Columns to avoid
                    elif any(keyword in col_lower for keyword in [
                        'id', 'ssn', 'social', 'number', 'code', 'key', 'index', 'row',
                        'name', 'date', 'time', 'timestamp', 'url', 'email', 'phone'
                    ]):
                        avoid_indicators.append(col)
                
                # Show classification recommendations
                if classification_indicators:
                    st.write("**🎯 Good for Classification:**")
                    for col in classification_indicators[:5]:  # Show top 5
                        col_info = f"• `{col}`"
                        if 'class' in col.lower():
                            col_info += " - Predicting categories/classes"
                        elif 'status' in col.lower():
                            col_info += " - Predicting status (active/inactive, approved/rejected)"
                        elif 'type' in col.lower():
                            col_info += " - Predicting types/categories"
                        elif 'outcome' in col.lower():
                            col_info += " - Predicting outcomes (success/failure)"
                        elif 'rating' in col.lower():
                            col_info += " - Predicting quality ratings"
                        st.write(col_info)
                
                # Show regression recommendations
                if regression_indicators:
                    st.write("**📈 Good for Regression:**")
                    for col in regression_indicators[:5]:  # Show top 5
                        col_info = f"• `{col}`"
                        if 'price' in col.lower() or 'cost' in col.lower():
                            col_info += " - Predicting monetary values"
                        elif 'salary' in col.lower() or 'income' in col.lower():
                            col_info += " - Predicting earnings"
                        elif 'score' in col.lower():
                            col_info += " - Predicting numerical scores"
                        elif 'age' in col.lower():
                            col_info += " - Predicting age values"
                        elif 'rate' in col.lower():
                            col_info += " - Predicting rates/percentages"
                        st.write(col_info)
                
                # Show columns to avoid
                if avoid_indicators:
                    st.warning("**⚠️ Avoid These Columns (Identifiers/Metadata):**")
                    for col in avoid_indicators[:3]:  # Show top 3 to avoid
                        col_info = f"• `{col}`"
                        if 'id' in col.lower():
                            col_info += " - Just an identifier, not predictive"
                        elif 'name' in col.lower():
                            col_info += " - Names are typically not predictive targets"
                        elif 'date' in col.lower() or 'time' in col.lower():
                            col_info += " - Timestamps are usually features, not targets"
                        st.write(col_info)
                
                # General guidance if no specific indicators found
                if not classification_indicators and not regression_indicators:
                    st.info("💡 **General Guidance:**")
                    st.write("• **For Classification**: Choose columns with categories (Yes/No, High/Medium/Low, etc.)")
                    st.write("• **For Regression**: Choose columns with continuous numbers (prices, scores, measurements)")
                    st.write("• **Avoid**: ID columns, names, dates (unless predicting time-based values)")
            
            # Add privacy-preserving ML recommendations if anonymized datasets are available
            if st.session_state.datasets_collection.get('anonymized_datasets'):
                st.markdown("**🔒 Privacy-Preserving ML Recommendations:**")
                st.info("**PPML Detected**: You have anonymized datasets loaded!")
                
                anon_methods = list(st.session_state.datasets_collection['anonymized_datasets'].keys())
                st.write(f"• **Available Privacy Methods**: {', '.join(anon_methods)}")
                st.write("• **Recommended Analysis**: Compare model performance across original and anonymized data")
                st.write("• **Key Metrics**: Utility retention, privacy-utility trade-off, performance degradation")
                
                # Method-specific recommendations
                for method in anon_methods:
                    if method == 'k-anonymity':
                        st.write("  - 📊 **k-anonymity**: Good for basic privacy, minimal utility loss")
                    elif method == 'l-diversity':
                        st.write("  - 🌈 **l-diversity**: Better privacy than k-anonymity, moderate utility loss")
                    elif method == 't-closeness':
                        st.write("  - 🎯 **t-closeness**: Strong privacy guarantees, higher utility loss")
                    elif method == 'differential-privacy':
                        st.write("  - 🔐 **Differential Privacy**: Mathematical privacy guarantees, variable utility impact")
                    else:
                        st.write(f"  - 🔧 **{method}**: Custom privacy method")
            
            # Task-specific feature recommendations
            st.markdown("**🧠 ML Task Strategy Recommendations:**")
            
            task_type = st.session_state.get('task_type', 'classification')
            if task_type == 'classification':
                st.write("**🎯 Classification Best Practices:**")
                st.write("• Ensure target has multiple classes (ideally 2-10 classes)")
                st.write("• Check class balance - avoid severely imbalanced targets")
                st.write("• Consider metrics: Accuracy, Precision, Recall, F1-Score")
                st.write("• Good algorithms: Random Forest, SVM, Logistic Regression")
            else:
                st.write("**📈 Regression Best Practices:**")
                st.write("• Target should be continuous numerical values")
                st.write("• Check for outliers that might skew predictions")
                st.write("• Consider metrics: MSE, MAE, R², RMSE")
                st.write("• Good algorithms: Linear Regression, Random Forest, Gradient Boosting")
            
            st.warning("⚠️ **Always Avoid**: ID columns (employee_id, customer_id), SSNs, names, or pure identifiers as targets!")
        # Target column selection with validation
        if available_columns:
            try:
                current_index = available_columns.index(st.session_state.target_column) if st.session_state.target_column in available_columns else 0
            except (ValueError, TypeError):
                current_index = 0
                st.session_state.target_column = available_columns[0]
            
            st.session_state.target_column = st.selectbox(
                "Select your target column:",
                options=available_columns,
                index=current_index,
                help="Choose the column you want to predict (avoid ID columns!)"
            )
            
            # ADD WARNING FOR BAD TARGET CHOICES:
            if st.session_state.target_column:
                target_col_lower = st.session_state.target_column.lower()
                if any(keyword in target_col_lower for keyword in ['id', 'ssn', 'name', 'date']):
                    st.warning(f"⚠️ **'{st.session_state.target_column}'** might not be a good target column. Consider choosing a column that represents what you want to predict (like performance_rating or salary).")            
            # Show target column info
            if st.session_state.target_column:
                target_series = st.session_state.df_uploaded[st.session_state.target_column]
                unique_values = target_series.nunique()
                
                # Enhanced target column information display with professional styling
                st.markdown("**🎯 Target Column Analysis**")
                
                # Professional target analysis table with improved styling
                target_analysis_data = []
                
                # Determine data quality indicator
                total_samples = len(target_series)
                if unique_values == total_samples:
                    quality_indicator = "⚠️ All Unique Values"
                    quality_description = "Every sample has different target - Check if this is correct"
                elif unique_values <= 10:
                    quality_indicator = "✅ Excellent for ML"
                    quality_description = "Good number of classes for classification/low cardinality"
                elif unique_values <= 50:
                    quality_indicator = "🟡 Many Classes"
                    quality_description = "High cardinality - Consider grouping similar values"
                else:
                    quality_indicator = "🔴 Too Many Classes"
                    quality_description = "Very high cardinality - May need preprocessing"
                
                # Determine task type recommendation
                auto_detected_task = determine_task_type(target_series)
                task_recommendation = f"📊 {auto_detected_task.title()}"
                
                # Calculate completeness
                missing_count = target_series.isnull().sum()
                completeness_pct = ((total_samples - missing_count) / total_samples) * 100
                
                target_analysis_data.append({
                    "🎯 Aspect": "Selected Target Column",
                    "📋 Value": st.session_state.target_column,
                    "📊 Details": f"Column selected for ML prediction task",
                    "✅ Status": "🟢 Ready"
                })
                
                target_analysis_data.append({
                    "🎯 Aspect": "Unique Values Count",
                    "📋 Value": f"{unique_values:,}",
                    "📊 Details": f"Distinct values in {total_samples:,} total samples",
                    "✅ Status": "🟢 Analyzed"
                })
                
                target_analysis_data.append({
                    "🎯 Aspect": "Data Quality Assessment",
                    "📋 Value": quality_indicator,
                    "📊 Details": quality_description,
                    "✅ Status": "🟢 Evaluated"
                })
                
                target_analysis_data.append({
                    "🎯 Aspect": "Task Type Recommendation",
                    "📋 Value": task_recommendation,
                    "📊 Details": f"Auto-detected based on data characteristics",
                    "✅ Status": "🟢 Detected"
                })
                
                target_analysis_data.append({
                    "🎯 Aspect": "Data Completeness",
                    "📋 Value": f"{completeness_pct:.1f}%",
                    "📊 Details": f"{total_samples - missing_count:,} valid samples, {missing_count:,} missing",
                    "✅ Status": "🟢 Complete" if completeness_pct == 100 else "🟡 Has Missing" if completeness_pct >= 95 else "🔴 Many Missing"
                })
                
                # Create and style the target analysis table
                target_analysis_df = pd.DataFrame(target_analysis_data)
                
                # Professional styling for target analysis table
                styled_target_analysis = target_analysis_df.style.set_properties(**{
                    'padding': '12px',
                    'font-size': '14px',
                    'border': '1px solid var(--text-color-secondary)',
                    'font-weight': '500'
                }).set_table_styles([
                    {'selector': 'th', 'props': [
                        ('background-color', '#fd7e14'),  # Orange header for target analysis
                        ('color', 'white'),
                        ('font-weight', 'bold'),
                        ('text-align', 'center'),
                        ('padding', '15px'),
                        ('border', '1px solid #fd7e14'),
                        ('font-size', '15px')
                    ]},
                    {'selector': 'td:first-child', 'props': [
                        ('background-color', 'rgba(253, 126, 20, 0.1)'),
                        ('font-weight', 'bold'),
                        ('text-align', 'left'),
                        ('color', '#fd7e14')
                    ]},
                    {'selector': 'td:nth-child(2)', 'props': [
                        ('text-align', 'center'),
                        ('font-weight', 'bold'),
                        ('font-size', '15px')
                    ]},
                    {'selector': 'td:nth-child(3)', 'props': [
                        ('text-align', 'left'),
                        ('font-style', 'italic'),
                        ('color', 'var(--text-color-secondary)')
                    ]},
                    {'selector': 'td:last-child', 'props': [
                        ('text-align', 'center'),
                        ('font-weight', 'bold')
                    ]},
                    {'selector': 'td', 'props': [
                        ('vertical-align', 'middle'),
                        ('border', '1px solid var(--text-color-secondary)')
                    ]},
                    {'selector': '', 'props': [
                        ('border-collapse', 'collapse'),
                        ('margin', '20px 0'),
                        ('border-radius', '8px'),
                        ('overflow', 'hidden'),
                        ('box-shadow', '0 4px 6px rgba(0, 0, 0, 0.1)')
                    ]}
                ])
                
                st.dataframe(styled_target_analysis, use_container_width=True, hide_index=True)
                
                # Professional value distribution table
                with st.expander("📊 Target Value Distribution", expanded=False):
                    sample_values = target_series.value_counts().head(10)
                    
                    if len(sample_values) > 0:
                        # Create professional distribution table
                        distribution_data = []
                        total_count = len(target_series)
                        
                        for value, count in sample_values.items():
                            percentage = (count / total_count) * 100
                            distribution_data.append({
                                "📋 Value": str(value),
                                "📊 Count": f"{count:,}",
                                "📈 Percentage": f"{percentage:.2f}%",
                                "🔍 Type": type(value).__name__
                            })
                        
                        distribution_df = pd.DataFrame(distribution_data)
                        
                        # Apply professional styling
                        styled_distribution = distribution_df.style.set_properties(**{
                            'padding': '8px',
                            'font-size': '13px',
                            'text-align': 'center',
                            'border': '1px solid var(--text-color-secondary)',
                            'font-weight': '500'
                        }).set_table_styles([
                            {'selector': 'th', 'props': [
                                ('background-color', '#007bff'),
                                ('color', 'white'),
                                ('font-weight', 'bold'),
                                ('text-align', 'center'),
                                ('padding', '10px'),
                                ('border', '1px solid #007bff'),
                                ('font-size', '14px')
                            ]},
                            {'selector': 'td', 'props': [
                                ('text-align', 'center'),
                                ('vertical-align', 'middle'),
                                ('border', '1px solid var(--text-color-secondary)')
                            ]},
                            {'selector': '', 'props': [
                                ('border-collapse', 'collapse'),
                                ('margin', '10px 0'),
                                ('border-radius', '6px'),
                                ('overflow', 'hidden'),
                                ('box-shadow', '0 2px 4px rgba(0, 0, 0, 0.1)')
                            ]}
                        ])
                        
                        st.dataframe(styled_distribution, use_container_width=True, hide_index=True)
                        
                        # Additional insights
                        st.markdown("**💡 Distribution Insights:**")
                        
                        insight_col1, insight_col2 = st.columns(2)
                        
                        with insight_col1:
                            # Class balance analysis
                            if len(sample_values) > 1:
                                max_count = sample_values.max()
                                min_count = sample_values.min()
                                balance_ratio = min_count / max_count
                                
                                if balance_ratio >= 0.8:
                                    balance_status = "🟢 Well Balanced"
                                elif balance_ratio >= 0.5:
                                    balance_status = "🟡 Moderately Balanced"
                                elif balance_ratio >= 0.2:
                                    balance_status = "🟠 Imbalanced"
                                else:
                                    balance_status = "🔴 Severely Imbalanced"
                                
                                st.write(f"**Class Balance:** {balance_status}")
                                st.write(f"**Balance Ratio:** {balance_ratio:.3f}")
                            else:
                                st.write("**Class Balance:** Single class detected")
                        
                        with insight_col2:
                            # Completeness analysis
                            missing_count = target_series.isnull().sum()
                            completeness = ((total_count - missing_count) / total_count) * 100
                            
                            if completeness == 100:
                                completeness_status = "🟢 Complete"
                            elif completeness >= 95:
                                completeness_status = "🟡 Mostly Complete"
                            elif completeness >= 80:
                                completeness_status = "🟠 Some Missing"
                            else:
                                completeness_status = "🔴 Many Missing"
                            
                            st.write(f"**Data Completeness:** {completeness_status}")
                            st.write(f"**Valid Samples:** {total_count - missing_count:,} / {total_count:,}")
                        
                        # Show additional values if truncated
                        if len(target_series.value_counts()) > 10:
                            remaining_count = len(target_series.value_counts()) - 10
                            st.info(f"📋 Showing top 10 values. {remaining_count} additional values not displayed.")
                    
                    else:
                        st.warning("⚠️ No valid target values found")

            # Auto-determine task type but allow manual override
            if st.session_state.target_column:
                auto_task_type = determine_task_type(st.session_state.df_uploaded[st.session_state.target_column])
                
                task_index = 0 if auto_task_type == "classification" else 1
                st.session_state.task_type = st.selectbox(
                    "Task Type:",
                    options=["classification", "regression"],
                    index=task_index,
                    help=f"Auto-detected: {auto_task_type}"
                )
                
                # Show task type validation
                if st.session_state.task_type != auto_task_type:
                    st.warning(f"⚠️ You selected '{st.session_state.task_type}' but auto-detection suggests '{auto_task_type}'")
        else:
            st.error("❌ No columns found in the dataset")
    else:
        st.info("👆 Upload a dataset to configure the task")
    # --- Algorithm Selection ---
    st.header("🤖 Algorithm Selection")
    
    if PLUGINS_AVAILABLE:
        # Get available plugins for the selected task type
        available_plugins = plugin_manager.get_available_plugins(st.session_state.task_type)
        
        # Add test plugin if available
        if (hasattr(st.session_state, 'test_plugin_available') and 
            st.session_state.test_plugin_available and 
            hasattr(st.session_state, 'test_plugin_instance')):
            
            test_plugin = st.session_state.test_plugin_instance
            test_plugin_name = st.session_state.test_plugin_name
            available_plugins[test_plugin_name] = test_plugin
        
        if not available_plugins:
            st.warning(f"No plugins available for {st.session_state.task_type} tasks.")
            
            # Add refresh button to reload plugins
            if st.button("🔄 Refresh Plugins", help="Reload plugins from disk"):
                try:
                    # Clear and reload plugins using correct methods
                    if hasattr(plugin_manager, '_loaded_plugins'):
                        plugin_manager._loaded_plugins = {}
                    if hasattr(plugin_manager, '_plugin_categories'):
                        plugin_manager._plugin_categories = {}
                    
                    # Use the correct reload method
                    plugin_manager._discover_and_load_plugins()
                    
                    st.success("All plugins reloaded!")
                    st.rerun()
                    
                except Exception as e:
                    st.error(f"Error reloading plugins: {e}")
                    st.write("**Debug info:**")
                    st.write(f"Plugin manager type: {type(plugin_manager)}")
                    st.write(f"Available methods: {[method for method in dir(plugin_manager) if not method.startswith('__')]}")
        else:
            # Add refresh option at the top
            col_refresh, col_info = st.columns([1, 3])
            with col_refresh:
                if st.button("🔄 Refresh", help="Reload plugins"):
                    try:
                        # Clear and reload plugins using correct methods
                        if hasattr(plugin_manager, '_loaded_plugins'):
                            plugin_manager._loaded_plugins = {}
                        if hasattr(plugin_manager, '_plugin_categories'):
                            plugin_manager._plugin_categories = {}
                        
                        # Use the correct reload method
                        plugin_manager._discover_and_load_plugins()
                        
                        st.success("Plugins refreshed!")
                        st.rerun()
                        
                    except Exception as e:
                        st.error(f"Error refreshing plugins: {e}")
                        st.info("Please restart the app to see new plugins.")
            
            with col_info:
                st.info(f"📊 {len(available_plugins)} algorithm(s) available for {st.session_state.task_type}")
            
            # Group plugins by category
            plugins_by_category = plugin_manager.get_plugins_by_category(st.session_state.task_type)
            
            # Add test plugin to appropriate category
            if (hasattr(st.session_state, 'test_plugin_available') and 
                st.session_state.test_plugin_available):
                
                test_plugin = st.session_state.test_plugin_instance
                test_plugin_name = st.session_state.test_plugin_name
                test_category = test_plugin.get_category()
                
                if test_category not in plugins_by_category:
                    plugins_by_category[test_category] = []
                plugins_by_category[test_category].append(test_plugin_name)
            
            # Display plugins organized by category
            selected_plugin_names = []
            for category, plugin_names in plugins_by_category.items():
                # Show category with plugin count
                category_display = f"📂 {category} Algorithms ({len(plugin_names)})"
                
                with st.expander(category_display, expanded=True):
                    for plugin_name in plugin_names:
                        plugin = available_plugins[plugin_name]
                        
                        # Special styling for test plugins
                        display_name = plugin_name
                        if plugin_name.startswith("[TEST]"):
                            display_name = f"🧪 {plugin_name}"
                        
                        # Check compatibility with current dataset
                        if st.session_state.df_uploaded is not None and st.session_state.target_column:
                            try:
                                is_compatible, reason = plugin.is_compatible_with_data(
                                    st.session_state.df_uploaded, st.session_state.target_column
                                )
                                
                                disabled = not is_compatible
                                help_text = f"{plugin.get_description()}\n\n"
                                if not is_compatible:
                                    help_text += f"⚠️ Not compatible: {reason}"
                                elif plugin_name.startswith("[TEST]"):
                                    help_text += "\n🧪 This is a test plugin from the development studio"
                            except Exception as e:
                                # Fallback if compatibility check fails
                                disabled = False
                                help_text = f"{plugin.get_description()}\n\n⚠️ Compatibility check failed: {str(e)}"
                        else:
                            disabled = False
                            help_text = plugin.get_description()
                            if plugin_name.startswith("[TEST]"):
                                help_text += "\n🧪 This is a test plugin from the development studio"
                        
                        is_selected = st.checkbox(
                            display_name,
                            key=f"select_{plugin_name}",
                            disabled=disabled,
                            help=help_text,
                            value=plugin_name in st.session_state.selected_plugins_config
                        )
                        
                        if is_selected:
                            selected_plugin_names.append(plugin_name)

            # Update selected plugins configuration
            current_config = {}
            for plugin_name in selected_plugin_names:
                current_config[plugin_name] = st.session_state.selected_plugins_config.get(plugin_name, {})
            st.session_state.selected_plugins_config = current_config
            
            # Show selection summary
            if selected_plugin_names:
                st.success(f"✅ {len(selected_plugin_names)} algorithm(s) selected")
            else:
                st.info("👆 Select algorithms above to configure and train")
                
    # Add this section after the Algorithm Selection section:

    # Plugin Management Section (for debugging)
    if st.session_state.ml_developer_mode:
        with st.expander("🔧 Plugin Management & Debugging", expanded=False):
            st.markdown("**Plugin Discovery Status:**")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if st.button("🔄 Force Reload All Plugins"):
                    try:
                        # Clear all caches using correct attribute names
                        if hasattr(plugin_manager, '_loaded_plugins'):
                            plugin_manager._loaded_plugins = {}
                        if hasattr(plugin_manager, '_plugin_categories'):
                            plugin_manager._plugin_categories = {}
                        
                        # Use the correct reload method
                        plugin_manager._discover_and_load_plugins()
                        
                        st.success("All plugins reloaded!")
                        st.rerun()
                        
                    except Exception as e:
                        st.error(f"Error reloading plugins: {e}")
                        st.write("**Debug info:**")
                        st.write(f"Plugin manager type: {type(plugin_manager)}")
                        st.write(f"Available methods: {[method for method in dir(plugin_manager) if not method.startswith('__')]}")
            
            with col2:
                if st.button("📋 Show Plugin Paths"):
                    algorithms_dir = os.path.join(PROJECT_ROOT, "src", "ml_plugins", "algorithms")
                    if os.path.exists(algorithms_dir):
                        files = [f for f in os.listdir(algorithms_dir) if f.endswith('.py') and f != '__init__.py']
                        st.write("**Found algorithm files:**")
                        for file in files:
                            st.write(f"- {file}")
                    else:
                        st.error("Algorithms directory not found")
            
            with col3:
                if st.button("🧪 Test Plugin Discovery"):
                    try:
                        # Test manual discovery
                        algorithms_dir = os.path.join(PROJECT_ROOT, "src", "ml_plugins", "algorithms")
                        found_plugins = []
                        
                        if os.path.exists(algorithms_dir):
                            for file in os.listdir(algorithms_dir):
                                if file.endswith('.py') and file != '__init__.py':
                                    found_plugins.append(file)
                        
                        st.write(f"**Discovered {len(found_plugins)} plugin files:**")
                        for plugin in found_plugins:
                            st.write(f"✓ {plugin}")
                            
                    except Exception as e:
                        st.error(f"Discovery test failed: {e}")
            
            # Show current plugin state
            if hasattr(plugin_manager, '_plugins'):
                current_plugins = list(plugin_manager._plugins.keys())
                st.write(f"**Currently loaded plugins ({len(current_plugins)}):**")
                for plugin_name in current_plugins:
                    st.write(f"• {plugin_name}")
            else:
                st.write("**No plugin cache found**")
                

with st.sidebar:
    # --- Metrics Selection ---
    st.header("📈 Metrics Selection")
    
    if PLUGINS_AVAILABLE:
        available_metrics = metric_manager.get_available_metrics(st.session_state.task_type)
        
        # Add test metric to available metrics if it exists
        if (hasattr(st.session_state, 'test_plugin_available') and 
            st.session_state.test_plugin_available and
            st.session_state.ml_plugin_type == "Metric" and
            hasattr(st.session_state, 'test_plugin_instance')):
            
            test_metric = st.session_state.test_plugin_instance
            test_metric_name = st.session_state.test_plugin_name
            available_metrics[test_metric_name] = test_metric
        
        if not available_metrics:
            st.warning(f"No metrics available for {st.session_state.task_type} tasks.")
            
            # Add refresh button to reload metrics
            if st.button("🔄 Refresh Metrics", help="Reload metrics from disk", key="metrics_refresh_btn"):
                # Store ALL plugin development state before refresh
                plugin_dev_state = {
                    'ml_plugin_type': st.session_state.get("ml_plugin_type", "ML Algorithm"),
                    'ml_plugin_category': st.session_state.get("ml_plugin_category", "Custom"),
                    'ml_plugin_raw_code': st.session_state.get("ml_plugin_raw_code", ""),
                    'ml_plugin_class_name': st.session_state.get("ml_plugin_class_name", ""),
                    'ml_plugin_display_name': st.session_state.get("ml_plugin_display_name", ""),
                    'ml_developer_mode': st.session_state.get("ml_developer_mode", False),
                    'ml_plugin_validation_results': st.session_state.get("ml_plugin_validation_results", []),
                    'ml_plugin_test_instance': st.session_state.get("ml_plugin_test_instance", None),
                    'ml_plugin_error': st.session_state.get("ml_plugin_error", None),
                    'ml_plugin_save_status': st.session_state.get("ml_plugin_save_status", ""),
                    'show_ml_plugin_snippets': st.session_state.get("show_ml_plugin_snippets", False),
                    'test_plugin_available': st.session_state.get("test_plugin_available", False),
                    'test_plugin_name': st.session_state.get("test_plugin_name", ""),
                    'test_plugin_instance': st.session_state.get("test_plugin_instance", None)
                }
                
                try:
                    # Clear and reload metrics using correct methods
                    if hasattr(metric_manager, '_loaded_metrics'):
                        metric_manager._loaded_metrics = {}
                    if hasattr(metric_manager, '_metric_categories'):
                        metric_manager._metric_categories = {}
                    
                    # Use the correct reload method for metrics
                    metric_manager._discover_and_load_metrics()
                    
                    # Restore ALL plugin development state
                    for key, value in plugin_dev_state.items():
                        st.session_state[key] = value
                    
                    st.success("All metrics reloaded!")
                    st.rerun()
                    
                except Exception as e:
                    st.error(f"Error reloading metrics: {e}")
                    # Restore state even on error
                    for key, value in plugin_dev_state.items():
                        st.session_state[key] = value
        else:
            # Add refresh option at the top
            col_refresh, col_info = st.columns([1, 3])
            with col_refresh:
                if st.button("🔄 Refresh", help="Reload metrics", key="metrics_refresh_main_btn"):
                    # Store ALL plugin development state before refresh
                    plugin_dev_state = {
                        'ml_plugin_type': st.session_state.get("ml_plugin_type", "ML Algorithm"),
                        'ml_plugin_category': st.session_state.get("ml_plugin_category", "Custom"),
                        'ml_plugin_raw_code': st.session_state.get("ml_plugin_raw_code", ""),
                        'ml_plugin_class_name': st.session_state.get("ml_plugin_class_name", ""),
                        'ml_plugin_display_name': st.session_state.get("ml_plugin_display_name", ""),
                        'ml_developer_mode': st.session_state.get("ml_developer_mode", False),
                        'ml_plugin_validation_results': st.session_state.get("ml_plugin_validation_results", []),
                        'ml_plugin_test_instance': st.session_state.get("ml_plugin_test_instance", None),
                        'ml_plugin_error': st.session_state.get("ml_plugin_error", None),
                        'ml_plugin_save_status': st.session_state.get("ml_plugin_save_status", ""),
                        'show_ml_plugin_snippets': st.session_state.get("show_ml_plugin_snippets", False),
                        'test_plugin_available': st.session_state.get("test_plugin_available", False),
                        'test_plugin_name': st.session_state.get("test_plugin_name", ""),
                        'test_plugin_instance': st.session_state.get("test_plugin_instance", None)
                    }
                    
                    try:
                        # Clear and reload metrics using correct methods
                        if hasattr(metric_manager, '_loaded_metrics'):
                            metric_manager._loaded_metrics = {}
                        if hasattr(metric_manager, '_metric_categories'):
                            metric_manager._metric_categories = {}
                        
                        # Use the correct reload method for metrics
                        metric_manager._discover_and_load_metrics()
                        
                        # Restore ALL plugin development state
                        for key, value in plugin_dev_state.items():
                            st.session_state[key] = value
                        
                        # Add explicit state preservation for UI checkboxes
                        st.session_state['ml_developer_mode_checkbox'] = plugin_dev_state['ml_developer_mode']
                        
                        st.success("Metrics refreshed!")
                        
                    except Exception as e:
                        st.error(f"Error refreshing metrics: {e}")
                        # Restore state even on error
                        for key, value in plugin_dev_state.items():
                            st.session_state[key] = value
                        st.info("Please restart the app to see new metrics.")
            
            with col_info:
                # Count metrics including test metric
                metric_count = len(available_metrics)
                st.info(f"📊 {metric_count} metric(s) available for {st.session_state.task_type}")
            
            metrics_by_category = metric_manager.get_metrics_by_category(st.session_state.task_type)
            
            # Add test metric to appropriate category if available
            if (hasattr(st.session_state, 'test_plugin_available') and 
                st.session_state.test_plugin_available and
                st.session_state.ml_plugin_type == "Metric" and
                hasattr(st.session_state, 'test_plugin_instance')):
                
                test_metric = st.session_state.test_plugin_instance
                test_metric_name = st.session_state.test_plugin_name
                test_category = test_metric.get_category() if hasattr(test_metric, 'get_category') else "Custom"
                
                if test_category not in metrics_by_category:
                    metrics_by_category[test_category] = []
                if test_metric_name not in metrics_by_category[test_category]:
                    metrics_by_category[test_category].append(test_metric_name)
            
            st.markdown("**Select evaluation metrics:**")
            selected_metric_names = []
            
            for category, metric_names in metrics_by_category.items():
                # Show category with metric count
                category_display = f"📊 {category} Metrics ({len(metric_names)})"
                
                with st.expander(category_display, expanded=True):
                    for metric_name in metric_names:
                        metric = available_metrics[metric_name]
                        
                        # Special styling for test metrics
                        display_name = metric_name
                        if metric_name.startswith("[TEST]"):
                            display_name = f"🧪 {metric_name}"
                        
                        help_text = f"{metric.get_description()}\n\n"
                        help_text += f"Range: {metric.get_value_range()}\n"
                        help_text += f"Higher is better: {metric.is_higher_better()}"
                        
                        if metric.requires_probabilities():
                            help_text += "\n⚠️ Requires prediction probabilities"
                        
                        if metric_name.startswith("[TEST]"):
                            help_text += "\n🧪 This is a test metric from the development studio"
                            help_text += "\n💡 Use the 'Remove' button in the development zone to delete this test metric"
                        
                        if st.checkbox(
                            display_name,
                            key=f"metric_{metric_name}",
                            help=help_text,
                            value=metric_name in st.session_state.selected_metrics
                        ):
                            selected_metric_names.append(metric_name)
            
            st.session_state.selected_metrics = selected_metric_names
            
            # Show selection summary
            if selected_metric_names:
                st.success(f"✅ {len(selected_metric_names)} metric(s) selected")
            else:
                st.info("👆 Select metrics above to evaluate model performance")
    else:
        st.info("Plugin system not available.")
                
    # --- Hyperparameter Configuration ---
    st.header("🔧Hyperparameter Configuration")
    
    if st.session_state.selected_plugins_config:
        for plugin_name in st.session_state.selected_plugins_config:
            plugin = available_plugins[plugin_name]
            
            with st.expander(f"⚙️ Configure {plugin_name}", expanded=True):
                st.markdown(f"**{plugin.get_description()}**")
                
                # Generate unique key prefix for this plugin's UI elements
                unique_key_prefix = f"{plugin_name.lower().replace(' ', '_').replace('-', '_')}"
                
                # Get hyperparameter configuration from the plugin
                hyperparams = plugin.get_hyperparameter_config(unique_key_prefix)
                st.session_state.selected_plugins_config[plugin_name] = hyperparams
    else:
        st.info("Select algorithms above to configure their hyperparameters.")

    # --- Training ---
    st.header("🚀 Training")
    
    col_train, col_clear = st.columns([3, 1])
    with col_train:
        if st.button(
            "🚀 Train Selected Models", 
            type="primary", 
            use_container_width=True, 
            disabled=not st.session_state.selected_plugins_config
        ):
            st.session_state.execute_training_flag = True
    
    with col_clear:
        if st.session_state.experiment_results:
            if st.button("🗑️ Clear Results", type="secondary", use_container_width=True):
                st.session_state.experiment_results = []
                st.rerun()
    
        # Add this after the training button section in the sidebar
    
    # Training Status Panel
    if st.session_state.experiment_results or st.session_state.trained_combinations:
        st.markdown("---")
        st.header("📈 Training Status")
        
        # Current results summary
        total_results = len(st.session_state.experiment_results)
        total_combinations = len(st.session_state.trained_combinations)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("🏆 Results", total_results, "trained models")
        
        with col2:
            st.metric("⚙️ Combinations", total_combinations, "tracked")
        
        # Advanced management
        with st.expander("🔧 Advanced Training Management", expanded=False):
            
            # Clear options
            clear_col1, clear_col2 = st.columns(2)
            
            with clear_col1:
                if st.button("🗑️ Clear Results", use_container_width=True, help="Remove all training results"):
                    st.session_state.experiment_results = []
                    st.rerun()
            
            with clear_col2:
                if st.button("🔄 Reset Training History", use_container_width=True, help="Clear training combinations tracking"):
                    st.session_state.trained_combinations.clear()
                    st.success("✅ Training history reset! Next training will process all combinations.")
                    st.rerun()
            
            # Show some trained combinations
            if st.session_state.trained_combinations:
                st.write(f"**Recent Combinations** (showing {min(5, len(st.session_state.trained_combinations))} of {len(st.session_state.trained_combinations)}):")
                for i, combo in enumerate(list(st.session_state.trained_combinations)[:5]):
                    # Parse combination ID
                    parts = combo.split('_')
                    if len(parts) >= 2:
                        dataset_part = '_'.join(parts[:-2]) if len(parts) > 2 else parts[0]
                        algorithm_part = parts[-2] if len(parts) > 2 else parts[1]
                        st.write(f"  {i+1}. 📊 {dataset_part} + 🤖 {algorithm_part}")
                    else:
                        st.write(f"  {i+1}. {combo}")

    st.sidebar.markdown("---")
    # --- Visualization Selection ---
    st.header("📊 Visualization Selection")
    
    if PLUGINS_AVAILABLE and st.session_state.selected_plugins_config:
        st.markdown("**Select visualizations to display for trained models:**")
        
        # Get available visualizations from selected plugins
        available_visualizations = {}
        for plugin_name in st.session_state.selected_plugins_config:
            plugin = available_plugins.get(plugin_name)
            if plugin and hasattr(plugin, 'get_available_visualizations'):
                plugin_viz = plugin.get_available_visualizations()
                for viz_key, viz_name in plugin_viz.items():
                    viz_full_key = f"{plugin_name}_{viz_key}"
                    available_visualizations[viz_full_key] = f"{plugin_name}: {viz_name}"
        
        # Initialize selected visualizations in session state
        if 'selected_visualizations' not in st.session_state:
            st.session_state.selected_visualizations = []
        
        # Create checkboxes for each available visualization
        selected_viz_names = []
        if available_visualizations:
            for viz_key, viz_display_name in available_visualizations.items():
                is_selected = st.checkbox(
                    viz_display_name,
                    key=f"viz_{viz_key}",
                    value=viz_key in st.session_state.selected_visualizations,
                    help=f"Include {viz_display_name.split(': ')[1]} in results"
                )
                
                if is_selected:
                    selected_viz_names.append(viz_key)
            
            st.session_state.selected_visualizations = selected_viz_names
            
            # Show selection summary
            if selected_viz_names:
                st.success(f"✅ {len(selected_viz_names)} visualization(s) selected")
            else:
                st.info("👆 Select visualizations above to include in results")
        else:
            st.info("Select algorithms first to see available visualizations")
    else:
        st.info("Select algorithms to see available visualizations")
        
    # Add this section to the sidebar in ml_app.py (after the existing sections)
    st.sidebar.markdown("---")
    st.sidebar.header("🛠️ Plugin Development")

    # Developer mode toggle
    st.sidebar.checkbox(
        "Enable Plugin Developer Mode", 
        key="ml_developer_mode_checkbox",
        value=st.session_state.ml_developer_mode,
        on_change=lambda: setattr(st.session_state, 'ml_developer_mode', st.session_state.ml_developer_mode_checkbox)
    )

    if st.sidebar.button("View Plugin Code Snippets", key="toggle_ml_snippets"):
        st.session_state.show_ml_plugin_snippets = not st.session_state.show_ml_plugin_snippets


# --- Main Area for Results ---
# Enhanced Training Loop with Multi-Dataset Support
def apply_preprocessing(df, config):
    """Apply preprocessing based on configuration"""
    processed_df = df.copy()
    
    # Handle missing values
    if config.get('missing_strategy') == 'drop':
        processed_df = processed_df.dropna()
    elif config.get('missing_strategy') in ['mean', 'median', 'most_frequent']:
        from sklearn.impute import SimpleImputer
        imputer = SimpleImputer(strategy=config['missing_strategy'])
        
        numeric_cols = processed_df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            processed_df[numeric_cols] = imputer.fit_transform(processed_df[numeric_cols])
    
    # Categorical encoding
    if config.get('encoding') == 'OneHotEncoder':
        processed_df = pd.get_dummies(processed_df, drop_first=True)
    elif config.get('encoding') == 'LabelEncoder':
        from sklearn.preprocessing import LabelEncoder
        categorical_cols = processed_df.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            if col != st.session_state.target_column:  # Don't encode target
                le = LabelEncoder()
                processed_df[col] = le.fit_transform(processed_df[col].astype(str))
    
    # Scaling
    if config.get('scaling') != 'None':
        from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
        
        scaler_map = {
            'StandardScaler': StandardScaler(),
            'MinMaxScaler': MinMaxScaler(),
            'RobustScaler': RobustScaler()
        }
        
        scaler = scaler_map[config['scaling']]
        numeric_cols = processed_df.select_dtypes(include=[np.number]).columns
        target_col = st.session_state.target_column
        
        # Don't scale the target column
        feature_cols = [col for col in numeric_cols if col != target_col]
        if feature_cols:
            processed_df[feature_cols] = scaler.fit_transform(processed_df[feature_cols])
    
    return processed_df

def train_models_on_multiple_datasets(datasets_dict, target_column, selected_plugins, preprocessing_config, selected_metrics, incremental=True):
    """Train models on multiple datasets for PPML comparison with incremental training support"""
    
    all_results = {}
    
    # Flatten the datasets_dict to handle multiple datasets per method
    flattened_datasets = {}
    
    # Add original dataset
    if 'original' in datasets_dict:
        flattened_datasets['original'] = datasets_dict['original']
    
    # Flatten anonymized datasets (now supporting multiple per method)
    for method, dataset_entries in st.session_state.datasets_collection['anonymized_datasets'].items():
        for dataset_entry in dataset_entries:
            dataset_name = f"{method}_{dataset_entry['number']}"
            flattened_datasets[dataset_name] = dataset_entry['data']
    
    # INCREMENTAL TRAINING LOGIC: Filter out already trained combinations
    datasets_to_process = {}
    plugins_to_process = {}
    
    if incremental:
        st.info("🔄 **Incremental Training Mode**: Only processing new dataset+algorithm combinations")
        
        # Check which combinations need training
        for dataset_name, dataset_df in flattened_datasets.items():
            for plugin_name in selected_plugins.keys():
                combination_id = f"{dataset_name}_{plugin_name}_{hash(str(selected_plugins[plugin_name]))}"
                
                if combination_id not in st.session_state.trained_combinations:
                    # This combination hasn't been trained yet
                    if dataset_name not in datasets_to_process:
                        datasets_to_process[dataset_name] = dataset_df
                        plugins_to_process[dataset_name] = {}
                    plugins_to_process[dataset_name][plugin_name] = selected_plugins[plugin_name]
        
        # Show what will be processed
        total_new_combinations = sum(len(plugins) for plugins in plugins_to_process.values())
        
        if total_new_combinations == 0:
            st.success("✅ **All combinations already trained!** No new training needed.")
            st.info("💡 **Tip**: Add new datasets or algorithms to train additional combinations.")
            return {}
        else:
            st.info(f"🎯 **Processing {total_new_combinations} new combination(s)** across {len(datasets_to_process)} dataset(s)")
            
            # Show breakdown
            for dataset_name, plugins in plugins_to_process.items():
                if plugins:  # Only show if there are plugins to process
                    plugin_names = list(plugins.keys())
                    st.write(f"  📊 **{dataset_name}**: {len(plugins)} algorithm(s) - {', '.join(plugin_names)}")
    else:
        # Process all combinations (original behavior)
        st.warning("🔄 **Full Training Mode**: Processing all dataset+algorithm combinations")
        datasets_to_process = flattened_datasets
        plugins_to_process = {dataset_name: selected_plugins for dataset_name in flattened_datasets.keys()}
    
    # Process only the required combinations
    total_datasets_to_process = len(datasets_to_process)
    dataset_counter = 0

    for dataset_name, dataset_df in datasets_to_process.items():
        dataset_counter += 1
        st.info(f"🔄 Processing dataset: {dataset_name} ({dataset_counter}/{total_datasets_to_process})")
        
        current_dataset_results = []
        plugins_for_this_dataset = plugins_to_process.get(dataset_name, {})
        
        if not plugins_for_this_dataset:
            st.info(f"⏭️ Skipping {dataset_name} - no new algorithms to train")
            continue
        
        # ... [Keep all the existing preprocessing and data preparation code] ...
        st.write(f"⏳ [Dataset: {dataset_name}] Applying general preprocessing...")
        if preprocessing_config.get('enabled', False):
            processed_df = apply_preprocessing(dataset_df, preprocessing_config)
            st.write(f"✅ [Dataset: {dataset_name}] General preprocessing applied.")
        else:
            processed_df = dataset_df.copy()
            st.write(f"ℹ️ [Dataset: {dataset_name}] General preprocessing skipped.")
        
        st.write(f"⏳ [Dataset: {dataset_name}] Preparing features and target...")
        features = [col for col in processed_df.columns if col != target_column]
        X = processed_df[features]
        y = processed_df[target_column]
        st.write(f"✅ [Dataset: {dataset_name}] Features and target prepared.")

        st.write(f"⏳ [Dataset: {dataset_name}] Performing final data type conversions and imputation for X...")
        X_processed = pd.DataFrame(index=X.index)
        
        for col_idx, col in enumerate(X.columns):
            if X[col].dtype in ['object', 'category']:
                numeric_converted = pd.to_numeric(X[col], errors='coerce')
                if numeric_converted.notna().sum() / len(numeric_converted) > 0.5:
                    X_processed[col] = numeric_converted
                else:
                    from sklearn.preprocessing import LabelEncoder
                    le = LabelEncoder()
                    col_filled = X[col].fillna('missing_value_for_encoding')
                    X_processed[col] = le.fit_transform(col_filled.astype(str))
            else:
                X_processed[col] = X[col]
        
        X_final = X_processed.select_dtypes(include=[np.number])
        
        if X_final.isnull().sum().sum() > 0:
            st.write(f"  ➡️ [Dataset: {dataset_name}] Imputing missing values in X_final...")
            from sklearn.impute import SimpleImputer
            imputer = SimpleImputer(strategy='median')
            X_final = pd.DataFrame(
                imputer.fit_transform(X_final), 
                columns=X_final.columns, 
                index=X_final.index
            )
        st.write(f"✅ [Dataset: {dataset_name}] Final data type conversions and imputation for X complete.")
        
        y_final = y.loc[X_final.index]
        
        st.write(f"⏳ [Dataset: {dataset_name}] Splitting data into train/test sets...")
        try:
            if st.session_state.task_type == "classification":
                y_value_counts = y_final.value_counts()
                min_samples_for_stratify = 2
                
                if (y_value_counts < min_samples_for_stratify).any():
                    st.write(f"  ⚠️ [Dataset: {dataset_name}] Not enough samples in some classes for stratification. Using non-stratified split.")
                    X_train, X_test, y_train, y_test = train_test_split(
                        X_final, y_final, test_size=0.2, random_state=42
                    )
                else:
                    X_train, X_test, y_train, y_test = train_test_split(
                        X_final, y_final, test_size=0.2, random_state=42, stratify=y_final
                    )
            else: # Regression
                X_train, X_test, y_train, y_test = train_test_split(
                    X_final, y_final, test_size=0.2, random_state=42
                )
            st.write(f"✅ [Dataset: {dataset_name}] Data split complete. Train shape: {X_train.shape}, Test shape: {X_test.shape}")
        except Exception as e:
            st.error(f"❌ Error splitting {dataset_name}: {e}")
            st.write(f"Data y_final value counts:\n{y_final.value_counts()}")
            continue 
        
        total_plugins_for_dataset = len(plugins_for_this_dataset)
        plugin_counter = 0
        
        for plugin_name, hyperparams in plugins_for_this_dataset.items():
            plugin_counter += 1
            st.write(f"🔄 [Dataset: {dataset_name}] Training model {plugin_counter}/{total_plugins_for_dataset}: {plugin_name}")
            plugin = available_plugins[plugin_name]
            
            result = train_and_evaluate_model(
                plugin, hyperparams, X_train, y_train, X_test, y_test, 
                target_column, selected_metrics
            )
            
            if "error" not in result:
                result['dataset_name'] = dataset_name
                result['dataset_type'] = 'original' if dataset_name == 'original' else 'anonymized'
                
                # Enhanced metadata for multiple datasets per method
                if dataset_name != 'original':
                    method_name = dataset_name.rsplit('_', 1)[0]  # Extract method name
                    dataset_number = dataset_name.rsplit('_', 1)[1]  # Extract dataset number
                    result['anonymization_method'] = method_name
                    result['dataset_number'] = dataset_number
                    result['full_dataset_id'] = dataset_name
                
                original_shape = datasets_dict['original'].shape if 'original' in datasets_dict else processed_df.shape
                result['data_utility'] = {
                    'row_retention': processed_df.shape[0] / original_shape[0] if original_shape[0] > 0 else 1.0,
                    'column_retention': processed_df.shape[1] / original_shape[1] if original_shape[1] > 0 else 1.0,
                    'total_samples': processed_df.shape[0],
                    'total_features': processed_df.shape[1]
                }
                
                current_dataset_results.append(result)
                
                # MARK COMBINATION AS TRAINED
                combination_id = f"{dataset_name}_{plugin_name}_{hash(str(hyperparams))}"
                st.session_state.trained_combinations.add(combination_id)
                
                st.write(f"✅ [Dataset: {dataset_name}] Model {plugin_name} trained successfully.")
            else:
                st.write(f"❌ [Dataset: {dataset_name}] Model {plugin_name} failed: {result['error']}")
        
        all_results[dataset_name] = current_dataset_results
        st.success(f"🎉 Finished processing dataset: {dataset_name}")
    
    return all_results

# MAIN TRAINING EXECUTION
if st.session_state.execute_training_flag:
    # Check prerequisites for training
    can_train = True
    error_messages = []
    
    if st.session_state.df_uploaded is None:
        error_messages.append("Please upload a dataset.")
        can_train = False
    if not st.session_state.target_column:
        error_messages.append("Please select a target column.")
        can_train = False
    if not st.session_state.selected_plugins_config:
        error_messages.append("Please select at least one algorithm.")
        can_train = False

    if not can_train:
        for msg in error_messages:
            st.warning(msg)
    else:
        # Enhanced multi-dataset training for PPML benchmarking with incremental support
        
        # Training mode selection
        training_mode_col1, training_mode_col2 = st.columns([3, 1])
        
        with training_mode_col1:
            st.markdown("**🎯 Training Mode:**")
        
        with training_mode_col2:
            force_retrain = st.checkbox(
                "🔄 Force Retrain All",
                value=False,
                help="Check this to retrain all combinations, including already trained ones"
            )
        
        with st.spinner("🚀 Enhanced PPML Training Pipeline - Processing datasets..."):
            
            # Prepare datasets for training
            datasets_to_process = {}
            
            # Always include the original dataset
            if st.session_state.datasets_collection['original'] is not None:
                datasets_to_process['original'] = st.session_state.datasets_collection['original']
            
            # Include anonymized datasets if available
            for method, anon_df in st.session_state.datasets_collection['anonymized_datasets'].items():
                datasets_to_process[method] = anon_df
            
            # If no multi-dataset setup, fall back to single dataset
            if len(datasets_to_process) == 0:
                datasets_to_process['single_dataset'] = st.session_state.df_uploaded
            
            # Show current training status
            total_existing_results = len(st.session_state.experiment_results)
            if total_existing_results > 0 and not force_retrain:
                st.info(f"📊 **Current Results**: {total_existing_results} trained models in memory")
                st.info(f"🎯 **Incremental Mode**: Will only train new dataset+algorithm combinations")
            elif force_retrain:
                st.warning(f"🔄 **Force Retrain Mode**: Will retrain all combinations (existing {total_existing_results} results will be kept)")
                # Clear trained combinations to force retraining
                st.session_state.trained_combinations.clear()
            
            # Show preprocessing configuration if enabled
            if st.session_state.preprocessing_configs.get('enabled', False):
                with st.expander("⚙️ Preprocessing Configuration", expanded=False):
                    st.json(st.session_state.preprocessing_configs)
            
            try:
                # Use enhanced training function with incremental support
                all_training_results = train_models_on_multiple_datasets(
                    datasets_to_process,
                    st.session_state.target_column,
                    st.session_state.selected_plugins_config,
                    st.session_state.preprocessing_configs,
                    st.session_state.selected_metrics,
                    incremental=not force_retrain  # Use incremental mode unless force retrain is checked
                )
                
                # Add only NEW results to experiment results
                total_successful = 0
                total_failed = 0
                new_results_added = 0
                
                for dataset_name, dataset_results in all_training_results.items():
                    for result in dataset_results:
                        if "error" not in result:
                            st.session_state.experiment_results.append(result)
                            total_successful += 1
                            new_results_added += 1
                        else:
                            total_failed += 1
                
                # Enhanced completion message
                if total_successful > 0:
                    st.success(f"🎉 PPML Training Complete! {total_successful} new model(s) trained successfully")
                    st.info(f"📊 **Total Results Now**: {len(st.session_state.experiment_results)} trained models ({new_results_added} new)")
                    
                    # Show quick summary of new results
                    if len(datasets_to_process) > 1 and new_results_added > 0:
                        with st.expander("📊 New Training Results Summary", expanded=True):
                            summary_data = []
                            for dataset_name, dataset_results in all_training_results.items():
                                successful_models = [r for r in dataset_results if "error" not in r]
                                if successful_models:
                                    avg_accuracy = np.mean([r['accuracy'] for r in successful_models])
                                    avg_f1 = np.mean([r['f1_score'] for r in successful_models])
                                    
                                    summary_data.append({
                                        "📊 Dataset": dataset_name.title(),
                                        "🤖 New Models": len(successful_models),
                                        "🎯 Avg Accuracy": f"{avg_accuracy:.4f}",
                                        "🏆 Avg F1-Score": f"{avg_f1:.4f}",
                                        "📈 Status": "✅ Complete"
                                    })
                            
                            if summary_data:
                                summary_df = pd.DataFrame(summary_data)
                                st.dataframe(summary_df, use_container_width=True, hide_index=True)
                elif len(all_training_results) == 0:
                    st.info("✅ **No new training needed!** All selected combinations are already trained.")
                    st.info(f"📊 **Current Results**: {len(st.session_state.experiment_results)} trained models available for analysis")
                else:
                    st.error("❌ No models trained successfully")
                
                if total_failed > 0:
                    st.warning(f"⚠️ {total_failed} model(s) failed during training")

            except Exception as e:
                st.error(f"❌ Enhanced training pipeline error: {e}")
                with st.expander("🐛 Error Details", expanded=False):
                    st.code(traceback.format_exc())
    
    # Reset the training flag
    st.session_state.execute_training_flag = False
    

# Helper function to display metrics in a consistent table format
def _display_metrics_table(metrics_dict, title):
    """Helper function to display algorithm metrics in a consistent table format"""
    
    if not metrics_dict:
        st.info(f"No {title.lower()} available")
        return
    
    st.markdown(f"**{title}:**")
    
    # Convert metrics to display format
    display_data = []
    
    for key, value in metrics_dict.items():
        # Format the metric name for display
        display_name = key.replace('_', ' ').title()
        
        # Handle different value types
        if isinstance(value, bool):
            formatted_value = "✅ Yes" if value else "❌ No"
            value_type = "Boolean"
        elif isinstance(value, int):
            formatted_value = f"{value:,}"
            value_type = "Integer"
        elif isinstance(value, float):
            if abs(value) >= 1000:
                formatted_value = f"{value:,.2f}"
            elif abs(value) >= 1:
                formatted_value = f"{value:.4f}"
            else:
                formatted_value = f"{value:.6f}"
            value_type = "Float"
        elif isinstance(value, str):
            formatted_value = value
            value_type = "String"
        else:
            formatted_value = str(value)
            value_type = "Other"
        
        # Add interpretation for common metrics
        interpretation = ""
        if "accuracy" in key.lower() or "score" in key.lower():
            if isinstance(value, (int, float)):
                if value >= 0.9:
                    interpretation = "🟢 Excellent"
                elif value >= 0.8:
                    interpretation = "🟡 Good"
                elif value >= 0.7:
                    interpretation = "🟠 Fair"
                else:
                    interpretation = "🔴 Poor"
        elif "error" in key.lower() or "loss" in key.lower():
            if isinstance(value, (int, float)):
                if value <= 0.1:
                    interpretation = "🟢 Low"
                elif value <= 0.3:
                    interpretation = "🟡 Moderate"
                else:
                    interpretation = "🔴 High"
        elif "ratio" in key.lower() or "percentage" in key.lower():
            if isinstance(value, (int, float)):
                if value >= 0.8:
                    interpretation = "🟢 High"
                elif value >= 0.5:
                    interpretation = "🟡 Medium"
                else:
                    interpretation = "🔴 Low"
        
        display_data.append({
            "📊 Metric": display_name,
            "📈 Value": formatted_value,
            "🏷️ Type": value_type,
            "💡 Assessment": interpretation if interpretation else "—"
        })
    
    # Create and display the table
    if display_data:
        metrics_df = pd.DataFrame(display_data)
        
        # Style the table
        styled_metrics = metrics_df.style.set_properties(**{
            'padding': '8px',
            'font-size': '13px',
            'text-align': 'center',
            'border': '1px solid var(--text-color-secondary)'
        }).set_table_styles([
            {'selector': 'th', 'props': [
                ('background-color', '#17a2b8'),
                ('color', 'white'),
                ('font-weight', 'bold'),
                ('text-align', 'center'),
                ('padding', '10px'),
                ('border', '1px solid #17a2b8')
            ]},
            {'selector': 'td:first-child', 'props': [
                ('text-align', 'left'),
                ('font-weight', 'bold')
            ]},
            {'selector': 'td', 'props': [
                ('text-align', 'center'),
                ('vertical-align', 'middle'),
                ('border', '1px solid var(--text-color-secondary)')
            ]},
            {'selector': '', 'props': [
                ('border-collapse', 'collapse'),
                ('margin', '10px 0'),
                ('border-radius', '6px'),
                ('overflow', 'hidden')
            ]}
        ])
        
        st.dataframe(styled_metrics, use_container_width=True, hide_index=True)
        
        # Add summary
        st.caption(f"📊 {len(display_data)} {title.lower()} displayed")
    
def _display_algorithm_metrics_table_with_comparison(metrics_dict, title, baseline_metrics=None, current_dataset_name="unknown", task_type="classification"):
    """Enhanced helper function to display algorithm metrics with task-aware comparison arrows"""
    
    if not metrics_dict:
        st.info(f"No {title.lower()} available")
        return
    
    st.markdown(f"**{title}:**")
    
    # Define metrics where LOWER is better (mainly regression error metrics)
    lower_is_better_metrics = {
        'mse', 'rmse', 'mae', 'mape', 'max_error', 'mean_squared_error', 
        'root_mean_squared_error', 'mean_absolute_error', 'error', 'loss',
        'deviation', 'variance', 'std', 'standard_deviation', 'residual',
        'absolute_error', 'squared_error', 'huber_loss', 'quantile_loss'
    }
    
    # Convert metrics to display format with task-aware comparison
    display_data = []
    
    for key, value in metrics_dict.items():
        # Format the metric name for display
        display_name = key.replace('_', ' ').title()
        
        # Handle different value types
        if isinstance(value, bool):
            formatted_value = "✅ Yes" if value else "❌ No"
            value_type = "Boolean"
        elif isinstance(value, int):
            formatted_value = f"{value:,}"
            value_type = "Integer"
        elif isinstance(value, float):
            if abs(value) >= 1000:
                formatted_value = f"{value:,.2f}"
            elif abs(value) >= 1:
                formatted_value = f"{value:.4f}"
            else:
                formatted_value = f"{value:.6f}"
            value_type = "Float"
        elif isinstance(value, str):
            formatted_value = value
            value_type = "String"
        else:
            formatted_value = str(value)
            value_type = "Other"
        
        # Task-aware interpretation for common metrics
        interpretation = ""
        
        if task_type == "regression":
            # Regression-specific interpretations
            if any(err_term in key.lower() for err_term in lower_is_better_metrics):
                # Error metrics: lower is better
                if isinstance(value, (int, float)):
                    if value <= 0.01:
                        interpretation = "🟢 Excellent (Very Low Error)"
                    elif value <= 0.1:
                        interpretation = "🟡 Good (Low Error)"
                    elif value <= 0.3:
                        interpretation = "🟠 Fair (Moderate Error)"
                    elif value <= 0.5:
                        interpretation = "🔴 Poor (High Error)"
                    else:
                        interpretation = "⚫ Critical (Very High Error)"
            elif "r2" in key.lower() or ("score" in key.lower() and "error" not in key.lower()):
                # R² and similar performance scores: higher is better
                if isinstance(value, (int, float)):
                    if value >= 0.95:
                        interpretation = "🟢 Excellent Fit"
                    elif value >= 0.85:
                        interpretation = "🟡 Good Fit"
                    elif value >= 0.70:
                        interpretation = "🟠 Fair Fit"
                    elif value >= 0.50:
                        interpretation = "🔴 Poor Fit"
                    else:
                        interpretation = "⚫ Very Poor Fit"
            elif "explained" in key.lower() and "variance" in key.lower():
                # Explained variance: higher is better
                if isinstance(value, (int, float)):
                    if value >= 0.9:
                        interpretation = "🟢 Excellent Explanation"
                    elif value >= 0.8:
                        interpretation = "🟡 Good Explanation"
                    elif value >= 0.7:
                        interpretation = "🟠 Fair Explanation"
                    else:
                        interpretation = "🔴 Poor Explanation"
        else:
            # Classification-specific interpretations (original logic)
            if "accuracy" in key.lower() or "score" in key.lower():
                if isinstance(value, (int, float)):
                    if value >= 0.95:
                        interpretation = "🟢 Excellent"
                    elif value >= 0.85:
                        interpretation = "🟡 Good"
                    elif value >= 0.75:
                        interpretation = "🟠 Fair"
                    elif value >= 0.65:
                        interpretation = "🔴 Poor"
                    else:
                        interpretation = "⚫ Critical"
            elif "precision" in key.lower() or "recall" in key.lower() or "f1" in key.lower():
                if isinstance(value, (int, float)):
                    if value >= 0.9:
                        interpretation = "🟢 Excellent"
                    elif value >= 0.8:
                        interpretation = "🟡 Good"
                    elif value >= 0.7:
                        interpretation = "🟠 Fair"
                    else:
                        interpretation = "🔴 Poor"
        
        # General interpretations for both task types
        if not interpretation:
            if "error" in key.lower() or "loss" in key.lower():
                if isinstance(value, (int, float)):
                    if value <= 0.1:
                        interpretation = "🟢 Low"
                    elif value <= 0.3:
                        interpretation = "🟡 Moderate"
                    else:
                        interpretation = "🔴 High"
            elif "ratio" in key.lower() or "percentage" in key.lower():
                if isinstance(value, (int, float)):
                    if value >= 0.8:
                        interpretation = "🟢 High"
                    elif value >= 0.5:
                        interpretation = "🟡 Medium"
                    else:
                        interpretation = "🔴 Low"
        
        # Task-aware comparison with baseline (triangle arrows)
        comparison_str = "—"
        if (baseline_metrics and 
            key in baseline_metrics and 
            current_dataset_name != 'original' and
            isinstance(value, (int, float)) and 
            isinstance(baseline_metrics[key], (int, float))):
            
            baseline_value = baseline_metrics[key]
            
            if baseline_value != 0:
                # Calculate percentage difference
                diff_percentage = ((value - baseline_value) / abs(baseline_value)) * 100
                threshold = 0.1
                
                # Check if this metric type has reversed logic (lower is better)
                is_lower_better = any(err_term in key.lower() for err_term in lower_is_better_metrics)
                
                if is_lower_better:
                    # For error metrics: decrease (negative diff) is GOOD, increase is BAD
                    if diff_percentage < -threshold:
                        comparison_str = f"▲ {diff_percentage:.2f}%"  # Green arrow for improvement (reduction in error)
                    elif diff_percentage > threshold:
                        comparison_str = f"▼ +{diff_percentage:.2f}%"  # Red arrow for degradation (increase in error)
                    else:
                        comparison_str = f"► {diff_percentage:.2f}%"
                else:
                    # For performance metrics: increase is GOOD, decrease is BAD (current logic)
                    if diff_percentage > threshold:
                        comparison_str = f"▲ +{diff_percentage:.2f}%"
                    elif diff_percentage < -threshold:
                        comparison_str = f"▼ {diff_percentage:.2f}%"
                    else:
                        comparison_str = f"► {diff_percentage:.2f}%"
            elif value > 0:
                comparison_str = "▲ New"
            else:
                comparison_str = "► 0.0%"
        
        # Build the row data
        row_data = {
            "📊 Metric": display_name,
            "📈 Value": formatted_value,
            "🏷️ Type": value_type,
            "💡 Assessment": interpretation if interpretation else "—"
        }
        
        # Add comparison column only if we have baseline data and it's not the original dataset
        if baseline_metrics and current_dataset_name != 'original':
            row_data["🆚 vs Original"] = comparison_str
        
        display_data.append(row_data)
    
    # Create and display the table
    if display_data:
        metrics_df = pd.DataFrame(display_data)
        
        # Helper function to color triangle arrow difference cells for algorithm metrics
        def color_algorithm_triangle_diff_metric_val(val_str):
            if isinstance(val_str, str):
                if val_str.startswith("▲"):
                    return 'color: green; font-weight: bold'
                elif val_str.startswith("▼"):
                    return 'color: red; font-weight: bold'
                elif val_str.startswith("►"):
                    return 'color: gray; font-weight: normal'
            return ''
        
        # Style the table with enhanced comparison support
        styled_metrics = metrics_df.style.set_properties(**{
            'padding': '8px',
            'font-size': '13px',
            'text-align': 'center',
            'border': '1px solid var(--text-color-secondary)'
        }).set_table_styles([
            {'selector': 'th', 'props': [
                ('background-color', '#17a2b8'),
                ('color', 'white'),
                ('font-weight', 'bold'),
                ('text-align', 'center'),
                ('padding', '10px'),
                ('border', '1px solid #17a2b8')
            ]},
            {'selector': 'td:first-child', 'props': [
                ('text-align', 'left'),
                ('font-weight', 'bold')
            ]},
            {'selector': 'td', 'props': [
                ('text-align', 'center'),
                ('vertical-align', 'middle'),
                ('border', '1px solid var(--text-color-secondary)')
            ]},
            {'selector': '', 'props': [
                ('border-collapse', 'collapse'),
                ('margin', '10px 0'),
                ('border-radius', '6px'),
                ('overflow', 'hidden')
            ]}
        ])
        
        # Apply triangle arrow coloring if comparison column exists
        if "🆚 vs Original" in metrics_df.columns:
            styled_metrics = styled_metrics.apply(
                lambda x: x.map(color_algorithm_triangle_diff_metric_val) if x.name == "🆚 vs Original" else [''] * len(x),
                axis=0
            )
        
        st.dataframe(styled_metrics, use_container_width=True, hide_index=True)
        
        # Add task-aware summary with comparison insights
        if baseline_metrics and current_dataset_name != 'original':
            # Count improvements, degradations, and neutral changes in algorithm metrics
            if "🆚 vs Original" in metrics_df.columns:
                comparison_values = metrics_df["🆚 vs Original"].tolist()
                improvements = sum(1 for val in comparison_values if isinstance(val, str) and val.startswith('▲'))
                degradations = sum(1 for val in comparison_values if isinstance(val, str) and val.startswith('▼'))
                neutral = sum(1 for val in comparison_values if isinstance(val, str) and val.startswith('►'))
                
                if improvements + degradations + neutral > 0:
                    st.markdown(f"**📊 Algorithm Metrics Comparison Summary ({task_type.title()}):**")
                    
                    summary_col1, summary_col2, summary_col3 = st.columns(3)
                    
                    with summary_col1:
                        improvement_label = "Lower Error" if task_type == "regression" else "Better Performance"
                        st.metric("🟢 Improved", improvements, improvement_label)
                    
                    with summary_col2:
                        degradation_label = "Higher Error" if task_type == "regression" else "Reduced Performance"
                        st.metric("🔴 Declined", degradations, degradation_label)
                    
                    with summary_col3:
                        stable_label = "Similar Error" if task_type == "regression" else "Similar Performance"
                        st.metric("⚪ Stable", neutral, stable_label)
        
        # Add summary with task context
        comparison_text = f" with comparison to original dataset" if baseline_metrics and current_dataset_name != 'original' else ""
        task_context = f" ({task_type} task)" if task_type else ""
        st.caption(f"📊 {len(display_data)} {title.lower()} displayed{comparison_text}{task_context}")
# --- Display Results ---

# === EXTERNAL DASHBOARD SECTION ===
if st.session_state.get('external_dashboard_enabled', False):
    st.markdown("---")
    st.header("📁 External Results Dashboard")
    st.markdown("*Analyzing uploaded experiment results with full PPML capabilities*")
    
    if st.session_state.get('external_dashboard_data') is not None:
        external_data = st.session_state.external_dashboard_data
        
        # Display file info
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("📊 Rows", external_data.shape[0])
        with col2:
            st.metric("📈 Columns", external_data.shape[1])
        with col3:
            st.metric("💾 Size", f"{external_data.memory_usage(deep=True).sum() / 1024**2:.1f} MB")
        
        # Column selection and filtering
        st.markdown("### 📋 Data Configuration")
        
        # Get available columns
        available_columns = external_data.columns.tolist()
          # Smart column detection and pre-selection
        core_columns = [col for col in available_columns if any(pattern in col.lower() for pattern in ['dataset', 'model', 'type', 'method'])]
        metric_columns = [col for col in available_columns if any(pattern in col.lower() for pattern in ['accuracy', 'f1', 'precision', 'recall', 'score', 'rmse', 'mae']) and col not in core_columns]
        comparison_columns = [col for col in available_columns if ('% diff' in col.lower() or 'diff' in col.lower()) and col not in core_columns and col not in metric_columns]
        
        # Combine and remove duplicates while preserving order
        default_columns = []
        for col_list in [core_columns, metric_columns, comparison_columns]:
            for col in col_list:
                if col not in default_columns:
                    default_columns.append(col)
        
        # If no smart detection worked, use first 10 columns
        if not default_columns:
            default_columns = available_columns[:min(10, len(available_columns))]
        
        # Column selection interface
        selected_external_columns = st.multiselect(
            "📊 Select columns to analyze:",
            available_columns,
            default=default_columns,
            key="external_dashboard_columns",
            help="Choose which columns to include in the analysis"        )
        
        if selected_external_columns:
            # Remove duplicates while preserving order
            unique_columns = []
            for col in selected_external_columns:
                if col not in unique_columns:
                    unique_columns.append(col)
            
            # Ensure all selected columns exist in the dataframe
            valid_columns = [col for col in unique_columns if col in external_data.columns]
            
            if len(valid_columns) != len(unique_columns):
                missing_cols = [col for col in unique_columns if col not in external_data.columns]
                st.warning(f"⚠️ Some selected columns were not found in the data: {missing_cols}")
            
            if valid_columns:
                filtered_external_data = external_data[valid_columns].copy()
                
                # Display selected data
                st.markdown("### 📊 Selected Data Preview")
                st.dataframe(filtered_external_data, use_container_width=True)
                  # Advanced filtering options
                st.markdown("### 🔍 Advanced Filtering")
                filter_col1, filter_col2, filter_col3 = st.columns(3)
                
                # Model filter
                model_columns = [col for col in valid_columns if 'model' in col.lower()]
                if model_columns:
                    with filter_col1:
                        model_col = st.selectbox("🤖 Model Column:", model_columns, key="ext_model_col")
                        available_models = filtered_external_data[model_col].unique()
                        selected_models = st.multiselect(
                            "Filter by Models:",
                            available_models,
                            default=available_models,
                            key="ext_model_filter"
                        )
                        if selected_models:
                            filtered_external_data = filtered_external_data[filtered_external_data[model_col].isin(selected_models)]
                
                # Dataset filter
                dataset_columns = [col for col in valid_columns if any(pattern in col.lower() for pattern in ['dataset', 'data'])]
                if dataset_columns:
                    with filter_col2:
                        dataset_col = st.selectbox("🗂️ Dataset Column:", dataset_columns, key="ext_dataset_col")
                        available_datasets = filtered_external_data[dataset_col].unique()
                        selected_datasets = st.multiselect(
                            "Filter by Datasets:",
                            available_datasets,
                            default=available_datasets,
                            key="ext_dataset_filter"
                        )
                        if selected_datasets:
                            filtered_external_data = filtered_external_data[filtered_external_data[dataset_col].isin(selected_datasets)]
                
                # Privacy method filter
                privacy_columns = [col for col in valid_columns if any(pattern in col.lower() for pattern in ['type', 'method', 'privacy'])]
                if privacy_columns:
                    with filter_col3:
                        privacy_col = st.selectbox("🔒 Privacy Column:", privacy_columns, key="ext_privacy_col")
                        available_methods = filtered_external_data[privacy_col].unique()
                        selected_methods = st.multiselect(
                            "Filter by Methods:",
                            available_methods,
                            default=available_methods,
                            key="ext_privacy_filter"
                        )
                        if selected_methods:
                            filtered_external_data = filtered_external_data[filtered_external_data[privacy_col].isin(selected_methods)]                
                # Show filtered results
                if len(filtered_external_data) > 0:
                    st.markdown(f"### 📈 Filtered Results ({len(filtered_external_data)} rows)")
                    st.dataframe(filtered_external_data, use_container_width=True)
                    
                    # === INTEGRATED PPML DASHBOARD ===
                    st.markdown("---")
                    st.markdown("### 🛡️ Professional PPML Analysis")
                    st.markdown("*Full PPML dashboard capabilities applied to your external data*")
                    
                    # Initialize the PPML visualizer if not already done
                    if 'ppml_dashboard_visualizer' in st.session_state:
                        ppml_viz_external = st.session_state.ppml_dashboard_visualizer
                        
                        # Prepare data for PPML analysis (same format as table-driven analysis)
                        st.session_state.filtered_comprehensive_df = filtered_external_data
                        st.session_state.comprehensive_selected_columns = valid_columns
                        
                        st.info(f"📊 **External Data Ready:** {len(filtered_external_data)} rows × {len(valid_columns)} columns loaded for professional PPML analysis")
                        
                        # Configuration UI for external dashboard
                        st.markdown("#### ⚙️ Professional Dashboard Configuration")
                        external_config_key_prefix = "ppml_external_dashboard_v2"
                        external_config = ppml_viz_external.get_config_ui(key_prefix=external_config_key_prefix)
                        
                        # Render the full PPML dashboard with external data
                        if external_config:
                            with st.spinner("🔄 Loading Professional PPML Dashboard for External Data..."):
                                try:
                                    # Use the same render method as the main dashboard
                                    success = ppml_viz_external.render(
                                        data=filtered_external_data,  # Pass the external data
                                        model_results=[],  # No model results needed for file-based analysis
                                        config=external_config
                                    )
                                    
                                    if not success:
                                        st.warning("⚠️ External dashboard rendering encountered issues. Please check your data format and column selection.")
                                        st.markdown("**Troubleshooting Tips:**")
                                        st.markdown("- Ensure your data has model performance metrics")
                                        st.markdown("- Check that column names match expected patterns")
                                        st.markdown("- Verify data types are numeric for metric columns")
                                except Exception as e:
                                    st.error(f"❌ Error rendering PPML dashboard: {str(e)}")
                                    st.markdown("**Debug Information:**")
                                    st.markdown(f"- Data shape: {filtered_external_data.shape}")
                                    st.markdown(f"- Columns: {list(filtered_external_data.columns)}")
                                    st.markdown(f"- Data types: {filtered_external_data.dtypes.to_dict()}")
                        else:
                            st.info("ℹ️ Please configure the dashboard settings in the sidebar to activate the full PPML analysis.")
                    
                    else:
                        st.error("⚠️ PPML Dashboard not initialized. Please restart the application.")
                        
                    # === EXPORT FUNCTIONALITY ===
                    st.markdown("---")
                    st.markdown("### 💾 Export Options")
                    
                    export_ext_col1, export_ext_col2, export_ext_col3 = st.columns(3)
                    
                    with export_ext_col1:
                        if st.button("📥 Download Filtered Data", key="export_external_filtered"):
                            csv_data = filtered_external_data.to_csv(index=False)
                            st.download_button(
                                label="💾 Download CSV",
                                data=csv_data,
                                file_name=f"external_analysis_filtered_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
                                mime="text/csv",
                                use_container_width=True
                            )
                    
                    with export_ext_col2:
                        if st.button("📊 Download Complete Data", key="export_external_complete"):
                            csv_data = external_data.to_csv(index=False)
                            st.download_button(
                                label="💾 Download All Data",
                                data=csv_data,
                                file_name=f"external_analysis_complete_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
                                mime="text/csv",
                                use_container_width=True
                            )                    
                    with export_ext_col3:
                        st.info(f"📊 Current view: {len(filtered_external_data)} rows × {len(valid_columns)} columns")
                    
                else:
                    st.warning("🔍 No data matches your current filters. Please adjust the filter criteria.")
            else:
                st.error("❌ No valid columns found in the uploaded data.")
                
        else:
            st.warning("📊 Please select at least one column to analyze your external data.")
    
    else:
        st.info("📁 Upload a file in the Dashboard tab (sidebar) to activate the external results analysis.")
        
        # Show upload reminder
        st.markdown("### 🚀 Quick Start Guide")
        st.markdown("""
        1. Go to the **📈 Dashboard** tab in the sidebar
        2. Check **🔓 Activate External Dashboard** 
        3. Upload your Excel/CSV/JSON file with experiment results
        4. Return here to see the full PPML analysis dashboard
        """)
    
    st.markdown("---")

# === ORIGINAL EXPERIMENT RESULTS SECTION ===
if st.session_state.experiment_results:
    st.header("📊 Experiment Results")

    for idx, res in enumerate(st.session_state.experiment_results):
        if "error" in res:
            continue
        
        model_display = res['model_name']
        target_name = res.get('target_column', 'Unknown')
        
        # Calculate run number
        current_run_number = sum(1 for i, r in enumerate(st.session_state.experiment_results[:idx+1]) 
                               if r.get('model_name') == res['model_name'] 
                               and r.get('target_column') == res['target_column']
                               and "error" not in r)
        
        result_title = f"{model_display} - Target: {target_name} - Run #{current_run_number}"
        
        # Enhanced expander title with dataset and model information
        dataset_info = ""
        if res.get('dataset_name'):
            dataset_name = res['dataset_name']
            
            # Check if this is an anonymized dataset with user label
            if dataset_name != 'original' and res.get('full_dataset_id'):
                full_id = res['full_dataset_id']
                if full_id in st.session_state.dataset_metadata:
                    user_label = st.session_state.dataset_metadata[full_id].get('user_label')
                    if user_label:
                        dataset_info = f"📊 {user_label} (#{res.get('dataset_number', 'N/A')})"
                    else:
                        method_name = res.get('anonymization_method', dataset_name)
                        dataset_info = f"📊 {method_name.title()} #{res.get('dataset_number', 'N/A')}"
                else:
                    dataset_info = f"📊 {dataset_name.title()}"
            elif dataset_name == 'original':
                dataset_info = "📊 Original"
            else:
                dataset_info = f"📊 {dataset_name.title()}"

        # Build enhanced title with dataset and model information
        enhanced_title = f"{result_title} | {dataset_info}" if dataset_info else result_title
        
        # Make each result collapsible with key metrics in the header
        if res.get('task_type', 'classification') == 'regression':
            # For regression, show R² and RMSE
            primary_display = f"{res.get('r2_score', res.get('accuracy', 0)):.3f}"
            secondary_display = f"{res.get('rmse', 0):.3f}"
            header_metrics = f"R²: {primary_display} | RMSE: {secondary_display}"
        else:
            # For classification, show Accuracy and F1
            accuracy_display = f"{res['accuracy']:.3f}"
            f1_display = f"{res['f1_score']:.3f}"
            header_metrics = f"Accuracy: {accuracy_display} | F1: {f1_display}"

        with st.expander(f"📈 {enhanced_title} | {header_metrics}", expanded=False):
            # Display model description
            if res.get('model_description'):
                st.caption(res['model_description'])
            
            # REPLACE THE TOGGLE SECTION WITH TABS
            st.markdown("---")
            
            # Create tabs for different sections
            tab_names = ["📊 Performance Metrics"]
            
            # Conditionally add tabs based on available data
            if res.get('hyperparameters'):
                tab_names.append("⚙️ Hyperparameters")
            
            if res.get('algorithm_specific_metrics'):
                tab_names.append("💡 Algorithm Metrics")
            
            # Only add confusion matrix tab for classification tasks
            if res.get('confusion_matrix') is not None and res.get('task_type', 'classification') == 'classification':
                tab_names.append("🔀 Confusion Matrix")
            
            # Only add regression analysis tab for regression tasks  
            if res.get('task_type', 'classification') == 'regression':
                tab_names.append("📊 Regression Analysis")
            
            if st.session_state.get('selected_visualizations', []):
                # Check if this model has visualizations
                plugin_name = res['model_name']
                plugin_visualizations = [
                    viz_key for viz_key in st.session_state.selected_visualizations 
                    if viz_key.startswith(f"{plugin_name}_")
                ]
                if plugin_visualizations:
                    tab_names.append("📈 Visualizations")
            
            # Create the tabs
            tabs = st.tabs(tab_names)
            
            # Performance Metrics Tab (always first - index 0)
            with tabs[0]:
                st.markdown("**📊 Performance Metrics Dashboard**")
                
                # Check task type to display appropriate metrics
                task_type = res.get('task_type', 'classification')
                
                if task_type == 'regression':
                    # Regression metrics display
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("🎯 R² Score", f"{res.get('r2_score', res.get('accuracy', 0)):.4f}", 
                                 help="Coefficient of determination - proportion of variance explained")
                    with col2:
                        st.metric("📏 RMSE", f"{res.get('rmse', 0):.4f}",
                                 help="Root Mean Square Error - prediction error magnitude")
                    with col3:
                        st.metric("📊 MAE", f"{res.get('mae', 0):.4f}",
                                 help="Mean Absolute Error - average prediction error")
                    with col4:
                        st.metric("📈 MAPE", f"{res.get('mape', 0):.2f}%",
                                 help="Mean Absolute Percentage Error")
                    
                    # Additional regression metrics in a second row
                    col5, col6, col7, col8 = st.columns(4)
                    with col5:
                        st.metric("🔍 MSE", f"{res.get('mse', 0):.4f}",
                                 help="Mean Square Error")
                    with col6:
                        st.metric("📈 Explained Variance", f"{res.get('explained_variance', 0):.4f}",
                                 help="Explained variance score")
                    with col7:
                        st.metric("⚠️ Max Error", f"{res.get('max_error', 0):.4f}",
                                 help="Maximum residual error")
                    with col8:
                        # Primary metric indicator
                        primary_metric = res.get('primary_metric_name', 'R² Score')
                        primary_value = res.get(res.get('primary_metric', 'r2_score'), 0)
                        st.metric(f"🏆 {primary_metric}", f"{primary_value:.4f}",
                                 help="Primary evaluation metric for this regression task")
                
                else:
                    # Classification metrics display (existing code)
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("🎯 Accuracy", f"{res['accuracy']:.4f}", 
                                 help="Overall correctness of predictions")
                    with col2:
                        st.metric("⚖️ Precision", f"{res['precision']:.4f}",
                                 help="Accuracy of positive predictions")
                    with col3:
                        st.metric("🔍 Recall", f"{res['recall']:.4f}",
                                 help="Coverage of actual positive cases")
                    with col4:
                        st.metric("🏆 F1-Score", f"{res['f1_score']:.4f}",
                                 help="Harmonic mean of precision and recall")
                
                # Display custom metrics if available (works for both task types)
                if res.get('custom_metrics'):
                    st.markdown("**🎯 Advanced Evaluation Metrics**")
                    custom_metrics_data = []
                    for metric_name, value in res['custom_metrics'].items():
                        if isinstance(value, (int, float)):
                            interpretation = ""
                            quality_indicator = ""
                            if PLUGINS_AVAILABLE:
                                metric_plugin = metric_manager.get_metric(metric_name)
                                if metric_plugin:
                                    interpretation = metric_plugin.get_interpretation(value)
                                    # Add quality indicators
                                    if interpretation in ["Excellent", "Very strong positive correlation"]:
                                        quality_indicator = "🟢"
                                    elif interpretation in ["Good", "Strong positive correlation"]:
                                        quality_indicator = "🟡"
                                    elif interpretation in ["Fair", "Moderate positive correlation"]:
                                        quality_indicator = "🟠"
                                    else:
                                        quality_indicator = "🔴"
                            
                            custom_metrics_data.append({
                                "🏷️ Metric": metric_name,
                                "📊 Value": f"{value:.4f}",
                                "💡 Interpretation": interpretation or "N/A",
                                "📈 Quality": quality_indicator
                            })
                        else:
                            custom_metrics_data.append({
                                "🏷️ Metric": metric_name,
                                "📊 Value": str(value),
                                "💡 Interpretation": "Error occurred",
                                "📈 Quality": "❌"
                            })
                    
                    if custom_metrics_data:
                        custom_df = pd.DataFrame(custom_metrics_data)
                        st.dataframe(custom_df, use_container_width=True, hide_index=True)
            
            # Now handle other tabs with proper index tracking
            current_tab_index = 1
            
            # Hyperparameters Tab (if it exists)
            if res.get('hyperparameters'):
                with tabs[current_tab_index]:
                    st.markdown("**⚙️ Hyperparameters Configuration**")
                    
                    # [Keep all the existing hyperparameters display logic here]
                    # This is the same code you already have for hyperparameters
                    hyperparam_items = list(res['hyperparameters'].items())
                    
                    # Enhanced Professional Table for key parameters
                    st.markdown("**📋 Quick Parameter Overview:**")
                    
                    # Convert hyperparameters to table format
                    param_table_data = []
                    
                    for param, value in hyperparam_items:
                        # Format value for display
                        if isinstance(value, float):
                            display_value = f"{value:.4f}"
                            param_type = "🔢 Float"
                        elif isinstance(value, bool):
                            display_value = "✅ True" if value else "❌ False"
                            param_type = "🔘 Boolean"
                        elif isinstance(value, int):
                            display_value = f"{value:,}"
                            param_type = "🔢 Integer"
                        else:
                            display_value = str(value)
                            if len(display_value) > 20:
                                display_value = f"{display_value[:20]}..."
                            param_type = "📝 String"
                        
                        # Format parameter name
                        param_display = param.replace('_', ' ').title()
                        
                        # Add contextual description for common parameters
                        description = ""
                        if 'learning' in param.lower():
                            description = "Controls learning speed"
                        elif 'depth' in param.lower() or 'estimator' in param.lower():
                            description = "Model complexity control"
                        elif 'regularization' in param.lower() or param.lower().startswith('c'):
                            description = "Overfitting prevention"
                        elif 'random' in param.lower() and 'state' in param.lower():
                            description = "Reproducibility seed"
                        elif 'max' in param.lower() and 'iter' in param.lower():
                            description = "Training iterations limit"
                        else:
                            description = "Algorithm parameter"
                        
                        param_table_data.append({
                            "⚙️ Parameter": param_display,
                            "📊 Value": display_value,
                            "🏷️ Type": param_type,
                            "💡 Description": description
                        })
                    
                    # Create and display the professional parameters table
                    if param_table_data:
                        params_df = pd.DataFrame(param_table_data)
                        st.dataframe(params_df, use_container_width=True, hide_index=True)
                
                current_tab_index += 1
            
            # Algorithm-Specific Metrics Tab (if it exists)
            if res.get('algorithm_specific_metrics'):
                with tabs[current_tab_index]:
                    st.markdown("**💡 Algorithm-Specific Metrics Dashboard**")
                    
                    algorithm_metrics = res['algorithm_specific_metrics']
                    
                    if algorithm_metrics and isinstance(algorithm_metrics, dict):
                        # Filter out non-metric entries
                        filtered_metrics = {k: v for k, v in algorithm_metrics.items() 
                                          if k != 'status' and not isinstance(v, str) or k == 'status'}
                        
                        if 'status' in algorithm_metrics and algorithm_metrics['status'] == "Model not fitted":
                            st.warning("⚠️ Algorithm-specific metrics not available - model was not properly fitted")
                        elif not filtered_metrics:
                            st.info("ℹ️ No algorithm-specific metrics available for this model")
                        else:
                            # NEW: Get baseline algorithm metrics for comparison
                            baseline_algorithm_metrics = None
                            current_dataset_name = res.get('dataset_name', 'unknown')
                            current_model_name = res.get('model_name', '')
                            
                            # Find original dataset result for the same model to use as baseline
                            for baseline_res in st.session_state.experiment_results:
                                if (baseline_res.get('dataset_name') == 'original' and 
                                    baseline_res.get('model_name') == current_model_name and
                                    baseline_res.get('algorithm_specific_metrics') and
                                    "error" not in baseline_res):
                                    baseline_algorithm_metrics = baseline_res.get('algorithm_specific_metrics', {})
                                    break
                            
                            # Group metrics by category/prefix for better organization
                            metric_groups = {}
                            ungrouped_metrics = {}
                            
                            for key, value in filtered_metrics.items():
                                if '_' in key:
                                    prefix = key.split('_')[0]
                                    if prefix in ['boost', 'fi', 'lt', 'res', 'es', 'conv', 'bv', 'comp', 'cv', 'alpha', 'fs', 'sparsity', 'stability', 'reg']:
                                        if prefix not in metric_groups:
                                            metric_groups[prefix] = {}
                                        metric_groups[prefix][key] = value
                                    else:
                                        ungrouped_metrics[key] = value
                                else:
                                    ungrouped_metrics[key] = value
                            
                            # Display grouped metrics in organized sections with comparison
                            if metric_groups:
                                st.markdown("**📊 Categorized Algorithm Metrics:**")
                                
                                # Create tabs for metric groups if there are multiple groups
                                if len(metric_groups) > 1:
                                    group_names = []
                                    group_labels = {
                                        'boost': '🚀 Boosting',
                                        'fi': '🎯 Feature Importance', 
                                        'lt': '📈 Learning Trajectory',
                                        'res': '📊 Residual Analysis',
                                        'es': '⏹️ Early Stopping',
                                        'conv': '🎯 Convergence',
                                        'bv': '⚖️ Bias-Variance',
                                        'comp': '⚡ Computational',
                                        'cv': '🔄 Cross-Validation',
                                        'alpha': '🎛️ Alpha Selection',
                                        'fs': '🎯 Feature Selection',
                                        'sparsity': '🔗 Sparsity',
                                        'stability': '🏗️ Stability',
                                        'reg': '📏 Regularization'
                                    }
                                    
                                    for group_key in metric_groups.keys():
                                        label = group_labels.get(group_key, f"📊 {group_key.title()}")
                                        group_names.append(label)
                                    
                                    metric_tabs = st.tabs(group_names)
                                    
                                    for i, (group_key, group_metrics) in enumerate(metric_groups.items()):
                                        with metric_tabs[i]:
                                            # NEW: Call enhanced display function with comparison
                                            _display_algorithm_metrics_table_with_comparison(
                                                group_metrics, 
                                                f"{group_key.title()} Metrics",
                                                baseline_algorithm_metrics,
                                                current_dataset_name,
                                                res.get('task_type', 'classification')
                                            )
                                else:
                                    # Single group - display directly with comparison
                                    group_key, group_metrics = next(iter(metric_groups.items()))
                                    _display_algorithm_metrics_table_with_comparison(
                                        group_metrics, 
                                        f"{group_key.title()} Metrics",
                                        baseline_algorithm_metrics,
                                        current_dataset_name,
                                        res.get('task_type', 'classification')
                                    )
                            
                            # Display ungrouped metrics with comparison
                            if ungrouped_metrics:
                                st.markdown("**🔧 General Algorithm Metrics:**")
                                _display_algorithm_metrics_table_with_comparison(
                                    ungrouped_metrics, 
                                    "General Metrics",
                                    baseline_algorithm_metrics,
                                    current_dataset_name,
                                    res.get('task_type', 'classification')
                                )
                            
                            # Summary statistics
                            total_metrics = len(filtered_metrics)
                            st.markdown(f"**📈 Summary:** {total_metrics} algorithm-specific metrics available")
                            
                            # Add metric export option
                            if st.button("📥 Export Algorithm Metrics", key=f"export_alg_metrics_{idx}"):
                                metrics_df = pd.DataFrame(list(filtered_metrics.items()), columns=['Metric', 'Value'])
                                csv_data = metrics_df.to_csv(index=False)
                                st.download_button(
                                    label="💾 Download Algorithm Metrics CSV",
                                    data=csv_data,
                                    file_name=f"algorithm_metrics_{res['model_name']}_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
                                    mime="text/csv"
                                )
                    else:
                        st.info("ℹ️ No algorithm-specific metrics available for this model")
                
                current_tab_index += 1
            
            # Confusion Matrix Tab (only for classification)
            if res.get('confusion_matrix') is not None and res.get('task_type', 'classification') == 'classification':
                with tabs[current_tab_index]:
                    st.markdown("**🔀 Confusion Matrix Analysis**")
                    
                    # [Keep all the existing confusion matrix display logic here]
                    if res.get('class_labels'):
                        cm_df = pd.DataFrame(
                            res['confusion_matrix'], 
                            index=[f"📍 Actual: {label}" for label in res['class_labels']], 
                            columns=[f"🎯 Predicted: {label}" for label in res['class_labels']]
                        )
                    else:
                        cm_df = pd.DataFrame(
                            res['confusion_matrix'], 
                            index=[f"📍 Actual {i}" for i in range(res['confusion_matrix'].shape[0])], 
                            columns=[f"🎯 Predicted {i}" for i in range(res['confusion_matrix'].shape[1])]
                        )
                    
                    st.dataframe(cm_df, use_container_width=True)
                    
                    # Add detailed analysis for binary classification
                    if res.get('class_labels') and len(res['class_labels']) == 2:
                        cm = res['confusion_matrix']
                        tn, fp, fn, tp = cm.ravel()
                        
                        st.markdown("**📊 Detailed Classification Analysis:**")
                        
                        # Primary metrics
                        metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
                        with metric_col1:
                            st.metric("✅ True Positives", tp, 
                                     help="Correctly identified positive cases")
                        with metric_col2:
                            st.metric("✅ True Negatives", tn, 
                                     help="Correctly identified negative cases")
                        with metric_col3:
                            st.metric("❌ False Positives", fp, 
                                     help="Incorrectly identified as positive (Type I Error)")
                        with metric_col4:
                            st.metric("❌ False Negatives", fn, 
                                     help="Incorrectly identified as negative (Type II Error)")
                
                current_tab_index += 1
            
            # Regression Analysis Tab (only for regression)
            if res.get('task_type', 'classification') == 'regression':
                with tabs[current_tab_index]:
                    st.markdown("**📊 Regression Analysis**")
                    
                    # Prediction vs Actual scatter plot (if you have the data)
                    if res.get('y_test') is not None and hasattr(res['trained_model'], 'predict'):
                        try:
                            y_test = res['y_test']
                            y_pred = res['trained_model'].predict(res.get('X_test'))
                            
                            # Create scatter plot data
                            import plotly.express as px
                            
                            plot_df = pd.DataFrame({
                                'Actual': y_test,
                                'Predicted': y_pred
                            })
                            
                            fig = px.scatter(
                                plot_df, 
                                x='Actual', 
                                y='Predicted',
                                title='Predicted vs Actual Values',
                                labels={'Actual': 'Actual Values', 'Predicted': 'Predicted Values'}
                            )
                            
                            # Add perfect prediction line
                            min_val = min(plot_df['Actual'].min(), plot_df['Predicted'].min())
                            max_val = max(plot_df['Actual'].max(), plot_df['Predicted'].max())
                            fig.add_shape(
                                type="line",
                                x0=min_val, y0=min_val,
                                x1=max_val, y1=max_val,
                                line=dict(color="red", dash="dash"),
                                name="Perfect Prediction"
                            )
                            
                            st.plotly_chart(fig, use_container_width=True)
                            
                            # Residuals analysis
                            residuals = y_test - y_pred
                            
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                st.metric("Mean Residual", f"{residuals.mean():.4f}",
                                         help="Average prediction error (should be close to 0)")
                            
                            with col2:
                                st.metric("Std Residual", f"{residuals.std():.4f}",
                                         help="Standard deviation of residuals")
                            
                        except Exception as e:
                            st.warning(f"Could not generate regression plots: {e}")
                            
                            # Fallback: just show the metrics we have
                            st.markdown("**📊 Regression Performance Summary:**")
                            
                            summary_data = [
                                ["🎯 R² Score", f"{res.get('r2_score', res.get('accuracy', 0)):.4f}"],
                                ["📏 RMSE", f"{res.get('rmse', 0):.4f}"],
                                ["📊 MAE", f"{res.get('mae', 0):.4f}"],
                                ["📈 MAPE", f"{res.get('mape', 0):.2f}%"]
                            ]
                            
                            summary_df = pd.DataFrame(summary_data, columns=['Metric', 'Value'])
                            st.dataframe(summary_df, use_container_width=True, hide_index=True)
                    
                    else:
                        st.info("ℹ️ No test data available for detailed regression analysis")
                
                current_tab_index += 1
            
            # Visualizations Tab (if available)
            if st.session_state.get('selected_visualizations', []):
                plugin_name = res['model_name']
                plugin_visualizations = [
                    viz_key for viz_key in st.session_state.selected_visualizations 
                    if viz_key.startswith(f"{plugin_name}_")
                ]
                
                if plugin_visualizations:
                    with tabs[current_tab_index]:
                        st.markdown("**📈 Algorithm-Specific Visualizations**")
                        
                        # [Keep all the existing visualization display logic here]
                        # This is the same code you already have for visualizations
                        plugin = available_plugins.get(plugin_name)
                        if plugin:
                            for viz_key in plugin_visualizations:
                                viz_type = viz_key.replace(f"{plugin_name}_", "")
                                
                                try:
                                    trained_model = res.get('trained_model')
                                    X_test = res.get('X_test')
                                    y_test = res.get('y_test')
                                    original_feature_names = res.get('feature_names', [])
                                    
                                    if not original_feature_names and hasattr(X_test, 'columns'):
                                        original_feature_names = list(X_test.columns)
                                    elif not original_feature_names and X_test is not None:
                                        original_feature_names = [f"feature_{i}" for i in range(X_test.shape[1])]
                                    
                                    if not original_feature_names and hasattr(trained_model, 'feature_names_'):
                                        original_feature_names = trained_model.feature_names_
                                    
                                    if trained_model is not None:
                                        st.markdown(f"#### {plugin.get_available_visualizations().get(viz_type, viz_type)}")
                                        
                                        fig = trained_model.render_visualization(
                                            viz_type, 
                                            X_test=X_test,
                                            y_test=y_test,
                                            feature_names=original_feature_names
                                        )
                                        
                                        if fig is not None:
                                            st.pyplot(fig)
                                            import matplotlib.pyplot as plt
                                            plt.close(fig)
                                    else:
                                        st.warning(f"Cannot display {viz_type} - trained model not available")
                                        
                                except Exception as e:
                                    st.error(f"Error rendering {viz_type}: {str(e)}")
            
            # Delete button at the bottom
            st.markdown("---")
            col_spacer, col_delete = st.columns([3, 1])
            with col_delete:
                if st.button(
                    "🗑️ Delete",
                    key=f"delete_{idx}_{id(res)}", 
                    help=f"Remove {result_title} from results",
                    type="secondary",
                    use_container_width=True
                ):
                    delete_experiment_result(idx)
                    st.rerun()

# Enhanced PPML Comparison Dashboard
if st.session_state.experiment_results:
    # Check if we have multi-dataset results (PPML scenario)
    has_multi_dataset = any(
        'dataset_name' in result and result.get('dataset_name') != 'single_dataset' 
        for result in st.session_state.experiment_results 
        if "error" not in result
    )
    
    if has_multi_dataset:
        st.header("🏆 PPML Benchmarking Results Dashboard")
        
        # Organize results by dataset for PPML analysis
        all_results = {}
        for result in st.session_state.experiment_results:
            if "error" not in result:
                dataset_name = result.get('dataset_name', 'unknown')
                if dataset_name not in all_results:
                    all_results[dataset_name] = []
                all_results[dataset_name].append(result)
        
    # Enhanced metrics overview for PPML
    st.markdown("#### 🔒 **Privacy-Preserving ML Overview**")
    
    col1, col2, col3, col4 = st.columns(4)
    
    # Count successful models
    successful_models = [res for res in st.session_state.experiment_results if "error" not in res]
    total_datasets = len(all_results)
    unique_models = len(set(res['model_name'] for res in successful_models))
    anonymization_methods = len([k for k in all_results.keys() if k != 'original'])
    
    with col1:
        st.metric(
            label="📊 Datasets Compared",
            value=total_datasets,
            delta=f"Original + {anonymization_methods} anonymized",
            help="Number of datasets in the PPML comparison"
        )
    
    with col2:
        st.metric(
            label="🤖 Unique Models",
            value=unique_models,
            delta=f"{len(successful_models)} total runs",
            help="Distinct algorithms tested across all datasets"
        )
    
    with col3:
        # Calculate average utility retention
        utility_retentions = []
        for res in successful_models:
            if 'data_utility' in res:
                retention = res['data_utility'].get('row_retention', 1.0) * 100
                utility_retentions.append(retention)
        
        avg_retention = sum(utility_retentions) / len(utility_retentions) if utility_retentions else 100
        st.metric(
            label="📈 Avg Data Retention",
            value=f"{avg_retention:.1f}%",
            delta="🟢 High" if avg_retention >= 90 else "🟡 Medium" if avg_retention >= 75 else "🔴 Low",
            help="Average data retention across anonymized datasets"
        )
    
    with col4:
        # Best performing anonymized method
        anon_results = [res for res in successful_models if res.get('dataset_type') == 'anonymized']
        if anon_results:
            best_anon = max(anon_results, key=lambda x: x['f1_score'])
            st.metric(
                label="🏆 Best Anonymized F1",
                value=f"{best_anon['f1_score']:.4f}",
                delta=f"🔒 {best_anon.get('anonymization_method', 'Unknown').title()}",
                help="Best F1-Score achieved on anonymized data"
            )
        else:
            st.metric(
                label="🏆 Best Anonymized F1",
                value="N/A",
                delta="❌ No anonymized results",
                help="No anonymized dataset results available"
            )
    
    # Create comprehensive PPML comparison table
    st.markdown("#### 📋 **Comprehensive PPML Comparison Table**")

    comparison_data = []
    for dataset_name_key, results_list in all_results.items(): # dataset_name_key is like "k-anonymity_1" or "original"
        for result in results_list:
            if "error" not in result:
                data_utility = result.get('data_utility', {})
                row_retention = data_utility.get('row_retention', 1.0) * 100
                col_retention = data_utility.get('column_retention', 1.0) * 100
                
                # Get task type to determine appropriate metrics
                task_type = result.get('task_type', 'classification')
                
                # --- ADDED/MODIFIED for user label ---
                display_dataset_name_for_table = "Original"
                privacy_method_for_table = "Original"

                if dataset_name_key != 'original':
                    method_type = result.get('anonymization_method', 'UnknownMethod')
                    dataset_num = result.get('dataset_number', 'N/A')
                    full_id = result.get('full_dataset_id')
                    
                    user_label = None
                    if full_id and full_id in st.session_state.dataset_metadata:
                        user_label = st.session_state.dataset_metadata[full_id].get('user_label')
                    
                    if user_label:
                        display_dataset_name_for_table = f"{user_label} (#{dataset_num})"
                    else:
                        display_dataset_name_for_table = f"{method_type.title()} (#{dataset_num})"
                    privacy_method_for_table = method_type.title()
                # --- END ADDED/MODIFIED ---

                # Create row data based on task type
                if task_type == 'regression':
                    comparison_data.append({
                        "🗂️ Dataset": display_dataset_name_for_table,
                        "🔒 Type": result.get('dataset_type', 'unknown').title(),
                        "🤖 Model": result['model_name'],
                        "🎯 R² Score": result.get('r2_score', result.get('accuracy', 0)),
                        "📏 RMSE": result.get('rmse', 0),
                        "📊 MAE": result.get('mae', 0),
                        "📈 MAPE": result.get('mape', 0),
                        "📊 Privacy Method": privacy_method_for_table,
                        "📈 Row Retention": f"{row_retention:.1f}%",
                        "📊 Col Retention": f"{col_retention:.1f}%"
                    })
                else:  # classification
                    comparison_data.append({
                        "🗂️ Dataset": display_dataset_name_for_table,
                        "🔒 Type": result.get('dataset_type', 'unknown').title(),
                        "🤖 Model": result['model_name'],
                        "🎯 Accuracy": result['accuracy'],
                        "⚖️ Precision": result['precision'], 
                        "🔍 Recall": result['recall'],
                        "🏆 F1-Score": result['f1_score'],
                        "📊 Privacy Method": privacy_method_for_table,
                        "📈 Row Retention": f"{row_retention:.1f}%",
                        "📊 Col Retention": f"{col_retention:.1f}%"
                    })

    if comparison_data:
        comparison_df = pd.DataFrame(comparison_data)
        
        # Add sorting and filtering options - task-aware
        sort_col1, sort_col2, filter_col = st.columns([2, 1, 1])
        
        with sort_col1:
            # Determine available metrics based on task type
            if any('🎯 R² Score' in str(comparison_df.columns) for _ in [1]):
                # Regression metrics available
                sort_options = ["🎯 R² Score", "📏 RMSE", "📊 MAE", "📈 MAPE"]
                default_sort = "🎯 R² Score"
            else:
                # Classification metrics available
                sort_options = ["🏆 F1-Score", "🎯 Accuracy", "⚖️ Precision", "🔍 Recall"]
                default_sort = "🏆 F1-Score"
            
            # Filter to only available columns
            available_sort_options = [opt for opt in sort_options if opt in comparison_df.columns]
            
            sort_by = st.selectbox(
                "📊 Sort by metric:",
                options=available_sort_options,
                index=0 if available_sort_options else None,
                key="ppml_sort_metric"
            )
        
        with sort_col2:
            # For regression: higher R² is better, lower RMSE/MAE/MAPE is better
            # For classification: higher is generally better
            if sort_by in ["📏 RMSE", "📊 MAE", "📈 MAPE"]:
                sort_ascending = st.checkbox(
                    "📈 Ascending (Lower is Better)",
                    value=True,
                    key="ppml_sort_ascending"
                )
            else:
                sort_ascending = st.checkbox(
                    "📈 Ascending",
                    value=False,
                    key="ppml_sort_ascending"
                )
        
        with filter_col:
            filter_dataset = st.selectbox(
                "🗂️ Filter Dataset:",
                options=["All"] + list(comparison_df['🗂️ Dataset'].unique()),
                key="ppml_filter_dataset"
            )
        
        # Apply filtering
        if filter_dataset != "All":
            filtered_df = comparison_df[comparison_df['🗂️ Dataset'] == filter_dataset]
        else:
            filtered_df = comparison_df
        
        # Apply sorting
        if sort_by:
            sorted_df = filtered_df.sort_values(by=sort_by, ascending=sort_ascending)
        else:
            sorted_df = filtered_df
        
        # Enhanced styling for PPML table with task-aware formatting
        if any('🎯 R² Score' in str(sorted_df.columns) for _ in [1]):
            # Regression formatting
            format_dict = {
                "🎯 R² Score": "{:.4f}",
                "📏 RMSE": "{:.4f}",
                "📊 MAE": "{:.4f}",
                "📈 MAPE": "{:.2f}"
            }
        else:
            # Classification formatting
            format_dict = {
                "🎯 Accuracy": "{:.4f}",
                "⚖️ Precision": "{:.4f}",
                "🔍 Recall": "{:.4f}",
                "🏆 F1-Score": "{:.4f}"
            }
        
        styled_ppml = sorted_df.style.format(format_dict).set_properties(**{
            'padding': '10px',
            'font-size': '13px',
            'text-align': 'center',
            'border': '1px solid var(--text-color-secondary)',
            'font-weight': 'bold'
        }).set_table_styles([
            {'selector': 'th', 'props': [
                ('background-color', '#e74c3c'),
                ('color', 'white'),
                ('font-weight', 'bold'),
                ('text-align', 'center'),
                ('padding', '12px'),
                ('border', '1px solid #e74c3c'),
                ('font-size', '14px')
            ]},
            {'selector': 'td', 'props': [
                ('text-align', 'center'),
                ('vertical-align', 'middle'),
                ('border', '1px solid var(--text-color-secondary)')
            ]},
            {'selector': '', 'props': [
                ('border-collapse', 'collapse'),
                ('margin', '15px 0'),
                ('border-radius', '8px'),
                ('overflow', 'hidden'),
                ('box-shadow', '0 3px 6px rgba(0, 0, 0, 0.15)')
            ]}
        ])
        
        # st.dataframe(styled_ppml, use_container_width=True, hide_index=True)
        
        # PPML-specific Privacy vs Utility Analysis - TASK AWARE
        st.markdown("#### 🔒 **Privacy vs Utility Analysis**")
        
        # Group by model and compare across datasets
        for model_name in comparison_df['🤖 Model'].unique():
            model_data = comparison_df[comparison_df['🤖 Model'] == model_name]
            
            # Determine task type from the available columns
            is_regression = any('🎯 R² Score' in str(model_data.columns) for _ in [1])
            task_label = "Regression" if is_regression else "Classification"
            
            with st.expander(f"🤖 {model_name} - Privacy-Utility Trade-off Analysis ({task_label})", expanded=True):
                
                # Create utility degradation metrics
                original_data = model_data[model_data['🗂️ Dataset'] == 'Original']
                
                if not original_data.empty:
                    if is_regression:
                        # Regression metrics
                        original_r2 = original_data['🎯 R² Score'].iloc[0]
                        original_rmse = original_data['📏 RMSE'].iloc[0]
                        original_mae = original_data['📊 MAE'].iloc[0]
                        original_mape = original_data['📈 MAPE'].iloc[0]
                        
                        primary_metric_col = '🎯 R² Score'
                        primary_metric_name = 'R² Score'
                        
                    else:
                        # Classification metrics
                        original_f1 = original_data['🏆 F1-Score'].iloc[0]
                        original_accuracy = original_data['🎯 Accuracy'].iloc[0]
                        original_precision = original_data['⚖️ Precision'].iloc[0]
                        original_recall = original_data['🔍 Recall'].iloc[0]
                        
                        primary_metric_col = '🏆 F1-Score'
                        primary_metric_name = 'F1-Score'
                    
                    degradation_data = []
                    
                    for _, row in model_data.iterrows():
                        if row['🗂️ Dataset'] != 'Original':
                            threshold = 0.05
                            
                            if is_regression:
                                # Calculate regression metric differences
                                r2_diff = ((row['🎯 R² Score'] - original_r2) / abs(original_r2)) * 100 if original_r2 != 0 else 0
                                rmse_diff = ((row['📏 RMSE'] - original_rmse) / abs(original_rmse)) * 100 if original_rmse != 0 else 0
                                mae_diff = ((row['📊 MAE'] - original_mae) / abs(original_mae)) * 100 if original_mae != 0 else 0
                                mape_diff = ((row['📈 MAPE'] - original_mape) / abs(original_mape)) * 100 if original_mape != 0 else 0
                                
                                # CORRECTED: Proper formatting for R² Score (higher is better)
                                if r2_diff > threshold:
                                    r2_diff_str = f"▲ +{r2_diff:.2f}%"  # Green up arrow - improvement
                                    r2_is_improvement = True
                                elif r2_diff < -threshold:
                                    r2_diff_str = f"▼ {r2_diff:.2f}%"   # Red down arrow - degradation
                                    r2_is_improvement = False
                                else:
                                    r2_diff_str = f"► {r2_diff:.2f}%"    # Gray right arrow - stable
                                    r2_is_improvement = None
                                
                                # CORRECTED: Proper formatting for error metrics (lower is better)
                                def format_error_diff_corrected(diff_val, metric_name):
                                    """Format error metrics where LOWER values are BETTER"""
                                    if diff_val < -threshold:  # Error decreased (GOOD)
                                        return f"▲ {diff_val:.2f}%", True  # Green up arrow - improvement
                                    elif diff_val > threshold:  # Error increased (BAD)
                                        return f"▼ +{diff_val:.2f}%", False  # Red down arrow - degradation
                                    else:  # Minimal change
                                        return f"► {diff_val:.2f}%", None  # Gray right arrow - stable
                                
                                rmse_diff_str, rmse_is_improvement = format_error_diff_corrected(rmse_diff, "RMSE")
                                mae_diff_str, mae_is_improvement = format_error_diff_corrected(mae_diff, "MAE")
                                mape_diff_str, mape_is_improvement = format_error_diff_corrected(mape_diff, "MAPE")
                                
                                # FIXED: Trade-off quality based on R² utility loss (primary metric)
                                if r2_diff < 0:  # R² decreased
                                    r2_loss = abs(r2_diff)
                                else:  # R² increased or stayed same
                                    r2_loss = 0
                                
                                if r2_loss < 2:
                                    tradeoff_quality = "🟢 Excellent"
                                elif r2_loss < 5:
                                    tradeoff_quality = "🟡 Good"
                                elif r2_loss < 10:
                                    tradeoff_quality = "🟠 Moderate"
                                elif r2_loss < 20:
                                    tradeoff_quality = "🔴 Poor"
                                else:
                                    tradeoff_quality = "⚫ Critical"
                                
                                degradation_data.append({
                                    "🔒 Privacy Method": row['📊 Privacy Method'],
                                    "🗂️ Dataset / Method": row['🗂️ Dataset'],
                                    "🎯 R² Score": f"{row['🎯 R² Score']:.4f}",
                                    "🆚 R² % Diff": r2_diff_str,
                                    "📏 RMSE": f"{row['📏 RMSE']:.4f}",
                                    "🆚 RMSE % Diff": rmse_diff_str,
                                    "📊 MAE": f"{row['📊 MAE']:.4f}",
                                    "🆚 MAE % Diff": mae_diff_str,
                                    "📈 MAPE": f"{row['📈 MAPE']:.2f}",
                                    "🆚 MAPE % Diff": mape_diff_str,
                                    "📈 Data Retention": row['📈 Row Retention'],
                                    "🔒 Privacy-Utility Trade-off": tradeoff_quality,
                                    # Store improvement flags for summary counting
                                    # "_r2_improvement": r2_is_improvement,
                                    # "_rmse_improvement": rmse_is_improvement,
                                    # "_mae_improvement": mae_is_improvement,
                                    # "_mape_improvement": mape_is_improvement
                                })
                                
                            else:
                                # Classification metric differences (existing logic)
                                f1_diff = ((row['🏆 F1-Score'] - original_f1) / abs(original_f1)) * 100 if original_f1 != 0 else 0
                                accuracy_diff = ((row['🎯 Accuracy'] - original_accuracy) / abs(original_accuracy)) * 100 if original_accuracy != 0 else 0
                                precision_diff = ((row['⚖️ Precision'] - original_precision) / abs(original_precision)) * 100 if original_precision != 0 else 0
                                recall_diff = ((row['🔍 Recall'] - original_recall) / abs(original_recall)) * 100 if original_recall != 0 else 0
                                
                                # Arrow formatting for classification
                                def format_classification_diff(diff_val):
                                    if diff_val > threshold:
                                        return f"▲ +{diff_val:.2f}%"
                                    elif diff_val < -threshold:
                                        return f"▼ {diff_val:.2f}%"
                                    else:
                                        return f"► {diff_val:.2f}%"
                                
                                f1_diff_str = format_classification_diff(f1_diff)
                                accuracy_diff_str = format_classification_diff(accuracy_diff)
                                precision_diff_str = format_classification_diff(precision_diff)
                                recall_diff_str = format_classification_diff(recall_diff)
                                
                                # Trade-off quality based on F1 utility loss
                                f1_loss = abs(f1_diff) if f1_diff < 0 else 0
                                if f1_loss < 2:
                                    tradeoff_quality = "🟢 Excellent"
                                elif f1_loss < 5:
                                    tradeoff_quality = "🟡 Good"
                                elif f1_loss < 10:
                                    tradeoff_quality = "🟠 Moderate"
                                elif f1_loss < 20:
                                    tradeoff_quality = "🔴 Poor"
                                else:
                                    tradeoff_quality = "⚫ Critical"
                                
                                degradation_data.append({
                                    "🔒 Privacy Method": row['📊 Privacy Method'],
                                    "🗂️ Dataset / Method": row['🗂️ Dataset'],
                                    "🏆 F1-Score": f"{row['🏆 F1-Score']:.4f}",
                                    "🆚 F1 % Diff": f1_diff_str,
                                    "🎯 Accuracy": f"{row['🎯 Accuracy']:.4f}",
                                    "🆚 Acc % Diff": accuracy_diff_str,
                                    "⚖️ Precision": f"{row['⚖️ Precision']:.4f}",
                                    "🆚 Prec % Diff": precision_diff_str,
                                    "🔍 Recall": f"{row['🔍 Recall']:.4f}",
                                    "🆚 Rec % Diff": recall_diff_str,
                                    "📈 Data Retention": row['📈 Row Retention'],
                                    "🔒 Privacy-Utility Trade-off": tradeoff_quality
                                })

                            if degradation_data:
                                degradation_df = pd.DataFrame(degradation_data)
                                
                                # Helper function to color triangle arrow difference cells
                                def color_triangle_diff_metric_val(val_str):
                                    if isinstance(val_str, str):
                                        if val_str.startswith("▲"):
                                            return 'color: green; font-weight: bold'  # Always green for improvements
                                        elif val_str.startswith("▼"):
                                            return 'color: red; font-weight: bold'    # Always red for degradations
                                        elif val_str.startswith("►"):
                                            return 'color: gray; font-weight: normal' # Always gray for stable
                                    return ''
                                
                                # Style the degradation analysis table with colored arrows
                                styled_degradation = degradation_df.style.set_properties(**{
                                    'padding': '10px',
                                    'font-size': '13px',
                                    'text-align': 'center',
                                    'border': '1px solid var(--text-color-secondary)',
                                    'font-weight': 'bold'
                                }).set_table_styles([
                                    {'selector': 'th', 'props': [
                                        ('background-color', '#9b59b6'),
                                        ('color', 'white'),
                                        ('font-weight', 'bold'),
                                        ('text-align', 'center'),
                                        ('padding', '12px'),
                                        ('border', '1px solid #9b59b6'),
                                        ('font-size', '13px')
                                    ]},
                                    {'selector': 'td', 'props': [
                                        ('text-align', 'center'),
                                        ('vertical-align', 'middle'),
                                        ('border', '1px solid var(--text-color-secondary)')
                                    ]},
                                    {'selector': '', 'props': [
                                        ('border-collapse', 'collapse'),
                                        ('margin', '10px 0'),
                                        ('border-radius', '6px'),
                                        ('overflow', 'hidden'),
                                        ('box-shadow', '0 2px 4px rgba(0, 0, 0, 0.1)')
                                    ]}
                                ])
                                
                                # FIXED: Apply coloring to ALL percentage difference columns at once
                                diff_columns = [col for col in degradation_df.columns if "% Diff" in col]
                                
                                if diff_columns:
                                    def color_all_diff_columns(row):
                                        styled_row = [''] * len(row)
                                        for i, (col_name, val) in enumerate(zip(row.index, row.values)):
                                            if col_name in diff_columns:
                                                styled_row[i] = color_triangle_diff_metric_val(val)
                                        return styled_row
                                    
                                    styled_degradation = styled_degradation.apply(color_all_diff_columns, axis=1)
                                
                                st.dataframe(styled_degradation, use_container_width=True, hide_index=True)
                                
                                # CORRECTED: Enhanced summary insights - task aware with corrected counting
                                if is_regression:
                                    # Count based on arrow indicators in the percentage difference strings
                                    improvements = 0
                                    degradations = 0
                                    neutral = 0
                                    
                                    # Count R² improvements/degradations from the arrow strings
                                    for item in degradation_data:
                                        r2_diff_str = item.get('🆚 R² % Diff', '')
                                        if isinstance(r2_diff_str, str):
                                            if r2_diff_str.startswith('▲'):  # R² improved
                                                improvements += 1
                                            elif r2_diff_str.startswith('▼'):  # R² degraded
                                                degradations += 1
                                            elif r2_diff_str.startswith('►'):  # R² stable
                                                neutral += 1
                                    
                                    # Find best and worst based on R² changes for regression
                                    best_method = max(degradation_data, key=lambda x: float(x['🎯 R² Score']))
                                    worst_method = min(degradation_data, key=lambda x: float(x['🎯 R² Score']))
                                    
                                    best_diff_str = best_method.get('🆚 R² % Diff', 'N/A')
                                    worst_diff_str = worst_method.get('🆚 R² % Diff', 'N/A')
                                    
                                    # Additional regression-specific insights
                                    best_rmse_method = min(degradation_data, key=lambda x: float(x['📏 RMSE']))
                                    best_mae_method = min(degradation_data, key=lambda x: float(x['📊 MAE']))
                                    
                                else:
                                    # Count based on F1 score changes (classification logic)
                                    primary_diff_column = f"🆚 {primary_metric_name} % Diff"
                                    
                                    # Safely extract improvements/degradations from arrow indicators
                                    improvements = 0
                                    degradations = 0
                                    neutral = 0
                                    
                                    for item in degradation_data:
                                        diff_str = item.get(primary_diff_column, item.get('🆚 F1 % Diff', ''))
                                        if isinstance(diff_str, str):
                                            if diff_str.startswith('▲'):
                                                improvements += 1
                                            elif diff_str.startswith('▼'):
                                                degradations += 1
                                            elif diff_str.startswith('►'):
                                                neutral += 1
                                    
                                    # Find best and worst for classification
                                    primary_score_column = "🏆 F1-Score"
                                    if primary_score_column in degradation_data[0]:
                                        best_method = max(degradation_data, key=lambda x: float(x.get(primary_score_column, '0')))
                                        worst_method = min(degradation_data, key=lambda x: float(x.get(primary_score_column, '0')))
                                    else:
                                        # Fallback to accuracy if F1-Score not available
                                        best_method = max(degradation_data, key=lambda x: float(x.get('🎯 Accuracy', '0')))
                                        worst_method = min(degradation_data, key=lambda x: float(x.get('🎯 Accuracy', '0')))
                                    
                                    best_diff_str = best_method.get(primary_diff_column, best_method.get('🆚 F1 % Diff', 'N/A'))
                                    worst_diff_str = worst_method.get(primary_diff_column, worst_method.get('🆚 F1 % Diff', 'N/A'))

                                # Display best and worst performers
                                insight_col1, insight_col2 = st.columns(2)
                                
                                with insight_col1:
                                    best_method_name = best_method.get('🔒 Privacy Method', 'Unknown')
                                    st.success(f"**🏆 Best Performance:** {best_method_name}")
                                    st.caption(f"Change vs Original: {best_diff_str}")
                                    
                                with insight_col2:
                                    worst_method_name = worst_method.get('🔒 Privacy Method', 'Unknown')
                                    st.error(f"**⚠️ Highest Utility Loss:** {worst_method_name}")
                                    st.caption(f"Change vs Original: {worst_diff_str}")
                                
                                # Task-aware performance change summary
                                st.markdown(f"**📊 {task_label} Performance Change Summary:**")
                                
                                summary_col1, summary_col2, summary_col3 = st.columns(3)
                                
                                with summary_col1:
                                    if is_regression:
                                        improvement_label = "Better R² / Lower Error"
                                        improvement_help = "Methods that improved R² score or reduced error metrics"
                                    else:
                                        improvement_label = "Better Performance"
                                        improvement_help = "Methods that improved classification metrics"
                                    
                                    st.metric(
                                        "🟢 Improvements", 
                                        improvements, 
                                        improvement_label,
                                        help=improvement_help
                                    )
                                
                                with summary_col2:
                                    if is_regression:
                                        degradation_label = "Lower R² / Higher Error"
                                        degradation_help = "Methods that reduced R² score or increased error metrics"
                                    else:
                                        degradation_label = "Worse Performance"
                                        degradation_help = "Methods that reduced classification performance"
                                    
                                    st.metric(
                                        "🔴 Degradations", 
                                        degradations, 
                                        degradation_label,
                                        help=degradation_help
                                    )
                                
                                with summary_col3:
                                    if is_regression:
                                        stable_label = "Similar R² / Error"
                                        stable_help = "Methods with minimal change in regression metrics"
                                    else:
                                        stable_label = "Similar Performance" 
                                        stable_help = "Methods with minimal change in classification metrics"
                                    
                                    st.metric(
                                        "⚪ Stable", 
                                        neutral, 
                                        stable_label,
                                        help=stable_help
                                    )
                                
                                # Additional regression-specific insights
                                if is_regression and len(degradation_data) > 1:
                                    st.markdown("**📈 Regression-Specific Insights:**")
                                    
                                    reg_insight_col1, reg_insight_col2 = st.columns(2)
                                    
                                    with reg_insight_col1:
                                        best_rmse_name = best_rmse_method.get('🔒 Privacy Method', 'Unknown')
                                        best_rmse_value = best_rmse_method.get('📏 RMSE', 'N/A')
                                        st.info(f"**📏 Lowest RMSE:** {best_rmse_name} ({best_rmse_value})")
                                    
                                    with reg_insight_col2:
                                        best_mae_name = best_mae_method.get('🔒 Privacy Method', 'Unknown')
                                        best_mae_value = best_mae_method.get('📊 MAE', 'N/A')
                                        st.info(f"**📊 Lowest MAE:** {best_mae_name} ({best_mae_value})")
                                
                                # Privacy-utility trade-off assessment
                                st.markdown("**⚖️ Privacy-Utility Trade-off Assessment:**")
                                
                                # Calculate overall trade-off quality
                                excellent_count = sum(1 for item in degradation_data if item.get('🔒 Privacy-Utility Trade-off', '').startswith('🟢'))
                                good_count = sum(1 for item in degradation_data if item.get('🔒 Privacy-Utility Trade-off', '').startswith('🟡'))
                                moderate_count = sum(1 for item in degradation_data if item.get('🔒 Privacy-Utility Trade-off', '').startswith('🟠'))
                                poor_count = sum(1 for item in degradation_data if item.get('🔒 Privacy-Utility Trade-off', '').startswith('🔴'))
                                
                                trade_off_col1, trade_off_col2, trade_off_col3, trade_off_col4 = st.columns(4)
                                
                                with trade_off_col1:
                                    st.metric("🟢 Excellent", excellent_count, "Minimal utility loss")
                                
                                with trade_off_col2:
                                    st.metric("🟡 Good", good_count, "Acceptable trade-off")
                                
                                with trade_off_col3:
                                    st.metric("🟠 Moderate", moderate_count, "Noticeable utility loss")
                                
                                with trade_off_col4:
                                    st.metric("🔴 Poor", poor_count, "Significant utility loss")
                                
                                # Recommendations based on analysis
                                st.markdown("**💡 Recommendations:**")
                                
                                total_methods = len(degradation_data)
                                if excellent_count >= total_methods * 0.5:
                                    st.success("✅ **Strong PPML Results:** Most methods maintain excellent utility. Consider deploying any of the excellent-rated methods.")
                                elif good_count + excellent_count >= total_methods * 0.7:
                                    st.info("🎯 **Good PPML Results:** Majority of methods show acceptable privacy-utility trade-offs. Focus on excellent and good-rated methods.")
                                elif moderate_count >= total_methods * 0.5:
                                    st.warning("⚠️ **Mixed PPML Results:** Consider parameter tuning or alternative anonymization approaches to improve utility retention.")
                                else:
                                    st.error("🔴 **Poor PPML Results:** Significant utility loss detected. Review anonymization parameters or consider less aggressive privacy techniques.")
                                
                            else:
                                # Handle case where no degradation data is available
                                st.info("ℹ️ No privacy method comparison data available for this model")
                                st.caption("This may occur when:")
                                st.caption("• Only original dataset results are available")
                                st.caption("• Anonymized datasets haven't been processed yet")
                                st.caption("• There are data processing errors")
                                
                else:
                    st.warning("❌ No original dataset results found for comparison")
                    st.caption("To enable privacy-utility analysis:")
                    st.caption("• Ensure the original dataset is loaded and processed")
                    st.caption("• Train models on the original dataset first")
                    st.caption("• Then train the same models on anonymized datasets")

        # Overall PPML Insights Dashboard - TASK AWARE
        st.markdown("#### 🎯 **Overall PPML Performance Insights**")

        insights_tab1, insights_tab2 = st.tabs(["🏆 Top Performing Privacy Methods", "📊 Dataset Quality Assessment"])
        
        # Determine if we have regression or classification data
        is_regression_data = any('🎯 R² Score' in str(comparison_df.columns) for _ in [1])
        primary_metric_for_ranking = '🎯 R² Score' if is_regression_data else '🏆 F1-Score'
        metric_name_for_display = 'R² Score' if is_regression_data else 'F1-Score'

        with insights_tab1:
            st.markdown("**🏆 Top Performing Privacy Methods:**")
            
            # Find best anonymization methods by primary metric
            anon_performance = {}
            for _, row in comparison_df.iterrows():
                if row['🗂️ Dataset'] != 'Original':
                    method = row['📊 Privacy Method']
                    primary_score = row[primary_metric_for_ranking]
                    
                    if method not in anon_performance:
                        anon_performance[method] = []
                    anon_performance[method].append(primary_score)
            
            # Calculate averages and create ranking
            method_rankings = []
            for method, scores in anon_performance.items():
                avg_score = sum(scores) / len(scores)
                method_rankings.append({
                    "🔒 Method": method,
                    f"🏆 Avg {metric_name_for_display}": f"{avg_score:.4f}",
                    "📊 Models Tested": len(scores)
                })
            
            method_rankings.sort(key=lambda x: float(x[f"🏆 Avg {metric_name_for_display}"]), reverse=True)
            
            if method_rankings:
                # Add ranking indicators
                for i, ranking in enumerate(method_rankings):
                    if i == 0:
                        ranking["🥇 Rank"] = "🥇 1st"
                    elif i == 1:
                        ranking["🥇 Rank"] = "🥈 2nd"
                    elif i == 2:
                        ranking["🥇 Rank"] = "🥉 3rd"
                    else:
                        ranking["🥇 Rank"] = f"#{i+1}"
                
                ranking_df = pd.DataFrame(method_rankings)
                ranking_df = ranking_df[["🥇 Rank", "🔒 Method", f"🏆 Avg {metric_name_for_display}", "📊 Models Tested"]]
                
                st.dataframe(ranking_df, use_container_width=True, hide_index=True)
            else:
                st.info("No anonymization method performance data available")

        with insights_tab2:
            st.markdown("**📊 Dataset Quality Assessment:**")
            
            # Assess each dataset using appropriate metric
            dataset_quality = []
            for dataset_name in comparison_df['🗂️ Dataset'].unique():
                dataset_results = comparison_df[comparison_df['🗂️ Dataset'] == dataset_name]
                
                if is_regression_data:
                    avg_primary = dataset_results['🎯 R² Score'].mean()
                    avg_secondary = dataset_results['📏 RMSE'].mean()
                    # For regression: high R² is good, low RMSE is good
                    combined_score = avg_primary  # Use R² as primary indicator
                else:
                    avg_primary = dataset_results['🏆 F1-Score'].mean()
                    avg_secondary = dataset_results['🎯 Accuracy'].mean()
                    combined_score = (avg_primary + avg_secondary) / 2
                
                model_count = len(dataset_results)
                
                # Determine quality grade
                if combined_score >= 0.9:
                    quality_grade = "🟢 Excellent"
                elif combined_score >= 0.8:
                    quality_grade = "🟡 Good"
                elif combined_score >= 0.7:
                    quality_grade = "🟠 Fair"
                elif combined_score >= 0.6:
                    quality_grade = "🔴 Poor"
                else:
                    quality_grade = "⚫ Critical"
                
                if is_regression_data:
                    dataset_quality.append({
                        "🗂️ Dataset": dataset_name,
                        "🎯 Avg R²": f"{avg_primary:.4f}",
                        "📏 Avg RMSE": f"{avg_secondary:.4f}",
                        "🤖 Models": model_count,
                        "📊 Quality": quality_grade,
                        # Store raw values for percentage calculation
                        "_raw_primary": avg_primary,
                        "_raw_secondary": avg_secondary
                    })
                else:
                    dataset_quality.append({
                        "🗂️ Dataset": dataset_name,
                        "🏆 Avg F1": f"{avg_primary:.4f}",
                        "🎯 Avg Accuracy": f"{avg_secondary:.4f}",
                        "🤖 Models": model_count,
                        "📊 Quality": quality_grade,
                        # Store raw values for percentage calculation
                        "_raw_primary": avg_primary,
                        "_raw_secondary": avg_secondary
                    })

            if dataset_quality:
                # Find original dataset for baseline comparison
                original_baseline = None
                for item in dataset_quality:
                    if item["🗂️ Dataset"] == "Original":
                        original_baseline = item
                        break

                # Add percentage difference columns BEFORE removing raw values
                for item in dataset_quality:
                    if item["🗂️ Dataset"] != "Original" and original_baseline:
                        primary_diff = ((item["_raw_primary"] - original_baseline["_raw_primary"]) / abs(original_baseline["_raw_primary"])) * 100 if original_baseline["_raw_primary"] != 0 else 0
                        secondary_diff = ((item["_raw_secondary"] - original_baseline["_raw_secondary"]) / abs(original_baseline["_raw_secondary"])) * 100 if original_baseline["_raw_secondary"] != 0 else 0

                        threshold = 0.1

                        if is_regression_data:
                            # R² Score: Higher is better, so positive diff is good (green up arrow)
                            if primary_diff > threshold:
                                primary_diff_str = f"▲ +{primary_diff:.1f}%"  # Green up arrow for improvement
                            elif primary_diff < -threshold:
                                primary_diff_str = f"▼ {primary_diff:.1f}%"   # Red down arrow for degradation
                            else:
                                primary_diff_str = f"► {primary_diff:.1f}%"    # Gray right arrow for stable

                            # RMSE: Lower is better, so negative diff is good (green up arrow)
                            if secondary_diff < -threshold:
                                secondary_diff_str = f"▲ {secondary_diff:.1f}%"  # Green up arrow for improvement (reduction)
                            elif secondary_diff > threshold:
                                secondary_diff_str = f"▼ +{secondary_diff:.1f}%"  # Red down arrow for degradation (increase)
                            else:
                                secondary_diff_str = f"► {secondary_diff:.1f}%"    # Gray right arrow for stable

                            item["🆚 R² vs Original"] = primary_diff_str
                            item["🆚 RMSE vs Original"] = secondary_diff_str
                        else:
                            # Classification: Higher is better for both F1 and Accuracy
                            if primary_diff > threshold:
                                primary_diff_str = f"▲ +{primary_diff:.1f}%"
                            elif primary_diff < -threshold:
                                primary_diff_str = f"▼ {primary_diff:.1f}%"
                            else:
                                primary_diff_str = f"► {primary_diff:.1f}%"

                            if secondary_diff > threshold:
                                secondary_diff_str = f"▲ +{secondary_diff:.1f}%"
                            elif secondary_diff < -threshold:
                                secondary_diff_str = f"▼ {secondary_diff:.1f}%"
                            else:
                                secondary_diff_str = f"► {secondary_diff:.1f}%"

                            item["🆚 F1 vs Original"] = primary_diff_str
                            item["🆚 Acc vs Original"] = secondary_diff_str
                    else:
                        # Original dataset or no baseline - no comparison
                        if is_regression_data:
                            item["🆚 R² vs Original"] = "—"
                            item["🆚 RMSE vs Original"] = "—"
                        else:
                            item["🆚 F1 vs Original"] = "—"
                            item["🆚 Acc vs Original"] = "—"

                # NOW remove raw values from display (after percentage calculations)
                for item in dataset_quality:
                    item.pop("_raw_primary", None)
                    item.pop("_raw_secondary", None)

                quality_df = pd.DataFrame(dataset_quality)

                # Helper function to color triangle arrows
                def color_triangle_arrows(val_str):
                    if isinstance(val_str, str):
                        if val_str.startswith("▲"):
                            return 'color: green; font-weight: bold'
                        elif val_str.startswith("▼"):
                            return 'color: red; font-weight: bold'
                        elif val_str.startswith("►"):
                            return 'color: gray; font-weight: normal'
                    return ''

                # Style the table with colored arrows - MOVED BEFORE applying colors
                styled_quality = quality_df.style.set_properties(**{
                    'padding': '8px',
                    'font-size': '13px',
                    'text-align': 'center',
                    'border': '1px solid var(--text-color-secondary)'
                }).set_table_styles([
                    {'selector': 'th', 'props': [
                        ('background-color', '#17a2b8'),
                        ('color', 'white'),
                        ('font-weight', 'bold'),
                        ('text-align', 'center'),
                        ('padding', '10px'),
                        ('border', '1px solid #17a2b8')
                    ]},
                    {'selector': 'td', 'props': [
                        ('text-align', 'center'),
                        ('vertical-align', 'middle'),
                        ('border', '1px solid var(--text-color-secondary)')
                    ]},
                    {'selector': '', 'props': [
                        ('border-collapse', 'collapse'),
                        ('margin', '10px 0'),
                        ('border-radius', '6px'),
                        ('overflow', 'hidden')
                    ]}
                ])

                # Apply coloring to ALL columns that contain arrow symbols - FIXED VERSION
                arrow_columns = []
                for col in quality_df.columns:
                    # Check if this column contains any arrow symbols
                    column_values = quality_df[col].astype(str).tolist()
                    has_arrows = any(isinstance(val, str) and any(arrow in val for arrow in ['▲', '▼', '►']) for val in column_values)
                    
                    if has_arrows:
                        arrow_columns.append(col)

                # Apply coloring to all arrow columns at once
                if arrow_columns:
                    def color_all_arrow_columns(row):
                        styled_row = [''] * len(row)
                        for i, (col_name, val) in enumerate(zip(row.index, row.values)):
                            if col_name in arrow_columns:
                                styled_row[i] = color_triangle_arrows(val)
                        return styled_row
                    
                    styled_quality = styled_quality.apply(color_all_arrow_columns, axis=1)

                st.dataframe(styled_quality, use_container_width=True, hide_index=True)

            else:
                st.info("No dataset quality data available.")

        # Add this section after the existing PPML comparison dashboard (around line 5881)

        # <<< --- NEW COMPREHENSIVE COMPARISON TABLE --- >>>
        st.markdown("---")
        st.subheader("📊 Comprehensive Results Comparison Table")
        st.markdown("**All experiment results with percentage differences in one unified view**")

        # Prepare data for the comprehensive table
        comprehensive_data = []
        
        # Get original baseline results for comparison
        original_baselines = {}
        for result in st.session_state.experiment_results:
            if "error" not in result and result.get('dataset_name') == 'original':
                model_name = result.get('model_name')
                original_baselines[model_name] = result
        
        # Process all results
        for result in st.session_state.experiment_results:
            if "error" in result:
                continue
                
            model_name = result.get('model_name', 'Unknown')
            dataset_name = result.get('dataset_name', 'unknown')
            task_type = result.get('task_type', 'classification')
            
            # Determine display name for dataset with user label support
            if dataset_name == 'original':
                dataset_display = "Original"
                privacy_method = "Original"
                dataset_type = "Original"
            else:
                # Handle anonymized datasets with user labels
                method_type = result.get('anonymization_method', 'Unknown Method')
                dataset_num = result.get('dataset_number', 'N/A')
                full_id = result.get('full_dataset_id')
                
                user_label = None
                if full_id and full_id in st.session_state.dataset_metadata:
                    user_label = st.session_state.dataset_metadata[full_id].get('user_label')
                
                if user_label:
                    dataset_display = f"{user_label} (#{dataset_num})"
                else:
                    dataset_display = f"{method_type.title()} (#{dataset_num})"
                
                privacy_method = method_type.title()
                dataset_type = "Anonymized"
            
            # Get baseline for this model
            baseline = original_baselines.get(model_name)
            
            # Helper function to calculate percentage difference with arrows
            def calc_percentage_diff(current_val, baseline_val, is_lower_better=False):
                if baseline is None or baseline_val == 0:
                    return "—"
                
                diff_pct = ((current_val - baseline_val) / abs(baseline_val)) * 100
                threshold = 0.1
                
                if is_lower_better:
                    # For error metrics: lower is better
                    if diff_pct < -threshold:
                        return f"▲ {diff_pct:.1f}%"  # Green - improvement (reduction)
                    elif diff_pct > threshold:
                        return f"▼ +{diff_pct:.1f}%"  # Red - degradation (increase)
                    else:
                        return f"► {diff_pct:.1f}%"  # Gray - stable
                else:
                    # For performance metrics: higher is better
                    if diff_pct > threshold:
                        return f"▲ +{diff_pct:.1f}%"  # Green - improvement
                    elif diff_pct < -threshold:
                        return f"▼ {diff_pct:.1f}%"  # Red - degradation
                    else:
                        return f"► {diff_pct:.1f}%"  # Gray - stable
            
            # Build row data based on task type
            row_data = {
                "🗂️ Dataset": dataset_display,
                "🔒 Type": dataset_type,
                "🤖 Model": model_name,
                "📊 Privacy Method": privacy_method
            }
            
            if task_type == 'regression':
                # Base regression metrics
                r2_score = result.get('r2_score', result.get('accuracy', 0))
                rmse = result.get('rmse', 0)
                mae = result.get('mae', 0)
                mape = result.get('mape', 0)
                
                row_data.update({
                    "🎯 R² Score": f"{r2_score:.4f}",
                    "📏 RMSE": f"{rmse:.4f}",
                    "📊 MAE": f"{mae:.4f}",
                    "📈 MAPE": f"{mape:.2f}"
                })
                
                # Add percentage differences for regression
                if baseline and dataset_name != 'original':
                    baseline_r2 = baseline.get('r2_score', baseline.get('accuracy', 0))
                    baseline_rmse = baseline.get('rmse', 0)
                    baseline_mae = baseline.get('mae', 0)
                    baseline_mape = baseline.get('mape', 0)
                    
                    row_data.update({
                        "🆚 R² % Diff": calc_percentage_diff(r2_score, baseline_r2, False),
                        "🆚 RMSE % Diff": calc_percentage_diff(rmse, baseline_rmse, True),
                        "🆚 MAE % Diff": calc_percentage_diff(mae, baseline_mae, True),
                        "🆚 MAPE % Diff": calc_percentage_diff(mape, baseline_mape, True)
                    })
                else:
                    # Original dataset - no comparison
                    row_data.update({
                        "🆚 R² % Diff": "—",
                        "🆚 RMSE % Diff": "—", 
                        "🆚 MAE % Diff": "—",
                        "🆚 MAPE % Diff": "—"
                    })
            
            else:  # Classification
                # Base classification metrics
                accuracy = result.get('accuracy', 0)
                precision = result.get('precision', 0)
                recall = result.get('recall', 0)
                f1_score = result.get('f1_score', 0)
                
                row_data.update({
                    "🎯 Accuracy": f"{accuracy:.4f}",
                    "⚖️ Precision": f"{precision:.4f}",
                    "🔍 Recall": f"{recall:.4f}",
                    "🏆 F1-Score": f"{f1_score:.4f}"
                })
                
                # Add percentage differences for classification
                if baseline and dataset_name != 'original':
                    baseline_accuracy = baseline.get('accuracy', 0)
                    baseline_precision = baseline.get('precision', 0)
                    baseline_recall = baseline.get('recall', 0)
                    baseline_f1 = baseline.get('f1_score', 0)
                    
                    row_data.update({
                        "🆚 Accuracy % Diff": calc_percentage_diff(accuracy, baseline_accuracy, False),
                        "🆚 Precision % Diff": calc_percentage_diff(precision, baseline_precision, False),
                        "🆚 Recall % Diff": calc_percentage_diff(recall, baseline_recall, False),
                        "🆚 F1-Score % Diff": calc_percentage_diff(f1_score, baseline_f1, False)
                    })
                else:
                    # Original dataset - no comparison
                    row_data.update({
                        "🆚 Accuracy % Diff": "—",
                        "🆚 Precision % Diff": "—",
                        "🆚 Recall % Diff": "—",
                        "🆚 F1-Score % Diff": "—"
                    })
            
            # Add custom metrics if available
            if result.get('custom_metrics'):
                for metric_name, value in result.get('custom_metrics', {}).items():
                    if isinstance(value, (int, float)):
                        metric_display_name = f"📊 {metric_name}"
                        row_data[metric_display_name] = f"{value:.4f}"
                        
                        # Add percentage difference for custom metrics
                        if baseline and dataset_name != 'original':
                            baseline_custom = baseline.get('custom_metrics', {})
                            if metric_name in baseline_custom:
                                baseline_value = baseline_custom[metric_name]
                                if isinstance(baseline_value, (int, float)):
                                    diff_display_name = f"🆚 {metric_name} % Diff"
                                    row_data[diff_display_name] = calc_percentage_diff(value, baseline_value, False)
                                else:
                                    row_data[f"🆚 {metric_name} % Diff"] = "—"
                            else:
                                row_data[f"🆚 {metric_name} % Diff"] = "—"
                        else:
                            row_data[f"🆚 {metric_name} % Diff"] = "—"
            
            # Add algorithm-specific metrics if available
            if result.get('algorithm_specific_metrics'):
                algo_metrics = result.get('algorithm_specific_metrics', {})
                # Only include numeric algorithm metrics (filter out status messages)
                for metric_name, value in algo_metrics.items():
                    if metric_name != 'status' and isinstance(value, (int, float)):
                        algo_display_name = f"🔧 {metric_name}"
                        row_data[algo_display_name] = f"{value:.4f}"
                        
                        # Add percentage difference for algorithm metrics
                        if baseline and dataset_name != 'original':
                            baseline_algo = baseline.get('algorithm_specific_metrics', {})
                            if metric_name in baseline_algo:
                                baseline_value = baseline_algo[metric_name]
                                if isinstance(baseline_value, (int, float)):
                                    # Determine if this is an error metric (lower is better)
                                    is_error_metric = any(err_term in metric_name.lower() for err_term in 
                                                        ['error', 'loss', 'mse', 'rmse', 'mae', 'deviation'])
                                    diff_display_name = f"🆚 {metric_name} % Diff"
                                    row_data[diff_display_name] = calc_percentage_diff(value, baseline_value, is_error_metric)
                                else:
                                    row_data[f"🆚 {metric_name} % Diff"] = "—"
                            else:
                                row_data[f"🆚 {metric_name} % Diff"] = "—"
                        else:
                            row_data[f"🆚 {metric_name} % Diff"] = "—"
            
            comprehensive_data.append(row_data)
        
        # Create the comprehensive DataFrame
        if comprehensive_data:
            comprehensive_df = pd.DataFrame(comprehensive_data)
            
            # Add filtering controls
            # Model filter (NOT in columns)
            available_models = comprehensive_df['🤖 Model'].unique()
            selected_models = st.multiselect(
                "🤖 Filter by Model:",
                options=available_models,
                default=available_models,
                key="comprehensive_model_filter"
            )

            # The rest as columns
            filter_col2, filter_col3 = st.columns(2)
            
            with filter_col2:
                # Dataset type filter
                available_types = comprehensive_df['🔒 Type'].unique()
                selected_types = st.multiselect(
                    "🔒 Filter by Type:",
                    options=available_types,
                    default=available_types,
                    key="comprehensive_type_filter"
                )
            
            with filter_col3:
                # Improvement threshold slider
                improvement_threshold = st.slider(
                    "📊 Show results with |change| ≥",
                    min_value=0.0,
                    max_value=20.0,
                    value=0.0,
                    step=0.5,
                    key="comprehensive_threshold_filter",
                    help="Filter to show only results with percentage changes above this threshold"
                )
            # NEW: Column Selection Controls
            st.markdown("---")
            st.markdown("**📋 Column Selection & Display Options:**")
            
            # Organize columns by category for better UX
            all_columns = list(comprehensive_df.columns)
            
            # Categorize columns
            core_columns = []
            base_metrics = []
            percentage_diffs = []
            custom_metrics = []
            algorithm_metrics = []
            
            for col in all_columns:
                if col in ["🗂️ Dataset", "🔒 Type", "🤖 Model", "📊 Privacy Method"]:
                    core_columns.append(col)
                elif "% Diff" in col:
                    percentage_diffs.append(col)
                elif col.startswith("📊 ") and "% Diff" not in col:
                    custom_metrics.append(col)
                elif col.startswith("🔧 ") and "% Diff" not in col:
                    algorithm_metrics.append(col)
                elif any(base_metric in col for base_metric in ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'R² Score', 'RMSE', 'MAE', 'MAPE']):
                    base_metrics.append(col)
            
            # Column selection interface with expandable sections
            col_selection_tab1, col_selection_tab2 = st.tabs(["🎯 Quick Selection", "🔧 Advanced Selection"])
            
            with col_selection_tab1:
                st.markdown("**⚡ Quick Column Presets:**")
                
                preset_col1, preset_col2, preset_col3, preset_col4 = st.columns(4)
                
# Replace the preset button section (around lines 6415-6435) with corrected logic:

                with preset_col1:
                    if st.button("📊 Core Only", key="preset_core_only", use_container_width=True):
                        selected_columns = core_columns + base_metrics
                        st.session_state.comprehensive_selected_columns = selected_columns
                        st.rerun()
                
# Replace the "With Comparisons" preset button (around lines 6437-6465) with this FIXED version:

                with preset_col2:
                    if st.button("📈 With Comparisons", key="preset_with_comparisons", use_container_width=True):
                        # FIXED: Create proper paired ordering (metric → diff → metric → diff)
                        selected_columns = core_columns.copy()
                        
                        # Add base metrics with their percentage differences in paired order
                        for metric in base_metrics:
                            selected_columns.append(metric)
                            
                            # FIXED: Direct mapping for all metrics to ensure pairing
                            diff_col = None
                            
                            if "R²" in metric or "R2" in metric:
                                diff_col = "🆚 R² % Diff"
                            elif "F1-Score" in metric or "F1 Score" in metric:
                                diff_col = "🆚 F1-Score % Diff"
                            elif "Accuracy" in metric:
                                diff_col = "🆚 Accuracy % Diff"
                            elif "Precision" in metric:
                                diff_col = "🆚 Precision % Diff"
                            elif "Recall" in metric:
                                diff_col = "🆚 Recall % Diff"
                            elif "RMSE" in metric:
                                diff_col = "🆚 RMSE % Diff"
                            elif "MAE" in metric:
                                diff_col = "🆚 MAE % Diff"
                            elif "MAPE" in metric:
                                diff_col = "🆚 MAPE % Diff"
                            else:
                                # Fallback: extract last word and create expected diff column name
                                metric_short_name = metric.split()[-1].replace(':', '')
                                diff_col = f"🆚 {metric_short_name} % Diff"
                            
                            # Add the diff column immediately after the metric if it exists
                            if diff_col and diff_col in percentage_diffs:
                                selected_columns.append(diff_col)
                        
                        st.session_state.comprehensive_selected_columns = selected_columns
                        st.rerun()
                        
                with preset_col3:
                    # REMOVED: Algorithm Focus preset as it was confusing
                    if st.button("🔧 Custom Metrics", key="preset_custom_focus", use_container_width=True):
                        # NEW: Focus on custom and algorithm metrics with their comparisons
                        selected_columns = core_columns + custom_metrics + algorithm_metrics
                        # Add relevant percentage differences
                        for col in custom_metrics + algorithm_metrics:
                            metric_name = col.replace("📊 ", "").replace("🔧 ", "")
                            diff_col = f"🆚 {metric_name} % Diff"
                            if diff_col in percentage_diffs:
                                selected_columns.append(diff_col)
                        st.session_state.comprehensive_selected_columns = selected_columns
                        st.rerun()
                
                with preset_col4:
                    if st.button("🌐 Everything", key="preset_everything", use_container_width=True):
                        selected_columns = all_columns
                        st.session_state.comprehensive_selected_columns = selected_columns
                        st.rerun()

            with col_selection_tab2:
                st.markdown("**🎛️ Detailed Column Selection:**")
                
                # Store current selections in temporary variables (not session state)
                # This prevents auto-refresh while user is selecting
                
                # Advanced column selection with categories
                adv_col1, adv_col2 = st.columns(2)
                
                with adv_col1:
                    st.markdown("**📋 Essential Columns:**")
                    
                    # Core columns (always recommended)
                    current_selection = st.session_state.get('comprehensive_selected_columns', all_columns)
                    filtered_core_defaults = [col for col in current_selection if col in core_columns]
                    
                    temp_selected_core = st.multiselect(
                        "🎯 Core Information (Dataset, Model, etc.):",
                        options=core_columns,
                        default=filtered_core_defaults,
                        key="temp_comprehensive_core_columns",  # CHANGED: Use temp key
                        help="Essential identification columns - highly recommended to keep"
                    )
                    
                    # Base metrics
                    if base_metrics:
                        filtered_base_defaults = [col for col in current_selection if col in base_metrics]
                        
                        temp_selected_base_metrics = st.multiselect(
                            "📊 Base Performance Metrics:",
                            options=base_metrics,
                            default=filtered_base_defaults,
                            key="temp_comprehensive_base_metrics",  # CHANGED: Use temp key
                            help="Core ML performance metrics (Accuracy, F1-Score, R², RMSE, etc.)"
                        )
                    else:
                        temp_selected_base_metrics = []
                    
                    # Percentage differences
                    if percentage_diffs:
                        filtered_percentage_defaults = [col for col in current_selection if col in percentage_diffs]
                        
                        temp_selected_percentage_diffs = st.multiselect(
                            "🆚 Percentage Comparisons:",
                            options=percentage_diffs,
                            default=filtered_percentage_defaults,
                            key="temp_comprehensive_percentage_diffs",  # CHANGED: Use temp key
                            help="Percentage changes compared to original dataset"
                        )
                    else:
                        temp_selected_percentage_diffs = []
                
                with adv_col2:
                    st.markdown("**🔧 Extended Metrics:**")
                    
                    # Custom metrics
                    if custom_metrics:
                        filtered_custom_defaults = [col for col in current_selection if col in custom_metrics]
                        
                        temp_selected_custom_metrics = st.multiselect(
                            "📊 Custom Evaluation Metrics:",
                            options=custom_metrics,
                            default=filtered_custom_defaults,
                            key="temp_comprehensive_custom_metrics",  # CHANGED: Use temp key
                            help="Additional evaluation metrics you selected"
                        )
                    else:
                        temp_selected_custom_metrics = []
                        st.info("ℹ️ No custom metrics available")
                    
                    # Algorithm-specific metrics
                    if algorithm_metrics:
                        filtered_algorithm_defaults = [col for col in current_selection if col in algorithm_metrics]
                        
                        temp_selected_algorithm_metrics = st.multiselect(
                            "🔧 Algorithm-Specific Metrics:",
                            options=algorithm_metrics,
                            default=filtered_algorithm_defaults,
                            key="temp_comprehensive_algorithm_metrics",  # CHANGED: Use temp key
                            help="Metrics specific to each algorithm implementation"
                        )
                    else:
                        temp_selected_algorithm_metrics = []
                        st.info("ℹ️ No algorithm-specific metrics available")
                
                # ADDED: Apply button section
                """
                Advanced Column Selection User Interface
                
                This section provides a comprehensive column selection and preview system
                for academic research data presentation. Essential for thesis-quality
                data visualization and comparative analysis.
                
                Academic Research Applications:
                - Selective metric presentation for focused analysis
                - Dynamic data filtering for research presentations
                - Professional table customization for publication standards
                - Interactive data exploration for academic review
                """
                
                st.markdown("---")
                
                # Aggregate all temporary column selections for comprehensive preview
                all_temp_selected_columns = (temp_selected_core + temp_selected_base_metrics + 
                                          temp_selected_percentage_diffs + temp_selected_custom_metrics + 
                                          temp_selected_algorithm_metrics)
                
                """
                Selection Preview and Validation System
                
                Provides real-time feedback on column selection choices with
                categorical breakdown for academic presentation quality assurance.
                Essential for maintaining thesis documentation standards.
                """
                # Two-column layout for selection preview and control interface
                selection_preview_col1, selection_preview_col2 = st.columns(2)
                
                with selection_preview_col1:
                    """
                    Real-time Selection Summary Dashboard
                    
                    Displays comprehensive breakdown of selected columns by category,
                    enabling researchers to validate their data presentation choices
                    before applying changes to the analysis view.
                    """
                    st.markdown("**📊 Selection Preview:**")
                    # Quantitative summary of total column selection
                    st.info(f"✅ {len(all_temp_selected_columns)} columns selected")
                    
                    # Categorical breakdown for academic transparency
                    breakdown_text = []
                    if temp_selected_core:
                        breakdown_text.append(f"🎯 Core: {len(temp_selected_core)}")           # Essential identification columns
                    if temp_selected_base_metrics:
                        breakdown_text.append(f"📊 Base: {len(temp_selected_base_metrics)}")   # Primary performance metrics
                    if temp_selected_percentage_diffs:
                        breakdown_text.append(f"🆚 Comparisons: {len(temp_selected_percentage_diffs)}")  # Comparative analysis columns
                    if temp_selected_custom_metrics:
                        breakdown_text.append(f"📊 Custom: {len(temp_selected_custom_metrics)}")  # User-defined evaluation metrics
                    if temp_selected_algorithm_metrics:
                        breakdown_text.append(f"🔧 Algorithm: {len(temp_selected_algorithm_metrics)}")  # Implementation-specific metrics
                    
                    # Display categorical breakdown with professional formatting
                    if breakdown_text:
                        st.caption(" | ".join(breakdown_text))
                
                with selection_preview_col2:
                    """
                    Column Selection Application Controls
                    
                    Provides user interface for applying temporary selections to the main
                    data view, ensuring controlled and validated changes to the analysis
                    presentation. Critical for maintaining academic workflow integrity.
                    """
                    # Professional action button layout for academic interface standards
                    apply_col1, apply_col2 = st.columns([3, 1])
                    
                    with apply_col1:
                        """
                        Primary Selection Application Interface
                        
                        Validates and applies the temporary column selections to the session state,
                        ensuring persistence across user interactions and maintaining analysis
                        continuity for academic research workflows.
                        """
                        if st.button("Apply Column Selection", 
                                   key="apply_column_selection", 
                                   type="primary", 
                                   use_container_width=True,
                                   help="Apply your column selection to the table"):
                            
                            # Store validated selections in persistent session state for academic continuity
                            st.session_state.comprehensive_selected_columns = all_temp_selected_columns
                            st.success(f"✅ Applied! Now showing {len(all_temp_selected_columns)} columns")
                            st.rerun()  # Refresh interface to reflect applied changes
                    
                    with apply_col2:
                        """
                        Selection Reset Functionality
                        
                        Provides quick reset capability to restore full column visibility,
                        ensuring researchers can easily return to comprehensive data view
                        when needed for complete analysis validation.
                        """
                        if st.button("Reset", 
                                   key="reset_column_selection", 
                                   help="Reset to show all columns"):
                            # Restore complete column visibility for comprehensive analysis
                            st.session_state.comprehensive_selected_columns = all_columns
                            st.info("🔄 Reset to show all columns")
                            st.rerun()  # Refresh interface to reflect reset state

            
            # Apply validated column selection with academic presentation standards
            if 'comprehensive_selected_columns' in st.session_state and st.session_state.comprehensive_selected_columns:
                # Filter DataFrame to validated selected columns for academic analysis
                selected_columns = [col for col in st.session_state.comprehensive_selected_columns if col in comprehensive_df.columns]
                
                if selected_columns:

                    # Apply sophisticated column ordering based on user preference
                    if st.session_state.get('comprehensive_column_order', 'Default') == 'Alphabetical':
                        """
                        Alphabetical Ordering with Core Column Preservation
                        
                        Maintains essential identification columns at the beginning while
                        alphabetizing remaining columns for consistent academic presentation.
                        """
                        # Preserve core columns for academic context, alphabetize others
                        core_selected = [col for col in selected_columns if col in core_columns]
                        other_selected = [col for col in selected_columns if col not in core_columns]
                        other_selected.sort()                                                    # Lexicographic ordering
                        selected_columns = core_selected + other_selected
                        
                    elif st.session_state.get('comprehensive_column_order', 'Default') == 'By Type':
                        """
                        Categorical Type-Based Ordering
                        
                        Groups columns by functional category (core, metrics, comparisons, etc.)
                        for logical academic presentation and enhanced research comprehension.
                        """
                        # Group by academic functional category
                        selected_columns = []
                        for column_group in [core_columns, base_metrics, percentage_diffs, custom_metrics, algorithm_metrics]:
                            selected_columns.extend([col for col in column_group if col in st.session_state.comprehensive_selected_columns])
                            
                    elif st.session_state.get('comprehensive_column_order', 'Default') == 'By Importance':
                        """
                        Academic Importance-Based Ordering
                        
                        Prioritizes columns by research relevance: Core identification → Primary metrics 
                        → Comparative analysis → Extended evaluation → Implementation-specific data.
                        """
                        # Order by academic research importance hierarchy
                        importance_order = core_columns + base_metrics + percentage_diffs + custom_metrics + algorithm_metrics
                        selected_columns = [col for col in importance_order if col in st.session_state.comprehensive_selected_columns]
                    
                    else:  # Default ordering - Advanced paired metric-comparison algorithm

                        # Initialize ordered column collection for academic presentation
                        ordered_columns = []
                        
                        # Phase 1: Add core identification columns for academic context
                        for col in core_columns:
                            if col in selected_columns:
                                ordered_columns.append(col)
                        
                        # Phase 2: Advanced metric-comparison pairing for base performance metrics
                        for metric in base_metrics:
                            if metric in selected_columns:
                                ordered_columns.append(metric)                                    # Add primary metric
                                
                                # Sophisticated percentage difference column matching algorithm
                                diff_col = None
                                
                                # Specialized handling for mathematical notation and metric variations
                                if "R²" in metric or "R2" in metric:
                                    diff_col = "🆚 R² % Diff"                                    # R-squared coefficient matching
                                elif "F1-Score" in metric or "F1 Score" in metric:
                                    diff_col = "🆚 F1-Score % Diff"                              # F1-score harmonic mean matching
                                elif "Accuracy" in metric:
                                    diff_col = "🆚 Accuracy % Diff"                              # Classification accuracy matching
                                elif "Precision" in metric:
                                    diff_col = "🆚 Precision % Diff"                             # Precision metric matching
                                elif "Recall" in metric:
                                    diff_col = "🆚 Recall % Diff"                                # Recall/sensitivity matching
                                elif "RMSE" in metric:
                                    diff_col = "🆚 RMSE % Diff"                                  # Root mean square error matching
                                elif "MAE" in metric:
                                    diff_col = "🆚 MAE % Diff"                                   # Mean absolute error matching
                                elif "MAPE" in metric:
                                    diff_col = "🆚 MAPE % Diff"                                  # Mean absolute percentage error matching
                                else:
                                    # Advanced fallback algorithm for complex metric name variations
                                    metric_short_name = metric.split()[-1].replace(':', '')      # Extract base metric name
                                    possible_diff_cols = [
                                        f"🆚 {metric_short_name} % Diff",                        # Direct name matching
                                        f"🆚 {metric_short_name.replace('²', '2')} % Diff",      # Mathematical notation normalization
                                        f"🆚 {metric_short_name.replace('-', ' ')} % Diff"       # Hyphen handling for compound names
                                    ]
                                    
                                    # Iterative matching for robust metric-diff association
                                    for possible_diff_col in possible_diff_cols:
                                        if possible_diff_col in selected_columns:
                                            diff_col = possible_diff_col
                                            break                                                 # Use first successful match
                                
                                # Add paired comparison column immediately after metric for academic clarity
                                if diff_col and diff_col in selected_columns and diff_col not in ordered_columns:
                                    ordered_columns.append(diff_col)
                        
                        # Phase 3: Custom metric pairing for user-defined evaluation criteria
                        for metric in custom_metrics:
                            if metric in selected_columns:
                                ordered_columns.append(metric)
                                
                                # Extract metric name and find corresponding percentage difference
                                metric_name = metric.replace("📊 ", "")                          # Remove emoji prefix
                                diff_col = f"🆚 {metric_name} % Diff"
                                if diff_col in selected_columns and diff_col not in ordered_columns:
                                    ordered_columns.append(diff_col)
                        
                        # Phase 4: Algorithm-specific metric pairing for implementation analysis
                        for metric in algorithm_metrics:
                            if metric in selected_columns:
                                ordered_columns.append(metric)
                                
                                # Extract algorithm metric name and find corresponding comparison
                                metric_name = metric.replace("🔧 ", "")                          # Remove technical emoji prefix
                                diff_col = f"🆚 {metric_name} % Diff"
                                if diff_col in selected_columns and diff_col not in ordered_columns:
                                    ordered_columns.append(diff_col)
                        
                        # Phase 5: Orphan column integration for comprehensive coverage
                        for col in selected_columns:
                            if col not in ordered_columns:
                                ordered_columns.append(col)                                      # Ensure no column is excluded
                        
                        # Apply final ordering for academic presentation
                        selected_columns = ordered_columns
                    
                    # Create filtered DataFrame with academic presentation standards
                    display_df = comprehensive_df[selected_columns]
                    
                    # Professional selection summary for thesis documentation
                    st.info(f"📊 Displaying {len(selected_columns)} of {len(comprehensive_df.columns)} available columns")
                    
                else:
                    # Validation error handling for academic workflow continuity
                    st.warning("⚠️ No valid columns selected. Showing all columns.")
                    display_df = comprehensive_df
                    selected_columns = list(comprehensive_df.columns)
            else:
                # Default comprehensive view for initial academic analysis
                display_df = comprehensive_df
                selected_columns = list(comprehensive_df.columns)
            
            
            # Apply comprehensive multi-criteria filtering for academic analysis
            filtered_df = display_df[
                (display_df['🤖 Model'].isin(selected_models)) &                              # Model selection filter
                (display_df['🔒 Type'].isin(selected_types))                                   # Dataset type filter
            ]
            
            # Advanced threshold-based filtering for significant change analysis
            if improvement_threshold > 0:
                """
                Significance Threshold Analysis Filter
                
                Implements statistical significance filtering based on percentage change
                thresholds, enabling researchers to focus on experimentally significant
                results for academic analysis and thesis documentation.
                """
                # Identify all percentage difference columns in the selected view
                pct_diff_columns = [col for col in selected_columns if "% Diff" in col]
                
                def meets_threshold(row):
                    """
                    Statistical Significance Evaluation Function
                    
                    Analyzes each row to determine if any percentage difference meets
                    the specified threshold for academic significance analysis.
                    
                    Args:
                        row: DataFrame row containing performance metrics and differences
                    
                    Returns:
                        bool: True if row contains at least one significant change
                    """
                    for col in pct_diff_columns:
                        if col in row.index:  # Validate column existence for robust analysis
                            val_str = str(row[col])
                            # Parse directional indicators (▲ increase, ▼ decrease)
                            if val_str != "—" and any(arrow in val_str for arrow in ['▲', '▼']):
                                try:
                                    # Extract numeric percentage value for threshold comparison
                                    numeric_part = val_str.split()[1].rstrip('%')               # Remove percentage symbol
                                    numeric_value = abs(float(numeric_part))                    # Absolute value for significance
                                    if numeric_value >= improvement_threshold:
                                        return True                                             # Meets academic significance threshold
                                except (IndexError, ValueError):
                                    continue                                                    # Skip malformed data gracefully
                    return False                                                                # No significant changes detected
                
                # Apply threshold filter with original data preservation for baseline comparison
                if improvement_threshold > 0:
                    # Maintain original datasets for academic baseline comparison
                    mask = filtered_df.apply(meets_threshold, axis=1) | (filtered_df['🔒 Type'] == 'Original')
                    filtered_df = filtered_df[mask]
            
            # Apply display density styling
            density_styles = {
                "Compact": {'padding': '6px', 'font-size': '11px'},
                "Normal": {'padding': '8px', 'font-size': '12px'},
                "Spacious": {'padding': '12px', 'font-size': '13px'}
            }
            
            current_density = st.session_state.get('comprehensive_display_density', 'Normal')
            density_style = density_styles[current_density]
            
            # Helper function to color arrows in the comprehensive table
            def color_comprehensive_arrows(val_str):
                if isinstance(val_str, str):
                    if val_str.startswith("▲"):
                        return 'color: green; font-weight: bold'
                    elif val_str.startswith("▼"):
                        return 'color: red; font-weight: bold'
                    elif val_str.startswith("►"):
                        return 'color: gray; font-weight: normal'
                return ''
            
            # Style the comprehensive table with density options
            styled_comprehensive = filtered_df.style.set_properties(**{
                'padding': density_style['padding'],
                'font-size': density_style['font-size'],
                'text-align': 'center',
                'border': '1px solid var(--text-color-secondary)',
                'font-weight': 'bold'
            }).set_table_styles([
                {'selector': 'th', 'props': [
                    ('background-color', '#dc3545'),  # Red header for comprehensive view
                    ('color', 'white'),
                    ('font-weight', 'bold'),
                    ('text-align', 'center'),
                    ('padding', '10px'),
                    ('border', '1px solid #dc3545'),
                    ('font-size', density_style['font-size'])
                ]},
                {'selector': 'td', 'props': [
                    ('text-align', 'center'),
                    ('vertical-align', 'middle'),
                    ('border', '1px solid var(--text-color-secondary)')
                ]},
                {'selector': '', 'props': [
                    ('border-collapse', 'collapse'),
                    ('margin', '15px 0'),
                    ('border-radius', '6px'),
                    ('overflow', 'hidden'),
                    ('box-shadow', '0 2px 4px rgba(0, 0, 0, 0.1)')
                ]}
            ])
            
            # Apply coloring to all percentage difference columns in the filtered dataframe
            pct_diff_columns = [col for col in filtered_df.columns if "% Diff" in col]
            if pct_diff_columns:
                def color_all_comprehensive_arrows(row):
                    styled_row = [''] * len(row)
                    for i, (col_name, val) in enumerate(zip(row.index, row.values)):
                        if col_name in pct_diff_columns:
                            styled_row[i] = color_comprehensive_arrows(val)
                    return styled_row
                
                styled_comprehensive = styled_comprehensive.apply(color_all_comprehensive_arrows, axis=1)
            
            # Display the table
            st.dataframe(styled_comprehensive, use_container_width=True, hide_index=True)
            
            # Enhanced summary statistics
            st.markdown("**📊 Table Summary:**")
            summary_col1, summary_col2, summary_col3, summary_col4 = st.columns(4)
            
            with summary_col1:
                st.metric("📊 Displayed Rows", len(filtered_df), "Experiment results")
            
            with summary_col2:
                original_count = len(filtered_df[filtered_df['🔒 Type'] == 'Original']) if '🔒 Type' in filtered_df.columns else 0
                st.metric("🗂️ Original", original_count, "Baseline results")
            
            with summary_col3:
                anonymized_count = len(filtered_df[filtered_df['🔒 Type'] == 'Anonymized']) if '🔒 Type' in filtered_df.columns else 0
                st.metric("🔒 Anonymized", anonymized_count, "Privacy-preserved")
            
            with summary_col4:
                unique_methods = len(filtered_df['📊 Privacy Method'].unique()) if '📊 Privacy Method' in filtered_df.columns else 0
                st.metric("🔧 Methods", unique_methods, "Privacy techniques")

            # ============================================================================
            # PROFESSIONAL THESIS-QUALITY VISUALIZATION SECTION
            # ============================================================================
            st.markdown("---")
            st.markdown("#### 📊 **Professional Thesis Visualization Suite**")
            st.markdown("*Generate publication-ready charts for your research*")
            
            # Import PURE visualization integration (EXACT original functions)
            try:
                from visualization_integration_pure import pure_viz_integrator
                VISUALIZATION_INTEGRATION_AVAILABLE = True
            except ImportError as e:
                st.error(f"❌ Pure visualization integration not available: {e}")
                VISUALIZATION_INTEGRATION_AVAILABLE = False
            
            if VISUALIZATION_INTEGRATION_AVAILABLE:
                # Create professional control panel
                viz_control_container = st.container()
                
                with viz_control_container:
                    # Main control buttons
                    col1, col2, col3 = st.columns([2, 2, 1])
                    
                    with col1:
                        show_visualizations = st.button(
                            "👁️ **Show Visualizations**",
                            type="primary",
                            use_container_width=True,
                            help="Display professional charts within the app for interactive analysis",
                            key="show_thesis_visualizations"
                        )
                    
                    with col2:
                        export_png_files = st.button(
                            "💾 **Export as PNG Files**",
                            type="secondary", 
                            use_container_width=True,
                            help="Generate and download publication-ready PNG files for thesis inclusion",
                            key="export_thesis_pngs"
                        )
                    
                    with col3:
                        st.markdown("**📋 Status:**")
                        if not filtered_df.empty:
                            st.success(f"✅ {len(filtered_df)} rows ready")
                        else:
                            st.warning("⚠️ No data available")
                
                # Anonymization level configuration
                with st.expander("⚙️ **Anonymization Level Configuration**", expanded=False):
                    st.markdown("**Configure how dataset names map to anonymization levels:**")
                    
                    # Get unique privacy methods from the data
                    if '📊 Privacy Method' in filtered_df.columns:
                        unique_methods = filtered_df['📊 Privacy Method'].unique()
                        
                        st.markdown("**Current Privacy Methods in Data:**")
                        mapping_configs = {}
                        
                        col_config1, col_config2, col_config3 = st.columns(3)
                        
                        for i, method in enumerate(unique_methods):
                            with [col_config1, col_config2, col_config3][i % 3]:
                                # Default mapping
                                method_clean = str(method).lower().replace(' ', '_')
                                default_level = 'Original' if 'original' in method_clean else 'Medium'
                                
                                level = st.selectbox(
                                    f"**{method}**",
                                    options=['Original', 'Minimal', 'Medium', 'High'],
                                    index=['Original', 'Minimal', 'Medium', 'High'].index(default_level),
                                    key=f"anon_level_{method}"
                                )
                                # Use original method name as key for proper mapping
                                mapping_configs[method] = level
                        
                        # Update the anonymization level mapping
                        if mapping_configs:
                            # Store the mapping in session state for use by visualization system
                            st.session_state['anonymization_configs'] = mapping_configs
                    else:
                        st.info("No privacy method column found in the data.")
                
                # NEW: Dataset Name Configuration
                with st.expander("🏷️ **Dataset Name Configuration**", expanded=True):
                    st.markdown("**Specify the dataset name for visualizations:**")
                    
                    # Dataset name input with smart default
                    default_name = "Current Dataset"
                    if '🗂️ Dataset' in filtered_df.columns and not filtered_df.empty:
                        # Try to get dataset name from data
                        unique_datasets = filtered_df['🗂️ Dataset'].unique()
                        if len(unique_datasets) == 1:
                            default_name = str(unique_datasets[0])
                        elif len(unique_datasets) > 1:
                            default_name = f"Multi-Dataset Analysis ({len(unique_datasets)} datasets)"
                    
                    dataset_name = st.text_input(
                        "Dataset Name (will appear in visualization titles):",
                        value=default_name,
                        help="This name will be used in all visualization titles and file names",
                        key="viz_dataset_name"
                    )
                    
                    st.info(f"📊 **Visualization titles will use:** '{dataset_name}'")
                
                # NEW: Metric Selection for Visualizations
                with st.expander("📊 **Metric Selection for Visualizations**", expanded=True):
                    st.markdown("**Select which metrics to include in visualizations:**")
                    
                    # Get all numeric columns (excluding identifiers)
                    identifier_cols = ['🗂️ Dataset', '🔒 Type', '🤖 Model', '📊 Privacy Method']
                    numeric_cols = []
                    
                    for col in filtered_df.columns:
                        if col not in identifier_cols:
                            # Check if column contains numeric data (even if mixed with arrows/percentages)
                            sample_values = filtered_df[col].dropna().head(10)
                            has_numeric = any(
                                bool(re.search(r'\d+\.?\d*', str(val))) 
                                for val in sample_values
                            )
                            if has_numeric:
                                numeric_cols.append(col)
                    
                    if numeric_cols:
                        # Categorize metrics for better organization
                        core_metrics = []
                        custom_metrics = []
                        
                        for col in numeric_cols:
                            col_lower = col.lower()
                            if any(term in col_lower for term in ['accuracy', 'precision', 'recall', 'f1', 'r²', 'rmse', 'mae']):
                                core_metrics.append(col)
                            else:
                                custom_metrics.append(col)
                        
                        # Metric selection interface
                        metric_col1, metric_col2 = st.columns(2)
                        
                        with metric_col1:
                            st.markdown("**🎯 Core Performance Metrics:**")
                            selected_core = st.multiselect(
                                "Select core metrics:",
                                options=core_metrics,
                                default=core_metrics[:4] if len(core_metrics) >= 4 else core_metrics,
                                key="viz_core_metrics"
                            )
                        
                        with metric_col2:
                            st.markdown("**📊 Custom/Extended Metrics:**")
                            selected_custom = st.multiselect(
                                "Select custom metrics:",
                                options=custom_metrics,
                                default=[],
                                key="viz_custom_metrics"
                            )
                        
                        # Combine selected metrics
                        selected_metrics = selected_core + selected_custom
                        
                        st.info(f"✅ **{len(selected_metrics)} metrics selected** for visualization: {', '.join(selected_metrics[:3])}{'...' if len(selected_metrics) > 3 else ''}")
                        
                    else:
                        st.warning("⚠️ No numeric columns found for visualization.")
                        selected_metrics = []
                
                # Handle visualization display with PURE original functions
                if show_visualizations:
                    if selected_metrics:
                        with st.spinner("🎨 Generating visualizations using EXACT original functions..."):
                            # Get anonymization configurations and dataset name
                            anonymization_configs = st.session_state.get('anonymization_configs', {})
                            dataset_name = st.session_state.get('viz_dataset_name', 'Current Dataset')
                            
                            # Extract data with proper label mapping using PURE system
                            results = pure_viz_integrator.extract_table_data_to_results_format(
                                filtered_df, selected_metrics, anonymization_configs, dataset_name
                            )
                            
                            if results:
                                # Generate ALL visualizations using EXACT original functions
                                visualization_files = pure_viz_integrator.generate_all_visualizations(
                                    results, dataset_name, selected_metrics
                                )
                                
                                if visualization_files:
                                    st.success("✅ **Visualizations generated using EXACT original visualization_summary.py functions!**")
                                    st.markdown("---")
                                    
                                    # Display tab-based visualizations
                                    pure_viz_integrator.display_visualizations_with_tabs(visualization_files, selected_metrics)
                                    
                                    # Create and offer download package
                                    st.markdown("---")
                                    st.markdown("### 📦 **Download Complete Package**")
                                    
                                    zip_path = pure_viz_integrator.create_download_package(visualization_files, dataset_name)
                                    if zip_path and os.path.exists(zip_path):
                                        with open(zip_path, 'rb') as f:
                                            zip_data = f.read()
                                        
                                        col1, col2, col3 = st.columns([1, 2, 1])
                                        with col2:
                                            st.download_button(
                                                label=f"📥 **Download {dataset_name} Visualizations (ZIP)**",
                                                data=zip_data,
                                                file_name=f"{dataset_name.replace(' ', '_')}_visualizations_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip",
                                                mime="application/zip",
                                                type="primary",
                                                use_container_width=True
                                            )
                                        
                                        st.success("✅ **Download package ready!** Contains all visualizations using EXACT original functions with Porsche Next TT font and cleaned metric names.")
                                else:
                                    st.error("❌ **Failed to generate visualizations.** Please check your data format.")
                            else:
                                st.error("❌ **No valid data extracted.** Please check your table structure and metric selection.")
                    else:
                        st.warning("⚠️ **Please select at least one metric** in the 'Metric Selection' section above.")
                
                # Handle PNG export with PURE original functions
                if export_png_files:
                    if selected_metrics:
                        with st.spinner("📸 Creating PNG exports using EXACT original functions..."):
                            # Get anonymization configurations and dataset name
                            anonymization_configs = st.session_state.get('anonymization_configs', {})
                            dataset_name = st.session_state.get('viz_dataset_name', 'Current Dataset')
                            
                            # Extract data with proper label mapping using PURE system
                            results = pure_viz_integrator.extract_table_data_to_results_format(
                                filtered_df, selected_metrics, anonymization_configs, dataset_name
                            )
                            
                            if results:
                                # Generate ALL visualizations using EXACT original functions
                                visualization_files = pure_viz_integrator.generate_all_visualizations(
                                    results, dataset_name, selected_metrics
                                )
                                
                                if visualization_files:
                                    # Create download package
                                    zip_path = pure_viz_integrator.create_download_package(visualization_files, dataset_name)
                                    
                                    if zip_path and os.path.exists(zip_path):
                                        # Create download button for ZIP file
                                        with open(zip_path, 'rb') as f:
                                            zip_data = f.read()
                                        
                                        st.download_button(
                                            label=f"📥 **Download {dataset_name} Visualization Suite (ZIP)**",
                                            data=zip_data,
                                            file_name=f"{dataset_name.replace(' ', '_')}_visualizations_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip",
                                            mime="application/zip",
                                            type="primary",
                                            use_container_width=True
                                        )
                                        
                                        st.success("✅ **Pure visualization package created successfully!**")
                                        st.info(f"""
                                        📊 **Export Summary:** 
                                        - All visualizations using EXACT original visualization_summary.py functions
                                        - Cleaned metric names (no icons/emojis)
                                        - Proper anonymization labels: Method + Configuration Level
                                        - Horizontal labels throughout
                                        - {len(selected_metrics)} metrics analyzed for '{dataset_name}'
                                        - Porsche Next TT font applied
                                        - 300 DPI publication-ready quality
                                        """)
                                    else:
                                        st.error("❌ **Failed to create download package.**")
                                else:
                                    st.error("❌ **Failed to generate visualizations for export.**")
                            else:
                                st.error("❌ **No valid data extracted for export.**")
                    else:
                        st.warning("⚠️ **Please select at least one metric** in the 'Metric Selection' section above.")
                
                # Professional usage instructions
                with st.expander("📖 **Pure Visualization_Summary Integration Guide**", expanded=False):
                    st.markdown("""
                    #### 🎯 **Complete Professional Visualization System**
                    
                    **Tab-Based Interactive Analysis:**
                    - Use **Show Visualizations** for comprehensive interactive analysis
                    - **7 Professional Visualization Categories** organized in tabs:
                      - 📊 **Metric Performance:** Individual metric analysis with bar charts
                      - 🔥 **Impact Heatmap:** Comprehensive anonymization impact matrix
                      - 📉 **Performance Degradation:** Track performance changes across techniques
                      - 📈 **Percentage Analysis:** Quantified performance loss percentages
                      - 🤖 **Model Impact:** Model robustness comparison analysis  
                      - 🏆 **Technique Ranking:** Anonymization technique effectiveness ranking
                      - ⚖️ **Privacy vs Utility:** Trade-off analysis with scatter plots
                    
                    **Professional Export Features:**
                    - **Dual-Size System:** Half-size for app display, full-size for download
                    - **Porsche Next TT Font:** Applied to all visualizations for brand consistency
                    - **300 DPI Quality:** Publication-ready high-resolution exports
                    - **Horizontal Labels:** Improved readability as requested
                    - **Complete Package:** All visualizations in organized ZIP structure
                    
                    #### 🔧 **Advanced Label Mapping System**
                    - **Smart Privacy Method Detection:** Automatically extracts from table columns
                    - **Anonymization Level Integration:** Uses your configuration from ⚙️ section above
                    - **Proper Format Generation:** Creates "Differential Privacy (High)" style labels
                    - **Exact Original Logic:** Uses same calculation functions as visualization_summary.py
                    
                    #### � **Complete Visualization Suite Includes:**
                    1. **Individual Metric Charts** - Bar charts for each selected metric
                    2. **Anonymization Impact Heatmap** - Color-coded performance matrix
                    3. **Performance Degradation Analysis** - Line plots showing degradation trends
                    4. **Percentage Degradation** - Quantified performance loss analysis
                    5. **Model Impact Analysis** - Box plots comparing model robustness
                    6. **Technique Ranking** - Bar chart ranking anonymization effectiveness
                    7. **Privacy-Utility Trade-off** - Scatter plot analysis of trade-offs
                    
                    #### 📁 **Export Package Structure**
                    ```
                    complete_thesis_visualizations_YYYYMMDD_HHMMSS.zip
                    ├── accuracy_performance_chart.png
                    ├── precision_performance_chart.png
                    ├── anonymization_impact_heatmap.png
                    ├── performance_degradation.png
                    ├── percentage_degradation.png
                    ├── model_impact_analysis.png
                    ├── technique_ranking.png
                    └── privacy_utility_tradeoff.png
                    ```
                    
                    #### ⚙️ **Configuration Requirements:**
                    - **Select Metrics:** Choose core and custom metrics above
                    - **Configure Anonymization Levels:** Set mapping in ⚙️ section above
                    - **Data Structure:** Ensure table has Privacy Method and Model columns
                    - **Horizontal Layout:** All labels optimized for horizontal display
                    """)
            else:
                st.error("❌ **Pure visualization integration not available.** Please check if `visualization_integration_pure.py` is in the correct directory.")
            
            # END PROFESSIONAL VISUALIZATION SECTION
            # ============================================================================

            # --- ENHANCED PROFESSIONAL INSIGHTS SECTION ---
            st.markdown("---")
            st.markdown("**🔎 Professional PPML Insights & Analysis**")
            if not filtered_df.empty:
                # Create professional tabs for different types of analysis
                insights_tab1, insights_tab2, insights_tab3, insights_tab4 = st.tabs([
                    "📊 Performance Analysis", 
                    "🔒 Privacy Impact Assessment", 
                    "⚖️ Utility-Privacy Trade-offs",
                    "💼 Business Recommendations"
                ])
                
                # Determine task type for context-aware analysis
                is_regression = "🎯 R² Score" in filtered_df.columns
                is_classification = "🏆 F1-Score" in filtered_df.columns
                task_type_name = "Regression" if is_regression else "Classification"
                
                with insights_tab1:
                    st.markdown(f"### 📊 **{task_type_name} Performance Analysis**")
                    
                    # Performance metrics analysis
                    if is_regression:
                        # REGRESSION ANALYSIS
                        r2_vals = pd.to_numeric(filtered_df["🎯 R² Score"], errors="coerce").dropna()
                        rmse_vals = pd.to_numeric(filtered_df["📏 RMSE"], errors="coerce").dropna()
                        mae_vals = pd.to_numeric(filtered_df["📊 MAE"], errors="coerce").dropna()
                        
                        # Key Performance Indicators
                        st.markdown("#### 🎯 **Key Performance Indicators**")
                        
                        perf_col1, perf_col2, perf_col3, perf_col4 = st.columns(4)
                        
                        with perf_col1:
                            st.metric(
                                "🏆 Best R² Score",
                                f"{r2_vals.max():.4f}",
                                f"Model Excellence",
                                help="Highest coefficient of determination achieved"
                            )
                        
                        with perf_col2:
                            r2_spread = r2_vals.max() - r2_vals.min()
                            spread_quality = "🟢 Excellent" if r2_spread < 0.02 else "🟡 Good" if r2_spread < 0.05 else "🔴 High"
                            st.metric(
                                "📊 Performance Spread",
                                f"{r2_spread:.4f}",
                                spread_quality,
                                help="Variation in R² scores across all combinations"
                            )
                        
                        with perf_col3:
                            r2_consistency = 1 - (r2_vals.std() / r2_vals.mean()) if r2_vals.mean() > 0 else 0
                            consistency_grade = "🟢 High" if r2_consistency > 0.95 else "🟡 Medium" if r2_consistency > 0.85 else "🔴 Low"
                            st.metric(
                                "📈 Consistency Index",
                                f"{r2_consistency:.3f}",
                                consistency_grade,
                                help="Coefficient of variation for model consistency"
                            )
                        
                        with perf_col4:
                            best_rmse = rmse_vals.min()
                            rmse_quality = "🟢 Excellent" if best_rmse < 0.1 else "🟡 Good" if best_rmse < 0.3 else "🔴 High"
                            st.metric(
                                "📉 Best RMSE",
                                f"{best_rmse:.4f}",
                                rmse_quality,
                                help="Lowest root mean square error achieved"
                            )
                        
                        # Performance Distribution Analysis
                        st.markdown("#### 📈 **Performance Distribution Analysis**")
                        
                        # Create performance tiers
                        excellent_r2 = len(r2_vals[r2_vals >= 0.9])
                        good_r2 = len(r2_vals[(r2_vals >= 0.8) & (r2_vals < 0.9)])
                        fair_r2 = len(r2_vals[(r2_vals >= 0.7) & (r2_vals < 0.8)])
                        poor_r2 = len(r2_vals[r2_vals < 0.7])
                        
                        tier_data = [
                            {"🏆 Performance Tier": "🟢 Excellent (R² ≥ 0.90)", "📊 Count": excellent_r2, "📈 Percentage": f"{(excellent_r2/len(r2_vals)*100):.1f}%"},
                            {"🏆 Performance Tier": "🟡 Good (0.80-0.89)", "📊 Count": good_r2, "📈 Percentage": f"{(good_r2/len(r2_vals)*100):.1f}%"},
                            {"🏆 Performance Tier": "🟠 Fair (0.70-0.79)", "📊 Count": fair_r2, "📈 Percentage": f"{(fair_r2/len(r2_vals)*100):.1f}%"},
                            {"🏆 Performance Tier": "🔴 Needs Improvement (<0.70)", "📊 Count": poor_r2, "📈 Percentage": f"{(poor_r2/len(r2_vals)*100):.1f}%"}
                        ]
                        
                        tier_df = pd.DataFrame(tier_data)
                        
                        # Style the tier table
                        styled_tier = tier_df.style.set_properties(**{
                            'padding': '8px',
                            'font-size': '13px',
                            'text-align': 'center',
                            'border': '1px solid #dee2e6'
                        }).set_table_styles([
                            {'selector': 'th', 'props': [
                                ('background-color', '#0d6efd'),
                                ('color', 'white'),
                                ('font-weight', 'bold'),
                                ('text-align', 'center'),
                                ('padding', '10px')
                            ]},
                            {'selector': '', 'props': [
                                ('border-collapse', 'collapse'),
                                ('border-radius', '6px'),
                                ('overflow', 'hidden')
                            ]}
                        ])
                        
                        st.dataframe(styled_tier, use_container_width=True, hide_index=True)
                        
                        # Statistical Summary
                        st.markdown("#### 📊 **Statistical Summary**")
                        
                        stats_data = [
                            {"📊 Metric": "R² Score", "📈 Mean": f"{r2_vals.mean():.4f}", "📊 Median": f"{r2_vals.median():.4f}", "📉 Std Dev": f"{r2_vals.std():.4f}", "🎯 Range": f"{r2_vals.min():.4f} - {r2_vals.max():.4f}"},
                            {"📊 Metric": "RMSE", "📈 Mean": f"{rmse_vals.mean():.4f}", "📊 Median": f"{rmse_vals.median():.4f}", "📉 Std Dev": f"{rmse_vals.std():.4f}", "🎯 Range": f"{rmse_vals.min():.4f} - {rmse_vals.max():.4f}"},
                            {"📊 Metric": "MAE", "📈 Mean": f"{mae_vals.mean():.4f}", "📊 Median": f"{mae_vals.median():.4f}", "📉 Std Dev": f"{mae_vals.std():.4f}", "🎯 Range": f"{mae_vals.min():.4f} - {mae_vals.max():.4f}"}
                        ]
                        
                        stats_df = pd.DataFrame(stats_data)
                        st.dataframe(stats_df, use_container_width=True, hide_index=True)
                        
                    else:
                        # CLASSIFICATION ANALYSIS
                        f1_vals = pd.to_numeric(filtered_df["🏆 F1-Score"], errors="coerce").dropna()
                        acc_vals = pd.to_numeric(filtered_df["🎯 Accuracy"], errors="coerce").dropna()
                        precision_vals = pd.to_numeric(filtered_df["⚖️ Precision"], errors="coerce").dropna()
                        recall_vals = pd.to_numeric(filtered_df["🔍 Recall"], errors="coerce").dropna()
                        
                        # Key Performance Indicators
                        st.markdown("#### 🎯 **Key Performance Indicators**")
                        
                        perf_col1, perf_col2, perf_col3, perf_col4 = st.columns(4)
                        
                        with perf_col1:
                            st.metric(
                                "🏆 Best F1-Score",
                                f"{f1_vals.max():.4f}",
                                "Model Excellence",
                                help="Highest F1-Score achieved across all combinations"
                            )
                        
                        with perf_col2:
                            f1_spread = f1_vals.max() - f1_vals.min()
                            spread_quality = "🟢 Excellent" if f1_spread < 0.02 else "🟡 Good" if f1_spread < 0.05 else "🔴 High"
                            st.metric(
                                "📊 Performance Spread",
                                f"{f1_spread:.4f}",
                                spread_quality,
                                help="Variation in F1-Scores across all combinations"
                            )
                        
                        with perf_col3:
                            f1_consistency = 1 - (f1_vals.std() / f1_vals.mean()) if f1_vals.mean() > 0 else 0
                            consistency_grade = "🟢 High" if f1_consistency > 0.95 else "🟡 Medium" if f1_consistency > 0.85 else "🔴 Low"
                            st.metric(
                                "📈 Consistency Index",
                                f"{f1_consistency:.3f}",
                                consistency_grade,
                                help="Coefficient of variation for model consistency"
                            )
                        
                        with perf_col4:
                            balanced_score = (acc_vals.mean() + precision_vals.mean() + recall_vals.mean()) / 3
                            balance_quality = "🟢 Excellent" if balanced_score > 0.9 else "🟡 Good" if balanced_score > 0.8 else "🔴 Fair"
                            st.metric(
                                "⚖️ Balance Score",
                                f"{balanced_score:.4f}",
                                balance_quality,
                                help="Average of accuracy, precision, and recall"
                            )
                        
                        # Performance Distribution Analysis
                        st.markdown("#### 📈 **Performance Distribution Analysis**")
                        
                        # Create performance tiers for classification
                        excellent_f1 = len(f1_vals[f1_vals >= 0.9])
                        good_f1 = len(f1_vals[(f1_vals >= 0.8) & (f1_vals < 0.9)])
                        fair_f1 = len(f1_vals[(f1_vals >= 0.7) & (f1_vals < 0.8)])
                        poor_f1 = len(f1_vals[f1_vals < 0.7])
                        
                        tier_data = [
                            {"🏆 Performance Tier": "🟢 Excellent (F1 ≥ 0.90)", "📊 Count": excellent_f1, "📈 Percentage": f"{(excellent_f1/len(f1_vals)*100):.1f}%"},
                            {"🏆 Performance Tier": "🟡 Good (0.80-0.89)", "📊 Count": good_f1, "📈 Percentage": f"{(good_f1/len(f1_vals)*100):.1f}%"},
                            {"🏆 Performance Tier": "🟠 Fair (0.70-0.79)", "📊 Count": fair_f1, "📈 Percentage": f"{(fair_f1/len(f1_vals)*100):.1f}%"},
                            {"🏆 Performance Tier": "🔴 Needs Improvement (<0.70)", "📊 Count": poor_f1, "📈 Percentage": f"{(poor_f1/len(f1_vals)*100):.1f}%"}
                        ]
                        
                        tier_df = pd.DataFrame(tier_data)
                        st.dataframe(tier_df, use_container_width=True, hide_index=True)
                        
                        # Statistical Summary
                        st.markdown("#### 📊 **Statistical Summary**")
                        
                        stats_data = [
                            {"📊 Metric": "F1-Score", "📈 Mean": f"{f1_vals.mean():.4f}", "📊 Median": f"{f1_vals.median():.4f}", "📉 Std Dev": f"{f1_vals.std():.4f}", "🎯 Range": f"{f1_vals.min():.4f} - {f1_vals.max():.4f}"},
                            {"📊 Metric": "Accuracy", "📈 Mean": f"{acc_vals.mean():.4f}", "📊 Median": f"{acc_vals.median():.4f}", "📉 Std Dev": f"{acc_vals.std():.4f}", "🎯 Range": f"{acc_vals.min():.4f} - {acc_vals.max():.4f}"},
                            {"📊 Metric": "Precision", "📈 Mean": f"{precision_vals.mean():.4f}", "📊 Median": f"{precision_vals.median():.4f}", "📉 Std Dev": f"{precision_vals.std():.4f}", "🎯 Range": f"{precision_vals.min():.4f} - {precision_vals.max():.4f}"},
                            {"📊 Metric": "Recall", "📈 Mean": f"{recall_vals.mean():.4f}", "📊 Median": f"{recall_vals.median():.4f}", "📉 Std Dev": f"{recall_vals.std():.4f}", "🎯 Range": f"{recall_vals.min():.4f} - {recall_vals.max():.4f}"}
                        ]
                        
                        stats_df = pd.DataFrame(stats_data)
                        st.dataframe(stats_df, use_container_width=True, hide_index=True)
                
                with insights_tab2:
                    st.markdown("### 🔒 **Privacy Impact Assessment**")
                    
                    # Privacy method effectiveness analysis
                    original_data = filtered_df[filtered_df['🔒 Type'] == 'Original']
                    anonymized_data = filtered_df[filtered_df['🔒 Type'] == 'Anonymized']
                    
                    if len(original_data) > 0 and len(anonymized_data) > 0:
                        
                        # Privacy Method Effectiveness
                        st.markdown("#### 🛡️ **Privacy Method Effectiveness**")
                        
                        # Get unique privacy methods and their performance
                        privacy_methods = anonymized_data['📊 Privacy Method'].unique()
                        method_analysis = []
                        
                        for method in privacy_methods:
                            method_data = anonymized_data[anonymized_data['📊 Privacy Method'] == method]
                            
                            if is_regression:
                                avg_performance = pd.to_numeric(method_data["🎯 R² Score"], errors="coerce").mean()
                                original_avg = pd.to_numeric(original_data["🎯 R² Score"], errors="coerce").mean()
                                performance_retention = (avg_performance / original_avg * 100) if original_avg > 0 else 0
                                metric_name = "R² Score"
                            else:
                                avg_performance = pd.to_numeric(method_data["🏆 F1-Score"], errors="coerce").mean()
                                original_avg = pd.to_numeric(original_data["🏆 F1-Score"], errors="coerce").mean()
                                performance_retention = (avg_performance / original_avg * 100) if original_avg > 0 else 0
                                metric_name = "F1-Score"
                            
                            # Determine effectiveness grade
                            if performance_retention >= 95:
                                effectiveness_grade = "🟢 Excellent"
                                privacy_impact = "Minimal Impact"
                            elif performance_retention >= 90:
                                effectiveness_grade = "🟡 Good"
                                privacy_impact = "Low Impact"
                            elif performance_retention >= 80:
                                effectiveness_grade = "🟠 Fair"
                                privacy_impact = "Moderate Impact"
                            else:
                                effectiveness_grade = "🔴 Poor"
                                privacy_impact = "High Impact"
                            
                            method_analysis.append({
                                "🔒 Privacy Method": method,
                                "📊 Dataset Count": len(method_data),
                                f"📈 Avg {metric_name}": f"{avg_performance:.4f}",
                                "🎯 Performance Retention": f"{performance_retention:.1f}%",
                                "📊 Effectiveness Grade": effectiveness_grade,
                                "⚠️ Privacy Impact": privacy_impact
                            })
                        
                        if method_analysis:
                            method_df = pd.DataFrame(method_analysis)
                            method_df = method_df.sort_values("🎯 Performance Retention", ascending=False)
                            
                            # Style the method analysis table
                            styled_method = method_df.style.set_properties(**{
                                'padding': '8px',
                                'font-size': '13px',
                                'text-align': 'center',
                                'border': '1px solid #dee2e6'
                            }).set_table_styles([
                                {'selector': 'th', 'props': [
                                    ('background-color', '#6f42c1'),
                                    ('color', 'white'),
                                    ('font-weight', 'bold'),
                                    ('text-align', 'center'),
                                    ('padding', '10px')
                                ]},
                                {'selector': '', 'props': [
                                    ('border-collapse', 'collapse'),
                                    ('border-radius', '6px'),
                                    ('overflow', 'hidden')
                                ]}
                            ])
                            
                            st.dataframe(styled_method, use_container_width=True, hide_index=True)
                        
                        # Privacy-Utility Trade-off Analysis
                        st.markdown("#### ⚖️ **Privacy-Utility Trade-off Metrics**")
                        
                        # Calculate overall trade-off metrics
                        total_anonymized = len(anonymized_data)
                        
                        if is_regression:
                            original_performance = pd.to_numeric(original_data["🎯 R² Score"], errors="coerce").mean()
                            anon_performance = pd.to_numeric(anonymized_data["🎯 R² Score"], errors="coerce")
                        else:
                            original_performance = pd.to_numeric(original_data["🏆 F1-Score"], errors="coerce").mean()
                            anon_performance = pd.to_numeric(anonymized_data["🏆 F1-Score"], errors="coerce")
                        
                        # Calculate trade-off statistics
                        performance_drops = ((original_performance - anon_performance) / original_performance * 100).dropna()
                        
                        minimal_loss = len(performance_drops[performance_drops <= 5])  # <= 5% loss
                        moderate_loss = len(performance_drops[(performance_drops > 5) & (performance_drops <= 15)])
                        significant_loss = len(performance_drops[performance_drops > 15])
                        
                        trade_off_col1, trade_off_col2, trade_off_col3, trade_off_col4 = st.columns(4)
                        
                        with trade_off_col1:
                            st.metric(
                                "🟢 Minimal Loss",
                                minimal_loss,
                                f"{(minimal_loss/total_anonymized*100):.1f}% of anonymized",
                                help="Datasets with ≤5% performance loss"
                            )
                        
                        with trade_off_col2:
                            st.metric(
                                "🟡 Moderate Loss",
                                moderate_loss,
                                f"{(moderate_loss/total_anonymized*100):.1f}% of anonymized",
                                help="Datasets with 5-15% performance loss"
                            )
                        
                        with trade_off_col3:
                            st.metric(
                                "🔴 Significant Loss",
                                significant_loss,
                                f"{(significant_loss/total_anonymized*100):.1f}% of anonymized",
                                help="Datasets with >15% performance loss"
                            )
                        
                        with trade_off_col4:
                            avg_performance_loss = performance_drops.mean()
                            loss_quality = "🟢 Low" if avg_performance_loss <= 5 else "🟡 Moderate" if avg_performance_loss <= 15 else "🔴 High"
                            st.metric(
                                "📊 Avg Performance Loss",
                                f"{avg_performance_loss:.1f}%",
                                loss_quality,
                                help="Average performance degradation across all anonymized datasets"
                            )
                    
                    else:
                        st.info("⚠️ Privacy impact analysis requires both original and anonymized datasets")
                        st.markdown("**Required for Analysis:**")
                        st.markdown("- At least one original dataset result")
                        st.markdown("- At least one anonymized dataset result")
                        st.markdown("- Same models trained on both dataset types")
                
                with insights_tab3:
                    st.markdown("### ⚖️ **Utility-Privacy Trade-off Analysis**")
                    
                    # Advanced trade-off analysis
                    if len(anonymized_data) > 0 and len(original_data) > 0:
                        
                        # Trade-off Efficiency Analysis
                        st.markdown("#### 🎯 **Trade-off Efficiency Analysis**")
                        
                        # Model-specific trade-off analysis
                        models = filtered_df['🤖 Model'].unique()
                        
                        model_trade_offs = []
                        
                        for model in models:
                            model_original = original_data[original_data['🤖 Model'] == model]
                            model_anon = anonymized_data[anonymized_data['🤖 Model'] == model]
                            
                            if len(model_original) > 0 and len(model_anon) > 0:
                                if is_regression:
                                    orig_perf = pd.to_numeric(model_original["🎯 R² Score"], errors="coerce").iloc[0]
                                    anon_perfs = pd.to_numeric(model_anon["🎯 R² Score"], errors="coerce")
                                    metric_name = "R² Score"
                                else:
                                    orig_perf = pd.to_numeric(model_original["🏆 F1-Score"], errors="coerce").iloc[0]
                                    anon_perfs = pd.to_numeric(model_anon["🏆 F1-Score"], errors="coerce")
                                    metric_name = "F1-Score"
                                
                                # Calculate trade-off metrics
                                best_anon_perf = anon_perfs.max()
                                worst_anon_perf = anon_perfs.min()
                                avg_anon_perf = anon_perfs.mean()
                                
                                trade_off_range = ((orig_perf - worst_anon_perf) / orig_perf * 100) if orig_perf > 0 else 0
                                best_retention = (best_anon_perf / orig_perf * 100) if orig_perf > 0 else 0
                                
                                # Determine trade-off quality
                                if best_retention >= 95:
                                    trade_off_quality = "🟢 Excellent"
                                elif best_retention >= 90:
                                    trade_off_quality = "🟡 Good"
                                elif best_retention >= 80:
                                    trade_off_quality = "🟠 Fair"
                                else:
                                    trade_off_quality = "🔴 Poor"
                                
                                model_trade_offs.append({
                                    "🤖 Model": model,
                                    f"📊 Original {metric_name}": f"{orig_perf:.4f}",
                                    f"🔒 Best Anonymized": f"{best_anon_perf:.4f}",
                                    "🎯 Best Retention": f"{best_retention:.1f}%",
                                    "📉 Max Loss": f"{trade_off_range:.1f}%",
                                    "⚖️ Trade-off Quality": trade_off_quality,
                                    "📊 Anon Datasets": len(model_anon)
                                })
                        
                        if model_trade_offs:
                            trade_off_df = pd.DataFrame(model_trade_offs)
                            trade_off_df = trade_off_df.sort_values("🎯 Best Retention", ascending=False)
                            
                            st.dataframe(trade_off_df, use_container_width=True, hide_index=True)
                        
                        # Privacy Budget Analysis
                        st.markdown("#### 💰 **Privacy Budget Efficiency**")
                        
                        st.info("💡 **Privacy Budget Concept**: The amount of utility you're willing to sacrifice for privacy protection")
                        
                        # Create privacy budget scenarios
                        budget_scenarios = [
                            {"🏷️ Scenario": "High Utility (95%+ retention)", "📊 Description": "Minimal privacy, maximum utility", "🎯 Use Case": "Internal analytics, low-risk data"},
                            {"🏷️ Scenario": "Balanced (85-95% retention)", "📊 Description": "Good privacy-utility balance", "🎯 Use Case": "Shared research, moderate-risk data"},
                            {"🏷️ Scenario": "High Privacy (70-85% retention)", "📊 Description": "Strong privacy, acceptable utility", "🎯 Use Case": "Public release, high-risk data"},
                            {"🏷️ Scenario": "Maximum Privacy (<70% retention)", "📊 Description": "Maximum privacy, limited utility", "🎯 Use Case": "Regulatory compliance, very high-risk data"}
                        ]
                        
                        scenario_df = pd.DataFrame(budget_scenarios)
                        st.dataframe(scenario_df, use_container_width=True, hide_index=True)
                        
                        # Recommendation Matrix
                        st.markdown("#### 🎯 **Method Selection Matrix**")
                        
                        # Create recommendation based on performance
                        recommendations = []
                        
                        for method in privacy_methods:
                            method_data = anonymized_data[anonymized_data['📊 Privacy Method'] == method]
                            
                            if is_regression:
                                avg_performance = pd.to_numeric(method_data["🎯 R² Score"], errors="coerce").mean()
                                original_avg = pd.to_numeric(original_data["🎯 R² Score"], errors="coerce").mean()
                            else:
                                avg_performance = pd.to_numeric(method_data["🏆 F1-Score"], errors="coerce").mean()
                                original_avg = pd.to_numeric(original_data["🏆 F1-Score"], errors="coerce").mean()
                            
                            retention = (avg_performance / original_avg * 100) if original_avg > 0 else 0
                            
                            # Determine recommendation
                            if retention >= 95:
                                recommendation = "🟢 Recommended for all use cases"
                                risk_level = "Low Risk"
                            elif retention >= 90:
                                recommendation = "🟡 Good for most use cases"
                                risk_level = "Low-Medium Risk"
                            elif retention >= 80:
                                recommendation = "🟠 Suitable for high-privacy needs"
                                risk_level = "Medium Risk"
                            else:
                                recommendation = "🔴 Only for maximum privacy requirements"
                                risk_level = "High Risk"
                            
                            recommendations.append({
                                "🔒 Privacy Method": method,
                                "📊 Utility Retention": f"{retention:.1f}%",
                                "⚠️ Risk Level": risk_level,
                                "💡 Recommendation": recommendation
                            })
                        
                        if recommendations:
                            rec_df = pd.DataFrame(recommendations)
                            rec_df = rec_df.sort_values("📊 Utility Retention", ascending=False)
                            st.dataframe(rec_df, use_container_width=True, hide_index=True)
                    
                    else:
                        st.info("⚠️ Trade-off analysis requires both original and anonymized datasets")
                
                with insights_tab4:
                    st.markdown("### 💼 **Business Recommendations & Implementation Guide**")
                    
                    # Executive Summary
                    st.markdown("#### 📋 **Executive Summary**")
                    
                    total_experiments = len(filtered_df)
                    unique_models = filtered_df['🤖 Model'].nunique()
                    unique_datasets = filtered_df['🗂️ Dataset'].nunique()
                    
                    exec_col1, exec_col2, exec_col3 = st.columns(3)
                    
                    with exec_col1:
                        st.metric(
                            "📊 Total Experiments",
                            total_experiments,
                            f"{unique_models} models × {unique_datasets} datasets",
                            help="Complete experimental coverage"
                        )
                    
                    with exec_col2:
                        if is_regression:
                            best_score = pd.to_numeric(filtered_df["🎯 R² Score"], errors="coerce").max()
                            score_name = "R² Score"
                        else:
                            best_score = pd.to_numeric(filtered_df["🏆 F1-Score"], errors="coerce").max()
                            score_name = "F1-Score"
                        
                        performance_grade = "🟢 Excellent" if best_score >= 0.9 else "🟡 Good" if best_score >= 0.8 else "🔴 Needs Improvement"
                        st.metric(
                            f"🏆 Best {score_name}",
                            f"{best_score:.4f}",
                            performance_grade,
                            help="Highest performance achieved"
                        )
                    
                    with exec_col3:
                        if len(anonymized_data) > 0:
                            privacy_coverage = len(privacy_methods)
                            coverage_grade = "🟢 Comprehensive" if privacy_coverage >= 3 else "🟡 Good" if privacy_coverage >= 2 else "🔴 Limited"
                            st.metric(
                                "🔒 Privacy Methods",
                                privacy_coverage,
                                coverage_grade,
                                help="Number of privacy techniques evaluated"
                            )
                        else:
                            st.metric(
                                "🔒 Privacy Methods",
                                "0",
                                "❌ No privacy analysis",
                                help="No anonymized datasets found"
                            )
                    
                    # Strategic Recommendations
                    st.markdown("#### 🎯 **Strategic Recommendations**")
                    
                    if len(anonymized_data) > 0 and len(original_data) > 0:
                        # Calculate overall recommendations
                        if is_regression:
                            original_avg = pd.to_numeric(original_data["🎯 R² Score"], errors="coerce").mean()
                            anon_avg = pd.to_numeric(anonymized_data["🎯 R² Score"], errors="coerce").mean()
                        else:
                            original_avg = pd.to_numeric(original_data["🏆 F1-Score"], errors="coerce").mean()
                            anon_avg = pd.to_numeric(anonymized_data["🏆 F1-Score"], errors="coerce").mean()
                        
                        overall_retention = (anon_avg / original_avg * 100) if original_avg > 0 else 0
                        
                        # Strategic recommendations based on results
                        recommendations = []
                        
                        # Performance recommendation
                        if overall_retention >= 95:
                            recommendations.append({
                                "🎯 Area": "Performance Assessment",
                                "📊 Finding": f"Excellent utility retention ({overall_retention:.1f}%)",
                                "💡 Recommendation": "Proceed with confidence - minimal privacy-utility trade-off",
                                "⚡ Action": "Deploy in production environment"
                            })
                        elif overall_retention >= 85:
                            recommendations.append({
                                "🎯 Area": "Performance Assessment", 
                                "📊 Finding": f"Good utility retention ({overall_retention:.1f}%)",
                                "💡 Recommendation": "Acceptable for most business use cases",
                                "⚡ Action": "Consider deployment with monitoring"
                            })
                        else:
                            recommendations.append({
                                "🎯 Area": "Performance Assessment",
                                "📊 Finding": f"Moderate utility retention ({overall_retention:.1f}%)",
                                "💡 Recommendation": "Evaluate if utility loss is acceptable for privacy gains",
                                "⚡ Action": "Consider alternative privacy methods or parameter tuning"
                            })
                        
                        # Best method recommendation
                        best_method_row = method_df.iloc[0] if 'method_df' in locals() and len(method_df) > 0 else None
                        if best_method_row is not None:
                            best_method = best_method_row['🔒 Privacy Method']
                            best_retention = best_method_row['🎯 Performance Retention']
                            
                            recommendations.append({
                                "🎯 Area": "Method Selection",
                                "📊 Finding": f"{best_method} shows best performance ({best_retention})",
                                "💡 Recommendation": f"Prioritize {best_method} for production deployment",
                                "⚡ Action": "Conduct detailed testing with larger datasets"
                            })
                        
                        # Implementation recommendation
                        high_performing_methods = len([m for m in method_analysis if float(m['🎯 Performance Retention'].rstrip('%')) >= 90])
                        
                        if high_performing_methods >= 2:
                            recommendations.append({
                                "🎯 Area": "Implementation Strategy",
                                "📊 Finding": f"{high_performing_methods} methods show high performance",
                                "💡 Recommendation": "Implement multi-method approach for different risk scenarios",
                                "⚡ Action": "Create method selection framework based on data sensitivity"
                            })
                        
                        # Display recommendations table
                        if recommendations:
                            rec_df = pd.DataFrame(recommendations)
                            
                            styled_rec = rec_df.style.set_properties(**{
                                'padding': '12px',
                                'font-size': '13px',
                                'text-align': 'left',
                                'border': '1px solid #dee2e6'
                            }).set_table_styles([
                                {'selector': 'th', 'props': [
                                    ('background-color', '#28a745'),
                                    ('color', 'white'),
                                    ('font-weight', 'bold'),
                                    ('text-align', 'center'),
                                    ('padding', '12px')
                                ]},
                                {'selector': 'td:first-child', 'props': [
                                    ('font-weight', 'bold'),
                                    ('color', '#28a745')
                                ]},
                                {'selector': '', 'props': [
                                    ('border-collapse', 'collapse'),
                                    ('border-radius', '6px'),
                                    ('overflow', 'hidden')
                                ]}
                            ])
                            
                            st.dataframe(styled_rec, use_container_width=True, hide_index=True)
                    
                    # Implementation Roadmap
                    st.markdown("#### 🛣️ **Implementation Roadmap**")
                    
                    roadmap_phases = [
                        {
                            "🎯 Phase": "1. Proof of Concept",
                            "📅 Timeline": "2-4 weeks",
                            "📊 Scope": "Test with small, non-sensitive dataset",
                            "🎯 Success Criteria": f">80% {score_name} retention",
                            "💡 Key Activities": "Method validation, performance benchmarking"
                        },
                        {
                            "🎯 Phase": "2. Pilot Implementation", 
                            "📅 Timeline": "4-8 weeks",
                            "📊 Scope": "Deploy on moderate-risk datasets",
                            "🎯 Success Criteria": f">85% {score_name} retention",
                            "💡 Key Activities": "Integration testing, stakeholder training"
                        },
                        {
                            "🎯 Phase": "3. Production Rollout",
                            "📅 Timeline": "8-12 weeks", 
                            "📊 Scope": "Full production deployment",
                            "🎯 Success Criteria": "Business requirements met",
                            "💡 Key Activities": "Monitoring setup, documentation, support"
                        },
                        {
                            "🎯 Phase": "4. Optimization",
                            "📅 Timeline": "Ongoing",
                            "📊 Scope": "Continuous improvement",
                            "🎯 Success Criteria": "Improved performance over time",
                            "💡 Key Activities": "Parameter tuning, method refinement"
                        }
                    ]
                    
                    roadmap_df = pd.DataFrame(roadmap_phases)
                    st.dataframe(roadmap_df, use_container_width=True, hide_index=True)
                    
                    # Risk Assessment & Mitigation
                    st.markdown("#### ⚠️ **Risk Assessment & Mitigation**")
                    
                    risk_factors = [
                        {
                            "⚠️ Risk Factor": "Performance Degradation",
                            "📊 Impact": "Medium" if overall_retention >= 85 else "High",
                            "🎯 Probability": "Low" if overall_retention >= 90 else "Medium",
                            "💡 Mitigation": "Implement performance monitoring and alerting",
                            "📋 Owner": "Data Science Team"
                        },
                        {
                            "⚠️ Risk Factor": "Privacy Re-identification",
                            "📊 Impact": "High",
                            "🎯 Probability": "Low" if len(privacy_methods) >= 2 else "Medium",
                            "💡 Mitigation": "Regular privacy audits and compliance reviews",
                            "📋 Owner": "Privacy Officer"
                        },
                        {
                            "⚠️ Risk Factor": "Regulatory Compliance",
                            "📊 Impact": "High",
                            "🎯 Probability": "Low",
                            "💡 Mitigation": "Legal review and documentation of privacy measures",
                            "📋 Owner": "Legal/Compliance Team"
                        }
                    ]
                    
                    risk_df = pd.DataFrame(risk_factors)
                    st.dataframe(risk_df, use_container_width=True, hide_index=True)
            
            else:
                st.info("No data available for professional insights analysis.")
                            
            # Enhanced export functionality
            st.markdown("**💾 Export Options:**")
            export_col1, export_col2, export_col3 = st.columns(3)
            
            with export_col1:
                if st.button("📥 Download Current View", key="export_comprehensive_current_csv"):
                    csv_data = filtered_df.to_csv(index=False)
                    st.download_button(
                        label="💾 Download Filtered Results",
                        data=csv_data,
                        file_name=f"comprehensive_results_filtered_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv",
                        use_container_width=True
                    )
            
            with export_col2:
                if st.button("📊 Download All Data", key="export_comprehensive_full_csv"):
                    csv_data = comprehensive_df.to_csv(index=False)
                    st.download_button(
                        label="💾 Download Complete Dataset",
                        data=csv_data,
                        file_name=f"comprehensive_results_complete_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv",
                        use_container_width=True
                    )
            
            with export_col3:
                # Show summary of what would be included in export
                total_columns_filtered = len(filtered_df.columns)
                metric_columns = len([col for col in filtered_df.columns if any(metric in col for metric in ['Score', 'Accuracy', 'Precision', 'Recall', 'F1', 'RMSE', 'MAE', 'MAPE'])])
                diff_columns = len([col for col in filtered_df.columns if "% Diff" in col])
                
                st.info(f"📊 Current view: {total_columns_filtered} columns ({metric_columns} metrics, {diff_columns} comparisons)")

        else:
            st.info("No experiment results available for comprehensive comparison")
        
        # --- FILE-BASED DASHBOARD SECTION ---
        if st.session_state.get('file_dashboard_enabled', False) and st.session_state.get('uploaded_results_data') is not None:
            st.markdown("---")
            st.markdown("#### 📁 **External Results Dashboard**")
            st.markdown("*Analyzing uploaded experiment results*")
            
            uploaded_data = st.session_state.uploaded_results_data
            
            # Display uploaded data info
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("📊 Rows", uploaded_data.shape[0])
            with col2:
                st.metric("📈 Columns", uploaded_data.shape[1])
            with col3:
                st.metric("💾 Size", f"{uploaded_data.memory_usage(deep=True).sum() / 1024**2:.1f} MB")
            
            # Show the data with column selection
            st.markdown("**📋 Uploaded Data Preview:**")
            
            # Column selection for the uploaded data
            available_columns = uploaded_data.columns.tolist()
            default_columns = available_columns[:min(10, len(available_columns))]  # Show first 10 columns by default
            
            selected_file_columns = st.multiselect(
                "Select columns to display and analyze:",
                available_columns,
                default=default_columns,
                key="file_dashboard_columns"
            )
            
            if selected_file_columns:
                filtered_file_data = uploaded_data[selected_file_columns]
                st.dataframe(filtered_file_data, use_container_width=True)
                
                # Create the PPML dashboard with file data
                if 'ppml_dashboard_visualizer' in st.session_state:
                    ppml_viz_file = st.session_state.ppml_dashboard_visualizer
                    
                    # Prepare the file data for the dashboard (same structure as table data)
                    st.session_state.filtered_comprehensive_df = filtered_file_data
                    st.session_state.comprehensive_selected_columns = selected_file_columns
                    
                    st.info(f"📊 **File Data Ready:** {len(filtered_file_data)} rows × {len(selected_file_columns)} columns loaded for analysis")
                    
                    # Configuration UI for file-based dashboard
                    st.markdown("**⚙️ Dashboard Configuration:**")
                    ppml_file_config_key_prefix = "ppml_file_dashboard_v2"
                    ppml_file_config = ppml_viz_file.get_config_ui(key_prefix=ppml_file_config_key_prefix)
                    
                    # Render the file-based dashboard
                    if ppml_file_config:
                        with st.spinner("Loading File-Based PPML Dashboard..."):
                            # Use the same render method but with file data
                            success = ppml_viz_file.render(
                                data=filtered_file_data,
                                model_results=[],  # No model results for file-based analysis
                                config=ppml_file_config
                            )
                            
                            if not success:
                                st.warning("⚠️ File dashboard rendering encountered issues. Please check your uploaded data format.")
                    else:
                        st.info("ℹ️ Please configure the file dashboard settings above.")
                else:
                    st.error("⚠️ Dashboard visualizer not available. Please restart the application.")
            else:
                st.warning("Please select at least one column to display.")
        
          # --- PROFESSIONAL PPML DASHBOARD SECTION ---
        st.markdown("---")
        st.markdown("#### 🛡️ **Professional PPML Analysis Dashboard**")
        
        if 'ppml_dashboard_visualizer' in st.session_state:
            ppml_viz_instance = st.session_state.ppml_dashboard_visualizer
            
            # Define valid results for PPML dashboard
            valid_results_for_ppml_dashboard = [
                res for res in st.session_state.experiment_results if "error" not in res
            ]
            
            # Add the filtered table data and selected columns to session state for the plugin
            if 'filtered_df' in locals() and 'selected_columns' in locals():
                st.session_state.filtered_comprehensive_df = filtered_df
                st.session_state.comprehensive_selected_columns = selected_columns
                
                # Show table data summary for transparency
                st.info(f"📊 **Table Data Ready:** {len(filtered_df)} rows × {len(selected_columns)} columns selected for professional analysis")
            else:
                st.info("ℹ️ **Professional Dashboard:** Select columns in the table above and click Apply to activate advanced analysis")
            
            # 1. Display the professional configuration UI
            ppml_main_config_key_prefix = "ppml_professional_dashboard_v2"
            ppml_main_config = ppml_viz_instance.get_config_ui(key_prefix=ppml_main_config_key_prefix)
            
            # 2. Render the professional dashboard
            if ppml_main_config:
                current_df_for_viz = st.session_state.df_uploaded if st.session_state.df_uploaded is not None else pd.DataFrame()
                
                with st.spinner("Loading Professional PPML Dashboard..."):
                    # Table-driven approach: plugin extracts data from session state
                    success = ppml_viz_instance.render(
                        data=current_df_for_viz,
                        model_results=valid_results_for_ppml_dashboard,
                        config=ppml_main_config
                    )
                    
                    if not success:
                        st.warning("⚠️ Dashboard rendering encountered issues. Please check your data selection.")
            else:
                st.info("ℹ️ Please configure the professional dashboard settings in the sidebar.")
        else:
            st.error("⚠️ Professional PPML Dashboard not initialized. Please restart the application.")

else:  # This is the `else` for `if has_multi_dataset:`
    # Fall back to regular comparison dashboard if no multi-dataset results
    st.header("📈 Model Comparison Dashboard")
    
    if len(st.session_state.experiment_results) > 1:
        st.markdown("#### 📊 **Comparison Overview**")
        
        col1, col2, col3, col4 = st.columns(4)
        
        # Count successful models
        successful_models = [res for res in st.session_state.experiment_results if "error" not in res]
        
        with col1:
            st.metric(
                label="🤖 Models Compared",
                value=len(successful_models),
                delta="✅ Successful",
                help="Number of successfully trained models in comparison"
            )
        
        with col2:
            # Find best performing model by F1-Score
            if successful_models:
                best_f1 = max(res['f1_score'] for res in successful_models)
                best_model = next(res for res in successful_models if res['f1_score'] == best_f1)
                st.metric(
                    label="🏆 Best F1-Score",
                    value=f"{best_f1:.4f}",
                    delta=f"🥇 {best_model['model_name']}",
                    help="Highest F1-Score achieved and the model that achieved it"
                )
        
        with col3:
            # Calculate performance spread
            if successful_models:
                f1_scores = [res['f1_score'] for res in successful_models]
                f1_spread = max(f1_scores) - min(f1_scores)
                spread_status = "🟢 Low" if f1_spread < 0.1 else "🟡 Medium" if f1_spread < 0.2 else "🔴 High"
                st.metric(
                    label="📊 Performance Spread",
                    value=f"{f1_spread:.4f}",
                    delta=spread_status,
                    help="Difference between best and worst F1-Score performance"
                )
        
        with col4:
            # Task type and target info
            st.metric(
                label="🎯 Task Type",
                value=st.session_state.task_type.title(),
                delta=f"Target: {st.session_state.target_column}",
                help="Current machine learning task type and target column"
            )
        
        # Prepare comparison data with enhanced processing
        comparison_data = []
        for idx, res in enumerate(st.session_state.experiment_results):
            if "error" not in res:
                current_run_number = sum(1 for i, r in enumerate(st.session_state.experiment_results[:idx+1]) 
                                       if r.get('model_name') == res['model_name'] 
                                       and r.get('target_column') == res['target_column']
                                       and "error" not in r)
                
                model_identifier = f"{res['model_name']} - Run #{current_run_number}"
                
                row_data = {
                    "🤖 Model": model_identifier,
                    "🎯 Accuracy": res['accuracy'],
                    "⚖️ Precision": res['precision'],
                    "🔍 Recall": res['recall'],
                    "🏆 F1-Score": res['f1_score']
                }
                
                # Add custom metrics to comparison if available
                if res.get('custom_metrics'):
                    for metric_name, value in res['custom_metrics'].items():
                        if isinstance(value, (int, float)):
                            # Add emoji prefix for custom metrics
                            display_name = f"📊 {metric_name}"
                            row_data[display_name] = value
                
                comparison_data.append(row_data)
        
        if comparison_data:
            comparison_df = pd.DataFrame(comparison_data).set_index("🤖 Model")
            
            # Enhanced comparison table with adaptive theme styling
            st.markdown("#### 📋 **Detailed Metrics Comparison Table**")
            
            # Add sorting options
            sort_col1, sort_col2, sort_col3 = st.columns([2, 1, 1])
            
            with sort_col1:
                # Get available numeric columns for sorting
                numeric_columns = [col for col in comparison_df.columns 
                                 if comparison_df[col].dtype in ['float64', 'float32', 'int64', 'int32']]
                
                if numeric_columns:
                    sort_by = st.selectbox(
                        "📊 Sort by metric:",
                        options=numeric_columns,
                        index=numeric_columns.index("🏆 F1-Score") if "🏆 F1-Score" in numeric_columns else 0,
                        key="comparison_sort_metric_main"
                    )
                else:
                    sort_by = None
            
            with sort_col2:
                if sort_by:
                    sort_ascending = st.checkbox(
                        "📈 Ascending",
                        value=False,
                        key="comparison_sort_ascending_main", 
                        help="Check for ascending order, uncheck for descending"
                    )
            
            with sort_col3:
                if st.button("🔄 Reset Sort", key="reset_comparison_sort_main"): 
                    # Reset to default F1-Score descending
                    st.session_state.comparison_sort_metric_main = "🏆 F1-Score" if "🏆 F1-Score" in numeric_columns else numeric_columns[0] if numeric_columns else None
                    st.session_state.comparison_sort_ascending_main = False
                    st.rerun()
            
            # Apply sorting if requested
            if sort_by:
                comparison_df_sorted = comparison_df.sort_values(
                    by=sort_by, 
                    ascending=sort_ascending
                )
                
                # Add rank column
                rank_values = []
                for i in range(len(comparison_df_sorted)):
                    if i == 0:
                        rank_values.append("🥇 1st")
                    elif i == 1:
                        rank_values.append("🥈 2nd")
                    elif i == 2:
                        rank_values.append("🥉 3rd")
                    else:
                        rank_values.append(f"#{i+1}")
                
                comparison_df_sorted.insert(0, "🏅 Rank", rank_values)
            else:
                comparison_df_sorted = comparison_df
                            
            # Enhanced styling with adaptive theme support
            formatted_df = comparison_df_sorted.style.format({
                col: "{:.4f}" for col in comparison_df_sorted.columns 
                if comparison_df_sorted[col].dtype in ['float64', 'float32'] and col != "🏅 Rank"
            }).set_properties(**{
                'padding': '12px',
                'font-size': '14px',
                'text-align': 'center',
                'border': '1px solid var(--text-color-secondary)',
                'font-weight': 'bold'
            }).set_table_styles([
                {'selector': 'th', 'props': [
                    ('background-color', '#6f42c1'),  # Purple header works on both themes
                    ('color', 'white'),
                    ('font-weight', 'bold'),
                    ('text-align', 'center'),
                    ('padding', '15px'),
                    ('border', '1px solid #6f42c1'),
                    ('font-size', '16px')
                ]},
                {'selector': 'td', 'props': [
                    ('text-align', 'center'),
                    ('vertical-align', 'middle'),
                    ('border', '1px solid var(--text-color-secondary)'),
                    ('font-size', '14px')
                ]},
                {'selector': '', 'props': [
                    ('border-collapse', 'collapse'),
                    ('margin', '20px 0'),
                    ('border-radius', '10px'),
                    ('overflow', 'hidden'),
                    ('box-shadow', '0 4px 6px rgba(0, 0, 0, 0.1)')
                ]},
                # Highlight the rank column if it exists
                {'selector': 'td:first-child', 'props': [
                    ('background-color', 'rgba(111, 66, 193, 0.1)'),
                    ('font-weight', 'bold'),
                    ('font-size', '16px')
                ]} if "🏅 Rank" in comparison_df_sorted.columns else {}
            ])
            
            # Add performance-based row highlighting
            if sort_by and len(comparison_df_sorted) > 1:
                # Highlight top performer
                formatted_df = formatted_df.apply(
                    lambda x: ['background-color: rgba(40, 167, 69, 0.2)' if x.name == comparison_df_sorted.index[0] 
                              else 'background-color: rgba(220, 53, 69, 0.1)' if x.name == comparison_df_sorted.index[-1]
                              else '' for i in x], 
                    axis=1
                )
            
            st.dataframe(formatted_df, use_container_width=True)
            
            # Performance insights section with modern table design
            st.markdown("#### 🔍 **Performance Insights**")
            
            # Create two-column layout for insights
            insight_col1, insight_col2 = st.columns(2)
            
            with insight_col1:
                st.markdown("**🏆 Top Performers Analysis**")
                
                # Top 3 by F1-Score in modern table format
                f1_sorted = comparison_df.sort_values("🏆 F1-Score", ascending=False)
                top_performers_data = []
                
                for i, (model_name, row) in enumerate(f1_sorted.head(3).iterrows()):
                    rank_emojis = ["🥇", "🥈", "🥉"]
                    rank_emoji = rank_emojis[i] if i < 3 else f"#{i+1}"
                    rank_text = ["1st Place", "2nd Place", "3rd Place"][i] if i < 3 else f"{i+1}th Place"
                    
                    f1_score = row["🏆 F1-Score"]
                    accuracy = row["🎯 Accuracy"]
                    precision = row["⚖️ Precision"]
                    recall = row["🔍 Recall"]
                    
                    # Calculate performance grade
                    avg_score = (f1_score + accuracy + precision + recall) / 4
                    if avg_score >= 0.95:
                        grade = "🏆 Excellent"
                    elif avg_score >= 0.85:
                        grade = "🥇 Very Good"
                    elif avg_score >= 0.75:
                        grade = "🥈 Good"
                    elif avg_score >= 0.65:
                        grade = "🥉 Fair"
                    else:
                        grade = "📈 Needs Improvement"
                    
                    top_performers_data.append({
                        "🏅 Rank": f"{rank_emoji} {rank_text}",
                        "🤖 Model": model_name,
                        "🏆 F1-Score": f"{f1_score:.4f}",
                        "🎯 Accuracy": f"{accuracy:.4f}",
                        "📊 Grade": grade
                    })
                
                if top_performers_data:
                    top_performers_df = pd.DataFrame(top_performers_data)
                    
                    # Modern styling for top performers table
                    styled_top_performers = top_performers_df.style.set_properties(**{
                        'padding': '10px',
                        'font-size': '13px',
                        'text-align': 'center',
                        'border': '1px solid var(--text-color-secondary)',
                        'font-weight': 'bold'
                    }).set_table_styles([
                        {'selector': 'th', 'props': [
                            ('background-color', '#28a745'),  # Green header for top performers
                            ('color', 'white'),
                            ('font-weight', 'bold'),
                            ('text-align', 'center'),
                            ('padding', '12px'),
                            ('border', '1px solid #28a745'),
                            ('font-size', '14px')
                        ]},
                        {'selector': 'td', 'props': [
                            ('text-align', 'center'),
                            ('vertical-align', 'middle'),
                            ('border', '1px solid var(--text-color-secondary)')
                        ]},
                        {'selector': '', 'props': [
                            ('border-collapse', 'collapse'),
                            ('margin', '10px 0'),
                            ('border-radius', '8px'),
                            ('overflow', 'hidden'),
                            ('box-shadow', '0 3px 6px rgba(0, 0, 0, 0.1)')
                        ]},
                        # Highlight first place
                        {'selector': 'tbody tr:first-child', 'props': [
                            ('background-color', 'rgba(40, 167, 69, 0.1)')
                        ]}
                    ])
                    
                    st.dataframe(styled_top_performers, use_container_width=True, hide_index=True)
                else:
                    st.info("No performance data available for ranking")
            
            with insight_col2:
                st.markdown("**📊 Statistical Analysis Dashboard**")
                
                # Calculate comprehensive statistics for all metrics
                key_metrics = ["🎯 Accuracy", "⚖️ Precision", "🔍 Recall", "🏆 F1-Score"]
                stats_data = []
                
                for metric in key_metrics:
                    if metric in comparison_df.columns:
                        metric_values = comparison_df[metric]
                        
                        # Calculate advanced statistics
                        mean_val = metric_values.mean()
                        std_val = metric_values.std()
                        min_val = metric_values.min()
                        max_val = metric_values.max()
                        median_val = metric_values.median()
                        
                        # Performance consistency assessment
                        if std_val < 0.01:
                            consistency = "🟢 Very Consistent"
                        elif std_val < 0.05:
                            consistency = "🟡 Consistent"
                        elif std_val < 0.1:
                            consistency = "🟠 Moderate"
                        else:
                            consistency = "🔴 Variable"
                        
                        # Get metric short name for display
                        metric_short = metric.split()[-1] if len(metric.split()) > 1 else metric.replace("🎯 ", "").replace("⚖️ ", "").replace("🔍 ", "").replace("🏆 ", "")
                        
                        stats_data.append({
                            "📊 Metric": metric_short,
                            "📈 Mean": f"{mean_val:.4f}",
                            "📉 Std Dev": f"{std_val:.4f}",
                            "🎯 Range": f"{min_val:.4f} - {max_val:.4f}",
                            "⚖️ Consistency": consistency
                        })
                
                if stats_data:
                    stats_df = pd.DataFrame(stats_data)
                    
                    # Modern styling for statistics table
                    styled_stats = stats_df.style.set_properties(**{
                        'padding': '10px',
                        'font-size': '13px',
                        'text-align': 'center',
                        'border': '1px solid var(--text-color-secondary)',
                        'font-weight': 'bold'
                    }).set_table_styles([
                        {'selector': 'th', 'props': [
                            ('background-color', '#17a2b8'),  # Info blue header for statistics
                            ('color', 'white'),
                            ('font-weight', 'bold'),
                            ('text-align', 'center'),
                            ('padding', '12px'),
                            ('border', '1px solid #17a2b8'),
                            ('font-size', '14px')
                        ]},
                        {'selector': 'td', 'props': [
                            ('text-align', 'center'),
                            ('vertical-align', 'middle'),
                            ('border', '1px solid var(--text-color-secondary)')
                        ]},
                        {'selector': '', 'props': [
                            ('border-collapse', 'collapse'),
                            ('margin', '10px 0'),
                            ('border-radius', '8px'),
                            ('overflow', 'hidden'),
                            ('box-shadow', '0 3px 6px rgba(0, 0, 0, 0.1)')
                        ]}
                    ])
                    
                    st.dataframe(styled_stats, use_container_width=True, hide_index=True)
                
                # Additional insights metrics below the table with modern table design
                st.markdown("**🔍 Quick Performance Insights Dashboard**")
                
                # Performance spread analysis
                if len(comparison_df) > 1:
                    f1_values = comparison_df["🏆 F1-Score"]
                    performance_spread = f1_values.max() - f1_values.min()
                    
                    if performance_spread < 0.01:
                        spread_assessment = "🟢 Very similar performance across models"
                        spread_color = "success"
                    elif performance_spread < 0.05:
                        spread_assessment = "🟡 Minor performance differences"
                        spread_color = "warning"
                    elif performance_spread < 0.1:
                        spread_assessment = "🟠 Moderate performance variation"
                        spread_color = "info"
                    else:
                        spread_assessment = "🔴 Significant performance differences"
                        spread_color = "error"
                    
                    # Best metric identification
                    best_f1_model = f1_values.idxmax()
                    
                    # Create insights data for modern table
                    insights_data = [
                        {
                            "📊 Insight Type": "📈 Performance Spread",
                            "📋 Details": spread_assessment,
                            "📏 Value": f"{performance_spread:.4f}",
                            "🎯 Significance": "Model consistency indicator"
                        },
                        {
                            "📊 Insight Type": "🎯 F1-Score Range",
                            "📋 Details": f"Range: {f1_values.min():.4f} to {f1_values.max():.4f}",
                            "📏 Value": f"Δ {f1_values.max() - f1_values.min():.4f}",
                            "🎯 Significance": "Performance distribution span"
                        },
                        {
                            "📊 Insight Type": "🏆 Best Overall Model",
                            "📋 Details": f"{best_f1_model}",
                            "📏 Value": f"{f1_values.max():.4f}",
                            "🎯 Significance": "Highest F1-Score achieved"
                        }
                    ]
                    
                    # Create modern insights table
                    insights_df = pd.DataFrame(insights_data)
                    
                    # Modern styling for insights table
                    styled_insights = insights_df.style.set_properties(**{
                        'padding': '10px',
                        'font-size': '13px',
                        'text-align': 'center',
                        'border': '1px solid var(--text-color-secondary)',
                        'font-weight': 'bold'
                    }).set_table_styles([
                        {'selector': 'th', 'props': [
                            ('background-color', '#6f42c1'),  # Purple header matching comparison table
                            ('color', 'white'),
                            ('font-weight', 'bold'),
                            ('text-align', 'center'),
                            ('padding', '12px'),
                            ('border', '1px solid #6f42c1'),
                            ('font-size', '14px')
                        ]},
                        {'selector': 'td', 'props': [
                            ('text-align', 'center'),
                            ('vertical-align', 'middle'),
                            ('border', '1px solid var(--text-color-secondary)')
                        ]},
                        {'selector': '', 'props': [
                            ('border-collapse', 'collapse'),
                            ('margin', '10px 0'),
                            ('border-radius', '8px'),
                            ('overflow', 'hidden'),
                            ('box-shadow', '0 3px 6px rgba(0, 0, 0, 0.1)')
                        ]},
                        # Highlight best overall model row
                        {'selector': 'tbody tr:last-child', 'props': [
                            ('background-color', 'rgba(111, 66, 193, 0.1)')
                        ]}
                    ])
                    
                    st.dataframe(styled_insights, use_container_width=True, hide_index=True)
                    
                    # Additional summary metrics in a compact card layout
                    st.markdown("**⚡ Summary Metrics:**")
                    
                    summary_col1, summary_col2, summary_col3 = st.columns(3)
                    
                    with summary_col1:
                        # Calculate model consistency score
                        consistency_score = (1 - performance_spread) * 100
                        consistency_grade = "🏆 Excellent" if consistency_score >= 99 else "🥇 Very Good" if consistency_score >= 95 else "🥈 Good" if consistency_score >= 90 else "🥉 Fair"
                        
                        # Custom HTML for smaller font
                        st.markdown(f"""
                        <div style="text-align: center; padding: 1rem; border: 1px solid #ddd; border-radius: 0.5rem;">
                            <div style="color: #666; font-size: 0.75rem; margin-bottom: 0.25rem;">🎯 Model Consistency</div>
                            <div style="font-size: 1rem; font-weight: bold; margin-bottom: 0.25rem;">{consistency_score:.1f}%</div>
                            <div style="color: #28a745; font-size: 0.75rem;">{consistency_grade}</div>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with summary_col2:
                        # Performance tier analysis
                        high_performers = sum(1 for score in f1_values if score >= 0.9)
                        performance_tier = f"{high_performers}/{len(f1_values)} High Performers"
                        tier_status = "🟢 Excellent" if high_performers == len(f1_values) else "🟡 Good" if high_performers >= len(f1_values)/2 else "🔴 Mixed"
                        
                        # Custom HTML for smaller font
                        st.markdown(f"""
                        <div style="text-align: center; padding: 1rem; border: 1px solid #ddd; border-radius: 0.5rem;">
                            <div style="color: #666; font-size: 0.75rem; margin-bottom: 0.25rem;">📊 Performance Tier</div>
                            <div style="font-size: 1rem; font-weight: bold; margin-bottom: 0.25rem;">{performance_tier}</div>
                            <div style="color: #28a745; font-size: 0.75rem;">{tier_status}</div>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with summary_col3:
                        # Improvement potential
                        max_possible = 1.0
                        current_best = f1_values.max()
                        improvement_potential = (max_possible - current_best) * 100
                        
                        if improvement_potential < 1:
                            potential_status = "🏆 Minimal"
                        elif improvement_potential < 5:
                            potential_status = "🟢 Low"
                        elif improvement_potential < 10:
                            potential_status = "🟡 Moderate"
                        else:
                            potential_status = "🔴 High"
                        
                        # Custom HTML for smaller font
                        st.markdown(f"""
                        <div style="text-align: center; padding: 1rem; border: 1px solid #ddd; border-radius: 0.5rem;">
                            <div style="color: #666; font-size: 0.75rem; margin-bottom: 0.25rem;">📈 Improvement Potential</div>
                            <div style="font-size: 1rem; font-weight: bold; margin-bottom: 0.25rem;">{improvement_potential:.1f}%</div>
                            <div style="color: #28a745; font-size: 0.75rem;">{potential_status}</div>
                        </div>
                        """, unsafe_allow_html=True)                
                else:
                    # Single model case with modern styling
                    single_insights_data = [{
                        "📊 Insight Type": "📊 Analysis Status",
                        "📋 Details": "Need multiple models for comparative analysis",
                        "📏 Value": "1 Model",
                        "🎯 Significance": "Train more models to enable insights"
                    }]
                    
                    single_insights_df = pd.DataFrame(single_insights_data)
                    
                    # Style single model insights table
                    styled_single_insights = single_insights_df.style.set_properties(**{
                        'padding': '10px',
                        'font-size': '13px',
                        'text-align': 'center',
                        'border': '1px solid var(--text-color-secondary)',
                        'font-weight': 'bold'
                    }).set_table_styles([
                        {'selector': 'th', 'props': [
                            ('background-color', '#ffc107'), 
                            ('color', 'black'),
                            ('font-weight', 'bold'),
                            ('text-align', 'center'),
                            ('padding', '12px'),
                            ('border', '1px solid #ffc107'),
                            ('font-size', '14px')
                        ]},
                        {'selector': 'td', 'props': [
                            ('text-align', 'center'),
                            ('vertical-align', 'middle'),
                            ('border', '1px solid var(--text-color-secondary)')
                        ]},
                        {'selector': '', 'props': [
                            ('border-collapse', 'collapse'),
                            ('margin', '10px 0'),
                            ('border-radius', '8px'),
                            ('overflow', 'hidden'),
                            ('box-shadow', '0 3px 6px rgba(0, 0, 0, 0.1)')
                        ]}
                    ])
                    
                    st.dataframe(styled_single_insights, use_container_width=True, hide_index=True)
            
            # Enhanced interactive chart section with more chart types
            st.markdown("#### 📈 **Advanced Performance Visualization Suite**")
            
            chart_control_col1, chart_control_col2 = st.columns([1, 2])
            
            with chart_control_col1:
                chart_type = st.selectbox(
                    "📊 Visualization Type:",
                    options=[
                        "Bar Chart", 
                        "Line Chart", 
                        "Radar Chart", 
                        "Scatter Plot",
                        "Heatmap",
                        "Box Plot",
                        "Violin Plot",
                        "Parallel Coordinates",
                        "Bubble Chart",
                        "Sunburst Chart"
                    ],
                    key="comparison_chart_type",
                    help="Choose how to visualize model performance comparison"
                )
            
            with chart_control_col2:
                chart_options = [col for col in comparison_df.columns if comparison_df[col].dtype in ['float64', 'float32']]
                selected_metrics_for_chart = st.multiselect(
                    "Select metrics to visualize:",
                    options=chart_options,
                    default=["🏆 F1-Score", "🎯 Accuracy"] if all(m in chart_options for m in ["🏆 F1-Score", "🎯 Accuracy"]) else chart_options[:2],
                    key="chart_metrics_selection",
                    help="Choose which performance metrics to include in the visualization"
                )
            
            # Chart generation
            if selected_metrics_for_chart:
                chart_data = comparison_df[selected_metrics_for_chart]
                
                # Import plotly for advanced charts
                try:
                    import plotly.express as px
                    import plotly.graph_objects as go
                    from plotly.subplots import make_subplots
                    
                    if chart_type == "Bar Chart":
                        st.bar_chart(chart_data, use_container_width=True)
                    
                    elif chart_type == "Line Chart":
                        st.line_chart(chart_data, use_container_width=True)
                    
                    elif chart_type == "Radar Chart":
                        fig = go.Figure()
                        
                        for model_name in chart_data.index:
                            values = chart_data.loc[model_name].values.tolist()
                            values += [values[0]]  # Close the radar chart
                            
                            fig.add_trace(go.Scatterpolar(
                                r=values,
                                theta=selected_metrics_for_chart + [selected_metrics_for_chart[0]],
                                fill='toself',
                                name=model_name,
                                line=dict(width=3),
                                opacity=0.7
                            ))
                        
                        fig.update_layout(
                            polar=dict(
                                radialaxis=dict(
                                    visible=True,
                                    range=[0, 1],
                                    tickmode='linear',
                                    tick0=0,
                                    dtick=0.2
                                )),
                            showlegend=True,
                            title="Model Performance Radar Comparison",
                            height=600,
                            font=dict(size=14)
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                    
                    elif chart_type == "Scatter Plot":
                        """
                        Bivariate Correlation Analysis Visualization
                        
                        Creates an interactive scatter plot for analyzing relationships between two performance metrics.
                        Essential for identifying correlations and trade-offs between different evaluation criteria.
                        
                        Academic Research Applications:
                        - Precision-Recall trade-off analysis
                        - Bias-Variance relationship exploration
                        - Multi-objective optimization visualization
                        - Performance correlation studies
                        """
                        if len(selected_metrics_for_chart) >= 2:
                            # Prepare scatter plot data with model identification for academic analysis
                            scatter_data = chart_data.reset_index()
                            
                            # Create professional scatter plot with academic presentation standards
                            fig = px.scatter(
                                scatter_data,
                                x=selected_metrics_for_chart[0],                    # X-axis: Primary performance metric
                                y=selected_metrics_for_chart[1],                    # Y-axis: Secondary performance metric
                                hover_name="🤖 Model",                              # Interactive model identification
                                title=f"{selected_metrics_for_chart[1]} vs {selected_metrics_for_chart[0]}",  # Descriptive academic title
                                height=500,                                         # Standardized height for presentation
                                size_max=20                                         # Maximum marker size for visibility
                            )
                            
                            # Apply professional styling for thesis presentation quality
                            fig.update_traces(marker=dict(
                                size=12,                                            # Visible marker size for academic presentation
                                line=dict(width=2, color='DarkSlateGrey')           # Professional marker border styling
                            ))
                            
                            st.plotly_chart(fig, use_container_width=True)
                        else:
                            # Academic requirement validation with user guidance
                            st.warning("Scatter plot requires at least 2 metrics. Please select more metrics.")
                    
                    elif chart_type == "Heatmap":
                        """
                        Comprehensive Performance Matrix Visualization
                        
                        Generates a correlation heatmap displaying all selected metrics across all models,
                        enabling comprehensive comparative analysis and pattern identification.
                        
                        Academic Research Applications:
                        - Comprehensive performance matrix analysis
                        - Model-metric correlation visualization
                        - Performance pattern identification
                        - Comparative algorithm assessment
                        """
                        # Create professional heatmap for comprehensive model comparison
                        fig = px.imshow(
                            chart_data.T,                                           # Transpose for metrics-vs-models layout
                            labels=dict(x="Models", y="Metrics", color="Score"),    # Professional axis labeling
                            title="Performance Heatmap",                           # Academic title
                            color_continuous_scale="Viridis",                      # Colorblind-friendly scale
                            height=500                                              # Standardized presentation height
                        )
                        
                        # Optimize readability for academic presentation
                        fig.update_xaxes(tickangle=45)                            # Angled labels for readability
                        st.plotly_chart(fig, use_container_width=True)
                    
                    elif chart_type == "Box Plot":
                        """
                        Statistical Distribution Analysis Visualization
                        
                        Creates box plots for analyzing performance metric distributions across models,
                        providing insights into variance, outliers, and statistical reliability.
                        
                        Academic Research Applications:
                        - Performance variance analysis
                        - Statistical outlier identification
                        - Distribution comparison studies
                        - Model reliability assessment
                        """
                        fig = go.Figure()
                        
                        # Generate individual box plots for each selected metric
                        for metric in selected_metrics_for_chart:
                            fig.add_trace(go.Box(
                                y=chart_data[metric],                               # Performance values for statistical analysis
                                name=metric,                                        # Metric identification
                                boxpoints='all',                                    # Show all data points for transparency
                                jitter=0.3,                                         # Point spreading for visibility
                                pointpos=-1.8                                       # Position points outside box
                            ))
                        
                        # Apply professional layout for academic presentation
                        fig.update_layout(
                            title="Performance Distribution Box Plot",             # Descriptive academic title
                            yaxis_title="Score",                                   # Professional axis labeling
                            height=500,                                            # Standardized height
                            showlegend=False                                       # Clean presentation without redundant legend
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                    
                    elif chart_type == "Violin Plot":
                        """
                        Advanced Distribution Shape Analysis
                        
                        Creates violin plots combining box plot statistics with kernel density estimation,
                        providing detailed insights into performance distribution characteristics.
                        
                        Academic Research Applications:
                        - Distribution shape analysis
                        - Multi-modal performance detection
                        - Advanced statistical visualization
                        - Comprehensive density estimation
                        """
                        fig = go.Figure()
                        
                        # Generate violin plots for detailed distribution analysis
                        for metric in selected_metrics_for_chart:
                            fig.add_trace(go.Violin(
                                y=chart_data[metric],                               # Performance data for density estimation
                                name=metric,                                        # Metric identification
                                box_visible=True,                                   # Include box plot for statistical summary
                                meanline_visible=True                               # Show mean line for central tendency
                            ))
                        
                        # Professional layout configuration for academic standards
                        fig.update_layout(
                            title="Performance Distribution Violin Plot",          # Academic title with methodology reference
                            yaxis_title="Score",                                   # Professional axis labeling
                            height=500                                             # Standardized presentation height
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                    
                    elif chart_type == "Parallel Coordinates":
                        """
                        Multi-dimensional Performance Analysis Visualization
                        
                        Creates parallel coordinates plot for analyzing relationships across multiple metrics simultaneously,
                        enabling comprehensive multi-criteria decision analysis and pattern recognition.
                        
                        Academic Research Applications:
                        - Multi-criteria decision analysis (MCDA)
                        - High-dimensional performance visualization
                        - Trade-off analysis across multiple objectives
                        - Pareto efficiency identification
                        """
                        # Prepare data structure for parallel coordinates analysis
                        parallel_data = chart_data.reset_index()
                        
                        # Create sophisticated parallel coordinates visualization
                        fig = go.Figure(data=
                            go.Parcoords(
                                line=dict(
                                    color=range(len(parallel_data)),               # Color coding for model identification
                                    colorscale='Viridis',                          # Professional colorblind-friendly scale
                                    showscale=True                                 # Include color scale for reference
                                ),
                                dimensions=list([
                                    dict(
                                        range=[0, 1],                              # Normalized range for comparison
                                        label=metric,                              # Metric axis labeling
                                        values=parallel_data[metric]               # Performance values
                                    ) for metric in selected_metrics_for_chart
                                ])
                            )
                        )
                        
                        # Professional layout for academic presentation
                        fig.update_layout(
                            title="Parallel Coordinates Plot",                    # Academic methodology title
                            height=500                                            # Standardized presentation height
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                    
                    elif chart_type == "Bubble Chart":
                        """
                        Three-Dimensional Performance Relationship Analysis
                        
                        Creates bubble chart visualization for analyzing relationships between three performance metrics,
                        where bubble size represents the third dimension of analysis.
                        
                        Academic Research Applications:
                        - Three-dimensional trade-off analysis
                        - Multi-objective optimization visualization
                        - Performance efficiency frontier analysis
                        - Complex relationship exploration
                        """
                        if len(selected_metrics_for_chart) >= 3:
                            # Prepare three-dimensional bubble chart data
                            bubble_data = chart_data.reset_index()
                            
                            # Create sophisticated bubble chart for multi-dimensional analysis
                            fig = px.scatter(
                                bubble_data,
                                x=selected_metrics_for_chart[0],                    # Primary performance metric (X-axis)
                                y=selected_metrics_for_chart[1],                    # Secondary performance metric (Y-axis)
                                size=selected_metrics_for_chart[2],                 # Tertiary metric represented by bubble size
                                hover_name="🤖 Model",                              # Interactive model identification
                                title=f"Bubble Chart: {selected_metrics_for_chart[0]} vs {selected_metrics_for_chart[1]} (size: {selected_metrics_for_chart[2]})",
                                height=500,                                         # Standardized presentation height
                                size_max=30                                         # Maximum bubble size for visibility
                            )
                            
                            st.plotly_chart(fig, use_container_width=True)
                        else:
                            # Academic requirement validation with methodological guidance
                            st.warning("Bubble chart requires at least 3 metrics. Please select more metrics.")
                    
                    elif chart_type == "Sunburst Chart":
                        """
                        Hierarchical Performance Decomposition Visualization
                        
                        Creates sunburst chart for hierarchical analysis of model performance across metrics,
                        enabling detailed decomposition and comparative assessment of performance components.
                        
                        Academic Research Applications:
                        - Hierarchical performance decomposition
                        - Component-wise comparative analysis
                        - Multi-level performance assessment
                        - Structured performance evaluation
                        """
                        # Construct hierarchical data structure for sunburst visualization
                        sunburst_data = []
                        
                        # Build hierarchical performance data for academic analysis
                        for model_name in chart_data.index:
                            # Add metric-level performance data
                            for metric in selected_metrics_for_chart:
                                value = chart_data.loc[model_name, metric]
                                sunburst_data.append({
                                    'ids': f"{model_name}_{metric}",               # Unique hierarchical identifier
                                    'labels': metric,                              # Metric label for display
                                    'parents': model_name,                         # Hierarchical parent relationship
                                    'values': value                                # Performance value
                                })
                            
                            # Add model-level aggregated performance data
                            sunburst_data.append({
                                'ids': model_name,                                 # Model identifier
                                'labels': model_name,                              # Model name for display
                                'parents': "",                                     # Root level indicator
                                'values': chart_data.loc[model_name].sum()         # Aggregated performance score
                            })
                        
                        # Convert to structured DataFrame for visualization
                        sunburst_df = pd.DataFrame(sunburst_data)
                        
                        # Create professional hierarchical sunburst visualization
                        fig = go.Figure(go.Sunburst(
                            ids=sunburst_df['ids'],                               # Hierarchical identifiers
                            labels=sunburst_df['labels'],                         # Display labels
                            parents=sunburst_df['parents'],                       # Parent-child relationships
                            values=sunburst_df['values'],                         # Performance values
                            branchvalues="total"                                  # Hierarchical value calculation
                        ))
                        
                        # Professional layout for academic presentation
                        fig.update_layout(
                            title="Performance Sunburst Chart",                   # Academic methodology title
                            height=600                                            # Enhanced height for hierarchical detail
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                
                except ImportError:
                    st.warning("⚠️ Advanced chart types require plotly. Showing basic charts instead.")
                    if chart_type in ["Bar Chart", "Line Chart"]:
                        if chart_type == "Bar Chart":
                            st.bar_chart(chart_data, use_container_width=True)
                        else:
                            st.line_chart(chart_data, use_container_width=True)
                    else:
                        st.info("📦 Install plotly (`pip install plotly`) to use advanced visualization types")
                        st.bar_chart(chart_data, use_container_width=True)
            else:
                st.info("Select at least one metric to display the visualization.")
                                
            # Export options
            st.markdown("#### 💾 **Export Comparison Results**")
            
            export_col1, export_col2, export_col3 = st.columns(3)
            
            with export_col1:
                if st.button("📊 Download CSV", key="export_csv", use_container_width=True):
                    csv_data = comparison_df.to_csv()
                    st.download_button(
                        label="💾 Download Comparison.csv",
                        data=csv_data,
                        file_name=f"model_comparison_{st.session_state.task_type}_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv",
                        use_container_width=True
                    )
            
            with export_col2:
                if st.button("📋 Copy Table", key="copy_table", use_container_width=True):
                    # Create a formatted string for copying
                    table_string = comparison_df.to_string()
                    st.text_area(
                        "Copy this text:",
                        value=table_string,
                        height=200,
                        key="copy_table_text"
                    )
            
            with export_col3:
                if st.button("📸 Save Chart", key="save_chart", use_container_width=True):
                    st.info("💡 Use your browser's screenshot feature or right-click on charts to save them as images.")
        
        else:
            st.info("No successful model runs to compare.")
    
    elif len(st.session_state.experiment_results) == 1:
        # Single model case with enhanced display
        st.markdown("#### 🤖 **Single Model Performance**")
        
        single_model = [res for res in st.session_state.experiment_results if "error" not in res][0]
        
        st.info("🎯 Train multiple models to enable detailed comparisons and competitive analysis.")
        
        # Show single model performance in a nice format
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**📊 Current Model Performance:**")
            st.write(f"**🤖 Model:** {single_model['model_name']}")
            st.write(f"**🎯 Accuracy:** {single_model['accuracy']:.4f}")
            st.write(f"**🏆 F1-Score:** {single_model['f1_score']:.4f}")
        
        with col2:
            st.markdown("**💡 Next Steps:**")
            st.write("• Train additional models for comparison")
            st.write("• Try different algorithms or hyperparameters")
            st.write("• Use the Plugin Development Studio to create custom models")
            st.write("• Explore different evaluation metrics")
    
    else:
        # No models case with guidance
        st.markdown("#### 🚀 **Get Started with Model Comparison**")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("**📤 1. Upload Data**")
            st.write("• Load your dataset")
            st.write("• Select target column")
            st.write("• Choose task type")
        
        with col2:
            st.markdown("**🤖 2. Select Models**")
            st.write("• Choose multiple algorithms")
            st.write("• Configure hyperparameters")
            st.write("• Add evaluation metrics")
        
        with col3:
            st.markdown("**🚀 3. Train & Compare**")
            st.write("• Train selected models")
            st.write("• Compare performance")
            st.write("• Visualize results")
        
        st.info("👆 Upload a dataset and select algorithms to get started with model comparison!")

# Update the plugin system status section
if PLUGINS_AVAILABLE:
    st.markdown("---")
    
    # Modern Plugin System Status with enhanced design
    st.markdown("### 🔌 **Plugin Ecosystem Overview**")
    
    # Create a modern status dashboard
    status_container = st.container()
    
    with status_container:
        # Main metrics in a professional card layout
        col1, col2, col3, col4 = st.columns(4)
        
        # ML Algorithms Status
        with col1:
            try:
                if hasattr(plugin_manager, 'get_statistics'):
                    plugin_stats = plugin_manager.get_statistics()
                    plugin_count = plugin_stats.get('total_plugins', 0)
                    plugin_errors = plugin_stats.get('errors', 0)
                elif hasattr(plugin_manager, '_loaded_plugins'):
                    plugin_count = len(plugin_manager._loaded_plugins)
                    plugin_errors = 0
                else:
                    available_plugins = plugin_manager.get_available_plugins(st.session_state.task_type)
                    plugin_count = len(available_plugins)
                    plugin_errors = 0
                
                # Status indicator
                if plugin_errors > 0:
                    status_color = "🔴"
                    status_text = f"⚠️ {plugin_errors} errors"
                elif plugin_count > 0:
                    status_color = "🟢"
                    status_text = "✅ Active"
                else:
                    status_color = "🟡"
                    status_text = "⚠️ None loaded"
                
                st.metric(
                    label="🤖 ML Algorithms",
                    value=plugin_count,
                    delta=status_text,
                    help="Available machine learning algorithms"
                )
                
            except Exception as e:
                st.metric(
                    label="🤖 ML Algorithms", 
                    value="Error",
                    delta="❌ Loading failed",
                    help=f"Error: {str(e)[:50]}..."
                )
        
        # Evaluation Metrics Status
        with col2:
            try:
                metric_count = 0
                metric_errors = 0
                
                # Try multiple methods to get metric count
                if hasattr(metric_manager, 'get_statistics'):
                    try:
                        metric_stats = metric_manager.get_statistics()
                        metric_count = metric_stats.get('total_metrics', 0)
                        metric_errors = metric_stats.get('errors', 0)
                    except Exception:
                        pass
                
                if metric_count == 0 and hasattr(metric_manager, '_loaded_metrics'):
                    metric_count = len(metric_manager._loaded_metrics)
                
                if metric_count == 0:
                    try:
                        available_metrics = metric_manager.get_available_metrics(st.session_state.task_type)
                        metric_count = len(available_metrics)
                    except Exception:
                        pass
                
                # Status indicator
                if metric_errors > 0:
                    status_color = "🔴"
                    status_text = f"⚠️ {metric_errors} errors"
                elif metric_count > 0:
                    status_color = "🟢"
                    status_text = "✅ Active"
                else:
                    status_color = "🟡"
                    status_text = "⚠️ None loaded"
                
                st.metric(
                    label="📊 Evaluation Metrics",
                    value=metric_count,
                    delta=status_text,
                    help="Available evaluation metrics for model assessment"
                )
                
            except Exception as e:
                st.metric(
                    label="📊 Evaluation Metrics",
                    value="Error", 
                    delta="❌ Loading failed",
                    help=f"Error: {str(e)[:50]}..."
                )
        
        # Visualization Plugins Status
        with col3:
            if VISUALIZATION_PLUGINS_AVAILABLE:
                try:
                    viz_manager = get_visualization_manager()
                    viz_count = 0
                    viz_errors = 0
                    
                    if hasattr(viz_manager, 'get_plugin_statistics'):
                        try:
                            viz_stats = viz_manager.get_plugin_statistics()
                            viz_count = viz_stats.get('total_plugins', 0)
                            viz_errors = viz_stats.get('errors', 0)
                        except Exception:
                            pass
                    
                    if viz_count == 0 and hasattr(viz_manager, '_loaded_plugins'):
                        viz_count = len(viz_manager._loaded_plugins)
                    
                    # Status indicator
                    if viz_errors > 0:
                        status_text = f"⚠️ {viz_errors} errors"
                    elif viz_count > 0:
                        status_text = "✅ Active"
                    else:
                        status_text = "⚠️ None loaded"
                    
                    st.metric(
                        label="📈 Visualizations",
                        value=viz_count,
                        delta=status_text,
                        help="Available visualization plugins"
                    )
                    
                except Exception as e:
                    st.metric(
                        label="📈 Visualizations",
                        value="Error",
                        delta="❌ Loading failed", 
                        help=f"Error: {str(e)[:50]}..."
                    )
            else:
                st.metric(
                    label="📈 Visualizations",
                    value="N/A",
                    delta="❌ Unavailable",
                    help="Visualization system not available"
                )
        
        # System Health Status
        with col4:
            try:
                # Calculate overall system health
                total_components = 3  # Algorithms, Metrics, Visualizations
                healthy_components = 0
                
                # Check each component
                if plugin_count > 0:
                    healthy_components += 1
                if metric_count > 0:
                    healthy_components += 1
                if VISUALIZATION_PLUGINS_AVAILABLE and viz_count > 0:
                    healthy_components += 1
                
                health_percentage = (healthy_components / total_components) * 100
                
                if health_percentage >= 100:
                    health_status = "🟢 Excellent"
                    health_color = "🟢"
                elif health_percentage >= 66:
                    health_status = "🟡 Good"
                    health_color = "🟡"
                elif health_percentage >= 33:
                    health_status = "🟠 Fair"
                    health_color = "🟠"
                else:
                    health_status = "🔴 Poor"
                    health_color = "🔴"
                
                st.metric(
                    label="🏥 System Health",
                    value=f"{health_percentage:.0f}%",
                    delta=health_status,
                    help=f"Overall plugin system health: {healthy_components}/{total_components} components active"
                )
                
            except Exception:
                st.metric(
                    label="🏥 System Health",
                    value="Unknown",
                    delta="❓ Cannot assess",
                    help="Unable to determine system health"
                )
    
    # Modern status table with plugin details
    st.markdown("#### 📋 **Component Details**")
    
    # Create detailed status table
    status_data = []
    
    # Add algorithm status
    try:
        algo_categories = plugin_manager.get_plugins_by_category(st.session_state.task_type)
        for category, plugins in algo_categories.items():
            status_data.append({
                "🔧 Component": f"🤖 Algorithms - {category}",
                "📊 Count": len(plugins),
                "🎯 Task Type": st.session_state.task_type.title(),
                "✅ Status": "🟢 Active" if len(plugins) > 0 else "🔴 Empty"
            })
    except Exception:
        status_data.append({
            "🔧 Component": "🤖 ML Algorithms",
            "📊 Count": "Error",
            "🎯 Task Type": st.session_state.task_type.title(),
            "✅ Status": "🔴 Error"
        })
    
    # Add metrics status
    try:
        metrics_categories = metric_manager.get_metrics_by_category(st.session_state.task_type)
        for category, metrics in metrics_categories.items():
            status_data.append({
                "🔧 Component": f"📊 Metrics - {category}",
                "📊 Count": len(metrics),
                "🎯 Task Type": st.session_state.task_type.title(),
                "✅ Status": "🟢 Active" if len(metrics) > 0 else "🔴 Empty"
            })
    except Exception:
        status_data.append({
            "🔧 Component": "📊 Evaluation Metrics",
            "📊 Count": "Error",
            "🎯 Task Type": st.session_state.task_type.title(),
            "✅ Status": "🔴 Error"
        })
    
    # Create and display the status table
    if status_data:
        status_df = pd.DataFrame(status_data)
        
        # Style the table with modern design
        styled_status = status_df.style.set_properties(**{
            'padding': '12px',
            'font-size': '14px',
            'text-align': 'center',
            'border': '1px solid var(--text-color-secondary)'
        }).set_table_styles([
            {'selector': 'th', 'props': [
                ('background-color', '#20c997'),  # Teal header
                ('color', 'white'),
                ('font-weight', 'bold'),
                ('text-align', 'center'),
                ('padding', '15px'),
                ('border', '1px solid #20c997')
            ]},
            {'selector': 'td', 'props': [
                ('text-align', 'center'),
                ('vertical-align', 'middle'),
                ('border', '1px solid var(--text-color-secondary)')
            ]},
            {'selector': '', 'props': [
                ('border-collapse', 'collapse'),
                ('margin', '15px 0'),
                ('border-radius', '8px'),
                ('overflow', 'hidden'),
                ('box-shadow', '0 2px 4px rgba(0, 0, 0, 0.1)')
            ]}
        ])
        
        st.dataframe(styled_status, use_container_width=True, hide_index=True)
    
    # Quick actions panel
    st.markdown("#### ⚡ **Quick Actions**")
    action_col1, action_col2, action_col3 = st.columns(3)
    
    with action_col1:
        if st.button("🔄 Refresh All Plugins", type="secondary", use_container_width=True):
            with st.spinner("Refreshing all plugins..."):
                try:
                    # Refresh algorithms
                    if hasattr(plugin_manager, '_loaded_plugins'):
                        plugin_manager._loaded_plugins = {}
                    if hasattr(plugin_manager, '_plugin_categories'):
                        plugin_manager._plugin_categories = {}
                    plugin_manager._discover_and_load_plugins()
                    
                    # Refresh metrics
                    if hasattr(metric_manager, '_loaded_metrics'):
                        metric_manager._loaded_metrics = {}
                    if hasattr(metric_manager, '_metric_categories'):
                        metric_manager._metric_categories = {}
                    metric_manager._discover_and_load_metrics()
                    
                    st.success("✅ All plugins refreshed successfully!")
                    st.rerun()
                except Exception as e:
                    st.error(f"❌ Refresh failed: {e}")
    
    with action_col2:
        if st.button("🧹 Clean Temp Files", type="secondary", use_container_width=True):
            try:
                temp_plugins_dir = os.path.join(PROJECT_ROOT, "src", "ml_plugins", "temp_plugins")
                if os.path.exists(temp_plugins_dir):
                    cleaned_count = 0
                    for temp_file in os.listdir(temp_plugins_dir):
                        if temp_file.endswith(('.py', '.pyc')) and temp_file != '__init__.py':
                            temp_file_path = os.path.join(temp_plugins_dir, temp_file)
                            try:
                                os.remove(temp_file_path)
                                cleaned_count += 1
                            except OSError:
                                pass
                    st.success(f"🧹 Cleaned {cleaned_count} temporary files!")
                else:
                    st.info("📁 No temp directory found")
            except Exception as e:
                st.error(f"❌ Cleanup failed: {e}")
    
    with action_col3:
        if st.button("📊 Show Plugin Paths", type="secondary", use_container_width=True):
            with st.expander("📁 Plugin Directory Structure", expanded=True):
                plugin_dirs = {
                    "🤖 Algorithms": os.path.join(PROJECT_ROOT, "src", "ml_plugins", "algorithms"),
                    "📊 Metrics": os.path.join(PROJECT_ROOT, "src", "ml_plugins", "metrics"), 
                    "📈 Visualizations": os.path.join(PROJECT_ROOT, "src", "ml_plugins", "visualizations")
                }
                
                for dir_name, dir_path in plugin_dirs.items():
                    if os.path.exists(dir_path):
                        files = [f for f in os.listdir(dir_path) if f.endswith('.py') and f != '__init__.py']
                        st.write(f"**{dir_name}** ({len(files)} files):")
                        for file in files:
                            st.write(f"  • {file}")
                    else:
                        st.write(f"**{dir_name}**: Directory not found")
                                
    # Show any plugin loading errors
    try:
        total_errors = 0
        error_details = []
        
        # Check plugin manager errors
        if 'plugin_stats' in locals() and isinstance(plugin_stats, dict):
            plugin_errors = plugin_stats.get('errors', 0)
            total_errors += plugin_errors
            if plugin_errors > 0:
                error_details.append(f"ML Algorithms: {plugin_errors}")
        
        # Check metric manager errors - use multiple methods
        metric_errors = 0
        if hasattr(metric_manager, 'get_statistics'):
            try:
                metric_stats = metric_manager.get_statistics()
                metric_errors = metric_stats.get('errors', 0)
            except Exception:
                pass
        
        total_errors += metric_errors
        if metric_errors > 0:
            error_details.append(f"Metrics: {metric_errors}")
        
        # Check visualization errors
        if VISUALIZATION_PLUGINS_AVAILABLE and 'viz_stats' in locals():
            viz_errors = viz_stats.get('errors', 0)
            total_errors += viz_errors
            if viz_errors > 0:
                error_details.append(f"Visualizations: {viz_errors}")
        
        if total_errors > 0:
            st.warning(f"⚠️ {total_errors} plugin loading error(s) detected: {', '.join(error_details)}")
            
    except Exception as e:
        st.warning(f"⚠️ Error checking plugin status: {str(e)}")
        
# Plugin Development Studio
if st.session_state.ml_developer_mode:
    st.markdown("---")
    st.header("🛠️ Plugin Development Studio")
    
    with st.expander("ML Plugin Developer Zone", expanded=True):
        st.subheader("Create Custom ML Components")
        
        # Plugin type selection
        col1, col2 = st.columns(2)
        with col1:
            plugin_type = st.selectbox(
                "Plugin Type:",
                options=["ML Algorithm", "Metric", "Visualization"],
                key="ml_plugin_type_selector"
            )
            st.session_state.ml_plugin_type = plugin_type
        
        with col2:
            # Show type-specific info
            type_info = {
                "ML Algorithm": "Create custom ML models and algorithms",
                "Metric": "Create custom evaluation metrics",
                "Visualization": "Create custom charts and plots"
            }
            st.info(f"**{plugin_type}:** {type_info[plugin_type]}")
        
        # Code editor
        st.markdown("### 📝 Code Editor")
        
        # Get appropriate template based on type
        if plugin_type == "ML Algorithm":
            default_template = ML_PLUGIN_SNIPPETS["ML Algorithm - Basic Template"]
        elif plugin_type == "Metric":
            default_template = ML_PLUGIN_SNIPPETS["Metric - Basic Template"]
        else:  # Visualization
            default_template = ML_PLUGIN_SNIPPETS["Visualization - Basic Template"]
        
        current_code = st.session_state.get("ml_plugin_raw_code", "")
        if not current_code:
            current_code = default_template
        
        try:
            from code_editor import code_editor
            CODE_EDITOR_AVAILABLE = True
        except ImportError:
            CODE_EDITOR_AVAILABLE = False

        # Code editor section
        if CODE_EDITOR_AVAILABLE:
            code_editor_result = code_editor(
                current_code,
                lang="python",
                theme="dark",
                height=[20, 30],
                key="ml_plugin_editor"
            )
            # Store the result properly in session state
            if code_editor_result["text"] != current_code:
                st.session_state.ml_plugin_raw_code = code_editor_result["text"]
        else:
            # Fallback to text_area
            code_editor_text = st.text_area(
                "Plugin Code:",
                value=current_code,
                height=600,
                key="ml_plugin_editor_textarea",
                help="Write your plugin code here. Install streamlit-code-editor for syntax highlighting."
            )
            # Store the text area result properly in session state
            if code_editor_text != current_code:
                st.session_state.ml_plugin_raw_code = code_editor_text
                        
        # Plugin details
        col1, col2 = st.columns(2)
        with col1:
            st.session_state.ml_plugin_class_name = st.text_input(
                "Plugin Class Name:",
                value=st.session_state.get("ml_plugin_class_name", ""),
                key="ml_plugin_class_input"
            )
        
        with col2:
            st.session_state.ml_plugin_display_name = st.text_input(
                "Plugin Display Name:",
                value=st.session_state.get("ml_plugin_display_name", ""),
                key="ml_plugin_display_input"
            )
        
        # Add category selection for ML Algorithms, Metrics, and Visualizations
        if plugin_type == "ML Algorithm":
            st.markdown("**📂 Algorithm Category Selection:**")
            algorithm_categories = [
                "Tree-Based",
                "Linear Models", 
                "Neural Networks",
                "Ensemble Algorithms",
                "Instance-Based",
                "Naive Bayes",
                "SVM",
                "Clustering",
                "Custom"
            ]
            
            # Get current category, defaulting to "Custom" if not set or invalid
            current_category = st.session_state.get("ml_plugin_category", "Custom")
            if current_category not in algorithm_categories:
                current_category = "Custom"
            
            selected_category = st.selectbox(
                "Select Algorithm Category:",
                options=algorithm_categories,
                index=algorithm_categories.index(current_category),
                key="ml_plugin_category_selector",
                help="Choose the category where your algorithm will appear in the Algorithm Selection section"
            )
            st.session_state.ml_plugin_category = selected_category
            
            # Update the template code with the selected category
            if st.session_state.ml_plugin_raw_code:
                # Update category in the code
                updated_code = st.session_state.ml_plugin_raw_code.replace(
                    'self._category = "Custom"',
                    f'self._category = "{selected_category}"'
                )
                if updated_code != st.session_state.ml_plugin_raw_code:
                    st.session_state.ml_plugin_raw_code = updated_code
                    st.rerun()
        
        elif plugin_type == "Metric":
            st.markdown("**📊 Metric Category Selection:**")
            metric_categories = [
                "Classification",
                "Regression", 
                "Clustering",
                "Ranking",
                "Probability",
                "Distance",
                "Information Theory",
                "Statistical",
                "Business",
                "Custom"
            ]
            
            # Get current category, defaulting to "Custom" if not set or invalid
            current_category = st.session_state.get("ml_plugin_category", "Custom")
            if current_category not in metric_categories:
                current_category = "Custom"
            
            selected_category = st.selectbox(
                "Select Metric Category:",
                options=metric_categories,
                index=metric_categories.index(current_category),
                key="ml_plugin_category_selector",
                help="Choose the category where your metric will appear in the Metrics Selection section"
            )
            st.session_state.ml_plugin_category = selected_category
            
            # Update the template code with the selected category
            if st.session_state.ml_plugin_raw_code:
                # Update category in the code
                updated_code = st.session_state.ml_plugin_raw_code.replace(
                    'self._category = "Custom"',
                    f'self._category = "{selected_category}"'
                )
                if updated_code != st.session_state.ml_plugin_raw_code:
                    st.session_state.ml_plugin_raw_code = updated_code
                    st.rerun()
        
        elif plugin_type == "Visualization":
            st.markdown("**📈 Visualization Category Selection:**")
            visualization_categories = [
                "Performance Charts",
                "Data Distribution",
                "Model Comparison", 
                "Feature Analysis",
                "Prediction Analysis",
                "Statistical Plots",
                "Interactive Dashboards",
                "Custom"
            ]
            
            # Get current category, defaulting to "Custom" if not set or invalid
            current_category = st.session_state.get("ml_plugin_category", "Custom")
            if current_category not in visualization_categories:
                current_category = "Custom"
            
            selected_category = st.selectbox(
                "Select Visualization Category:",
                options=visualization_categories,
                index=visualization_categories.index(current_category),
                key="ml_plugin_category_selector",
                help="Choose the category where your visualization will appear in the Visualization Selection section"
            )
            st.session_state.ml_plugin_category = selected_category
            
            # Update the template code with the selected category
            if st.session_state.ml_plugin_raw_code:
                # Update category in the code
                updated_code = st.session_state.ml_plugin_raw_code.replace(
                    'self._category = "Custom"',
                    f'self._category = "{selected_category}"'
                )
                if updated_code != st.session_state.ml_plugin_raw_code:
                    st.session_state.ml_plugin_raw_code = updated_code
                    st.rerun()
        # Action buttons
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("🔍 Validate Code", key="validate_ml_plugin"):
                st.session_state.ml_plugin_validation_results = validate_ml_plugin_code(
                    st.session_state.ml_plugin_raw_code,
                    st.session_state.ml_plugin_class_name,
                    st.session_state.ml_plugin_type
                )
        
        with col2:
            # Create sub-columns for Test and Remove buttons
            test_col, remove_col = st.columns([2, 1])
            
            
            with test_col:
                if st.button("🧪 Test Plugin", key="test_ml_plugin", use_container_width=True):
                    st.session_state.ml_plugin_error = None
                    st.session_state.ml_plugin_save_status = ""
                    
                    code_str = st.session_state.ml_plugin_raw_code
                    class_name_str = st.session_state.ml_plugin_class_name
                    display_name_str = st.session_state.ml_plugin_display_name
                    plugin_type = st.session_state.ml_plugin_type
                    
                    # Enhanced validation with detailed error messages
                    validation_errors = []
                    
                    # Check each field individually
                    if not code_str or not code_str.strip():
                        validation_errors.append("❌ **Plugin Code** is empty - Please write or paste your plugin code")
                    
                    if not class_name_str or not class_name_str.strip():
                        validation_errors.append("❌ **Plugin Class Name** is empty - Please enter the main class name (e.g., 'MyMLAlgorithm')")
                    
                    if not display_name_str or not display_name_str.strip():
                        validation_errors.append("❌ **Plugin Display Name** is empty - Please enter a user-friendly name (e.g., 'My Custom Algorithm')")
                    
                    # Additional validation checks
                    if class_name_str and not class_name_str.replace('_', '').isalnum():
                        validation_errors.append("⚠️ **Plugin Class Name** should only contain letters, numbers, and underscores")
                    
                    if class_name_str and not class_name_str[0].isupper():
                        validation_errors.append("⚠️ **Plugin Class Name** should start with a capital letter (Python convention)")
                    
                    if code_str and f"class {class_name_str}" not in code_str:
                        validation_errors.append(f"❌ **Class Definition Missing** - Your code should contain 'class {class_name_str}'")
                    
                    if code_str and "def get_plugin" not in code_str and plugin_type != "Metric":
                        validation_errors.append("❌ **get_plugin() Function Missing** - Your code must include a 'def get_plugin()' function")
                    elif code_str and plugin_type == "Metric" and "def get_metric_plugin" not in code_str:
                        validation_errors.append("❌ **get_metric_plugin() Function Missing** - Metric plugins must include a 'def get_metric_plugin()' function")
                    
                    # Show detailed validation errors (NOT inside expander since we're already in one)
                    if validation_errors:
                        st.session_state.ml_plugin_error = "Validation failed - please fix the following issues:"
                        
                        # Display errors in an organized way
                        st.error("🚫 **Plugin Validation Failed**")
                        
                        # Use container instead of expander for nested content
                        st.markdown("**📋 Detailed Validation Errors:**")
                        st.markdown("**Please fix the following issues before testing:**")
                        
                        for i, error in enumerate(validation_errors, 1):
                            st.markdown(f"{i}. {error}")
                        
                        # Helpful tips
                        st.markdown("---")
                        st.markdown("**💡 Quick Tips:**")
                        st.markdown("""
                        - **Plugin Code**: Use the code editor above or select a template from snippets
                        - **Class Name**: Should match the class in your code (e.g., `MyMLAlgorithm`)
                        - **Display Name**: User-friendly name that will appear in the UI (e.g., `My Custom Algorithm`)
                        - **Code Structure**: Make sure your code includes the required class and `get_plugin()` function
                        """)
                        
                        # Show current field values for debugging
                        st.markdown("**🔍 Current Field Values:**")
                        st.markdown(f"- **Plugin Type**: `{plugin_type}`")
                        st.markdown(f"- **Class Name**: `{repr(class_name_str)}` ({'✅ Set' if class_name_str else '❌ Empty'})")
                        st.markdown(f"- **Display Name**: `{repr(display_name_str)}` ({'✅ Set' if display_name_str else '❌ Empty'})")
                        st.markdown(f"- **Code Length**: {len(code_str)} characters ({'✅ Has content' if code_str else '❌ Empty'})")
                        
                        if code_str:
                            # Basic code analysis
                            has_class = f"class {class_name_str}" in code_str if class_name_str else False
                            has_get_plugin = "def get_plugin" in code_str
                            has_imports = any(keyword in code_str for keyword in ["import", "from"])
                            
                            st.markdown("**📊 Code Analysis:**")
                            st.markdown(f"- **Class Definition Found**: {'✅ Yes' if has_class else '❌ No'}")
                            st.markdown(f"- **get_plugin() Function**: {'✅ Found' if has_get_plugin else '❌ Missing'}")
                            st.markdown(f"- **Has Imports**: {'✅ Yes' if has_imports else '⚠️ No imports detected'}")
                            st.markdown(f"- **Lines of Code**: {len(code_str.splitlines())}")
                
                    else:
                        # All validations passed, proceed with testing
                        st.success("✅ **Initial validation passed** - Proceeding with plugin testing...")
                        
                        # Test load the plugin
                        try:
                            with st.spinner("🔄 Testing plugin loading..."):
                                # Create temp directory structure
                                temp_dir = os.path.join(PROJECT_ROOT, "src", "ml_plugins", "temp_plugins")
                                os.makedirs(temp_dir, exist_ok=True)
                                
                                st.info(f"📁 Creating temporary plugin file in: {temp_dir}")
                                
                                # Fix the import paths for testing
                                code_to_test = code_str
                                
                                if plugin_type == "ML Algorithm":
                                    # Replace relative import with absolute import for testing
                                    code_to_test = code_to_test.replace(
                                        "from ....base_ml_plugin import MLPlugin",
                                        f"import sys; sys.path.append(r'{PROJECT_ROOT}'); from src.ml_plugins.base_ml_plugin import MLPlugin"
                                    )
                                    st.info("🔧 Fixed ML Algorithm imports for testing")
                                elif plugin_type == "Metric":
                                    code_to_test = code_to_test.replace(
                                        "from ....base_metric import MetricPlugin",
                                        f"import sys; sys.path.append(r'{PROJECT_ROOT}'); from src.ml_plugins.base_metric import MetricPlugin"
                                    )
                                    st.info("🔧 Fixed Metric imports for testing")
                                else:  # Visualization
                                    code_to_test = code_to_test.replace(
                                        "from ....base_visualization import BaseVisualization",
                                        f"import sys; sys.path.append(r'{PROJECT_ROOT}'); from src.ml_plugins.base_visualization import BaseVisualization"
                                    )
                                    st.info("🔧 Fixed Visualization imports for testing")
                                
                                # Create temporary file with correct naming convention
                                if plugin_type == "Metric":
                                    temp_suffix = "_metric.py"  # Use _metric for consistency
                                else:
                                    temp_suffix = "_plugin.py"
                                
                                with tempfile.NamedTemporaryFile(
                                    mode="w",
                                    suffix=temp_suffix,
                                    delete=False,
                                    dir=temp_dir,
                                    encoding="utf-8"
                                ) as tmp_file:
                                    tmp_file.write(code_to_test)
                                    temp_file_path = tmp_file.name
                                
                                st.info(f"📝 Temporary plugin file created: {os.path.basename(temp_file_path)}")
                                
                                # Load and test
                                temp_module_name = f"test_ml_plugin_{os.urandom(4).hex()}"
                                st.info(f"🔄 Loading module as: {temp_module_name}")
                                
                                spec = importlib.util.spec_from_file_location(temp_module_name, temp_file_path)
                                
                                if spec and spec.loader:
                                    st.info("✅ Module specification created successfully")
                                    
                                    module = importlib.util.module_from_spec(spec)
                                    st.info("✅ Module object created")
                                    
                                    spec.loader.exec_module(module)
                                    st.info("✅ Module executed successfully")
                                    
                                    # Check for correct factory function based on plugin type
                                    if plugin_type == "Metric":
                                        factory_function_name = "get_metric_plugin"
                                    else:
                                        factory_function_name = "get_plugin"
                                    
                                    if hasattr(module, factory_function_name):
                                        st.info(f"✅ {factory_function_name}() function found")
                                        
                                        test_instance = getattr(module, factory_function_name)()
                                        st.info(f"✅ Plugin instance created: {type(test_instance).__name__}")
                                        
                                        st.session_state.ml_plugin_test_instance = test_instance
                                        
                                        # Store for testing integration
                                        test_plugin_key = f"[TEST] {display_name_str}"
                                        st.session_state[f"test_plugin_{test_plugin_key}"] = test_instance
                                        
                                        st.success(f"🎉 **{plugin_type} plugin '{display_name_str}' loaded successfully!**")
                                        
                                        # Test basic functionality with detailed logging
                                        try:
                                            if plugin_type == "ML Algorithm":
                                                st.info("🧪 Testing ML Algorithm methods...")
                                                name = test_instance.get_name()
                                                desc = test_instance.get_description()
                                                category = test_instance.get_category()
                                                
                                                # Test hyperparameter config
                                                try:
                                                    test_hyperparams = test_instance.get_hyperparameter_config("test_prefix")
                                                    st.info(f"✅ Hyperparameter config test passed: {len(test_hyperparams)} parameters")
                                                except Exception as hp_error:
                                                    st.warning(f"⚠️ Hyperparameter config test failed: {hp_error}")
                                                
                                                # Test model creation
                                                try:
                                                    test_model = test_instance.create_model_instance({})
                                                    st.info(f"✅ Model instance creation test passed: {type(test_model).__name__}")
                                                except Exception as model_error:
                                                    st.warning(f"⚠️ Model creation test failed: {model_error}")
                                                
                                                st.info(f"📝 **Plugin Name**: {name}")
                                                st.info(f"📋 **Description**: {desc}")
                                                st.info(f"📂 **Category**: {category}")
                                                
                                                # Add to test selection
                                                st.session_state.test_plugin_available = True
                                                st.session_state.test_plugin_name = test_plugin_key
                                                st.session_state.test_plugin_instance = test_instance
                                                
                                                st.success("🚀 **Plugin is ready for testing!** You can now select it in the Algorithm Selection section.")
                                                
                                            elif plugin_type == "Metric":
                                                st.info("🧪 Testing Metric methods...")
                                                name = test_instance.get_name()
                                                desc = test_instance.get_description()
                                                value_range = test_instance.get_value_range()
                                                
                                                # Test metric calculation with dummy data
                                                try:
                                                    import numpy as np
                                                    dummy_y_true = np.array([0, 1, 0, 1])
                                                    dummy_y_pred = np.array([0, 1, 1, 1])
                                                    test_result = test_instance.calculate(dummy_y_true, dummy_y_pred)
                                                    st.info(f"✅ Metric calculation test passed: {test_result}")
                                                except Exception as calc_error:
                                                    st.warning(f"⚠️ Metric calculation test failed: {calc_error}")
                                                
                                                st.info(f"📝 **Plugin Name**: {name}")
                                                st.info(f"📋 **Description**: {desc}")
                                                st.info(f"📊 **Value Range**: {value_range}")
                                                
                                                # Add to test selection for metrics
                                                st.session_state.test_plugin_available = True
                                                st.session_state.test_plugin_name = test_plugin_key
                                                st.session_state.test_plugin_instance = test_instance
                                                
                                                st.success("🚀 **Metric plugin is ready for testing!** You can now select it in the Metrics Selection section.")
                                                
                                            elif plugin_type == "Visualization":
                                                st.info("🧪 Testing Visualization methods...")
                                                name = test_instance.get_name()
                                                desc = test_instance.get_description()
                                                supported_types = test_instance.get_supported_data_types()
                                                
                                                # Test configuration UI
                                                try:
                                                    test_config = test_instance.get_config_ui("test_prefix")
                                                    st.info(f"✅ Configuration UI test passed: {len(test_config)} config options")
                                                except Exception as config_error:
                                                    st.warning(f"⚠️ Configuration UI test failed: {config_error}")
                                                
                                                st.info(f"📝 **Plugin Name**: {name}")
                                                st.info(f"📋 **Description**: {desc}")
                                                st.info(f"🎯 **Supported Types**: {', '.join(supported_types)}")
                                                
                                                # Add to test selection for visualizations
                                                st.session_state.test_plugin_available = True
                                                st.session_state.test_plugin_name = test_plugin_key
                                                st.session_state.test_plugin_instance = test_instance
                                                
                                                st.success("🚀 **Visualization plugin is ready for testing!** You can now select it in the Visualization section.")
                                                
                                        except Exception as method_error:
                                            st.warning(f"⚠️ **Plugin loaded but some methods failed**: {method_error}")
                                            st.info("💡 The plugin may still work, but some functionality might be limited")
                                    else:
                                        st.error(f"❌ **Plugin missing {factory_function_name}() function**")
                                        st.error(f"Your {plugin_type.lower()} plugin code must include a function named `{factory_function_name}()` that returns an instance of your plugin class")
                                else:
                                    st.error("❌ **Failed to create module specification**")
                                    st.error(f"Could not load plugin from: {temp_file_path}")
                                # Clean up temp file and temp directory
                                if os.path.exists(temp_file_path):
                                    os.remove(temp_file_path)
                                    st.info("🧹 Temporary file cleaned up")
                                
                                # Clean up old temp files in temp_plugins directory
                                try:
                                    temp_plugins_dir = os.path.join(PROJECT_ROOT, "src", "ml_plugins", "temp_plugins")
                                    if os.path.exists(temp_plugins_dir):
                                        for temp_file in os.listdir(temp_plugins_dir):
                                            if temp_file.endswith(('.py', '.pyc')) and temp_file != '__init__.py':
                                                temp_file_full_path = os.path.join(temp_plugins_dir, temp_file)
                                                try:
                                                    os.remove(temp_file_full_path)
                                                except OSError:
                                                    pass  # Ignore if file is in use
                                        st.info("🧹 Cleaned up old temporary plugin files")
                                except Exception:
                                    pass  # Ignore cleanup errors
                                    
                        except Exception as e:
                            error_msg = f"Error testing plugin: {e}"
                            st.session_state.ml_plugin_error = error_msg
                            
                            # Enhanced error reporting (NO NESTED EXPANDER)
                            st.error("❌ **Plugin Testing Failed**")
                            
                            # Use containers instead of expanders for nested content
                            st.markdown("**🐛 Detailed Error Information:**")
                            st.markdown("**Error Details:**")
                            st.code(str(e))
                            
                            # Show error type and location
                            import traceback
                            import pandas as pd
                            st.markdown("**Full Traceback:**")
                            st.code(traceback.format_exc())
                            
                            st.markdown("**💡 Common Solutions:**")
                            st.markdown("""
                            - **Import Errors**: Make sure your base class imports are correct
                            - **Syntax Errors**: Check your Python syntax in the code editor
                            - **Missing Methods**: Ensure all required methods are implemented
                            - **Indentation**: Check that your code is properly indented
                            - **Class Name**: Verify the class name matches what you entered above
                            """)
                            
                            # Clean up on error
                            if 'temp_file_path' in locals() and os.path.exists(temp_file_path):
                                os.remove(temp_file_path)
                                st.info("🧹 Temporary file cleaned up after error")
                            
                            # Clean up temp directory on error too
                            try:
                                temp_plugins_dir = os.path.join(PROJECT_ROOT, "src", "ml_plugins", "temp_plugins")
                                if os.path.exists(temp_plugins_dir):
                                    for temp_file in os.listdir(temp_plugins_dir):
                                        if temp_file.endswith(('.py', '.pyc')) and temp_file != '__init__.py':
                                            temp_file_full_path = os.path.join(temp_plugins_dir, temp_file)
                                            try:
                                                os.remove(temp_file_full_path)
                                            except OSError:
                                                pass
                            except Exception:
                                pass           
            with remove_col:
                # Only show remove button if there's a test plugin available
                if (hasattr(st.session_state, 'test_plugin_available') and 
                    st.session_state.test_plugin_available and 
                    hasattr(st.session_state, 'test_plugin_name')):
                    
                    if st.button("🗑️ Remove", 
                                key="remove_test_plugin", 
                                help=f"Remove test plugin: {st.session_state.test_plugin_name}",
                                type="secondary",
                                use_container_width=True):
                        
                        # Get the test plugin name for cleanup
                        test_plugin_name = st.session_state.test_plugin_name
                        
                        # Remove from session state
                        if hasattr(st.session_state, 'test_plugin_instance'):
                            del st.session_state.test_plugin_instance
                        
                        # Remove from selected plugins if it was selected
                        if test_plugin_name in st.session_state.selected_plugins_config:
                            del st.session_state.selected_plugins_config[test_plugin_name]
                        
                        # Remove the test plugin storage key
                        test_storage_key = f"test_plugin_{test_plugin_name}"
                        if test_storage_key in st.session_state:
                            del st.session_state[test_storage_key]
                        
                        # Reset test plugin flags
                        st.session_state.test_plugin_available = False
                        st.session_state.test_plugin_name = ""
                        
                        # Clear any test-related errors or status
                        st.session_state.ml_plugin_error = None
                        st.session_state.ml_plugin_save_status = ""
                        
                        # Show success message
                        st.success(f"🗑️ Test plugin '{test_plugin_name}' removed successfully!")
                        
                        # Force refresh to update the algorithm selection
                        st.rerun()
                
                else:
                    # Show disabled remove button when no test plugin is available
                    st.button("🗑️ Remove", 
                             key="remove_test_plugin_disabled", 
                             disabled=True,
                             help="No test plugin to remove",
                             use_container_width=True)
        

        with col3:
            if st.button("💾 Save Plugin", key="save_ml_plugin"):
                code_str = st.session_state.ml_plugin_raw_code
                class_name_str = st.session_state.ml_plugin_class_name
                display_name_str = st.session_state.ml_plugin_display_name
                plugin_type = st.session_state.ml_plugin_type
                
                if not all([code_str, class_name_str, display_name_str]):
                    st.session_state.ml_plugin_save_status = "Error: All fields required"
                else:
                    try:
                        # Determine target directory based on plugin type
                        if plugin_type == "ML Algorithm":
                            target_dir = os.path.join(PROJECT_ROOT, "src", "ml_plugins", "algorithms")
                            import_fix = (
                                "from ....base_ml_plugin import MLPlugin",
                                "from ...base_ml_plugin import MLPlugin"
                            )
                        elif plugin_type == "Metric":
                            target_dir = os.path.join(PROJECT_ROOT, "src", "ml_plugins", "metrics")
                            import_fix = (
                                "from ....base_metric import MetricPlugin",
                                "from ...base_metric import MetricPlugin"
                            )
                        else:  # Visualization
                            target_dir = os.path.join(PROJECT_ROOT, "src", "ml_plugins", "visualizations")
                            import_fix = (
                                "from ....base_visualization import BaseVisualization",
                                "from ...base_visualization import BaseVisualization"
                            )
                        
                        # Create target directory if it doesn't exist
                        os.makedirs(target_dir, exist_ok=True)
                        
                        # Create filename with correct suffix based on plugin type
                        safe_name = display_name_str.lower().replace(' ', '_').replace('(', '').replace(')', '').replace('-', '_')
                        
                        # Use correct filename patterns
                        if plugin_type == "Metric":
                            filename = f"{safe_name}_metric.py"
                        elif plugin_type == "ML Algorithm":
                            filename = f"{safe_name}_plugin.py"
                        else:  # Visualization
                            filename = f"{safe_name}_plugin.py"
                        
                        file_path = os.path.join(target_dir, filename)
                        
                        if os.path.exists(file_path):
                            # Ask for confirmation to overwrite
                            st.warning(f"⚠️ File {filename} already exists in {os.path.basename(target_dir)}/")
                            
                            col_cancel, col_overwrite = st.columns(2)
                            with col_cancel:
                                if st.button("❌ Cancel", key="cancel_save"):
                                    st.session_state.ml_plugin_save_status = "Save cancelled"
                                    st.rerun()
                            
                            with col_overwrite:
                                if st.button("✅ Overwrite", key="confirm_overwrite"):
                                    pass  # Continue with save
                                else:
                                    st.stop()
                        
                        # Fix import paths for permanent storage
                        code_to_save = code_str.replace(import_fix[0], import_fix[1])
                        
                        # Also handle the fallback import section
                        if plugin_type == "ML Algorithm":
                            code_to_save = code_to_save.replace(
                                f"sys.path.append(project_root)\n    from src.ml_plugins.base_ml_plugin import MLPlugin",
                                f"sys.path.append(project_root)\n    from ...base_ml_plugin import MLPlugin"
                            )
                        elif plugin_type == "Metric":
                            code_to_save = code_to_save.replace(
                                f"sys.path.append(project_root)\n    from src.ml_plugins.base_metric import MetricPlugin",
                                f"sys.path.append(project_root)\n    from ...base_metric import MetricPlugin"
                            )
                        else:  # Visualization
                            code_to_save = code_to_save.replace(
                                f"sys.path.append(project_root)\n    from src.ml_plugins.base_visualization import BaseVisualization",
                                f"sys.path.append(project_root)\n    from ...base_visualization import BaseVisualization"
                            )
                        
                        # Write the file
                        with open(file_path, "w", encoding="utf-8") as f:
                            f.write(code_to_save)
                        
                        # Success message with directory info
                        relative_path = os.path.relpath(file_path, PROJECT_ROOT)
                        st.session_state.ml_plugin_save_status = f"✅ Success: {plugin_type} saved as {relative_path}"
                        
                        # Auto-refresh plugins after save
                        if PLUGINS_AVAILABLE:
                            try:
                                # Clear plugin cache using correct attributes
                                if plugin_type == "Metric":
                                    # For metrics, refresh the metric manager
                                    if hasattr(metric_manager, '_loaded_metrics'):
                                        metric_manager._loaded_metrics = {}
                                    if hasattr(metric_manager, '_metric_categories'):
                                        metric_manager._metric_categories = {}
                                    
                                    # Use correct method to reload metrics
                                    metric_manager._discover_and_load_metrics()
                                    st.success("✅ Metric plugin saved and automatically loaded!")
                                else:
                                    # For algorithms and visualizations, refresh the plugin manager
                                    if hasattr(plugin_manager, '_loaded_plugins'):
                                        plugin_manager._loaded_plugins = {}
                                    if hasattr(plugin_manager, '_plugin_categories'):
                                        plugin_manager._plugin_categories = {}
                                    
                                    # Use correct method to reload
                                    plugin_manager._discover_and_load_plugins()
                                    st.success("✅ Plugin saved and automatically loaded!")
                                    
                            except Exception as e:
                                st.warning(f"Plugin saved but auto-reload failed: {e}")
                                st.info("Please restart the app to use your new plugin.")
                        
                        # Clean up test plugin if it was saved
                        if hasattr(st.session_state, 'test_plugin_available'):
                            st.session_state.test_plugin_available = False
                            if hasattr(st.session_state, 'test_plugin_name'):
                                test_name = st.session_state.test_plugin_name
                                if test_name in st.session_state.selected_plugins_config:
                                    del st.session_state.selected_plugins_config[test_name]
                        
                        # Clear form after successful save
                        st.session_state.ml_plugin_raw_code = ""
                        st.session_state.ml_plugin_class_name = ""
                        st.session_state.ml_plugin_display_name = ""
                        st.session_state.ml_plugin_test_instance = None
                        
                        st.rerun()
                        
                    except Exception as e:
                        st.session_state.ml_plugin_save_status = f"❌ Error: {e}"
        # Display validation results
        if st.session_state.ml_plugin_validation_results:
            st.markdown("### 📋 Validation Results")
            
            for result in st.session_state.ml_plugin_validation_results:
                status = result['status']
                check = result['check']
                message = result['message']
                
                if "❌ Failed" in status:
                    st.error(f"{status} {check}" + (f": {message}" if message else ""))
                elif "⚠️ Warning" in status:
                    st.warning(f"{status} {check}" + (f": {message}" if message else ""))
                else:
                    st.success(f"{status} {check}" + (f": {message}" if message else ""))
        
        # Display errors
        if st.session_state.ml_plugin_error:
            st.error(f"❌ Error: {st.session_state.ml_plugin_error}")
        
        # Display save status
        if st.session_state.ml_plugin_save_status:
            if "✅ Success" in st.session_state.ml_plugin_save_status:
                st.success(st.session_state.ml_plugin_save_status)
            else:
                st.error(st.session_state.ml_plugin_save_status)

# Plugin snippets viewer
if st.session_state.show_ml_plugin_snippets:
    st.markdown("---")
    with st.expander("📚 ML Plugin Code Snippets", expanded=True):
        st.markdown("Copy these templates to get started with plugin development:")
        
        for snippet_name, snippet_code in ML_PLUGIN_SNIPPETS.items():
            if snippet_name == "--- Select a Snippet ---":
                continue
            
            st.subheader(snippet_name)
            st.code(snippet_code, language="python")
            st.markdown("---")
        
        if st.button("Close Snippets", key="close_ml_snippets"):
            st.session_state.show_ml_plugin_snippets = False
            st.rerun()
               

# Ensure directories exist
ml_plugin_dirs = [
    os.path.join(PROJECT_ROOT, "src", "ml_plugins"),
    os.path.join(PROJECT_ROOT, "src", "ml_plugins", "algorithms"),
    os.path.join(PROJECT_ROOT, "src", "ml_plugins", "metrics"),
    os.path.join(PROJECT_ROOT, "src", "ml_plugins", "visualizations"),
    os.path.join(PROJECT_ROOT, "src", "ml_plugins", "temp_plugins")
]

for dir_path in ml_plugin_dirs:
    os.makedirs(dir_path, exist_ok=True)
    
    # Create __init__.py files if they don't exist
    init_file = os.path.join(dir_path, "__init__.py")
    if not os.path.exists(init_file):
        with open(init_file, "w") as f:
            f.write("# This file makes Python treat the directory as a package\n")