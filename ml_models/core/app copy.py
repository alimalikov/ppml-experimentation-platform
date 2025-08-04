# filepath: c:\Users\alise\OneDrive\Desktop\Bachelor Thesis\ml_models\core\app.py
import streamlit as st
import pandas as pd
import os
import sys
import json
import io
import importlib.util
import numpy as np
import glob                                                                 # For finding plugin files in directory [*_plugin.py]
import traceback                                                            # For error handling and debugging
import ast                                                                  # For parsing and analyzing Python code [used in plugin code validation]
import tempfile                                                             # For creating temporary files
from streamlit_ace import st_ace                                            # For code editor component
from sklearn import datasets                                                # For sample datasets
from sklearn.datasets import make_classification, make_regression           # For dataset generation
from sklearn.datasets import fetch_openml                                   # For fetching datasets
import warnings

# --- Example Plugin Snippets ---
# Predefined code templates for common anonymization patterns designed for educational purposes
# These snippets serve as foundational resources and starting points for custom plugin development
EXAMPLE_PLUGIN_SNIPPETS = {
    "--- Select a Snippet ---": "",  # Placeholder option for UI selection
    
    # Essential template for developing custom anonymization plugins
    "Minimal Plugin Skeleton": '''import streamlit as st
import pandas as pd
from typing import List, Dict, Any
import json
# For test environment, use ...base_anonymizer
# When saved, this will be converted to ..base_anonymizer
from ...base_anonymizer import Anonymizer

class MyNewPlugin(Anonymizer):
    """Template for implementing custom anonymization techniques"""
    def __init__(self):
        self._name = "My New Plugin"  # TODO: Update with technique name
        self._description = "Description of what my new plugin does."  # TODO: Update description

    def get_name(self) -> str:
        return self._name

    def get_description(self) -> str:
        return self._description

    def get_sidebar_ui(self, all_cols: List[str], sa_col_to_pass: str | None, df_raw: pd.DataFrame | None, unique_key_prefix: str) -> Dict[str, Any]:
        st.sidebar.subheader(f"{self.get_name()} Configuration")
        # TODO: Add your UI elements here
        # Example: param = st.sidebar.text_input("My Parameter", key=f"{unique_key_prefix}_my_param")
        st.sidebar.info("Configure your plugin parameters here.")
        return {}  # TODO: Return a dictionary of parameters

    def anonymize(self, df_input: pd.DataFrame, parameters: Dict[str, Any], sa_col: str | None) -> pd.DataFrame:
        """Core anonymization logic - implement your data transformation here"""
        df_anonymized = df_input.copy()
        st.info(f"'{self.get_name()}' anonymize method called. Implement your logic.")
        # TODO: Access parameters: my_param_value = parameters.get("my_param_key_from_get_sidebar_ui")
        # TODO: Modify df_anonymized here
        return df_anonymized

    def build_config_export(self, unique_key_prefix: str, sa_col: str | None) -> Dict[str, Any]:
        """Export current configuration for reproducibility"""
        # TODO: Example: return {"my_param": st.session_state.get(f"{unique_key_prefix}_my_param")}
        return {}

    def apply_config_import(self, config_params: Dict[str, Any], all_cols: List[str], unique_key_prefix: str):
        """Import and apply saved configuration"""
        # TODO: Example: st.session_state[f"{unique_key_prefix}_my_param"] = config_params.get("my_param")
        pass

    def get_export_button_ui(self, config_to_export: dict, unique_key_prefix: str):
        json_string = json.dumps(config_to_export, indent=4)
        st.sidebar.download_button(
            label=f"Export {self.get_name()} Config",
            data=json_string,
            file_name=f"{self.get_name().lower().replace(' ', '_')}_config.json",
            mime="application/json",
            key=f"{unique_key_prefix}_export_button"
        )

    def get_anonymize_button_ui(self, unique_key_prefix: str) -> bool:
        return st.button(f"Anonymize with {self.get_name()}", key=f"{unique_key_prefix}_anonymize_button")

def get_plugin():
    return MyNewPlugin()  # TODO: Update class name
''',

    # Implementation of simple data redaction for sensitive column suppression
    "Basic Redaction Plugin": '''import streamlit as st
import pandas as pd
from typing import List, Dict, Any
import json
from ...base_anonymizer import Anonymizer  # For test, use ...base_anonymizer

class BasicRedactionPlugin(Anonymizer):
    """Simple redaction technique for suppressing sensitive information"""
    def __init__(self):
        self._name = "Basic Redactor"
        self._description = "Replaces values in a selected column with '[REDACTED]'."
        
    def get_name(self) -> str: 
        return self._name
        
    def get_description(self) -> str: 
        return self._description
    
    def get_sidebar_ui(self, all_cols: List[str], sa_col_to_pass: str | None, df_raw: pd.DataFrame | None, unique_key_prefix: str) -> Dict[str, Any]:
        st.sidebar.subheader(f"{self.get_name()} Config")
        col_to_redact = st.sidebar.selectbox(
            "Column to Redact:", 
            options=all_cols if all_cols else ["N/A"], 
            key=f"{unique_key_prefix}_col_redact", 
            help="Select the column whose values will be replaced."
        )
        return {"column_to_redact": col_to_redact}
        
    def anonymize(self, df_input: pd.DataFrame, parameters: Dict[str, Any], sa_col: str | None) -> pd.DataFrame:
        """Apply redaction by replacing all values in selected column with placeholder text"""
        df_copy = df_input.copy()
        col = parameters.get("column_to_redact")
        if col and col != "N/A" and col in df_copy.columns:
            df_copy[col] = "[REDACTED]"
            st.info(f"Redacted column: {col}")
        elif col and col != "N/A":
            st.warning(f"Column '{col}' not found for redaction.")
        else:
            st.warning("No column selected for redaction or no columns available.")
        return df_copy
        
    def build_config_export(self, unique_key_prefix: str, sa_col: str | None) -> Dict[str, Any]:
        return {"column_to_redact": st.session_state.get(f"{unique_key_prefix}_col_redact")}
        
    def apply_config_import(self, config_params: Dict[str, Any], all_cols: List[str], unique_key_prefix: str):
        st.session_state[f"{unique_key_prefix}_col_redact"] = config_params.get("column_to_redact")
        
    def get_export_button_ui(self, config_to_export: dict, unique_key_prefix: str):
        st.sidebar.download_button(
            label=f"Export {self.get_name()} Config", 
            data=json.dumps(config_to_export, indent=2), 
            file_name=f"{self.get_name().lower().replace(' ', '_')}_config.json", 
            mime="application/json", 
            key=f"{unique_key_prefix}_export"
        )
        
    def get_anonymize_button_ui(self, unique_key_prefix: str) -> bool:
        return st.button(f"Anonymize with {self.get_name()}", key=f"{unique_key_prefix}_anon_btn")

def get_plugin():
    return BasicRedactionPlugin()
''',
    
    # Numeric data transformation implementation for quantitative anonymization
    "Numeric Scaler Plugin": '''import streamlit as st
import pandas as pd
from typing import List, Dict, Any
import json
from ...base_anonymizer import Anonymizer  # For test, use ...base_anonymizer

class BasicNumericScalerPlugin(Anonymizer):
    """Numeric scaling technique for quantitative data anonymization"""
    def __init__(self):
        self._name = "Numeric Scaler"
        self._description = "Scales a selected numeric column by a factor."
        self.default_factor = 1.0
        
    def get_name(self) -> str: 
        return self._name
        
    def get_description(self) -> str: 
        return self._description
    
    def get_sidebar_ui(self, all_cols: List[str], sa_col_to_pass: str | None, df_raw: pd.DataFrame | None, unique_key_prefix: str) -> Dict[str, Any]:
        st.sidebar.subheader(f"{self.get_name()} Config")
        # Identify numeric columns for scaling operations
        numeric_cols = []
        if df_raw is not None:
            numeric_cols = df_raw.select_dtypes(include='number').columns.tolist()
        
        col_to_scale = st.sidebar.selectbox(
            "Column to Scale:", 
            options=numeric_cols if numeric_cols else ["N/A"], 
            key=f"{unique_key_prefix}_col_scale", 
            help="Select a numeric column to apply scaling."
        )
        scale_factor = st.sidebar.number_input(
            "Scale Factor:", 
            value=self.default_factor, 
            step=0.1, 
            key=f"{unique_key_prefix}_factor", 
            help="Enter the factor to multiply the column values by."
        )
        return {"column_to_scale": col_to_scale, "scale_factor": scale_factor}
        
    def anonymize(self, df_input: pd.DataFrame, parameters: Dict[str, Any], sa_col: str | None) -> pd.DataFrame:
        """Apply multiplicative scaling to numeric data for anonymization"""
        df_copy = df_input.copy()
        col = parameters.get("column_to_scale")
        factor = parameters.get("scale_factor", self.default_factor)
        if col and col != "N/A" and col in df_copy.columns:
            try:
                df_copy[col] = pd.to_numeric(df_copy[col], errors='coerce') * factor
                st.info(f"Scaled column '{col}' by factor {factor}.")
            except Exception as e:
                st.error(f"Error scaling column '{col}': {e}")
        elif col and col != "N/A":
            st.warning(f"Column '{col}' not found for scaling.")
        else:
            st.warning("No column selected for scaling or no numeric columns available.")
        return df_copy
        
    def build_config_export(self, unique_key_prefix: str, sa_col: str | None) -> Dict[str, Any]:
        return {
            "column_to_scale": st.session_state.get(f"{unique_key_prefix}_col_scale"), 
            "scale_factor": st.session_state.get(f"{unique_key_prefix}_factor")
        }
        
    def apply_config_import(self, config_params: Dict[str, Any], all_cols: List[str], unique_key_prefix: str):
        st.session_state[f"{unique_key_prefix}_col_scale"] = config_params.get("column_to_scale")
        st.session_state[f"{unique_key_prefix}_factor"] = config_params.get("scale_factor", self.default_factor)
        
    def get_export_button_ui(self, config_to_export: dict, unique_key_prefix: str):
        st.sidebar.download_button(
            label=f"Export {self.get_name()} Config", 
            data=json.dumps(config_to_export, indent=2), 
            file_name=f"{self.get_name().lower().replace(' ', '_')}_config.json", 
            mime="application/json", 
            key=f"{unique_key_prefix}_export"
        )
        
    def get_anonymize_button_ui(self, unique_key_prefix: str) -> bool:
        return st.button(f"Anonymize with {self.get_name()}", key=f"{unique_key_prefix}_anon_btn")

def get_plugin():
    return BasicNumericScalerPlugin()
'''
}

def validate_plugin_code(code_str: str, class_name_str: str) -> list:
    """
    Validates plugin code against common issues and requirements using AST.
    """
    results = []
    tree = None

    # 1. Check if code is empty
    if not code_str.strip():
        results.append({
            'status': '❌ Failed',
            'check': 'Code Presence',
            'message': 'No code provided. Plugin code cannot be empty.'
        })
        return results # Cannot proceed if code is empty

    # 2. Basic Python syntax check
    try:
        tree = ast.parse(code_str)
        results.append({
            'status': '✅ Passed',
            'check': 'Python Syntax',
            'message': 'Code is syntactically valid.'
        })
    except SyntaxError as e:
        results.append({
            'status': '❌ Failed',
            'check': 'Python Syntax',
            'message': f"Invalid syntax on line {e.lineno}: {e.msg}"
        })
        return results  # Stop validation if basic syntax is incorrect

    # Initialize flags for checks
    base_anonymizer_imported = False
    get_plugin_defined = False
    plugin_class_defined = False
    inherits_anonymizer = False
    defined_methods = set()

    # Expected elements
    # For permanent plugins, '..base_anonymizer' is standard.
    # For test environment, '...base_anonymizer' might be used due to temp_plugins depth.
    # We will check for either and guide if necessary.
    expected_import_module_level2 = "..base_anonymizer"
    expected_import_module_level3 = "...base_anonymizer"
    expected_import_name = "Anonymizer"
    
    expected_factory_function = "get_plugin"
    
    # These are methods that a typical plugin would override or implement from the Anonymizer interface
    # Add or remove methods based on your Anonymizer base class's abstract methods or expected interface
    expected_class_methods = {
        'get_name', 
        'get_description', 
        'get_sidebar_ui', 
        'anonymize',
        'build_config_export',
        'apply_config_import',
        'get_export_button_ui',
        'get_anonymize_button_ui'
        # __init__ is implicitly checked by class instantiation if get_plugin works
    }

    # 3. Traverse AST for detailed checks
    for node in ast.walk(tree):
        # Check for 'from ..base_anonymizer import Anonymizer' or 'from ...base_anonymizer import Anonymizer'
        if isinstance(node, ast.ImportFrom):
            is_level2_import = node.level == 2 and node.module == "base_anonymizer" # from ..base_anonymizer
            is_level3_import = node.level == 3 and node.module == "base_anonymizer" # from ...base_anonymizer
            
            if is_level2_import or is_level3_import:
                for alias in node.names:
                    if alias.name == expected_import_name:
                        base_anonymizer_imported = True
                        break
            if base_anonymizer_imported: # Found one, no need to check other ImportFrom nodes for this
                break 
                
    # Re-walk for other checks to ensure import is found first if it exists anywhere
    for node in ast.walk(tree):
        # Check for 'get_plugin' factory function
        if isinstance(node, ast.FunctionDef) and node.name == expected_factory_function:
            get_plugin_defined = True

        # Check for plugin class definition and its methods
        elif isinstance(node, ast.ClassDef) and node.name == class_name_str:
            plugin_class_defined = True
            # Check for Anonymizer inheritance
            for base in node.bases:
                if isinstance(base, ast.Name) and base.id == expected_import_name:
                    inherits_anonymizer = True
                    break
            # Collect defined methods within this class
            for sub_node in node.body:
                if isinstance(sub_node, ast.FunctionDef):
                    defined_methods.add(sub_node.name)

    # 4. Compile results based on findings

    # Base Anonymizer Import Check
    if base_anonymizer_imported:
        results.append({
            'status': '✅ Passed',
            'check': 'Base Anonymizer Import',
            'message': f"'{expected_import_name}' is imported from '{expected_import_module_level2}' or '{expected_import_module_level3}'."
        })
    else:
        results.append({
            'status': '❌ Failed',
            'check': 'Base Anonymizer Import',
            'message': f"Missing import: 'from ..base_anonymizer import {expected_import_name}' (recommended) or 'from ...base_anonymizer import {expected_import_name}'."
        })

    # get_plugin() Function Check
    if get_plugin_defined:
        results.append({
            'status': '✅ Passed',
            'check': f"'{expected_factory_function}()' Function",
            'message': 'Factory function is defined.'
        })
    else:
        results.append({
            'status': '❌ Failed',
            'check': f"'{expected_factory_function}()' Function",
            'message': f"Missing '{expected_factory_function}()' factory function."
        })

    # Plugin Class Definition Check
    if plugin_class_defined:
        results.append({
            'status': '✅ Passed',
            'check': f"Class '{class_name_str}' Definition",
            'message': f"Class '{class_name_str}' is defined."
        })

        # Anonymizer Inheritance Check (only if class is defined)
        if inherits_anonymizer:
            results.append({
                'status': '✅ Passed',
                'check': 'Anonymizer Inheritance',
                'message': f"Class '{class_name_str}' correctly inherits from '{expected_import_name}'."
            })
        else:
            results.append({
                'status': '❌ Failed', # This is crucial for functionality
                'check': 'Anonymizer Inheritance',
                'message': f"Class '{class_name_str}' does not appear to inherit from '{expected_import_name}'. Should be 'class {class_name_str}({expected_import_name}):'."
            })

        # Required Methods Check (only if class is defined)
        missing_methods = expected_class_methods - defined_methods
        if not missing_methods:
            results.append({
                'status': '✅ Passed',
                'check': 'Required Class Methods',
                'message': f"All expected methods ({', '.join(sorted(list(expected_class_methods)))}) are present in class '{class_name_str}'."
            })
        else:
            results.append({
                'status': '❌ Failed',
                'check': 'Required Class Methods',
                'message': f"Class '{class_name_str}' is missing methods: {', '.join(sorted(list(missing_methods)))}."
            })
            
    # Plugin class not defined
    else: 
        results.append({
            'status': '❌ Failed',
            'check': f"Class '{class_name_str}' Definition",
            'message': f"Class '{class_name_str}' is not defined in the code."
        })
        # Cannot check inheritance or methods if class is not found
        results.append({
            'status': '❌ Failed',
            'check': 'Anonymizer Inheritance',
            'message': f"Cannot check inheritance; class '{class_name_str}' not found."
        })
        results.append({
            'status': '❌ Failed',
            'check': 'Required Class Methods',
            'message': f"Cannot check methods; class '{class_name_str}' not found."
        })

    # 5. Optional: Check for common anti-patterns or provide style warnings
    # Warn if print() is used, as Streamlit provides better feedback mechanisms.
    if "print(" in code_str:  # Simple check; could be improved with AST for accuracy.
        results.append({
            'status': '⚠️ Warning',
            'check': 'Use of print()',
            'message': "Found 'print()' statements. For user feedback in Streamlit, prefer 'st.write', 'st.info', 'st.warning', 'st.error', or logging."
        })
        
    return results


# --- STREAMLIT APPLICATION CONFIGURATION ---
st.set_page_config(layout="wide")

# --- PROJECT PATH SETUP ---
# Add project root to Python path for relative imports
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# --- CORE ANONYMIZER IMPORT ---
# Import the base anonymizer class that all plugins must inherit from
try:
    from src.anonymizers.base_anonymizer import Anonymizer
except ImportError as e:
    st.error(f"CRITICAL ERROR: Could not import Anonymizer base class. Ensure 'src/anonymizers/base_anonymizer.py' exists. Error: {e}")
    st.stop()

# --- PLUGIN SYSTEM INITIALIZATION ---
# Global registry for loaded anonymization plugins
ANONYMIZER_PLUGINS = {}

# Initialize session state for plugin development and testing functionality
if "test_plugin_instance" not in st.session_state:
    st.session_state.test_plugin_instance = None
if "test_plugin_error" not in st.session_state:
    st.session_state.test_plugin_error = None
if "test_plugin_raw_code" not in st.session_state:
    st.session_state.test_plugin_raw_code = ""
if "test_plugin_class_name" not in st.session_state:
    st.session_state.test_plugin_class_name = ""
if "test_plugin_display_name" not in st.session_state:
    st.session_state.test_plugin_display_name = ""
if "plugin_editor_save_status" not in st.session_state:
    st.session_state.plugin_editor_save_status = ""

# Developer mode and validation state
if "developer_mode_enabled" not in st.session_state:
    st.session_state.developer_mode_enabled = False
if "plugin_validation_results" not in st.session_state:
    st.session_state.plugin_validation_results = []
if "show_code_snippets_expander" not in st.session_state:
    st.session_state.show_code_snippets_expander = False

# --- DATA PERSISTENCE SYSTEM ---
# Maintain uploaded file data across UI interactions and reruns
if "persisted_uploaded_file_bytes" not in st.session_state:
    st.session_state.persisted_uploaded_file_bytes = None
if "persisted_uploaded_file_name" not in st.session_state:
    st.session_state.persisted_uploaded_file_name = None
if "last_uploader_key_persisted" not in st.session_state:
    st.session_state.last_uploader_key_persisted = None

# Persist anonymized results to maintain state between operations
if 'df_anonymized_data' not in st.session_state:
    st.session_state.df_anonymized_data = None
if 'df_anonymized_source_technique' not in st.session_state:
    st.session_state.df_anonymized_source_technique = None

# --- SAMPLE DATASET SYSTEM ---
# Track loaded sample datasets for demonstrations
if "sample_dataset_info" not in st.session_state:
    st.session_state.sample_dataset_info = None
if "loaded_sample_dataset_name" not in st.session_state:
    st.session_state.loaded_sample_dataset_name = None

# --- VALIDATE PLUGIN CODE ---
def validate_plugin_code(code_str, class_name_str):
    """Validate plugin code against common issues and requirements.
    
    This function performs static analysis on plugin code to ensure it meets
    the framework's requirements before attempting to load it dynamically.
    
    Args:
        code_str:           The plugin source code as a string
        class_name_str:     Expected name of the plugin class
        
    Returns:
        List of validation results, each containing status, check name, and message
    """
    results = []
    
    # Check if code is empty
    if not code_str.strip():
        results.append({
            'status': '❌ Failed',
            'check': 'Code is not empty',
            'message': 'No code provided'
        })
        return results
    
    # Basic syntax check - ensures code is valid Python before further analysis
    try:
        ast.parse(code_str)
        results.append({
            'status': '✅ Passed',
            'check': 'Python syntax',
            'message': ''
        })
    except SyntaxError as e:
        results.append({
            'status': '❌ Failed',
            'check': 'Python syntax',
            'message': f"Line {e.lineno}: {e.msg}"
        })
        return results  # Return early on syntax error
    
    # Verify required import for base class inheritance
    if "from ...base_anonymizer import Anonymizer" not in code_str:
        results.append({
            'status': '❌ Failed',
            'check': 'Base Anonymizer import',
            'message': "Missing 'from ...base_anonymizer import Anonymizer'"
        })
    else:
        results.append({
            'status': '✅ Passed',
            'check': 'Base Anonymizer import',
            'message': ''
        })
    
    # Ensure plugin factory function exists for dynamic loading
    if "def get_plugin" not in code_str:
        results.append({
            'status': '❌ Failed',
            'check': 'get_plugin() function',
            'message': "Missing 'def get_plugin()' factory function"
        })
    else:
        results.append({
            'status': '✅ Passed',
            'check': 'get_plugin() function',
            'message': ''
        })
    
    # Verify the expected class name matches what's in the code
    if f"class {class_name_str}" not in code_str:
        results.append({
            'status': '❌ Failed',
            'check': f"Class '{class_name_str}' exists",
            'message': f"Class '{class_name_str}' not found in code"
        })
    else:
        results.append({
            'status': '✅ Passed',
            'check': f"Class '{class_name_str}' exists",
            'message': ''
        })
    
    # Validate presence of required interface methods
    required_methods = ['anonymize', 'get_name', 'get_sidebar_ui']
    missing_methods = []
    
    for method in required_methods:
        if f"def {method}" not in code_str:
            missing_methods.append(method)
    
    if missing_methods:
        results.append({
            'status': '❌ Failed',
            'check': 'Required methods',
            'message': f"Missing: {', '.join(missing_methods)}"
        })
    else:
        results.append({
            'status': '✅ Passed',
            'check': 'Required methods',
            'message': ''
        })
    
    # Additional checks can be added here
    
    # Non-critical check for proper inheritance pattern
    if "class" in code_str and "(Anonymizer)" not in code_str:
        results.append({
            'status': '⚠️ Warning',
            'check': 'Anonymizer inheritance',
            'message': "Class may not inherit from Anonymizer"
        })
    
    return results

# --- LOAD ANONYMIZER PLUGINS ---
def load_anonymizer_plugins(include_test_plugin=True):
    """Dynamically load all anonymizer plugins from the plugins directory.
    
    This function discovers and loads plugin modules following the naming convention
    '*_plugin.py'. It handles module reloading for development and validates each
    plugin before registration in the global ANONYMIZER_PLUGINS dictionary.
    
    Args:
        include_test_plugin: Whether to include the test plugin from session state
    """
    global ANONYMIZER_PLUGINS
    ANONYMIZER_PLUGINS = {} # Reset before loading
    plugin_dir_path = os.path.join(PROJECT_ROOT, "src", "anonymizers", "plugins")

    if not os.path.isdir(plugin_dir_path):
        st.warning(f"Plugin directory not found: {plugin_dir_path}. No file-based plugins will be loaded.")
    else:
        # Discover all plugin files matching the naming convention
        for filepath in glob.glob(os.path.join(plugin_dir_path, "*_plugin.py")):
            module_name = os.path.splitext(os.path.basename(filepath))[0]
            full_module_name = f"src.anonymizers.plugins.{module_name}"
            try:
                if full_module_name in sys.modules: # Ensure fresh import
                    del sys.modules[full_module_name]
                
                # Load module dynamically using importlib
                spec = importlib.util.spec_from_file_location(full_module_name, filepath)
                if spec and spec.loader:
                    module = importlib.util.module_from_spec(spec)
                    sys.modules[full_module_name] = module # Add to sys.modules for relative imports
                    spec.loader.exec_module(module)
                    
                    # Validate and instantiate plugin
                    if hasattr(module, "get_plugin") and callable(module.get_plugin):
                        plugin_instance = module.get_plugin()
                        if isinstance(plugin_instance, Anonymizer) and hasattr(plugin_instance, "get_name") and callable(plugin_instance.get_name):
                            plugin_name = plugin_instance.get_name()
                            if plugin_name in ANONYMIZER_PLUGINS:
                                st.warning(f"Duplicate plugin name '{plugin_name}' from {filepath}. Overwriting.")
                            ANONYMIZER_PLUGINS[plugin_name] = plugin_instance
                        else:
                            st.warning(f"Plugin from {filepath} invalid Anonymizer or missing get_name().")
                    else:
                        st.warning(f"Plugin file {filepath} missing 'get_plugin()' factory.")
                else:
                    st.warning(f"Could not create import spec for plugin {filepath}.")
            except Exception as e:
                st.error(f"Failed to load plugin from {filepath}: {e}")
                st.exception(e)

    # Include dynamically created test plugin if available
    if include_test_plugin and st.session_state.test_plugin_instance is not None and st.session_state.test_plugin_display_name:
        test_plugin_name_key = f"[TEST] {st.session_state.test_plugin_display_name}"
        if test_plugin_name_key in ANONYMIZER_PLUGINS:
            st.warning(f"Test plugin name '{test_plugin_name_key}' conflicts. May be overshadowed.")
        ANONYMIZER_PLUGINS[test_plugin_name_key] = st.session_state.test_plugin_instance

# Initialize plugins on module load
load_anonymizer_plugins()

# --- Helper function for XLSX download ---
@st.cache_data
def convert_df_to_xlsx(df):
    """Convert DataFrame to XLSX format for download.
    
    Uses openpyxl engine to create Excel files with proper formatting.
    Cached to avoid recomputation for identical DataFrames.
    """
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df.to_excel(writer, index=False, sheet_name='Anonymized_Data')
    output.seek(0)
    return output.getvalue()

# --- Helper function for CSV download ---
@st.cache_data
def convert_df_to_csv(df):
    """Convert DataFrame to CSV format using semicolon delimiter for European compatibility."""
    return df.to_csv(index=False, sep=';').encode('utf-8')

# --- Config Import/Export Helper ---
def apply_imported_config_to_ui(cfg: dict, all_cols_from_df: list | None):
    """Apply imported configuration to the UI state.
    
    This function restores a saved anonymization configuration, including
    the selected technique, sensitive attribute, and plugin-specific parameters.
    
    Args:
        cfg: Configuration dictionary containing technique and parameters
        all_cols_from_df: List of available columns from the loaded dataset
        
    Raises:
        ValueError: If config format is invalid or technique is unavailable
    """
    if not isinstance(cfg, dict): raise ValueError("Config must be a dict.")
    if "technique" not in cfg: raise ValueError("Missing 'technique' in config.")

    technique_name = cfg["technique"]
    load_anonymizer_plugins() # Ensure plugins (esp. test plugin) are current
    plugin_to_apply = ANONYMIZER_PLUGINS.get(technique_name)

    if not plugin_to_apply:
        raise ValueError(f"Technique '{technique_name}' not available. Available: {list(ANONYMIZER_PLUGINS.keys())}")

    # Update main UI state
    st.session_state["selected_technique_main"] = technique_name
    st.session_state["sa_col_selector_main"] = cfg.get("sa_col", "<None>") or "<None>"

    plugin_params = cfg.get("parameters", {})
    # Determine plugin_key_prefix for applying config
    plugin_key_prefix_base = plugin_to_apply.get_name().lower().replace(' ', '_').replace('-', '_').replace('(', '').replace(')', '')
    if technique_name.startswith("[TEST]") and st.session_state.test_plugin_display_name:
         # Use the display name for the test plugin's key prefix for consistency
        plugin_key_prefix = f"test_{st.session_state.test_plugin_display_name.lower().replace(' ', '_').replace('-', '_').replace('(', '').replace(')', '')}"
    else:
        plugin_key_prefix = plugin_key_prefix_base

    if all_cols_from_df is None:
         st.warning(f"Applying config for '{technique_name}' without loaded data. Column settings may not init correctly.")
    
    # Delegate parameter restoration to the plugin
    plugin_to_apply.apply_config_import(plugin_params, all_cols_from_df if all_cols_from_df else [], plugin_key_prefix)
    st.sidebar.success(f"Config for '{technique_name}' applied to UI.")


# --- Streamlit App UI ---
st.title("Modular Dataset Anonymization Tool")

# ========================= DEVELOPER MODE TOGGLE IN SIDEBAR ========================= #
# Enable plugin development features for creating and testing custom anonymizers
st.sidebar.checkbox("Enable Plugin Developer Mode", key="developer_mode_enabled_checkbox",
                    on_change=lambda: setattr(st.session_state, 'developer_mode_enabled', st.session_state.developer_mode_enabled_checkbox))

# --- Button to show/hide code snippets ---
if st.sidebar.button("View Plugin Code Snippets", key="toggle_snippets_button"):
    st.session_state.show_code_snippets_expander = not st.session_state.show_code_snippets_expander

st.sidebar.markdown("---") # Separator

# ========================= CONFIG IMPORT / EXPORT UI (Still in Sidebar) ========================= #
st.sidebar.header("Configuration Import / Export")

# Initialize session state for configuration management
if "pending_parsed_config" not in st.session_state: st.session_state.pending_parsed_config = None
if "pending_config_filename" not in st.session_state: st.session_state.pending_config_filename = None
if "last_uploaded_config_name_in_widget" not in st.session_state: st.session_state.last_uploaded_config_name_in_widget = None

# File uploader for importing saved configurations
imported_config_file = st.sidebar.file_uploader("Import configuration (JSON)", type=["json"], key="config_file_uploader_main")
if imported_config_file is not None:
    # Process new config file only if it's different from the last one
    if imported_config_file.name != st.session_state.last_uploaded_config_name_in_widget:
        try:
            imported_config_file.seek(0)
            cfg_json_data = json.load(imported_config_file)
            st.session_state.pending_parsed_config = cfg_json_data
            st.session_state.pending_config_filename = imported_config_file.name
            st.session_state.last_uploaded_config_name_in_widget = imported_config_file.name
            st.sidebar.info(f"Config '{imported_config_file.name}' loaded. Click 'Apply'.")
        except Exception as e:
            st.sidebar.error(f"Failed to parse '{imported_config_file.name}': {e}")
            st.session_state.pending_parsed_config = None
# File removed
elif imported_config_file is None and st.session_state.last_uploaded_config_name_in_widget is not None:
    # Clear pending config when file is removed
    st.session_state.pending_parsed_config = None
    st.session_state.pending_config_filename = None
    st.session_state.last_uploaded_config_name_in_widget = None

# Apply button for loaded configuration
df_columns_for_config_apply = st.session_state.get('df_raw_columns_for_config_apply', None)
if st.session_state.pending_parsed_config and st.session_state.pending_config_filename:
    if st.sidebar.button(f"Apply Config: '{st.session_state.pending_config_filename}'", key="apply_imported_config_button"):
        try:
            apply_imported_config_to_ui(st.session_state.pending_parsed_config, df_columns_for_config_apply)
            st.session_state.pending_parsed_config = None
            st.session_state.pending_config_filename = None
            st.rerun()
        except Exception as e:
            st.sidebar.error(f"Error applying config: {e}")
            st.exception(e)


# ========================= MAIN WIDGETS [Plugin-Driven - Selection in Sidebar] ========================= #

# ========================= CATEGORIZED PLUGIN SELECTION ========================= #
# Organize plugins by category
plugins_by_category = {}
for plugin_name, plugin_instance in ANONYMIZER_PLUGINS.items():
    try:
        category = plugin_instance.get_category() if hasattr(plugin_instance, 'get_category') else "Other"
    except Exception:
        category = "Other"
    
    if category not in plugins_by_category:
        plugins_by_category[category] = []
    plugins_by_category[category].append(plugin_name)

# Sort plugins within each category (Test plugins last within their category)
for category in plugins_by_category:
    plugins_by_category[category].sort(key=lambda x: (x.startswith("[TEST]"), x))

# Get all available categories
available_categories = list(plugins_by_category.keys())
available_categories.sort()

if not available_categories:
    st.error("No anonymization plugins loaded. Check 'src/anonymizers/plugins'.")
    st.stop()

# Initialize session state for category selection
if "selected_category" not in st.session_state:
    st.session_state.selected_category = available_categories[0]

st.sidebar.subheader("🏷️ Plugin Categories")
selected_category = st.sidebar.selectbox(
    "Select Category:",
    options=available_categories,
    key="plugin_category_selector",
    help="Choose a category to filter anonymization techniques"
)

# Get techniques for selected category
techniques_in_category = plugins_by_category.get(selected_category, [])

if not techniques_in_category:
    st.sidebar.error(f"No techniques available in category '{selected_category}'")
    st.stop()

# Show category description
category_descriptions = {
    "Privacy Models": "🔒 Classical privacy models like k-anonymity, l-diversity, t-closeness providing syntactic privacy guarantees",
    "Differential Privacy": "📊 Formal privacy mechanisms providing mathematical privacy guarantees with calibrated noise",
    "Generative Models": "🤖 Machine learning models that generate synthetic data preserving statistical properties",
    "Perturbation Methods": "🌊 Data modification techniques using noise addition and randomization for privacy",
    "Suppression & Generalization": "📝 Data hiding and generalization techniques for privacy protection",
    "Utility Preserving": "⚡ Advanced techniques that maintain data utility while providing privacy",
    "Advanced Techniques": "🔬 Cutting-edge privacy-preserving methods for specialized use cases",
    "Custom/Experimental": "🧪 User-defined or experimental anonymization methods",
    "Other": "📦 Miscellaneous anonymization techniques"
}

if selected_category in category_descriptions:
    st.sidebar.info(category_descriptions[selected_category])

# Show technique count in category
st.sidebar.caption(f"📋 {len(techniques_in_category)} technique(s) in this category")

# Technique selection within category
st.sidebar.subheader("🔧 Anonymization Techniques")

selected_technique_name = st.sidebar.selectbox(
    "Select Technique:",
    options=techniques_in_category,
    key="plugin_technique_selector",
    help="Choose a specific anonymization technique from the selected category"
)

current_plugin = ANONYMIZER_PLUGINS.get(selected_technique_name)

# Clear persisted anonymized data if the technique has changed from the one that generated it
if selected_technique_name != st.session_state.get('df_anonymized_source_technique'):
    st.session_state.df_anonymized_data = None
    st.session_state.df_anonymized_source_technique = None


if not current_plugin:
    st.error(f"Plugin '{selected_technique_name}' not loaded. This shouldn't happen.")
    st.stop()

st.header(f"{selected_technique_name} Technique Options") # Use selected_technique_name for header

# --- File Uploader ---
uploaded_file_key = "file_uploader_main_data"
uploaded_file = st.file_uploader("Upload your CSV or Excel file", type=["csv", "xlsx"], key=uploaded_file_key)
df_raw = None

# --- Sample Datasets Section ---
st.sidebar.markdown("---")
st.sidebar.header("🎲 Sample Datasets")
st.sidebar.markdown("*Try the platform with built-in datasets containing sensitive information*")

# === DATASET SIZE SELECTION ===
st.sidebar.markdown("**Choose Dataset Size:**")
dataset_size_options = {
    "📊 Sample (Quick Test)": "sample",
    "📈 Medium (Performance Test)": "medium", 
    "📉 Full (Real-world Scale)": "full"
}

# Select dataset size from sidebar
selected_size = st.sidebar.selectbox(
    "Dataset size:",
    options=list(dataset_size_options.keys()),
    key="dataset_size_selector",
    help="Choose between sample (1x), medium (3x), or full (10x) dataset sizes"
)

dataset_size_key = dataset_size_options[selected_size]

# Show size info
size_descriptions = {
    "sample": "**Sample Size:** Quick testing with smaller datasets (1,000-2,000 rows)",
    "medium": "**Medium Size:** Performance testing with moderate datasets (3,000-6,000 rows)", 
    "full": "**Full Size:** Real-world scale testing with large datasets (10,000-20,000 rows)"
}

# Display size info in sidebar
if dataset_size_key in size_descriptions:
    st.sidebar.info(size_descriptions[dataset_size_key])

# === CATEGORY FILTER ===
st.sidebar.markdown("**Filter by Industry & Use Case:**")
dataset_categories = {
    "All Datasets": "all",
    "🔬 Classic ML (Small & Real)": "classic_ml",
    "🎯 Binary Classification": "binary_classification", 
    "🌈 Multi-class Classification": "multi_class",
    "📈 Regression": "regression",
    "🏭 Large Scale (10K+ samples)": "large_scale",
    "🏥 Healthcare & Medical": "healthcare",
    "💰 Financial & Banking": "financial",
    "👥 Human Resources": "hr",
    "📊 Research & Analytics": "research"
}

# Select dataset category from sidebar
selected_category = st.sidebar.selectbox(
    "Choose dataset category:",
    options=list(dataset_categories.keys()),
    key="dataset_category_filter",
    help="Filter datasets by their industry domain and use case"
)

# Define comprehensive dataset categorization
if selected_category == "All Datasets":
    sample_dataset_options = {
        "None": None,
        # Classic ML
        "🌸 Iris Flowers": "iris",
        "🍷 Wine Classification": "wine", 
        "🎗️ Breast Cancer": "breast_cancer",
        "🏠 Boston Housing (Synthetic)": "boston_synthetic",
        "💉 Diabetes Progression": "diabetes",
        "🔢 Handwritten Digits": "digits",
        "🏘️ California Housing": "california_housing",
        # Binary Classification
        "💳 Credit Approval (Synthetic)": "credit_approval_synthetic",
        "🚢 Titanic Survival": "titanic",
        "❤️ Heart Disease (Cleveland)": "heart_disease_cleveland",
        "📢 Marketing Campaign (Synthetic)": "marketing_campaign_synthetic",
        # Multi-class
        "🌺 Flower Species (Synthetic)": "flower_species_synthetic",
        "🎵 Music Genres (Synthetic)": "music_genre_synthetic",
        # Regression
        "🚗 Auto MPG": "auto_mpg",
        "📈 Stock Prices (Synthetic)": "stock_prices_synthetic",
        # Large Scale
        "💼 Adult Income (Census)": "adult_income",
        # Healthcare
        "🏥 Medical Records": "medical_records",
        "🧪 Clinical Trials": "clinical_trials", 
        # Financial
        "💰 Financial Transactions": "financial_transactions",
        # HR
        "🏢 Employee Records": "employee_records",
        # Research
        "📊 Survey Responses": "survey_responses",
        "👥 Customer Data": "customer_data"
    }
else:
    # Filter datasets based on selected category
    # Use the key from dataset_categories to determine which datasets to show
    selected_cat_key = dataset_categories[selected_category]
    
    if selected_cat_key == "classic_ml":
        sample_dataset_options = {
            "None": None,
            "🌸 Iris Flowers": "iris",
            "🍷 Wine Classification": "wine", 
            "🎗️ Breast Cancer": "breast_cancer",
            "🏠 Boston Housing (Synthetic)": "boston_synthetic",
            "💉 Diabetes Progression": "diabetes",
            "🔢 Handwritten Digits": "digits",
            "🏘️ California Housing": "california_housing"
        }
    elif selected_cat_key == "binary_classification":
        sample_dataset_options = {
            "None": None,
            "💳 Credit Approval (Synthetic)": "credit_approval_synthetic",
            "🚢 Titanic Survival": "titanic",
            "❤️ Heart Disease (Cleveland)": "heart_disease_cleveland",
            "📢 Marketing Campaign (Synthetic)": "marketing_campaign_synthetic"
        }
    elif selected_cat_key == "multi_class":
        sample_dataset_options = {
            "None": None,
            "🌸 Iris Flowers": "iris",
            "🔢 Handwritten Digits": "digits",
            "🌺 Flower Species (Synthetic)": "flower_species_synthetic",
            "🎵 Music Genres (Synthetic)": "music_genre_synthetic"
        }
    elif selected_cat_key == "regression":
        sample_dataset_options = {
            "None": None,
            "🏠 Boston Housing (Synthetic)": "boston_synthetic",
            "💉 Diabetes Progression": "diabetes",
            "🏘️ California Housing": "california_housing",
            "🚗 Auto MPG": "auto_mpg",
            "📈 Stock Prices (Synthetic)": "stock_prices_synthetic"
        }
    elif selected_cat_key == "large_scale":
        sample_dataset_options = {
            "None": None,
            "💼 Adult Income (Census)": "adult_income",
            "🏘️ California Housing": "california_housing"
        }
    elif selected_cat_key == "healthcare":
        sample_dataset_options = {
            "None": None,
            "🎗️ Breast Cancer": "breast_cancer",
            "💉 Diabetes Progression": "diabetes",
            "❤️ Heart Disease (Cleveland)": "heart_disease_cleveland",
            "🏥 Medical Records": "medical_records",
            "🧪 Clinical Trials": "clinical_trials"
        }
    elif selected_cat_key == "financial":
        sample_dataset_options = {
            "None": None,
            "💳 Credit Approval (Synthetic)": "credit_approval_synthetic",
            "💰 Financial Transactions": "financial_transactions"
        }
    elif selected_cat_key == "hr":
        sample_dataset_options = {
            "None": None,
            "💼 Adult Income (Census)": "adult_income",
            "🏢 Employee Records": "employee_records"
        }
    elif selected_cat_key == "research":
        sample_dataset_options = {
            "None": None,
            "📊 Survey Responses": "survey_responses",
            "👥 Customer Data": "customer_data"
        }
    else:
        sample_dataset_options = {"None": None}

# Show comprehensive category info
if selected_category != "All Datasets":
    category_descriptions = {
        "🔬 Classic ML (Small & Real)": "Well-known datasets from machine learning literature, mostly small and real-world.",
        "🎯 Binary Classification": "Datasets suitable for tasks predicting one of two outcomes (e.g., yes/no, true/false).",
        "🌈 Multi-class Classification": "Datasets for tasks predicting one of several (more than two) categories.",
        "📈 Regression": "Datasets for tasks predicting a continuous numerical value.",
        "🏭 Large Scale (10K+ samples)": "Larger datasets to test performance and scalability, typically over 10,000 samples.",
        "🏥 Healthcare & Medical": "Medical records, clinical trials, pharmacy data with HIPAA-protected information or similar health data.",
        "💰 Financial & Banking": "Banking, loans, credit cards with PCI-DSS regulated financial data or similar financial info.",
        "👥 Human Resources": "Employee records, payroll data with workplace privacy information or salary prediction tasks.",
        "📊 Research & Analytics": "Survey responses, customer analytics with research participant data or general customer behavior."
    }
    
    if selected_category in category_descriptions:
        st.sidebar.info(f"**{selected_category}:** {category_descriptions[selected_category]}")

# Show filtered dataset count with industry context
total_datasets = len([k for k in sample_dataset_options.keys() if k != "None"])
if selected_category == "All Datasets":
    st.sidebar.caption(f"📋 Showing all {total_datasets} datasets across multiple industries")
else:
    st.sidebar.caption(f"📋 Showing {total_datasets} datasets in {selected_category}")

# --- Logic to handle file upload and persistence ---
# Try to restore "df_raw" if uploader is empty but we have persisted data for this uploader key
if uploaded_file is None and \
   st.session_state.persisted_uploaded_file_bytes is not None and \
   st.session_state.last_uploader_key_persisted == uploaded_file_key:
    
    file_bytes_io = io.BytesIO(st.session_state.persisted_uploaded_file_bytes)
    file_name_persisted = st.session_state.persisted_uploaded_file_name
    # Inform user about using persisted data
    st.info(f"Using previously uploaded data: {file_name_persisted}")
    try:
        if file_name_persisted.endswith('.csv'):
            df_raw = pd.read_csv(file_bytes_io, sep=None, engine='python')
        elif file_name_persisted.endswith('.xlsx'):
            df_raw = pd.read_excel(file_bytes_io, engine='openpyxl')

    # Note: "uploaded_file" object is still None. If downstream code strictly needs "uploaded_file.name", we'd use file_name_persisted.
    except Exception as e:
        st.error(f"Error re-processing persisted file '{file_name_persisted}': {e}")
        # Clear potentially corrupted persisted data
        st.session_state.persisted_uploaded_file_bytes = None
        st.session_state.persisted_uploaded_file_name = None
        st.session_state.last_uploader_key_persisted = None
        df_raw = None

elif uploaded_file is not None:
    # A new file is uploaded, or the file_uploader widget retained its state correctly.
    try:
        # Persist this newly uploaded file's content and name.
        uploaded_file.seek(0)
        st.session_state.persisted_uploaded_file_bytes = uploaded_file.getvalue()
        st.session_state.persisted_uploaded_file_name = uploaded_file.name
        st.session_state.last_uploader_key_persisted = uploaded_file_key
        
        # Clear any previous anonymized data as new raw data is being processed/re-processed
        st.session_state.df_anonymized_data = None
        st.session_state.df_anonymized_source_technique = None

        # Now, read "df_raw" from the current "uploaded_file" object for immediate use.
        uploaded_file.seek(0) # Reset pointer again for pandas to read
        if uploaded_file.name.endswith('.csv'):
            df_raw = pd.read_csv(uploaded_file, sep=None, engine='python')
        elif uploaded_file.name.endswith('.xlsx'):
            df_raw = pd.read_excel(uploaded_file, engine='openpyxl')
        
        # This success message was inside the 'if uploaded_file is not None:' block originally.
        # It should only show if "df_raw" was successfully loaded from the uploader.
        if df_raw is not None:
             st.success(f"Uploaded '{uploaded_file.name}' successfully!")

    except Exception as e:
        st.error(f"Error processing uploaded file: {e}")
        st.exception(e)
        df_raw = None
        # Clear persisted data if processing this new file failed
        st.session_state.persisted_uploaded_file_bytes = None
        st.session_state.persisted_uploaded_file_name = None
        st.session_state.last_uploader_key_persisted = None

# --- Main data processing block [if "df_raw" is available] ---
if df_raw is not None:
    # Ensure df_raw is a DataFrame
    sa_col_to_pass = None
    try:
        # This subheader and dataframe display should be here, active if df_raw exists
        st.subheader("Original Data Preview (First 5 rows)")
        st.dataframe(df_raw.head())
        all_cols = df_raw.columns.tolist()
        st.session_state['df_raw_columns_for_config_apply'] = all_cols

        # --- General Configuration [SA Column] ---
        st.sidebar.header("General Configuration")
        sa_col_options = ["<None>"] + all_cols
        current_sa_val = st.session_state.get("sa_col_selector_main", "<None>")
        sa_index = sa_col_options.index(current_sa_val) if current_sa_val in sa_col_options else 0
        
        # Sensitive Attribute (SA) column selection
        sa_col_selected = st.sidebar.selectbox(
            "Select Sensitive Attribute (SA) column (optional):",
            options=sa_col_options, 
            index=sa_index, 
            key="sa_col_selector_main"
        )
        # Re-assigns "sa_col_to_pass" based on selection.
        sa_col_to_pass = None if sa_col_selected == "<None>" else sa_col_selected

        # Initialize "df_anonymized_current_run" for this script run
        df_anonymized_current_run = None

        # --- Technique-Specific UI and Logic ---
        plugin_key_prefix_base = current_plugin.get_name().lower().replace(' ', '_').replace('-', '_').replace('(', '').replace(')', '')
        # If the selected technique is a test plugin, use a different prefix to avoid conflicts
        if selected_technique_name.startswith("[TEST]") and st.session_state.test_plugin_display_name:
            plugin_key_prefix_base = st.session_state.test_plugin_display_name.lower().replace(' ', '_').replace('-', '_').replace('(', '').replace(')', '')
            plugin_key_prefix = f"test_{plugin_key_prefix_base}"
        # If the technique is not a test plugin, use the base prefix
        else:
            plugin_key_prefix = plugin_key_prefix_base
        
        # Get the sidebar UI elements for the current plugin
        current_parameters = current_plugin.get_sidebar_ui(all_cols, sa_col_to_pass, df_raw, plugin_key_prefix)
        
        # If the plugin has a sidebar UI, it will return parameters or None
        full_config_for_export = {
            "technique": selected_technique_name, 
            "sa_col": sa_col_to_pass,
            "parameters": current_plugin.build_config_export(plugin_key_prefix, sa_col_to_pass)
        }
        # Add the plugin key prefix to the config for export
        current_plugin.get_export_button_ui(full_config_for_export, plugin_key_prefix)

        # If the plugin has an anonymize button UI, show it and handle anonymization
        if current_plugin.get_anonymize_button_ui(plugin_key_prefix): 
            if df_raw is not None and not df_raw.empty:
                with st.spinner(f"Applying {selected_technique_name}..."):
                    try:
                        # Calculate and immediately store in session state
                        anonymized_result = current_plugin.anonymize(df_raw.copy(), current_parameters, sa_col_to_pass)
                        st.session_state.df_anonymized_data = anonymized_result
                        st.session_state.df_anonymized_source_technique = selected_technique_name
                        # Update the current run variable
                        df_anonymized_current_run = anonymized_result
                    # Handle any exceptions during anonymization
                    except Exception as e:
                        st.error(f"{selected_technique_name} Error: {e}")
                        st.exception(e)
                        # Clear on error to avoid stale data
                        st.session_state.df_anonymized_data = None
                        st.session_state.df_anonymized_source_technique = None
                        df_anonymized_current_run = None
            # If no file is uploaded or df_raw is empty, show a warning
            else:
                st.warning("Please upload a file before anonymizing.")
                # Clear stale anonymized data if raw data is gone or empty for this attempt
                st.session_state.df_anonymized_data = None
                st.session_state.df_anonymized_source_technique = None
                df_anonymized_current_run = None
        
        # Display logic uses "df_anonymized_current_run" which is set above
        if df_anonymized_current_run is not None and not df_anonymized_current_run.empty:
            st.success(f"{selected_technique_name} anonymization complete!")
            st.subheader(f"Anonymized Data Preview ({selected_technique_name})")
            st.dataframe(df_anonymized_current_run.head()) # Use df_anonymized_current_run
            
            # Change from CSV to XLSX download
            xlsx_anonymized = convert_df_to_xlsx(df_anonymized_current_run) # Use XLSX function
            
            current_file_name_for_download = st.session_state.persisted_uploaded_file_name
             # Prefer current uploader name if available
            if uploaded_file is not None:
                 current_file_name_for_download = uploaded_file.name
            # Fallback if persisted is also somehow gone
            elif not current_file_name_for_download: 
                 current_file_name_for_download = "data"

            # Download button for XLSX
            st.download_button(
                label=f"📥 Download Anonymized ({selected_technique_name}) Data as XLSX", 
                # Use the converted XLSX data
                data=xlsx_anonymized, 
                # Use the original file name with anonymized prefix
                file_name=f"anonymized_{plugin_key_prefix_base}_{current_file_name_for_download.split('.')[0]}.xlsx", 
                # Set MIME type for XLSX
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                # Use a unique key for this button 
                key=f"download_{plugin_key_prefix}",
            )
        
        elif df_anonymized_current_run is not None and df_anonymized_current_run.empty:
            anonymize_button_key = f"{plugin_key_prefix}_anonymize_button" 
            # Check if the button was actually clicked in this run or if we are just showing empty persisted data
            if st.session_state.get(anonymize_button_key, False) or \
               (st.session_state.get('df_anonymized_data') is not None and st.session_state.df_anonymized_data.empty and \
                st.session_state.get('df_anonymized_source_technique') == selected_technique_name):
                st.warning(f"Anonymization with {selected_technique_name} resulted in an empty DataFrame.")
    # This is the except for "if df_raw is not None:"
    except Exception as e: 
        st.error(f"Error during data display or plugin UI setup: {e}")
        st.exception(e)
# This 'else' corresponds to 'if df_raw is not None:'
else:
    st.info("Awaiting file upload to configure and anonymize.")
    if 'df_raw_columns_for_config_apply' in st.session_state:
        del st.session_state['df_raw_columns_for_config_apply']
    
    # Clear persisted anonymized data if raw data is gone
    st.session_state.df_anonymized_data = None
    st.session_state.df_anonymized_source_technique = None

# ========================= PERSISTENCE OF UPLOADED FILES ========================= #
# Ensure session state for uploaded file is initialized
    if st.session_state.get(uploaded_file_key) is None:
        st.session_state.persisted_uploaded_file_bytes = None
        st.session_state.persisted_uploaded_file_name = None
        st.session_state.last_uploader_key_persisted = None


# ========================= DEVELOPER TOOLS IN MAIN AREA (if enabled) ========================= #
if st.session_state.developer_mode_enabled:
    # === Developer Zone: Plugin Creation & Testing ===
    with st.expander("Developer Zone: Create & Test Plugins", expanded=True):
        st.subheader("Plugin Code Editor")

        # Ensure session state for plugin code editor is initialized
        current_code_for_editor = st.session_state.get("test_plugin_raw_code", "")
        code_returned_by_editor = st_ace(
            value=current_code_for_editor,
            language="python",
            theme="tomorrow_night",
            keybinding="vscode",
            font_size=14,
            height=600,
            show_gutter=True,
            wrap=True,
            auto_update=False,
            readonly=False,
            placeholder=""" # Paste or type your Python plugin code here...
                            # You can view example snippets via the sidebar button.
                            # Required:
                            # - from ...base_anonymizer import Anonymizer
                            # - class YourPluginName(Anonymizer): ...
                            # - def get_plugin(): return YourPluginName()
                            # Implement all methods from the Anonymizer base class.
                            # Use st.sidebar for UI elements within get_sidebar_ui.
                        """,
            key="plugin_editor_code_area_ace_main" 
        )
        if code_returned_by_editor != current_code_for_editor:
            st.session_state.test_plugin_raw_code = code_returned_by_editor
        
        st.caption("Write your plugin code above. View example snippets via the sidebar. Ensure your code defines a class inheriting from `Anonymizer` and includes a `get_plugin()` factory function.")

        col1_dev, col2_dev = st.columns(2)
        with col1_dev:
            st.session_state.test_plugin_class_name = st.text_input(
                "Plugin Class Name (e.g., MyNewPlugin):", 
                value=st.session_state.get("test_plugin_class_name", ""), 
                key="plugin_editor_class_name_input_main"
            )
        with col2_dev:
            st.session_state.test_plugin_display_name = st.text_input(
                "Plugin Display Name (for dropdown):", 
                value=st.session_state.get("test_plugin_display_name", ""), 
                key="plugin_editor_display_name_input_main"
            )

        # "Validate Code" button
        if st.button("Validate Plugin Code", key="plugin_editor_validate_button"):
            st.session_state.plugin_validation_results = validate_plugin_code(
                st.session_state.test_plugin_raw_code,
                st.session_state.test_plugin_class_name
            )
            st.session_state.test_plugin_error = None
            st.session_state.plugin_editor_save_status = ""

        # Display Validation Results
        # Use .get for safety
        if st.session_state.get("plugin_validation_results"):
            st.markdown("---")
            st.subheader("Plugin Code Validation Checklist:")
            # Track if all are either passed or warning
            all_passed_or_warning = True 
            has_failures = False

            for res in st.session_state.plugin_validation_results:
                icon = res['status'].split(" ")[0] # Get the icon part
                message = f" {res['check']}"
                if res['message']:
                    message += f" ({res['message']})"
                
                if "❌ Failed" in res['status']:
                    st.error(icon + message)
                    all_passed_or_warning = False
                    has_failures = True
                elif "⚠️ Warning" in res['status']:
                    st.warning(icon + message)
                else:
                    st.success(icon + message)
            # If no failures and at least one check ran
            if not has_failures and any(r['status'] != '❌ Failed' for r in st.session_state.plugin_validation_results):
                 st.toast("Validation successful! All checks passed.", icon="✅")

        if st.button("Test This Plugin In Session", key="plugin_editor_test_button"):
            st.session_state.plugin_validation_results = []
            st.session_state.test_plugin_instance = None
            st.session_state.test_plugin_error = None
            st.session_state.plugin_editor_save_status = ""
            
            code_str = st.session_state.test_plugin_raw_code
            class_name_str = st.session_state.test_plugin_class_name
            display_name_str = st.session_state.test_plugin_display_name

            if not all([code_str, class_name_str, display_name_str]):
                st.session_state.test_plugin_error = "Plugin Code, Class Name, and Display Name are required."
            else:
                # Attempt to load the test plugin using a temporary file to better handle imports
                temp_module_name = f"temp_test_plugin_{os.urandom(4).hex()}" # Unique temp module name
                temp_file_path = ""
                try:
                    # Create a temporary directory if it doesn't exist
                    temp_dir = os.path.join(PROJECT_ROOT, "src", "anonymizers", "plugins", "temp_plugins")
                    os.makedirs(temp_dir, exist_ok=True)

                    with tempfile.NamedTemporaryFile(
                        mode="w", 
                        suffix="_plugin.py", # Ensure it ends with _plugin.py if any loading logic depends on it
                        delete=False,        # Delete it manually after loading
                        dir=temp_dir,        # Place it within a known package structure
                        encoding="utf-8"
                    ) as tmp_file:
                        tmp_file.write(code_str)
                        temp_file_path = tmp_file.name
                    
                    # Construct a module name that importlib can understand relative to 'src'
                    # e.g., src.anonymizers.plugins.temp_plugins.temp_filename_without_py
                    relative_path_from_src = os.path.relpath(temp_file_path, PROJECT_ROOT)
                    module_spec_name_parts = os.path.splitext(relative_path_from_src)[0].split(os.sep)
                    # Ensure it's a valid Python module path (dots, no leading slashes)
                    module_spec_name = ".".join(part for part in module_spec_name_parts if part)


                    if module_spec_name in sys.modules: # Clean up if somehow loaded before
                        del sys.modules[module_spec_name]

                    spec = importlib.util.spec_from_file_location(module_spec_name, temp_file_path)
                    if not spec or not spec.loader:
                        raise ImportError(f"Could not create module spec for temporary plugin at {temp_file_path}")
                    
                    module = importlib.util.module_from_spec(spec)
                    sys.modules[module_spec_name] = module # Crucial for relative imports
                    spec.loader.exec_module(module)

                    if not hasattr(module, "get_plugin") or not callable(module.get_plugin):
                        raise AttributeError("Test plugin code is missing the 'get_plugin()' factory function.")
                    
                    temp_instance = module.get_plugin()

                    # Validate the instance (optional, but good practice)
                    if not isinstance(temp_instance, Anonymizer):
                        raise TypeError(f"The 'get_plugin()' function in the test code did not return an Anonymizer instance. Got: {type(temp_instance)}")
                    
                    # Check if the returned instance's class name matches the expected one (optional)
                    # This assumes get_plugin() returns an instance of the class_name_str
                    # This check might be too strict if get_plugin() instantiates a different class defined in the string
                    # if temp_instance.__class__.__name__ != class_name_str:
                    #    st.warning(f"Plugin's get_plugin() returned an instance of '{temp_instance.__class__.__name__}' instead of expected '{class_name_str}'.")


                    st.session_state.test_plugin_instance = temp_instance
                    st.success(
                        f"Plugin '{display_name_str}' (from code editor) loaded for testing! "
                        "It's now in the technique dropdown. "
                        "If your data disappeared, it should reload. Then, select the [TEST] plugin to use it."
                    )
                    load_anonymizer_plugins() 
                    st.rerun()

                except Exception as e:
                    st.session_state.test_plugin_error = f"Error loading test plugin: {e}\n{traceback.format_exc()}"
                finally:
                    if temp_file_path and os.path.exists(temp_file_path):
                        try:
                            os.remove(temp_file_path) # Delete the .py file
                        except Exception as e_del:
                            st.warning(f"Could not delete temporary test plugin file (.py) {temp_file_path}: {e_del}")
                    
                    # Attempt to delete the corresponding .pyc file if it exists alongside
                    # temp_file_path would be something like ".../temp_plugins/tmpXXXX_plugin.py"
                    if temp_file_path:
                        pyc_file_path_alongside = temp_file_path.replace(".py", f".cpython-{sys.version_info.major}{sys.version_info.minor}.pyc")
                        if os.path.exists(pyc_file_path_alongside):
                            try:
                                os.remove(pyc_file_path_alongside)
                            except Exception as e_del_pyc:
                                st.warning(f"Could not delete temporary .pyc file {pyc_file_path_alongside}: {e_del_pyc}")

                        # Attempt to delete from __pycache__ if it exists
                        # e.g. .../temp_plugins/__pycache__/tmpXXXX_plugin.cpython-XY.pyc
                        temp_dir_of_file = os.path.dirname(temp_file_path)
                        temp_file_basename = os.path.basename(temp_file_path)
                        pyc_filename_in_cache = temp_file_basename.replace(".py", f".cpython-{sys.version_info.major}{sys.version_info.minor}.pyc")
                        pyc_file_path_in_cache = os.path.join(temp_dir_of_file, "__pycache__", pyc_filename_in_cache)

                        if os.path.exists(pyc_file_path_in_cache):
                            try:
                                os.remove(pyc_file_path_in_cache)
                            except Exception as e_del_pyc_cache:
                                st.warning(f"Could not delete temporary .pyc file from __pycache__ {pyc_file_path_in_cache}: {e_del_pyc_cache}")
                        
                        # Optional: Clean up the __pycache__ directory itself if it's empty and belongs to temp_plugins
                        pycache_dir = os.path.join(temp_dir_of_file, "__pycache__")
                        if os.path.exists(pycache_dir) and not os.listdir(pycache_dir) and "temp_plugins" in pycache_dir:
                            try:
                                os.rmdir(pycache_dir)
                            except Exception as e_del_pycache_dir:
                                st.warning(f"Could not delete empty __pycache__ directory {pycache_dir}: {e_del_pycache_dir}")
                    
                    # The temp_dir (e.g. .../temp_plugins) itself is not deleted here,
                    # as it's a general workspace. Only the files related to the specific test run.
        
        if st.session_state.test_plugin_error: st.error(st.session_state.test_plugin_error)

        st.markdown("---")
        st.subheader("Save Tested Plugin")
        plugin_filename = st.text_input(
            "Filename (e.g., my_plugin.py):", 
            value=f"{st.session_state.test_plugin_display_name.lower().replace(' ', '_').replace('(', '').replace(')', '')}_plugin.py" if st.session_state.test_plugin_display_name else "new_plugin.py",
            key="plugin_editor_filename_input"
        )
        if st.button("Save Plugin to File System", key="plugin_editor_save_button"):
            st.session_state.plugin_validation_results = [] # Clear validation on save attempt
            st.session_state.plugin_editor_save_status = ""
            code_to_save_original, fname = st.session_state.test_plugin_raw_code, plugin_filename # Get original code
            
            if not code_to_save_original: st.session_state.plugin_editor_save_status = "Error: No code to save."
            elif not fname: st.session_state.plugin_editor_save_status = "Error: Filename required."
            elif not fname.endswith("_plugin.py"): st.session_state.plugin_editor_save_status = "Error: Filename must end with '_plugin.py'."
            elif ".." in fname or "/" in fname or "\\" in fname: st.session_state.plugin_editor_save_status = "Error: Invalid filename characters."
            else:
                try:
                    plugin_dir = os.path.join(PROJECT_ROOT, "src", "anonymizers", "plugins")
                    if not os.path.exists(plugin_dir): os.makedirs(plugin_dir)
                    file_path = os.path.join(plugin_dir, fname)
                    
                    if os.path.exists(file_path):
                        st.session_state.plugin_editor_save_status = f"Warning: File '{fname}' exists. Not overwritten."
                    else:
                        # Modify the import path before saving
                        # This specifically targets the case where the test plugin might use three dots
                        # and the saved file (in src/anonymizers/plugins/) needs two dots.
                        code_to_save_modified = code_to_save_original.replace(
                            "from ...base_anonymizer import Anonymizer", 
                            "from ..base_anonymizer import Anonymizer"
                        )
                        # You might want to add a more specific check or a warning if other three-dot imports exist
                        # that are not related to base_anonymizer, but for now, this addresses the direct request.

                        with open(file_path, "w", encoding="utf-8") as f:
                            f.write(code_to_save_modified) # Save the potentially modified code
                        
                        st.session_state.plugin_editor_save_status = f"Success: Plugin saved as '{fname}'. Reloading..."
                        # Clear test plugin state after successful save
                        st.session_state.test_plugin_instance = None
                        st.session_state.test_plugin_display_name = ""
                        st.session_state.test_plugin_raw_code = ""      # Clear editor
                        st.session_state.test_plugin_class_name = ""    # Clear editor
                        load_anonymizer_plugins() # Reload all plugins
                        st.rerun()
                except Exception as e:
                    st.session_state.plugin_editor_save_status = f"Error saving plugin: {e}\n{traceback.format_exc()}"
        
        if st.session_state.plugin_editor_save_status:
            if "Success" in st.session_state.plugin_editor_save_status: st.success(st.session_state.plugin_editor_save_status)
            elif "Warning" in st.session_state.plugin_editor_save_status: st.warning(st.session_state.plugin_editor_save_status)
            else: st.error(st.session_state.plugin_editor_save_status)
else: # If developer mode is disabled
    st.session_state.plugin_validation_results = [] # Clear validation results
    if st.session_state.test_plugin_instance is not None: # Clear active test plugin
        st.session_state.test_plugin_instance = None
        st.session_state.test_plugin_display_name = ""
        load_anonymizer_plugins() # Reload to remove it from list
        st.rerun()

# --- Ensure plugin directories exist on startup ---
plugin_base_dir = os.path.join(PROJECT_ROOT, "src", "anonymizers", "plugins")
os.makedirs(plugin_base_dir, exist_ok=True)

# Specific check for mondrian_plugin.py presence
mondrian_plugin_path = os.path.join(plugin_base_dir, "mondrian_plugin.py")
if not os.path.isfile(mondrian_plugin_path):
    st.warning(f"Expected plugin file not found: {mondrian_plugin_path}. Ensure the file exists.")

# Ensure temp_plugins directory exists (app should create this too)
temp_plugins_dir = os.path.join(plugin_base_dir, "temp_plugins")
os.makedirs(temp_plugins_dir, exist_ok=True)

# Inform about the temp_plugins directory
st.info(f"Temporary plugins directory: {temp_plugins_dir}. This is where test plugins are loaded from.")

# --- Example plugin snippets (read-only and copyable) ---
# This section is controlled by the show_code_snippets_expander state
if st.session_state.get("show_code_snippets_expander", False):
    with st.expander("Plugin Code Snippets (Read-Only & Copy)", expanded=True):
        st.markdown("Below are some example plugin snippets. You can manually select and copy the code to paste into the editor in the 'Developer Zone'.")
        
        # The EXAMPLE_PLUGIN_SNIPPETS dictionary should be defined earlier in your script
        # (e.g., near the top, after imports).
        if 'EXAMPLE_PLUGIN_SNIPPETS' in globals() and isinstance(EXAMPLE_PLUGIN_SNIPPETS, dict):
            for snippet_name, snippet_code in EXAMPLE_PLUGIN_SNIPPETS.items():
                if snippet_name == "--- Select a Snippet ---": # Skip placeholder
                    continue
                st.subheader(snippet_name)
                st.code(snippet_code, language="python")
                st.markdown("---") # Separator between snippets
        else:
            st.warning("EXAMPLE_PLUGIN_SNIPPETS dictionary not found or not a dictionary.")
        
        if st.button("Close Snippet Viewer", key="close_snippet_viewer_button"):
            st.session_state.show_code_snippets_expander = False
            st.rerun() # Rerun to immediately hide the expander

# ========================================================================================
# DATASET LOADING AND MANAGEMENT SYSTEM - CORE COMPONENT FOR ANONYMIZATION TESTING
# ========================================================================================
# This section implements the dataset loading infrastructure that provides both real-world
# and synthetic datasets for comprehensive privacy-preserving anonymization research.
# The system supports scalable dataset generation for performance testing and academic evaluation.

@st.cache_data
def load_sample_dataset(dataset_name, dataset_size="sample"):
    """
    Advanced Dataset Loading System for Privacy Research and Anonymization Testing
    
    This function serves as the central dataset provider for the anonymization platform,
    offering a comprehensive collection of both classic machine learning datasets and
    synthetic datasets specifically designed for privacy research. The system implements
    intelligent size scaling to support performance testing across different data volumes.
    
    Core Functionality:
    - Multi-source dataset integration (scikit-learn, OpenML, synthetic generation)
    - Intelligent size scaling for performance benchmarking (1x, 3x, 10x multipliers)
    - Reproducible data generation using fixed random seeds
    - Privacy-aware dataset curation with realistic sensitive information patterns
    
    Research Applications:
    - Anonymization algorithm performance evaluation across varying dataset sizes
    - Privacy technique comparison using standardized benchmark datasets

    
    Parameters:
    -----------
    dataset_name : str
        Identifier for the specific dataset to load. Supports classic ML datasets
        (iris, wine, breast_cancer, etc.) and privacy-focused synthetic datasets
    dataset_size : str, default="sample"
        Scale factor for dataset generation:
        - "sample": 1x base size for quick testing and development
        - "medium": 3x base size for moderate performance evaluation
        - "full": 10x base size for large-scale testing and real-world simulation
    
    Returns:
    --------
    pandas.DataFrame or None
        Loaded dataset with appropriate size scaling, or None if loading fails
        
    Technical Implementation Notes:
    - Utilizes Streamlit caching for optimal performance across user sessions
    - Implements defensive programming with comprehensive exception handling
    - Maintains reproducibility through consistent random seed management (seed=42)
    - Supports both classification and regression tasks across multiple domains
    """
    import pandas as pd
    import numpy as np
    
    try:
        # Set deterministic random seed for reproducible research results
        # Essential for academic validation and comparative studies
        np.random.seed(42)  # Standard seed value ensuring consistent dataset generation
        
        # ====================================================================
        # DATASET SIZE SCALING CONFIGURATION
        # ====================================================================
        # Define intelligent scaling multipliers for comprehensive performance testing
        # This system enables researchers to evaluate anonymization techniques across
        # different data volumes, simulating real-world deployment scenarios
        
        if dataset_size == "sample":
            size_multiplier = 1      # Base size: Quick prototyping and algorithm validation
        elif dataset_size == "medium":
            size_multiplier = 3      # Medium scale: Performance characterization testing
        elif dataset_size == "full":
            size_multiplier = 10     # Large scale: Enterprise deployment simulation
        else:
            size_multiplier = 1      # Default fallback: Conservative resource usage
        
        df = None  # Initialize dataset container for subsequent population

        # ====================================================================
        # SCIKIT-LEARN CLASSIC DATASETS - ACADEMIC BENCHMARK COLLECTION
        # ====================================================================
        # This section provides access to well-established machine learning datasets
        # from scikit-learn, serving as standardized benchmarks for privacy research.
        # Each dataset has been extensively used in academic literature, providing
        # reliable baselines for anonymization technique comparison and validation.
        
        if dataset_name == "iris":
            """
            Iris Flower Classification Dataset - Multi-class Privacy Challenge
            
            Academic Provenance: Fisher, R.A. (1936) - Classic statistical dataset
            Privacy Research Value: Small-scale multi-class classification with botanical features
            Dataset Characteristics: 150 samples, 4 features, 3 species classes
            Anonymization Challenges: Feature correlation patterns, small dataset size effects
            """
            data = datasets.load_iris(as_frame=True)
            df = data.frame
            # Apply intelligent replication for small datasets to enable scalability testing
            if size_multiplier > 1 and len(df) < 500:  # Replication threshold for small datasets
                df = pd.concat([df.copy() for _ in range(size_multiplier)], ignore_index=True)
                df = df.sample(frac=1, random_state=42).reset_index(drop=True)  # Randomize order post-replication
                
        elif dataset_name == "wine":
            """
            Wine Quality Classification Dataset - Chemical Analysis Privacy Case
            
            Academic Provenance: UCI Repository - Chemical constituent analysis
            Privacy Research Value: Multi-class classification with chemical measurements
            Dataset Characteristics: 178 samples, 13 chemical features, 3 wine classes
            Anonymization Challenges: High-dimensional feature space, chemical fingerprinting
            """
            data = datasets.load_wine(as_frame=True)
            df = data.frame
            if size_multiplier > 1 and len(df) < 600:
                df = pd.concat([df.copy() for _ in range(size_multiplier)], ignore_index=True)
                df = df.sample(frac=1, random_state=42).reset_index(drop=True)
                
        elif dataset_name == "breast_cancer":
            """
            Breast Cancer Diagnostic Dataset - Medical Privacy Research
            
            Academic Provenance: Wisconsin Diagnostic Breast Cancer (WDBC) Database
            Privacy Research Value: Medical diagnostic data with patient privacy implications
            Dataset Characteristics: 569 samples, 30 clinical features, binary diagnosis
            Anonymization Challenges: Medical data sensitivity, diagnostic accuracy preservation
            """
            data = datasets.load_breast_cancer(as_frame=True)
            df = data.frame
            if size_multiplier > 1 and len(df) < 2000:
                 df = pd.concat([df.copy() for _ in range(size_multiplier)], ignore_index=True)
                 df = df.sample(frac=1, random_state=42).reset_index(drop=True)
                 
        elif dataset_name == "diabetes":
            """
            Diabetes Progression Dataset - Medical Regression Privacy Challenge
            
            Academic Provenance: Efron et al. (2004) - Medical progression modeling
            Privacy Research Value: Medical regression with patient health measurements
            Dataset Characteristics: 442 samples, 10 physiological features, progression target
            Anonymization Challenges: Health data privacy, regression accuracy maintenance
            """
            data = datasets.load_diabetes(as_frame=True)
            df = data.frame
            if size_multiplier > 1 and len(df) < 1500:
                df = pd.concat([df.copy() for _ in range(size_multiplier)], ignore_index=True)
                df = df.sample(frac=1, random_state=42).reset_index(drop=True)
                
        elif dataset_name == "digits":
            """
            Handwritten Digits Recognition Dataset - Image-based Privacy Research
            
            Academic Provenance: Optical recognition of handwritten digits (UCI)
            Privacy Research Value: High-dimensional image data with classification challenges
            Dataset Characteristics: 1,797 samples, 64 pixel features, 10 digit classes
            Anonymization Challenges: High dimensionality, handwriting pattern privacy
            """
            data = datasets.load_digits(as_frame=True)
            df = data.frame
            if size_multiplier > 1 and len(df) < 6000:  # Digits dataset is moderately larger
                df = pd.concat([df.copy() for _ in range(size_multiplier)], ignore_index=True)
                df = df.sample(frac=1, random_state=42).reset_index(drop=True)
                
        elif dataset_name == "california_housing":
            """
            California Housing Prices Dataset - Large-Scale Geographic Privacy Challenge
            
            Academic Provenance: California housing census data (1990)
            Privacy Research Value: Large-scale geographic and economic data
            Dataset Characteristics: 20,640 samples, 8 features, housing price regression
            Anonymization Challenges: Geographic privacy, economic inference, large-scale processing
            
            Technical Note: Size multiplier not applied due to already large dataset size
            """
            data = datasets.fetch_california_housing(as_frame=True)
            df = data.frame  # Large dataset - size multiplication bypassed for performance
            # For enterprise-scale datasets, size multiplier could enable sampling strategies,
            # but here we preserve the complete dataset for comprehensive testing

        # ====================================================================
        # SYNTHETIC DATASETS - PRIVACY-FOCUSED RESEARCH COLLECTION
        # ====================================================================
        # This section generates synthetic datasets specifically designed for privacy research,
        # incorporating realistic patterns of sensitive information while providing controlled
        # environments for anonymization technique development and validation.
        
        elif dataset_name == "boston_synthetic":
            """
            Synthetic Boston Housing Dataset - Real Estate Privacy Simulation
            
            Research Purpose: Simulates housing market data with privacy-sensitive attributes
            Original Inspiration: Boston Housing dataset (ethical considerations addressed via synthesis)
            Privacy Research Value: Regression with real estate and demographic privacy challenges
            
            Dataset Characteristics:
            - Original Boston dataset size: 506 samples (historically scaled)
            - 13 numerical features representing housing and neighborhood characteristics
            - Includes PII-like attributes for comprehensive anonymization testing
            - Synthetic owner initials and street number hashes simulate real-world identifiers
            
            Anonymization Test Cases:
            - Personal identifier redaction (owner initials)
            - Location-based quasi-identifier anonymization (street numbers)
            - Regression accuracy preservation under privacy constraints
            """
            n_samples = 506 * size_multiplier  # Scale from original Boston dataset size
            n_features = 13  # Match original Boston Housing feature count
            
            # Generate realistic housing data using controlled regression synthesis
            X, y = make_regression(n_samples=n_samples, n_features=n_features, n_informative=10,
                                   noise=20.0, random_state=42)
            df = pd.DataFrame(X, columns=[f'feature_{i+1}' for i in range(n_features)])
            df['target_price'] = y
            
            # Incorporate realistic PII-like attributes for anonymization research
            df['owner_name_initials'] = [f"{chr(65+np.random.randint(0,26))}{chr(65+np.random.randint(0,26))}" for _ in range(n_samples)]
            df['street_number_hash'] = [hash(f"Street {i}") % 10000 for i in range(n_samples)]

        elif dataset_name == "credit_approval_synthetic":
            """
            Synthetic Credit Approval Dataset - Financial Privacy Challenge
            
            Research Purpose: Simulates financial decision-making with credit assessment privacy
            Original Inspiration: UCI Credit Approval dataset patterns
            Privacy Research Value: Binary classification with financial privacy implications
            
            Dataset Characteristics:
            - 690 base samples (scaled from original credit approval dataset patterns)
            - 15 applicant attributes representing financial and personal information
            - Binary approval outcome for supervised learning validation
            - Realistic applicant IDs and geographic indicators (zip code prefixes)
            
            Anonymization Test Cases:
            - Financial information anonymization (attribute generalization)
            - Applicant identifier management and pseudonymization
            - Geographic privacy (zip code truncation and generalization)
            - Decision outcome preservation under privacy transformation
            """
            n_samples = 690 * size_multiplier  # Based on original UCI credit approval dataset
            X, y = make_classification(n_samples=n_samples, n_features=15, n_informative=8,
                                       n_redundant=2, n_clusters_per_class=2, random_state=42)
            df = pd.DataFrame(X, columns=[f'attr_{i+1}' for i in range(15)])
            df['approved'] = y
            df['applicant_id'] = [f"APP-{1000+i}" for i in range(n_samples)]
            df['zip_code_prefix'] = np.random.randint(100, 999, n_samples)

        elif dataset_name == "marketing_campaign_synthetic":
            """
            Synthetic Marketing Campaign Dataset - Customer Privacy Challenge
            
            Research Purpose: Simulates marketing analytics with customer privacy considerations
            Privacy Research Value: Customer behavior analysis with contact information privacy
            
            Dataset Characteristics:
            - 2,240 base samples (typical marketing campaign dataset scale)
            - 10 behavioral and demographic factors influencing campaign response
            - Binary subscription outcome for campaign effectiveness analysis
            - Realistic email domains and temporal contact patterns
            
            Anonymization Test Cases:
            - Contact information anonymization (email domain generalization)
            - Temporal pattern disruption (contact timing anonymization)
            - Behavioral profiling prevention while preserving campaign insights
            """
            n_samples = 2240 * size_multiplier  # Inspired by typical marketing dataset scales
            X, y = make_classification(n_samples=n_samples, n_features=10, n_informative=5, random_state=42)
            df = pd.DataFrame(X, columns=[f'factor_{i+1}' for i in range(10)])
            df['subscribed'] = y
            df['customer_email_domain'] = np.random.choice(['@gmail.com', '@yahoo.com', '@hotmail.com', '@company.com'], n_samples)
            df['last_contact_days_ago'] = np.random.randint(1, 365, n_samples)

        elif dataset_name == "flower_species_synthetic":
            """
            Synthetic Flower Species Dataset - Biological Classification Privacy Case
            
            Research Purpose: Multi-class classification with biological observation privacy
            Privacy Research Value: Scientific data collection with observer identity protection
            
            Dataset Characteristics:
            - 300 base samples for manageable multi-class evaluation
            - 4 morphological features mimicking botanical measurement patterns
            - 3-class species classification for multi-class anonymization testing
            - Observation IDs simulating research data collection scenarios
            
            Anonymization Test Cases:
            - Research identifier anonymization (observation ID management)
            - Scientific data accuracy preservation under privacy constraints
            - Multi-class classification performance under anonymization stress testing
            """
            n_samples = 300 * size_multiplier
            X, y = make_classification(n_samples=n_samples, n_features=4, n_informative=3, n_classes=3, 
                                       n_clusters_per_class=1, random_state=42)
            df = pd.DataFrame(X, columns=['sepal_length_syn', 'sepal_width_syn', 'petal_length_syn', 'petal_width_syn'])
            df['species_type'] = y
            df['observation_id'] = [f"OBS-{2000+i}" for i in range(n_samples)]

        elif dataset_name == "music_genre_synthetic":
            """
            Synthetic Music Genre Dataset - Audio Analytics Privacy Challenge
            
            Research Purpose: Audio feature analysis with user behavior privacy protection
            Privacy Research Value: High-dimensional feature space with track identification risks
            
            Dataset Characteristics:
            - 1,000 base samples representing diverse music library scale
            - 20 audio features capturing spectral and temporal characteristics
            - 5-class genre classification for multi-class privacy evaluation
            - Track identifiers simulating music streaming platform scenarios
            
            Anonymization Test Cases:
            - High-dimensional feature space anonymization techniques
            - Track identifier management and pseudonymization strategies
            - User listening pattern privacy while preserving genre classification accuracy
            """
            n_samples = 1000 * size_multiplier
            X, y = make_classification(n_samples=n_samples, n_features=20, n_informative=10, n_classes=5, random_state=42)
            df = pd.DataFrame(X, columns=[f'audio_feature_{i+1}' for i in range(20)])
            df['genre'] = y
            df['track_id_suffix'] = [f"-TRK{np.random.randint(1000,9999)}" for _ in range(n_samples)]

        elif dataset_name == "stock_prices_synthetic":
            """
            Synthetic Stock Price Dataset - Financial Market Privacy Challenge
            
            Research Purpose: Financial time series analysis with trading privacy protection
            Privacy Research Value: Market data with trader identity and strategy privacy concerns
            
            Dataset Characteristics:
            - 500 base samples representing trading session observations
            - 5 market factors influencing price movements (sentiment, volume, rates, news, timing)
            - Regression target for price change prediction
            - Generic company tickers and temporal indicators for realistic market simulation
            
            Anonymization Test Cases:
            - Financial time series anonymization while preserving market trend accuracy
            - Company identifier generalization and ticker anonymization
            - Trading pattern privacy protection with temporal anonymization techniques
            """
            n_samples = 500 * size_multiplier
            # Generate realistic market-influencing features using controlled regression
            X_reg, y_reg = make_regression(n_samples=n_samples, n_features=5, n_informative=3, noise=10, random_state=42)
            df = pd.DataFrame(X_reg, columns=['market_sentiment', 'prev_volume', 'interest_rate', 'news_score', 'day_of_week'])
            df['price_change'] = y_reg
            df['company_ticker_generic'] = np.random.choice(['AAA', 'BBB', 'CCC', 'DDD'], n_samples)
            df['trade_date_offset'] = range(n_samples)


        # ========================================================================
        # OPENML REPOSITORY DATASETS - REAL-WORLD DATA FOR PRIVACY RESEARCH
        # ========================================================================
        # These datasets are sourced from OpenML, a collaborative machine learning platform
        # providing access to curated, real-world datasets with documented provenance.
        # Each dataset represents different domains and privacy challenges for comprehensive testing.
        
        elif dataset_name == "titanic":
            """
            RMS Titanic Passenger Dataset - Historical Survival Analysis
            
            Source: OpenML Repository (titanic dataset, version 1)
            Description: Complete passenger manifest from the RMS Titanic disaster (1912)
            Privacy Challenges: Contains personal identifiers, demographic data, and family relationships
            Research Value: Classic dataset for binary classification with inherent privacy concerns
            
            Dataset Characteristics:
            - Historical passenger records with names, ages, cabin assignments
            - Socioeconomic indicators (passenger class, fare paid)
            - Family structure data (siblings, parents/children aboard)
            - Binary survival outcome for supervised learning tasks
            """
            try:
                # Fetch official Titanic dataset from OpenML repository
                data = fetch_openml(name='titanic', version=1, as_frame=True, parser='auto')
                df = data.frame
                
                # Apply size scaling for performance testing scenarios
                # Titanic dataset is relatively small (~1,300 records), suitable for replication
                if size_multiplier > 1 and len(df) < 3000:
                     # Create multiple copies to simulate larger datasets for scalability testing
                     df = pd.concat([df.copy() for _ in range(size_multiplier)], ignore_index=True)
                     # Randomize order while maintaining reproducibility for consistent testing
                     df = df.sample(frac=1, random_state=42).reset_index(drop=True)
            except Exception as e:
                st.error(f"❌ Failed to fetch Titanic dataset from OpenML repository: {e}")
                return None
                
        elif dataset_name == "heart_disease_cleveland":
            """
            Cleveland Heart Disease Dataset - Medical Diagnosis Privacy Challenge
            
            Source: UCI Repository via OpenML (heart-disease dataset)
            Description: Clinical data for heart disease diagnosis from Cleveland Clinic Foundation
            Privacy Challenges: Protected Health Information (PHI) under HIPAA regulations
            Research Value: Medical data anonymization with diagnostic outcome preservation
            
            Dataset Characteristics:
            - Clinical measurements (blood pressure, cholesterol levels, ECG results)
            - Patient demographic information (age, sex, chest pain type)
            - Diagnostic test results requiring privacy protection
            - Multi-class heart disease severity classification
            
            Technical Implementation:
            - Primary fetch by dataset name with fallback to data ID for robustness
            - Handles OpenML repository changes and version inconsistencies
            - Implements defensive programming for reliable data access
            """
            try:
                # Primary attempt: Fetch by well-known dataset name
                # Note: OpenML data IDs can change; using name provides better stability
                data = fetch_openml(name='heart-disease', version=1, as_frame=True, parser='auto')
                df = data.frame
                
                # Scale dataset size for comprehensive privacy testing
                if size_multiplier > 1 and len(df) < 1000:
                     df = pd.concat([df.copy() for _ in range(size_multiplier)], ignore_index=True)
                     df = df.sample(frac=1, random_state=42).reset_index(drop=True)
                     
            except Exception as e_name:
                # Fallback mechanism: Use alternative dataset ID if name-based fetch fails
                try:
                    # Alternative: Statlog Heart dataset (data_id=53) - another common medical dataset
                    data = fetch_openml(data_id=53, as_frame=True, parser='auto')
                    df = data.frame
                    if size_multiplier > 1 and len(df) < 1000:
                        df = pd.concat([df.copy() for _ in range(size_multiplier)], ignore_index=True)
                        df = df.sample(frac=1, random_state=42).reset_index(drop=True)
                except Exception as e_id:
                    st.error(f"❌ Failed to fetch Heart Disease dataset from OpenML - Primary error: '{e_name}', Fallback error: '{e_id}'")
                    return None
                    
        elif dataset_name == "auto_mpg":
            """
            Automotive Fuel Efficiency Dataset - Regression with Identifying Information
            
            Source: UCI Repository via OpenML (autoMpg dataset)
            Description: Automotive fuel consumption data from 1970s-1980s vehicle models
            Privacy Challenges: Vehicle identification through model names and specifications
            Research Value: Regression tasks with quasi-identifiers requiring anonymization
            
            Dataset Characteristics:
            - Continuous target variable (miles per gallon fuel efficiency)
            - Technical specifications (cylinders, displacement, horsepower, weight)
            - Temporal information (model year) serving as quasi-identifier
            - Vehicle names that could enable re-identification
            """
            try:
                # Fetch automotive dataset for regression-based privacy research
                data = fetch_openml(name='autoMpg', version=1, as_frame=True, parser='auto')
                df = data.frame
                
                # Apply controlled replication for medium-scale testing
                if size_multiplier > 1 and len(df) < 1200:
                     df = pd.concat([df.copy() for _ in range(size_multiplier)], ignore_index=True)
                     df = df.sample(frac=1, random_state=42).reset_index(drop=True)
            except Exception as e:
                st.error(f"❌ Failed to fetch Auto MPG dataset from OpenML repository: {e}")
                return None
                
        elif dataset_name == "adult_income":
            """
            Adult Census Income Dataset - Large-Scale Demographic Privacy Challenge
            
            Source: UCI Repository via OpenML (adult dataset, version 2)
            Description: 1994 US Census Bureau data for income prediction analysis
            Privacy Challenges: Comprehensive demographic profiling with multiple quasi-identifiers
            Research Value: Large-scale dataset (~48K records) for scalability testing
            
            Dataset Characteristics:
            - Extensive demographic attributes (age, education, marital status, occupation)
            - Geographic indicators (workclass, native country)
            - Sensitive attributes (race, sex, relationship status)
            - Binary income classification (>50K vs <=50K annual income)
            
            Technical Considerations:
            - Large dataset size prohibits size multiplication (performance optimization)
            - Represents real-world scale for enterprise anonymization scenarios
            - Contains multiple types of sensitive and quasi-identifying information
            """
            try:
                # Fetch large-scale census dataset - version 2 provides cleaned, standardized format
                data = fetch_openml(name='adult', version=2, as_frame=True, parser='auto')
                df = data.frame  # Note: Large dataset - size multiplication not applied for performance
            except Exception as e:
                st.error(f"❌ Failed to fetch Adult Income dataset from OpenML repository: {e}")
                return None

        # ========================================================================
        # SYNTHETIC DATASET GENERATION - PRIVACY-PRESERVING RESEARCH DATASETS
        # ========================================================================
        
        elif dataset_name == "customer_data":
            """
            Customer Database - Comprehensive PII and Financial Information Challenge
            
            Research Purpose: Simulates customer relationship management (CRM) database privacy challenges
            Privacy Domain: Personal Identifiable Information (PII) and financial data protection
            Industry Context: E-commerce, banking, retail analytics with GDPR/CCPA compliance requirements
            
            Privacy Challenges Simulated:
            - Direct identifiers: Names, email addresses, phone numbers
            - Quasi-identifiers: Age, income correlation patterns, geographic information
            - Financial sensitive data: Income levels, credit scores requiring special protection
            - Contact information: Phone patterns, email domains enabling re-identification
            
            Anonymization Test Cases:
            - Name redaction and pseudonymization techniques
            - Income generalization and noise addition methods
            - Geographic data suppression (zip code anonymization)
            - Contact information masking and tokenization
            """
            # Calculate sample size based on selected multiplier for scalability testing
            n_samples = 1000 * size_multiplier
            
            # Generate realistic age distribution using normal distribution with realistic bounds
            # Mean age: 40, Standard deviation: 15, Clipped to realistic human age range
            age = np.random.normal(40, 15, n_samples).clip(18, 80)
            
            # Create income correlation with age using exponential base + age factor
            # Simulates realistic income progression with age while adding variability
            income = np.random.exponential(50000, n_samples) + age * 1000
            
            # Generate credit scores with normal distribution around typical average (650)
            # Standard deviation: 100, Clipped to standard credit score range (300-850)
            credit_score = np.random.normal(650, 100, n_samples).clip(300, 850)
            
            # Create realistic name pools with sufficient diversity for anonymization testing
            # Scaled proportionally to dataset size for consistent name diversity ratios
            first_names = ['John', 'Jane', 'Michael', 'Sarah', 'David', 'Emily', 'Robert', 'Lisa'] * (size_multiplier * 125 // 8 +1)
            last_names = ['Smith', 'Johnson', 'Williams', 'Brown', 'Jones', 'Garcia', 'Miller', 'Davis'] * (size_multiplier * 125 // 8+1)
            
            # Construct comprehensive customer dataset with multiple privacy challenge types
            df = pd.DataFrame({
                'customer_id': range(1, n_samples + 1),                                                                             # Sequential identifier for tracking
                'first_name': np.random.choice(first_names, n_samples, replace=True),                                               # Direct PII
                'last_name': np.random.choice(last_names, n_samples, replace=True),                                                 # Direct PII
                'age': age.astype(int),                                                                                             # Quasi-identifier
                'annual_income': income.round(2),                                                                                   # Sensitive financial data
                'credit_score': credit_score.astype(int),                                                                           # Sensitive financial data
                'phone_number': [f"555-{np.random.randint(100, 999)}-{np.random.randint(1000, 9999)}" for _ in range(n_samples)],   # PII contact
                'email': [f"user{i}@example.com" for i in range(n_samples)],                                                        # PII contact
                'city': np.random.choice(['New York', 'Los Angeles', 'Chicago', 'Houston', 'Phoenix'], n_samples),                  # Quasi-identifier
                'zip_code': np.random.randint(10000, 99999, n_samples)                                                              # Quasi-identifier location
            })
            
        elif dataset_name == "employee_records":
            """
            Employee HR Records - Workplace Privacy and Compliance Challenge
            
            Research Purpose: Simulates Human Resources database with employment privacy concerns
            Privacy Domain: Workplace privacy, salary confidentiality, personal employment data
            Industry Context: HR management systems requiring GDPR, employment law compliance
            Regulatory Framework: Employee privacy rights, salary disclosure restrictions
            
            Privacy Challenges Simulated:
            - Personal identifiers: Employee names, SSNs, birth dates
            - Sensitive employment data: Salary information, performance evaluations
            - Temporal quasi-identifiers: Hire dates, birth dates enabling inference attacks
            - Organizational structure: Department/position combinations reducing anonymity
            
            Anonymization Test Cases:
            - SSN redaction and partial masking techniques
            - Salary range generalization and noise addition
            - Date generalization (year-only, quarter-based anonymization)
            - Organizational hierarchy anonymization methods
            """
            # Scale dataset for different testing scenarios
            n_samples = 800 * size_multiplier
            
            # Define realistic organizational structure for privacy testing
            departments = ['Engineering', 'Marketing', 'Sales', 'HR', 'Finance']
            positions = ['Manager', 'Senior', 'Junior', 'Lead', 'Analyst']
            
            # Generate comprehensive employee record dataset
            df = pd.DataFrame({
                'employee_id': [f"EMP{str(i).zfill(6)}" for i in range(1, n_samples + 1)],                                                          # Organizational identifier
                'full_name': [f"Employee {i}" for i in range(n_samples)],                                                                           # Direct PII
                'department': np.random.choice(departments, n_samples),                                                                             # Quasi-identifier
                'position': np.random.choice(positions, n_samples),                                                                                 # Quasi-identifier
                'salary': np.random.normal(75000, 25000, n_samples).clip(30000, 200000).round(2),                                                   # Highly sensitive financial
                'years_experience': np.random.exponential(5, n_samples).clip(0, 40).astype(int),                                                    # Quasi-identifier
                'performance_rating': np.random.choice([1, 2, 3, 4, 5], n_samples, p=[0.05, 0.15, 0.4, 0.3, 0.1]),                                  # Sensitive evaluation
                'ssn': [f"{np.random.randint(100, 999)}-{np.random.randint(10, 99)}-{np.random.randint(1000, 9999)}" for _ in range(n_samples)],    # Highly sensitive PII
                'birth_date': pd.to_datetime('2000-01-01') - pd.to_timedelta(np.random.randint(20*365, 60*365, n_samples), unit='D'),               # Sensitive temporal
                'hire_date': pd.to_datetime('2023-01-01') - pd.to_timedelta(np.random.randint(1*365, 10*365, n_samples), unit='D')                  # Quasi-identifier temporal
            })
            # Convert datetime objects to string format for consistent handling
            df['birth_date'] = df['birth_date'].dt.strftime('%Y-%m-%d')
            df['hire_date'] = df['hire_date'].dt.strftime('%Y-%m-%d')

        elif dataset_name == "medical_records":
            """
            Medical Patient Records - Healthcare Privacy and HIPAA Compliance Challenge
            
            Research Purpose: Simulates electronic health records (EHR) privacy protection requirements
            Privacy Domain: Protected Health Information (PHI) under HIPAA regulations
            Industry Context: Healthcare systems, medical research, clinical data sharing
            Regulatory Framework: HIPAA Safe Harbor method, expert determination requirements
            
            Privacy Challenges Simulated:
            - Direct medical identifiers: Patient names, medical record numbers, insurance IDs
            - Health information: Diagnoses, treatment costs, clinical measurements
            - Provider information: Doctor names, facility identifiers
            - Temporal health data: Visit dates, treatment timelines
            - Biometric data: Blood pressure readings, vital signs
            
            Anonymization Test Cases:
            - Medical identifier redaction and code substitution
            - Diagnosis generalization (ICD-10 hierarchy anonymization)
            - Cost data noise addition and range generalization
            - Date shifting and temporal anonymization methods
            - Clinical measurement anonymization techniques
            """
            # Generate medical dataset with HIPAA-relevant privacy challenges
            n_samples = 1200 * size_multiplier
            
            # Define realistic medical conditions with distribution reflecting real-world prevalence
            conditions = ['Diabetes', 'Hypertension', 'Asthma', 'Heart Disease', 'None']
            
            df = pd.DataFrame({
                'patient_id': [f"P{str(i).zfill(8)}" for i in range(1, n_samples + 1)],                                         # Medical record identifier
                'patient_name': [f"Patient {i}" for i in range(n_samples)],                                                     # Direct PII (HIPAA identifier)
                'age': np.random.gamma(2, 30, n_samples).clip(1, 100).astype(int),                                              # Quasi-identifier with realistic age distribution
                'gender': np.random.choice(['Male', 'Female', 'Other'], n_samples, p=[0.48, 0.50, 0.02]),                       # Demographic quasi-identifier
                'diagnosis': np.random.choice(conditions, n_samples, p=[0.15, 0.20, 0.15, 0.10, 0.40]),                         # Protected health information
                'treatment_cost': np.random.exponential(5000, n_samples).round(2),                                              # Sensitive financial health data
                'insurance_id': [f"INS{np.random.randint(100000, 999999)}" for _ in range(n_samples)],                          # HIPAA identifier
                'doctor_name': np.random.choice(['Dr. Smith', 'Dr. Johnson', 'Dr. Williams', 'Dr. Brown'], n_samples),          # Provider identifier
                'visit_date': pd.to_datetime('2023-01-01') - pd.to_timedelta(np.random.randint(1, 3*365, n_samples), unit='D'), # Temporal health data
                'blood_pressure': [f"{np.random.randint(90, 180)}/{np.random.randint(60, 120)}" for _ in range(n_samples)]      # Clinical measurement data
            })
            # Standardize date format for consistent temporal anonymization testing
            df['visit_date'] = df['visit_date'].dt.strftime('%Y-%m-%d')
            
        elif dataset_name == "financial_transactions":
            """
            Financial Transaction Records - Banking Privacy and PCI-DSS Compliance Challenge
            
            Research Purpose: Simulates banking transaction database with financial privacy requirements
            Privacy Domain: Financial transaction data under PCI-DSS and banking regulations
            Industry Context: Banking systems, payment processing, financial analytics
            Regulatory Framework: PCI-DSS compliance, banking secrecy laws, financial privacy regulations
            
            Privacy Challenges Simulated:
            - Financial identifiers: Account numbers, card information, transaction IDs
            - Customer identification: Names linked to financial behavior patterns
            - Transaction patterns: Amounts, timestamps, merchant data enabling profiling
            - Location tracking: Geographic transaction patterns, spending locations
            - Balance information: Financial status inference from account balances
            
            Anonymization Test Cases:
            - Account number tokenization and masking
            - Transaction amount noise addition and bucketing
            - Timestamp anonymization and temporal pattern disruption
            - Merchant information generalization and suppression
            - Location data anonymization (zip code masking)
            """
            # Generate comprehensive financial transaction dataset
            n_samples = 2000 * size_multiplier
            
            # Define realistic transaction types found in banking systems
            transaction_types = ['Purchase', 'Transfer', 'Deposit', 'Withdrawal', 'Payment']
            
            df = pd.DataFrame({
                'transaction_id': [f"TXN{str(i).zfill(10)}" for i in range(1, n_samples + 1)],                                                     # Unique transaction identifier
                'account_number': [f"{np.random.randint(1000000000, 9999999999)}" for _ in range(n_samples)],                                      # Sensitive financial identifier
                'customer_name': [f"Customer {i}" for i in range(n_samples)],                                                                      # Direct PII linked to finances
                'transaction_type': np.random.choice(transaction_types, n_samples),                                                                # Transaction classification
                'amount': np.random.exponential(500, n_samples).round(2),                                                                          # Sensitive financial amount
                'balance_after': np.random.exponential(10000, n_samples).round(2),                                                                 # Sensitive account balance
                'merchant': np.random.choice(['Amazon', 'Walmart', 'Target', 'Starbucks', 'Other'], n_samples),                                    # Spending pattern data
                'location_zip_prefix': np.random.randint(100, 999, n_samples).astype(str),                                                         # Geographic quasi-identifier
                'timestamp': pd.to_datetime('2023-01-01 00:00:00') + pd.to_timedelta(np.random.randint(0, 365*24*60*60, n_samples), unit='s'),     # Temporal transaction pattern
                'card_last_four': [f"{np.random.randint(1000, 9999)}" for _ in range(n_samples)]                                                   # Partial payment card identifier
            })
            # Standardize timestamp format for temporal anonymization consistency
            df['timestamp'] = df['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')

        elif dataset_name == "survey_responses":
            """
            Survey Response Data - Research Privacy and Participant Protection Challenge
            
            Research Purpose: Simulates survey research database with participant privacy concerns
            Privacy Domain: Research participant data, survey respondent anonymity
            Industry Context: Market research, academic studies, opinion polling
            Regulatory Framework: Research ethics, informed consent, participant confidentiality
            
            Privacy Challenges Simulated:
            - Participant identification: Email hashes, IP address fragments
            - Demographic profiling: Age groups, income brackets, education combinations
            - Response patterns: Satisfaction scores, feedback text analysis
            - Temporal tracking: Submission timestamps enabling behavior analysis
            - Network information: IP address data for geographic inference
            
            Anonymization Test Cases:
            - Email hash anonymization and participant de-linkage
            - Demographic generalization and category broadening
            - Text anonymization in feedback fields
            - Timestamp anonymization and submission pattern disruption
            - Network identifier anonymization methods
            """
            # Generate research survey dataset with participant privacy challenges
            n_samples = 1500 * size_multiplier
            
            df = pd.DataFrame({
                'response_id': range(1, n_samples + 1),                                                                                                                       # Survey response identifier
                'respondent_email_hash': [hash(f"resp{i}@survey.com") % (10**8) for i in range(n_samples)],                                                                   # Participant identifier hash
                'age_group': np.random.choice(['18-25', '26-35', '36-45', '46-55', '56-65', '65+'], n_samples),                                                               # Demographic quasi-identifier
                'income_bracket': np.random.choice(['<30K', '30-50K', '50-75K', '75-100K', '>150K'], n_samples),                                                              # Economic quasi-identifier
                'education_level': np.random.choice(['High School', 'Bachelor', 'Master', 'PhD'], n_samples),                                                                 # Educational quasi-identifier
                'satisfaction_score': np.random.choice([1, 2, 3, 4, 5], n_samples),                                                                                           # Survey response data
                'feedback_keywords': [",".join(np.random.choice(['good', 'bad', 'neutral', 'improve', 'great', 'poor'], np.random.randint(1,4))) for _ in range(n_samples)],  # Text response data
                'submission_timestamp': pd.to_datetime('2023-06-01 00:00:00') + pd.to_timedelta(np.random.randint(0, 180*24*60*60, n_samples), unit='s'),                     # Temporal participation pattern
                'ip_octet_1_2': [f"{np.random.randint(1, 255)}.{np.random.randint(1, 255)}" for _ in range(n_samples)]                                                        # Partial network identifier
            })
            # Standardize timestamp format for temporal anonymization testing
            df['submission_timestamp'] = df['submission_timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')
            
        elif dataset_name == "clinical_trials":
            """
            Clinical Trial Participant Data - Medical Research Privacy Challenge
            
            Research Purpose: Simulates clinical trial database with research participant protection
            Privacy Domain: Clinical research data, trial participant anonymity, medical research ethics
            Industry Context: Pharmaceutical research, medical device trials, clinical studies
            Regulatory Framework: FDA regulations, GCP compliance, research ethics requirements
            
            Privacy Challenges Simulated:
            - Participant identifiers: Partial SSNs, initials, contact relationships
            - Medical information: Blood types, pre-existing conditions, drug responses
            - Trial data: Drug assignments, dosage information, physician identifiers
            - Temporal data: Birth years, trial timelines
            - Emergency contacts: Family relationship information
            
            Anonymization Test Cases:
            - Partial SSN masking and participant de-identification
            - Medical information generalization and code substitution
            - Trial arm anonymization while preserving analytical value
            - Age generalization from birth year data
            - Relationship information anonymization methods
            """
            # Generate clinical trial dataset with medical research privacy challenges
            n_samples = 950 * size_multiplier
            
            df = pd.DataFrame({
                'trial_id': [f"CT{str(i).zfill(7)}" for i in range(1, n_samples + 1)],                                                      # Clinical trial identifier
                'participant_ssn_last4': [f"{np.random.randint(1000, 9999)}" for _ in range(n_samples)],                                    # Partial sensitive identifier
                'participant_initials': [f"{chr(65+np.random.randint(0,25))}{chr(65+np.random.randint(0,25))}" for _ in range(n_samples)],  # Partial name identifier
                'birth_year': np.random.randint(1950, 1996, n_samples),                                                                     # Age-related quasi-identifier
                'trial_drug_group': np.random.choice(['DrugA', 'DrugB', 'Placebo'], n_samples, p=[0.4, 0.4, 0.2]),                          # Trial assignment data
                'dosage_mg': np.random.choice([10, 25, 50, 100], n_samples),                                                                # Medical dosage information
                'blood_type_group': np.random.choice(['A', 'B', 'AB', 'O'], n_samples),                                                     # Medical characteristic (simplified)
                'has_preexisting_condition': np.random.choice([True, False], n_samples, p=[0.3, 0.7]),                                      # Medical history flag
                'emergency_contact_relation': np.random.choice(['Spouse', 'Parent', 'Sibling', 'Friend'], n_samples),                       # Relationship information
                'physician_id': [f"DOC{np.random.randint(100,199)}" for _ in range(n_samples)]                                              # Medical provider identifier
            })
            
        else:
            # Handle unknown dataset requests with informative error message
            st.error(f"❌ Unknown dataset requested: {dataset_name}")
            return None
            
        # Return successfully generated dataset for anonymization testing
        return df
        
    except Exception as e:
        # Comprehensive error handling with detailed debugging information
        st.error(f"🚨 Error loading dataset {dataset_name}: {e}\n{traceback.format_exc()}")
        return None

def get_dataset_info(dataset_name, dataset_size="sample"):
    """Get information about the dataset with size-specific details"""
    
    base_info = {
        # Scikit-learn Classic
        "iris": {"description": "Iris flower species classification (3 classes). Famous ML dataset.", "base_samples": 150, "features": 4, "target_cols": ["target"], "sensitive_columns": [], "recommended_techniques": ["Classification-aware methods"]},
        "wine": {"description": "Wine recognition dataset; chemical analysis of wines (3 classes).", "base_samples": 178, "features": 13, "target_cols": ["target"], "sensitive_columns": [], "recommended_techniques": ["Generalization"]},
        "breast_cancer": {"description": "Breast cancer diagnosis (binary classification). Features are computed from digitized image of a fine needle aspirate (FNA) of a breast mass.", "base_samples": 569, "features": 30, "target_cols": ["target"], "sensitive_columns": ["target"], "recommended_techniques": ["Suppression", "Randomization"]},
        "diabetes": {"description": "Diabetes progression one year after baseline. Ten baseline variables, age, sex, body mass index, average blood pressure, and six blood serum measurements.", "base_samples": 442, "features": 10, "target_cols": ["target"], "sensitive_columns": ["age", "sex"], "recommended_techniques": ["Numeric Perturbation", "Aggregation"]},
        "digits": {"description": "Handwritten digits recognition (10 classes). Each feature is an 8x8 image of a digit.", "base_samples": 1797, "features": 64, "target_cols": ["target"], "sensitive_columns": [], "recommended_techniques": ["Dimensionality Reduction based Anonymization"]},
        "california_housing": {"description": "California housing prices. Based on the 1990 California census data.", "base_samples": 20640, "features": 8, "target_cols": ["MedHouseVal"], "sensitive_columns": ["MedInc", "HouseAge", "AveRooms"], "recommended_techniques": ["Microaggregation", "Top/Bottom Coding"]},

        # Synthetic ML Archetypes
        "boston_synthetic": {"description": "Synthetic regression dataset inspired by Boston Housing.", "base_samples": 506, "features": 13 + 2, "target_cols": ["target_price"], "sensitive_columns": ["owner_name_initials", "street_number_hash"], "recommended_techniques": ["Redaction", "Hashing"]},
        "credit_approval_synthetic": {"description": "Synthetic binary classification for credit approval.", "base_samples": 690, "features": 15 + 2, "target_cols": ["approved"], "sensitive_columns": ["applicant_id", "zip_code_prefix", "attr_1", "attr_2"], "recommended_techniques": ["Pseudonymization", "Suppression"]},
        "marketing_campaign_synthetic": {"description": "Synthetic binary classification for marketing campaign subscription.", "base_samples": 2240, "features": 10 + 2, "target_cols": ["subscribed"], "sensitive_columns": ["customer_email_domain", "last_contact_days_ago"], "recommended_techniques": ["Generalization", "Redaction"]},
        "flower_species_synthetic": {"description": "Synthetic multi-class classification for flower species.", "base_samples": 300, "features": 4 + 1, "target_cols": ["species_type"], "sensitive_columns": ["observation_id"], "recommended_techniques": ["k-Anonymity"]},
        "music_genre_synthetic": {"description": "Synthetic multi-class classification for music genres.", "base_samples": 1000, "features": 20 + 1, "target_cols": ["genre"], "sensitive_columns": ["track_id_suffix", "audio_feature_1"], "recommended_techniques": ["l-Diversity"]},
        "stock_prices_synthetic": {"description": "Synthetic regression for stock price changes.", "base_samples": 500, "features": 5 + 2, "target_cols": ["price_change"], "sensitive_columns": ["company_ticker_generic", "trade_date_offset"], "recommended_techniques": ["Noise Addition", "Aggregation"]},

        # OpenML Datasets
        "titanic": {"description": "Titanic passenger survival data (binary classification).", "base_samples": 1309, "features": 13, "target_cols": ["survived"], "sensitive_columns": ["name", "ticket", "cabin", "age"], "recommended_techniques": ["Redaction", "Categorization"]},
        "heart_disease_cleveland": {"description": "Cleveland Heart Disease dataset for predicting presence of heart disease.", "base_samples": 303, "features": 13, "target_cols": ["class"], "sensitive_columns": ["age", "sex"], "recommended_techniques": ["Suppression", "Generalization"]},
        "auto_mpg": {"description": "Predicting city-cycle fuel consumption in miles per gallon.", "base_samples": 398, "features": 8, "target_cols": ["mpg"], "sensitive_columns": ["name", "origin"], "recommended_techniques": ["Rounding", "Top/Bottom Coding"]},
        "adult_income": {"description": "Predict whether income exceeds $50K/yr based on census data (Adult dataset).", "base_samples": 48842, "features": 14, "target_cols": ["class"], "sensitive_columns": ["age", "education", "marital-status", "occupation", "race", "sex", "native-country"], "recommended_techniques": ["Generalization", "Suppression", "k-Anonymity"]},
        
        # Existing Synthetic Datasets
        "customer_data": {
            "description": "Customer database with PII and financial information.",
            "base_samples": 1000, "features": 10, 
            "sensitive_columns": ["first_name", "last_name", "phone_number", "email", "zip_code"],
            "recommended_techniques": ["Basic Redaction", "Numeric Scaler", "Pseudonymization"]
        },
        "employee_records": {
            "description": "Employee HR records with salaries and personal data.",
            "base_samples": 800, "features": 10, 
            "sensitive_columns": ["full_name", "ssn", "birth_date", "salary"],
            "recommended_techniques": ["Basic Redaction", "Numeric Perturbation", "Hashing"]
        },
        "medical_records": {
            "description": "Patient medical records with health information.",
            "base_samples": 1200, "features": 10, 
            "sensitive_columns": ["patient_name", "insurance_id", "doctor_name", "blood_pressure", "diagnosis"],
            "recommended_techniques": ["Basic Redaction", "Generalization", "Suppression"]
        },
        "financial_transactions": {
            "description": "Bank transaction records with account details.",
            "base_samples": 2000, "features": 10, 
            "sensitive_columns": ["account_number", "customer_name", "card_last_four", "balance_after"],
            "recommended_techniques": ["Basic Redaction", "Masking", "Noise Addition"]
        },
        "survey_responses": {
            "description": "Survey data with demographic and contact information.",
            "base_samples": 1500, "features": 9,
            "sensitive_columns": ["respondent_email_hash", "ip_octet_1_2", "feedback_keywords"],
            "recommended_techniques": ["Basic Redaction", "Aggregation", "Top/Bottom Coding"]
        },
        "clinical_trials": {
            "description": "Clinical trial participant data with medical history.",
            "base_samples": 950, "features": 10, 
            "sensitive_columns": ["participant_ssn_last4", "participant_initials", "birth_year", "emergency_contact_relation"],
            "recommended_techniques": ["Basic Redaction", "Pseudonymization", "Generalization"]
        }
    }
    
    info = base_info.get(dataset_name, {})
    if not info:
        return {}
    
    # Calculate actual samples based on size multiplier
    size_multipliers = {"sample": 1, "medium": 3, "full": 10}
    multiplier = size_multipliers.get(dataset_size, 1)
    
    info = info.copy()

    # For large datasets, don't multiply base_samples
    if dataset_name not in ["california_housing", "adult_income"]:
        info["samples"] = info["base_samples"] * multiplier
    else:
        info["samples"] = info["base_samples"]

    info["dataset_size"] = dataset_size.title()
    
    return info

# Update the main area display when sample dataset is loaded
# Add sample dataset info display if loaded
if (hasattr(st.session_state, 'sample_dataset_info') and 
    hasattr(st.session_state, 'loaded_sample_dataset_name') and
    st.session_state.sample_dataset_info and
    st.session_state.persisted_uploaded_file_bytes is not None):
    
    st.info(f"📊 **Sample Dataset Loaded:** {st.session_state.loaded_sample_dataset_name}")
    
    # Show dataset overview WITH SIZE INFO
    info = st.session_state.sample_dataset_info
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric("📏 Samples", f"{info.get('samples', 0):,}")
    with col2:
        st.metric("📊 Features", info.get('features', 0))
    with col3:
        st.metric("🔒 Sensitive Columns", len(info.get('sensitive_columns', [])))
    with col4:
        st.metric("📦 Dataset Size", info.get('dataset_size', 'sample').title())
    with col5:
        if st.button("🗑️ Clear", help="Clear sample dataset"):
            st.rerun()
    
    # Show performance expectations
    size_key = info.get('dataset_size', 'sample')
    if size_key == "sample":
        st.success("⚡ **Fast Processing:** This sample size is optimized for quick testing and learning.")
    elif size_key == "medium":
        st.warning("⏱️ **Moderate Processing:** This medium size tests performance with realistic data volumes.")
    else:
        st.error("🐌 **Slower Processing:** This full size simulates real-world data volumes. Processing may take longer.")
    
    # Show sensitive columns info with size context
    if info.get('sensitive_columns'):
        with st.expander("🔒 Sensitive Columns Details", expanded=False):
            st.write("**Columns containing sensitive information:**")
            for col in info['sensitive_columns']:
                st.write(f"• **{col}**")
            st.write(f"**💡 Tip:** Consider applying anonymization to these columns using techniques like {', '.join(info.get('recommended_techniques', ['Basic Redaction']))}.")
            st.write(f"**📊 Scale:** With {info.get('samples', 0):,} rows, this {size_key} dataset will help you understand anonymization performance at different scales.")
            

# Dataset selection dropdown
selected_dataset = st.sidebar.selectbox(
    "Choose a sample dataset:",
    options=list(sample_dataset_options.keys()),
    key="sample_dataset_select",
    help="Select a built-in dataset to test anonymization techniques"
)

if selected_dataset != "None":
    dataset_key = sample_dataset_options[selected_dataset]
    
    # Show quick info about the dataset WITH SIZE INFO
    info = get_dataset_info(dataset_key, dataset_size_key)
    if info:
        with st.sidebar.expander(f"📋 About {selected_dataset} ({selected_size})", expanded=False):
            col1, col2 = st.sidebar.columns(2)
            with col1:
                st.write(f"**📏 Samples:** {info.get('samples', 'N/A'):,}")
                st.write(f"**📊 Features:** {info.get('features', 'N/A')}")
            with col2:
                st.write(f"**🔒 Sensitive Cols:** {len(info.get('sensitive_columns', []))}")
                st.write(f"**📦 Size:** {info.get('dataset_size', 'sample').title()}")
            
            st.write(f"**📝 Description:** {info.get('description', 'No description available')}")
            if info.get('sensitive_columns'):
                st.write(f"**🔒 Sensitive Fields:** {', '.join(info['sensitive_columns'][:3])}{'...' if len(info['sensitive_columns']) > 3 else ''}")
            if info.get('recommended_techniques'):
                st.write(f"**💡 Recommended:** {', '.join(info['recommended_techniques'])}")
    
    # Load button WITH SIZE INDICATOR
    col1, col2 = st.sidebar.columns([2, 1])
    with col1:
        load_button_text = f"📥 Load {selected_size.split('(')[0].strip()}"
        if st.sidebar.button(load_button_text, type="secondary", key="load_sample_btn"):
            with st.spinner(f"Loading {selected_dataset} ({dataset_size_key} size)..."):
                # Load the dataset WITH SIZE PARAMETER
                sample_df = load_sample_dataset(dataset_key, dataset_size_key)
                
                if sample_df is not None:
                    # Clear any existing data
                    st.session_state.persisted_uploaded_file_bytes = None
                    st.session_state.persisted_uploaded_file_name = None
                    st.session_state.last_uploader_key_persisted = None
                    st.session_state.df_anonymized_data = None
                    st.session_state.df_anonymized_source_technique = None
                    
                    # Store sample dataset as if it were uploaded
                    csv_bytes = sample_df.to_csv(index=False).encode('utf-8')
                    st.session_state.persisted_uploaded_file_bytes = csv_bytes
                    st.session_state.persisted_uploaded_file_name = f"{dataset_key}_{dataset_size_key}.csv"
                    st.session_state.last_uploader_key_persisted = "file_uploader_main_data"
                    
                    # Store dataset info for display (with size info)
                    st.session_state.sample_dataset_info = info
                    st.session_state.loaded_sample_dataset_name = f"{selected_dataset} ({dataset_size_key.title()})"
                    
                    st.sidebar.success(f"✅ {selected_dataset} ({dataset_size_key}) loaded successfully!")
                    st.sidebar.info(f"📊 Loaded {len(sample_df):,} rows × {len(sample_df.columns)} columns")
                    st.rerun()
                else:
                    st.sidebar.error(f"❌ Failed to load {selected_dataset}")
    
    with col2:
        if st.sidebar.button("🔍", key="preview_sample_btn", help="Quick preview"):
            with st.spinner("Loading preview..."):
                preview_df = load_sample_dataset(dataset_key, dataset_size_key)
                if preview_df is not None:
                    st.sidebar.write(f"**👀 Preview: {selected_dataset} ({dataset_size_key})**")
                    st.sidebar.dataframe(preview_df.head(5), use_container_width=True)
                    st.sidebar.caption(f"Showing first 5 rows of {len(preview_df):,} total rows")

# Show loaded sample dataset info WITH SIZE
if (hasattr(st.session_state, 'sample_dataset_info') and 
    hasattr(st.session_state, 'loaded_sample_dataset_name') and
    st.session_state.sample_dataset_info):
    
    info = st.session_state.sample_dataset_info
    st.sidebar.success(f"📊 Dataset loaded: **{st.session_state.loaded_sample_dataset_name}**")
    
    # Quick stats with size info
    col1, col2 = st.sidebar.columns(2)
    with col1:
        st.metric("📏 Samples", f"{info.get('samples', 0):,}")
        st.metric("🔒 Sensitive", len(info.get('sensitive_columns', [])))
    with col2:
        st.metric("📦 Size", info.get('dataset_size', 'sample').title())
        
        # Performance indicator
        size_key = info.get('dataset_size', 'sample')
        if size_key == "sample":
            st.caption("⚡ Fast processing")
        elif size_key == "medium":  
            st.caption("⏱️ Moderate processing")
        else:
            st.caption("🐌 Slower processing")
    
    # Download button for original dataset
    if (st.session_state.persisted_uploaded_file_bytes and 
        st.session_state.persisted_uploaded_file_name):
        st.sidebar.download_button(
            label="📥 Download Original Dataset",
            data=st.session_state.persisted_uploaded_file_bytes,
            file_name=st.session_state.persisted_uploaded_file_name,
            mime="text/csv",
            key="download_original_sample_dataset",
            help="Download the loaded original dataset for future use",
            use_container_width=True
        )
    
    # Recommended techniques
    if info.get('recommended_techniques'):
        st.sidebar.info(f"💡 **Recommended techniques:** {', '.join(info['recommended_techniques'])}")

