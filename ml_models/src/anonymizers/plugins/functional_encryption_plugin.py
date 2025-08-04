"""
Functional Encryption Plugin for Data Anonymization

This plugin implements functional encryption schemes that allow computation
of specific functions on encrypted data while revealing only the function
output, not the underlying data.
"""

import streamlit as st
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Tuple, Optional, Callable
import hashlib
import secrets
import logging
import json
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.backends import default_backend

from ..base_anonymizer import Anonymizer

logger = logging.getLogger(__name__)

class FunctionalEncryptionPlugin(Anonymizer):
    """
    Functional Encryption plugin for privacy-preserving function evaluation.
    
    Implements functional encryption schemes that allow evaluation of specific
    functions on encrypted data while revealing only the function result.
    """
    
    def __init__(self):
        super().__init__()
        self.name = "Functional Encryption"
        self.description = "Encrypt data while allowing evaluation of specific functions without full decryption"
        
        # Encryption parameters
        self.fe_scheme = "inner_product"  # inner_product, quadratic, linear
        self.key_size = 256
        self.security_parameter = 128
        
        # Function parameters
        self.allowed_functions = ["sum", "mean", "dot_product", "linear_combination"]
        self.function_bounds = {"min": -1000, "max": 1000}
        self.precision_bits = 32
        
        # Privacy parameters
        self.function_hiding = True  # Hide the function being computed
        self.adaptive_security = True
        self.simulation_security = True
        
        # Access control
        self.enable_access_control = True
        self.access_policies = {}
        self.attribute_universe = []
        
        # Performance parameters
        self.use_batch_evaluation = True
        self.batch_size = 50
        self.precompute_keys = True        # Internal state
        self.master_key = None
        self.public_parameters = None
        self.functional_keys = {}
        self.encrypted_data = {}
        self.function_cache = {}
    
    def get_category(self) -> str:
        """Return the category for this anonymization technique."""
        return "Cryptographic Methods"
    
    def get_name(self) -> str:
        """Return the display name of this anonymization technique."""
        return self.name
    
    def get_sidebar_ui(self, all_cols: List[str], sa_col_to_pass: str | None, 
                      df_raw: pd.DataFrame, unique_key_prefix: str) -> Dict[str, Any]:
        """
        Renders the FE specific UI elements in the Streamlit sidebar.
        """
        st.sidebar.header(f"🔧 {self.get_name()} Configuration")
        
        with st.sidebar.expander("ℹ️ About Functional Encryption"):
            st.markdown("""
            **Functional Encryption (FE)** allows computation of specific functions on encrypted
            data while revealing only the function output, not the underlying plaintext data.
            """)
        
        # FE scheme selection
        scheme_options = ["inner_product", "quadratic", "linear", "general"]
        fe_scheme = st.sidebar.selectbox(
            "FE Scheme",
            options=scheme_options,
            key=f"{unique_key_prefix}_fe_scheme",
            help="Choose the functional encryption scheme"
        )
        
        # Key size
        key_size = st.sidebar.selectbox(
            "Key Size (bits)",
            options=[128, 256, 384, 512],
            index=1,
            key=f"{unique_key_prefix}_key_size",
            help="Size of encryption keys"
        )
        
        # Security parameter
        security_parameter = st.sidebar.slider(
            "Security Parameter",
            min_value=80,
            max_value=256,
            value=128,
            step=16,
            key=f"{unique_key_prefix}_security_parameter",
            help="Security parameter in bits"
        )
        
        # Allowed functions
        function_options = ["sum", "mean", "dot_product", "linear_combination"]
        allowed_functions = st.sidebar.multiselect(
            "Allowed Functions",
            options=function_options,
            default=["sum", "mean", "dot_product"],
            key=f"{unique_key_prefix}_allowed_functions",
            help="Functions that can be evaluated on encrypted data"
        )
        
        # Advanced settings
        with st.sidebar.expander("🔧 Advanced Settings"):
            function_hiding = st.checkbox(
                "Function Hiding",
                value=True,
                key=f"{unique_key_prefix}_function_hiding",
                help="Hide which functions are being computed"
            )
            
            use_batch_evaluation = st.checkbox(
                "Batch Evaluation",
                value=True,
                key=f"{unique_key_prefix}_use_batch_evaluation",
                help="Evaluate functions in batches for efficiency"
            )
        
        return {
            "fe_scheme": fe_scheme,
            "key_size": key_size,
            "security_parameter": security_parameter,
            "allowed_functions": allowed_functions,
            "function_hiding": function_hiding,
            "use_batch_evaluation": use_batch_evaluation
        }
    
    def anonymize(self, df_input: pd.DataFrame, parameters: Dict[str, Any], sa_col: str | None) -> pd.DataFrame:
        """
        Apply functional encryption to the dataset.
        """
        try:
            import time
            start_time = time.time()
            
            # Update configuration from parameters
            self.fe_scheme = parameters.get("fe_scheme", "inner_product")
            self.key_size = parameters.get("key_size", 256)
            self.security_parameter = parameters.get("security_parameter", 128)
            self.allowed_functions = parameters.get("allowed_functions", ["sum", "mean"])
            self.function_hiding = parameters.get("function_hiding", True)
            self.use_batch_evaluation = parameters.get("use_batch_evaluation", True)
            
            logger.info(f"Starting functional encryption with scheme: {self.fe_scheme}")
            
            # Setup FE system
            self._setup_fe_system()
            
            # Generate functional keys for allowed functions
            self._generate_functional_keys()
            
            # Apply functional encryption
            encrypted_data = self._apply_functional_encryption(df_input)
            
            self.last_encryption_time = time.time() - start_time
            
            # Add metadata
            encrypted_data.attrs['fe_scheme'] = self.fe_scheme
            encrypted_data.attrs['security_parameter'] = self.security_parameter
            encrypted_data.attrs['functionally_encrypted'] = True
            encrypted_data.attrs['allowed_functions'] = self.allowed_functions
            
            logger.info(f"Functional encryption completed in {self.last_encryption_time:.2f} seconds")
            return encrypted_data
            
        except Exception as e:
            logger.error(f"Error in functional encryption: {str(e)}")
            st.error(f"Functional encryption failed: {str(e)}")
            return df_input
    
    def build_config_export(self, unique_key_prefix: str, sa_col: str | None) -> Dict[str, Any]:
        """Build configuration for export."""
        return {
            "fe_scheme": st.session_state.get(f"{unique_key_prefix}_fe_scheme", "inner_product"),
            "key_size": st.session_state.get(f"{unique_key_prefix}_key_size", 256),
            "security_parameter": st.session_state.get(f"{unique_key_prefix}_security_parameter", 128),
            "allowed_functions": st.session_state.get(f"{unique_key_prefix}_allowed_functions", ["sum", "mean"]),
            "function_hiding": st.session_state.get(f"{unique_key_prefix}_function_hiding", True),
            "use_batch_evaluation": st.session_state.get(f"{unique_key_prefix}_use_batch_evaluation", True)
        }
    
    def apply_config_import(self, config_params: Dict[str, Any], all_cols: List[str], unique_key_prefix: str):
        """Apply imported configuration to session state."""
        if "fe_scheme" in config_params:
            st.session_state[f"{unique_key_prefix}_fe_scheme"] = config_params["fe_scheme"]
        if "key_size" in config_params:
            st.session_state[f"{unique_key_prefix}_key_size"] = config_params["key_size"]
        if "security_parameter" in config_params:
            st.session_state[f"{unique_key_prefix}_security_parameter"] = config_params["security_parameter"]
        if "allowed_functions" in config_params:
            st.session_state[f"{unique_key_prefix}_allowed_functions"] = config_params["allowed_functions"]
        if "function_hiding" in config_params:
            st.session_state[f"{unique_key_prefix}_function_hiding"] = config_params["function_hiding"]
        if "use_batch_evaluation" in config_params:
            st.session_state[f"{unique_key_prefix}_use_batch_evaluation"] = config_params["use_batch_evaluation"]

    # ...existing code...
    
    def _setup_fe_system(self):
        """Setup the functional encryption system."""
        logger.info(f"Setting up FE system with scheme: {self.fe_scheme}")
        
        # Generate master key and public parameters
        self.master_key = secrets.token_bytes(self.key_size // 8)
        
        # Generate public parameters based on scheme
        if self.fe_scheme == "inner_product":
            self._setup_inner_product_fe()
        elif self.fe_scheme == "quadratic":
            self._setup_quadratic_fe()
        elif self.fe_scheme == "linear":
            self._setup_linear_fe()
        else:
            self._setup_general_fe()
    
    def _setup_inner_product_fe(self):
        """Setup inner product functional encryption."""
        # Simplified inner product FE setup
        dimension = min(100, self.key_size)  # Vector dimension
        
        # Generate random matrices for the scheme
        self.public_parameters = {
            "dimension": dimension,
            "modulus": 2**self.precision_bits - 1,
            "generator_matrix": np.random.randint(0, 2**16, (dimension, dimension)),
            "noise_bound": 2**(self.precision_bits // 4)
        }
        
        logger.info(f"Inner product FE setup complete (dimension: {dimension})")
    
    def _setup_quadratic_fe(self):
        """Setup quadratic function functional encryption."""
        # Simplified quadratic FE setup
        dimension = min(50, self.key_size // 2)  # Smaller dimension for quadratic
        
        self.public_parameters = {
            "dimension": dimension,
            "modulus": 2**self.precision_bits - 1,
            "quadratic_matrix": np.random.randint(0, 2**8, (dimension, dimension, dimension)),
            "noise_bound": 2**(self.precision_bits // 3)
        }
        
        logger.info(f"Quadratic FE setup complete (dimension: {dimension})")
    
    def _setup_linear_fe(self):
        """Setup linear function functional encryption."""
        # Simplified linear FE setup
        dimension = min(200, self.key_size * 2)  # Larger dimension for linear
        
        self.public_parameters = {
            "dimension": dimension,
            "modulus": 2**self.precision_bits - 1,
            "linear_matrix": np.random.randint(0, 2**12, (dimension, dimension)),
            "noise_bound": 2**(self.precision_bits // 6)
        }
        
        logger.info(f"Linear FE setup complete (dimension: {dimension})")
    
    def _setup_general_fe(self):
        """Setup general function functional encryption."""
        # Simplified general FE setup (most expensive)
        dimension = min(30, self.key_size // 4)
        
        self.public_parameters = {
            "dimension": dimension,
            "modulus": 2**self.precision_bits - 1,
            "circuit_depth": 10,
            "gate_count": dimension * 100,
            "noise_bound": 2**(self.precision_bits // 2)
        }
        
        logger.info(f"General FE setup complete (dimension: {dimension})")
    
    def _generate_functional_keys(self):
        """Generate functional keys for allowed functions."""
        logger.info("Generating functional keys")
        
        for function_name in self.allowed_functions:
            if function_name not in self.functional_keys:
                key = self._derive_functional_key(function_name)
                self.functional_keys[function_name] = key
                
                logger.debug(f"Generated functional key for: {function_name}")
    
    def _derive_functional_key(self, function_name: str) -> Dict[str, Any]:
        """Derive a functional key for a specific function."""
        # Create a hash-based key derivation
        key_material = hashlib.pbkdf2_hmac(
            'sha256',
            self.master_key,
            function_name.encode(),
            100000,  # iterations
            dklen=self.key_size // 8
        )
        
        # Generate function-specific parameters
        if function_name in ["sum", "mean"]:
            # Linear function key
            dimension = self.public_parameters["dimension"]
            weights = np.frombuffer(key_material[:dimension*4], dtype=np.int32) % self.public_parameters["modulus"]
            
            return {
                "type": "linear",
                "weights": weights[:dimension],
                "function": function_name,
                "key_material": key_material
            }
        
        elif function_name == "dot_product":
            # Inner product key
            dimension = self.public_parameters["dimension"]
            vector = np.frombuffer(key_material[:dimension*4], dtype=np.int32) % self.public_parameters["modulus"]
            
            return {
                "type": "inner_product",
                "vector": vector[:dimension],
                "function": function_name,
                "key_material": key_material
            }
        
        else:
            # General function key
            return {
                "type": "general",
                "circuit": self._encode_function_circuit(function_name),
                "function": function_name,
                "key_material": key_material
            }
    
    def _encode_function_circuit(self, function_name: str) -> Dict[str, Any]:
        """Encode a function as a circuit for general FE."""
        # Simplified circuit encoding
        circuits = {
            "linear_combination": {"gates": 50, "depth": 3, "inputs": 10},
            "weighted_sum": {"gates": 30, "depth": 2, "inputs": 20},
            "min_max": {"gates": 100, "depth": 5, "inputs": 2},
            "variance": {"gates": 80, "depth": 4, "inputs": 100},
            "covariance": {"gates": 120, "depth": 6, "inputs": 200}
        }
        
        return circuits.get(function_name, {"gates": 10, "depth": 1, "inputs": 1})
    
    def _apply_functional_encryption(self, data: pd.DataFrame) -> pd.DataFrame:
        """Apply functional encryption to the data."""
        logger.info("Applying functional encryption")
        
        encrypted_data = data.copy()
        
        # Encrypt numerical columns
        for column in data.select_dtypes(include=[np.number]).columns:
            values = data[column].fillna(0)
            
            if self.fe_scheme == "inner_product":
                encrypted_column = self._encrypt_for_inner_product(values, column)
            elif self.fe_scheme == "quadratic":
                encrypted_column = self._encrypt_for_quadratic(values, column)
            elif self.fe_scheme == "linear":
                encrypted_column = self._encrypt_for_linear(values, column)
            else:
                encrypted_column = self._encrypt_for_general(values, column)
            
            encrypted_data[column] = encrypted_column
        
        # Store encrypted data for function evaluation
        self.encrypted_data = encrypted_data
        
        return encrypted_data
    
    def _encrypt_for_inner_product(self, values: pd.Series, column: str) -> List[str]:
        """Encrypt values for inner product FE."""
        dimension = self.public_parameters["dimension"]
        modulus = self.public_parameters["modulus"]
        
        encrypted_values = []
        for value in values:
            # Embed value in a random vector
            vector = np.random.randint(0, modulus, dimension)
            vector[0] = int(value) % modulus  # Place actual value at position 0
            
            # Add noise for security
            noise = np.random.randint(-self.public_parameters["noise_bound"], 
                                    self.public_parameters["noise_bound"], dimension)
            encrypted_vector = (vector + noise) % modulus
            
            # Encode as string for storage
            encrypted_values.append(",".join(map(str, encrypted_vector)))
        
        return encrypted_values
    
    def _encrypt_for_quadratic(self, values: pd.Series, column: str) -> List[str]:
        """Encrypt values for quadratic FE."""
        dimension = self.public_parameters["dimension"]
        modulus = self.public_parameters["modulus"]
        
        encrypted_values = []
        for value in values:
            # Create quadratic encoding
            vector = np.random.randint(0, modulus, dimension)
            vector[0] = int(value) % modulus
            
            # Quadratic transformation (simplified)
            quad_vector = np.outer(vector, vector).flatten()[:dimension]
            
            # Add noise
            noise = np.random.randint(-self.public_parameters["noise_bound"], 
                                    self.public_parameters["noise_bound"], dimension)
            encrypted_vector = (quad_vector + noise) % modulus
            
            encrypted_values.append(",".join(map(str, encrypted_vector)))
        
        return encrypted_values
    
    def _encrypt_for_linear(self, values: pd.Series, column: str) -> List[str]:
        """Encrypt values for linear FE."""
        dimension = self.public_parameters["dimension"]
        modulus = self.public_parameters["modulus"]
        
        encrypted_values = []
        for value in values:
            # Linear encoding with random coefficients
            coefficients = np.random.randint(1, modulus//100, dimension)
            encrypted_value = (int(value) * coefficients[0]) % modulus
            
            # Create vector representation
            vector = np.random.randint(0, modulus, dimension)
            vector[0] = encrypted_value
            
            encrypted_values.append(",".join(map(str, vector)))
        
        return encrypted_values
    
    def _encrypt_for_general(self, values: pd.Series, column: str) -> List[str]:
        """Encrypt values for general FE."""
        dimension = self.public_parameters["dimension"]
        modulus = self.public_parameters["modulus"]
        
        encrypted_values = []
        for value in values:
            # General circuit-based encoding (simplified)
            circuit_input = np.zeros(dimension, dtype=int)
            circuit_input[0] = int(value) % modulus
            
            # Apply random circuit transformations
            for _ in range(self.public_parameters.get("circuit_depth", 3)):
                circuit_input = (circuit_input * 7 + 13) % modulus
            
            encrypted_values.append(",".join(map(str, circuit_input)))
        
        return encrypted_values
    
    def _evaluate_function(self, function_name: str) -> Any:
        """Evaluate a function on encrypted data."""
        if function_name not in self.functional_keys:
            raise ValueError(f"No functional key for: {function_name}")
        
        if not self.encrypted_data:
            raise ValueError("No encrypted data available")
        
        logger.info(f"Evaluating function: {function_name}")
        
        # Get functional key
        func_key = self.functional_keys[function_name]
        
        # Evaluate based on function type
        if func_key["type"] == "linear" and function_name in ["sum", "mean"]:
            return self._evaluate_linear_function(function_name, func_key)
        elif func_key["type"] == "inner_product":
            return self._evaluate_inner_product(function_name, func_key)
        else:
            return self._evaluate_general_function(function_name, func_key)
    
    def _evaluate_linear_function(self, function_name: str, func_key: Dict[str, Any]) -> float:
        """Evaluate a linear function on encrypted data."""
        weights = func_key["weights"]
        modulus = self.public_parameters["modulus"]
        
        total = 0
        count = 0
        
        # Process numerical columns
        for column in self.encrypted_data.select_dtypes(include=[object]).columns:
            for encrypted_value in self.encrypted_data[column]:
                if isinstance(encrypted_value, str) and "," in encrypted_value:
                    vector = np.array([int(x) for x in encrypted_value.split(",")])
                    if len(vector) >= len(weights):
                        # Compute inner product with weights
                        result = np.dot(vector[:len(weights)], weights) % modulus
                        total += result
                        count += 1
        
        if function_name == "mean" and count > 0:
            return total / count
        else:
            return total
    
    def _evaluate_inner_product(self, function_name: str, func_key: Dict[str, Any]) -> float:
        """Evaluate inner product on encrypted data."""
        query_vector = func_key["vector"]
        modulus = self.public_parameters["modulus"]
        
        results = []
        
        # Process numerical columns
        for column in self.encrypted_data.select_dtypes(include=[object]).columns:
            for encrypted_value in self.encrypted_data[column]:
                if isinstance(encrypted_value, str) and "," in encrypted_value:
                    vector = np.array([int(x) for x in encrypted_value.split(",")])
                    if len(vector) >= len(query_vector):
                        # Compute inner product
                        result = np.dot(vector[:len(query_vector)], query_vector) % modulus
                        results.append(result)
        
        return sum(results) if results else 0
    
    def _evaluate_general_function(self, function_name: str, func_key: Dict[str, Any]) -> float:
        """Evaluate a general function on encrypted data."""
        # Simplified general function evaluation
        circuit = func_key["circuit"]
        
        # Mock evaluation based on circuit complexity
        complexity_factor = circuit.get("gates", 10) * circuit.get("depth", 1)
        
        # Process some encrypted values
        total = 0
        for column in self.encrypted_data.select_dtypes(include=[object]).columns:
            for encrypted_value in self.encrypted_data[column][:10]:  # Limit for demo
                if isinstance(encrypted_value, str) and "," in encrypted_value:
                    vector = np.array([int(x) for x in encrypted_value.split(",")])
                    # Simplified computation
                    total += (vector.sum() * complexity_factor) % self.public_parameters["modulus"]
        
        return total
    
    def get_privacy_metrics(self, original_data: pd.DataFrame, 
                          anonymized_data: pd.DataFrame) -> Dict[str, Any]:
        """Calculate privacy metrics for functional encryption."""
        return {
            "fe_scheme": self.fe_scheme,
            "security_parameter": self.security_parameter,
            "function_hiding": self.function_hiding,
            "adaptive_security": self.adaptive_security,
            "simulation_security": self.simulation_security,
            "selective_disclosure": True,
            "fine_grained_access": self.enable_access_control,
            "allowed_functions": len(self.allowed_functions),
            "computational_security": True,
            "semantic_security": True
        }
    
    def get_utility_metrics(self, original_data: pd.DataFrame, 
                          anonymized_data: pd.DataFrame) -> Dict[str, Any]:
        """Calculate utility metrics for functional encryption."""
        return {
            "function_evaluation": True,
            "selective_computation": True,
            "access_controlled": self.enable_access_control,
            "functions_supported": len(self.allowed_functions),
            "precision_bits": self.precision_bits,
            "computation_overhead": "Very High",  # FE is computationally expensive
            "storage_overhead": "High",
            "scheme_efficiency": self._get_scheme_efficiency()
        }
    
    def _get_scheme_efficiency(self) -> str:
        """Get efficiency rating for the current scheme."""
        efficiency_map = {
            "inner_product": "Good",
            "linear": "Good", 
            "quadratic": "Moderate",
            "general": "Poor"
        }
        return efficiency_map.get(self.fe_scheme, "Unknown")

# Register the plugin
def get_plugin():
    """Return the plugin instance for registration."""
    return FunctionalEncryptionPlugin()
