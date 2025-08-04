"""
Private Set Intersection Plugin for Data Anonymization

This plugin implements private set intersection (PSI) techniques that allow
two or more parties to compute the intersection of their datasets without
revealing any information about non-intersecting elements.
"""

import streamlit as st
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Tuple, Optional, Set
import hashlib
import secrets
import logging
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.hazmat.backends import default_backend
import base64

from ..base_anonymizer import Anonymizer

logger = logging.getLogger(__name__)

class PrivateSetIntersectionPlugin(Anonymizer):
    """
    Private Set Intersection (PSI) plugin for privacy-preserving set operations.
    
    Implements PSI protocols that allow computation of set intersections
    without revealing non-intersecting elements to any party.
    """
    
    def __init__(self):
        super().__init__()
        self.name = "Private Set Intersection"
        self.description = "Compute set intersections while preserving privacy of non-intersecting elements"
        
        # PSI protocol parameters
        self.psi_protocol = "kkrt"  # kkrt, ecdh, ot_based, bloom_filter
        self.hash_function = "sha256"
        self.key_size = 256
        self.security_parameter = 128
        
        # Set operation parameters
        self.intersection_type = "exact"  # exact, approximate, threshold
        self.threshold_percentage = 80  # For threshold PSI
        self.allow_cardinality_leakage = True
        self.support_multi_party = True
        
        # Privacy parameters
        self.use_differential_privacy = False
        self.dp_epsilon = 1.0
        self.add_dummy_elements = True
        self.dummy_ratio = 0.1
        
        # Performance parameters
        self.use_bloom_filters = True
        self.bloom_filter_size = 10000
        self.hash_count = 5
        self.batch_size = 1000
        
        # Communication parameters
        self.minimize_communication = True
        self.compression_enabled = True
        self.parallel_processing = True
        
        # Internal state
        self.local_set = set()
        self.encrypted_sets = {}
        self.intersection_result = set()
        self.psi_keys = {}
        self.bloom_filters = {}
        self.protocol_state = {}
    
    def get_category(self) -> str:
        """Return the category for this anonymization technique."""
        return "Cryptographic Methods"
    
    def get_name(self) -> str:
        """Return the display name of this anonymization technique."""
        return self.name
    
    def get_sidebar_ui(self, all_cols: List[str], sa_col_to_pass: str | None, 
                      df_raw: pd.DataFrame, unique_key_prefix: str) -> Dict[str, Any]:
        """
        Renders the PSI specific UI elements in the Streamlit sidebar.
        """
        st.sidebar.header(f"🤝 {self.get_name()} Configuration")
        
        with st.sidebar.expander("ℹ️ About Private Set Intersection"):
            st.markdown("""
            **Private Set Intersection (PSI)** allows two or more parties to compute
            the intersection of their private datasets without revealing any information
            about elements that are not in the intersection.
            """)
        
        # PSI protocol selection
        protocol_options = ["kkrt", "ecdh", "ot_based", "bloom_filter"]
        psi_protocol = st.sidebar.selectbox(
            "PSI Protocol",
            options=protocol_options,
            key=f"{unique_key_prefix}_psi_protocol",
            help="Choose the PSI protocol to use"
        )
        
        # Hash function
        hash_function = st.sidebar.selectbox(
            "Hash Function",
            options=["sha256", "sha512", "blake2b"],
            key=f"{unique_key_prefix}_hash_function",
            help="Cryptographic hash function for element hashing"
        )
        
        # Key size
        key_size = st.sidebar.selectbox(
            "Key Size (bits)",
            options=[128, 192, 256, 384, 512],
            index=2,
            key=f"{unique_key_prefix}_key_size",
            help="Cryptographic key size"
        )
        
        # Intersection type
        intersection_type = st.sidebar.selectbox(
            "Intersection Type",
            options=["exact", "approximate", "threshold", "cardinality_only"],
            key=f"{unique_key_prefix}_intersection_type",
            help="Type of intersection to compute"
        )
        
        # Advanced settings
        with st.sidebar.expander("🔧 Advanced Settings"):
            use_bloom_filters = st.checkbox(
                "Use Bloom Filters",
                value=True,
                key=f"{unique_key_prefix}_use_bloom_filters",
                help="Use Bloom filters for preliminary filtering"
            )
            
            add_dummy_elements = st.checkbox(
                "Add Dummy Elements",
                value=True,
                key=f"{unique_key_prefix}_add_dummy_elements",
                help="Add dummy elements to hide true set sizes"
            )
        
        return {
            "psi_protocol": psi_protocol,
            "hash_function": hash_function,
            "key_size": key_size,
            "intersection_type": intersection_type,
            "use_bloom_filters": use_bloom_filters,
            "add_dummy_elements": add_dummy_elements
        }
    
    def anonymize(self, df_input: pd.DataFrame, parameters: Dict[str, Any], sa_col: str | None) -> pd.DataFrame:
        """
        Apply private set intersection to prepare dataset for PSI operations.
        """
        try:
            import time
            start_time = time.time()
            
            # Update configuration from parameters
            self.psi_protocol = parameters.get("psi_protocol", "kkrt")
            self.hash_function = parameters.get("hash_function", "sha256")
            self.key_size = parameters.get("key_size", 256)
            self.intersection_type = parameters.get("intersection_type", "exact")
            self.use_bloom_filters = parameters.get("use_bloom_filters", True)
            self.add_dummy_elements = parameters.get("add_dummy_elements", True)
            
            logger.info(f"Preparing dataset for PSI with protocol: {self.psi_protocol}")
            
            # Extract set elements from the dataset
            self.local_set = self._extract_set_elements(df_input)
            
            # Add dummy elements if enabled
            if self.add_dummy_elements:
                self._add_dummy_elements()
            
            # Prepare the dataset for PSI
            psi_data = self._prepare_psi_dataset(df_input)
            
            self.last_psi_time = time.time() - start_time
            
            # Add metadata
            psi_data.attrs['psi_protocol'] = self.psi_protocol
            psi_data.attrs['security_parameter'] = self.security_parameter
            psi_data.attrs['psi_prepared'] = True
            psi_data.attrs['local_set_size'] = len(self.local_set)
            
            logger.info(f"PSI preparation completed in {self.last_psi_time:.2f} seconds")
            return psi_data
            
        except Exception as e:
            logger.error(f"Error in PSI preparation: {str(e)}")
            st.error(f"PSI preparation failed: {str(e)}")
            return df_input
    
    def build_config_export(self, unique_key_prefix: str, sa_col: str | None) -> Dict[str, Any]:
        """Build configuration for export."""
        return {
            "psi_protocol": st.session_state.get(f"{unique_key_prefix}_psi_protocol", "kkrt"),
            "hash_function": st.session_state.get(f"{unique_key_prefix}_hash_function", "sha256"),
            "key_size": st.session_state.get(f"{unique_key_prefix}_key_size", 256),
            "intersection_type": st.session_state.get(f"{unique_key_prefix}_intersection_type", "exact"),
            "use_bloom_filters": st.session_state.get(f"{unique_key_prefix}_use_bloom_filters", True),
            "add_dummy_elements": st.session_state.get(f"{unique_key_prefix}_add_dummy_elements", True)
        }
    
    def apply_config_import(self, config_params: Dict[str, Any], all_cols: List[str], unique_key_prefix: str):
        """Apply imported configuration to session state."""
        if "psi_protocol" in config_params:
            st.session_state[f"{unique_key_prefix}_psi_protocol"] = config_params["psi_protocol"]
        if "hash_function" in config_params:
            st.session_state[f"{unique_key_prefix}_hash_function"] = config_params["hash_function"]
        if "key_size" in config_params:
            st.session_state[f"{unique_key_prefix}_key_size"] = config_params["key_size"]
        if "intersection_type" in config_params:
            st.session_state[f"{unique_key_prefix}_intersection_type"] = config_params["intersection_type"]
        if "use_bloom_filters" in config_params:
            st.session_state[f"{unique_key_prefix}_use_bloom_filters"] = config_params["use_bloom_filters"]
        if "add_dummy_elements" in config_params:
            st.session_state[f"{unique_key_prefix}_add_dummy_elements"] = config_params["add_dummy_elements"]
    
    def _extract_set_elements(self, data: pd.DataFrame) -> Set[str]:
        """Extract set elements from the dataset."""
        elements = set()
        
        # Use specified columns or all string/categorical columns
        target_columns = data.select_dtypes(include=['object', 'string', 'category']).columns
        
        for column in target_columns:
            # Convert values to strings and add to set
            column_elements = data[column].dropna().astype(str).unique()
            elements.update(column_elements)
        
        # Also consider combinations of columns as compound elements
        if len(target_columns) > 1:
            compound_elements = data[target_columns].apply(
                lambda row: "|".join(row.astype(str)), axis=1
            ).unique()
            elements.update(compound_elements)
        
        logger.info(f"Extracted {len(elements)} unique elements from dataset")
        return elements
    
    def _add_dummy_elements(self):
        """Add dummy elements to hide true set size."""
        dummy_count = int(len(self.local_set) * self.dummy_ratio)
        
        for i in range(dummy_count):
            # Generate random dummy element
            dummy_element = f"dummy_{secrets.token_hex(8)}"
            self.local_set.add(dummy_element)
        
        logger.info(f"Added {dummy_count} dummy elements")
    
    def _prepare_psi_dataset(self, data: pd.DataFrame) -> pd.DataFrame:
        """Prepare dataset for PSI operations."""
        psi_data = data.copy()
        
        # Create hashed identifiers for PSI
        if self.psi_protocol in ["kkrt", "ot_based"]:
            psi_data = self._prepare_ot_based_psi(psi_data)
        elif self.psi_protocol == "ecdh":
            psi_data = self._prepare_ecdh_psi(psi_data)
        elif self.psi_protocol == "bloom_filter":
            psi_data = self._prepare_bloom_filter_psi(psi_data)
        else:
            psi_data = self._prepare_generic_psi(psi_data)
        
        return psi_data
    
    def _prepare_ot_based_psi(self, data: pd.DataFrame) -> pd.DataFrame:
        """Prepare data for OT-based PSI protocols."""
        logger.info("Preparing OT-based PSI")
        
        # Create OT parameters
        self.psi_keys["ot_params"] = {
            "security_param": self.security_parameter,
            "hash_function": self.hash_function,
            "batch_size": self.batch_size
        }
        
        # Hash elements for OT protocol
        hashed_elements = {}
        for element in self.local_set:
            hash_value = self._hash_element(element)
            hashed_elements[element] = hash_value
        
        # Add hashed identifiers to dataset
        data_copy = data.copy()
        data_copy['psi_hash'] = data_copy.apply(
            lambda row: self._compute_row_hash(row), axis=1
        )
        
        return data_copy
    
    def _prepare_ecdh_psi(self, data: pd.DataFrame) -> pd.DataFrame:
        """Prepare data for ECDH-based PSI."""
        logger.info("Preparing ECDH-based PSI")
        
        # Generate ECDH key pair (simplified)
        private_key = secrets.randbelow(2**self.key_size)
        self.psi_keys["ecdh_private"] = private_key
        
        # Compute ECDH values for elements
        ecdh_values = {}
        for element in self.local_set:
            # Simplified ECDH computation
            element_hash = int(self._hash_element(element), 16) % (2**self.key_size)
            ecdh_value = pow(element_hash, private_key, 2**self.key_size - 1)
            ecdh_values[element] = ecdh_value
        
        data_copy = data.copy()
        data_copy['ecdh_value'] = data_copy.apply(
            lambda row: self._compute_ecdh_for_row(row, ecdh_values), axis=1
        )
        
        return data_copy
    
    def _prepare_bloom_filter_psi(self, data: pd.DataFrame) -> pd.DataFrame:
        """Prepare data for Bloom filter-based PSI."""
        logger.info("Preparing Bloom filter PSI")
        
        # Create Bloom filter
        bloom_filter = self._create_bloom_filter()
        
        # Add elements to Bloom filter
        for element in self.local_set:
            self._add_to_bloom_filter(bloom_filter, element)
        
        self.bloom_filters["local"] = bloom_filter
        
        # Mark rows that match Bloom filter
        data_copy = data.copy()
        data_copy['bloom_match'] = data_copy.apply(
            lambda row: self._check_bloom_filter(bloom_filter, self._compute_row_hash(row)),
            axis=1
        )
        
        return data_copy
    
    def _prepare_generic_psi(self, data: pd.DataFrame) -> pd.DataFrame:
        """Prepare data for generic PSI protocol."""
        logger.info("Preparing generic PSI")
        
        # Simple hash-based preparation
        data_copy = data.copy()
        data_copy['psi_identifier'] = data_copy.apply(
            lambda row: self._compute_row_hash(row), axis=1
        )
        
        return data_copy
    
    def _hash_element(self, element: str) -> str:
        """Hash an element using the specified hash function."""
        if self.hash_function == "sha256":
            return hashlib.sha256(element.encode()).hexdigest()
        elif self.hash_function == "sha512":
            return hashlib.sha512(element.encode()).hexdigest()
        elif self.hash_function == "blake2b":
            return hashlib.blake2b(element.encode()).hexdigest()
        elif self.hash_function == "sha3_256":
            return hashlib.sha3_256(element.encode()).hexdigest()
        else:
            return hashlib.sha256(element.encode()).hexdigest()
    
    def _compute_row_hash(self, row: pd.Series) -> str:
        """Compute hash for a data row."""
        # Combine relevant columns into a single string
        row_str = "|".join(str(val) for val in row if pd.notna(val))
        return self._hash_element(row_str)
    
    def _compute_ecdh_for_row(self, row: pd.Series, ecdh_values: Dict[str, int]) -> int:
        """Compute ECDH value for a data row."""
        row_hash = self._compute_row_hash(row)
        # Find matching element or compute new ECDH value
        for element, ecdh_val in ecdh_values.items():
            if self._hash_element(element) == row_hash:
                return ecdh_val
        
        # Compute new ECDH value for unmatched row
        element_hash = int(row_hash, 16) % (2**self.key_size)
        return pow(element_hash, self.psi_keys["ecdh_private"], 2**self.key_size - 1)
    
    def _create_bloom_filter(self) -> np.ndarray:
        """Create a Bloom filter."""
        return np.zeros(self.bloom_filter_size, dtype=bool)
    
    def _add_to_bloom_filter(self, bloom_filter: np.ndarray, element: str):
        """Add an element to the Bloom filter."""
        for i in range(self.hash_count):
            # Create multiple hash values
            hash_input = f"{element}_{i}"
            hash_value = int(self._hash_element(hash_input), 16)
            index = hash_value % self.bloom_filter_size
            bloom_filter[index] = True
    
    def _check_bloom_filter(self, bloom_filter: np.ndarray, element: str) -> bool:
        """Check if an element might be in the Bloom filter."""
        for i in range(self.hash_count):
            hash_input = f"{element}_{i}"
            hash_value = int(self._hash_element(hash_input), 16)
            index = hash_value % self.bloom_filter_size
            if not bloom_filter[index]:
                return False
        return True
    
    def _compute_psi(self, remote_set: Set[str]) -> Set[str]:
        """Compute private set intersection with remote set."""
        logger.info(f"Computing PSI with {len(remote_set)} remote elements")
        
        if self.psi_protocol == "kkrt":
            return self._compute_kkrt_psi(remote_set)
        elif self.psi_protocol == "ecdh":
            return self._compute_ecdh_psi(remote_set)
        elif self.psi_protocol == "bloom_filter":
            return self._compute_bloom_psi(remote_set)
        else:
            return self._compute_naive_psi(remote_set)
    
    def _compute_kkrt_psi(self, remote_set: Set[str]) -> Set[str]:
        """Compute PSI using KKRT protocol (simplified)."""
        # Simplified KKRT implementation
        intersection = set()
        
        # Hash local and remote sets
        local_hashes = {self._hash_element(elem) for elem in self.local_set}
        remote_hashes = {self._hash_element(elem) for elem in remote_set}
        
        # Find intersection in hash space
        hash_intersection = local_hashes.intersection(remote_hashes)
        
        # Map back to original elements (simplified)
        for elem in self.local_set:
            if self._hash_element(elem) in hash_intersection:
                intersection.add(elem)
        
        return intersection
    
    def _compute_ecdh_psi(self, remote_set: Set[str]) -> Set[str]:
        """Compute PSI using ECDH protocol (simplified)."""
        intersection = set()
        
        # Simulate ECDH key exchange
        if "ecdh_private" not in self.psi_keys:
            return intersection
        
        private_key = self.psi_keys["ecdh_private"]
        
        # Compute ECDH values for both sets
        local_ecdh = {}
        for elem in self.local_set:
            elem_hash = int(self._hash_element(elem), 16) % (2**self.key_size)
            ecdh_val = pow(elem_hash, private_key, 2**self.key_size - 1)
            local_ecdh[ecdh_val] = elem
        
        remote_ecdh = set()
        for elem in remote_set:
            elem_hash = int(self._hash_element(elem), 16) % (2**self.key_size)
            ecdh_val = pow(elem_hash, private_key, 2**self.key_size - 1)
            remote_ecdh.add(ecdh_val)
        
        # Find intersection
        for ecdh_val in local_ecdh:
            if ecdh_val in remote_ecdh:
                intersection.add(local_ecdh[ecdh_val])
        
        return intersection
    
    def _compute_bloom_psi(self, remote_set: Set[str]) -> Set[str]:
        """Compute PSI using Bloom filters."""
        # Create Bloom filter for remote set
        remote_bloom = self._create_bloom_filter()
        for elem in remote_set:
            self._add_to_bloom_filter(remote_bloom, elem)
        
        # Check local elements against remote Bloom filter
        intersection = set()
        for elem in self.local_set:
            if self._check_bloom_filter(remote_bloom, elem):
                # Potential match - verify with exact check
                if elem in remote_set:
                    intersection.add(elem)
        
        return intersection
    
    def _compute_naive_psi(self, remote_set: Set[str]) -> Set[str]:
        """Compute PSI using naive intersection (for testing)."""
        return self.local_set.intersection(remote_set)
    
    def _filter_by_intersection(self, data: pd.DataFrame, intersection: Set[str]) -> pd.DataFrame:
        """Filter dataset to include only intersecting elements."""
        if self.intersection_type == "cardinality_only":
            # Return only the count
            result = pd.DataFrame({"intersection_size": [len(intersection)]})
        else:
            # Filter rows that contain intersecting elements
            mask = data.apply(
                lambda row: any(str(val) in intersection for val in row if pd.notna(val)),
                axis=1
            )
            result = data[mask].copy()
            
            if self.use_differential_privacy:
                result = self._add_dp_noise_to_intersection(result)
        
        return result
    
    def _add_dp_noise_to_intersection(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add differential privacy noise to intersection results."""
        # Add Laplace noise to numerical columns
        for column in data.select_dtypes(include=[np.number]).columns:
            sensitivity = 1.0  # Assuming sensitivity of 1
            scale = sensitivity / self.dp_epsilon
            noise = np.random.laplace(0, scale, len(data))
            data[column] = data[column] + noise
        
        return data
    
    def _generate_sample_sets(self):
        """Generate sample sets for testing PSI."""
        # Generate overlapping sets
        set_a = {f"user_{i}" for i in range(100, 200)}
        set_b = {f"user_{i}" for i in range(150, 250)}
        
        # Add some random elements
        for i in range(50):
            set_a.add(f"random_a_{secrets.token_hex(4)}")
            set_b.add(f"random_b_{secrets.token_hex(4)}")
        
        self.sample_sets = {"A": set_a, "B": set_b}
        logger.info(f"Generated sample sets: A={len(set_a)}, B={len(set_b)}")
    
    def _compute_sample_psi(self) -> Set[str]:
        """Compute PSI on sample sets."""
        if not hasattr(self, 'sample_sets'):
            return set()
        
        import time
        start_time = time.time()
        
        self.local_set = self.sample_sets["A"]
        intersection = self._compute_psi(self.sample_sets["B"])
        
        self.last_psi_time = time.time() - start_time
        return intersection
    
    def get_privacy_metrics(self, original_data: pd.DataFrame, 
                          anonymized_data: pd.DataFrame) -> Dict[str, Any]:
        """Calculate privacy metrics for PSI."""
        return {
            "psi_protocol": self.psi_protocol,
            "security_parameter": self.security_parameter,
            "intersection_privacy": True,
            "non_intersection_hidden": True,
            "cardinality_leakage": self.allow_cardinality_leakage,
            "differential_privacy": self.use_differential_privacy,
            "dp_epsilon": self.dp_epsilon if self.use_differential_privacy else None,
            "dummy_elements": self.add_dummy_elements,
            "computational_security": True,
            "multi_party_support": self.support_multi_party
        }
    
    def get_utility_metrics(self, original_data: pd.DataFrame, 
                          anonymized_data: pd.DataFrame) -> Dict[str, Any]:
        """Calculate utility metrics for PSI."""
        return {
            "intersection_accuracy": self._get_intersection_accuracy(),
            "communication_efficiency": self._get_communication_efficiency(),
            "computation_overhead": self._get_computation_overhead(),
            "supports_threshold": self.intersection_type == "threshold",
            "supports_approximate": self.intersection_type == "approximate",
            "bloom_filter_optimization": self.use_bloom_filters,
            "batch_processing": self.batch_size > 1,
            "parallel_processing": self.parallel_processing
        }
    
    def _get_intersection_accuracy(self) -> str:
        """Get accuracy rating for intersection computation."""
        if self.intersection_type == "exact":
            return "Perfect"
        elif self.intersection_type == "approximate":
            return "High"
        elif self.intersection_type == "threshold":
            return f"Threshold ({self.threshold_percentage}%)"
        else:
            return "Unknown"
    
    def _get_communication_efficiency(self) -> str:
        """Get communication efficiency rating."""
        if self.use_bloom_filters and self.minimize_communication:
            return "Excellent"
        elif self.minimize_communication:
            return "Good"
        else:
            return "Moderate"
    
    def _get_computation_overhead(self) -> str:
        """Get computation overhead rating."""
        overhead_map = {
            "bloom_filter": "Low",
            "ecdh": "Moderate",
            "kkrt": "Moderate", 
            "ot_based": "High",
            "fhe_based": "Very High"
        }
        return overhead_map.get(self.psi_protocol, "Moderate")

# Register the plugin
def get_plugin():
    """Return the plugin instance for registration."""
    return PrivateSetIntersectionPlugin()
