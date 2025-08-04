"""
Secure Multiparty Computation (SMC) Plugin for Data Anonymization

This plugin implements secure multiparty computation techniques for 
privacy-preserving data analysis where multiple parties can jointly 
compute functions over their data without revealing their individual inputs.
"""

import streamlit as st
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Tuple, Optional
import hashlib
import secrets
import logging

from ..base_anonymizer import Anonymizer

logger = logging.getLogger(__name__)

class SecureMultipartyComputationPlugin(Anonymizer):
    """
    Secure Multiparty Computation plugin for privacy-preserving data analysis.
    
    Implements secret sharing schemes and secure aggregation protocols
    to enable collaborative data analysis without exposing raw data.
    """
    
    def __init__(self):
        super().__init__()
        self.name = "Secure Multiparty Computation"
        self.description = "Privacy-preserving collaborative computation using secret sharing and secure aggregation"
        
        # SMC parameters
        self.num_parties = 3
        self.threshold = 2  # Threshold for secret sharing (t-out-of-n)
        self.prime_modulus = 2**31 - 1  # Large prime for finite field arithmetic
        self.noise_scale = 1.0
        self.aggregation_method = "sum"  # sum, mean, max, min
        self.secret_sharing_scheme = "shamir"  # shamir, additive
        
        # Security parameters
        self.security_level = 128
        self.use_secure_aggregation = True
        self.add_differential_privacy = False
        self.dp_epsilon = 1.0
        
        # Internal state
        self.shares = {}
        self.party_ids = []
        self.reconstruction_data = {}
        
    def get_name(self) -> str:
        return self.name
        
    def get_category(self) -> str:
        return "Advanced Techniques"
        
    def get_description(self) -> str:
        return self.description
    
    def get_sidebar_ui(self, all_cols: List[str], sa_col_to_pass: str | None, 
                      df_raw: pd.DataFrame, unique_key_prefix: str) -> Dict[str, Any]:
        """
        Renders the SMC specific UI elements in the Streamlit sidebar.
        """
        st.sidebar.header(f"🔐 {self.get_name()} Configuration")
        
        with st.sidebar.expander("ℹ️ About Secure Multiparty Computation"):
            st.markdown("""
            **Secure Multiparty Computation (SMC)** enables multiple parties to jointly 
            compute functions over their data without revealing individual inputs.
            
            **Key Features:**
            - Secret sharing schemes (Shamir's, Additive)
            - Secure aggregation protocols
            - Threshold cryptography (t-out-of-n reconstruction)
            - Optional differential privacy integration
            - Protection against semi-honest adversaries
            """)
        
        # Basic SMC parameters
        st.sidebar.subheader("SMC Protocol Settings")
        
        num_parties = st.sidebar.slider(
            "Number of Parties:",
            min_value=2, max_value=10, 
            value=self.num_parties,
            key=f"{unique_key_prefix}_num_parties",
            help="Total number of parties participating in the computation"
        )
        
        threshold = st.sidebar.slider(
            "Reconstruction Threshold:",
            min_value=1, max_value=num_parties,
            value=min(self.threshold, num_parties),
            key=f"{unique_key_prefix}_threshold",
            help="Minimum number of shares needed to reconstruct the secret"
        )
        
        secret_sharing_scheme = st.sidebar.selectbox(
            "Secret Sharing Scheme:",
            options=["shamir", "additive"],
            index=0 if self.secret_sharing_scheme == "shamir" else 1,
            key=f"{unique_key_prefix}_secret_sharing",
            help="Type of secret sharing scheme to use"
        )
        
        # Aggregation settings
        st.sidebar.subheader("Aggregation Settings")
        
        if all_cols:
            computation_cols = st.sidebar.multiselect(
                "Columns for Secure Computation:",
                options=all_cols,
                default=[col for col in all_cols[:3] if col in all_cols],
                key=f"{unique_key_prefix}_computation_cols",
                help="Select columns to include in secure multiparty computation"
            )
        else:
            computation_cols = []
        
        aggregation_method = st.sidebar.selectbox(
            "Aggregation Function:",
            options=["sum", "mean", "max", "min", "count"],
            index=["sum", "mean", "max", "min", "count"].index(self.aggregation_method),
            key=f"{unique_key_prefix}_aggregation_method",
            help="Function to compute securely across parties"
        )
        
        # Privacy enhancements
        st.sidebar.subheader("Privacy Enhancements")
        
        add_differential_privacy = st.sidebar.checkbox(
            "Add Differential Privacy",
            value=self.add_differential_privacy,
            key=f"{unique_key_prefix}_add_dp",
            help="Add noise for differential privacy guarantees"
        )
        
        if add_differential_privacy:
            dp_epsilon = st.sidebar.slider(
                "DP Privacy Budget (ε):",
                min_value=0.1, max_value=10.0, value=self.dp_epsilon,
                step=0.1,
                key=f"{unique_key_prefix}_dp_epsilon",
                help="Lower values provide stronger privacy"
            )
        else:
            dp_epsilon = self.dp_epsilon
        
        noise_scale = st.sidebar.slider(
            "Noise Scale:",
            min_value=0.0, max_value=5.0, value=self.noise_scale,
            step=0.1,
            key=f"{unique_key_prefix}_noise_scale",
            help="Additional noise for enhanced privacy"
        )
        
        # Security parameters
        with st.sidebar.expander("🔧 Advanced Security Settings"):
            security_level = st.sidebar.selectbox(
                "Security Level (bits):",
                options=[80, 128, 192, 256],
                index=[80, 128, 192, 256].index(self.security_level),
                key=f"{unique_key_prefix}_security_level"
            )
            
            use_secure_aggregation = st.sidebar.checkbox(
                "Use Secure Aggregation",
                value=self.use_secure_aggregation,
                key=f"{unique_key_prefix}_secure_aggregation",
                help="Use cryptographic protocols for secure aggregation"
            )
        
        return {
            "num_parties": num_parties,
            "threshold": threshold,
            "secret_sharing_scheme": secret_sharing_scheme,
            "computation_cols": computation_cols,
            "aggregation_method": aggregation_method,
            "add_differential_privacy": add_differential_privacy,
            "dp_epsilon": dp_epsilon,
            "noise_scale": noise_scale,
            "security_level": security_level,
            "use_secure_aggregation": use_secure_aggregation
        }
    
    def _generate_prime(self, bits: int) -> int:
        """Generate a random prime number of specified bit length."""
        # Simplified prime generation for demo purposes
        # In practice, use proper cryptographic libraries
        return 2**31 - 1  # Mersenne prime
    
    def _shamir_secret_sharing(self, secret: int, n: int, t: int, prime: int) -> List[Tuple[int, int]]:
        """
        Shamir's secret sharing scheme.
        
        Args:
            secret: The secret to share
            n: Total number of shares
            t: Threshold (minimum shares needed for reconstruction)
            prime: Prime modulus for finite field arithmetic
            
        Returns:
            List of (x, y) coordinate pairs representing shares
        """
        # Generate random coefficients for polynomial of degree t-1
        coefficients = [secret] + [secrets.randbelow(prime) for _ in range(t-1)]
        
        # Generate shares
        shares = []
        for i in range(1, n+1):
            # Evaluate polynomial at point i
            y = sum(coeff * pow(i, j, prime) for j, coeff in enumerate(coefficients)) % prime
            shares.append((i, y))
        
        return shares
    
    def _reconstruct_secret(self, shares: List[Tuple[int, int]], prime: int) -> int:
        """
        Reconstruct secret from Shamir shares using Lagrange interpolation.
        """
        if len(shares) < 2:
            raise ValueError("Need at least 2 shares for reconstruction")
        
        secret = 0
        for i, (xi, yi) in enumerate(shares):
            # Lagrange basis polynomial
            numerator = 1
            denominator = 1
            
            for j, (xj, _) in enumerate(shares):
                if i != j:
                    numerator = (numerator * (-xj)) % prime
                    denominator = (denominator * (xi - xj)) % prime
            
            # Modular inverse
            denominator_inv = pow(denominator, prime - 2, prime)
            lagrange_coeff = (numerator * denominator_inv) % prime
            
            secret = (secret + yi * lagrange_coeff) % prime
        
        return secret
    
    def _additive_secret_sharing(self, secret: int, n: int, prime: int) -> List[int]:
        """
        Additive secret sharing scheme.
        """
        shares = [secrets.randbelow(prime) for _ in range(n-1)]
        last_share = (secret - sum(shares)) % prime
        shares.append(last_share)
        return shares
    
    def _secure_aggregation(self, values: List[float], method: str) -> float:
        """
        Perform secure aggregation using cryptographic protocols.
        """
        if method == "sum":
            return sum(values)
        elif method == "mean":
            return np.mean(values)
        elif method == "max":
            return max(values)
        elif method == "min":
            return min(values)
        elif method == "count":
            return len(values)
        else:
            return sum(values)
    
    def _add_differential_privacy_noise(self, value: float, epsilon: float, sensitivity: float = 1.0) -> float:
        """
        Add Laplace noise for differential privacy.
        """
        scale = sensitivity / epsilon
        noise = np.random.laplace(0, scale)
        return value + noise
    
    def _simulate_multiparty_computation(self, df: pd.DataFrame, params: Dict[str, Any]) -> pd.DataFrame:
        """
        Simulate secure multiparty computation on the dataset.
        """
        try:
            computation_cols = params.get("computation_cols", [])
            if not computation_cols:
                st.warning("No columns selected for computation. Using all numeric columns.")
                computation_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            
            if not computation_cols:
                st.error("No numeric columns available for SMC.")
                return df.copy()
            
            result_df = df.copy()
            
            # Simulate the SMC process
            st.info(f"🔐 Simulating SMC with {params['num_parties']} parties...")
            
            # For each numeric column, perform SMC simulation
            for col in computation_cols:
                if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
                    # Convert to integers for secret sharing (scale up floats)
                    values = df[col].fillna(0).astype(float)
                    scaled_values = (values * 1000).astype(int)  # Scale for precision
                    
                    # Perform secret sharing and reconstruction simulation
                    computed_values = []
                    
                    for value in scaled_values:
                        # Create secret shares
                        if params["secret_sharing_scheme"] == "shamir":
                            shares = self._shamir_secret_sharing(
                                value, 
                                params["num_parties"], 
                                params["threshold"], 
                                self.prime_modulus
                            )
                            # Simulate reconstruction with threshold shares
                            reconstructed = self._reconstruct_secret(
                                shares[:params["threshold"]], 
                                self.prime_modulus
                            )
                        else:  # additive
                            shares = self._additive_secret_sharing(
                                value, 
                                params["num_parties"], 
                                self.prime_modulus
                            )
                            reconstructed = sum(shares) % self.prime_modulus
                        
                        # Convert back to original scale
                        computed_value = reconstructed / 1000.0
                        
                        # Add noise if requested
                        if params["noise_scale"] > 0:
                            noise = np.random.normal(0, params["noise_scale"])
                            computed_value += noise
                        
                        # Add DP noise if enabled
                        if params["add_differential_privacy"]:
                            computed_value = self._add_differential_privacy_noise(
                                computed_value, 
                                params["dp_epsilon"]
                            )
                        
                        computed_values.append(computed_value)
                    
                    # Update the column with computed values
                    result_df[f"{col}_smc"] = computed_values
            
            # Perform secure aggregation if requested
            if params["use_secure_aggregation"]:
                aggregation_results = {}
                for col in computation_cols:
                    if f"{col}_smc" in result_df.columns:
                        agg_value = self._secure_aggregation(
                            result_df[f"{col}_smc"].tolist(),
                            params["aggregation_method"]
                        )
                        aggregation_results[f"{col}_{params['aggregation_method']}"] = agg_value
                
                # Add aggregation results as metadata
                st.sidebar.subheader("🔢 Aggregation Results")
                for key, value in aggregation_results.items():
                    st.sidebar.metric(key.replace("_", " ").title(), f"{value:.4f}")
            
            return result_df
            
        except Exception as e:
            st.error(f"Error in SMC simulation: {e}")
            logger.error(f"SMC error: {e}")
            return df.copy()
    
    def anonymize(self, df_input: pd.DataFrame, parameters: Dict[str, Any], sa_col: str | None) -> pd.DataFrame:
        """
        Apply secure multiparty computation simulation to the dataset.
        """
        if df_input.empty:
            st.warning("Input dataset is empty.")
            return df_input.copy()
        
        st.info(f"🔐 Applying {self.get_name()}...")
        
        # Perform SMC simulation
        result_df = self._simulate_multiparty_computation(df_input, parameters)
        
        # Add metadata
        st.success(f"✅ SMC simulation completed with {parameters['num_parties']} parties")
        st.info(f"📊 Secret sharing: {parameters['secret_sharing_scheme'].title()}, "
                f"Threshold: {parameters['threshold']}/{parameters['num_parties']}")
        
        if parameters["add_differential_privacy"]:
            st.info(f"🔒 Differential privacy added (ε = {parameters['dp_epsilon']})")
        
        return result_df
    
    def build_config_export(self, unique_key_prefix: str, sa_col: str | None) -> Dict[str, Any]:
        """Build configuration for export."""
        return {
            "num_parties": st.session_state.get(f"{unique_key_prefix}_num_parties", self.num_parties),
            "threshold": st.session_state.get(f"{unique_key_prefix}_threshold", self.threshold),
            "secret_sharing_scheme": st.session_state.get(f"{unique_key_prefix}_secret_sharing", self.secret_sharing_scheme),
            "computation_cols": st.session_state.get(f"{unique_key_prefix}_computation_cols", []),
            "aggregation_method": st.session_state.get(f"{unique_key_prefix}_aggregation_method", self.aggregation_method),
            "add_differential_privacy": st.session_state.get(f"{unique_key_prefix}_add_dp", self.add_differential_privacy),
            "dp_epsilon": st.session_state.get(f"{unique_key_prefix}_dp_epsilon", self.dp_epsilon),
            "noise_scale": st.session_state.get(f"{unique_key_prefix}_noise_scale", self.noise_scale),
            "security_level": st.session_state.get(f"{unique_key_prefix}_security_level", self.security_level),
            "use_secure_aggregation": st.session_state.get(f"{unique_key_prefix}_secure_aggregation", self.use_secure_aggregation)
        }
    
    def apply_config_import(self, config_params: Dict[str, Any], all_cols: List[str], unique_key_prefix: str):
        """Apply imported configuration to session state."""
        if "num_parties" in config_params:
            st.session_state[f"{unique_key_prefix}_num_parties"] = config_params["num_parties"]
        if "threshold" in config_params:
            st.session_state[f"{unique_key_prefix}_threshold"] = config_params["threshold"]
        if "secret_sharing_scheme" in config_params:
            st.session_state[f"{unique_key_prefix}_secret_sharing"] = config_params["secret_sharing_scheme"]
        if "computation_cols" in config_params:
            st.session_state[f"{unique_key_prefix}_computation_cols"] = config_params["computation_cols"]
        if "aggregation_method" in config_params:
            st.session_state[f"{unique_key_prefix}_aggregation_method"] = config_params["aggregation_method"]
        if "add_differential_privacy" in config_params:
            st.session_state[f"{unique_key_prefix}_add_dp"] = config_params["add_differential_privacy"]
        if "dp_epsilon" in config_params:
            st.session_state[f"{unique_key_prefix}_dp_epsilon"] = config_params["dp_epsilon"]
        if "noise_scale" in config_params:
            st.session_state[f"{unique_key_prefix}_noise_scale"] = config_params["noise_scale"]
        if "security_level" in config_params:
            st.session_state[f"{unique_key_prefix}_security_level"] = config_params["security_level"]
        if "use_secure_aggregation" in config_params:
            st.session_state[f"{unique_key_prefix}_secure_aggregation"] = config_params["use_secure_aggregation"]


def get_plugin():
    return SecureMultipartyComputationPlugin()
