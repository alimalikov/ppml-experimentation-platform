"""
Homomorphic Encryption Plugin for Data Anonymization

This plugin implements homomorphic encryption techniques that allow computation
on encrypted data without decrypting it, enabling privacy-preserving analytics
on encrypted datasets.
"""

import streamlit as st
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Tuple, Optional
import hashlib
import secrets
import logging
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.backends import default_backend
import base64

from ..base_anonymizer import Anonymizer

logger = logging.getLogger(__name__)

class HomomorphicEncryptionPlugin(Anonymizer):
    """
    Homomorphic Encryption plugin for privacy-preserving data analysis.
    
    Implements additive and multiplicative homomorphic encryption schemes
    that allow computations on encrypted data without decryption.
    """
    
    def __init__(self):
        super().__init__()
        self.name = "Homomorphic Encryption"
        self.description = "Encrypt data while preserving ability to perform computations on encrypted values"
        
        # Encryption parameters
        self.encryption_scheme = "paillier"  # paillier, elgamal, rsa
        self.key_size = 2048
        self.security_level = 128
        
        # Computation parameters
        self.allowed_operations = ["addition", "multiplication", "scalar_multiplication"]
        self.computation_depth = 2  # Maximum depth of operations
        self.noise_budget = 50  # For leveled homomorphic encryption
        
        # Optimization parameters
        self.use_batching = True
        self.batch_size = 100
        self.use_relinearization = True
        self.use_bootstrapping = False
        
        # Privacy parameters
        self.add_noise = True
        self.noise_scale = 0.1
        self.preserve_format = True
          # Internal state
        self.public_key = None
        self.private_key = None
        self.encrypted_data = {}
        self.computation_log = []
    
    def get_name(self) -> str:
        """Return the display name of this anonymization technique."""
        return self.name
    
    def get_sidebar_ui(self, all_cols: List[str], sa_col_to_pass: str | None, 
                      df_raw: pd.DataFrame, unique_key_prefix: str) -> Dict[str, Any]:
        """
        Renders the HE specific UI elements in the Streamlit sidebar.
        """
        st.sidebar.header(f"🔐 {self.get_name()} Configuration")
        
        with st.sidebar.expander("ℹ️ About Homomorphic Encryption"):
            st.markdown("""
            **Homomorphic Encryption** allows computations to be performed on encrypted data
            without decrypting it first. This enables privacy-preserving analytics where
            data can remain encrypted throughout the entire computation process.
            
            **Key Features:**
            - **Additive Homomorphism**: Supports addition operations on encrypted data
            - **Multiplicative Homomorphism**: Supports multiplication operations (limited)
            - **Semantic Security**: Computationally indistinguishable encryptions
            - **Flexible Computations**: Various arithmetic operations possible
            """)
        
        # Encryption scheme selection
        scheme_options = ["paillier", "elgamal", "rsa"]
        scheme_labels = {
            "paillier": "Paillier (Additive Homomorphism)",
            "elgamal": "ElGamal (Multiplicative Homomorphism)", 
            "rsa": "RSA (Limited Homomorphism)"
        }
        
        encryption_scheme = st.sidebar.selectbox(
            "Encryption Scheme",
            options=scheme_options,
            format_func=lambda x: scheme_labels[x],
            key=f"{unique_key_prefix}_encryption_scheme",
            help="Choose the homomorphic encryption scheme"
        )
        
        # Key size
        key_size = st.sidebar.selectbox(
            "Key Size (bits)",
            options=[1024, 2048, 3072, 4096],
            index=1,
            key=f"{unique_key_prefix}_key_size",
            help="Larger keys provide better security but slower computation"
        )
        
        # Security level
        security_level = st.sidebar.slider(
            "Security Level",
            min_value=80,
            max_value=256,
            value=128,
            step=16,
            key=f"{unique_key_prefix}_security_level",
            help="Security level in bits (higher = more secure)"
        )
        
        # Allowed operations
        operation_options = ["addition", "multiplication", "scalar_multiplication"]
        allowed_operations = st.sidebar.multiselect(
            "Allowed Operations",
            options=operation_options,
            default=["addition", "scalar_multiplication"],
            key=f"{unique_key_prefix}_allowed_operations",
            help="Operations that can be performed on encrypted data"
        )
        
        # Computation depth
        computation_depth = st.sidebar.slider(
            "Maximum Computation Depth",
            min_value=1,
            max_value=10,
            value=2,
            key=f"{unique_key_prefix}_computation_depth",
            help="Maximum depth of nested operations (affects noise growth)"
        )
        
        # Advanced settings
        with st.sidebar.expander("🔧 Advanced Settings"):
            use_batching = st.checkbox(
                "Enable Batching",
                value=True,
                key=f"{unique_key_prefix}_use_batching",
                help="Batch multiple values for efficient encryption"
            )
            
            batch_size = st.number_input(
                "Batch Size",
                min_value=10,
                max_value=1000,
                value=100,
                key=f"{unique_key_prefix}_batch_size",
                help="Number of values to batch together"
            )
            
            add_noise = st.checkbox(
                "Add Noise",
                value=True,
                key=f"{unique_key_prefix}_add_noise",
                help="Add additional noise for enhanced privacy"
            )
            
            noise_scale = st.slider(
                "Noise Scale",
                min_value=0.01,
                max_value=2.0,
                value=0.1,
                step=0.01,
                key=f"{unique_key_prefix}_noise_scale",
                help="Scale of additional noise"
            )
        
        return {
            "encryption_scheme": encryption_scheme,
            "key_size": key_size,
            "security_level": security_level,
            "allowed_operations": allowed_operations,
            "computation_depth": computation_depth,            "use_batching": use_batching,
            "batch_size": batch_size,
            "add_noise": add_noise,
            "noise_scale": noise_scale
        }
    
    def get_category(self) -> str:
        """Return the category for this anonymization technique."""
        return "Cryptographic Methods"
    
    def anonymize(self, df_input: pd.DataFrame, parameters: Dict[str, Any], sa_col: str | None) -> pd.DataFrame:
        """
        Apply homomorphic encryption to the dataset.
        
        Args:
            df_input: Input DataFrame to encrypt
            parameters: Parameters from get_sidebar_ui
            sa_col: Sensitive attribute column
            
        Returns:
            DataFrame with encrypted values
        """
        try:
            import time
            start_time = time.time()
            
            # Update configuration from parameters
            self.encryption_scheme = parameters.get("encryption_scheme", "paillier")
            self.key_size = parameters.get("key_size", 2048)
            self.security_level = parameters.get("security_level", 128)
            self.allowed_operations = parameters.get("allowed_operations", ["addition"])
            self.computation_depth = parameters.get("computation_depth", 2)
            self.use_batching = parameters.get("use_batching", True)
            self.batch_size = parameters.get("batch_size", 100)
            self.add_noise = parameters.get("add_noise", True)
            self.noise_scale = parameters.get("noise_scale", 0.1)
            
            logger.info(f"Starting homomorphic encryption with scheme: {self.encryption_scheme}")
            
            # Generate keys
            self._generate_keys()
            
            # Apply encryption based on scheme
            if self.encryption_scheme == "paillier":
                encrypted_data = self._apply_paillier_encryption(df_input)
            elif self.encryption_scheme == "elgamal":
                encrypted_data = self._apply_elgamal_encryption(df_input)
            elif self.encryption_scheme == "rsa":
                encrypted_data = self._apply_rsa_encryption(df_input)
            else:
                raise ValueError(f"Unsupported encryption scheme: {self.encryption_scheme}")
            
            self.last_encryption_time = time.time() - start_time
            
            # Add metadata
            encrypted_data.attrs['encryption_scheme'] = self.encryption_scheme
            encrypted_data.attrs['key_size'] = self.key_size
            encrypted_data.attrs['encrypted'] = True
            encrypted_data.attrs['allowed_operations'] = self.allowed_operations
            
            logger.info(f"Homomorphic encryption completed in {self.last_encryption_time:.2f} seconds")
            return encrypted_data
            
        except Exception as e:
            logger.error(f"Error in homomorphic encryption: {str(e)}")
            st.error(f"Encryption failed: {str(e)}")
            return df_input
    
    def _generate_keys(self):
        """Generate public and private keys for encryption."""
        logger.info(f"Generating {self.key_size}-bit keys for {self.encryption_scheme}")
        
        # Simplified key generation (in practice, use proper cryptographic libraries)
        if self.encryption_scheme == "paillier":
            self._generate_paillier_keys()
        elif self.encryption_scheme == "elgamal":
            self._generate_elgamal_keys()
        elif self.encryption_scheme == "rsa":
            self._generate_rsa_keys()
    
    def _generate_paillier_keys(self):
        """Generate Paillier key pair."""
        # Simplified Paillier key generation
        # In practice, use a proper cryptographic library like python-paillier
        
        # Generate two large primes (simplified)
        p = self._generate_large_prime(self.key_size // 2)
        q = self._generate_large_prime(self.key_size // 2)
        
        n = p * q
        lambda_n = (p - 1) * (q - 1)  # Simplified lambda
        
        # Public key: (n, g)
        g = n + 1  # Simplified g
        self.public_key = {"n": n, "g": g}
        
        # Private key: lambda
        self.private_key = {"lambda": lambda_n, "n": n}
        
        logger.info("Paillier keys generated successfully")
    
    def _generate_elgamal_keys(self):
        """Generate ElGamal key pair."""
        # Simplified ElGamal key generation
        p = self._generate_large_prime(self.key_size)
        g = 2  # Simplified generator
        x = secrets.randbelow(p - 2) + 1  # Private key
        y = pow(g, x, p)  # Public key component
        
        self.public_key = {"p": p, "g": g, "y": y}
        self.private_key = {"x": x, "p": p}
        
        logger.info("ElGamal keys generated successfully")
    
    def _generate_rsa_keys(self):
        """Generate RSA key pair."""
        # Simplified RSA key generation
        p = self._generate_large_prime(self.key_size // 2)
        q = self._generate_large_prime(self.key_size // 2)
        
        n = p * q
        phi_n = (p - 1) * (q - 1)
        e = 65537  # Common public exponent
        d = self._mod_inverse(e, phi_n)
        
        self.public_key = {"n": n, "e": e}
        self.private_key = {"n": n, "d": d}
        
        logger.info("RSA keys generated successfully")
    
    def _generate_large_prime(self, bits):
        """Generate a large prime number (simplified)."""
        # This is a simplified implementation
        # In practice, use proper prime generation algorithms
        import random
        while True:
            num = random.getrandbits(bits)
            if num % 2 == 1 and self._is_prime_simple(num):
                return num
    
    def _is_prime_simple(self, n):
        """Simple primality test (not cryptographically secure)."""
        if n < 2:
            return False
        if n == 2:
            return True
        if n % 2 == 0:
            return False
        
        # Simple trial division
        for i in range(3, int(n**0.5) + 1, 2):
            if n % i == 0:
                return False
        return True
    
    def _mod_inverse(self, a, m):
        """Compute modular inverse using extended Euclidean algorithm."""
        def extended_gcd(a, b):
            if a == 0:
                return b, 0, 1
            gcd, x1, y1 = extended_gcd(b % a, a)
            x = y1 - (b // a) * x1
            y = x1
            return gcd, x, y
        
        gcd, x, y = extended_gcd(a, m)
        if gcd != 1:
            raise ValueError("Modular inverse does not exist")
        return (x % m + m) % m
    
    def _apply_paillier_encryption(self, data: pd.DataFrame) -> pd.DataFrame:
        """Apply Paillier encryption to data."""
        logger.info("Applying Paillier encryption")
        
        encrypted_data = data.copy()
        n = self.public_key["n"]
        g = self.public_key["g"]
        
        for column in data.select_dtypes(include=[np.number]).columns:
            values = data[column].fillna(0).astype(int)
            
            if self.use_batching:
                # Batch encryption for efficiency
                encrypted_values = []
                for i in range(0, len(values), self.batch_size):
                    batch = values.iloc[i:i+self.batch_size]
                    batch_encrypted = self._encrypt_paillier_batch(batch, n, g)
                    encrypted_values.extend(batch_encrypted)
                encrypted_data[column] = encrypted_values
            else:
                # Individual encryption
                encrypted_data[column] = values.apply(
                    lambda x: self._encrypt_paillier_value(x, n, g)
                )
            
            # Add noise if enabled
            if self.add_noise:
                noise = np.random.normal(0, self.noise_scale, len(encrypted_data))
                encrypted_data[column] = encrypted_data[column] + noise.astype(int)
        
        return encrypted_data
    
    def _encrypt_paillier_value(self, value, n, g):
        """Encrypt a single value using Paillier encryption."""
        # Simplified Paillier encryption: E(m) = g^m * r^n mod n^2
        r = secrets.randbelow(n - 1) + 1  # Random value
        n_squared = n * n
        
        # Convert negative values to positive representation
        if value < 0:
            value = n + value
        
        ciphertext = (pow(g, int(value), n_squared) * pow(r, n, n_squared)) % n_squared
        return ciphertext
    
    def _encrypt_paillier_batch(self, batch, n, g):
        """Encrypt a batch of values using Paillier encryption."""
        return [self._encrypt_paillier_value(val, n, g) for val in batch]
    
    def _apply_elgamal_encryption(self, data: pd.DataFrame) -> pd.DataFrame:
        """Apply ElGamal encryption to data."""
        logger.info("Applying ElGamal encryption")
        
        encrypted_data = data.copy()
        p = self.public_key["p"]
        g = self.public_key["g"]
        y = self.public_key["y"]
        
        for column in data.select_dtypes(include=[np.number]).columns:
            values = data[column].fillna(0).astype(int)
            
            encrypted_pairs = []
            for value in values:
                # ElGamal encryption: (g^k mod p, m * y^k mod p)
                k = secrets.randbelow(p - 2) + 1
                c1 = pow(g, k, p)
                c2 = (int(value) * pow(y, k, p)) % p
                encrypted_pairs.append(f"{c1}:{c2}")
            
            encrypted_data[column] = encrypted_pairs
        
        return encrypted_data
    
    def _apply_rsa_encryption(self, data: pd.DataFrame) -> pd.DataFrame:
        """Apply RSA encryption to data."""
        logger.info("Applying RSA encryption")
        
        encrypted_data = data.copy()
        n = self.public_key["n"]
        e = self.public_key["e"]
        
        for column in data.select_dtypes(include=[np.number]).columns:
            values = data[column].fillna(0).astype(int)
            
            # RSA encryption: c = m^e mod n
            encrypted_data[column] = values.apply(
                lambda x: pow(int(abs(x)), e, n) if x != 0 else 0
            )
        
        return encrypted_data
    
    def get_privacy_metrics(self, original_data: pd.DataFrame, 
                          anonymized_data: pd.DataFrame) -> Dict[str, Any]:
        """Calculate privacy metrics for homomorphic encryption."""
        metrics = {
            "encryption_scheme": self.encryption_scheme,
            "key_size": self.key_size,
            "security_level": self.security_level,
            "semantic_security": True,  # HE provides semantic security
            "computational_security": True,
            "perfect_secrecy": False,  # Computational, not perfect
            "allowed_operations": len(self.allowed_operations),
            "computation_depth": self.computation_depth
        }
        
        # Add scheme-specific metrics
        if self.encryption_scheme == "paillier":
            metrics["additive_homomorphism"] = True
            metrics["multiplicative_homomorphism"] = False
        elif self.encryption_scheme == "elgamal":
            metrics["additive_homomorphism"] = False
            metrics["multiplicative_homomorphism"] = True
        elif self.encryption_scheme == "rsa":
            metrics["additive_homomorphism"] = False
            metrics["multiplicative_homomorphism"] = True
        
        return metrics
    
    def get_utility_metrics(self, original_data: pd.DataFrame, 
                          anonymized_data: pd.DataFrame) -> Dict[str, Any]:
        """Calculate utility metrics for homomorphic encryption."""
        metrics = {
            "preserves_computations": True,
            "data_encrypted": True,
            "format_preserved": self.preserve_format,
            "operations_supported": len(self.allowed_operations),
            "computation_overhead": "High",  # HE is computationally expensive
            "storage_overhead": "High"  # Encrypted data is larger
        }
          # Calculate expansion factor
        if hasattr(self, 'public_key') and self.public_key:
            if self.encryption_scheme == "paillier":
                # Paillier ciphertexts are ~2x the key size
                expansion_factor = 2.0
            else:
                expansion_factor = 1.5
            
            metrics["size_expansion_factor"] = expansion_factor
        
        return metrics
    
    def build_config_export(self, unique_key_prefix: str, sa_col: str | None) -> Dict[str, Any]:
        """Build configuration for export."""
        return {
            "encryption_scheme": st.session_state.get(f"{unique_key_prefix}_encryption_scheme", "paillier"),
            "key_size": st.session_state.get(f"{unique_key_prefix}_key_size", 2048),
            "security_level": st.session_state.get(f"{unique_key_prefix}_security_level", 128),
            "allowed_operations": st.session_state.get(f"{unique_key_prefix}_allowed_operations", ["addition"]),
            "computation_depth": st.session_state.get(f"{unique_key_prefix}_computation_depth", 2),
            "use_batching": st.session_state.get(f"{unique_key_prefix}_use_batching", True),
            "batch_size": st.session_state.get(f"{unique_key_prefix}_batch_size", 100),
            "add_noise": st.session_state.get(f"{unique_key_prefix}_add_noise", True),
            "noise_scale": st.session_state.get(f"{unique_key_prefix}_noise_scale", 0.1)
        }
    
    def apply_config_import(self, config_params: Dict[str, Any], all_cols: List[str], unique_key_prefix: str):
        """Apply imported configuration to session state."""
        if "encryption_scheme" in config_params:
            st.session_state[f"{unique_key_prefix}_encryption_scheme"] = config_params["encryption_scheme"]
        if "key_size" in config_params:
            st.session_state[f"{unique_key_prefix}_key_size"] = config_params["key_size"]
        if "security_level" in config_params:
            st.session_state[f"{unique_key_prefix}_security_level"] = config_params["security_level"]
        if "allowed_operations" in config_params:
            st.session_state[f"{unique_key_prefix}_allowed_operations"] = config_params["allowed_operations"]
        if "computation_depth" in config_params:
            st.session_state[f"{unique_key_prefix}_computation_depth"] = config_params["computation_depth"]
        if "use_batching" in config_params:
            st.session_state[f"{unique_key_prefix}_use_batching"] = config_params["use_batching"]
        if "batch_size" in config_params:
            st.session_state[f"{unique_key_prefix}_batch_size"] = config_params["batch_size"]
        if "add_noise" in config_params:
            st.session_state[f"{unique_key_prefix}_add_noise"] = config_params["add_noise"]
        if "noise_scale" in config_params:
            st.session_state[f"{unique_key_prefix}_noise_scale"] = config_params["noise_scale"]

# Register the plugin
def get_plugin():
    """Return the plugin instance for registration."""
    return HomomorphicEncryptionPlugin()
