"""
Additive Noise Plugin for Data Anonymization

This plugin implements basic additive noise perturbation mechanisms for data anonymization.
Supports various noise distributions and provides configurable noise parameters.
"""

import streamlit as st
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional
import json
import logging
from ..base_anonymizer import Anonymizer

logger = logging.getLogger(__name__)

class AdditiveNoisePlugin(Anonymizer):
    """
    Additive noise perturbation plugin.
    
    Adds noise directly to numerical data using various probability distributions
    to obscure original values while preserving statistical properties.
    """

    def __init__(self):
        """Initialize the additive noise plugin."""
        self._name = "Additive Noise"
        self._description = ("Basic additive noise perturbation for numerical data anonymization. "
                           "Supports multiple noise distributions including Gaussian, Laplace, and Uniform. "
                           "Provides configurable noise scale and clipping options.")

    def get_name(self) -> str:
        """Returns the display name of the anonymization technique."""
        return self._name

    def get_category(self) -> str:
        """Returns the category of the anonymization technique."""
        return "Perturbation Methods"

    def get_description(self) -> str:
        """Returns detailed description of the technique."""
        return self._description

    def get_sidebar_ui(self, all_cols: List[str], sa_col_to_pass: str | None, 
                      df_raw: pd.DataFrame, unique_key_prefix: str) -> Dict[str, Any]:
        """
        Renders the additive noise specific UI elements in the Streamlit sidebar.
        """
        st.sidebar.header(f"🔊 {self.get_name()} Configuration")
        
        # Show description
        with st.sidebar.expander("ℹ️ About Additive Noise"):
            st.markdown(self._description)
            st.markdown("""
            **Noise Distributions:**
            - **Gaussian**: Normal distribution noise (most common)
            - **Laplace**: Double exponential distribution
            - **Uniform**: Uniform random noise in specified range
            
            **Use Cases:**
            - Basic data obfuscation
            - Statistical analysis preservation
            - Simple privacy protection
            
            **Best for:** Numerical data where exact values are less critical than statistical trends
            """)

        # Define session state keys
        cols_key = f"{unique_key_prefix}_selected_cols"
        noise_type_key = f"{unique_key_prefix}_noise_type"
        noise_scale_key = f"{unique_key_prefix}_noise_scale"
        relative_noise_key = f"{unique_key_prefix}_relative_noise"
        clip_values_key = f"{unique_key_prefix}_clip_values"
        clip_min_key = f"{unique_key_prefix}_clip_min"
        clip_max_key = f"{unique_key_prefix}_clip_max"
        random_seed_key = f"{unique_key_prefix}_random_seed"

        # Column Selection
        st.sidebar.subheader("📊 Column Selection")
        numeric_cols = df_raw.select_dtypes(include=[np.number]).columns.tolist() if df_raw is not None else []
        
        if not numeric_cols:
            st.sidebar.warning("No numerical columns found in the dataset.")
            selected_cols = []
        else:
            default_selected = st.session_state.get(cols_key, numeric_cols[:3])  # Default to first 3
            valid_default = [col for col in default_selected if col in numeric_cols]
            
            selected_cols = st.sidebar.multiselect(
                "Select columns to add noise to:",
                options=numeric_cols,
                default=valid_default,
                key=cols_key,
                help="Choose numerical columns for noise addition"
            )

        # Noise Configuration
        st.sidebar.subheader("🎛️ Noise Configuration")
        
        # Noise type selection
        noise_types = {
            'gaussian': 'Gaussian (Normal)',
            'laplace': 'Laplace (Double Exponential)', 
            'uniform': 'Uniform Random'
        }
        
        noise_type = st.sidebar.selectbox(
            "Noise Distribution:",
            options=list(noise_types.keys()),
            format_func=lambda x: noise_types[x],
            key=noise_type_key,
            help="Type of probability distribution for noise generation"
        )

        # Noise scale/magnitude
        if df_raw is not None and selected_cols:
            # Calculate data statistics for guidance
            data_stats = {}
            for col in selected_cols:
                if col in df_raw.columns:
                    col_data = df_raw[col].dropna()
                    data_stats[col] = {
                        'mean': col_data.mean(),
                        'std': col_data.std(),
                        'range': col_data.max() - col_data.min()
                    }
            
            # Show data statistics
            with st.sidebar.expander("📈 Data Statistics (for guidance)"):
                for col, stats in data_stats.items():
                    st.write(f"**{col}:**")
                    st.write(f"  - Mean: {stats['mean']:.2f}")
                    st.write(f"  - Std Dev: {stats['std']:.2f}")
                    st.write(f"  - Range: {stats['range']:.2f}")

        # Relative vs absolute noise
        relative_noise = st.sidebar.checkbox(
            "Use Relative Noise",
            value=st.session_state.get(relative_noise_key, True),
            key=relative_noise_key,
            help="Scale noise relative to data standard deviation (recommended)"
        )

        if relative_noise:
            noise_scale = st.sidebar.slider(
                "Relative Noise Scale (× std dev):",
                min_value=0.01,
                max_value=2.0,
                value=st.session_state.get(noise_scale_key, 0.1),
                step=0.01,
                key=noise_scale_key,
                help="Noise magnitude as multiple of standard deviation"
            )
            st.sidebar.caption(f"Noise scale: {noise_scale}× standard deviation")
        else:
            noise_scale = st.sidebar.number_input(
                "Absolute Noise Scale:",
                min_value=0.001,
                max_value=1000.0,
                value=st.session_state.get(noise_scale_key, 1.0),
                step=0.1,
                key=noise_scale_key,
                help="Fixed noise magnitude"
            )

        # Value clipping options
        st.sidebar.subheader("✂️ Output Clipping")
        clip_values = st.sidebar.checkbox(
            "Clip Output Values",
            value=st.session_state.get(clip_values_key, False),
            key=clip_values_key,
            help="Limit output values to specified range"
        )

        if clip_values:
            col1, col2 = st.sidebar.columns(2)
            with col1:
                clip_min = st.number_input(
                    "Min Value:",
                    value=st.session_state.get(clip_min_key, 0.0),
                    key=clip_min_key
                )
            with col2:
                clip_max = st.number_input(
                    "Max Value:",
                    value=st.session_state.get(clip_max_key, 100.0),
                    key=clip_max_key
                )
        else:
            clip_min = None
            clip_max = None

        # Advanced options
        st.sidebar.subheader("⚙️ Advanced Options")
        random_seed = st.sidebar.number_input(
            "Random Seed:",
            min_value=0,
            max_value=9999,
            value=st.session_state.get(random_seed_key, 42),
            key=random_seed_key,
            help="Seed for reproducible noise generation"
        )

        return {
            'selected_cols': selected_cols,
            'noise_type': noise_type,
            'noise_scale': noise_scale,
            'relative_noise': relative_noise,
            'clip_values': clip_values,
            'clip_min': clip_min,
            'clip_max': clip_max,
            'random_seed': random_seed
        }

    def anonymize(self, df_input: pd.DataFrame, parameters: Dict[str, Any], sa_col: str | None) -> pd.DataFrame:
        """
        Performs additive noise anonymization on the specified columns.
        """
        try:
            # Extract parameters
            selected_cols = parameters.get('selected_cols', [])
            noise_type = parameters.get('noise_type', 'gaussian')
            noise_scale = parameters.get('noise_scale', 0.1)
            relative_noise = parameters.get('relative_noise', True)
            clip_values = parameters.get('clip_values', False)
            clip_min = parameters.get('clip_min', None)
            clip_max = parameters.get('clip_max', None)
            random_seed = parameters.get('random_seed', 42)

            if not selected_cols:
                st.warning("No columns selected for noise addition. Returning original data.")
                return df_input.copy()

            # Set random seed for reproducibility
            np.random.seed(random_seed)
            
            # Create copy of input DataFrame
            df_result = df_input.copy()
            
            for col in selected_cols:
                if col not in df_input.columns:
                    st.warning(f"Column '{col}' not found in dataset. Skipping.")
                    continue
                
                if not pd.api.types.is_numeric_dtype(df_input[col]):
                    st.warning(f"Column '{col}' is not numerical. Skipping.")
                    continue

                # Get column data
                col_data = df_result[col].copy()
                non_null_mask = col_data.notna()
                
                if not non_null_mask.any():
                    st.warning(f"Column '{col}' has no non-null values. Skipping.")
                    continue

                # Calculate noise scale
                if relative_noise:
                    data_std = col_data[non_null_mask].std()
                    if data_std == 0:
                        st.warning(f"Column '{col}' has zero standard deviation. Using absolute noise.")
                        actual_noise_scale = noise_scale
                    else:
                        actual_noise_scale = noise_scale * data_std
                else:
                    actual_noise_scale = noise_scale

                # Generate noise based on type
                n_values = non_null_mask.sum()
                
                if noise_type == 'gaussian':
                    noise = np.random.normal(0, actual_noise_scale, n_values)
                elif noise_type == 'laplace':
                    # For Laplace, scale parameter = std_dev / sqrt(2)
                    laplace_scale = actual_noise_scale / np.sqrt(2)
                    noise = np.random.laplace(0, laplace_scale, n_values)
                elif noise_type == 'uniform':
                    # Uniform noise in [-scale, scale] range
                    noise = np.random.uniform(-actual_noise_scale, actual_noise_scale, n_values)
                else:
                    st.error(f"Unknown noise type: {noise_type}")
                    continue

                # Add noise to non-null values
                noisy_values = col_data[non_null_mask] + noise

                # Apply clipping if specified
                if clip_values and clip_min is not None and clip_max is not None:
                    noisy_values = np.clip(noisy_values, clip_min, clip_max)

                # Update column
                df_result.loc[non_null_mask, col] = noisy_values

            st.success(f"Additive {noise_type} noise applied to {len(selected_cols)} columns.")
            
            return df_result

        except Exception as e:
            st.error(f"Error in additive noise anonymization: {str(e)}")
            logger.error(f"Additive noise error: {str(e)}")
            return df_input.copy()

    def build_config_export(self, unique_key_prefix: str, sa_col: str | None) -> Dict[str, Any]:
        """Build configuration for export."""
        return {
            'selected_cols': st.session_state.get(f"{unique_key_prefix}_selected_cols", []),
            'noise_type': st.session_state.get(f"{unique_key_prefix}_noise_type", 'gaussian'),
            'noise_scale': st.session_state.get(f"{unique_key_prefix}_noise_scale", 0.1),
            'relative_noise': st.session_state.get(f"{unique_key_prefix}_relative_noise", True),
            'clip_values': st.session_state.get(f"{unique_key_prefix}_clip_values", False),
            'clip_min': st.session_state.get(f"{unique_key_prefix}_clip_min", None),
            'clip_max': st.session_state.get(f"{unique_key_prefix}_clip_max", None),
            'random_seed': st.session_state.get(f"{unique_key_prefix}_random_seed", 42)
        }

    def apply_config_import(self, config_params: Dict[str, Any], all_cols: List[str], unique_key_prefix: str):
        """Apply imported configuration."""
        # Validate columns exist
        selected_cols = config_params.get('selected_cols', [])
        valid_cols = [col for col in selected_cols if col in all_cols]
        
        # Set session state
        st.session_state[f"{unique_key_prefix}_selected_cols"] = valid_cols
        st.session_state[f"{unique_key_prefix}_noise_type"] = config_params.get('noise_type', 'gaussian')
        st.session_state[f"{unique_key_prefix}_noise_scale"] = config_params.get('noise_scale', 0.1)
        st.session_state[f"{unique_key_prefix}_relative_noise"] = config_params.get('relative_noise', True)
        st.session_state[f"{unique_key_prefix}_clip_values"] = config_params.get('clip_values', False)
        st.session_state[f"{unique_key_prefix}_clip_min"] = config_params.get('clip_min', None)
        st.session_state[f"{unique_key_prefix}_clip_max"] = config_params.get('clip_max', None)
        st.session_state[f"{unique_key_prefix}_random_seed"] = config_params.get('random_seed', 42)

def get_plugin():
    """Factory function to get plugin instance."""
    return AdditiveNoisePlugin()
