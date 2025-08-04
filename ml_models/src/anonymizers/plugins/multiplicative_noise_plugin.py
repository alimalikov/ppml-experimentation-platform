"""
Multiplicative Noise Plugin for Data Anonymization

This plugin implements multiplicative noise perturbation mechanisms for data anonymization.
Multiplies data by random noise factors to preserve relative relationships while adding privacy.
"""

import streamlit as st
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional
import json
import logging
from ..base_anonymizer import Anonymizer

logger = logging.getLogger(__name__)

class MultiplicativeNoisePlugin(Anonymizer):
    """
    Multiplicative noise perturbation plugin.
    
    Multiplies numerical data by noise factors drawn from various probability distributions
    to preserve relative relationships while adding uncertainty to exact values.
    """

    def __init__(self):
        """Initialize the multiplicative noise plugin."""
        self._name = "Multiplicative Noise"
        self._description = ("Multiplicative noise perturbation for numerical data anonymization. "
                           "Multiplies values by random noise factors to preserve relative relationships "
                           "while obscuring exact values. Supports log-normal, gamma, and uniform distributions.")

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
        Renders the multiplicative noise specific UI elements in the Streamlit sidebar.
        """
        st.sidebar.header(f"✖️ {self.get_name()} Configuration")
        
        # Show description
        with st.sidebar.expander("ℹ️ About Multiplicative Noise"):
            st.markdown(self._description)
            st.markdown("""
            **Noise Distributions:**
            - **Log-Normal**: Preserves positive values, commonly used
            - **Gamma**: Flexible shape, always positive
            - **Uniform**: Simple uniform multiplier in specified range
            
            **Advantages:**
            - Preserves relative relationships (ratios)
            - Maintains data distribution shape
            - Good for percentage-based analysis
            
            **Best for:** Financial data, counts, measurements where relative values matter
            
            **Note:** Only works with positive numerical data
            """)

        # Define session state keys
        cols_key = f"{unique_key_prefix}_selected_cols"
        noise_type_key = f"{unique_key_prefix}_noise_type"
        noise_factor_key = f"{unique_key_prefix}_noise_factor"
        preserve_zero_key = f"{unique_key_prefix}_preserve_zero"
        handle_negative_key = f"{unique_key_prefix}_handle_negative"
        clip_multiplier_key = f"{unique_key_prefix}_clip_multiplier"
        min_multiplier_key = f"{unique_key_prefix}_min_multiplier"
        max_multiplier_key = f"{unique_key_prefix}_max_multiplier"
        random_seed_key = f"{unique_key_prefix}_random_seed"

        # Column Selection
        st.sidebar.subheader("📊 Column Selection")
        numeric_cols = df_raw.select_dtypes(include=[np.number]).columns.tolist() if df_raw is not None else []
        
        if not numeric_cols:
            st.sidebar.warning("No numerical columns found in the dataset.")
            selected_cols = []
        else:
            default_selected = st.session_state.get(cols_key, numeric_cols[:3])
            valid_default = [col for col in default_selected if col in numeric_cols]
            
            # Show data characteristics
            if df_raw is not None:
                with st.sidebar.expander("📈 Data Characteristics"):
                    for col in numeric_cols[:5]:  # Show first 5 columns
                        col_data = df_raw[col].dropna()
                        min_val = col_data.min()
                        max_val = col_data.max()
                        has_zero = (col_data == 0).any()
                        has_negative = (col_data < 0).any()
                        
                        st.write(f"**{col}:**")
                        if has_negative:
                            st.write(f"  ⚠️ Has negative values ({min_val:.2f} to {max_val:.2f})")
                        elif has_zero:
                            st.write(f"  ⚠️ Has zeros (0 to {max_val:.2f})")
                        else:
                            st.write(f"  ✅ All positive ({min_val:.2f} to {max_val:.2f})")
            
            selected_cols = st.sidebar.multiselect(
                "Select columns for multiplicative noise:",
                options=numeric_cols,
                default=valid_default,
                key=cols_key,
                help="Choose numerical columns for multiplicative noise"
            )

        # Noise Configuration
        st.sidebar.subheader("🎛️ Noise Configuration")
        
        # Noise type selection
        noise_types = {
            'lognormal': 'Log-Normal (recommended)',
            'gamma': 'Gamma Distribution',
            'uniform': 'Uniform Range'
        }
        
        noise_type = st.sidebar.selectbox(
            "Noise Distribution:",
            options=list(noise_types.keys()),
            format_func=lambda x: noise_types[x],
            key=noise_type_key,
            help="Type of distribution for noise multipliers"
        )

        # Noise factor configuration
        if noise_type == 'lognormal':
            noise_factor = st.sidebar.slider(
                "Log-Normal Scale (σ):",
                min_value=0.01,
                max_value=1.0,
                value=st.session_state.get(noise_factor_key, 0.1),
                step=0.01,
                key=noise_factor_key,
                help="Standard deviation of log-normal distribution (higher = more noise)"
            )
            st.sidebar.caption(f"Mean multiplier ≈ 1.0, typical range: {np.exp(-2*noise_factor):.2f} - {np.exp(2*noise_factor):.2f}")
            
        elif noise_type == 'gamma':
            noise_factor = st.sidebar.slider(
                "Gamma Shape Parameter:",
                min_value=0.5,
                max_value=10.0,
                value=st.session_state.get(noise_factor_key, 2.0),
                step=0.1,
                key=noise_factor_key,
                help="Shape parameter for gamma distribution (higher = less variance)"
            )
            # Scale parameter is set to maintain mean ≈ 1
            scale = 1.0 / noise_factor
            st.sidebar.caption(f"Mean multiplier ≈ 1.0, variance ≈ {scale:.3f}")
            
        elif noise_type == 'uniform':
            noise_factor = st.sidebar.slider(
                "Uniform Range (±):",
                min_value=0.01,
                max_value=0.5,
                value=st.session_state.get(noise_factor_key, 0.1),
                step=0.01,
                key=noise_factor_key,
                help="Uniform multiplier range around 1.0"
            )
            st.sidebar.caption(f"Multiplier range: {1-noise_factor:.2f} - {1+noise_factor:.2f}")

        # Special value handling
        st.sidebar.subheader("🔧 Special Value Handling")
        
        preserve_zero = st.sidebar.checkbox(
            "Preserve Zero Values",
            value=st.session_state.get(preserve_zero_key, True),
            key=preserve_zero_key,
            help="Keep zero values unchanged (recommended)"
        )

        handle_negative_options = {
            'absolute': 'Take Absolute Value',
            'skip': 'Skip Negative Values',
            'shift': 'Shift to Positive Range'
        }
        
        handle_negative = st.sidebar.selectbox(
            "Handle Negative Values:",
            options=list(handle_negative_options.keys()),
            format_func=lambda x: handle_negative_options[x],
            key=handle_negative_key,
            help="How to handle negative values"
        )

        # Multiplier clipping
        st.sidebar.subheader("✂️ Multiplier Constraints")
        clip_multiplier = st.sidebar.checkbox(
            "Clip Multiplier Range",
            value=st.session_state.get(clip_multiplier_key, True),
            key=clip_multiplier_key,
            help="Limit multiplier values to reasonable range"
        )

        if clip_multiplier:
            col1, col2 = st.sidebar.columns(2)
            with col1:
                min_multiplier = st.number_input(
                    "Min Multiplier:",
                    min_value=0.01,
                    max_value=0.99,
                    value=st.session_state.get(min_multiplier_key, 0.5),
                    step=0.01,
                    key=min_multiplier_key
                )
            with col2:
                max_multiplier = st.number_input(
                    "Max Multiplier:",
                    min_value=1.01,
                    max_value=10.0,
                    value=st.session_state.get(max_multiplier_key, 2.0),
                    step=0.01,
                    key=max_multiplier_key
                )
        else:
            min_multiplier = None
            max_multiplier = None

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
            'noise_factor': noise_factor,
            'preserve_zero': preserve_zero,
            'handle_negative': handle_negative,
            'clip_multiplier': clip_multiplier,
            'min_multiplier': min_multiplier,
            'max_multiplier': max_multiplier,
            'random_seed': random_seed
        }

    def _generate_multipliers(self, noise_type: str, noise_factor: float, n_values: int) -> np.ndarray:
        """Generate multiplicative noise factors."""
        if noise_type == 'lognormal':
            # Log-normal with mean ≈ 1
            multipliers = np.random.lognormal(mean=0, sigma=noise_factor, size=n_values)
            
        elif noise_type == 'gamma':
            # Gamma with mean ≈ 1
            shape = noise_factor
            scale = 1.0 / shape  # This makes mean = shape * scale = 1
            multipliers = np.random.gamma(shape, scale, size=n_values)
            
        elif noise_type == 'uniform':
            # Uniform around 1
            low = 1 - noise_factor
            high = 1 + noise_factor
            multipliers = np.random.uniform(low, high, size=n_values)
            
        else:
            # Fallback to uniform
            multipliers = np.random.uniform(0.9, 1.1, size=n_values)
            
        return multipliers

    def anonymize(self, df_input: pd.DataFrame, parameters: Dict[str, Any], sa_col: str | None) -> pd.DataFrame:
        """
        Performs multiplicative noise anonymization on the specified columns.
        """
        try:
            # Extract parameters
            selected_cols = parameters.get('selected_cols', [])
            noise_type = parameters.get('noise_type', 'lognormal')
            noise_factor = parameters.get('noise_factor', 0.1)
            preserve_zero = parameters.get('preserve_zero', True)
            handle_negative = parameters.get('handle_negative', 'absolute')
            clip_multiplier = parameters.get('clip_multiplier', True)
            min_multiplier = parameters.get('min_multiplier', 0.5)
            max_multiplier = parameters.get('max_multiplier', 2.0)
            random_seed = parameters.get('random_seed', 42)

            if not selected_cols:
                st.warning("No columns selected for multiplicative noise. Returning original data.")
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

                # Handle zeros
                if preserve_zero:
                    zero_mask = (col_data == 0) & non_null_mask
                    process_mask = non_null_mask & ~zero_mask
                else:
                    process_mask = non_null_mask

                if not process_mask.any():
                    continue

                # Handle negative values
                process_data = col_data[process_mask].copy()
                negative_mask = process_data < 0
                
                if negative_mask.any():
                    if handle_negative == 'absolute':
                        process_data = np.abs(process_data)
                    elif handle_negative == 'skip':
                        # Only process positive values
                        positive_mask = process_mask & (col_data > 0)
                        process_mask = positive_mask
                        process_data = col_data[process_mask].copy()
                    elif handle_negative == 'shift':
                        # Shift to positive range
                        min_val = process_data.min()
                        if min_val < 0:
                            shift = abs(min_val) + 1
                            process_data += shift

                if len(process_data) == 0:
                    continue

                # Generate multipliers
                multipliers = self._generate_multipliers(noise_type, noise_factor, len(process_data))

                # Apply clipping if specified
                if clip_multiplier and min_multiplier is not None and max_multiplier is not None:
                    multipliers = np.clip(multipliers, min_multiplier, max_multiplier)

                # Apply multiplicative noise
                noisy_values = process_data * multipliers

                # Handle shifting back if needed
                if handle_negative == 'shift':
                    min_val = col_data[process_mask].min()
                    if min_val < 0:
                        shift = abs(min_val) + 1
                        noisy_values -= shift

                # Update column
                df_result.loc[process_mask, col] = noisy_values

            st.success(f"Multiplicative {noise_type} noise applied to {len(selected_cols)} columns.")
            
            return df_result

        except Exception as e:
            st.error(f"Error in multiplicative noise anonymization: {str(e)}")
            logger.error(f"Multiplicative noise error: {str(e)}")
            return df_input.copy()

    def build_config_export(self, unique_key_prefix: str, sa_col: str | None) -> Dict[str, Any]:
        """Build configuration for export."""
        return {
            'selected_cols': st.session_state.get(f"{unique_key_prefix}_selected_cols", []),
            'noise_type': st.session_state.get(f"{unique_key_prefix}_noise_type", 'lognormal'),
            'noise_factor': st.session_state.get(f"{unique_key_prefix}_noise_factor", 0.1),
            'preserve_zero': st.session_state.get(f"{unique_key_prefix}_preserve_zero", True),
            'handle_negative': st.session_state.get(f"{unique_key_prefix}_handle_negative", 'absolute'),
            'clip_multiplier': st.session_state.get(f"{unique_key_prefix}_clip_multiplier", True),
            'min_multiplier': st.session_state.get(f"{unique_key_prefix}_min_multiplier", 0.5),
            'max_multiplier': st.session_state.get(f"{unique_key_prefix}_max_multiplier", 2.0),
            'random_seed': st.session_state.get(f"{unique_key_prefix}_random_seed", 42)
        }

    def apply_config_import(self, config_params: Dict[str, Any], all_cols: List[str], unique_key_prefix: str):
        """Apply imported configuration."""
        # Validate columns exist
        selected_cols = config_params.get('selected_cols', [])
        valid_cols = [col for col in selected_cols if col in all_cols]
        
        # Set session state
        st.session_state[f"{unique_key_prefix}_selected_cols"] = valid_cols
        st.session_state[f"{unique_key_prefix}_noise_type"] = config_params.get('noise_type', 'lognormal')
        st.session_state[f"{unique_key_prefix}_noise_factor"] = config_params.get('noise_factor', 0.1)
        st.session_state[f"{unique_key_prefix}_preserve_zero"] = config_params.get('preserve_zero', True)
        st.session_state[f"{unique_key_prefix}_handle_negative"] = config_params.get('handle_negative', 'absolute')
        st.session_state[f"{unique_key_prefix}_clip_multiplier"] = config_params.get('clip_multiplier', True)
        st.session_state[f"{unique_key_prefix}_min_multiplier"] = config_params.get('min_multiplier', 0.5)
        st.session_state[f"{unique_key_prefix}_max_multiplier"] = config_params.get('max_multiplier', 2.0)
        st.session_state[f"{unique_key_prefix}_random_seed"] = config_params.get('random_seed', 42)

def get_plugin():
    """Factory function to get plugin instance."""
    return MultiplicativeNoisePlugin()
