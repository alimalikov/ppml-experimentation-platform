"""
Standalone Gaussian Mechanism Plugin for Data Anonymization

This plugin implements the standalone Gaussian mechanism for adding differential privacy noise
to numerical data. This provides approximate differential privacy (ε,δ)-DP and is separate 
from the comprehensive differential privacy plugins.
"""

import streamlit as st
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional
import json
import logging
from ..base_anonymizer import Anonymizer

logger = logging.getLogger(__name__)

class GaussianMechanismPlugin(Anonymizer):
    """
    Standalone Gaussian mechanism plugin for approximate differential privacy.
    
    Implements the Gaussian mechanism for adding calibrated noise
    to numerical data to achieve (ε,δ)-differential privacy.
    """

    def __init__(self):
        """Initialize the Gaussian mechanism plugin."""
        self._name = "Gaussian Mechanism"
        self._description = ("Standalone Gaussian mechanism for approximate differential privacy. "
                           "Adds calibrated Gaussian noise to numerical data based on "
                           "sensitivity analysis and privacy parameters (epsilon, delta). Provides "
                           "(ε,δ)-differential privacy guarantees with better utility than Laplace.")

    def get_name(self) -> str:
        """Returns the display name of the anonymization technique."""
        return self._name

    def get_category(self) -> str:
        """Returns the category of the anonymization technique."""
        return "Differential Privacy"

    def get_description(self) -> str:
        """Returns detailed description of the technique."""
        return self._description

    def get_sidebar_ui(self, all_cols: List[str], sa_col_to_pass: str | None, 
                      df_raw: pd.DataFrame, unique_key_prefix: str) -> Dict[str, Any]:
        """
        Renders the Gaussian mechanism specific UI elements in the Streamlit sidebar.
        """
        st.sidebar.header(f"🔔 {self.get_name()} Configuration")
        
        # Show description
        with st.sidebar.expander("ℹ️ About Gaussian Mechanism"):
            st.markdown(self._description)
            st.markdown("""
            **Key Properties:**
            - **Approximate DP**: Provides (ε,δ)-differential privacy
            - **Gaussian Noise**: Adds symmetric normal distribution noise
            - **Better Utility**: Often provides better utility than Laplace
            
            **Formula**: Noise ~ N(0, σ²) where σ = √(2ln(1.25/δ)) × Δf/ε
            
            **Use Cases:**
            - Large-scale data analysis
            - When some privacy leakage (δ) is acceptable
            - Better utility requirements
            """)

        # Define session state keys
        cols_key = f"{unique_key_prefix}_selected_cols"
        epsilon_key = f"{unique_key_prefix}_epsilon"
        delta_key = f"{unique_key_prefix}_delta"
        sensitivity_key = f"{unique_key_prefix}_sensitivity"
        auto_sensitivity_key = f"{unique_key_prefix}_auto_sensitivity"
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
            default_selected = st.session_state.get(cols_key, numeric_cols[:3])
            valid_default = [col for col in default_selected if col in numeric_cols]
            
            selected_cols = st.sidebar.multiselect(
                "Select columns for Gaussian mechanism:",
                options=numeric_cols,
                default=valid_default,
                key=cols_key,
                help="Choose numerical columns for Gaussian noise addition"
            )

        # Privacy Parameters
        st.sidebar.subheader("🔐 Privacy Configuration")
        
        # Epsilon (privacy budget)
        epsilon = st.sidebar.number_input(
            "Privacy Budget (ε):",
            min_value=0.01,
            max_value=10.0,
            value=st.session_state.get(epsilon_key, 1.0),
            step=0.1,
            key=epsilon_key,
            help="Lower epsilon = stronger privacy, higher noise"
        )

        # Delta (failure probability)
        delta = st.sidebar.number_input(
            "Failure Probability (δ):",
            min_value=1e-10,
            max_value=0.1,
            value=st.session_state.get(delta_key, 1e-5),
            step=1e-6,
            format="%.2e",
            key=delta_key,
            help="Probability of privacy failure. Typically much smaller than 1/n"
        )

        # Sensitivity Configuration
        auto_sensitivity = st.sidebar.checkbox(
            "Auto-calculate sensitivity",
            value=st.session_state.get(auto_sensitivity_key, True),
            key=auto_sensitivity_key,
            help="Automatically estimate global sensitivity from data range"
        )

        if auto_sensitivity:
            sensitivity_info = "Auto-calculated from data range"
            manual_sensitivity = None
        else:
            manual_sensitivity = st.sidebar.number_input(
                "Manual Sensitivity (Δf):",
                min_value=0.01,
                max_value=1000.0,
                value=st.session_state.get(sensitivity_key, 1.0),
                step=0.1,
                key=sensitivity_key,
                help="Maximum change in output due to single record change"
            )
            sensitivity_info = f"Manual: {manual_sensitivity}"

        st.sidebar.info(f"**Sensitivity**: {sensitivity_info}")

        # Data Preprocessing
        st.sidebar.subheader("📏 Data Preprocessing")
        
        clip_values = st.sidebar.checkbox(
            "Clip values to range",
            value=st.session_state.get(clip_values_key, False),
            key=clip_values_key,
            help="Clip data to specified range before adding noise"
        )

        if clip_values:
            col1, col2 = st.sidebar.columns(2)
            with col1:
                clip_min = st.number_input(
                    "Min:",
                    value=st.session_state.get(clip_min_key, 0.0),
                    key=clip_min_key
                )
            with col2:
                clip_max = st.number_input(
                    "Max:",
                    value=st.session_state.get(clip_max_key, 100.0),
                    key=clip_max_key
                )
        else:
            clip_min = clip_max = None

        # Reproducibility
        st.sidebar.subheader("🎲 Reproducibility")
        random_seed = st.sidebar.number_input(
            "Random Seed (optional):",
            min_value=0,
            max_value=999999,
            value=st.session_state.get(random_seed_key, 42),
            key=random_seed_key,
            help="Set seed for reproducible results"
        )

        # Privacy Analysis
        if selected_cols and epsilon > 0 and delta > 0:
            st.sidebar.subheader("📈 Privacy Analysis")
            
            # Calculate noise parameters
            if auto_sensitivity and df_raw is not None:
                data_ranges = []
                for col in selected_cols:
                    if col in df_raw.columns:
                        col_range = df_raw[col].max() - df_raw[col].min()
                        data_ranges.append(col_range)
                
                if data_ranges:
                    estimated_sensitivity = max(data_ranges)
                    # Gaussian mechanism standard deviation
                    sigma = np.sqrt(2 * np.log(1.25 / delta)) * estimated_sensitivity / epsilon
                    
                    st.sidebar.metric("Estimated Sensitivity", f"{estimated_sensitivity:.3f}")
                    st.sidebar.metric("Noise Std Dev (σ)", f"{sigma:.3f}")
                    st.sidebar.metric("SNR (Signal/Noise)", f"{estimated_sensitivity/sigma:.2f}")
            
            elif manual_sensitivity:
                sigma = np.sqrt(2 * np.log(1.25 / delta)) * manual_sensitivity / epsilon
                st.sidebar.metric("Noise Std Dev (σ)", f"{sigma:.3f}")
                st.sidebar.metric("SNR (Signal/Noise)", f"{manual_sensitivity/sigma:.2f}")

        return {
            'selected_cols': selected_cols,
            'epsilon': epsilon,
            'delta': delta,
            'auto_sensitivity': auto_sensitivity,
            'manual_sensitivity': manual_sensitivity,
            'clip_values': clip_values,
            'clip_min': clip_min,
            'clip_max': clip_max,
            'random_seed': random_seed
        }

    def anonymize(self, df_input: pd.DataFrame, parameters: Dict[str, Any], sa_col: str | None) -> pd.DataFrame:
        """
        Apply Gaussian mechanism to the input DataFrame.
        """
        if df_input.empty:
            return df_input.copy()

        # Extract parameters
        selected_cols = parameters.get('selected_cols', [])
        epsilon = parameters.get('epsilon', 1.0)
        delta = parameters.get('delta', 1e-5)
        auto_sensitivity = parameters.get('auto_sensitivity', True)
        manual_sensitivity = parameters.get('manual_sensitivity', 1.0)
        clip_values = parameters.get('clip_values', False)
        clip_min = parameters.get('clip_min', None)
        clip_max = parameters.get('clip_max', None)
        random_seed = parameters.get('random_seed', None)

        if not selected_cols:
            logger.warning("No columns selected for Gaussian mechanism")
            return df_input.copy()

        # Set random seed for reproducibility
        if random_seed is not None:
            np.random.seed(random_seed)

        result_df = df_input.copy()

        for col in selected_cols:
            if col not in result_df.columns:
                logger.warning(f"Column {col} not found in dataset")
                continue

            if not pd.api.types.is_numeric_dtype(result_df[col]):
                logger.warning(f"Column {col} is not numeric, skipping")
                continue

            # Get column data
            col_data = result_df[col].values.astype(float)

            # Clip values if requested
            if clip_values and clip_min is not None and clip_max is not None:
                col_data = np.clip(col_data, clip_min, clip_max)

            # Calculate sensitivity
            if auto_sensitivity:
                col_range = np.max(col_data) - np.min(col_data)
                sensitivity = max(col_range, 0.001)  # Avoid division by zero
            else:
                sensitivity = manual_sensitivity if manual_sensitivity else 1.0

            # Calculate noise standard deviation
            sigma = np.sqrt(2 * np.log(1.25 / delta)) * sensitivity / epsilon

            # Generate Gaussian noise
            gaussian_noise = np.random.normal(0, sigma, size=len(col_data))

            # Add noise to data
            noisy_data = col_data + gaussian_noise

            # Update result
            result_df[col] = noisy_data

            logger.info(f"Applied Gaussian mechanism to column {col}: "
                       f"sensitivity={sensitivity:.3f}, sigma={sigma:.3f}")

        return result_df

    def build_config_export(self, unique_key_prefix: str, sa_col: str | None) -> Dict[str, Any]:
        """
        Build configuration for export.
        """
        return {
            "selected_cols": st.session_state.get(f"{unique_key_prefix}_selected_cols", []),
            "epsilon": st.session_state.get(f"{unique_key_prefix}_epsilon", 1.0),
            "delta": st.session_state.get(f"{unique_key_prefix}_delta", 1e-5),
            "auto_sensitivity": st.session_state.get(f"{unique_key_prefix}_auto_sensitivity", True),
            "manual_sensitivity": st.session_state.get(f"{unique_key_prefix}_manual_sensitivity", 1.0),
            "clip_values": st.session_state.get(f"{unique_key_prefix}_clip_values", False),
            "clip_min": st.session_state.get(f"{unique_key_prefix}_clip_min", 0.0),
            "clip_max": st.session_state.get(f"{unique_key_prefix}_clip_max", 100.0),
            "random_seed": st.session_state.get(f"{unique_key_prefix}_random_seed", 42)
        }

    def apply_config_import(self, config_params: Dict[str, Any], all_cols: List[str], unique_key_prefix: str):
        """
        Apply imported configuration to session state.
        """
        st.session_state[f"{unique_key_prefix}_selected_cols"] = config_params.get("selected_cols", [])
        st.session_state[f"{unique_key_prefix}_epsilon"] = config_params.get("epsilon", 1.0)
        st.session_state[f"{unique_key_prefix}_delta"] = config_params.get("delta", 1e-5)
        st.session_state[f"{unique_key_prefix}_auto_sensitivity"] = config_params.get("auto_sensitivity", True)
        st.session_state[f"{unique_key_prefix}_manual_sensitivity"] = config_params.get("manual_sensitivity", 1.0)
        st.session_state[f"{unique_key_prefix}_clip_values"] = config_params.get("clip_values", False)
        st.session_state[f"{unique_key_prefix}_clip_min"] = config_params.get("clip_min", 0.0)
        st.session_state[f"{unique_key_prefix}_clip_max"] = config_params.get("clip_max", 100.0)
        st.session_state[f"{unique_key_prefix}_random_seed"] = config_params.get("random_seed", 42)

def get_plugin():
    """Factory function to get plugin instance."""
    return GaussianMechanismPlugin()
