"""
Standalone Exponential Mechanism Plugin for Data Anonymization

This plugin implements the standalone exponential mechanism for differential privacy.
The exponential mechanism is designed for selecting from a discrete set of alternatives
based on a quality/utility function while preserving differential privacy.
"""

import streamlit as st
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional, Callable
import json
import logging
from ..base_anonymizer import Anonymizer

logger = logging.getLogger(__name__)

class ExponentialMechanismPlugin(Anonymizer):
    """
    Standalone exponential mechanism plugin for differential privacy.
    
    Implements the exponential mechanism for private selection from discrete alternatives.
    Useful for categorical data, mode selection, and other discrete choice problems.
    """

    def __init__(self):
        """Initialize the exponential mechanism plugin."""
        self._name = "Exponential Mechanism"
        self._description = ("Standalone exponential mechanism for differential privacy. "
                           "Selects from discrete alternatives based on a utility function "
                           "while providing epsilon-differential privacy. Ideal for categorical "
                           "data anonymization and private selection problems.")

    def get_name(self) -> str:
        """Returns the display name of the anonymization technique."""
        return "Exponential Mechanism"

    def get_category(self) -> str:
        """Returns the category of the anonymization technique."""
        return "Differential Privacy"

    def get_description(self) -> str:
        """Returns detailed description of the technique."""
        return "Standalone exponential mechanism for differential privacy. Selects from discrete alternatives based on a utility function while providing epsilon-differential privacy. Ideal for categorical data and discrete choice problems."

    def get_sidebar_ui(self, all_cols: List[str], sa_col_to_pass: str | None, 
                      df_raw: pd.DataFrame, unique_key_prefix: str) -> Dict[str, Any]:
        """
        Renders the exponential mechanism specific UI elements in the Streamlit sidebar.
        """
        st.sidebar.header(f"🎯 {self.get_name()} Configuration")
        
        # Show description
        with st.sidebar.expander("ℹ️ About Exponential Mechanism"):
            st.markdown(self._description)
            st.markdown("""
            **Key Properties:**
            - **Pure DP**: Provides ε-differential privacy
            - **Discrete Selection**: Works with categorical/discrete data
            - **Utility-Based**: Selection based on utility function
            
            **Formula**: P(output = r) ∝ exp(ε × u(x,r) / (2×Δu))
            where u(x,r) is the utility function and Δu is its sensitivity.
            
            **Use Cases:**
            - Categorical data anonymization
            - Private mode/median selection
            - Discrete optimization problems
            """)

        # Define session state keys
        cols_key = f"{unique_key_prefix}_selected_cols"
        epsilon_key = f"{unique_key_prefix}_epsilon"
        mechanism_type_key = f"{unique_key_prefix}_mechanism_type"
        top_k_key = f"{unique_key_prefix}_top_k"
        utility_sensitivity_key = f"{unique_key_prefix}_utility_sensitivity"
        random_seed_key = f"{unique_key_prefix}_random_seed"

        # Column Selection
        st.sidebar.subheader("📊 Column Selection")
        categorical_cols = []
        numeric_cols = []
        
        if df_raw is not None:
            categorical_cols = df_raw.select_dtypes(include=['object', 'category']).columns.tolist()
            numeric_cols = df_raw.select_dtypes(include=[np.number]).columns.tolist()
        
        all_applicable_cols = categorical_cols + numeric_cols
        
        if not all_applicable_cols:
            st.sidebar.warning("No suitable columns found in the dataset.")
            selected_cols = []
        else:
            default_selected = st.session_state.get(cols_key, all_applicable_cols[:2])
            valid_default = [col for col in default_selected if col in all_applicable_cols]
            
            selected_cols = st.sidebar.multiselect(
                "Select columns for exponential mechanism:",
                options=all_applicable_cols,
                default=valid_default,
                key=cols_key,
                help="Choose columns for private selection"
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
            help="Lower epsilon = stronger privacy, more randomness"
        )

        # Mechanism Configuration
        st.sidebar.subheader("⚙️ Mechanism Configuration")
        
        mechanism_types = {
            'mode_selection': 'Private Mode Selection',
            'top_k_selection': 'Top-K Private Selection',
            'weighted_sampling': 'Weighted Private Sampling',
            'range_query': 'Private Range Query (Numeric)'
        }
        
        mechanism_type = st.sidebar.selectbox(
            "Mechanism Type:",
            options=list(mechanism_types.keys()),
            format_func=lambda x: mechanism_types[x],
            index=list(mechanism_types.keys()).index(
                st.session_state.get(mechanism_type_key, 'mode_selection')
            ),
            key=mechanism_type_key,
            help="Choose the type of exponential mechanism to apply"
        )

        # Mechanism-specific parameters
        if mechanism_type == 'top_k_selection':
            top_k = st.sidebar.number_input(
                "Top-K value:",
                min_value=1,
                max_value=10,
                value=st.session_state.get(top_k_key, 3),
                key=top_k_key,
                help="Number of top items to select privately"
            )
        else:
            top_k = None

        # Utility sensitivity
        utility_sensitivity = st.sidebar.number_input(
            "Utility Sensitivity (Δu):",
            min_value=0.1,
            max_value=10.0,
            value=st.session_state.get(utility_sensitivity_key, 1.0),
            step=0.1,
            key=utility_sensitivity_key,
            help="Maximum change in utility function due to single record"
        )

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
        if selected_cols and epsilon > 0:
            st.sidebar.subheader("📈 Privacy Analysis")
            
            # Calculate selection probabilities for small examples
            if df_raw is not None and len(selected_cols) > 0:
                sample_col = selected_cols[0]
                if sample_col in df_raw.columns:
                    if mechanism_type == 'mode_selection':
                        value_counts = df_raw[sample_col].value_counts()
                        if len(value_counts) <= 5:  # Show probabilities for small sets
                            utilities = value_counts.values
                            exp_utilities = np.exp(epsilon * utilities / (2 * utility_sensitivity))
                            probabilities = exp_utilities / np.sum(exp_utilities)
                            
                            st.sidebar.text("Selection Probabilities:")
                            for val, prob in zip(value_counts.index, probabilities):
                                st.sidebar.text(f"  {val}: {prob:.3f}")

        return {
            'selected_cols': selected_cols,
            'epsilon': epsilon,
            'mechanism_type': mechanism_type,
            'top_k': top_k,
            'utility_sensitivity': utility_sensitivity,
            'random_seed': random_seed
        }

    def _exponential_mechanism_select(self, utilities: np.ndarray, epsilon: float, 
                                    sensitivity: float) -> int:
        """
        Core exponential mechanism selection.
        """
        # Calculate selection probabilities
        exp_utilities = np.exp(epsilon * utilities / (2 * sensitivity))
        probabilities = exp_utilities / np.sum(exp_utilities)
        
        # Sample according to probabilities
        return np.random.choice(len(utilities), p=probabilities)

    def _mode_selection(self, series: pd.Series, epsilon: float, sensitivity: float) -> Any:
        """
        Private mode selection using exponential mechanism.
        """
        value_counts = series.value_counts()
        utilities = value_counts.values.astype(float)
        
        selected_idx = self._exponential_mechanism_select(utilities, epsilon, sensitivity)
        return value_counts.index[selected_idx]

    def _top_k_selection(self, series: pd.Series, k: int, epsilon: float, sensitivity: float) -> List[Any]:
        """
        Private top-k selection using exponential mechanism.
        """
        value_counts = series.value_counts()
        utilities = value_counts.values.astype(float)
        
        selected_items = []
        remaining_indices = list(range(len(utilities)))
        
        for _ in range(min(k, len(remaining_indices))):
            remaining_utilities = utilities[remaining_indices]
            selected_pos = self._exponential_mechanism_select(remaining_utilities, epsilon, sensitivity)
            actual_idx = remaining_indices[selected_pos]
            
            selected_items.append(value_counts.index[actual_idx])
            remaining_indices.remove(actual_idx)
        
        return selected_items

    def _weighted_sampling(self, series: pd.Series, epsilon: float, sensitivity: float) -> Any:
        """
        Weighted private sampling using exponential mechanism.
        """
        if pd.api.types.is_numeric_dtype(series):
            # For numeric data, use values as utilities
            utilities = series.values.astype(float)
            selected_idx = self._exponential_mechanism_select(utilities, epsilon, sensitivity)
            return series.iloc[selected_idx]
        else:
            # For categorical data, use frequency as utility
            return self._mode_selection(series, epsilon, sensitivity)

    def _range_query(self, series: pd.Series, epsilon: float, sensitivity: float) -> float:
        """
        Private range query using exponential mechanism.
        """
        if not pd.api.types.is_numeric_dtype(series):
            logger.warning("Range query requires numeric data")
            return series.iloc[0] if len(series) > 0 else 0.0
        
        # Use values as utilities
        utilities = series.values.astype(float)
        selected_idx = self._exponential_mechanism_select(utilities, epsilon, sensitivity)
        return utilities[selected_idx]

    def anonymize(self, df_input: pd.DataFrame, parameters: Dict[str, Any], sa_col: str | None) -> pd.DataFrame:
        """
        Apply exponential mechanism to the input DataFrame.
        """
        if df_input.empty:
            return df_input.copy()

        # Extract parameters
        selected_cols = parameters.get('selected_cols', [])
        epsilon = parameters.get('epsilon', 1.0)
        mechanism_type = parameters.get('mechanism_type', 'mode_selection')
        top_k = parameters.get('top_k', 3)
        utility_sensitivity = parameters.get('utility_sensitivity', 1.0)
        random_seed = parameters.get('random_seed', None)

        if not selected_cols:
            logger.warning("No columns selected for exponential mechanism")
            return df_input.copy()

        # Set random seed for reproducibility
        if random_seed is not None:
            np.random.seed(random_seed)

        result_df = df_input.copy()

        for col in selected_cols:
            if col not in result_df.columns:
                logger.warning(f"Column {col} not found in dataset")
                continue

            col_series = result_df[col].dropna()
            if len(col_series) == 0:
                logger.warning(f"Column {col} has no valid data")
                continue

            try:
                if mechanism_type == 'mode_selection':
                    selected_value = self._mode_selection(col_series, epsilon, utility_sensitivity)
                    # Replace all values with the privately selected mode
                    result_df[col] = selected_value
                    
                elif mechanism_type == 'top_k_selection':
                    selected_values = self._top_k_selection(col_series, top_k, epsilon, utility_sensitivity)
                    # Replace values with randomly chosen from top-k
                    replacement_values = np.random.choice(selected_values, size=len(result_df))
                    result_df[col] = replacement_values
                    
                elif mechanism_type == 'weighted_sampling':
                    # Sample new values for each row
                    new_values = []
                    for _ in range(len(result_df)):
                        sampled_value = self._weighted_sampling(col_series, epsilon, utility_sensitivity)
                        new_values.append(sampled_value)
                    result_df[col] = new_values
                    
                elif mechanism_type == 'range_query':
                    selected_value = self._range_query(col_series, epsilon, utility_sensitivity)
                    result_df[col] = selected_value

                logger.info(f"Applied {mechanism_type} exponential mechanism to column {col}")

            except Exception as e:
                logger.error(f"Error applying exponential mechanism to column {col}: {e}")
                continue

        return result_df

    def build_config_export(self, unique_key_prefix: str, sa_col: str | None) -> Dict[str, Any]:
        """
        Build configuration for export.
        """
        return {
            "selected_cols": st.session_state.get(f"{unique_key_prefix}_selected_cols", []),
            "epsilon": st.session_state.get(f"{unique_key_prefix}_epsilon", 1.0),
            "mechanism_type": st.session_state.get(f"{unique_key_prefix}_mechanism_type", 'mode_selection'),
            "top_k": st.session_state.get(f"{unique_key_prefix}_top_k", 3),
            "utility_sensitivity": st.session_state.get(f"{unique_key_prefix}_utility_sensitivity", 1.0),
            "random_seed": st.session_state.get(f"{unique_key_prefix}_random_seed", 42)
        }

    def apply_config_import(self, config_params: Dict[str, Any], all_cols: List[str], unique_key_prefix: str):
        """
        Apply imported configuration to session state.
        """
        st.session_state[f"{unique_key_prefix}_selected_cols"] = config_params.get("selected_cols", [])
        st.session_state[f"{unique_key_prefix}_epsilon"] = config_params.get("epsilon", 1.0)
        st.session_state[f"{unique_key_prefix}_mechanism_type"] = config_params.get("mechanism_type", 'mode_selection')
        st.session_state[f"{unique_key_prefix}_top_k"] = config_params.get("top_k", 3)
        st.session_state[f"{unique_key_prefix}_utility_sensitivity"] = config_params.get("utility_sensitivity", 1.0)
        st.session_state[f"{unique_key_prefix}_random_seed"] = config_params.get("random_seed", 42)

def get_plugin():
    """Factory function to get plugin instance."""
    return ExponentialMechanismPlugin()
