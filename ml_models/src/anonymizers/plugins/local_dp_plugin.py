"""
Professional Local Differential Privacy plugin for the anonymization tool.
Provides local differential privacy where noise is added at the individual record level.
"""

import streamlit as st
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional
import json
from ..base_anonymizer import Anonymizer
from ..differential_privacy_core import DifferentialPrivacyCore

class LocalDifferentialPrivacyPlugin(Anonymizer):
    """
    Professional local differential privacy plugin with record-level privacy protection.
    """

    def __init__(self):
        """Initialize the local differential privacy plugin."""
        self._name = "Local Differential Privacy"
        self._description = ("Local differential privacy implementation where each individual's "
                           "data is perturbed locally before collection. Provides stronger privacy "
                           "guarantees but may have higher utility loss compared to central DP.")
        self.dp_core = DifferentialPrivacyCore()

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
        Renders the local differential privacy specific UI elements in the Streamlit sidebar.
        """
        st.sidebar.header(f"🔐 {self.get_name()} Configuration")
        
        # Show description
        with st.sidebar.expander("ℹ️ About Local Differential Privacy"):
            st.markdown(self._description)
            st.markdown("""
            **Key Features:**
            - Record-level privacy protection
            - No trusted curator required
            - Randomized response for categorical data
            - Per-record noise calibration
            - Strong privacy guarantees
            
            **Best for:** Scenarios without trusted data curator, survey data, sensitive personal information
            
            **Differences from Central DP:**
            - Noise added at individual level (not aggregate level)
            - Higher privacy protection but potentially lower utility
            - No need for trusted central server
            - Each record is independently protected
            """)

        # Define session state keys
        cols_key = f"{unique_key_prefix}_ldp_cols"
        epsilon_key = f"{unique_key_prefix}_ldp_epsilon"
        mechanism_key = f"{unique_key_prefix}_ldp_mechanism"
        perturbation_key = f"{unique_key_prefix}_perturbation_type"
        domain_key = f"{unique_key_prefix}_domain_size"
        show_metrics_key = f"{unique_key_prefix}_show_metrics"

        # Column Selection
        st.sidebar.subheader("📊 Column Selection")
        default_cols = st.session_state.get(cols_key, [])
        valid_default_cols = [col for col in default_cols if col in all_cols]

        # Separate numeric and categorical columns
        if df_raw is not None and not df_raw.empty:
            numeric_cols = [col for col in all_cols if pd.api.types.is_numeric_dtype(df_raw[col])]
            categorical_cols = [col for col in all_cols if not pd.api.types.is_numeric_dtype(df_raw[col])]
        else:
            numeric_cols = all_cols
            categorical_cols = []

        if numeric_cols:
            st.sidebar.info(f"📈 Numeric columns: {', '.join(numeric_cols[:3])}{'...' if len(numeric_cols) > 3 else ''}")
        if categorical_cols:
            st.sidebar.info(f"📝 Categorical columns: {', '.join(categorical_cols[:3])}{'...' if len(categorical_cols) > 3 else ''}")

        selected_cols = st.sidebar.multiselect(
            "Select columns for Local DP:",
            options=all_cols,
            default=valid_default_cols,
            key=cols_key,
            help="Choose columns to apply local differential privacy. Each record will be independently perturbed."
        )

        # Privacy Parameters
        st.sidebar.subheader("🎯 Privacy Parameters")

        # Epsilon (Privacy Budget) - typically smaller for LDP
        current_epsilon = st.session_state.get(epsilon_key, 0.5)
        epsilon = st.sidebar.number_input(
            "Privacy Budget per Record (ε):",
            min_value=0.01,
            max_value=5.0,
            value=current_epsilon,
            step=0.05,
            key=epsilon_key,
            help="Privacy budget applied to each individual record. "
                 "Smaller values provide stronger privacy. "
                 "For LDP, typical values are smaller than central DP."
        )

        # Privacy level indicator
        if epsilon <= 0.1:
            privacy_level = "🔒 Extremely High Privacy"
            level_color = "green"
        elif epsilon <= 0.5:
            privacy_level = "🔐 Very High Privacy"
            level_color = "blue"
        elif epsilon <= 1.0:
            privacy_level = "🔓 High Privacy"
            level_color = "orange"
        else:
            privacy_level = "⚠️ Moderate Privacy"
            level_color = "red"

        st.sidebar.markdown(f"<span style='color: {level_color}'>{privacy_level}</span>", 
                           unsafe_allow_html=True)

        # Mechanism Selection
        st.sidebar.subheader("🔧 Local DP Mechanism")
        
        mechanism_options = {
            "randomized_response": "Randomized Response (Categorical)",
            "laplace_local": "Local Laplace Mechanism (Numeric)",
            "gaussian_local": "Local Gaussian Mechanism (Numeric)",
            "binary_randomized_response": "Binary Randomized Response",
            "unary_encoding": "Unary Encoding (for categorical)"
        }
        
        current_mechanism = st.session_state.get(mechanism_key, "randomized_response")
        mechanism = st.sidebar.selectbox(
            "Local DP Mechanism:",
            options=list(mechanism_options.keys()),
            format_func=lambda x: mechanism_options[x],
            index=list(mechanism_options.keys()).index(current_mechanism),
            key=mechanism_key,
            help="Choose the local differential privacy mechanism. "
                 "Randomized Response is best for categorical data, "
                 "Laplace/Gaussian for numeric data."
        )

        # Perturbation Configuration
        st.sidebar.subheader("⚙️ Perturbation Settings")
        
        perturbation_types = {
            "full": "Full Perturbation (all records)",
            "selective": "Selective Perturbation (probability-based)",
            "threshold": "Threshold-based Perturbation"
        }
        
        current_perturbation = st.session_state.get(perturbation_key, "full")
        perturbation_type = st.sidebar.selectbox(
            "Perturbation Type:",
            options=list(perturbation_types.keys()),
            format_func=lambda x: perturbation_types[x],
            index=list(perturbation_types.keys()).index(current_perturbation),
            key=perturbation_key,
            help="How to apply perturbation to records"
        )

        # Domain size estimation for categorical columns
        domain_sizes = {}
        if selected_cols and df_raw is not None and not df_raw.empty:
            categorical_selected = [col for col in selected_cols if col in categorical_cols]
            
            if categorical_selected:
                st.sidebar.subheader("📐 Domain Configuration")
                
                for col in categorical_selected:
                    unique_vals = df_raw[col].nunique()
                    domain_key_col = f"{domain_key}_{col}"
                    
                    current_domain = st.session_state.get(domain_key_col, unique_vals)
                    domain_size = st.sidebar.number_input(
                        f"Domain size for {col}:",
                        min_value=2,
                        max_value=max(100, unique_vals * 2),
                        value=current_domain,
                        key=domain_key_col,
                        help=f"Size of the domain for {col}. Current unique values: {unique_vals}"
                    )
                    domain_sizes[col] = domain_size

        # Advanced Options
        with st.sidebar.expander("🔧 Advanced Local DP Options"):
            # Show detailed metrics
            show_metrics = st.checkbox(
                "Show Detailed Privacy Metrics",
                value=st.session_state.get(show_metrics_key, True),
                key=show_metrics_key,
                help="Display comprehensive local privacy and utility metrics"
            )
            
            # Aggregation settings for utility estimation
            st.write("**Utility Estimation Settings:**")
            estimate_aggregates = st.checkbox(
                "Estimate aggregate statistics",
                value=True,
                help="Estimate means, counts after local perturbation for utility assessment"
            )

        # Privacy Analysis Preview
        if selected_cols and df_raw is not None and not df_raw.empty:
            st.sidebar.subheader("📊 Privacy Analysis Preview")
            
            # Estimate utility loss
            if mechanism == "randomized_response":
                # For randomized response
                categorical_selected = [col for col in selected_cols if col in categorical_cols]
                if categorical_selected:
                    for col in categorical_selected:
                        k = domain_sizes.get(col, df_raw[col].nunique())
                        p_truth = np.exp(epsilon) / (np.exp(epsilon) + k - 1)
                        st.sidebar.write(f"• {col}: Truth prob = {p_truth:.3f}")
                        
            elif mechanism in ["laplace_local", "gaussian_local"]:
                # For numeric mechanisms
                numeric_selected = [col for col in selected_cols if col in numeric_cols]
                if numeric_selected:
                    for col in numeric_selected:
                        col_range = df_raw[col].max() - df_raw[col].min()
                        sensitivity = col_range  # For local DP, sensitivity is typically the range
                        noise_scale = sensitivity / epsilon
                        st.sidebar.write(f"• {col}: Noise scale ≈ {noise_scale:.3f}")

        return {
            "columns": selected_cols,
            "epsilon": epsilon,
            "mechanism": mechanism,
            "perturbation_type": perturbation_type,
            "domain_sizes": domain_sizes,
            "show_metrics": show_metrics
        }

    def anonymize(self, df: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
        """
        Apply local differential privacy to the DataFrame.
        
        Args:
            df: Input DataFrame
            config: Configuration from sidebar UI
            
        Returns:
            DataFrame with local differential privacy applied
        """
        if df.empty:
            return df
            
        # Extract configuration
        columns = config.get("columns", [])
        epsilon = config.get("epsilon", 0.5)
        mechanism = config.get("mechanism", "randomized_response")
        perturbation_type = config.get("perturbation_type", "full")
        domain_sizes = config.get("domain_sizes", {})
        
        if not columns:
            st.warning("No columns selected for local differential privacy.")
            return df
            
        try:
            result_df = df.copy()
            
            # Separate numeric and categorical columns
            numeric_cols = [col for col in columns if col in df.columns 
                          and pd.api.types.is_numeric_dtype(df[col])]
            categorical_cols = [col for col in columns if col in df.columns 
                              and not pd.api.types.is_numeric_dtype(df[col])]
            
            # Apply local DP to categorical columns
            for col in categorical_cols:
                if mechanism in ["randomized_response", "binary_randomized_response", "unary_encoding"]:
                    # Get domain values
                    unique_vals = list(df[col].unique())
                    domain_size = domain_sizes.get(col, len(unique_vals))
                    
                    # Expand domain if needed
                    if domain_size > len(unique_vals):
                        # Add dummy values to expand domain
                        dummy_vals = [f"dummy_{i}" for i in range(domain_size - len(unique_vals))]
                        possible_values = unique_vals + dummy_vals
                    else:
                        possible_values = unique_vals
                    
                    # Apply randomized response
                    result_df[col] = self._apply_local_randomized_response(
                        df[col], epsilon, possible_values
                    )
            
            # Apply local DP to numeric columns
            for col in numeric_cols:
                if mechanism in ["laplace_local", "gaussian_local"]:
                    # Calculate local sensitivity (typically the range for numeric data)
                    col_min, col_max = df[col].min(), df[col].max()
                    sensitivity = col_max - col_min
                    
                    if mechanism == "laplace_local":
                        # Apply local Laplace noise to each record
                        noise_scale = sensitivity / epsilon
                        noise = np.random.laplace(0, noise_scale, size=len(df))
                        result_df[col] = df[col] + noise
                        
                    elif mechanism == "gaussian_local":
                        # Apply local Gaussian noise
                        # For local DP, we use a simpler formulation
                        sigma = sensitivity * np.sqrt(2) / epsilon
                        noise = np.random.normal(0, sigma, size=len(df))
                        result_df[col] = df[col] + noise
            
            # Apply perturbation type logic
            if perturbation_type == "selective":
                # Only perturb a fraction of records
                perturb_prob = min(1.0, epsilon)  # Simple heuristic
                mask = np.random.random(len(df)) < perturb_prob
                
                # Revert non-selected records
                for col in columns:
                    if col in result_df.columns:
                        result_df.loc[~mask, col] = df.loc[~mask, col]
            
            # Show success message
            st.success(f"✅ Local Differential Privacy applied successfully!")
            st.info(f"Applied {mechanism} with ε={epsilon:.3f} per record")
            
            return result_df
            
        except Exception as e:
            st.error(f"Error applying local differential privacy: {str(e)}")
            return df

    def _apply_local_randomized_response(self, data: pd.Series, epsilon: float, 
                                       possible_values: List[Any]) -> pd.Series:
        """Apply local randomized response to categorical data."""
        k = len(possible_values)
        
        # Calculate probabilities
        p_truth = np.exp(epsilon) / (np.exp(epsilon) + k - 1)
        p_other = 1 / (np.exp(epsilon) + k - 1)
        
        result = []
        for value in data:
            if np.random.random() < p_truth:
                # Output true value
                result.append(value)
            else:
                # Output random value from domain (excluding true value)
                other_values = [v for v in possible_values if v != value]
                if other_values:
                    result.append(np.random.choice(other_values))
                else:
                    result.append(value)
        
        return pd.Series(result, index=data.index)

    def get_privacy_metrics(self, original_df: pd.DataFrame, anonymized_df: pd.DataFrame, 
                          config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calculate privacy and utility metrics for local differential privacy.
        """
        if original_df.empty or anonymized_df.empty:
            return {}
            
        try:
            columns = config.get("columns", [])
            epsilon = config.get("epsilon", 0.5)
            mechanism = config.get("mechanism", "randomized_response")
            
            metrics = {
                "privacy_metrics": {
                    "epsilon_per_record": epsilon,
                    "mechanism": mechanism,
                    "privacy_level": self._classify_local_privacy_level(epsilon),
                    "privacy_type": "Local Differential Privacy",
                    "total_records_perturbed": len(anonymized_df)
                },
                "utility_metrics": {},
                "data_metrics": {
                    "original_rows": len(original_df),
                    "anonymized_rows": len(anonymized_df),
                    "columns_modified": len(columns),
                    "total_columns": len(original_df.columns)
                }
            }
            
            # Calculate utility metrics for different column types
            numeric_cols = [col for col in columns if col in original_df.columns 
                          and pd.api.types.is_numeric_dtype(original_df[col])]
            categorical_cols = [col for col in columns if col in original_df.columns 
                              and not pd.api.types.is_numeric_dtype(original_df[col])]
            
            utility_metrics = {}
            
            # Numeric columns
            for col in numeric_cols:
                if col in anonymized_df.columns:
                    orig_vals = original_df[col].dropna()
                    anon_vals = anonymized_df[col].dropna()
                    
                    if len(orig_vals) > 0 and len(anon_vals) > 0:
                        # Mean Absolute Error
                        mae = np.mean(np.abs(orig_vals.values - anon_vals.values[:len(orig_vals)]))
                        
                        # Relative Error
                        orig_range = orig_vals.max() - orig_vals.min()
                        relative_error = mae / orig_range if orig_range > 0 else 0
                        
                        utility_metrics[col] = {
                            "type": "numeric",
                            "mean_absolute_error": float(mae),
                            "relative_error": float(relative_error),
                            "original_mean": float(orig_vals.mean()),
                            "anonymized_mean": float(anon_vals.mean()),
                            "bias": float(anon_vals.mean() - orig_vals.mean()),
                            "noise_to_signal_ratio": float(mae / orig_vals.std()) if orig_vals.std() > 0 else 0
                        }
            
            # Categorical columns
            for col in categorical_cols:
                if col in anonymized_df.columns:
                    orig_counts = original_df[col].value_counts(normalize=True)
                    anon_counts = anonymized_df[col].value_counts(normalize=True)
                    
                    # Calculate distribution similarity
                    common_values = set(orig_counts.index) & set(anon_counts.index)
                    if common_values:
                        # Jensen-Shannon divergence approximation
                        js_div = 0
                        for val in common_values:
                            p = orig_counts[val]
                            q = anon_counts[val]
                            m = (p + q) / 2
                            if p > 0 and q > 0 and m > 0:
                                js_div += 0.5 * p * np.log2(p/m) + 0.5 * q * np.log2(q/m)
                        
                        utility_metrics[col] = {
                            "type": "categorical",
                            "js_divergence": float(js_div),
                            "distribution_similarity": float(1 - js_div),
                            "original_unique_values": len(orig_counts),
                            "anonymized_unique_values": len(anon_counts),
                            "value_overlap": len(common_values)
                        }
            
            metrics["utility_metrics"] = utility_metrics
            
            # Overall utility score
            if utility_metrics:
                numeric_scores = [1 - m["relative_error"] for m in utility_metrics.values() 
                                if m.get("type") == "numeric"]
                categorical_scores = [m["distribution_similarity"] for m in utility_metrics.values() 
                                    if m.get("type") == "categorical"]
                
                all_scores = numeric_scores + categorical_scores
                if all_scores:
                    metrics["overall_utility_score"] = float(np.mean(all_scores))
            
            return metrics
            
        except Exception as e:
            st.warning(f"Could not calculate all metrics: {str(e)}")
            return {"error": str(e)}

    def build_config_export(self, unique_key_prefix: str, sa_col: str | None) -> Dict[str, Any]:
        """
        Build the configuration export for local differential privacy.
        """
        # Export domain sizes
        domain_sizes = {}
        for key, value in st.session_state.items():
            if key.startswith(f"{unique_key_prefix}_domain_size_"):
                col_name = key.replace(f"{unique_key_prefix}_domain_size_", "")
                domain_sizes[col_name] = value
        
        return {
            "columns": st.session_state.get(f"{unique_key_prefix}_ldp_cols", []),
            "epsilon": st.session_state.get(f"{unique_key_prefix}_ldp_epsilon", 0.5),
            "mechanism": st.session_state.get(f"{unique_key_prefix}_ldp_mechanism", "randomized_response"),
            "perturbation_type": st.session_state.get(f"{unique_key_prefix}_perturbation_type", "full"),
            "domain_sizes": domain_sizes,
            "show_metrics": st.session_state.get(f"{unique_key_prefix}_show_metrics", True)
        }

    def apply_config_import(self, config_params: Dict[str, Any], all_cols: List[str], unique_key_prefix: str):
        """
        Apply imported configuration to session state for local differential privacy.
        """
        # Validate and set columns
        imported_cols = config_params.get("columns", [])
        valid_cols = [col for col in imported_cols if col in all_cols]
        st.session_state[f"{unique_key_prefix}_ldp_cols"] = valid_cols
        
        # Set other parameters with defaults
        st.session_state[f"{unique_key_prefix}_ldp_epsilon"] = config_params.get("epsilon", 0.5)
        st.session_state[f"{unique_key_prefix}_ldp_mechanism"] = config_params.get("mechanism", "randomized_response")
        st.session_state[f"{unique_key_prefix}_perturbation_type"] = config_params.get("perturbation_type", "full")
        st.session_state[f"{unique_key_prefix}_show_metrics"] = config_params.get("show_metrics", True)
        
        # Set domain sizes
        domain_sizes = config_params.get("domain_sizes", {})
        for col, size in domain_sizes.items():
            st.session_state[f"{unique_key_prefix}_domain_size_{col}"] = size

    def get_export_button_ui(self, config_to_export: dict, unique_key_prefix: str):
        """Export button UI for local differential privacy configuration."""
        json_string = json.dumps(config_to_export, indent=4)
        st.sidebar.download_button(
            label=f"Export {self.get_name()} Config",
            data=json_string,
            file_name=f"{self.get_name().lower().replace(' ', '_')}_config.json",
            mime="application/json",
            key=f"{unique_key_prefix}_export_button"
        )

    def get_anonymize_button_ui(self, unique_key_prefix: str) -> bool:
        """Anonymize button UI for local differential privacy."""
        return st.button(f"Anonymize with {self.get_name()}", key=f"{unique_key_prefix}_anonymize_button")

    def _classify_local_privacy_level(self, epsilon: float) -> str:
        """Classify local privacy level based on epsilon value."""
        if epsilon <= 0.1:
            return "Extremely High Privacy"
        elif epsilon <= 0.5:
            return "Very High Privacy"
        elif epsilon <= 1.0:
            return "High Privacy"
        elif epsilon <= 2.0:
            return "Moderate Privacy"
        else:
            return "Low Privacy"

    def export_config(self, config: Dict[str, Any]) -> str:
        """Export the current configuration as JSON string."""
        export_config = {
            "anonymizer": self.get_name(),
            "version": "1.0",
            "parameters": config,
            "timestamp": pd.Timestamp.now().isoformat()
        }
        return json.dumps(export_config, indent=2)

    def import_config(self, config_str: str) -> Dict[str, Any]:
        """Import configuration from JSON string."""
        try:
            config_data = json.loads(config_str)
            if config_data.get("anonymizer") != self.get_name():
                raise ValueError("Configuration is not for Local Differential Privacy")
            
            return config_data.get("parameters", {})
            
        except json.JSONDecodeError:
            raise ValueError("Invalid JSON configuration")
        except Exception as e:
            raise ValueError(f"Error importing configuration: {str(e)}")

# Create plugin instance
def get_plugin():
    """Factory function to create plugin instance."""
    return LocalDifferentialPrivacyPlugin()
