"""
Professional Standard Differential Privacy plugin for the anonymization tool.
Provides comprehensive standard differential privacy implementation with multiple noise mechanisms.
"""

import streamlit as st
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional
import json
from ..base_anonymizer import Anonymizer
from ..differential_privacy_core import DifferentialPrivacyCore

class StandardDifferentialPrivacyPlugin(Anonymizer):
    """
    Professional standard differential privacy plugin with multiple noise mechanisms.
    """

    def __init__(self):
        """Initialize the standard differential privacy plugin."""
        self._name = "Standard Differential Privacy"
        self._description = ("Professional differential privacy implementation providing formal "
                           "privacy guarantees through calibrated noise addition. Supports Laplace, "
                           "Gaussian, and Exponential mechanisms with privacy budget tracking.")
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
        Renders the standard differential privacy specific UI elements in the Streamlit sidebar.
        """
        st.sidebar.header(f"🔐 {self.get_name()} Configuration")
        
        # Show description
        with st.sidebar.expander("ℹ️ About Standard Differential Privacy"):
            st.markdown(self._description)
            st.markdown("""
            **Key Features:**
            - Multiple noise mechanisms (Laplace, Gaussian, Exponential)
            - Privacy budget management and tracking
            - Sensitivity analysis and automatic calibration
            - Support for both numerical and categorical data
            - Composition-aware privacy accounting
            
            **Best for:** Strong privacy guarantees, research applications, high-sensitivity data
            
            **Privacy Parameters:**
            - **ε (epsilon)**: Privacy budget - smaller values = stronger privacy
            - **δ (delta)**: Probability of privacy breach (for Gaussian mechanism)
            - **Sensitivity**: Maximum impact of a single record
            """)

        # Define session state keys
        cols_key = f"{unique_key_prefix}_dp_cols"
        epsilon_key = f"{unique_key_prefix}_epsilon"
        delta_key = f"{unique_key_prefix}_delta"
        mechanism_key = f"{unique_key_prefix}_mechanism"
        operation_key = f"{unique_key_prefix}_operation"
        budget_key = f"{unique_key_prefix}_budget_tracking"
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
            "Select columns to apply DP:",
            options=all_cols,
            default=valid_default_cols,
            key=cols_key,
            help="Choose columns to add differential privacy noise. Both numeric and categorical columns are supported."
        )

        # Privacy Parameters
        st.sidebar.subheader("🎯 Privacy Parameters")

        # Epsilon (Privacy Budget)
        current_epsilon = st.session_state.get(epsilon_key, 1.0)
        epsilon = st.sidebar.number_input(
            "Privacy Budget (ε - epsilon):",
            min_value=0.01,
            max_value=10.0,
            value=current_epsilon,
            step=0.1,
            key=epsilon_key,
            help="Smaller values provide stronger privacy but may reduce utility. "
                 "Typical values: 0.1 (high privacy) to 1.0 (moderate privacy)"
        )

        # Privacy level indicator
        if epsilon <= 0.1:
            privacy_level = "🔒 Very High Privacy"
            level_color = "green"
        elif epsilon <= 0.5:
            privacy_level = "🔐 High Privacy"
            level_color = "blue"
        elif epsilon <= 1.0:
            privacy_level = "🔓 Moderate Privacy"
            level_color = "orange"
        else:
            privacy_level = "⚠️ Low Privacy"
            level_color = "red"

        st.sidebar.markdown(f"<span style='color: {level_color}'>{privacy_level}</span>", 
                           unsafe_allow_html=True)

        # Noise Mechanism Selection
        st.sidebar.subheader("🔧 Mechanism Configuration")
        
        mechanism_options = {
            "laplace": "Laplace Mechanism (Pure DP)",
            "gaussian": "Gaussian Mechanism (Approximate DP)",
            "exponential": "Exponential Mechanism (for categorical)"
        }
        
        current_mechanism = st.session_state.get(mechanism_key, "laplace")
        mechanism = st.sidebar.selectbox(
            "Noise Mechanism:",
            options=list(mechanism_options.keys()),
            format_func=lambda x: mechanism_options[x],
            index=list(mechanism_options.keys()).index(current_mechanism),
            key=mechanism_key,
            help="Laplace: Pure DP, good for continuous data. "
                 "Gaussian: Better utility, requires delta parameter. "
                 "Exponential: For categorical/discrete choices."
        )

        # Delta parameter (only for Gaussian mechanism)
        delta = None
        if mechanism == "gaussian":
            current_delta = st.session_state.get(delta_key, 1e-5)
            delta = st.sidebar.number_input(
                "Delta (δ) parameter:",
                min_value=1e-10,
                max_value=1e-3,
                value=current_delta,
                format="%.2e",
                key=delta_key,
                help="Probability of privacy breach. Should be much smaller than 1/n where n is dataset size."
            )

        # Operation Type (for sensitivity calculation)
        st.sidebar.subheader("📐 Sensitivity Configuration")
        
        operation_options = {
            "identity": "Identity (add noise to raw values)",
            "mean": "Mean aggregation",
            "sum": "Sum aggregation", 
            "count": "Count queries",
            "max": "Maximum values",
            "min": "Minimum values"
        }
        
        current_operation = st.session_state.get(operation_key, "identity")
        operation = st.sidebar.selectbox(
            "Operation Type:",
            options=list(operation_options.keys()),
            format_func=lambda x: operation_options[x],
            index=list(operation_options.keys()).index(current_operation),
            key=operation_key,
            help="Type of operation for sensitivity calculation. Identity is most common for data release."
        )

        # Advanced Options
        with st.sidebar.expander("🔧 Advanced Options"):
            # Budget tracking
            budget_tracking = st.checkbox(
                "Enable Privacy Budget Tracking",
                value=st.session_state.get(budget_key, True),
                key=budget_key,
                help="Track cumulative privacy budget usage across operations"
            )
            
            # Show detailed metrics
            show_metrics = st.checkbox(
                "Show Detailed Privacy Metrics",
                value=st.session_state.get(show_metrics_key, True),
                key=show_metrics_key,
                help="Display comprehensive privacy and utility metrics"
            )

        # Privacy Budget Status
        if budget_tracking:
            st.sidebar.subheader("📊 Privacy Budget Status")
            current_budget = self.dp_core.privacy_budget_used
            remaining = max(0, epsilon - current_budget)
            
            if current_budget > 0:
                st.sidebar.progress(min(1.0, current_budget / epsilon))
                st.sidebar.write(f"Used: {current_budget:.3f} / {epsilon:.3f}")
                if remaining <= 0:
                    st.sidebar.error("⚠️ Privacy budget exhausted!")
                elif remaining < epsilon * 0.2:
                    st.sidebar.warning(f"⚠️ Low budget remaining: {remaining:.3f}")
                else:
                    st.sidebar.success(f"✅ Budget remaining: {remaining:.3f}")

        # Sensitivity Preview
        if selected_cols and df_raw is not None and not df_raw.empty:
            st.sidebar.subheader("📏 Sensitivity Preview")
            numeric_selected = [col for col in selected_cols if col in numeric_cols]
            
            if numeric_selected:
                try:
                    sensitivities = self.dp_core.calculate_global_sensitivity(
                        df_raw, numeric_selected, operation
                    )
                    
                    for col, sens in sensitivities.items():
                        st.sidebar.write(f"• {col}: {sens:.3f}")
                          # Estimate noise level
                    avg_sensitivity = np.mean(list(sensitivities.values()))
                    if mechanism == "laplace":
                        noise_scale = avg_sensitivity / epsilon
                        st.sidebar.info(f"Avg noise scale: {noise_scale:.3f}")
                    elif mechanism == "gaussian" and delta:
                        sigma = avg_sensitivity * np.sqrt(2 * np.log(1.25 / delta)) / epsilon
                        st.sidebar.info(f"Avg noise σ: {sigma:.3f}")
                        
                except Exception as e:
                    st.sidebar.warning(f"Could not calculate sensitivity: {str(e)}")
        
        return {
            "columns": selected_cols,
            "epsilon": epsilon,
            "delta": delta,
            "mechanism": mechanism,
            "operation": operation,
            "budget_tracking": budget_tracking,
            "show_metrics": show_metrics
        }

    def anonymize(self, df_input: pd.DataFrame, parameters: dict, sa_col: str | None) -> pd.DataFrame:
        """
        Apply standard differential privacy to the DataFrame.
        
        Args:
            df_input: Input DataFrame
            parameters: Configuration from sidebar UI
            sa_col: Selected sensitive attribute column (not used in DP)
            
        Returns:
            DataFrame with differential privacy applied
        """
        if df_input.empty:
            return df_input
            
        # Extract configuration
        columns = parameters.get("columns", [])
        epsilon = parameters.get("epsilon", 1.0)
        delta = parameters.get("delta", 1e-5)
        mechanism = parameters.get("mechanism", "laplace")
        operation = parameters.get("operation", "identity")
        budget_tracking = parameters.get("budget_tracking", True)
        
        if not columns:
            st.warning("No columns selected for differential privacy.")
            return df_input
            
        # Check privacy budget
        if budget_tracking:
            remaining_budget = epsilon - self.dp_core.privacy_budget_used
            if remaining_budget <= 0:
                st.error("Privacy budget exhausted! Reset budget or increase epsilon.")
                return df_input
            elif remaining_budget < epsilon:
                st.warning(f"Using remaining budget: {remaining_budget:.3f}")
                epsilon = remaining_budget

        try:
            # Apply differential privacy
            dp_config = {
                "epsilon": epsilon,
                "delta": delta,
                "mechanism": mechanism,
                "operation": operation,
                "columns": columns
            }
            
            result_df = self.dp_core.apply_differential_privacy(df_input, dp_config)
            
            # Show success message
            st.success(f"✅ Standard Differential Privacy applied successfully!")
            st.info(f"Applied {mechanism} mechanism with ε={epsilon:.3f}" + 
                   (f", δ={delta:.2e}" if delta else ""))
            
            return result_df
            
        except Exception as e:
            st.error(f"Error applying differential privacy: {str(e)}")
            return df_input

    def get_privacy_metrics(self, original_df: pd.DataFrame, anonymized_df: pd.DataFrame, 
                          config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calculate privacy and utility metrics for standard differential privacy.
        """
        if original_df.empty or anonymized_df.empty:
            return {}
            
        try:
            # Get basic DP metrics
            dp_metrics = self.dp_core.get_privacy_metrics()
            
            # Calculate utility metrics
            columns = config.get("columns", [])
            epsilon = config.get("epsilon", 1.0)
            mechanism = config.get("mechanism", "laplace")
            
            metrics = {
                "privacy_metrics": {
                    "privacy_budget_used": dp_metrics.get("total_epsilon_used", 0),
                    "epsilon": epsilon,
                    "delta": config.get("delta"),
                    "mechanism": mechanism,
                    "privacy_level": self.dp_core._classify_privacy_level(epsilon),
                    "operations_count": dp_metrics.get("number_of_operations", 0)
                },
                "utility_metrics": {},
                "data_metrics": {
                    "original_rows": len(original_df),
                    "anonymized_rows": len(anonymized_df),
                    "columns_modified": len(columns),
                    "total_columns": len(original_df.columns)
                }
            }
            
            # Calculate utility metrics for numeric columns
            numeric_cols = [col for col in columns if col in original_df.columns 
                          and pd.api.types.is_numeric_dtype(original_df[col])]
            
            if numeric_cols:
                utility_metrics = {}
                
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
                            
                            # Signal-to-Noise Ratio
                            signal_power = np.var(orig_vals)
                            noise_power = np.var(orig_vals.values - anon_vals.values[:len(orig_vals)])
                            snr = 10 * np.log10(signal_power / noise_power) if noise_power > 0 else float('inf')
                            
                            utility_metrics[col] = {
                                "mean_absolute_error": float(mae),
                                "relative_error": float(relative_error),
                                "signal_to_noise_ratio_db": float(snr) if not np.isinf(snr) else "∞",
                                "original_mean": float(orig_vals.mean()),
                                "anonymized_mean": float(anon_vals.mean()),
                                "original_std": float(orig_vals.std()),
                                "anonymized_std": float(anon_vals.std())
                            }
                
                metrics["utility_metrics"] = utility_metrics
                
                # Overall utility score
                if utility_metrics:
                    avg_relative_error = np.mean([m["relative_error"] for m in utility_metrics.values()])
                    utility_score = max(0, 1 - avg_relative_error)
                    metrics["overall_utility_score"] = float(utility_score)
            
            return metrics
            
        except Exception as e:
            st.warning(f"Could not calculate all metrics: {str(e)}")
            return {"error": str(e)}

    def build_config_export(self, unique_key_prefix: str, sa_col: str | None) -> Dict[str, Any]:
        """
        Build the configuration export for standard differential privacy.
        """
        return {
            "columns": st.session_state.get(f"{unique_key_prefix}_dp_cols", []),
            "epsilon": st.session_state.get(f"{unique_key_prefix}_epsilon", 1.0),
            "delta": st.session_state.get(f"{unique_key_prefix}_delta", 1e-5),
            "mechanism": st.session_state.get(f"{unique_key_prefix}_mechanism", "laplace"),
            "operation": st.session_state.get(f"{unique_key_prefix}_operation", "identity"),
            "budget_tracking": st.session_state.get(f"{unique_key_prefix}_budget_tracking", True),
            "show_metrics": st.session_state.get(f"{unique_key_prefix}_show_metrics", True)
        }

    def apply_config_import(self, config_params: Dict[str, Any], all_cols: List[str], unique_key_prefix: str):
        """
        Apply imported configuration to session state for standard differential privacy.
        """
        # Validate and set columns
        imported_cols = config_params.get("columns", [])
        valid_cols = [col for col in imported_cols if col in all_cols]
        st.session_state[f"{unique_key_prefix}_dp_cols"] = valid_cols
        
        # Set other parameters with defaults
        st.session_state[f"{unique_key_prefix}_epsilon"] = config_params.get("epsilon", 1.0)
        st.session_state[f"{unique_key_prefix}_delta"] = config_params.get("delta", 1e-5)
        st.session_state[f"{unique_key_prefix}_mechanism"] = config_params.get("mechanism", "laplace")
        st.session_state[f"{unique_key_prefix}_operation"] = config_params.get("operation", "identity")
        st.session_state[f"{unique_key_prefix}_budget_tracking"] = config_params.get("budget_tracking", True)
        st.session_state[f"{unique_key_prefix}_show_metrics"] = config_params.get("show_metrics", True)

    def get_export_button_ui(self, config_to_export: dict, unique_key_prefix: str):
        """Export button UI for standard differential privacy configuration."""
        json_string = json.dumps(config_to_export, indent=4)
        st.sidebar.download_button(
            label=f"Export {self.get_name()} Config",
            data=json_string,
            file_name=f"{self.get_name().lower().replace(' ', '_')}_config.json",
            mime="application/json",
            key=f"{unique_key_prefix}_export_button"
        )

    def get_anonymize_button_ui(self, unique_key_prefix: str) -> bool:
        """Anonymize button UI for standard differential privacy."""
        return st.button(f"Anonymize with {self.get_name()}", key=f"{unique_key_prefix}_anonymize_button")

    def export_config(self, config: Dict[str, Any]) -> str:
        """Export the current configuration as JSON string."""
        export_config = {
            "anonymizer": self.get_name(),
            "version": "1.0",
            "parameters": config,
            "privacy_budget_used": self.dp_core.privacy_budget_used,
            "timestamp": pd.Timestamp.now().isoformat()
        }
        return json.dumps(export_config, indent=2)

    def import_config(self, config_str: str) -> Dict[str, Any]:
        """Import configuration from JSON string."""
        try:
            config_data = json.loads(config_str)
            if config_data.get("anonymizer") != self.get_name():
                raise ValueError("Configuration is not for Standard Differential Privacy")
            
            params = config_data.get("parameters", {})
            
            # Restore privacy budget if available
            if "privacy_budget_used" in config_data:
                self.dp_core.privacy_budget_used = config_data["privacy_budget_used"]
            
            return params
            
        except json.JSONDecodeError:
            raise ValueError("Invalid JSON configuration")
        except Exception as e:
            raise ValueError(f"Error importing configuration: {str(e)}")

    def reset_privacy_budget(self):
        """Reset the privacy budget tracker."""
        self.dp_core.reset_privacy_budget()
        st.success("Privacy budget reset successfully!")

# Create plugin instance
def get_plugin():
    """Factory function to create plugin instance."""
    return StandardDifferentialPrivacyPlugin()
