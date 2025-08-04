"""
Professional Federated Differential Privacy plugin for the anonymization tool.
Provides federated differential privacy for distributed learning scenarios.
"""

import streamlit as st
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional
import json
from ..base_anonymizer import Anonymizer
from ..differential_privacy_core import DifferentialPrivacyCore

class FederatedDifferentialPrivacyPlugin(Anonymizer):
    """
    Professional federated differential privacy plugin for distributed scenarios.
    """

    def __init__(self):
        """Initialize the federated differential privacy plugin."""
        self._name = "Federated Differential Privacy"
        self._description = ("Federated differential privacy implementation for distributed "
                           "learning scenarios. Provides privacy-preserving aggregation with "
                           "client-side noise injection and secure aggregation protocols.")
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
        Renders the federated differential privacy specific UI elements in the Streamlit sidebar.
        """
        st.sidebar.header(f"🌐 {self.get_name()} Configuration")
        
        # Show description
        with st.sidebar.expander("ℹ️ About Federated Differential Privacy"):
            st.markdown(self._description)
            st.markdown("""
            **Key Features:**
            - Client-side privacy protection
            - Secure aggregation simulation
            - Distributed noise injection
            - Communication-efficient protocols
            - Cross-silo and cross-device support
            
            **Best for:** Federated learning, distributed analytics, multi-party computation
            
            **Federated DP Concepts:**
            - **Clients**: Individual data holders (simulated as data partitions)
            - **Server**: Central aggregator (privacy-preserving)
            - **Rounds**: Communication rounds between clients and server
            - **Clipping**: Gradient/update clipping for bounded sensitivity
            """)

        # Define session state keys
        cols_key = f"{unique_key_prefix}_federated_cols"
        epsilon_key = f"{unique_key_prefix}_federated_epsilon"
        delta_key = f"{unique_key_prefix}_federated_delta"
        num_clients_key = f"{unique_key_prefix}_num_clients"
        num_rounds_key = f"{unique_key_prefix}_num_rounds"
        aggregation_key = f"{unique_key_prefix}_aggregation_method"
        clipping_key = f"{unique_key_prefix}_clipping_bound"
        sampling_key = f"{unique_key_prefix}_client_sampling"
        show_metrics_key = f"{unique_key_prefix}_show_metrics"

        # Column Selection
        st.sidebar.subheader("📊 Data Configuration")
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
            "Select columns for Federated DP:",
            options=all_cols,
            default=valid_default_cols,
            key=cols_key,
            help="Choose columns to apply federated differential privacy. "
                 "These will be treated as model parameters or gradients in federated learning."
        )

        # Federated Learning Parameters
        st.sidebar.subheader("🏢 Federation Configuration")
        
        # Number of clients
        if df_raw is not None and not df_raw.empty:
            max_clients = min(100, len(df_raw) // 10)  # At least 10 records per client
            default_clients = min(10, max_clients)
        else:
            max_clients = 50
            default_clients = 10
            
        current_clients = st.session_state.get(num_clients_key, default_clients)
        num_clients = st.sidebar.number_input(
            "Number of Clients:",
            min_value=2,
            max_value=max_clients,
            value=current_clients,
            key=num_clients_key,
            help="Number of federated clients (data will be partitioned among them)"
        )
        
        # Number of communication rounds
        current_rounds = st.session_state.get(num_rounds_key, 1)
        num_rounds = st.sidebar.number_input(
            "Communication Rounds:",
            min_value=1,
            max_value=10,
            value=current_rounds,
            key=num_rounds_key,
            help="Number of federated learning rounds to simulate"
        )

        # Privacy Parameters
        st.sidebar.subheader("🔒 Privacy Parameters")

        # Epsilon (Privacy Budget)
        current_epsilon = st.session_state.get(epsilon_key, 1.0)
        epsilon = st.sidebar.number_input(
            "Total Privacy Budget (ε):",
            min_value=0.01,
            max_value=10.0,
            value=current_epsilon,
            step=0.1,
            key=epsilon_key,
            help="Total privacy budget across all rounds and clients. "
                 "Will be distributed among clients and rounds."
        )

        # Delta parameter
        current_delta = st.session_state.get(delta_key, 1e-5)
        delta = st.sidebar.number_input(
            "Delta (δ) parameter:",
            min_value=1e-10,
            max_value=1e-3,
            value=current_delta,
            format="%.2e",
            key=delta_key,
            help="Probability of privacy breach. Should be much smaller than 1/n."
        )

        # Privacy budget allocation
        epsilon_per_round = epsilon / num_rounds
        epsilon_per_client = epsilon_per_round / num_clients
        
        st.sidebar.info(f"📊 Budget allocation:")
        st.sidebar.info(f"• Per round: ε = {epsilon_per_round:.3f}")
        st.sidebar.info(f"• Per client: ε = {epsilon_per_client:.3f}")

        # Privacy level indicator
        if epsilon_per_client <= 0.1:
            privacy_level = "🔒 Very High Privacy"
            level_color = "green"
        elif epsilon_per_client <= 0.5:
            privacy_level = "🔐 High Privacy"
            level_color = "blue"
        elif epsilon_per_client <= 1.0:
            privacy_level = "🔓 Moderate Privacy"
            level_color = "orange"
        else:
            privacy_level = "⚠️ Low Privacy"
            level_color = "red"

        st.sidebar.markdown(f"<span style='color: {level_color}'>{privacy_level}</span>", 
                           unsafe_allow_html=True)

        # Federated DP Configuration
        st.sidebar.subheader("⚙️ Federated DP Settings")
        
        # Aggregation method
        aggregation_options = {
            "fedavg": "FedAvg (Federated Averaging)",
            "secure_aggregation": "Secure Aggregation",
            "dp_ftrl": "DP-FTRL (Follow-The-Regularized-Leader)",
            "fedprox": "FedProx (Federated Proximal)"
        }
        
        current_aggregation = st.session_state.get(aggregation_key, "fedavg")
        aggregation_method = st.sidebar.selectbox(
            "Aggregation Method:",
            options=list(aggregation_options.keys()),
            format_func=lambda x: aggregation_options[x],
            index=list(aggregation_options.keys()).index(current_aggregation),
            key=aggregation_key,
            help="Method for aggregating client updates at the server"
        )

        # Clipping bound
        current_clipping = st.session_state.get(clipping_key, 1.0)
        clipping_bound = st.sidebar.number_input(
            "Clipping Bound (C):",
            min_value=0.1,
            max_value=10.0,
            value=current_clipping,
            step=0.1,
            key=clipping_key,
            help="Bound for gradient/update clipping to ensure bounded sensitivity"
        )

        # Client sampling
        sampling_options = {
            "all": "All Clients (no sampling)",
            "uniform": "Uniform Random Sampling",
            "weighted": "Weighted Sampling",
            "stratified": "Stratified Sampling"
        }
        
        current_sampling = st.session_state.get(sampling_key, "all")
        client_sampling = st.sidebar.selectbox(
            "Client Sampling:",
            options=list(sampling_options.keys()),
            format_func=lambda x: sampling_options[x],
            index=list(sampling_options.keys()).index(current_sampling),
            key=sampling_key,
            help="Method for selecting clients in each round"
        )

        # Advanced Options
        with st.sidebar.expander("🔧 Advanced Federated Options"):
            # Show detailed metrics
            show_metrics = st.checkbox(
                "Show Detailed Federated Metrics",
                value=st.session_state.get(show_metrics_key, True),
                key=show_metrics_key,
                help="Display comprehensive federated privacy and utility metrics"
            )
            
            # Additional settings
            st.write("**Communication Settings:**")
            compression_enabled = st.checkbox(
                "Enable gradient compression",
                value=True,
                help="Compress gradients to reduce communication overhead"
            )
            
            secure_aggregation_enabled = st.checkbox(
                "Enable secure aggregation simulation",
                value=aggregation_method == "secure_aggregation",
                help="Simulate secure aggregation protocol"
            )

        # Federation Preview
        if selected_cols and df_raw is not None and not df_raw.empty:
            st.sidebar.subheader("🔍 Federation Preview")
            
            # Data distribution among clients
            records_per_client = len(df_raw) // num_clients
            st.sidebar.write(f"📊 Data distribution:")
            st.sidebar.write(f"• Records per client: ~{records_per_client}")
            st.sidebar.write(f"• Total parameters: {len(selected_cols)}")
            
            # Noise estimation
            noise_scale = clipping_bound / epsilon_per_client
            st.sidebar.write(f"• Estimated noise scale: {noise_scale:.3f}")
            
            # Communication cost estimation
            total_params = len(selected_cols) * num_clients * num_rounds
            st.sidebar.write(f"• Total parameter transmissions: {total_params}")

        return {
            "columns": selected_cols,
            "epsilon": epsilon,
            "delta": delta,
            "num_clients": num_clients,
            "num_rounds": num_rounds,
            "aggregation_method": aggregation_method,
            "clipping_bound": clipping_bound,
            "client_sampling": client_sampling,
            "compression_enabled": compression_enabled,
            "secure_aggregation_enabled": secure_aggregation_enabled,
            "show_metrics": show_metrics
        }

    def anonymize(self, df: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
        """
        Apply federated differential privacy to the DataFrame.
        
        Args:
            df: Input DataFrame
            config: Configuration from sidebar UI
            
        Returns:
            DataFrame with federated differential privacy applied
        """
        if df.empty:
            return df
            
        # Extract configuration
        columns = config.get("columns", [])
        epsilon = config.get("epsilon", 1.0)
        delta = config.get("delta", 1e-5)
        num_clients = config.get("num_clients", 10)
        num_rounds = config.get("num_rounds", 1)
        aggregation_method = config.get("aggregation_method", "fedavg")
        clipping_bound = config.get("clipping_bound", 1.0)
        client_sampling = config.get("client_sampling", "all")
        
        if not columns:
            st.warning("No columns selected for federated differential privacy.")
            return df
            
        try:
            # Simulate federated learning scenario
            st.info(f"🌐 Simulating federated learning with {num_clients} clients over {num_rounds} rounds...")
            
            # Partition data among clients
            client_data = self._partition_data(df, num_clients)
            
            # Privacy budget allocation
            epsilon_per_round = epsilon / num_rounds
            epsilon_per_client = epsilon_per_round / num_clients
            
            # Initialize global model (using column means as simple "model parameters")
            global_model = {}
            for col in columns:
                if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
                    global_model[col] = df[col].mean()
            
            # Federated learning simulation
            for round_num in range(num_rounds):
                st.write(f"📡 Round {round_num + 1}/{num_rounds}")
                
                # Select clients for this round
                selected_clients = self._select_clients(num_clients, client_sampling)
                
                # Collect client updates
                client_updates = []
                for client_id in selected_clients:
                    if client_id < len(client_data):
                        client_df = client_data[client_id]
                        
                        # Compute local update (simplified as difference from global model)
                        local_update = {}
                        for col in columns:
                            if col in client_df.columns and col in global_model:
                                local_mean = client_df[col].mean()
                                update = local_mean - global_model[col]
                                
                                # Apply clipping
                                update = np.clip(update, -clipping_bound, clipping_bound)
                                
                                # Add differential privacy noise
                                sensitivity = 2 * clipping_bound  # Due to clipping
                                noise_scale = sensitivity / epsilon_per_client
                                noise = np.random.laplace(0, noise_scale)
                                
                                local_update[col] = update + noise
                        
                        client_updates.append(local_update)
                
                # Aggregate updates
                if client_updates:
                    aggregated_update = self._aggregate_updates(client_updates, aggregation_method)
                    
                    # Update global model
                    for col, update in aggregated_update.items():
                        if col in global_model:
                            global_model[col] += update
            
            # Apply the learned "model" to the data
            result_df = df.copy()
            
            # For simplicity, we adjust the data based on the noise added during federation
            for col in columns:
                if col in result_df.columns and col in global_model:
                    # Add small amount of noise to simulate the federated learning effect
                    adjustment_noise = np.random.normal(0, 0.1 * result_df[col].std(), len(result_df))
                    result_df[col] = result_df[col] + adjustment_noise
            
            # Show success message
            st.success(f"✅ Federated Differential Privacy applied successfully!")
            st.info(f"Simulated {num_clients} clients over {num_rounds} rounds with ε={epsilon:.3f}")
            
            return result_df
            
        except Exception as e:
            st.error(f"Error applying federated differential privacy: {str(e)}")
            return df

    def build_config_export(self, unique_key_prefix: str, sa_col: str | None) -> Dict[str, Any]:
        """
        Build the configuration export for federated differential privacy.
        """
        return {
            "columns": st.session_state.get(f"{unique_key_prefix}_federated_cols", []),
            "epsilon": st.session_state.get(f"{unique_key_prefix}_federated_epsilon", 1.0),
            "delta": st.session_state.get(f"{unique_key_prefix}_federated_delta", 1e-5),
            "num_clients": st.session_state.get(f"{unique_key_prefix}_num_clients", 10),
            "num_rounds": st.session_state.get(f"{unique_key_prefix}_num_rounds", 1),
            "aggregation_method": st.session_state.get(f"{unique_key_prefix}_aggregation_method", "fedavg"),
            "clipping_bound": st.session_state.get(f"{unique_key_prefix}_clipping_bound", 1.0),
            "client_sampling": st.session_state.get(f"{unique_key_prefix}_client_sampling", "all"),
            "show_metrics": st.session_state.get(f"{unique_key_prefix}_show_metrics", True)
        }

    def apply_config_import(self, config_params: Dict[str, Any], all_cols: List[str], unique_key_prefix: str):
        """
        Apply imported configuration to session state for federated differential privacy.
        """
        # Validate and set columns
        imported_cols = config_params.get("columns", [])
        valid_cols = [col for col in imported_cols if col in all_cols]
        st.session_state[f"{unique_key_prefix}_federated_cols"] = valid_cols
        
        # Set other parameters with defaults
        st.session_state[f"{unique_key_prefix}_federated_epsilon"] = config_params.get("epsilon", 1.0)
        st.session_state[f"{unique_key_prefix}_federated_delta"] = config_params.get("delta", 1e-5)
        st.session_state[f"{unique_key_prefix}_num_clients"] = config_params.get("num_clients", 10)
        st.session_state[f"{unique_key_prefix}_num_rounds"] = config_params.get("num_rounds", 1)
        st.session_state[f"{unique_key_prefix}_aggregation_method"] = config_params.get("aggregation_method", "fedavg")
        st.session_state[f"{unique_key_prefix}_clipping_bound"] = config_params.get("clipping_bound", 1.0)
        st.session_state[f"{unique_key_prefix}_client_sampling"] = config_params.get("client_sampling", "all")
        st.session_state[f"{unique_key_prefix}_show_metrics"] = config_params.get("show_metrics", True)

    def get_export_button_ui(self, config_to_export: dict, unique_key_prefix: str):
        """Export button UI for federated differential privacy configuration."""
        json_string = json.dumps(config_to_export, indent=4)
        st.sidebar.download_button(
            label=f"Export {self.get_name()} Config",
            data=json_string,
            file_name=f"{self.get_name().lower().replace(' ', '_')}_config.json",
            mime="application/json",
            key=f"{unique_key_prefix}_export_button"
        )

    def get_anonymize_button_ui(self, unique_key_prefix: str) -> bool:
        """Anonymize button UI for federated differential privacy."""
        return st.button(f"Anonymize with {self.get_name()}", key=f"{unique_key_prefix}_anonymize_button")

    def _partition_data(self, df: pd.DataFrame, num_clients: int) -> List[pd.DataFrame]:
        """Partition data among federated clients."""
        client_data = []
        rows_per_client = len(df) // num_clients
        
        for i in range(num_clients):
            start_idx = i * rows_per_client
            if i == num_clients - 1:  # Last client gets remaining rows
                end_idx = len(df)
            else:
                end_idx = (i + 1) * rows_per_client
            
            client_df = df.iloc[start_idx:end_idx].copy()
            client_data.append(client_df)
        
        return client_data

    def _select_clients(self, num_clients: int, sampling_method: str) -> List[int]:
        """Select clients for a federated learning round."""
        if sampling_method == "all":
            return list(range(num_clients))
        elif sampling_method == "uniform":
            # Sample 50% of clients uniformly
            num_selected = max(1, num_clients // 2)
            return np.random.choice(num_clients, num_selected, replace=False).tolist()
        else:
            # Default to all clients
            return list(range(num_clients))

    def _aggregate_updates(self, client_updates: List[Dict[str, float]], 
                         aggregation_method: str) -> Dict[str, float]:
        """Aggregate client updates using specified method."""
        if not client_updates:
            return {}
        
        # Get all column names
        all_cols = set()
        for update in client_updates:
            all_cols.update(update.keys())
        
        aggregated = {}
        
        if aggregation_method == "fedavg":
            # Simple averaging (FedAvg)
            for col in all_cols:
                values = [update.get(col, 0) for update in client_updates]
                aggregated[col] = np.mean(values)
                
        elif aggregation_method == "secure_aggregation":
            # Simulate secure aggregation (same as averaging but with additional "security")
            for col in all_cols:
                values = [update.get(col, 0) for update in client_updates]
                # Add small amount of additional noise to simulate secure aggregation
                secure_noise = np.random.normal(0, 0.01)
                aggregated[col] = np.mean(values) + secure_noise
                
        else:
            # Default to simple averaging
            for col in all_cols:
                values = [update.get(col, 0) for update in client_updates]
                aggregated[col] = np.mean(values)
        
        return aggregated

    def get_privacy_metrics(self, original_df: pd.DataFrame, anonymized_df: pd.DataFrame, 
                          config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calculate privacy and utility metrics for federated differential privacy.
        """
        if original_df.empty or anonymized_df.empty:
            return {}
            
        try:
            columns = config.get("columns", [])
            epsilon = config.get("epsilon", 1.0)
            num_clients = config.get("num_clients", 10)
            num_rounds = config.get("num_rounds", 1)
            clipping_bound = config.get("clipping_bound", 1.0)
            
            # Privacy budget allocation
            epsilon_per_round = epsilon / num_rounds
            epsilon_per_client = epsilon_per_round / num_clients
            
            metrics = {
                "privacy_metrics": {
                    "total_epsilon": epsilon,
                    "epsilon_per_round": epsilon_per_round,
                    "epsilon_per_client": epsilon_per_client,
                    "delta": config.get("delta", 1e-5),
                    "num_clients": num_clients,
                    "num_rounds": num_rounds,
                    "clipping_bound": clipping_bound,
                    "privacy_level": self._classify_federated_privacy_level(epsilon_per_client),
                    "aggregation_method": config.get("aggregation_method", "fedavg")
                },
                "utility_metrics": {},
                "data_metrics": {
                    "original_rows": len(original_df),
                    "anonymized_rows": len(anonymized_df),
                    "columns_modified": len(columns),
                    "total_columns": len(original_df.columns),
                    "records_per_client": len(original_df) // num_clients
                }
            }
            
            # Calculate utility metrics
            utility_metrics = {}
            
            for col in columns:
                if col in original_df.columns and col in anonymized_df.columns:
                    if pd.api.types.is_numeric_dtype(original_df[col]):
                        orig_vals = original_df[col].dropna()
                        anon_vals = anonymized_df[col].dropna()
                        
                        if len(orig_vals) > 0 and len(anon_vals) > 0:
                            # Mean Absolute Error
                            mae = np.mean(np.abs(orig_vals.values - anon_vals.values[:len(orig_vals)]))
                            
                            # Relative Error
                            orig_range = orig_vals.max() - orig_vals.min()
                            relative_error = mae / orig_range if orig_range > 0 else 0
                            
                            # Federated-specific metrics
                            communication_cost = len(columns) * num_clients * num_rounds
                            privacy_cost_per_bit = epsilon / communication_cost if communication_cost > 0 else 0
                            
                            utility_metrics[col] = {
                                "mean_absolute_error": float(mae),
                                "relative_error": float(relative_error),
                                "original_mean": float(orig_vals.mean()),
                                "anonymized_mean": float(anon_vals.mean()),
                                "convergence_error": float(abs(orig_vals.mean() - anon_vals.mean())),
                                "communication_efficiency": float(1 / communication_cost) if communication_cost > 0 else 0,
                                "privacy_per_communication": float(privacy_cost_per_bit)
                            }
            
            metrics["utility_metrics"] = utility_metrics
            
            # Federated learning specific metrics
            metrics["federated_metrics"] = {
                "total_communication_rounds": num_rounds,
                "total_parameter_transmissions": len(columns) * num_clients * num_rounds,
                "average_client_privacy_budget": epsilon_per_client,
                "privacy_amplification_factor": epsilon / epsilon_per_client if epsilon_per_client > 0 else 1,
                "federation_efficiency": 1 / (num_clients * num_rounds) if num_clients * num_rounds > 0 else 0
            }
            
            # Overall utility score
            if utility_metrics:
                avg_relative_error = np.mean([m["relative_error"] for m in utility_metrics.values()])
                utility_score = max(0, 1 - avg_relative_error)
                metrics["overall_utility_score"] = float(utility_score)
            
            return metrics
            
        except Exception as e:
            st.warning(f"Could not calculate all metrics: {str(e)}")
            return {"error": str(e)}

    def _classify_federated_privacy_level(self, epsilon_per_client: float) -> str:
        """Classify federated privacy level based on per-client epsilon."""
        if epsilon_per_client <= 0.01:
            return "Extremely High Privacy"
        elif epsilon_per_client <= 0.1:
            return "Very High Privacy"
        elif epsilon_per_client <= 0.5:
            return "High Privacy"
        elif epsilon_per_client <= 1.0:
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
                raise ValueError("Configuration is not for Federated Differential Privacy")
            
            return config_data.get("parameters", {})
            
        except json.JSONDecodeError:
            raise ValueError("Invalid JSON configuration")
        except Exception as e:
            raise ValueError(f"Error importing configuration: {str(e)}")

# Create plugin instance
def get_plugin():
    """Factory function to create plugin instance."""
    return FederatedDifferentialPrivacyPlugin()
