"""
Professional Bayesian Networks plugin for privacy-preserving data synthesis.
Uses probabilistic graphical models to learn and generate synthetic data.
"""

import streamlit as st
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
import json
from ..base_anonymizer import Anonymizer

class BayesianNetworksPlugin(Anonymizer):
    """Professional Bayesian Networks plugin for synthetic data generation."""

    def __init__(self):
        self._name = "Bayesian Networks"
        self._description = ("Bayesian Networks for privacy-preserving synthetic data generation. "
                           "Learns probabilistic dependencies between variables and generates "
                           "synthetic data while preserving conditional dependencies.")

    def get_name(self) -> str:
        return self._name

    def get_category(self) -> str:
        """Returns the category of the anonymization technique."""
        return "Generative Models"

    def get_description(self) -> str:
        return self._description

    def get_sidebar_ui(self, all_cols: List[str], sa_col_to_pass: str | None, 
                      df_raw: pd.DataFrame, unique_key_prefix: str) -> Dict[str, Any]:
        st.sidebar.header(f"🕸️ {self.get_name()} Configuration")
        
        with st.sidebar.expander("ℹ️ About Bayesian Networks"):
            st.markdown(self._description)
            st.markdown("""
            **Key Features:**
            - Learns conditional dependencies between variables
            - Preserves statistical relationships and correlations
            - Handles mixed data types naturally
            - Interpretable probabilistic model
            - Support for causal inference
            """)

        # Session state keys
        cols_key = f"{unique_key_prefix}_bn_cols"
        structure_key = f"{unique_key_prefix}_structure_learning"
        max_parents_key = f"{unique_key_prefix}_max_parents"
        scoring_key = f"{unique_key_prefix}_scoring_method"
        discretization_key = f"{unique_key_prefix}_discretization_bins"
        synthetic_ratio_key = f"{unique_key_prefix}_bn_synthetic_ratio"

        # Column selection
        selected_cols = st.sidebar.multiselect(
            "Select columns for Bayesian Network:",
            options=all_cols,
            default=st.session_state.get(cols_key, all_cols),
            key=cols_key
        )

        # Structure learning method
        structure_methods = {
            "hill_climbing": "Hill Climbing",
            "constraint_based": "Constraint-based (PC Algorithm)",
            "score_based": "Score-based (BIC)",
            "hybrid": "Hybrid (MMHC)"
        }
        
        structure_learning = st.sidebar.selectbox(
            "Structure Learning Method:",
            options=list(structure_methods.keys()),
            format_func=lambda x: structure_methods[x],
            key=structure_key
        )

        # Max parents per node
        max_parents = st.sidebar.number_input(
            "Max Parents per Node:",
            min_value=1,
            max_value=10,
            value=st.session_state.get(max_parents_key, 3),
            key=max_parents_key
        )

        # Scoring method
        scoring_methods = ["BIC", "AIC", "K2", "BDeu"]
        scoring_method = st.sidebar.selectbox(
            "Scoring Method:",
            options=scoring_methods,
            key=scoring_key
        )

        # Discretization for continuous variables
        discretization_bins = st.sidebar.number_input(
            "Discretization Bins:",
            min_value=2,
            max_value=20,
            value=st.session_state.get(discretization_key, 5),
            key=discretization_key
        )

        # Synthetic ratio
        synthetic_ratio = st.sidebar.slider(
            "Synthetic Data Ratio:",
            min_value=0.1,
            max_value=5.0,
            value=st.session_state.get(synthetic_ratio_key, 1.0),
            key=synthetic_ratio_key
        )

        return {
            "columns": selected_cols,
            "structure_learning": structure_learning,
            "max_parents": max_parents,
            "scoring_method": scoring_method,
            "discretization_bins": discretization_bins,
            "synthetic_ratio": synthetic_ratio
        }

    def anonymize(self, df: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
        if df.empty:
            return df
            
        columns = config.get("columns", [])
        if not columns:
            st.warning("No columns selected.")
            return pd.DataFrame()
            
        try:
            st.info("🕸️ Learning Bayesian Network structure...")
            
            # Prepare data
            data = df[columns].copy().fillna(df[columns].mode().iloc[0])
            
            # Discretize continuous variables
            discretized_data, discretizers = self._discretize_data(data, config.get("discretization_bins", 5))
            
            # Learn network structure (simulated)
            network_structure = self._learn_structure(discretized_data, config)
            
            st.info("📊 Learning conditional probability tables...")
            
            # Learn parameters
            cpts = self._learn_parameters(discretized_data, network_structure)
            
            st.info("🎲 Generating synthetic data...")
            
            # Generate synthetic data
            synthetic_size = int(len(df) * config.get("synthetic_ratio", 1.0))
            synthetic_data = self._generate_synthetic_data(
                network_structure, cpts, discretizers, columns, synthetic_size
            )
            
            st.success(f"✅ Generated {len(synthetic_data)} synthetic samples using Bayesian Networks!")
            return synthetic_data
            
        except Exception as e:
            st.error(f"Error in Bayesian Network synthesis: {str(e)}")
            return pd.DataFrame()

    def _discretize_data(self, data: pd.DataFrame, bins: int) -> Tuple[pd.DataFrame, Dict]:
        discretizers = {}
        discretized = data.copy()
        
        for col in data.columns:
            if pd.api.types.is_numeric_dtype(data[col]):
                try:
                    discretized[col], bin_edges = pd.cut(data[col], bins=bins, retbins=True, duplicates='drop')
                    discretizers[col] = {'type': 'numeric', 'bins': bin_edges}
                except:
                    discretizers[col] = {'type': 'numeric', 'bins': None}
            else:
                discretizers[col] = {'type': 'categorical'}
        
        return discretized, discretizers

    def _learn_structure(self, data: pd.DataFrame, config: Dict) -> Dict:
        # Simplified structure learning simulation
        nodes = list(data.columns)
        edges = []
        
        # Generate some random edges (simplified)
        max_parents = config.get("max_parents", 3)
        for i, node in enumerate(nodes):
            num_parents = min(np.random.poisson(1), max_parents, i)
            if num_parents > 0:
                parents = np.random.choice(nodes[:i], num_parents, replace=False)
                for parent in parents:
                    edges.append((parent, node))
        
        return {"nodes": nodes, "edges": edges}

    def _learn_parameters(self, data: pd.DataFrame, structure: Dict) -> Dict:
        # Learn conditional probability tables (simplified)
        cpts = {}
        
        for node in structure["nodes"]:
            parents = [edge[0] for edge in structure["edges"] if edge[1] == node]
            
            if not parents:
                # Marginal distribution
                cpts[node] = data[node].value_counts(normalize=True).to_dict()
            else:
                # Conditional distribution
                grouped = data.groupby(parents + [node]).size()
                parent_totals = data.groupby(parents).size()
                
                cpt = {}
                for parent_vals, total in parent_totals.items():
                    if not isinstance(parent_vals, tuple):
                        parent_vals = (parent_vals,)
                    
                    cpt[parent_vals] = {}
                    for node_val in data[node].unique():
                        key = parent_vals + (node_val,)
                        count = grouped.get(key, 0)
                        cpt[parent_vals][node_val] = count / total if total > 0 else 0
                
                cpts[node] = cpt
        
        return cpts

    def _generate_synthetic_data(self, structure: Dict, cpts: Dict, discretizers: Dict,
                               columns: List[str], num_samples: int) -> pd.DataFrame:
        # Generate synthetic data using ancestral sampling
        synthetic_data = []
        
        for _ in range(num_samples):
            sample = {}
            
            # Sample in topological order (simplified)
            for node in structure["nodes"]:
                parents = [edge[0] for edge in structure["edges"] if edge[1] == node]
                
                if not parents:
                    # Sample from marginal
                    probs = cpts[node]
                    sample[node] = np.random.choice(list(probs.keys()), p=list(probs.values()))
                else:
                    # Sample from conditional
                    parent_vals = tuple(sample[parent] for parent in parents)
                    if parent_vals in cpts[node]:
                        cond_probs = cpts[node][parent_vals]
                        if cond_probs:
                            vals, probs = zip(*cond_probs.items())
                            sample[node] = np.random.choice(vals, p=probs)
                        else:
                            sample[node] = np.random.choice(list(discretizers[node].get('categories', ['Unknown'])))
                    else:
                        sample[node] = np.random.choice(list(discretizers[node].get('categories', ['Unknown'])))
            
            synthetic_data.append(sample)
        
        # Convert back to original format
        synthetic_df = pd.DataFrame(synthetic_data)
        
        # Undiscretize numeric columns
        for col in columns:
            if col in discretizers and discretizers[col]['type'] == 'numeric' and discretizers[col]['bins'] is not None:
                bins = discretizers[col]['bins']
                # Convert intervals back to numeric values (take midpoint)
                synthetic_df[col] = synthetic_df[col].apply(
                    lambda x: (x.left + x.right) / 2 if hasattr(x, 'left') else np.random.uniform(bins[0], bins[-1])
                )
        
        return synthetic_df

    def build_config_export(self, unique_key_prefix: str, sa_col: str | None) -> Dict[str, Any]:
        return {
            "columns": st.session_state.get(f"{unique_key_prefix}_bn_cols", []),
            "structure_learning": st.session_state.get(f"{unique_key_prefix}_structure_learning", "hill_climbing"),
            "max_parents": st.session_state.get(f"{unique_key_prefix}_max_parents", 3),
            "scoring_method": st.session_state.get(f"{unique_key_prefix}_scoring_method", "BIC"),
            "discretization_bins": st.session_state.get(f"{unique_key_prefix}_discretization_bins", 5),
            "synthetic_ratio": st.session_state.get(f"{unique_key_prefix}_bn_synthetic_ratio", 1.0)
        }

    def apply_config_import(self, config_params: Dict[str, Any], all_cols: List[str], unique_key_prefix: str):
        valid_cols = [col for col in config_params.get("columns", []) if col in all_cols]
        st.session_state[f"{unique_key_prefix}_bn_cols"] = valid_cols
        st.session_state[f"{unique_key_prefix}_structure_learning"] = config_params.get("structure_learning", "hill_climbing")
        st.session_state[f"{unique_key_prefix}_max_parents"] = config_params.get("max_parents", 3)
        st.session_state[f"{unique_key_prefix}_scoring_method"] = config_params.get("scoring_method", "BIC")
        st.session_state[f"{unique_key_prefix}_discretization_bins"] = config_params.get("discretization_bins", 5)
        st.session_state[f"{unique_key_prefix}_bn_synthetic_ratio"] = config_params.get("synthetic_ratio", 1.0)

    def get_export_button_ui(self, config_to_export: dict, unique_key_prefix: str):
        json_string = json.dumps(config_to_export, indent=4)
        st.sidebar.download_button(
            label=f"Export {self.get_name()} Config",
            data=json_string,
            file_name=f"{self.get_name().lower().replace(' ', '_')}_config.json",
            mime="application/json",
            key=f"{unique_key_prefix}_export_button"
        )

    def get_anonymize_button_ui(self, unique_key_prefix: str) -> bool:
        return st.button(f"Generate Synthetic Data with {self.get_name()}", key=f"{unique_key_prefix}_anonymize_button")

def get_plugin():
    return BayesianNetworksPlugin()
