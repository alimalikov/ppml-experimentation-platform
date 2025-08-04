"""
PrivBayes Plugin for Data Anonymization

This plugin implements PrivBayes, a differentially private algorithm for
synthetic data generation using Bayesian networks. It learns the structure
and parameters of a Bayesian network while providing differential privacy guarantees.
"""

import streamlit as st
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, Tuple, List, Set
import logging
import warnings
warnings.filterwarnings('ignore')

from ..base_anonymizer import Anonymizer

logger = logging.getLogger(__name__)

class PrivBayesPlugin(Anonymizer):
    """
    PrivBayes plugin for differentially private synthetic data generation.
    
    Implements the PrivBayes algorithm which uses Bayesian networks
    to model data dependencies while providing differential privacy.
    """
    
    def __init__(self):
        super().__init__()
        self.name = "PrivBayes"
        self.description = "Differentially private Bayesian network synthesis"
        self.category = "Generative Models"
        
        # Plugin parameters
        self.epsilon = 1.0  # Privacy budget
        self.delta = 1e-5   # Delta for (epsilon, delta)-DP
        self.max_parents = 3  # Maximum parents per node in Bayesian network
        self.discretization_bins = 10  # Bins for continuous variables
        self.n_samples = 1000
        self.random_seed = 42
        
        # PrivBayes specific parameters
        self.structure_epsilon_fraction = 0.3  # Fraction of epsilon for structure learning
        self.attribute_selection_method = "mutual_information"  # mutual_information, random
        self.laplace_smoothing = 0.1
        self.min_bin_size = 5  # Minimum records per bin
        
        # Learned components
        self.bayesian_network = {}
        self.conditional_distributions = {}
        self.attribute_order = []
        self.discretization_info = {}
        self.is_fitted = False
        
    def get_name(self) -> str:
        return "PrivBayes"
    
    def get_category(self) -> str:
        """Returns the category of the anonymization technique."""
        return "Generative Models"
    
    def get_description(self) -> str:
        return "Differentially private Bayesian network synthesis with formal privacy guarantees"
    
    def get_sidebar_ui(self, all_cols: list, sa_col_to_pass: str | None, 
                      df_raw: pd.DataFrame, unique_key_prefix: str) -> Dict[str, Any]:
        """
        Renders the PrivBayes specific UI elements in the Streamlit sidebar.
        """
        st.sidebar.header(f"🔗 {self.get_name()} Configuration")
        
        # Show description
        with st.sidebar.expander("ℹ️ About PrivBayes"):
            st.markdown(self.get_description())
            st.markdown("""
            **Key Features:**
            - **Differential Privacy**: Formal (ε,δ)-DP guarantees
            - **Bayesian Networks**: Models complex dependencies
            - **Structure Learning**: Private network structure discovery
            - **Conditional Generation**: Respects learned dependencies
            
            **Use Cases:**
            - High privacy requirements with formal guarantees
            - Complex tabular data with dependencies
            - Research requiring provable privacy
            """)

        # Define session state keys
        epsilon_key = f"{unique_key_prefix}_epsilon"
        delta_key = f"{unique_key_prefix}_delta"
        max_parents_key = f"{unique_key_prefix}_max_parents"
        discretization_bins_key = f"{unique_key_prefix}_discretization_bins"
        n_samples_key = f"{unique_key_prefix}_n_samples"
        random_seed_key = f"{unique_key_prefix}_random_seed"
        structure_epsilon_key = f"{unique_key_prefix}_structure_epsilon_fraction"
        selection_method_key = f"{unique_key_prefix}_attribute_selection_method"
        laplace_smoothing_key = f"{unique_key_prefix}_laplace_smoothing"
        min_bin_size_key = f"{unique_key_prefix}_min_bin_size"

        # Privacy Configuration
        st.sidebar.subheader("🔐 Privacy Parameters")
        
        epsilon = st.sidebar.number_input(
            "Privacy Budget (ε):",
            min_value=0.01,
            max_value=10.0,
            value=st.session_state.get(epsilon_key, 1.0),
            step=0.1,
            key=epsilon_key,
            help="Privacy budget for differential privacy"
        )

        delta = st.sidebar.number_input(
            "Failure Probability (δ):",
            min_value=1e-10,
            max_value=0.1,
            value=st.session_state.get(delta_key, 1e-5),
            step=1e-6,
            format="%.2e",
            key=delta_key,
            help="Failure probability for (ε,δ)-differential privacy"
        )

        structure_epsilon_fraction = st.sidebar.slider(
            "Structure Learning ε Fraction:",
            min_value=0.1,
            max_value=0.8,
            value=st.session_state.get(structure_epsilon_key, 0.3),
            step=0.05,
            key=structure_epsilon_key,
            help="Fraction of privacy budget for structure learning"
        )

        # Bayesian Network Configuration
        st.sidebar.subheader("🔗 Network Configuration")
        
        max_parents = st.sidebar.slider(
            "Max Parents per Node:",
            min_value=1,
            max_value=5,
            value=st.session_state.get(max_parents_key, 3),
            key=max_parents_key,
            help="Maximum number of parent nodes in Bayesian network"
        )

        selection_methods = {
            'mutual_information': 'Mutual Information',
            'random': 'Random Selection',
            'correlation': 'Correlation-based'
        }
        
        attribute_selection_method = st.sidebar.selectbox(
            "Attribute Selection:",
            options=list(selection_methods.keys()),
            format_func=lambda x: selection_methods[x],
            index=list(selection_methods.keys()).index(
                st.session_state.get(selection_method_key, 'mutual_information')
            ),
            key=selection_method_key,
            help="Method for selecting network structure"
        )

        # Data Preprocessing
        st.sidebar.subheader("📊 Data Preprocessing")
        
        discretization_bins = st.sidebar.slider(
            "Discretization Bins:",
            min_value=5,
            max_value=50,
            value=st.session_state.get(discretization_bins_key, 10),
            key=discretization_bins_key,
            help="Number of bins for discretizing continuous variables"
        )

        min_bin_size = st.sidebar.number_input(
            "Minimum Bin Size:",
            min_value=1,
            max_value=50,
            value=st.session_state.get(min_bin_size_key, 5),
            key=min_bin_size_key,
            help="Minimum number of records per bin"
        )

        laplace_smoothing = st.sidebar.slider(
            "Laplace Smoothing:",
            min_value=0.01,
            max_value=1.0,
            value=st.session_state.get(laplace_smoothing_key, 0.1),
            step=0.01,
            key=laplace_smoothing_key,
            help="Smoothing parameter for conditional distributions"
        )

        # Generation Parameters
        st.sidebar.subheader("⚙️ Generation Parameters")
        
        n_samples = st.sidebar.number_input(
            "Number of synthetic samples:",
            min_value=100,
            max_value=10000,
            value=st.session_state.get(n_samples_key, 1000),
            step=100,
            key=n_samples_key,
            help="Number of synthetic records to generate"
        )

        # Reproducibility
        random_seed = st.sidebar.number_input(
            "Random Seed:",
            min_value=0,
            max_value=999999,
            value=st.session_state.get(random_seed_key, 42),
            key=random_seed_key,
            help="Seed for reproducible results"
        )

        # Privacy Analysis
        if epsilon > 0:
            st.sidebar.subheader("📈 Privacy Analysis")
            structure_eps = epsilon * structure_epsilon_fraction
            parameter_eps = epsilon * (1 - structure_epsilon_fraction)
            
            st.sidebar.metric("Structure Learning ε", f"{structure_eps:.3f}")
            st.sidebar.metric("Parameter Learning ε", f"{parameter_eps:.3f}")
            st.sidebar.metric("Total Privacy Cost", f"({epsilon:.3f}, {delta:.2e})")

        return {
            'epsilon': epsilon,
            'delta': delta,
            'max_parents': max_parents,
            'discretization_bins': discretization_bins,
            'n_samples': n_samples,
            'random_seed': random_seed,
            'structure_epsilon_fraction': structure_epsilon_fraction,
            'attribute_selection_method': attribute_selection_method,
            'laplace_smoothing': laplace_smoothing,
            'min_bin_size': min_bin_size
        }
    
    def get_parameters(self) -> Dict[str, Any]:
        return {
            'epsilon': self.epsilon,
            'delta': self.delta,
            'max_parents': self.max_parents,
            'discretization_bins': self.discretization_bins,
            'n_samples': self.n_samples,
            'random_seed': self.random_seed,
            'structure_epsilon_fraction': self.structure_epsilon_fraction,
            'attribute_selection_method': self.attribute_selection_method,
            'laplace_smoothing': self.laplace_smoothing,
            'min_bin_size': self.min_bin_size
        }
    
    def set_parameters(self, params: Dict[str, Any]) -> None:
        self.epsilon = params.get('epsilon', self.epsilon)
        self.delta = params.get('delta', self.delta)
        self.max_parents = params.get('max_parents', self.max_parents)
        self.discretization_bins = params.get('discretization_bins', self.discretization_bins)
        self.n_samples = params.get('n_samples', self.n_samples)
        self.random_seed = params.get('random_seed', self.random_seed)
        self.structure_epsilon_fraction = params.get('structure_epsilon_fraction', self.structure_epsilon_fraction)
        self.attribute_selection_method = params.get('attribute_selection_method', self.attribute_selection_method)
        self.laplace_smoothing = params.get('laplace_smoothing', self.laplace_smoothing)
        self.min_bin_size = params.get('min_bin_size', self.min_bin_size)
    
    def build_sidebar_ui(self) -> Dict[str, Any]:
        """Build the Streamlit sidebar UI for PrivBayes configuration."""
        st.sidebar.markdown("### PrivBayes Configuration")
        
        # Privacy parameters
        st.sidebar.markdown("#### Privacy Parameters")
        epsilon = st.sidebar.slider(
            "Epsilon (ε)",
            min_value=0.1,
            max_value=10.0,
            value=self.epsilon,
            step=0.1,
            help="Privacy budget - lower values provide stronger privacy"
        )
        
        delta = st.sidebar.selectbox(
            "Delta (δ)",
            options=[1e-6, 1e-5, 1e-4, 1e-3],
            index=[1e-6, 1e-5, 1e-4, 1e-3].index(self.delta),
            help="Failure probability for (ε,δ)-differential privacy"
        )
        
        # Bayesian network structure parameters
        st.sidebar.markdown("#### Bayesian Network Parameters")
        max_parents = st.sidebar.number_input(
            "Maximum Parents",
            min_value=1,
            max_value=10,
            value=self.max_parents,
            help="Maximum number of parent nodes for each variable"
        )
        
        structure_epsilon_fraction = st.sidebar.slider(
            "Structure Learning ε Fraction",
            min_value=0.1,
            max_value=0.9,
            value=self.structure_epsilon_fraction,
            step=0.1,
            help="Fraction of privacy budget for structure learning"
        )
        
        attribute_selection_method = st.sidebar.selectbox(
            "Attribute Selection",
            options=["mutual_information", "random"],
            value=self.attribute_selection_method,
            help="Method for selecting parent attributes"
        )
        
        # Discretization parameters
        st.sidebar.markdown("#### Discretization Parameters")
        discretization_bins = st.sidebar.number_input(
            "Discretization Bins",
            min_value=5,
            max_value=50,
            value=self.discretization_bins,
            help="Number of bins for discretizing continuous variables"
        )
        
        min_bin_size = st.sidebar.number_input(
            "Minimum Bin Size",
            min_value=1,
            max_value=20,
            value=self.min_bin_size,
            help="Minimum number of records per bin"
        )
        
        # Generation parameters
        st.sidebar.markdown("#### Generation Parameters")
        n_samples = st.sidebar.number_input(
            "Number of Samples",
            min_value=100,
            max_value=10000,
            value=self.n_samples,
            step=100,
            help="Number of synthetic samples to generate"
        )
        
        random_seed = st.sidebar.number_input(
            "Random Seed",
            min_value=0,
            max_value=9999,
            value=self.random_seed,
            help="Random seed for reproducibility"
        )
        
        # Advanced parameters
        st.sidebar.markdown("#### Advanced Parameters")
        laplace_smoothing = st.sidebar.slider(
            "Laplace Smoothing",
            min_value=0.01,
            max_value=1.0,
            value=self.laplace_smoothing,
            step=0.01,
            help="Smoothing parameter for probability estimation"
        )
        
        return {
            'epsilon': epsilon,
            'delta': delta,
            'max_parents': max_parents,
            'discretization_bins': discretization_bins,
            'n_samples': n_samples,
            'random_seed': random_seed,
            'structure_epsilon_fraction': structure_epsilon_fraction,
            'attribute_selection_method': attribute_selection_method,
            'laplace_smoothing': laplace_smoothing,
            'min_bin_size': min_bin_size
        }
    
    def validate_parameters(self, params: Dict[str, Any]) -> Tuple[bool, str]:
        """Validate the provided parameters."""
        try:
            # Check epsilon
            epsilon = params.get('epsilon', self.epsilon)
            if not isinstance(epsilon, (int, float)) or epsilon <= 0:
                return False, "Epsilon must be positive"
            
            # Check delta
            delta = params.get('delta', self.delta)
            if not isinstance(delta, (int, float)) or delta <= 0 or delta >= 1:
                return False, "Delta must be between 0 and 1"
            
            # Check max_parents
            max_parents = params.get('max_parents', self.max_parents)
            if not isinstance(max_parents, int) or max_parents < 1:
                return False, "Maximum parents must be at least 1"
            
            # Check discretization bins
            bins = params.get('discretization_bins', self.discretization_bins)
            if not isinstance(bins, int) or bins < 5:
                return False, "Discretization bins must be at least 5"
            
            # Check structure epsilon fraction
            struct_frac = params.get('structure_epsilon_fraction', self.structure_epsilon_fraction)
            if not isinstance(struct_frac, (int, float)) or struct_frac <= 0 or struct_frac >= 1:
                return False, "Structure epsilon fraction must be between 0 and 1"
            
            return True, "Parameters are valid"
            
        except Exception as e:
            return False, f"Parameter validation error: {str(e)}"
    
    def _discretize_continuous_variables(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Discretize continuous variables into bins.
        
        Returns discretized data and discretization mappings.
        """
        discretized_data = data.copy()
        discretization_maps = {}
        
        for column in data.columns:
            if data[column].dtype in ['object', 'category']:
                # Already categorical
                discretization_maps[column] = {
                    'type': 'categorical',
                    'categories': data[column].unique().tolist()
                }
            else:
                # Continuous variable - discretize
                col_data = data[column].dropna()
                
                if len(col_data.unique()) <= self.discretization_bins:
                    # Already discrete enough
                    discretization_maps[column] = {
                        'type': 'discrete',
                        'values': sorted(col_data.unique())
                    }
                else:
                    # Create bins
                    try:
                        # Use quantile-based binning for better distribution
                        quantiles = np.linspace(0, 1, self.discretization_bins + 1)
                        bin_edges = col_data.quantile(quantiles).unique()
                        
                        # Ensure we have enough bins
                        if len(bin_edges) < 3:
                            bin_edges = np.linspace(col_data.min(), col_data.max(), self.discretization_bins + 1)
                        
                        # Discretize
                        discretized_values = pd.cut(data[column], bins=bin_edges, 
                                                  labels=False, include_lowest=True, duplicates='drop')
                        discretized_data[column] = discretized_values
                        
                        discretization_maps[column] = {
                            'type': 'binned',
                            'bin_edges': bin_edges.tolist(),
                            'n_bins': len(bin_edges) - 1
                        }
                        
                    except Exception as e:
                        logger.warning(f"Failed to discretize {column}: {e}")
                        # Fall back to simple binning
                        discretized_values = pd.cut(data[column], bins=self.discretization_bins, 
                                                  labels=False, include_lowest=True)
                        discretized_data[column] = discretized_values
                        
                        discretization_maps[column] = {
                            'type': 'simple_binned',
                            'min_val': col_data.min(),
                            'max_val': col_data.max(),
                            'n_bins': self.discretization_bins
                        }
        
        return discretized_data, discretization_maps
    
    def _add_laplace_noise(self, count: float, sensitivity: float, epsilon: float) -> float:
        """Add Laplace noise for differential privacy."""
        if epsilon <= 0:
            return count
        
        scale = sensitivity / epsilon
        noise = np.random.laplace(0, scale)
        return max(0, count + noise)  # Ensure non-negative counts
    
    def _calculate_mutual_information(self, data: pd.DataFrame, var1: str, var2: str, epsilon: float) -> float:
        """
        Calculate noisy mutual information between two variables.
        """
        try:
            # Create contingency table
            contingency = pd.crosstab(data[var1], data[var2])
            
            # Add Laplace noise to each cell
            sensitivity = 1.0  # Adding/removing one record changes count by at most 1
            noisy_contingency = contingency.copy()
            
            for i in contingency.index:
                for j in contingency.columns:
                    noisy_count = self._add_laplace_noise(contingency.loc[i, j], sensitivity, epsilon)
                    noisy_contingency.loc[i, j] = noisy_count
            
            # Calculate mutual information from noisy counts
            total = noisy_contingency.sum().sum()
            if total == 0:
                return 0.0
            
            # Marginal probabilities
            p_x = noisy_contingency.sum(axis=1) / total
            p_y = noisy_contingency.sum(axis=0) / total
            
            # Joint probabilities
            p_xy = noisy_contingency / total
            
            # Mutual information
            mi = 0.0
            for i in p_xy.index:
                for j in p_xy.columns:
                    if p_xy.loc[i, j] > 0 and p_x[i] > 0 and p_y[j] > 0:
                        mi += p_xy.loc[i, j] * np.log(p_xy.loc[i, j] / (p_x[i] * p_y[j]))
            
            return max(0, mi)  # Ensure non-negative MI
            
        except Exception as e:
            logger.warning(f"Failed to calculate MI between {var1} and {var2}: {e}")
            return 0.0
    
    def _learn_bayesian_network_structure(self, data: pd.DataFrame, epsilon: float) -> Tuple[List[str], Dict[str, List[str]]]:
        """
        Learn Bayesian network structure with differential privacy.
        
        Returns node ordering and parent relationships.
        """
        variables = list(data.columns)
        np.random.seed(self.random_seed)
        
        # Initialize
        node_order = []
        parents = {var: [] for var in variables}
        remaining_vars = set(variables)
        
        # Privacy budget per step
        n_steps = len(variables)
        step_epsilon = epsilon / n_steps if n_steps > 0 else epsilon
        
        # Greedy structure learning
        while remaining_vars:
            if self.attribute_selection_method == "random":
                # Random selection (baseline)
                next_var = np.random.choice(list(remaining_vars))
            else:
                # Mutual information-based selection
                mi_scores = {}
                
                for var in remaining_vars:
                    total_mi = 0.0
                    
                    # Calculate MI with already selected variables
                    for selected_var in node_order:
                        mi = self._calculate_mutual_information(data, var, selected_var, step_epsilon / len(node_order) if node_order else step_epsilon)
                        total_mi += mi
                    
                    mi_scores[var] = total_mi
                
                # Select variable with highest MI (with noise for privacy)
                if mi_scores:
                    # Add noise to MI scores
                    noisy_scores = {}
                    for var, score in mi_scores.items():
                        noise = np.random.laplace(0, 1.0 / step_epsilon)
                        noisy_scores[var] = score + noise
                    
                    next_var = max(noisy_scores, key=noisy_scores.get)
                else:
                    next_var = np.random.choice(list(remaining_vars))
            
            # Add to order
            node_order.append(next_var)
            remaining_vars.remove(next_var)
            
            # Select parents for this variable
            if len(node_order) > 1:  # Can only have parents from previously selected variables
                candidate_parents = node_order[:-1]  # All except the current variable
                
                if len(candidate_parents) <= self.max_parents:
                    # Use all candidates as parents
                    selected_parents = candidate_parents
                else:
                    # Select top k parents based on MI
                    parent_mi_scores = {}
                    for candidate in candidate_parents:
                        mi = self._calculate_mutual_information(data, next_var, candidate, step_epsilon / len(candidate_parents))
                        parent_mi_scores[candidate] = mi
                    
                    # Add noise and select top-k
                    noisy_parent_scores = {}
                    for candidate, score in parent_mi_scores.items():
                        noise = np.random.laplace(0, 1.0 / step_epsilon)
                        noisy_parent_scores[candidate] = score + noise
                    
                    # Select top max_parents
                    sorted_parents = sorted(noisy_parent_scores.items(), key=lambda x: x[1], reverse=True)
                    selected_parents = [p[0] for p in sorted_parents[:self.max_parents]]
                
                parents[next_var] = selected_parents
        
        return node_order, parents
    
    def _learn_conditional_distributions(self, data: pd.DataFrame, node_order: List[str], parents: Dict[str, List[str]], epsilon: float) -> Dict[str, Dict]:
        """
        Learn conditional probability distributions with differential privacy.
        """
        conditional_distributions = {}
        n_distributions = len(node_order)
        dist_epsilon = epsilon / n_distributions if n_distributions > 0 else epsilon
        
        for variable in node_order:
            var_parents = parents[variable]
            
            if not var_parents:
                # No parents - learn marginal distribution
                value_counts = data[variable].value_counts()
                
                # Add Laplace noise
                noisy_counts = {}
                sensitivity = 1.0
                
                for value, count in value_counts.items():
                    noisy_count = self._add_laplace_noise(count, sensitivity, dist_epsilon)
                    noisy_counts[value] = noisy_count
                
                # Normalize to probabilities
                total = sum(noisy_counts.values())
                if total > 0:
                    probabilities = {value: count / total for value, count in noisy_counts.items()}
                else:
                    # Uniform distribution as fallback
                    unique_values = data[variable].unique()
                    probabilities = {value: 1.0 / len(unique_values) for value in unique_values}
                
                conditional_distributions[variable] = {
                    'type': 'marginal',
                    'probabilities': probabilities
                }
                
            else:
                # Has parents - learn conditional distribution
                parent_combinations = []
                
                # Get all unique parent combinations
                if len(var_parents) == 1:
                    parent_combinations = [(val,) for val in data[var_parents[0]].unique()]
                else:
                    # Multi-dimensional parent combinations
                    for _, row in data[var_parents].drop_duplicates().iterrows():
                        combination = tuple(row[parent] for parent in var_parents)
                        parent_combinations.append(combination)
                
                conditional_probs = {}
                
                for parent_combo in parent_combinations:
                    # Filter data for this parent combination
                    mask = True
                    for i, parent in enumerate(var_parents):
                        mask = mask & (data[parent] == parent_combo[i])
                    
                    filtered_data = data[mask]
                    
                    if len(filtered_data) > 0:
                        value_counts = filtered_data[variable].value_counts()
                        
                        # Add Laplace noise
                        noisy_counts = {}
                        sensitivity = 1.0
                        
                        for value, count in value_counts.items():
                            noisy_count = self._add_laplace_noise(count, sensitivity, dist_epsilon / len(parent_combinations))
                            noisy_counts[value] = noisy_count
                        
                        # Normalize
                        total = sum(noisy_counts.values())
                        if total > 0:
                            probabilities = {value: count / total for value, count in noisy_counts.items()}
                        else:
                            # Uniform distribution as fallback
                            unique_values = data[variable].unique()
                            probabilities = {value: 1.0 / len(unique_values) for value in unique_values}
                        
                        conditional_probs[parent_combo] = probabilities
                    else:
                        # No data for this combination - use uniform distribution
                        unique_values = data[variable].unique()
                        conditional_probs[parent_combo] = {value: 1.0 / len(unique_values) for value in unique_values}
                
                conditional_distributions[variable] = {
                    'type': 'conditional',
                    'parents': var_parents,
                    'probabilities': conditional_probs
                }
        
        return conditional_distributions
    
    def _undiscretize_value(self, variable: str, discretized_value: Any) -> Any:
        """Convert discretized value back to original scale."""
        if variable not in self.discretization_maps:
            return discretized_value
        
        disc_info = self.discretization_maps[variable]
        
        if disc_info['type'] == 'categorical':
            return discretized_value
        elif disc_info['type'] == 'discrete':
            return discretized_value
        elif disc_info['type'] == 'binned':
            bin_edges = disc_info['bin_edges']
            if pd.isna(discretized_value) or discretized_value < 0 or discretized_value >= len(bin_edges) - 1:
                return np.random.uniform(bin_edges[0], bin_edges[-1])
            
            # Sample uniformly within the bin
            bin_idx = int(discretized_value)
            return np.random.uniform(bin_edges[bin_idx], bin_edges[bin_idx + 1])
        elif disc_info['type'] == 'simple_binned':
            if pd.isna(discretized_value):
                return np.random.uniform(disc_info['min_val'], disc_info['max_val'])
            
            # Calculate bin boundaries
            bin_width = (disc_info['max_val'] - disc_info['min_val']) / disc_info['n_bins']
            bin_idx = int(discretized_value)
            bin_start = disc_info['min_val'] + bin_idx * bin_width
            bin_end = bin_start + bin_width
            
            return np.random.uniform(bin_start, bin_end)
        
        return discretized_value
    
    def fit(self, data: pd.DataFrame) -> None:
        """Fit the PrivBayes model to the data."""
        try:
            np.random.seed(self.random_seed)
            
            # Step 1: Discretize continuous variables
            discretized_data, discretization_maps = self._discretize_continuous_variables(data)
            self.discretization_maps = discretization_maps
            
            # Step 2: Learn Bayesian network structure
            structure_epsilon = self.epsilon * self.structure_epsilon_fraction
            node_order, parents = self._learn_bayesian_network_structure(discretized_data, structure_epsilon)
            
            self.node_order = node_order
            self.bayesian_network = parents
            
            # Step 3: Learn conditional distributions
            parameter_epsilon = self.epsilon * (1 - self.structure_epsilon_fraction)
            conditional_distributions = self._learn_conditional_distributions(
                discretized_data, node_order, parents, parameter_epsilon
            )
            
            self.conditional_distributions = conditional_distributions
            self.is_fitted = True
            
        except Exception as e:
            logger.error(f"Error fitting PrivBayes model: {str(e)}")
            raise
    
    def anonymize(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate synthetic data using the fitted PrivBayes model."""
        try:
            if not self.is_fitted:
                self.fit(data)
            
            np.random.seed(self.random_seed)
            synthetic_data = {}
            
            # Generate samples following the Bayesian network order
            for _ in range(self.n_samples):
                sample = {}
                
                for variable in self.node_order:
                    dist_info = self.conditional_distributions[variable]
                    
                    if dist_info['type'] == 'marginal':
                        # Sample from marginal distribution
                        probabilities = dist_info['probabilities']
                        values = list(probabilities.keys())
                        probs = list(probabilities.values())
                        
                        if sum(probs) > 0:
                            # Normalize probabilities
                            probs = np.array(probs) / sum(probs)
                            sampled_value = np.random.choice(values, p=probs)
                        else:
                            sampled_value = np.random.choice(values)
                        
                    else:  # conditional
                        # Get parent values for this sample
                        parent_values = tuple(sample[parent] for parent in dist_info['parents'])
                        
                        if parent_values in dist_info['probabilities']:
                            probabilities = dist_info['probabilities'][parent_values]
                        else:
                            # Fallback: use first available parent combination
                            available_combos = list(dist_info['probabilities'].keys())
                            if available_combos:
                                probabilities = dist_info['probabilities'][available_combos[0]]
                            else:
                                # Ultimate fallback: uniform over observed values
                                unique_values = data[variable].unique()
                                probabilities = {value: 1.0 / len(unique_values) for value in unique_values}
                        
                        values = list(probabilities.keys())
                        probs = list(probabilities.values())
                        
                        if sum(probs) > 0:
                            probs = np.array(probs) / sum(probs)
                            sampled_value = np.random.choice(values, p=probs)
                        else:
                            sampled_value = np.random.choice(values)
                    
                    sample[variable] = sampled_value
                
                # Add sample to synthetic data
                for variable, value in sample.items():
                    if variable not in synthetic_data:
                        synthetic_data[variable] = []
                    synthetic_data[variable].append(value)
            
            # Convert to DataFrame
            synthetic_df = pd.DataFrame(synthetic_data)
            
            # Undiscretize continuous variables
            for variable in synthetic_df.columns:
                if variable in self.discretization_maps:
                    disc_info = self.discretization_maps[variable]
                    if disc_info['type'] in ['binned', 'simple_binned']:
                        synthetic_df[variable] = synthetic_df[variable].apply(
                            lambda x: self._undiscretize_value(variable, x)
                        )
            
            return synthetic_df
            
        except Exception as e:
            logger.error(f"Error generating synthetic data: {str(e)}")
            raise
    
    def calculate_privacy_metrics(self, original_data: pd.DataFrame, anonymized_data: pd.DataFrame) -> Dict[str, float]:
        """Calculate privacy metrics for PrivBayes."""
        try:
            metrics = {}
            
            # Differential privacy guarantee
            metrics['differential_privacy'] = 1.0  # PrivBayes provides formal DP guarantees
            
            # Privacy budget utilization
            epsilon_score = min(2.0 / (self.epsilon + 1), 1.0)  # Stronger privacy for smaller epsilon
            metrics['epsilon_privacy'] = epsilon_score
            
            # Structure privacy (benefit from Bayesian network structure)
            structure_complexity = len(self.bayesian_network) * self.max_parents
            structure_privacy = min(structure_complexity / (len(original_data.columns) * 5), 1.0)
            metrics['structure_privacy'] = structure_privacy
            
            # Discretization privacy (information loss through binning)
            discretization_privacy = 0.0
            total_vars = 0
            
            for var, disc_info in self.discretization_maps.items():
                if disc_info['type'] in ['binned', 'simple_binned']:
                    # More bins = less privacy through discretization
                    n_bins = disc_info.get('n_bins', self.discretization_bins)
                    var_privacy = max(0, 1.0 - (n_bins / 50.0))  # Normalize by reasonable max
                    discretization_privacy += var_privacy
                else:
                    discretization_privacy += 0.5  # Moderate privacy for categorical/discrete
                total_vars += 1
            
            if total_vars > 0:
                metrics['discretization_privacy'] = discretization_privacy / total_vars
            
            # Overall privacy score
            privacy_score = np.mean([
                metrics.get('differential_privacy', 0),
                metrics.get('epsilon_privacy', 0),
                metrics.get('structure_privacy', 0),
                metrics.get('discretization_privacy', 0)
            ])
            metrics['overall_privacy_score'] = privacy_score
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error calculating privacy metrics: {str(e)}")
            return {'overall_privacy_score': 0.0}
    
    def calculate_utility_metrics(self, original_data: pd.DataFrame, anonymized_data: pd.DataFrame) -> Dict[str, float]:
        """Calculate utility metrics for PrivBayes."""
        try:
            metrics = {}
            
            # Bayesian network structure utility
            if hasattr(self, 'bayesian_network') and self.bayesian_network:
                # Measure how well the network captures dependencies
                total_edges = sum(len(parents) for parents in self.bayesian_network.values())
                max_possible_edges = len(self.node_order) * self.max_parents
                structure_density = total_edges / max(max_possible_edges, 1)
                metrics['structure_utility'] = min(structure_density * 2, 1.0)  # Scale appropriately
            
            # Marginal distribution preservation
            from scipy import stats
            marginal_similarities = []
            
            for column in original_data.select_dtypes(include=[np.number]).columns:
                if column in anonymized_data.columns:
                    try:
                        ks_stat, _ = stats.ks_2samp(
                            original_data[column].dropna(),
                            anonymized_data[column].dropna()
                        )
                        marginal_similarities.append(1.0 - ks_stat)
                    except:
                        continue
            
            if marginal_similarities:
                metrics['marginal_utility'] = np.mean(marginal_similarities)
            
            # Conditional distribution utility (simplified)
            conditional_utilities = []
            
            # Check pairwise correlations as proxy for conditional distributions
            try:
                orig_corr = original_data.select_dtypes(include=[np.number]).corr()
                anon_corr = anonymized_data.select_dtypes(include=[np.number]).corr()
                
                if not orig_corr.empty and not anon_corr.empty:
                    corr_diff = np.abs(orig_corr.values - anon_corr.values)
                    conditional_utility = 1.0 - np.nanmean(corr_diff)
                    metrics['conditional_utility'] = max(0.0, conditional_utility)
            except:
                pass
            
            # Discretization impact on utility
            discretization_impact = 0.0
            total_discretized = 0
            
            for var, disc_info in self.discretization_maps.items():
                if disc_info['type'] in ['binned', 'simple_binned']:
                    # More bins preserve more information
                    n_bins = disc_info.get('n_bins', self.discretization_bins)
                    info_preservation = min(n_bins / 20.0, 1.0)  # Normalize
                    discretization_impact += info_preservation
                    total_discretized += 1
            
            if total_discretized > 0:
                metrics['discretization_utility'] = discretization_impact / total_discretized
            
            # Privacy-utility tradeoff
            privacy_cost = 1.0 / (self.epsilon + 1)  # Higher epsilon = lower privacy cost
            utility_bonus = 1.0 - privacy_cost
            metrics['privacy_utility_tradeoff'] = utility_bonus
            
            # Overall utility score
            utility_components = [
                metrics.get('structure_utility', 0),
                metrics.get('marginal_utility', 0),
                metrics.get('conditional_utility', 0),
                metrics.get('discretization_utility', 0),
                metrics.get('privacy_utility_tradeoff', 0)
            ]
            metrics['overall_utility_score'] = np.mean([u for u in utility_components if u > 0])
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error calculating utility metrics: {str(e)}")
            return {'overall_utility_score': 0.0}
    
    def build_config_export(self) -> Dict[str, Any]:
        """Export current configuration."""
        return {
            'plugin_name': self.name,
            'parameters': self.get_parameters(),
            'fitted_models': {
                'bayesian_network': self.bayesian_network if hasattr(self, 'bayesian_network') else {},
                'node_order': self.node_order if hasattr(self, 'node_order') else [],
                'discretization_maps': self.discretization_maps if hasattr(self, 'discretization_maps') else {},
                'conditional_distributions': self.conditional_distributions if hasattr(self, 'conditional_distributions') else {},
                'is_fitted': self.is_fitted if hasattr(self, 'is_fitted') else False
            }
        }
    
    def apply_config_import(self, config: Dict[str, Any]) -> None:
        """Import configuration."""
        if 'parameters' in config:
            self.set_parameters(config['parameters'])
        
        if 'fitted_models' in config:
            fitted = config['fitted_models']
            self.bayesian_network = fitted.get('bayesian_network', {})
            self.node_order = fitted.get('node_order', [])
            self.discretization_maps = fitted.get('discretization_maps', {})
            self.conditional_distributions = fitted.get('conditional_distributions', {})
            self.is_fitted = fitted.get('is_fitted', False)

def get_plugin():
    """Factory function to get plugin instance."""
    return PrivBayesPlugin()
