"""
Copula-based Synthesis Plugin for Data Anonymization

This plugin implements copula-based synthetic data generation, which models
the dependency structure between variables using copulas while fitting
marginal distributions independently.
"""

import streamlit as st
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, Tuple
import logging
from scipy import stats
from scipy.stats import gaussian_kde
import warnings
warnings.filterwarnings('ignore')

from ..base_anonymizer import Anonymizer

logger = logging.getLogger(__name__)

class CopulaSynthesisPlugin(Anonymizer):
    """
    Copula-based synthetic data generation plugin.
    
    Uses copulas to model the dependency structure between variables
    while fitting marginal distributions independently.
    """
    
    def __init__(self):
        super().__init__()
        self.name = "Copula-based Synthesis"
        self.description = "Generate synthetic data using copula-based modeling"
        self.category = "Generative Models"
        
        # Plugin parameters
        self.copula_type = "gaussian"  # gaussian, t, empirical
        self.marginal_method = "kde"   # kde, parametric, empirical
        self.n_samples = 1000
        self.random_seed = 42
        self.kde_bandwidth = "scott"
        self.correlation_threshold = 0.1
        
        # Fitted models
        self.marginal_models = {}
        self.copula_params = {}
        self.correlation_matrix = None
        self.original_columns = []
        self.is_fitted = False
        
    def get_name(self) -> str:
        return "Copula-based Synthesis"
    
    def get_category(self) -> str:
        return "Generative Models"

    def get_description(self) -> str:
        return "Generate synthetic data using copula-based modeling to preserve dependency structures"
    
    def get_parameters(self) -> Dict[str, Any]:
        return {
            'copula_type': self.copula_type,
            'marginal_method': self.marginal_method,
            'n_samples': self.n_samples,
            'random_seed': self.random_seed,
            'kde_bandwidth': self.kde_bandwidth,
            'correlation_threshold': self.correlation_threshold
        }
    
    def set_parameters(self, params: Dict[str, Any]) -> None:
        self.copula_type = params.get('copula_type', self.copula_type)
        self.marginal_method = params.get('marginal_method', self.marginal_method)
        self.n_samples = params.get('n_samples', self.n_samples)
        self.random_seed = params.get('random_seed', self.random_seed)
        self.kde_bandwidth = params.get('kde_bandwidth', self.kde_bandwidth)
        self.correlation_threshold = params.get('correlation_threshold', self.correlation_threshold)
    
    def build_sidebar_ui(self) -> Dict[str, Any]:
        """Build the Streamlit sidebar UI for copula synthesis configuration."""
        st.sidebar.markdown("### Copula-based Synthesis Configuration")
        
        # Copula type selection
        copula_type = st.sidebar.selectbox(
            "Copula Type",
            options=["gaussian", "t", "empirical"],
            value=self.copula_type,
            help="Type of copula to model dependencies"
        )
        
        # Marginal distribution method
        marginal_method = st.sidebar.selectbox(
            "Marginal Method",
            options=["kde", "parametric", "empirical"],
            value=self.marginal_method,
            help="Method to model marginal distributions"
        )
        
        # Number of synthetic samples
        n_samples = st.sidebar.number_input(
            "Number of Samples",
            min_value=100,
            max_value=10000,
            value=self.n_samples,
            step=100,
            help="Number of synthetic samples to generate"
        )
        
        # Random seed
        random_seed = st.sidebar.number_input(
            "Random Seed",
            min_value=0,
            max_value=9999,
            value=self.random_seed,
            help="Random seed for reproducibility"
        )
        
        # KDE bandwidth (if using KDE)
        if marginal_method == "kde":
            kde_bandwidth = st.sidebar.selectbox(
                "KDE Bandwidth",
                options=["scott", "silverman", "auto"],
                value=self.kde_bandwidth,
                help="Bandwidth selection method for KDE"
            )
        else:
            kde_bandwidth = self.kde_bandwidth
        
        # Correlation threshold
        correlation_threshold = st.sidebar.slider(
            "Correlation Threshold",
            min_value=0.0,
            max_value=1.0,
            value=self.correlation_threshold,
            step=0.05,
            help="Minimum correlation to consider for modeling"
        )
        
        return {
            'copula_type': copula_type,
            'marginal_method': marginal_method,
            'n_samples': n_samples,
            'random_seed': random_seed,
            'kde_bandwidth': kde_bandwidth,
            'correlation_threshold': correlation_threshold
        }
    
    def validate_parameters(self, params: Dict[str, Any]) -> Tuple[bool, str]:
        """Validate the provided parameters."""
        try:
            # Check copula type
            if params.get('copula_type') not in ['gaussian', 't', 'empirical']:
                return False, "Invalid copula type"
            
            # Check marginal method
            if params.get('marginal_method') not in ['kde', 'parametric', 'empirical']:
                return False, "Invalid marginal method"
            
            # Check n_samples
            n_samples = params.get('n_samples', self.n_samples)
            if not isinstance(n_samples, int) or n_samples < 100:
                return False, "Number of samples must be at least 100"
            
            # Check correlation threshold
            threshold = params.get('correlation_threshold', self.correlation_threshold)
            if not isinstance(threshold, (int, float)) or threshold < 0 or threshold > 1:
                return False, "Correlation threshold must be between 0 and 1"
            
            return True, "Parameters are valid"
            
        except Exception as e:
            return False, f"Parameter validation error: {str(e)}"
    
    def _fit_marginal_distributions(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Fit marginal distributions for each column."""
        marginal_models = {}
        
        for column in data.columns:
            values = data[column].dropna()
            
            if values.dtype in ['object', 'category']:
                # Categorical variable - use empirical distribution
                marginal_models[column] = {
                    'type': 'categorical',
                    'categories': values.value_counts(normalize=True).to_dict()
                }
            else:
                # Numerical variable
                if self.marginal_method == 'kde':
                    # Use KDE
                    kde = gaussian_kde(values, bw_method=self.kde_bandwidth)
                    marginal_models[column] = {
                        'type': 'kde',
                        'model': kde,
                        'min_val': values.min(),
                        'max_val': values.max()
                    }
                elif self.marginal_method == 'parametric':
                    # Fit normal distribution (can be extended to other distributions)
                    mean, std = values.mean(), values.std()
                    marginal_models[column] = {
                        'type': 'normal',
                        'mean': mean,
                        'std': std
                    }
                else:  # empirical
                    marginal_models[column] = {
                        'type': 'empirical',
                        'values': values.values,
                        'quantiles': np.percentile(values, np.linspace(0, 100, 1000))
                    }
        
        return marginal_models
    
    def _transform_to_uniform(self, data: pd.DataFrame) -> pd.DataFrame:
        """Transform data to uniform marginals using fitted distributions."""
        uniform_data = pd.DataFrame(index=data.index)
        
        for column in data.columns:
            values = data[column].dropna()
            model = self.marginal_models[column]
            
            if model['type'] == 'categorical':
                # For categorical, assign uniform random values
                uniform_values = np.random.uniform(0, 1, len(values))
                uniform_data[column] = uniform_values
            elif model['type'] == 'kde':
                # Use CDF of KDE (approximated)
                sorted_vals = np.sort(values)
                uniform_vals = np.searchsorted(sorted_vals, values) / len(sorted_vals)
                uniform_data[column] = uniform_vals
            elif model['type'] == 'normal':
                # Use normal CDF
                uniform_vals = stats.norm.cdf(values, model['mean'], model['std'])
                uniform_data[column] = uniform_vals
            else:  # empirical
                # Use empirical CDF
                uniform_vals = np.searchsorted(model['quantiles'], values) / len(model['quantiles'])
                uniform_data[column] = uniform_vals
        
        return uniform_data
    
    def _fit_copula(self, uniform_data: pd.DataFrame) -> Dict[str, Any]:
        """Fit copula to uniform-transformed data."""
        copula_params = {}
        
        if self.copula_type == 'gaussian':
            # Gaussian copula - estimate correlation matrix
            correlation_matrix = uniform_data.corr().values
            copula_params = {
                'type': 'gaussian',
                'correlation_matrix': correlation_matrix
            }
        elif self.copula_type == 't':
            # t-copula (simplified - just correlation for now)
            correlation_matrix = uniform_data.corr().values
            copula_params = {
                'type': 't',
                'correlation_matrix': correlation_matrix,
                'degrees_freedom': 4  # Default value
            }
        else:  # empirical
            # Empirical copula - store the data
            copula_params = {
                'type': 'empirical',
                'data': uniform_data.values
            }
        
        return copula_params
    
    def _sample_from_copula(self, n_samples: int) -> np.ndarray:
        """Sample from the fitted copula."""
        n_vars = len(self.original_columns)
        
        if self.copula_params['type'] == 'gaussian':
            # Sample from multivariate normal and transform to uniform
            mean = np.zeros(n_vars)
            cov = self.copula_params['correlation_matrix']
            normal_samples = np.random.multivariate_normal(mean, cov, n_samples)
            uniform_samples = stats.norm.cdf(normal_samples)
        elif self.copula_params['type'] == 't':
            # Sample from multivariate t-distribution (simplified)
            mean = np.zeros(n_vars)
            cov = self.copula_params['correlation_matrix']
            normal_samples = np.random.multivariate_normal(mean, cov, n_samples)
            uniform_samples = stats.norm.cdf(normal_samples)  # Approximation
        else:  # empirical
            # Sample from empirical copula
            data = self.copula_params['data']
            indices = np.random.choice(len(data), n_samples, replace=True)
            uniform_samples = data[indices]
        
        return uniform_samples
    
    def _transform_from_uniform(self, uniform_samples: np.ndarray) -> pd.DataFrame:
        """Transform uniform samples back to original scale."""
        synthetic_data = pd.DataFrame(columns=self.original_columns)
        
        for i, column in enumerate(self.original_columns):
            uniform_vals = uniform_samples[:, i]
            model = self.marginal_models[column]
            
            if model['type'] == 'categorical':
                # Sample from categorical distribution
                categories = list(model['categories'].keys())
                probabilities = list(model['categories'].values())
                synthetic_vals = np.random.choice(categories, len(uniform_vals), p=probabilities)
                synthetic_data[column] = synthetic_vals
            elif model['type'] == 'kde':
                # Sample from KDE
                synthetic_vals = model['model'].resample(len(uniform_vals))[0]
                # Clip to original range
                synthetic_vals = np.clip(synthetic_vals, model['min_val'], model['max_val'])
                synthetic_data[column] = synthetic_vals
            elif model['type'] == 'normal':
                # Use inverse normal CDF
                synthetic_vals = stats.norm.ppf(uniform_vals, model['mean'], model['std'])
                synthetic_data[column] = synthetic_vals
            else:  # empirical
                # Use inverse empirical CDF
                quantile_indices = (uniform_vals * (len(model['quantiles']) - 1)).astype(int)
                quantile_indices = np.clip(quantile_indices, 0, len(model['quantiles']) - 1)
                synthetic_vals = model['quantiles'][quantile_indices]
                synthetic_data[column] = synthetic_vals
        
        return synthetic_data
    
    def fit(self, data: pd.DataFrame) -> None:
        """Fit the copula model to the data."""
        try:
            np.random.seed(self.random_seed)
            self.original_columns = data.columns.tolist()
            
            # Fit marginal distributions
            self.marginal_models = self._fit_marginal_distributions(data)
            
            # Transform to uniform marginals
            uniform_data = self._transform_to_uniform(data)
            
            # Fit copula
            self.copula_params = self._fit_copula(uniform_data)
            
            # Store correlation matrix for metrics
            self.correlation_matrix = uniform_data.corr()
            
            self.is_fitted = True
            
        except Exception as e:
            logger.error(f"Error fitting copula model: {str(e)}")
            raise
    
    def anonymize(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate synthetic data using the fitted copula model."""
        try:
            if not self.is_fitted:
                self.fit(data)
            
            np.random.seed(self.random_seed)
            
            # Sample from copula
            uniform_samples = self._sample_from_copula(self.n_samples)
            
            # Transform back to original scale
            synthetic_data = self._transform_from_uniform(uniform_samples)
            
            return synthetic_data
            
        except Exception as e:
            logger.error(f"Error generating synthetic data: {str(e)}")
            raise
    
    def calculate_privacy_metrics(self, original_data: pd.DataFrame, anonymized_data: pd.DataFrame) -> Dict[str, float]:
        """Calculate privacy metrics for copula-based synthesis."""
        try:
            metrics = {}
            
            # Basic privacy through synthesis
            metrics['synthesis_privacy'] = 1.0  # Complete synthesis provides privacy
            
            # Correlation preservation
            orig_corr = original_data.select_dtypes(include=[np.number]).corr()
            anon_corr = anonymized_data.select_dtypes(include=[np.number]).corr()
            
            if not orig_corr.empty and not anon_corr.empty:
                # Compare correlation matrices
                corr_diff = np.abs(orig_corr.values - anon_corr.values)
                corr_preservation = 1.0 - np.nanmean(corr_diff)
                metrics['correlation_preservation'] = max(0.0, corr_preservation)
            
            # Distribution similarity (using KS test)
            ks_scores = []
            for column in original_data.select_dtypes(include=[np.number]).columns:
                if column in anonymized_data.columns:
                    try:
                        ks_stat, _ = stats.ks_2samp(
                            original_data[column].dropna(),
                            anonymized_data[column].dropna()
                        )
                        ks_scores.append(1.0 - ks_stat)  # Convert to similarity
                    except:
                        continue
            
            if ks_scores:
                metrics['distribution_similarity'] = np.mean(ks_scores)
            
            # Privacy score (weighted combination)
            privacy_score = (
                metrics.get('synthesis_privacy', 0) * 0.5 +
                metrics.get('correlation_preservation', 0) * 0.3 +
                metrics.get('distribution_similarity', 0) * 0.2
            )
            metrics['overall_privacy_score'] = privacy_score
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error calculating privacy metrics: {str(e)}")
            return {'overall_privacy_score': 0.0}
    
    def calculate_utility_metrics(self, original_data: pd.DataFrame, anonymized_data: pd.DataFrame) -> Dict[str, float]:
        """Calculate utility metrics for copula-based synthesis."""
        try:
            metrics = {}
            
            # Statistical similarity
            num_cols = original_data.select_dtypes(include=[np.number]).columns
            
            if len(num_cols) > 0:
                # Mean absolute error of means
                mean_errors = []
                std_errors = []
                
                for col in num_cols:
                    if col in anonymized_data.columns:
                        orig_mean = original_data[col].mean()
                        anon_mean = anonymized_data[col].mean()
                        mean_errors.append(abs(orig_mean - anon_mean) / (abs(orig_mean) + 1e-8))
                        
                        orig_std = original_data[col].std()
                        anon_std = anonymized_data[col].std()
                        std_errors.append(abs(orig_std - anon_std) / (abs(orig_std) + 1e-8))
                
                if mean_errors:
                    metrics['mean_preservation'] = 1.0 - np.mean(mean_errors)
                    metrics['std_preservation'] = 1.0 - np.mean(std_errors)
            
            # Copula quality (correlation preservation)
            if hasattr(self, 'correlation_matrix') and self.correlation_matrix is not None:
                synth_corr = anonymized_data.select_dtypes(include=[np.number]).corr()
                if not synth_corr.empty:
                    corr_diff = np.abs(self.correlation_matrix.values - synth_corr.values)
                    metrics['copula_quality'] = 1.0 - np.nanmean(corr_diff)
            
            # Overall utility score
            utility_score = np.mean([
                metrics.get('mean_preservation', 0),
                metrics.get('std_preservation', 0),
                metrics.get('copula_quality', 0)
            ])
            metrics['overall_utility_score'] = utility_score
            
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
                'marginal_models': self.marginal_models if hasattr(self, 'marginal_models') else {},
                'copula_params': self.copula_params if hasattr(self, 'copula_params') else {},
                'is_fitted': self.is_fitted if hasattr(self, 'is_fitted') else False
            }
        }
    
    def apply_config_import(self, config: Dict[str, Any]) -> None:
        """Import configuration."""
        if 'parameters' in config:
            self.set_parameters(config['parameters'])
        
        if 'fitted_models' in config:
            fitted = config['fitted_models']
            self.marginal_models = fitted.get('marginal_models', {})
            self.copula_params = fitted.get('copula_params', {})
            self.is_fitted = fitted.get('is_fitted', False)
    
    def get_sidebar_ui(self, all_cols: list, sa_col_to_pass: str | None, 
                      df_raw: pd.DataFrame, unique_key_prefix: str) -> Dict[str, Any]:
        """
        Renders the copula synthesis specific UI elements in the Streamlit sidebar.
        """
        st.sidebar.header(f"🔗 {self.get_name()} Configuration")
        
        # Show description
        with st.sidebar.expander("ℹ️ About Copula-based Synthesis"):
            st.markdown(self.get_description())
            st.markdown("""
            **Key Features:**
            - **Dependency Modeling**: Captures complex relationships between variables
            - **Marginal Preservation**: Maintains individual variable distributions
            - **Flexible Copulas**: Gaussian, t-copula, and empirical options
            
            **Use Cases:**
            - Financial data with complex dependencies
            - Multivariate data preservation
            - Research requiring realistic synthetic data
            """)

        # Define session state keys
        copula_type_key = f"{unique_key_prefix}_copula_type"
        marginal_method_key = f"{unique_key_prefix}_marginal_method"
        n_samples_key = f"{unique_key_prefix}_n_samples"
        random_seed_key = f"{unique_key_prefix}_random_seed"
        kde_bandwidth_key = f"{unique_key_prefix}_kde_bandwidth"
        correlation_threshold_key = f"{unique_key_prefix}_correlation_threshold"

        # Copula Configuration
        st.sidebar.subheader("🔗 Copula Configuration")
        
        copula_types = {
            'gaussian': 'Gaussian Copula',
            't': 'Student-t Copula', 
            'empirical': 'Empirical Copula'
        }
        
        copula_type = st.sidebar.selectbox(
            "Copula Type:",
            options=list(copula_types.keys()),
            format_func=lambda x: copula_types[x],
            index=list(copula_types.keys()).index(
                st.session_state.get(copula_type_key, 'gaussian')
            ),
            key=copula_type_key,
            help="Type of copula to model dependencies"
        )

        # Marginal Distribution Configuration
        st.sidebar.subheader("📊 Marginal Distributions")
        
        marginal_methods = {
            'kde': 'Kernel Density Estimation',
            'parametric': 'Parametric Fitting',
            'empirical': 'Empirical Distribution'
        }
        
        marginal_method = st.sidebar.selectbox(
            "Marginal Method:",
            options=list(marginal_methods.keys()),
            format_func=lambda x: marginal_methods[x],
            index=list(marginal_methods.keys()).index(
                st.session_state.get(marginal_method_key, 'kde')
            ),
            key=marginal_method_key,
            help="Method for fitting marginal distributions"
        )

        if marginal_method == 'kde':
            kde_bandwidth = st.sidebar.selectbox(
                "KDE Bandwidth:",
                options=['scott', 'silverman', 'auto'],
                index=['scott', 'silverman', 'auto'].index(
                    st.session_state.get(kde_bandwidth_key, 'scott')
                ),
                key=kde_bandwidth_key,
                help="Bandwidth selection method for kernel density estimation"
            )
        else:
            kde_bandwidth = 'scott'

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

        correlation_threshold = st.sidebar.slider(
            "Correlation Threshold:",
            min_value=0.0,
            max_value=1.0,
            value=st.session_state.get(correlation_threshold_key, 0.1),
            step=0.05,
            key=correlation_threshold_key,
            help="Minimum correlation to consider for modeling"
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

        return {
            'copula_type': copula_type,
            'marginal_method': marginal_method,
            'n_samples': n_samples,
            'random_seed': random_seed,
            'kde_bandwidth': kde_bandwidth,
            'correlation_threshold': correlation_threshold
        }
    
def get_plugin():
    """Factory function to get plugin instance."""
    return CopulaSynthesisPlugin()
