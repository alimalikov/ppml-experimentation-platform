"""
Marginal Synthesis Plugin for Data Anonymization

This plugin implements marginal-based synthetic data generation, which models
each variable independently without considering dependencies. This provides
strong privacy guarantees but may not preserve complex relationships.
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

class MarginalSynthesisPlugin(Anonymizer):
    """
    Marginal-based synthetic data generation plugin.
    
    Models each variable independently using various distribution fitting methods,
    providing strong privacy at the cost of not preserving correlations.
    """
    
    def __init__(self):
        super().__init__()
        self.name = "Marginal Synthesis"
        self.description = "Generate synthetic data using independent marginal distributions"
        self.category = "Generative Models"
        
        # Plugin parameters
        self.distribution_method = "kde"  # kde, parametric, empirical, histogram
        self.n_samples = 1000
        self.random_seed = 42
        self.kde_bandwidth = "scott"
        self.histogram_bins = 50
        self.parametric_distributions = ["norm", "lognorm", "gamma", "beta"]
        self.privacy_noise = 0.0  # Additional Laplace noise for privacy
        
        # Fitted models
        self.marginal_models = {}
        self.column_types = {}
        self.is_fitted = False
        
    def get_name(self) -> str:
        return "Marginal Synthesis"
    
    def get_category(self) -> str:
        """Returns the category of the anonymization technique."""
        return "Generative Models"
    
    def get_description(self) -> str:
        return "Generate synthetic data using independent marginal distributions for strong privacy preservation"
    
    def get_parameters(self) -> Dict[str, Any]:
        return {
            'distribution_method': self.distribution_method,
            'n_samples': self.n_samples,
            'random_seed': self.random_seed,
            'kde_bandwidth': self.kde_bandwidth,
            'histogram_bins': self.histogram_bins,
            'parametric_distributions': self.parametric_distributions,
            'privacy_noise': self.privacy_noise
        }
    
    def set_parameters(self, params: Dict[str, Any]) -> None:
        self.distribution_method = params.get('distribution_method', self.distribution_method)
        self.n_samples = params.get('n_samples', self.n_samples)
        self.random_seed = params.get('random_seed', self.random_seed)
        self.kde_bandwidth = params.get('kde_bandwidth', self.kde_bandwidth)
        self.histogram_bins = params.get('histogram_bins', self.histogram_bins)
        self.parametric_distributions = params.get('parametric_distributions', self.parametric_distributions)
        self.privacy_noise = params.get('privacy_noise', self.privacy_noise)
    
    def build_sidebar_ui(self) -> Dict[str, Any]:
        """Build the Streamlit sidebar UI for marginal synthesis configuration."""
        st.sidebar.markdown("### Marginal Synthesis Configuration")
        
        # Distribution method selection
        distribution_method = st.sidebar.selectbox(
            "Distribution Method",
            options=["kde", "parametric", "empirical", "histogram"],
            value=self.distribution_method,
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
        
        # Method-specific parameters
        if distribution_method == "kde":
            kde_bandwidth = st.sidebar.selectbox(
                "KDE Bandwidth",
                options=["scott", "silverman", "auto"],
                value=self.kde_bandwidth,
                help="Bandwidth selection method for KDE"
            )
        else:
            kde_bandwidth = self.kde_bandwidth
        
        if distribution_method == "histogram":
            histogram_bins = st.sidebar.number_input(
                "Histogram Bins",
                min_value=10,
                max_value=200,
                value=self.histogram_bins,
                help="Number of bins for histogram-based sampling"
            )
        else:
            histogram_bins = self.histogram_bins
        
        # Privacy noise
        privacy_noise = st.sidebar.slider(
            "Privacy Noise",
            min_value=0.0,
            max_value=1.0,
            value=self.privacy_noise,
            step=0.01,
            help="Additional Laplace noise for enhanced privacy"
        )
        
        # Advanced options
        st.sidebar.markdown("#### Advanced Options")
        
        if distribution_method == "parametric":
            parametric_distributions = st.sidebar.multiselect(
                "Parametric Distributions",
                options=["norm", "lognorm", "gamma", "beta", "expon", "uniform"],
                default=self.parametric_distributions,
                help="Distributions to consider for parametric fitting"
            )
        else:
            parametric_distributions = self.parametric_distributions
        
        return {
            'distribution_method': distribution_method,
            'n_samples': n_samples,
            'random_seed': random_seed,
            'kde_bandwidth': kde_bandwidth,
            'histogram_bins': histogram_bins,
            'parametric_distributions': parametric_distributions,
            'privacy_noise': privacy_noise
        }
    
    def validate_parameters(self, params: Dict[str, Any]) -> Tuple[bool, str]:
        """Validate the provided parameters."""
        try:
            # Check distribution method
            if params.get('distribution_method') not in ['kde', 'parametric', 'empirical', 'histogram']:
                return False, "Invalid distribution method"
            
            # Check n_samples
            n_samples = params.get('n_samples', self.n_samples)
            if not isinstance(n_samples, int) or n_samples < 100:
                return False, "Number of samples must be at least 100"
            
            # Check histogram bins
            bins = params.get('histogram_bins', self.histogram_bins)
            if not isinstance(bins, int) or bins < 10:
                return False, "Histogram bins must be at least 10"
            
            # Check privacy noise
            noise = params.get('privacy_noise', self.privacy_noise)
            if not isinstance(noise, (int, float)) or noise < 0 or noise > 1:
                return False, "Privacy noise must be between 0 and 1"
            
            return True, "Parameters are valid"
            
        except Exception as e:
            return False, f"Parameter validation error: {str(e)}"
    
    def _fit_kde_distribution(self, values: np.ndarray) -> Dict[str, Any]:
        """Fit KDE to numerical data."""
        try:
            kde = gaussian_kde(values, bw_method=self.kde_bandwidth)
            return {
                'type': 'kde',
                'model': kde,
                'min_val': values.min(),
                'max_val': values.max(),
                'std': values.std()
            }
        except Exception as e:
            logger.warning(f"KDE fitting failed: {e}, falling back to empirical")
            return self._fit_empirical_distribution(values)
    
    def _fit_parametric_distribution(self, values: np.ndarray) -> Dict[str, Any]:
        """Fit parametric distributions and select the best one."""
        best_dist = None
        best_params = None
        best_score = -np.inf
        
        for dist_name in self.parametric_distributions:
            try:
                dist = getattr(stats, dist_name)
                params = dist.fit(values)
                
                # Calculate goodness of fit (using log-likelihood)
                score = np.sum(dist.logpdf(values, *params))
                
                if score > best_score:
                    best_score = score
                    best_dist = dist_name
                    best_params = params
                    
            except Exception as e:
                logger.warning(f"Failed to fit {dist_name}: {e}")
                continue
        
        if best_dist is None:
            # Fall back to empirical if no parametric distribution works
            return self._fit_empirical_distribution(values)
        
        return {
            'type': 'parametric',
            'distribution': best_dist,
            'parameters': best_params,
            'score': best_score
        }
    
    def _fit_empirical_distribution(self, values: np.ndarray) -> Dict[str, Any]:
        """Fit empirical distribution."""
        return {
            'type': 'empirical',
            'values': values.copy(),
            'unique_values': np.unique(values),
            'quantiles': np.percentile(values, np.linspace(0, 100, 1000))
        }
    
    def _fit_histogram_distribution(self, values: np.ndarray) -> Dict[str, Any]:
        """Fit histogram-based distribution."""
        hist, bin_edges = np.histogram(values, bins=self.histogram_bins, density=True)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        
        return {
            'type': 'histogram',
            'hist': hist,
            'bin_edges': bin_edges,
            'bin_centers': bin_centers
        }
    
    def _fit_categorical_distribution(self, values: pd.Series) -> Dict[str, Any]:
        """Fit categorical distribution."""
        value_counts = values.value_counts(normalize=True)
        
        # Add privacy noise if specified
        if self.privacy_noise > 0:
            noise = np.random.laplace(0, self.privacy_noise, len(value_counts))
            noisy_probs = value_counts.values + noise
            noisy_probs = np.maximum(noisy_probs, 0)  # Ensure non-negative
            noisy_probs = noisy_probs / noisy_probs.sum()  # Normalize
            value_counts = pd.Series(noisy_probs, index=value_counts.index)
        
        return {
            'type': 'categorical',
            'categories': value_counts.index.tolist(),
            'probabilities': value_counts.values.tolist()
        }
    
    def fit(self, data: pd.DataFrame) -> None:
        """Fit marginal distributions to the data."""
        try:
            np.random.seed(self.random_seed)
            self.marginal_models = {}
            self.column_types = {}
            
            for column in data.columns:
                values = data[column].dropna()
                
                if values.dtype in ['object', 'category']:
                    # Categorical variable
                    self.column_types[column] = 'categorical'
                    self.marginal_models[column] = self._fit_categorical_distribution(values)
                else:
                    # Numerical variable
                    self.column_types[column] = 'numerical'
                    values_array = values.values
                    
                    if self.distribution_method == 'kde':
                        self.marginal_models[column] = self._fit_kde_distribution(values_array)
                    elif self.distribution_method == 'parametric':
                        self.marginal_models[column] = self._fit_parametric_distribution(values_array)
                    elif self.distribution_method == 'histogram':
                        self.marginal_models[column] = self._fit_histogram_distribution(values_array)
                    else:  # empirical
                        self.marginal_models[column] = self._fit_empirical_distribution(values_array)
            
            self.is_fitted = True
            
        except Exception as e:
            logger.error(f"Error fitting marginal models: {str(e)}")
            raise
    
    def _sample_from_model(self, column: str, n_samples: int) -> np.ndarray:
        """Sample from a fitted marginal model."""
        model = self.marginal_models[column]
        
        if model['type'] == 'categorical':
            # Sample from categorical distribution
            samples = np.random.choice(
                model['categories'],
                size=n_samples,
                p=model['probabilities']
            )
            return samples
        
        elif model['type'] == 'kde':
            # Sample from KDE
            samples = model['model'].resample(n_samples)[0]
            # Clip to original range
            samples = np.clip(samples, model['min_val'], model['max_val'])
            
            # Add privacy noise if specified
            if self.privacy_noise > 0:
                noise = np.random.laplace(0, self.privacy_noise * model['std'], n_samples)
                samples += noise
            
            return samples
        
        elif model['type'] == 'parametric':
            # Sample from parametric distribution
            dist = getattr(stats, model['distribution'])
            samples = dist.rvs(*model['parameters'], size=n_samples)
            
            # Add privacy noise if specified
            if self.privacy_noise > 0:
                noise = np.random.laplace(0, self.privacy_noise * np.std(samples), n_samples)
                samples += noise
            
            return samples
        
        elif model['type'] == 'histogram':
            # Sample from histogram
            # Choose bins according to histogram probabilities
            bin_probs = model['hist'] * np.diff(model['bin_edges'])
            bin_probs = bin_probs / bin_probs.sum()  # Normalize
            
            chosen_bins = np.random.choice(len(model['bin_centers']), size=n_samples, p=bin_probs)
            # Sample uniformly within chosen bins
            bin_width = model['bin_edges'][1] - model['bin_edges'][0]
            samples = model['bin_centers'][chosen_bins] + np.random.uniform(
                -bin_width/2, bin_width/2, n_samples
            )
            
            # Add privacy noise if specified
            if self.privacy_noise > 0:
                noise = np.random.laplace(0, self.privacy_noise * bin_width, n_samples)
                samples += noise
            
            return samples
        
        else:  # empirical
            # Sample from empirical distribution
            samples = np.random.choice(model['values'], size=n_samples, replace=True)
            
            # Add privacy noise if specified
            if self.privacy_noise > 0:
                noise = np.random.laplace(0, self.privacy_noise * np.std(model['values']), n_samples)
                samples += noise
            
            return samples
    
    def anonymize(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate synthetic data using fitted marginal distributions."""
        try:
            if not self.is_fitted:
                self.fit(data)
            
            np.random.seed(self.random_seed)
            
            synthetic_data = pd.DataFrame()
            
            for column in data.columns:
                samples = self._sample_from_model(column, self.n_samples)
                synthetic_data[column] = samples
            
            return synthetic_data
            
        except Exception as e:
            logger.error(f"Error generating synthetic data: {str(e)}")
            raise
    
    def calculate_privacy_metrics(self, original_data: pd.DataFrame, anonymized_data: pd.DataFrame) -> Dict[str, float]:
        """Calculate privacy metrics for marginal synthesis."""
        try:
            metrics = {}
            
            # Independence privacy (very high since no correlations preserved)
            metrics['independence_privacy'] = 1.0
            
            # Marginal privacy (based on method and noise)
            if self.distribution_method in ['kde', 'empirical']:
                base_privacy = 0.7  # Moderate privacy for non-parametric methods
            else:
                base_privacy = 0.8  # Higher privacy for parametric methods
            
            # Additional privacy from noise
            noise_privacy = min(self.privacy_noise * 0.5, 0.3)
            marginal_privacy = min(base_privacy + noise_privacy, 1.0)
            metrics['marginal_privacy'] = marginal_privacy
            
            # Distribution distance (privacy through distortion)
            distance_scores = []
            for column in original_data.select_dtypes(include=[np.number]).columns:
                if column in anonymized_data.columns:
                    try:
                        ks_stat, _ = stats.ks_2samp(
                            original_data[column].dropna(),
                            anonymized_data[column].dropna()
                        )
                        distance_scores.append(ks_stat)  # Higher distance = more privacy
                    except:
                        continue
            
            if distance_scores:
                metrics['distribution_distance'] = np.mean(distance_scores)
            
            # Overall privacy score
            privacy_score = (
                metrics.get('independence_privacy', 0) * 0.4 +
                metrics.get('marginal_privacy', 0) * 0.4 +
                metrics.get('distribution_distance', 0) * 0.2
            )
            metrics['overall_privacy_score'] = privacy_score
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error calculating privacy metrics: {str(e)}")
            return {'overall_privacy_score': 0.0}
    
    def calculate_utility_metrics(self, original_data: pd.DataFrame, anonymized_data: pd.DataFrame) -> Dict[str, float]:
        """Calculate utility metrics for marginal synthesis."""
        try:
            metrics = {}
            
            # Marginal utility (distribution similarity)
            marginal_similarities = []
            for column in original_data.columns:
                if column in anonymized_data.columns:
                    if self.column_types.get(column) == 'numerical':
                        try:
                            # KS test for numerical columns
                            ks_stat, _ = stats.ks_2samp(
                                original_data[column].dropna(),
                                anonymized_data[column].dropna()
                            )
                            similarity = 1.0 - ks_stat
                            marginal_similarities.append(similarity)
                        except:
                            continue
                    else:
                        # Chi-square test for categorical columns
                        try:
                            orig_counts = original_data[column].value_counts()
                            anon_counts = anonymized_data[column].value_counts()
                            
                            # Align categories
                            all_cats = set(orig_counts.index) | set(anon_counts.index)
                            orig_aligned = [orig_counts.get(cat, 0) for cat in all_cats]
                            anon_aligned = [anon_counts.get(cat, 0) for cat in all_cats]
                            
                            if sum(anon_aligned) > 0:
                                # Normalize to probabilities
                                orig_probs = np.array(orig_aligned) / sum(orig_aligned)
                                anon_probs = np.array(anon_aligned) / sum(anon_aligned)
                                
                                # Jensen-Shannon divergence
                                m = (orig_probs + anon_probs) / 2
                                js_div = (stats.entropy(orig_probs, m) + stats.entropy(anon_probs, m)) / 2
                                similarity = 1.0 - js_div
                                marginal_similarities.append(max(0, similarity))
                        except:
                            continue
            
            if marginal_similarities:
                metrics['marginal_utility'] = np.mean(marginal_similarities)
            
            # Statistical moments preservation
            num_cols = original_data.select_dtypes(include=[np.number]).columns
            if len(num_cols) > 0:
                mean_errors = []
                std_errors = []
                
                for col in num_cols:
                    if col in anonymized_data.columns:
                        orig_mean = original_data[col].mean()
                        anon_mean = anonymized_data[col].mean()
                        if abs(orig_mean) > 1e-8:
                            mean_errors.append(abs(orig_mean - anon_mean) / abs(orig_mean))
                        
                        orig_std = original_data[col].std()
                        anon_std = anonymized_data[col].std()
                        if abs(orig_std) > 1e-8:
                            std_errors.append(abs(orig_std - anon_std) / abs(orig_std))
                
                if mean_errors:
                    metrics['mean_preservation'] = 1.0 - np.mean(mean_errors)
                if std_errors:
                    metrics['std_preservation'] = 1.0 - np.mean(std_errors)
            
            # Note: Correlation preservation is intentionally not measured
            # as marginal synthesis explicitly ignores correlations
            
            # Overall utility score
            utility_components = [
                metrics.get('marginal_utility', 0),
                metrics.get('mean_preservation', 0),
                metrics.get('std_preservation', 0)
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
                'marginal_models': self.marginal_models if hasattr(self, 'marginal_models') else {},
                'column_types': self.column_types if hasattr(self, 'column_types') else {},
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
            self.column_types = fitted.get('column_types', {})
            self.is_fitted = fitted.get('is_fitted', False)
        
    def get_sidebar_ui(self, all_cols: list, sa_col_to_pass: str | None, 
                      df_raw: pd.DataFrame, unique_key_prefix: str) -> Dict[str, Any]:
        """
        Renders the marginal synthesis specific UI elements in the Streamlit sidebar.
        """
        st.sidebar.header(f"📊 {self.get_name()} Configuration")
        
        # Show description
        with st.sidebar.expander("ℹ️ About Marginal Synthesis"):
            st.markdown(self.get_description())
            st.markdown("""
            **Key Features:**
            - **Independence**: Models each variable separately
            - **Strong Privacy**: No cross-variable information leakage
            - **Multiple Methods**: KDE, parametric, empirical, histogram
            - **Privacy Noise**: Optional differential privacy noise
            
            **Use Cases:**
            - High privacy requirements
            - Simple data exploration
            - When correlations are not critical
            """)

        # Define session state keys
        distribution_method_key = f"{unique_key_prefix}_distribution_method"
        n_samples_key = f"{unique_key_prefix}_n_samples"
        random_seed_key = f"{unique_key_prefix}_random_seed"
        kde_bandwidth_key = f"{unique_key_prefix}_kde_bandwidth"
        histogram_bins_key = f"{unique_key_prefix}_histogram_bins"
        privacy_noise_key = f"{unique_key_prefix}_privacy_noise"

        # Distribution Configuration
        st.sidebar.subheader("📈 Distribution Modeling")
        
        distribution_methods = {
            'kde': 'Kernel Density Estimation',
            'parametric': 'Parametric Fitting',
            'empirical': 'Empirical Distribution',
            'histogram': 'Histogram-based'
        }
        
        distribution_method = st.sidebar.selectbox(
            "Distribution Method:",
            options=list(distribution_methods.keys()),
            format_func=lambda x: distribution_methods[x],
            index=list(distribution_methods.keys()).index(
                st.session_state.get(distribution_method_key, 'kde')
            ),
            key=distribution_method_key,
            help="Method for modeling marginal distributions"
        )

        # Method-specific parameters
        if distribution_method == 'kde':
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

        if distribution_method == 'histogram':
            histogram_bins = st.sidebar.slider(
                "Histogram Bins:",
                min_value=10,
                max_value=100,
                value=st.session_state.get(histogram_bins_key, 50),
                key=histogram_bins_key,
                help="Number of bins for histogram-based modeling"
            )
        else:
            histogram_bins = 50

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

        # Privacy Configuration
        st.sidebar.subheader("🔐 Privacy Enhancement")
        
        privacy_noise = st.sidebar.slider(
            "Privacy Noise Level:",
            min_value=0.0,
            max_value=1.0,
            value=st.session_state.get(privacy_noise_key, 0.0),
            step=0.05,
            key=privacy_noise_key,
            help="Additional Laplace noise for differential privacy (0 = no noise)"
        )

        if privacy_noise > 0:
            st.sidebar.info(f"Added DP noise level: {privacy_noise}")

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
            'distribution_method': distribution_method,
            'n_samples': n_samples,
            'random_seed': random_seed,
            'kde_bandwidth': kde_bandwidth,
            'histogram_bins': histogram_bins,
            'privacy_noise': privacy_noise
        }
    
def get_plugin():
    """Factory function to get plugin instance."""
    return MarginalSynthesisPlugin()
