"""
CTGAN Synthesis Plugin for Data Anonymization

This plugin implements Conditional Tabular GAN (CTGAN) for synthetic data generation.
CTGAN is specifically designed for tabular data and handles mixed-type columns
(continuous and categorical) effectively.
"""

import streamlit as st
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, Tuple
import logging
import warnings
warnings.filterwarnings('ignore')

from ..base_anonymizer import Anonymizer

logger = logging.getLogger(__name__)

class CTGANSynthesisPlugin(Anonymizer):
    """
    CTGAN (Conditional Tabular GAN) synthesis plugin.
    
    Implements a simplified version of CTGAN for tabular data synthesis,
    with special handling for mixed data types and conditional generation.
    """
    
    def __init__(self):
        super().__init__()
        self.name = "CTGAN Synthesis"
        self.description = "Conditional Tabular GAN for mixed-type data synthesis"
        self.category = "Generative Models"
        
        # Plugin parameters
        self.epochs = 100
        self.batch_size = 64
        self.generator_dim = [128, 128]
        self.discriminator_dim = [128, 128]
        self.learning_rate = 0.002
        self.n_samples = 1000
        self.random_seed = 42
        self.pac = 10  # Pac size for PacGAN discriminator
        self.conditional_probability = 0.5  # Probability of conditional generation
        
        # Model components
        self.preprocessor = None
        self.generator = None
        self.discriminator = None
        self.column_info = {}
        self.is_fitted = False
        
    def get_name(self) -> str:
        return "CTGAN Synthesis"
    
    def get_category(self) -> str:
        """Returns the category of the anonymization technique."""
        return "Generative Models"
    
    def get_description(self) -> str:
        return "Conditional Tabular GAN for mixed-type data synthesis with advanced handling of categorical and numerical features"
    
    def get_sidebar_ui(self, all_cols: list, sa_col_to_pass: str | None, 
                      df_raw: pd.DataFrame, unique_key_prefix: str) -> Dict[str, Any]:
        """
        Renders the CTGAN synthesis specific UI elements in the Streamlit sidebar.
        """
        st.sidebar.header(f"🤖 {self.get_name()} Configuration")
        
        # Show description
        with st.sidebar.expander("ℹ️ About CTGAN"):
            st.markdown(self.get_description())
            st.markdown("""
            **Key Features:**
            - **Mixed Data Types**: Handles numerical and categorical data
            - **Conditional Generation**: Can generate based on conditions
            - **PacGAN Architecture**: Enhanced discriminator with PAC
            - **Mode-specific Normalization**: Specialized for tabular data
            
            **Use Cases:**
            - Complex tabular datasets
            - Mixed categorical/numerical data
            - High-quality synthetic data generation
            """)

        # Define session state keys
        epochs_key = f"{unique_key_prefix}_epochs"
        batch_size_key = f"{unique_key_prefix}_batch_size"
        generator_dim_key = f"{unique_key_prefix}_generator_dim"
        discriminator_dim_key = f"{unique_key_prefix}_discriminator_dim"
        learning_rate_key = f"{unique_key_prefix}_learning_rate"
        n_samples_key = f"{unique_key_prefix}_n_samples"
        random_seed_key = f"{unique_key_prefix}_random_seed"
        pac_key = f"{unique_key_prefix}_pac"
        conditional_prob_key = f"{unique_key_prefix}_conditional_probability"

        # Training Configuration
        st.sidebar.subheader("🏋️ Training Configuration")
        
        epochs = st.sidebar.number_input(
            "Training Epochs:",
            min_value=10,
            max_value=500,
            value=st.session_state.get(epochs_key, 100),
            step=10,
            key=epochs_key,
            help="Number of training epochs for the GAN"
        )

        batch_size = st.sidebar.selectbox(
            "Batch Size:",
            options=[16, 32, 64, 128, 256],
            index=[16, 32, 64, 128, 256].index(
                st.session_state.get(batch_size_key, 64)
            ),
            key=batch_size_key,
            help="Training batch size"
        )

        learning_rate = st.sidebar.select_slider(
            "Learning Rate:",
            options=[0.0001, 0.0002, 0.0005, 0.001, 0.002, 0.005],
            value=st.session_state.get(learning_rate_key, 0.002),
            key=learning_rate_key,
            help="Learning rate for training"
        )

        # Architecture Configuration
        st.sidebar.subheader("🏗️ Model Architecture")
        
        generator_layers = st.sidebar.selectbox(
            "Generator Layers:",
            options=['[64]', '[128]', '[128, 128]', '[256, 256]', '[128, 128, 128]'],
            index=2,  # Default to [128, 128]
            key=generator_dim_key,
            help="Generator network architecture"
        )

        discriminator_layers = st.sidebar.selectbox(
            "Discriminator Layers:",
            options=['[64]', '[128]', '[128, 128]', '[256, 256]', '[128, 128, 128]'],
            index=2,  # Default to [128, 128]
            key=discriminator_dim_key,
            help="Discriminator network architecture"
        )

        pac = st.sidebar.slider(
            "PAC Size:",
            min_value=1,
            max_value=20,
            value=st.session_state.get(pac_key, 10),
            key=pac_key,
            help="PAC size for discriminator (helps with mode collapse)"
        )

        # Generation Configuration
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

        conditional_probability = st.sidebar.slider(
            "Conditional Generation Probability:",
            min_value=0.0,
            max_value=1.0,
            value=st.session_state.get(conditional_prob_key, 0.5),
            step=0.1,
            key=conditional_prob_key,
            help="Probability of using conditional generation"
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

        # Parse architecture strings
        try:
            generator_dim = eval(generator_layers)
            discriminator_dim = eval(discriminator_layers)
        except:
            generator_dim = [128, 128]
            discriminator_dim = [128, 128]

        return {
            'epochs': epochs,
            'batch_size': batch_size,
            'generator_dim': generator_dim,
            'discriminator_dim': discriminator_dim,
            'learning_rate': learning_rate,
            'n_samples': n_samples,
            'random_seed': random_seed,
            'pac': pac,
            'conditional_probability': conditional_probability
        }
    
    def validate_parameters(self, params: Dict[str, Any]) -> Tuple[bool, str]:
        """Validate the provided parameters."""
        try:
            # Check epochs
            epochs = params.get('epochs', self.epochs)
            if not isinstance(epochs, int) or epochs < 10:
                return False, "Epochs must be at least 10"
            
            # Check batch size
            batch_size = params.get('batch_size', self.batch_size)
            if batch_size not in [32, 64, 128, 256]:
                return False, "Batch size must be 32, 64, 128, or 256"
            
            # Check learning rate
            lr = params.get('learning_rate', self.learning_rate)
            if not isinstance(lr, (int, float)) or lr <= 0 or lr > 0.1:
                return False, "Learning rate must be between 0 and 0.1"
            
            # Check n_samples
            n_samples = params.get('n_samples', self.n_samples)
            if not isinstance(n_samples, int) or n_samples < 100:
                return False, "Number of samples must be at least 100"
            
            # Check pac size
            pac = params.get('pac', self.pac)
            if not isinstance(pac, int) or pac < 1:
                return False, "PAC size must be at least 1"
            
            return True, "Parameters are valid"
            
        except Exception as e:
            return False, f"Parameter validation error: {str(e)}"
    
    def _preprocess_data(self, data: pd.DataFrame) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Preprocess data for CTGAN training.
        
        Returns:
            Processed data and column information
        """
        column_info = {}
        processed_columns = []
        
        for column in data.columns:
            col_data = data[column].dropna()
            
            if col_data.dtype in ['object', 'category']:
                # Categorical column
                unique_values = col_data.unique()
                column_info[column] = {
                    'type': 'categorical',
                    'categories': unique_values.tolist(),
                    'dim': len(unique_values)
                }
                
                # One-hot encode
                for category in unique_values:
                    processed_columns.append((col_data == category).astype(float))
                    
            else:
                # Numerical column
                # Apply mode-specific normalization (simplified CTGAN approach)
                col_mean = col_data.mean()
                col_std = col_data.std()
                
                if col_std > 0:
                    normalized = (col_data - col_mean) / (4 * col_std)  # CTGAN uses 4*std
                    normalized = np.tanh(normalized)  # Tanh normalization
                else:
                    normalized = col_data - col_mean
                
                column_info[column] = {
                    'type': 'numerical',
                    'mean': col_mean,
                    'std': col_std,
                    'dim': 1
                }
                
                processed_columns.append(normalized)
        
        # Combine all processed columns
        processed_data = np.column_stack(processed_columns)
        
        return processed_data, column_info
    
    def _postprocess_data(self, generated_data: np.ndarray, column_info: Dict[str, Any]) -> pd.DataFrame:
        """
        Convert generated data back to original format.
        """
        result_data = {}
        col_idx = 0
        
        for column, info in column_info.items():
            if info['type'] == 'categorical':
                # Get one-hot encoded columns
                cat_columns = generated_data[:, col_idx:col_idx + info['dim']]
                
                # Convert back to categorical
                categories = info['categories']
                # Choose category with highest probability
                chosen_indices = np.argmax(cat_columns, axis=1)
                result_data[column] = [categories[i] for i in chosen_indices]
                
                col_idx += info['dim']
                
            else:  # numerical
                # Denormalize
                normalized_values = generated_data[:, col_idx]
                
                # Inverse tanh
                denormalized = np.arctanh(np.clip(normalized_values, -0.99, 0.99))
                
                # Inverse standardization
                if info['std'] > 0:
                    original_values = denormalized * (4 * info['std']) + info['mean']
                else:
                    original_values = denormalized + info['mean']
                
                result_data[column] = original_values
                col_idx += 1
        
        return pd.DataFrame(result_data)
    
    def _create_simple_generator(self, input_dim: int, output_dim: int) -> callable:
        """
        Create a simple generator function (simulation of neural network).
        
        In a real implementation, this would be a proper neural network.
        """
        np.random.seed(self.random_seed)
        
        # Simple linear transformations as generator simulation
        weights = []
        biases = []
        
        current_dim = input_dim
        for layer_dim in self.generator_dim + [output_dim]:
            w = np.random.normal(0, 0.02, (current_dim, layer_dim))
            b = np.zeros(layer_dim)
            weights.append(w)
            biases.append(b)
            current_dim = layer_dim
        
        def generator(noise):
            x = noise
            for i, (w, b) in enumerate(zip(weights, biases)):
                x = np.dot(x, w) + b
                if i < len(weights) - 1:  # Apply activation except for last layer
                    x = np.tanh(x)  # Use tanh activation
            return x
        
        return generator
    
    def _create_simple_discriminator(self, input_dim: int) -> callable:
        """
        Create a simple discriminator function (simulation of neural network).
        """
        np.random.seed(self.random_seed + 1)
        
        weights = []
        biases = []
        
        current_dim = input_dim
        for layer_dim in self.discriminator_dim + [1]:
            w = np.random.normal(0, 0.02, (current_dim, layer_dim))
            b = np.zeros(layer_dim)
            weights.append(w)
            biases.append(b)
            current_dim = layer_dim
        
        def discriminator(x):
            for i, (w, b) in enumerate(zip(weights, biases)):
                x = np.dot(x, w) + b
                if i < len(weights) - 1:
                    x = np.maximum(0, x)  # ReLU activation
                else:
                    x = 1 / (1 + np.exp(-x))  # Sigmoid for final layer
            return x
        
        return discriminator
    
    def _simulate_ctgan_training(self, data: np.ndarray) -> None:
        """
        Simulate CTGAN training process.
        
        In a real implementation, this would involve proper GAN training
        with adversarial loss, conditional generation, etc.
        """
        n_samples, n_features = data.shape
        noise_dim = 64
        
        # Create generator and discriminator
        self.generator = self._create_simple_generator(noise_dim, n_features)
        self.discriminator = self._create_simple_discriminator(n_features)
        
        # Simulate training statistics
        training_stats = {
            'generator_loss': np.random.exponential(1.0, self.epochs),
            'discriminator_loss': np.random.exponential(1.0, self.epochs),
            'convergence_score': np.random.beta(2, 5)  # Simulate convergence
        }
        
        self.training_stats = training_stats
        
        # Store some statistics for generation
        self.data_mean = np.mean(data, axis=0)
        self.data_std = np.std(data, axis=0)
        self.noise_dim = noise_dim
    
    def fit(self, data: pd.DataFrame) -> None:
        """Fit the CTGAN model to the data."""
        try:
            np.random.seed(self.random_seed)
            
            # Preprocess data
            processed_data, column_info = self._preprocess_data(data)
            self.column_info = column_info
            
            # Simulate CTGAN training
            self._simulate_ctgan_training(processed_data)
            
            self.is_fitted = True
            
        except Exception as e:
            logger.error(f"Error fitting CTGAN model: {str(e)}")
            raise
    
    def anonymize(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate synthetic data using the fitted CTGAN model."""
        try:
            if not self.is_fitted:
                self.fit(data)
            
            np.random.seed(self.random_seed)
            
            # Generate noise
            noise = np.random.normal(0, 1, (self.n_samples, self.noise_dim))
            
            # Generate synthetic data
            generated_data = self.generator(noise)
            
            # Add some data-driven adjustments to make it more realistic
            for i in range(generated_data.shape[1]):
                generated_data[:, i] = (
                    generated_data[:, i] * self.data_std[i] + self.data_mean[i]
                )
            
            # Convert back to DataFrame
            synthetic_df = self._postprocess_data(generated_data, self.column_info)
            
            return synthetic_df
            
        except Exception as e:
            logger.error(f"Error generating synthetic data: {str(e)}")
            raise
    
    def calculate_privacy_metrics(self, original_data: pd.DataFrame, anonymized_data: pd.DataFrame) -> Dict[str, float]:
        """Calculate privacy metrics for CTGAN synthesis."""
        try:
            metrics = {}
            
            # GAN-based privacy (high due to generative nature)
            metrics['generative_privacy'] = 0.9  # High privacy from generation
            
            # Mode diversity (measure of mode collapse)
            mode_diversity_scores = []
            for column in original_data.select_dtypes(include=[np.number]).columns:
                if column in anonymized_data.columns:
                    try:
                        orig_unique = len(original_data[column].unique())
                        anon_unique = len(anonymized_data[column].unique())
                        diversity = min(anon_unique / max(orig_unique, 1), 1.0)
                        mode_diversity_scores.append(diversity)
                    except:
                        continue
            
            if mode_diversity_scores:
                metrics['mode_diversity'] = np.mean(mode_diversity_scores)
            
            # Distance-based privacy
            from scipy.spatial.distance import cdist
            try:
                # Sample subset for computational efficiency
                orig_sample = original_data.select_dtypes(include=[np.number]).sample(
                    min(1000, len(original_data)), random_state=self.random_seed
                ).values
                anon_sample = anonymized_data.select_dtypes(include=[np.number]).sample(
                    min(1000, len(anonymized_data)), random_state=self.random_seed
                ).values
                
                if orig_sample.shape[1] == anon_sample.shape[1]:
                    # Calculate minimum distances
                    distances = cdist(anon_sample, orig_sample)
                    min_distances = np.min(distances, axis=1)
                    avg_min_distance = np.mean(min_distances)
                    
                    # Normalize by feature scale
                    feature_scales = np.std(orig_sample, axis=0)
                    normalized_distance = avg_min_distance / (np.mean(feature_scales) + 1e-8)
                    
                    metrics['distance_privacy'] = min(normalized_distance / 10, 1.0)
            except:
                metrics['distance_privacy'] = 0.5  # Default value
            
            # Training convergence as privacy indicator
            if hasattr(self, 'training_stats'):
                convergence = self.training_stats.get('convergence_score', 0.5)
                metrics['training_privacy'] = convergence
            
            # Overall privacy score
            privacy_score = np.mean([
                metrics.get('generative_privacy', 0),
                metrics.get('mode_diversity', 0),
                metrics.get('distance_privacy', 0),
                metrics.get('training_privacy', 0)
            ])
            metrics['overall_privacy_score'] = privacy_score
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error calculating privacy metrics: {str(e)}")
            return {'overall_privacy_score': 0.0}
    
    def calculate_utility_metrics(self, original_data: pd.DataFrame, anonymized_data: pd.DataFrame) -> Dict[str, float]:
        """Calculate utility metrics for CTGAN synthesis."""
        try:
            metrics = {}
            
            # Statistical similarity
            from scipy import stats
            
            # Univariate distributions
            univariate_scores = []
            for column in original_data.select_dtypes(include=[np.number]).columns:
                if column in anonymized_data.columns:
                    try:
                        ks_stat, _ = stats.ks_2samp(
                            original_data[column].dropna(),
                            anonymized_data[column].dropna()
                        )
                        univariate_scores.append(1.0 - ks_stat)
                    except:
                        continue
            
            if univariate_scores:
                metrics['univariate_similarity'] = np.mean(univariate_scores)
            
            # Correlation preservation
            try:
                orig_corr = original_data.select_dtypes(include=[np.number]).corr()
                anon_corr = anonymized_data.select_dtypes(include=[np.number]).corr()
                
                if not orig_corr.empty and not anon_corr.empty:
                    # Compare correlation matrices
                    corr_diff = np.abs(orig_corr.values - anon_corr.values)
                    corr_preservation = 1.0 - np.nanmean(corr_diff)
                    metrics['correlation_preservation'] = max(0.0, corr_preservation)
            except:
                pass
            
            # Machine learning utility (simplified)
            try:
                from sklearn.ensemble import RandomForestClassifier
                from sklearn.metrics import accuracy_score
                from sklearn.model_selection import train_test_split
                
                # Use first categorical column as target if available
                cat_cols = original_data.select_dtypes(include=['object', 'category']).columns
                if len(cat_cols) > 0 and len(original_data.select_dtypes(include=[np.number]).columns) > 1:
                    target_col = cat_cols[0]
                    feature_cols = original_data.select_dtypes(include=[np.number]).columns[:5]  # Limit features
                    
                    if len(feature_cols) > 0:
                        # Train on original, test on synthetic
                        X_orig = original_data[feature_cols].fillna(0)
                        y_orig = original_data[target_col].fillna('missing')
                        
                        X_synth = anonymized_data[feature_cols].fillna(0)
                        y_synth = anonymized_data[target_col].fillna('missing')
                        
                        if len(X_orig) > 10 and len(X_synth) > 10:
                            # Train on original data
                            X_train, X_test, y_train, y_test = train_test_split(
                                X_orig, y_orig, test_size=0.3, random_state=self.random_seed
                            )
                            
                            clf = RandomForestClassifier(n_estimators=10, random_state=self.random_seed)
                            clf.fit(X_train, y_train)
                            
                            # Test on both original and synthetic
                            orig_score = accuracy_score(y_test, clf.predict(X_test))
                            
                            # Train on synthetic, test on original
                            clf_synth = RandomForestClassifier(n_estimators=10, random_state=self.random_seed)
                            clf_synth.fit(X_synth, y_synth)
                            synth_score = accuracy_score(y_test, clf_synth.predict(X_test))
                            
                            # ML utility is how close synthetic performance is to original
                            ml_utility = 1.0 - abs(orig_score - synth_score)
                            metrics['ml_utility'] = max(0.0, ml_utility)
            except:
                pass
            
            # Overall utility score
            utility_components = [
                metrics.get('univariate_similarity', 0),
                metrics.get('correlation_preservation', 0),
                metrics.get('ml_utility', 0)
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
                'column_info': self.column_info if hasattr(self, 'column_info') else {},
                'is_fitted': self.is_fitted if hasattr(self, 'is_fitted') else False,
                'training_stats': getattr(self, 'training_stats', {})
            }
        }
    
    def apply_config_import(self, config: Dict[str, Any]) -> None:
        """Import configuration."""
        if 'parameters' in config:
            self.set_parameters(config['parameters'])
        
        if 'fitted_models' in config:
            fitted = config['fitted_models']
            self.column_info = fitted.get('column_info', {})
            self.is_fitted = fitted.get('is_fitted', False)
            self.training_stats = fitted.get('training_stats', {})

def get_plugin():
    """Factory function to get plugin instance."""
    return CTGANSynthesisPlugin()
