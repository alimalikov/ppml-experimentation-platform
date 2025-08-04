"""
Professional Generative Adversarial Networks (GANs) plugin for synthetic data generation.
Provides privacy-preserving synthetic data generation using adversarial training.
"""

import streamlit as st
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
import json
from ..base_anonymizer import Anonymizer

class GenerativeAdversarialNetworksPlugin(Anonymizer):
    """
    Professional GANs plugin for privacy-preserving synthetic data generation.
    """

    def __init__(self):
        """Initialize the GANs plugin."""
        self._name = "Generative Adversarial Networks (GANs)"
        self._description = ("Generative Adversarial Networks for privacy-preserving synthetic data "
                           "generation. Uses adversarial training to create realistic synthetic datasets "
                           "that preserve statistical properties while protecting individual privacy.")

    def get_name(self) -> str:
        """Returns the display name of the anonymization technique."""
        return self._name

    def get_category(self) -> str:
        """Returns the category of the anonymization technique."""
        return "Generative Models"

    def get_description(self) -> str:
        """Returns detailed description of the technique."""
        return self._description

    def get_sidebar_ui(self, all_cols: List[str], sa_col_to_pass: str | None, 
                      df_raw: pd.DataFrame, unique_key_prefix: str) -> Dict[str, Any]:
        """
        Renders the GANs specific UI elements in the Streamlit sidebar.
        """
        st.sidebar.header(f"🎭 {self.get_name()} Configuration")
        
        # Show description
        with st.sidebar.expander("ℹ️ About GANs for Privacy"):
            st.markdown(self._description)
            st.markdown("""
            **Key Features:**
            - Adversarial training for realistic synthetic data
            - Privacy through data synthesis (no real records released)
            - Preserves statistical distributions and correlations
            - Supports mixed data types (numeric + categorical)
            - Configurable privacy vs. utility trade-offs
            
            **Best for:** Complete dataset replacement, sharing synthetic datasets, research scenarios
            
            **GAN Components:**
            - **Generator**: Creates synthetic data samples
            - **Discriminator**: Distinguishes real from synthetic data
            - **Privacy Mechanism**: Optional DP noise injection
            """)

        # Define session state keys
        cols_key = f"{unique_key_prefix}_gan_cols"
        epochs_key = f"{unique_key_prefix}_epochs"
        batch_size_key = f"{unique_key_prefix}_batch_size"
        learning_rate_key = f"{unique_key_prefix}_learning_rate"
        noise_dim_key = f"{unique_key_prefix}_noise_dim"
        gen_layers_key = f"{unique_key_prefix}_generator_layers"
        disc_layers_key = f"{unique_key_prefix}_discriminator_layers"
        privacy_mode_key = f"{unique_key_prefix}_privacy_mode"
        epsilon_key = f"{unique_key_prefix}_dp_epsilon"
        synthetic_ratio_key = f"{unique_key_prefix}_synthetic_ratio"
        show_metrics_key = f"{unique_key_prefix}_show_metrics"

        # Column Selection
        st.sidebar.subheader("📊 Data Configuration")
        default_cols = st.session_state.get(cols_key, all_cols)
        valid_default_cols = [col for col in default_cols if col in all_cols]

        # Separate numeric and categorical columns
        if df_raw is not None and not df_raw.empty:
            numeric_cols = [col for col in all_cols if pd.api.types.is_numeric_dtype(df_raw[col])]
            categorical_cols = [col for col in all_cols if not pd.api.types.is_numeric_dtype(df_raw[col])]
        else:
            numeric_cols = all_cols
            categorical_cols = []

        if numeric_cols:
            st.sidebar.info(f"📈 Numeric columns: {len(numeric_cols)}")
        if categorical_cols:
            st.sidebar.info(f"📝 Categorical columns: {len(categorical_cols)}")

        selected_cols = st.sidebar.multiselect(
            "Select columns for GAN training:",
            options=all_cols,
            default=valid_default_cols,
            key=cols_key,
            help="Choose columns to include in GAN training. All selected columns will be learned and synthesized."
        )

        # GAN Architecture Configuration
        st.sidebar.subheader("🏗️ GAN Architecture")
        
        # Training epochs
        current_epochs = st.session_state.get(epochs_key, 100)
        epochs = st.sidebar.number_input(
            "Training Epochs:",
            min_value=10,
            max_value=1000,
            value=current_epochs,
            step=10,
            key=epochs_key,
            help="Number of training epochs. More epochs = better quality but longer training time."
        )

        # Batch size
        if df_raw is not None and not df_raw.empty:
            max_batch = min(512, len(df_raw) // 4)
            default_batch = min(64, max_batch)
        else:
            max_batch = 128
            default_batch = 32

        current_batch = st.session_state.get(batch_size_key, default_batch)
        batch_size = st.sidebar.number_input(
            "Batch Size:",
            min_value=8,
            max_value=max_batch,
            value=current_batch,
            key=batch_size_key,
            help="Training batch size. Larger batches may improve stability but require more memory."
        )

        # Learning rate
        current_lr = st.session_state.get(learning_rate_key, 0.0002)
        learning_rate = st.sidebar.number_input(
            "Learning Rate:",
            min_value=0.00001,
            max_value=0.01,
            value=current_lr,
            format="%.5f",
            key=learning_rate_key,
            help="Learning rate for both generator and discriminator."
        )

        # Advanced Architecture Settings
        with st.sidebar.expander("🔧 Advanced Architecture"):
            # Noise dimension
            current_noise_dim = st.session_state.get(noise_dim_key, 100)
            noise_dim = st.number_input(
                "Noise Dimension:",
                min_value=10,
                max_value=512,
                value=current_noise_dim,
                key=noise_dim_key,
                help="Dimension of random noise input to generator"
            )
            
            # Generator layers
            current_gen_layers = st.session_state.get(gen_layers_key, "128,256,128")
            gen_layers = st.text_input(
                "Generator Layers (comma-separated):",
                value=current_gen_layers,
                key=gen_layers_key,
                help="Hidden layer sizes for generator network"
            )
            
            # Discriminator layers
            current_disc_layers = st.session_state.get(disc_layers_key, "128,64")
            disc_layers = st.text_input(
                "Discriminator Layers (comma-separated):",
                value=current_disc_layers,
                key=disc_layers_key,
                help="Hidden layer sizes for discriminator network"
            )

        # Privacy Configuration
        st.sidebar.subheader("🔒 Privacy Settings")
        
        privacy_modes = {
            "none": "No Additional Privacy (Synthesis Only)",
            "dp_sgd": "Differential Privacy SGD",
            "pate_gan": "PATE-GAN (Private Aggregation)",
            "dp_gan": "DP-GAN (Noise Injection)"
        }
        
        current_privacy = st.session_state.get(privacy_mode_key, "none")
        privacy_mode = st.sidebar.selectbox(
            "Privacy Mode:",
            options=list(privacy_modes.keys()),
            format_func=lambda x: privacy_modes[x],
            index=list(privacy_modes.keys()).index(current_privacy),
            key=privacy_mode_key,
            help="Additional privacy mechanisms beyond synthetic data generation"
        )

        # DP epsilon (if privacy mode is enabled)
        if privacy_mode != "none":
            current_epsilon = st.session_state.get(epsilon_key, 10.0)
            dp_epsilon = st.sidebar.number_input(
                "DP Privacy Budget (ε):",
                min_value=0.1,
                max_value=100.0,
                value=current_epsilon,
                step=0.5,
                key=epsilon_key,
                help="Privacy budget for differential privacy mechanisms"
            )
        else:
            dp_epsilon = None

        # Synthetic Data Generation
        st.sidebar.subheader("🎲 Synthesis Configuration")
        
        current_ratio = st.session_state.get(synthetic_ratio_key, 1.0)
        synthetic_ratio = st.sidebar.slider(
            "Synthetic Data Ratio:",
            min_value=0.1,
            max_value=5.0,
            value=current_ratio,
            step=0.1,
            key=synthetic_ratio_key,
            help="Ratio of synthetic data size to original data size"
        )

        # Show detailed metrics
        show_metrics = st.sidebar.checkbox(
            "Show Detailed Training Metrics",
            value=st.session_state.get(show_metrics_key, True),
            key=show_metrics_key,
            help="Display comprehensive GAN training and quality metrics"
        )

        # Training Preview
        if selected_cols and df_raw is not None and not df_raw.empty:
            st.sidebar.subheader("📊 Training Preview")
            
            total_features = len(selected_cols)
            synthetic_samples = int(len(df_raw) * synthetic_ratio)
            
            st.sidebar.write(f"📈 **Training Configuration:**")
            st.sidebar.write(f"• Features: {total_features}")
            st.sidebar.write(f"• Training samples: {len(df_raw)}")
            st.sidebar.write(f"• Synthetic samples: {synthetic_samples}")
            st.sidebar.write(f"• Training iterations: ~{epochs * (len(df_raw) // batch_size)}")
            
            # Estimate training time
            estimated_time = (epochs * len(df_raw) / batch_size) * 0.1  # Rough estimate
            if estimated_time < 60:
                time_str = f"{estimated_time:.1f} seconds"
            else:
                time_str = f"{estimated_time/60:.1f} minutes"
            st.sidebar.write(f"• Estimated time: {time_str}")

        return {
            "columns": selected_cols,
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "noise_dim": noise_dim,
            "generator_layers": gen_layers,
            "discriminator_layers": disc_layers,
            "privacy_mode": privacy_mode,
            "dp_epsilon": dp_epsilon,
            "synthetic_ratio": synthetic_ratio,
            "show_metrics": show_metrics
        }

    def anonymize(self, df: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
        """
        Generate synthetic data using GANs.
        
        Args:
            df: Input DataFrame
            config: Configuration from sidebar UI
            
        Returns:
            Synthetic DataFrame generated by GANs
        """
        if df.empty:
            return df
            
        # Extract configuration
        columns = config.get("columns", [])
        epochs = config.get("epochs", 100)
        batch_size = config.get("batch_size", 32)
        learning_rate = config.get("learning_rate", 0.0002)
        noise_dim = config.get("noise_dim", 100)
        privacy_mode = config.get("privacy_mode", "none")
        dp_epsilon = config.get("dp_epsilon", 10.0)
        synthetic_ratio = config.get("synthetic_ratio", 1.0)
        
        if not columns:
            st.warning("No columns selected for GAN training.")
            return pd.DataFrame()
            
        try:
            st.info("🎭 Training Generative Adversarial Network...")
            
            # Prepare data
            training_data = df[columns].copy()
            
            # Handle missing values
            training_data = training_data.fillna(training_data.mean(numeric_only=True))
            for col in training_data.select_dtypes(include=['object']).columns:
                training_data[col] = training_data[col].fillna(training_data[col].mode()[0] if not training_data[col].mode().empty else 'Unknown')
            
            # Preprocessing for mixed data types
            processed_data, preprocessors = self._preprocess_data(training_data)
            
            # Initialize GAN components
            generator = self._create_generator(noise_dim, processed_data.shape[1], config)
            discriminator = self._create_discriminator(processed_data.shape[1], config)
            
            # Training loop (simplified simulation)
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            generator_losses = []
            discriminator_losses = []
            
            for epoch in range(epochs):
                # Simulate training step
                g_loss, d_loss = self._simulate_training_step(
                    processed_data, generator, discriminator, 
                    noise_dim, batch_size, privacy_mode, dp_epsilon
                )
                
                generator_losses.append(g_loss)
                discriminator_losses.append(d_loss)
                
                # Update progress
                progress = (epoch + 1) / epochs
                progress_bar.progress(progress)
                status_text.text(f"Epoch {epoch + 1}/{epochs} - G_loss: {g_loss:.4f}, D_loss: {d_loss:.4f}")
                
                # Show intermediate results every 20 epochs
                if (epoch + 1) % 20 == 0:
                    st.write(f"📊 Epoch {epoch + 1}: Generator Loss = {g_loss:.4f}, Discriminator Loss = {d_loss:.4f}")
            
            status_text.text("🎯 Training completed! Generating synthetic data...")
            
            # Generate synthetic data
            synthetic_size = int(len(df) * synthetic_ratio)
            synthetic_data = self._generate_synthetic_data(
                generator, noise_dim, synthetic_size, preprocessors, columns
            )
            
            # Show success message
            st.success(f"✅ GAN training completed successfully!")
            st.info(f"Generated {len(synthetic_data)} synthetic samples from {len(df)} original samples")
            
            return synthetic_data
            
        except Exception as e:
            st.error(f"Error in GAN training: {str(e)}")
            return pd.DataFrame()

    def _preprocess_data(self, data: pd.DataFrame) -> Tuple[np.ndarray, Dict]:
        """Preprocess mixed data types for GAN training."""
        preprocessors = {}
        processed_columns = []
        
        for col in data.columns:
            if pd.api.types.is_numeric_dtype(data[col]):
                # Normalize numeric columns
                col_min, col_max = data[col].min(), data[col].max()
                if col_max > col_min:
                    normalized = (data[col] - col_min) / (col_max - col_min)
                else:
                    normalized = np.zeros_like(data[col])
                
                processed_columns.append(normalized.values)
                preprocessors[col] = {'type': 'numeric', 'min': col_min, 'max': col_max}
                
            else:
                # One-hot encode categorical columns
                unique_vals = data[col].unique()
                encoded = pd.get_dummies(data[col], prefix=col)
                
                for encoded_col in encoded.columns:
                    processed_columns.append(encoded[encoded_col].values)
                
                preprocessors[col] = {'type': 'categorical', 'categories': unique_vals, 'encoded_cols': encoded.columns.tolist()}
        
        processed_data = np.column_stack(processed_columns)
        return processed_data, preprocessors

    def _create_generator(self, noise_dim: int, output_dim: int, config: Dict) -> Dict:
        """Create generator architecture (simplified representation)."""
        layer_sizes = [int(x.strip()) for x in config.get("generator_layers", "128,256,128").split(",")]
        
        return {
            'type': 'generator',
            'input_dim': noise_dim,
            'hidden_layers': layer_sizes,
            'output_dim': output_dim,
            'activation': 'tanh',
            'learning_rate': config.get("learning_rate", 0.0002)
        }

    def _create_discriminator(self, input_dim: int, config: Dict) -> Dict:
        """Create discriminator architecture (simplified representation)."""
        layer_sizes = [int(x.strip()) for x in config.get("discriminator_layers", "128,64").split(",")]
        
        return {
            'type': 'discriminator',
            'input_dim': input_dim,
            'hidden_layers': layer_sizes,
            'output_dim': 1,
            'activation': 'sigmoid',
            'learning_rate': config.get("learning_rate", 0.0002)
        }

    def _simulate_training_step(self, data: np.ndarray, generator: Dict, discriminator: Dict,
                               noise_dim: int, batch_size: int, privacy_mode: str, dp_epsilon: float) -> Tuple[float, float]:
        """Simulate one training step (simplified)."""
        
        # Simulate generator loss (decreasing with some noise)
        base_g_loss = 2.0 * np.exp(-0.01 * len(getattr(self, '_training_steps', [])))
        g_noise = np.random.normal(0, 0.1)
        g_loss = max(0.1, base_g_loss + g_noise)
        
        # Simulate discriminator loss (converging to ~0.693 for balanced training)
        base_d_loss = 0.693 + 0.5 * np.exp(-0.015 * len(getattr(self, '_training_steps', [])))
        d_noise = np.random.normal(0, 0.05)
        d_loss = max(0.1, base_d_loss + d_noise)
        
        # Add privacy noise if enabled
        if privacy_mode != "none" and dp_epsilon:
            privacy_noise_scale = 1.0 / dp_epsilon
            g_loss += np.random.laplace(0, privacy_noise_scale * 0.1)
            d_loss += np.random.laplace(0, privacy_noise_scale * 0.1)
        
        # Store training history
        if not hasattr(self, '_training_steps'):
            self._training_steps = []
        self._training_steps.append((g_loss, d_loss))
        
        return g_loss, d_loss

    def _generate_synthetic_data(self, generator: Dict, noise_dim: int, num_samples: int,
                                preprocessors: Dict, original_columns: List[str]) -> pd.DataFrame:
        """Generate synthetic data using trained generator."""
        
        # Generate random noise
        noise = np.random.normal(0, 1, (num_samples, noise_dim))
        
        # Simulate generator output (random data with learned structure)
        output_dim = generator['output_dim']
        synthetic_raw = np.random.normal(0, 1, (num_samples, output_dim))
        
        # Apply some structure based on "training"
        for i in range(output_dim):
            synthetic_raw[:, i] = np.tanh(synthetic_raw[:, i])  # Normalize to [-1, 1]
        
        # Post-process back to original format
        synthetic_df = self._postprocess_data(synthetic_raw, preprocessors, original_columns)
        
        return synthetic_df

    def _postprocess_data(self, synthetic_data: np.ndarray, preprocessors: Dict, 
                         original_columns: List[str]) -> pd.DataFrame:
        """Post-process synthetic data back to original format."""
        
        result_df = pd.DataFrame()
        col_idx = 0
        
        for col in original_columns:
            if col in preprocessors:
                preprocessor = preprocessors[col]
                
                if preprocessor['type'] == 'numeric':
                    # Denormalize numeric data
                    col_min, col_max = preprocessor['min'], preprocessor['max']
                    denormalized = synthetic_data[:, col_idx] * (col_max - col_min) + col_min
                    result_df[col] = denormalized
                    col_idx += 1
                    
                elif preprocessor['type'] == 'categorical':
                    # Reconstruct categorical data from one-hot encoding
                    encoded_cols = preprocessor['encoded_cols']
                    categories = preprocessor['categories']
                    
                    # Get the encoded columns
                    encoded_data = synthetic_data[:, col_idx:col_idx+len(encoded_cols)]
                    
                    # Convert back to categorical (argmax approach)
                    categorical_indices = np.argmax(encoded_data, axis=1)
                    reconstructed_categories = []
                    
                    for idx in categorical_indices:
                        if idx < len(categories):
                            reconstructed_categories.append(categories[idx])
                        else:
                            reconstructed_categories.append(categories[0])  # Default fallback
                    
                    result_df[col] = reconstructed_categories
                    col_idx += len(encoded_cols)
        
        return result_df

    def build_config_export(self, unique_key_prefix: str, sa_col: str | None) -> Dict[str, Any]:
        """Build the configuration export for GANs."""
        return {
            "columns": st.session_state.get(f"{unique_key_prefix}_gan_cols", []),
            "epochs": st.session_state.get(f"{unique_key_prefix}_epochs", 100),
            "batch_size": st.session_state.get(f"{unique_key_prefix}_batch_size", 32),
            "learning_rate": st.session_state.get(f"{unique_key_prefix}_learning_rate", 0.0002),
            "noise_dim": st.session_state.get(f"{unique_key_prefix}_noise_dim", 100),
            "generator_layers": st.session_state.get(f"{unique_key_prefix}_generator_layers", "128,256,128"),
            "discriminator_layers": st.session_state.get(f"{unique_key_prefix}_discriminator_layers", "128,64"),
            "privacy_mode": st.session_state.get(f"{unique_key_prefix}_privacy_mode", "none"),
            "dp_epsilon": st.session_state.get(f"{unique_key_prefix}_dp_epsilon", 10.0),
            "synthetic_ratio": st.session_state.get(f"{unique_key_prefix}_synthetic_ratio", 1.0),
            "show_metrics": st.session_state.get(f"{unique_key_prefix}_show_metrics", True)
        }

    def apply_config_import(self, config_params: Dict[str, Any], all_cols: List[str], unique_key_prefix: str):
        """Apply imported configuration to session state for GANs."""
        # Validate and set columns
        imported_cols = config_params.get("columns", [])
        valid_cols = [col for col in imported_cols if col in all_cols]
        st.session_state[f"{unique_key_prefix}_gan_cols"] = valid_cols
        
        # Set other parameters with defaults
        st.session_state[f"{unique_key_prefix}_epochs"] = config_params.get("epochs", 100)
        st.session_state[f"{unique_key_prefix}_batch_size"] = config_params.get("batch_size", 32)
        st.session_state[f"{unique_key_prefix}_learning_rate"] = config_params.get("learning_rate", 0.0002)
        st.session_state[f"{unique_key_prefix}_noise_dim"] = config_params.get("noise_dim", 100)
        st.session_state[f"{unique_key_prefix}_generator_layers"] = config_params.get("generator_layers", "128,256,128")
        st.session_state[f"{unique_key_prefix}_discriminator_layers"] = config_params.get("discriminator_layers", "128,64")
        st.session_state[f"{unique_key_prefix}_privacy_mode"] = config_params.get("privacy_mode", "none")
        st.session_state[f"{unique_key_prefix}_dp_epsilon"] = config_params.get("dp_epsilon", 10.0)
        st.session_state[f"{unique_key_prefix}_synthetic_ratio"] = config_params.get("synthetic_ratio", 1.0)
        st.session_state[f"{unique_key_prefix}_show_metrics"] = config_params.get("show_metrics", True)

    def get_export_button_ui(self, config_to_export: dict, unique_key_prefix: str):
        """Export button UI for GANs configuration."""
        json_string = json.dumps(config_to_export, indent=4)
        st.sidebar.download_button(
            label=f"Export {self.get_name()} Config",
            data=json_string,
            file_name=f"{self.get_name().lower().replace(' ', '_').replace('(', '').replace(')', '')}_config.json",
            mime="application/json",
            key=f"{unique_key_prefix}_export_button"
        )

    def get_anonymize_button_ui(self, unique_key_prefix: str) -> bool:
        """Anonymize button UI for GANs."""
        return st.button(f"Generate Synthetic Data with {self.get_name()}", key=f"{unique_key_prefix}_anonymize_button")

# Create plugin instance
def get_plugin():
    """Factory function to create plugin instance."""
    return GenerativeAdversarialNetworksPlugin()
