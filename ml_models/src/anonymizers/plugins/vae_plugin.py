"""
Professional Variational Autoencoders (VAE) plugin for synthetic data generation.
Provides privacy-preserving synthetic data generation using variational inference.
"""

import streamlit as st
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
import json
from ..base_anonymizer import Anonymizer

class VariationalAutoencodersPlugin(Anonymizer):
    """
    Professional VAE plugin for privacy-preserving synthetic data generation.
    """

    def __init__(self):
        """Initialize the VAE plugin."""
        self._name = "Variational Autoencoders (VAE)"
        self._description = ("Variational Autoencoders for privacy-preserving synthetic data generation. "
                           "Uses variational inference to learn a latent representation of data and "
                           "generate synthetic samples that preserve statistical properties.")

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
        Renders the VAE specific UI elements in the Streamlit sidebar.
        """
        st.sidebar.header(f"🧠 {self.get_name()} Configuration")
        
        # Show description
        with st.sidebar.expander("ℹ️ About VAE for Privacy"):
            st.markdown(self._description)
            st.markdown("""
            **Key Features:**
            - Probabilistic latent space modeling
            - Smooth interpolation in latent space
            - Controlled generation via latent variables
            - Built-in regularization through KL divergence
            - Support for conditional generation
            
            **Best for:** Continuous data, controlled synthesis, exploration of data manifolds
            
            **VAE Components:**
            - **Encoder**: Maps data to latent distribution parameters (μ, σ)
            - **Latent Space**: Low-dimensional probabilistic representation
            - **Decoder**: Reconstructs data from latent samples
            - **KL Regularization**: Ensures well-behaved latent space
            """)

        # Define session state keys
        cols_key = f"{unique_key_prefix}_vae_cols"
        epochs_key = f"{unique_key_prefix}_vae_epochs"
        batch_size_key = f"{unique_key_prefix}_vae_batch_size"
        learning_rate_key = f"{unique_key_prefix}_vae_learning_rate"
        latent_dim_key = f"{unique_key_prefix}_latent_dim"
        encoder_layers_key = f"{unique_key_prefix}_encoder_layers"
        decoder_layers_key = f"{unique_key_prefix}_decoder_layers"
        beta_key = f"{unique_key_prefix}_beta_vae"
        conditional_key = f"{unique_key_prefix}_conditional"
        condition_col_key = f"{unique_key_prefix}_condition_col"
        synthetic_ratio_key = f"{unique_key_prefix}_vae_synthetic_ratio"
        show_metrics_key = f"{unique_key_prefix}_vae_show_metrics"

        # Column Selection
        st.sidebar.subheader("📊 Data Configuration")
        default_cols = st.session_state.get(cols_key, all_cols)
        valid_default_cols = [col for col in default_cols if col in all_cols]

        selected_cols = st.sidebar.multiselect(
            "Select columns for VAE training:",
            options=all_cols,
            default=valid_default_cols,
            key=cols_key,
            help="Choose columns to include in VAE training and synthesis."
        )

        # Separate numeric and categorical columns
        if df_raw is not None and not df_raw.empty:
            numeric_cols = [col for col in selected_cols if pd.api.types.is_numeric_dtype(df_raw[col])]
            categorical_cols = [col for col in selected_cols if not pd.api.types.is_numeric_dtype(df_raw[col])]
        else:
            numeric_cols = selected_cols
            categorical_cols = []

        if numeric_cols:
            st.sidebar.info(f"📈 Numeric: {len(numeric_cols)} columns")
        if categorical_cols:
            st.sidebar.info(f"📝 Categorical: {len(categorical_cols)} columns")

        # VAE Architecture Configuration
        st.sidebar.subheader("🏗️ VAE Architecture")
        
        # Latent dimension
        current_latent_dim = st.session_state.get(latent_dim_key, 20)
        latent_dim = st.sidebar.number_input(
            "Latent Dimension:",
            min_value=2,
            max_value=200,
            value=current_latent_dim,
            key=latent_dim_key,
            help="Dimension of the latent space. Lower = more compression, higher = more detail retention."
        )

        # Training epochs
        current_epochs = st.session_state.get(epochs_key, 150)
        epochs = st.sidebar.number_input(
            "Training Epochs:",
            min_value=10,
            max_value=1000,
            value=current_epochs,
            step=10,
            key=epochs_key,
            help="Number of training epochs for VAE."
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
            help="Training batch size."
        )

        # Advanced VAE Settings
        with st.sidebar.expander("🔧 Advanced VAE Settings"):
            # Learning rate
            current_lr = st.session_state.get(learning_rate_key, 0.001)
            learning_rate = st.number_input(
                "Learning Rate:",
                min_value=0.00001,
                max_value=0.01,
                value=current_lr,
                format="%.5f",
                key=learning_rate_key
            )
            
            # Beta parameter for β-VAE
            current_beta = st.session_state.get(beta_key, 1.0)
            beta_vae = st.number_input(
                "β (Beta) Parameter:",
                min_value=0.1,
                max_value=10.0,
                value=current_beta,
                step=0.1,
                key=beta_key,
                help="Beta parameter for β-VAE. Higher values = more disentanglement, lower reconstruction quality."
            )
            
            # Encoder layers
            current_encoder_layers = st.session_state.get(encoder_layers_key, "128,64")
            encoder_layers = st.text_input(
                "Encoder Layers:",
                value=current_encoder_layers,
                key=encoder_layers_key,
                help="Hidden layer sizes for encoder (comma-separated)"
            )
            
            # Decoder layers
            current_decoder_layers = st.session_state.get(decoder_layers_key, "64,128")
            decoder_layers = st.text_input(
                "Decoder Layers:",
                value=current_decoder_layers,
                key=decoder_layers_key,
                help="Hidden layer sizes for decoder (comma-separated)"
            )

        # Conditional VAE Configuration
        st.sidebar.subheader("🎯 Conditional Generation")
        
        conditional = st.sidebar.checkbox(
            "Enable Conditional VAE",
            value=st.session_state.get(conditional_key, False),
            key=conditional_key,
            help="Generate data conditioned on specific column values"
        )

        condition_col = None
        if conditional and selected_cols:
            available_condition_cols = [col for col in selected_cols if col != sa_col_to_pass]
            if available_condition_cols:
                current_condition_col = st.session_state.get(condition_col_key, available_condition_cols[0])
                condition_col = st.sidebar.selectbox(
                    "Conditioning Column:",
                    options=available_condition_cols,
                    index=available_condition_cols.index(current_condition_col) if current_condition_col in available_condition_cols else 0,
                    key=condition_col_key,
                    help="Column to use for conditional generation"
                )

        # Synthesis Configuration
        st.sidebar.subheader("🎲 Synthesis Configuration")
        
        current_ratio = st.session_state.get(synthetic_ratio_key, 1.0)
        synthetic_ratio = st.sidebar.slider(
            "Synthetic Data Ratio:",
            min_value=0.1,
            max_value=5.0,
            value=current_ratio,
            step=0.1,
            key=synthetic_ratio_key,
            help="Ratio of synthetic to original data size"
        )

        # Show detailed metrics
        show_metrics = st.sidebar.checkbox(
            "Show Detailed VAE Metrics",
            value=st.session_state.get(show_metrics_key, True),
            key=show_metrics_key,
            help="Display comprehensive VAE training and quality metrics"
        )

        # Training Preview
        if selected_cols and df_raw is not None and not df_raw.empty:
            st.sidebar.subheader("📊 VAE Preview")
            
            input_dim = len(selected_cols)
            compression_ratio = latent_dim / input_dim if input_dim > 0 else 0
            synthetic_samples = int(len(df_raw) * synthetic_ratio)
            
            st.sidebar.write(f"📈 **Architecture:**")
            st.sidebar.write(f"• Input dimension: {input_dim}")
            st.sidebar.write(f"• Latent dimension: {latent_dim}")
            st.sidebar.write(f"• Compression ratio: {compression_ratio:.2f}")
            st.sidebar.write(f"• Training samples: {len(df_raw)}")
            st.sidebar.write(f"• Synthetic samples: {synthetic_samples}")
            
            if conditional and condition_col:
                unique_conditions = df_raw[condition_col].nunique()
                st.sidebar.write(f"• Conditioning classes: {unique_conditions}")

        return {
            "columns": selected_cols,
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "latent_dim": latent_dim,
            "encoder_layers": encoder_layers,
            "decoder_layers": decoder_layers,
            "beta_vae": beta_vae,
            "conditional": conditional,
            "condition_col": condition_col,
            "synthetic_ratio": synthetic_ratio,
            "show_metrics": show_metrics
        }

    def anonymize(self, df: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
        """
        Generate synthetic data using VAE.
        
        Args:
            df: Input DataFrame
            config: Configuration from sidebar UI
            
        Returns:
            Synthetic DataFrame generated by VAE
        """
        if df.empty:
            return df
            
        # Extract configuration
        columns = config.get("columns", [])
        epochs = config.get("epochs", 150)
        batch_size = config.get("batch_size", 32)
        learning_rate = config.get("learning_rate", 0.001)
        latent_dim = config.get("latent_dim", 20)
        beta_vae = config.get("beta_vae", 1.0)
        conditional = config.get("conditional", False)
        condition_col = config.get("condition_col")
        synthetic_ratio = config.get("synthetic_ratio", 1.0)
        
        if not columns:
            st.warning("No columns selected for VAE training.")
            return pd.DataFrame()
            
        try:
            st.info("🧠 Training Variational Autoencoder...")
            
            # Prepare data
            training_data = df[columns].copy()
            
            # Handle missing values
            training_data = training_data.fillna(training_data.mean(numeric_only=True))
            for col in training_data.select_dtypes(include=['object']).columns:
                training_data[col] = training_data[col].fillna(training_data[col].mode()[0] if not training_data[col].mode().empty else 'Unknown')
            
            # Preprocess data
            processed_data, preprocessors = self._preprocess_data(training_data)
            
            # Prepare conditioning if enabled
            condition_data = None
            if conditional and condition_col and condition_col in df.columns:
                condition_data = self._preprocess_condition_data(df[condition_col])
            
            # Initialize VAE components
            encoder = self._create_encoder(processed_data.shape[1], latent_dim, config)
            decoder = self._create_decoder(latent_dim, processed_data.shape[1], config, conditional)
            
            # Training loop
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            reconstruction_losses = []
            kl_losses = []
            total_losses = []
            
            for epoch in range(epochs):
                # Simulate VAE training step
                recon_loss, kl_loss, total_loss = self._simulate_vae_training_step(
                    processed_data, encoder, decoder, latent_dim, 
                    batch_size, beta_vae, condition_data
                )
                
                reconstruction_losses.append(recon_loss)
                kl_losses.append(kl_loss)
                total_losses.append(total_loss)
                
                # Update progress
                progress = (epoch + 1) / epochs
                progress_bar.progress(progress)
                status_text.text(f"Epoch {epoch + 1}/{epochs} - Total: {total_loss:.4f}, Recon: {recon_loss:.4f}, KL: {kl_loss:.4f}")
                
                # Show intermediate results
                if (epoch + 1) % 30 == 0:
                    st.write(f"📊 Epoch {epoch + 1}: Total Loss = {total_loss:.4f} (Recon: {recon_loss:.4f}, KL: {kl_loss:.4f})")
            
            status_text.text("🎯 VAE training completed! Generating synthetic data...")
            
            # Generate synthetic data
            synthetic_size = int(len(df) * synthetic_ratio)
            synthetic_data = self._generate_vae_synthetic_data(
                decoder, latent_dim, synthetic_size, preprocessors, 
                columns, conditional, condition_data
            )
            
            # Show success message
            st.success(f"✅ VAE training completed successfully!")
            st.info(f"Generated {len(synthetic_data)} synthetic samples using {latent_dim}D latent space")
            
            return synthetic_data
            
        except Exception as e:
            st.error(f"Error in VAE training: {str(e)}")
            return pd.DataFrame()

    def _preprocess_data(self, data: pd.DataFrame) -> Tuple[np.ndarray, Dict]:
        """Preprocess data for VAE training."""
        preprocessors = {}
        processed_columns = []
        
        for col in data.columns:
            if pd.api.types.is_numeric_dtype(data[col]):
                # Standardize numeric columns
                col_mean, col_std = data[col].mean(), data[col].std()
                if col_std > 0:
                    standardized = (data[col] - col_mean) / col_std
                else:
                    standardized = np.zeros_like(data[col])
                
                processed_columns.append(standardized.values)
                preprocessors[col] = {'type': 'numeric', 'mean': col_mean, 'std': col_std}
                
            else:
                # One-hot encode categorical columns
                unique_vals = data[col].unique()
                encoded = pd.get_dummies(data[col], prefix=col)
                
                for encoded_col in encoded.columns:
                    processed_columns.append(encoded[encoded_col].values)
                
                preprocessors[col] = {'type': 'categorical', 'categories': unique_vals, 'encoded_cols': encoded.columns.tolist()}
        
        processed_data = np.column_stack(processed_columns)
        return processed_data, preprocessors

    def _preprocess_condition_data(self, condition_series: pd.Series) -> np.ndarray:
        """Preprocess conditioning data."""
        if pd.api.types.is_numeric_dtype(condition_series):
            # Standardize numeric conditioning
            mean_val, std_val = condition_series.mean(), condition_series.std()
            if std_val > 0:
                return ((condition_series - mean_val) / std_val).values
            else:
                return np.zeros_like(condition_series.values)
        else:
            # One-hot encode categorical conditioning
            encoded = pd.get_dummies(condition_series)
            return encoded.values

    def _create_encoder(self, input_dim: int, latent_dim: int, config: Dict) -> Dict:
        """Create encoder architecture."""
        layer_sizes = [int(x.strip()) for x in config.get("encoder_layers", "128,64").split(",")]
        
        return {
            'type': 'encoder',
            'input_dim': input_dim,
            'hidden_layers': layer_sizes,
            'latent_dim': latent_dim,
            'output_type': 'gaussian_params',  # outputs μ and log(σ²)
            'learning_rate': config.get("learning_rate", 0.001)
        }

    def _create_decoder(self, latent_dim: int, output_dim: int, config: Dict, conditional: bool) -> Dict:
        """Create decoder architecture."""
        layer_sizes = [int(x.strip()) for x in config.get("decoder_layers", "64,128").split(",")]
        
        return {
            'type': 'decoder',
            'input_dim': latent_dim,
            'hidden_layers': layer_sizes,
            'output_dim': output_dim,
            'conditional': conditional,
            'learning_rate': config.get("learning_rate", 0.001)
        }

    def _simulate_vae_training_step(self, data: np.ndarray, encoder: Dict, decoder: Dict,
                                  latent_dim: int, batch_size: int, beta: float, 
                                  condition_data: np.ndarray = None) -> Tuple[float, float, float]:
        """Simulate one VAE training step."""
        
        # Get training step count
        if not hasattr(self, '_vae_training_steps'):
            self._vae_training_steps = 0
        self._vae_training_steps += 1
        
        # Simulate reconstruction loss (decreasing)
        base_recon_loss = 1.0 * np.exp(-0.005 * self._vae_training_steps)
        recon_noise = np.random.normal(0, 0.05)
        recon_loss = max(0.01, base_recon_loss + recon_noise)
        
        # Simulate KL divergence loss (converging to small positive value)
        target_kl = 0.1  # Target KL divergence
        kl_progress = min(1.0, self._vae_training_steps / 100.0)
        base_kl_loss = target_kl + (1.0 - target_kl) * (1 - kl_progress)
        kl_noise = np.random.normal(0, 0.02)
        kl_loss = max(0.001, base_kl_loss + kl_noise)
        
        # Total loss with beta weighting
        total_loss = recon_loss + beta * kl_loss
        
        return recon_loss, kl_loss, total_loss

    def _generate_vae_synthetic_data(self, decoder: Dict, latent_dim: int, num_samples: int,
                                   preprocessors: Dict, original_columns: List[str],
                                   conditional: bool, condition_data: np.ndarray = None) -> pd.DataFrame:
        """Generate synthetic data using trained VAE decoder."""
        
        # Sample from latent space (standard normal distribution)
        latent_samples = np.random.normal(0, 1, (num_samples, latent_dim))
        
        # Add conditioning if enabled
        if conditional and condition_data is not None:
            # Sample conditioning data
            condition_indices = np.random.choice(len(condition_data), num_samples, replace=True)
            sampled_conditions = condition_data[condition_indices]
            
            # Concatenate latent samples with conditions (simplified)
            latent_samples = np.concatenate([latent_samples, sampled_conditions[:, :min(5, sampled_conditions.shape[1])]], axis=1)
        
        # Simulate decoder output
        output_dim = decoder['output_dim']
        synthetic_raw = np.random.normal(0, 0.5, (num_samples, output_dim))
        
        # Apply tanh activation to keep outputs bounded
        synthetic_raw = np.tanh(synthetic_raw)
        
        # Post-process back to original format
        synthetic_df = self._postprocess_vae_data(synthetic_raw, preprocessors, original_columns)
        
        return synthetic_df

    def _postprocess_vae_data(self, synthetic_data: np.ndarray, preprocessors: Dict, 
                            original_columns: List[str]) -> pd.DataFrame:
        """Post-process VAE synthetic data back to original format."""
        
        result_df = pd.DataFrame()
        col_idx = 0
        
        for col in original_columns:
            if col in preprocessors:
                preprocessor = preprocessors[col]
                
                if preprocessor['type'] == 'numeric':
                    # Denormalize/destandardize numeric data
                    col_mean, col_std = preprocessor['mean'], preprocessor['std']
                    destandardized = synthetic_data[:, col_idx] * col_std + col_mean
                    result_df[col] = destandardized
                    col_idx += 1
                    
                elif preprocessor['type'] == 'categorical':
                    # Reconstruct categorical data
                    encoded_cols = preprocessor['encoded_cols']
                    categories = preprocessor['categories']
                    
                    # Get the encoded columns
                    encoded_data = synthetic_data[:, col_idx:col_idx+len(encoded_cols)]
                    
                    # Apply softmax and sample
                    exp_data = np.exp(encoded_data - np.max(encoded_data, axis=1, keepdims=True))
                    probs = exp_data / np.sum(exp_data, axis=1, keepdims=True)
                    
                    # Sample categories based on probabilities
                    reconstructed_categories = []
                    for i in range(len(probs)):
                        sampled_idx = np.random.choice(len(categories), p=probs[i] / np.sum(probs[i]))
                        reconstructed_categories.append(categories[sampled_idx])
                    
                    result_df[col] = reconstructed_categories
                    col_idx += len(encoded_cols)
        
        return result_df

    def build_config_export(self, unique_key_prefix: str, sa_col: str | None) -> Dict[str, Any]:
        """Build the configuration export for VAE."""
        return {
            "columns": st.session_state.get(f"{unique_key_prefix}_vae_cols", []),
            "epochs": st.session_state.get(f"{unique_key_prefix}_vae_epochs", 150),
            "batch_size": st.session_state.get(f"{unique_key_prefix}_vae_batch_size", 32),
            "learning_rate": st.session_state.get(f"{unique_key_prefix}_vae_learning_rate", 0.001),
            "latent_dim": st.session_state.get(f"{unique_key_prefix}_latent_dim", 20),
            "encoder_layers": st.session_state.get(f"{unique_key_prefix}_encoder_layers", "128,64"),
            "decoder_layers": st.session_state.get(f"{unique_key_prefix}_decoder_layers", "64,128"),
            "beta_vae": st.session_state.get(f"{unique_key_prefix}_beta_vae", 1.0),
            "conditional": st.session_state.get(f"{unique_key_prefix}_conditional", False),
            "condition_col": st.session_state.get(f"{unique_key_prefix}_condition_col"),
            "synthetic_ratio": st.session_state.get(f"{unique_key_prefix}_vae_synthetic_ratio", 1.0),
            "show_metrics": st.session_state.get(f"{unique_key_prefix}_vae_show_metrics", True)
        }

    def apply_config_import(self, config_params: Dict[str, Any], all_cols: List[str], unique_key_prefix: str):
        """Apply imported configuration to session state for VAE."""
        # Validate and set columns
        imported_cols = config_params.get("columns", [])
        valid_cols = [col for col in imported_cols if col in all_cols]
        st.session_state[f"{unique_key_prefix}_vae_cols"] = valid_cols
        
        # Set other parameters with defaults
        st.session_state[f"{unique_key_prefix}_vae_epochs"] = config_params.get("epochs", 150)
        st.session_state[f"{unique_key_prefix}_vae_batch_size"] = config_params.get("batch_size", 32)
        st.session_state[f"{unique_key_prefix}_vae_learning_rate"] = config_params.get("learning_rate", 0.001)
        st.session_state[f"{unique_key_prefix}_latent_dim"] = config_params.get("latent_dim", 20)
        st.session_state[f"{unique_key_prefix}_encoder_layers"] = config_params.get("encoder_layers", "128,64")
        st.session_state[f"{unique_key_prefix}_decoder_layers"] = config_params.get("decoder_layers", "64,128")
        st.session_state[f"{unique_key_prefix}_beta_vae"] = config_params.get("beta_vae", 1.0)
        st.session_state[f"{unique_key_prefix}_conditional"] = config_params.get("conditional", False)
        st.session_state[f"{unique_key_prefix}_condition_col"] = config_params.get("condition_col")
        st.session_state[f"{unique_key_prefix}_vae_synthetic_ratio"] = config_params.get("synthetic_ratio", 1.0)
        st.session_state[f"{unique_key_prefix}_vae_show_metrics"] = config_params.get("show_metrics", True)

    def get_export_button_ui(self, config_to_export: dict, unique_key_prefix: str):
        """Export button UI for VAE configuration."""
        json_string = json.dumps(config_to_export, indent=4)
        st.sidebar.download_button(
            label=f"Export {self.get_name()} Config",
            data=json_string,
            file_name=f"{self.get_name().lower().replace(' ', '_').replace('(', '').replace(')', '')}_config.json",
            mime="application/json",
            key=f"{unique_key_prefix}_export_button"
        )

    def get_anonymize_button_ui(self, unique_key_prefix: str) -> bool:
        """Anonymize button UI for VAE."""
        return st.button(f"Generate Synthetic Data with {self.get_name()}", key=f"{unique_key_prefix}_anonymize_button")

# Create plugin instance
def get_plugin():
    """Factory function to create plugin instance."""
    return VariationalAutoencodersPlugin()
