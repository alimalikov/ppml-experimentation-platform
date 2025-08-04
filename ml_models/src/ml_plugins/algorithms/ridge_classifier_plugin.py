import streamlit as st
import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple, Optional, List
from sklearn.linear_model import RidgeClassifier
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import validation_curve
import warnings

# Import for plugin system - will be auto-fixed during save
try:
    from src.ml_plugins.base_ml_plugin import MLPlugin
except ImportError:
    # Fallback for testing
    import sys
    import os
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))
    sys.path.append(project_root)
    from src.ml_plugins.base_ml_plugin import MLPlugin

class RidgeClassifierPlugin(BaseEstimator, ClassifierMixin, MLPlugin):
    """
    Ridge Classifier Plugin - Regularized Linear Model
    
    Ridge classifier uses L2 regularization to prevent overfitting and improve
    generalization. It's particularly effective when dealing with multicollinearity
    and high-dimensional data where the number of features approaches or exceeds
    the number of samples.
    """
    
    def __init__(self, 
                 alpha=1.0,
                 fit_intercept=True,
                 copy_X=True,
                 max_iter=None,
                 tol=1e-4,
                 class_weight=None,
                 solver='auto',
                 positive=False,
                 random_state=42):
        """
        Initialize Ridge Classifier with comprehensive parameter support
        
        Parameters:
        -----------
        alpha : float, default=1.0
            Regularization strength; must be positive. Higher values specify
            stronger regularization
        fit_intercept : bool, default=True
            Whether to calculate the intercept for this model
        copy_X : bool, default=True
            If True, X will be copied; else, it may be overwritten
        max_iter : int, default=None
            Maximum number of iterations for conjugate gradient solver
        tol : float, default=1e-4
            Precision of the solution
        class_weight : dict or 'balanced', default=None
            Weights associated with classes
        solver : {'auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag', 'saga', 'lbfgs'}
            Solver to use in the computational routines
        positive : bool, default=False
            When set to True, forces coefficients to be positive
        random_state : int, default=42
            Random state for reproducibility
        """
        super().__init__()
        
        # Algorithm parameters
        self.alpha = alpha
        self.fit_intercept = fit_intercept
        self.copy_X = copy_X
        self.max_iter = max_iter
        self.tol = tol
        self.class_weight = class_weight
        self.solver = solver
        self.positive = positive
        self.random_state = random_state
        
        # Plugin metadata
        self._name = "Ridge Classifier"
        self._description = "Regularized linear classifier using L2 penalty. Excellent for handling multicollinearity and preventing overfitting."
        self._category = "Linear Models"
        self._algorithm_type = "Regularized Linear Classifier"
        self._paper_reference = "Hoerl, A. E., & Kennard, R. W. (1970). Ridge regression: Biased estimation for nonorthogonal problems. Technometrics, 12(1), 55-67."
        
        # Capability flags
        self._supports_classification = True
        self._supports_regression = False
        self._supports_binary = True
        self._supports_multiclass = True
        self._min_samples_required = 10
        self._handles_missing_values = False
        self._requires_scaling = True
        self._supports_sparse = True
        self._is_linear = True
        self._provides_feature_importance = True
        self._provides_probabilities = False  # Ridge classifier doesn't provide probabilities directly
        self._handles_multicollinearity = True
        self._regularization_type = "L2"
        
        # Internal attributes
        self.model_ = None
        self.scaler_ = None
        self.label_encoder_ = None
        self.feature_names_ = None
        self.n_features_in_ = None
        self.classes_ = None
        self.is_fitted_ = False
        
    def get_name(self) -> str:
        """Return the algorithm name"""
        return self._name
        
    def get_description(self) -> str:
        """Return detailed algorithm description"""
        return self._description
        
    def get_category(self) -> str:
        """Return algorithm category"""
        return self._category
    
    def get_algorithm_info(self) -> Dict[str, Any]:
        """Return comprehensive algorithm information"""
        return {
            "name": self._name,
            "category": self._category,
            "type": self._algorithm_type,
            "description": self._description,
            "paper_reference": self._paper_reference,
            "regularization_type": self._regularization_type,
            "strengths": [
                "Handles multicollinearity well",
                "Prevents overfitting with L2 regularization",
                "Stable and robust predictions",
                "Works well with high-dimensional data",
                "Computational efficiency",
                "No hyperparameter tuning required (good defaults)",
                "Interpretable linear model"
            ],
            "weaknesses": [
                "Assumes linear relationship",
                "Sensitive to feature scaling",
                "No automatic feature selection",
                "Cannot handle non-linear patterns",
                "No direct probability estimates",
                "Requires tuning of alpha parameter"
            ],
            "use_cases": [
                "High-dimensional datasets",
                "When features are correlated",
                "Baseline linear classifier",
                "When interpretability is important",
                "Text classification with bag-of-words",
                "Genomic data analysis",
                "Financial modeling"
            ],
            "complexity": {
                "training": "O(n × m²) or O(n² × m)",
                "prediction": "O(m)",
                "space": "O(m)"
            },
            "mathematical_foundation": {
                "objective": "minimize ||Xw - y||² + α||w||²",
                "regularization": "L2 penalty (α||w||²)",
                "solution": "w = (X'X + αI)⁻¹X'y"
            }
        }
    
    def fit(self, X, y, sample_weight=None):
        """
        Fit the Ridge Classifier model
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Training data
        y : array-like, shape (n_samples,)
            Target values
        sample_weight : array-like, shape (n_samples,), optional
            Sample weights
            
        Returns:
        --------
        self : object
        """
        # Validate input
        X, y = check_X_y(X, y, accept_sparse=True, dtype=np.float64)
        
        # Store training info
        self.n_features_in_ = X.shape[1]
        if hasattr(X, 'columns'):
            self.feature_names_ = list(X.columns)
        else:
            self.feature_names_ = [f"feature_{i}" for i in range(X.shape[1])]
        
        # Encode labels if they're not numeric
        self.label_encoder_ = LabelEncoder()
        y_encoded = self.label_encoder_.fit_transform(y)
        self.classes_ = self.label_encoder_.classes_
        
        # Scale features (highly recommended for Ridge)
        self.scaler_ = StandardScaler()
        X_scaled = self.scaler_.fit_transform(X)
        
        # Auto-select solver based on data characteristics
        effective_solver = self.solver
        if self.solver == 'auto':
            n_samples, n_features = X.shape
            if n_samples > 10000 and n_features > 1000:
                effective_solver = 'sag'  # Fast for large datasets
            elif n_features > n_samples:
                effective_solver = 'lsqr'  # Good for underdetermined systems
            elif hasattr(X, 'sparse') or hasattr(X_scaled, 'sparse'):
                effective_solver = 'sparse_cg'  # Good for sparse data
            else:
                effective_solver = 'cholesky'  # Default for dense data
        
        # Create and configure the Ridge classifier
        self.model_ = RidgeClassifier(
            alpha=self.alpha,
            fit_intercept=self.fit_intercept,
            copy_X=self.copy_X,
            max_iter=self.max_iter,
            tol=self.tol,
            class_weight=self.class_weight,
            solver=effective_solver,
            positive=self.positive,
            random_state=self.random_state
        )
        
        # Train the model
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)
            if sample_weight is not None:
                self.model_.fit(X_scaled, y_encoded, sample_weight=sample_weight)
            else:
                self.model_.fit(X_scaled, y_encoded)
        
        self.is_fitted_ = True
        return self
    
    def predict(self, X):
        """
        Predict class labels for samples in X
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Samples
            
        Returns:
        --------
        y_pred : array, shape (n_samples,)
            Predicted class labels
        """
        check_is_fitted(self, 'is_fitted_')
        X = check_array(X, accept_sparse=True, dtype=np.float64)
        
        # Scale features
        X_scaled = self.scaler_.transform(X)
        
        # Make predictions
        y_pred_encoded = self.model_.predict(X_scaled)
        
        # Decode labels back to original format
        y_pred = self.label_encoder_.inverse_transform(y_pred_encoded)
        
        return y_pred
    
    def decision_function(self, X):
        """
        Decision function for samples in X
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Samples
            
        Returns:
        --------
        scores : array, shape (n_samples,) or (n_samples, n_classes)
            Decision function scores
        """
        check_is_fitted(self, 'is_fitted_')
        X = check_array(X, accept_sparse=True, dtype=np.float64)
        
        X_scaled = self.scaler_.transform(X)
        return self.model_.decision_function(X_scaled)
    
    def get_feature_importance(self) -> Optional[np.ndarray]:
        """
        Get feature importance (coefficients for Ridge classifier)
        
        Returns:
        --------
        importance : array, shape (n_features,)
            Feature importance scores based on absolute coefficients
        """
        if not self.is_fitted_:
            return None
            
        # For Ridge classifier, feature importance is based on coefficients
        if len(self.classes_) == 2:
            # Binary classification
            importance = np.abs(self.model_.coef_[0])
        else:
            # Multi-class: average absolute coefficients across all classes
            importance = np.mean(np.abs(self.model_.coef_), axis=0)
            
        return importance
    
    def get_regularization_path(self, X, y, alphas=None):
        """
        Compute Ridge coefficients for different regularization values
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Training data
        y : array-like, shape (n_samples,)
            Target values
        alphas : array-like, optional
            Alpha values to test
            
        Returns:
        --------
        alphas : array
            Alpha values used
        coefs : array, shape (n_alphas, n_features) or (n_alphas, n_classes, n_features)
            Coefficients for each alpha
        """
        if alphas is None:
            alphas = np.logspace(-4, 2, 50)
        
        X_scaled = self.scaler_.transform(X) if self.scaler_ else X
        y_encoded = self.label_encoder_.transform(y) if self.label_encoder_ else y
        
        coefs = []
        for alpha in alphas:
            ridge = RidgeClassifier(alpha=alpha, **{k: v for k, v in self.get_model_params().items() if k != 'alpha'})
            ridge.fit(X_scaled, y_encoded)
            coefs.append(ridge.coef_)
        
        return alphas, np.array(coefs)
    
    def get_hyperparameter_config(self, key_prefix: str) -> Dict[str, Any]:
        """Generate Streamlit UI for hyperparameters"""
        
        st.sidebar.markdown("### 🏔️ Ridge Classifier Configuration")
        
        # Create tabs for different parameter groups
        tab1, tab2, tab3 = st.sidebar.tabs(["Core", "Advanced", "Info"])
        
        with tab1:
            st.markdown("**Core Parameters**")
            
            # Alpha (regularization strength)
            alpha = st.number_input(
                "Regularization Strength (α):",
                value=float(self.alpha),
                min_value=0.0001,
                max_value=1000.0,
                step=0.1,
                help="Higher values = stronger regularization, smoother coefficients",
                key=f"{key_prefix}_alpha"
            )
            
            # Alpha selection helper
            if st.button("🎯 Suggest Alpha Range", key=f"{key_prefix}_suggest_alpha"):
                st.info("""
                **Alpha Guidelines:**
                • 0.01-0.1: Light regularization (many features, little noise)
                • 0.1-1.0: Moderate regularization (balanced)
                • 1.0-10.0: Strong regularization (few features, noisy data)
                • 10.0+: Very strong regularization (high noise, overfitting)
                """)
            
            # Solver selection
            solver_options = ['auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag', 'saga', 'lbfgs']
            solver = st.selectbox(
                "Solver:",
                options=solver_options,
                index=solver_options.index(self.solver),
                help="Optimization algorithm. 'auto' selects best based on data",
                key=f"{key_prefix}_solver"
            )
            
            # Class weight
            class_weight_option = st.selectbox(
                "Class Weight:",
                options=['None', 'balanced'],
                index=0 if self.class_weight is None else 1,
                help="Balanced: Adjust weights inversely proportional to class frequencies",
                key=f"{key_prefix}_class_weight"
            )
            class_weight = None if class_weight_option == 'None' else 'balanced'
        
        with tab2:
            st.markdown("**Advanced Parameters**")
            
            # Tolerance
            tol = st.number_input(
                "Tolerance:",
                value=float(self.tol),
                min_value=1e-6,
                max_value=1e-2,
                step=1e-6,
                format="%.2e",
                help="Precision of the solution",
                key=f"{key_prefix}_tol"
            )
            
            # Max iterations
            max_iter_enabled = st.checkbox(
                "Set Max Iterations",
                value=self.max_iter is not None,
                help="Limit iterations for iterative solvers",
                key=f"{key_prefix}_max_iter_enabled"
            )
            
            if max_iter_enabled:
                max_iter = st.number_input(
                    "Max Iterations:",
                    value=int(self.max_iter) if self.max_iter else 1000,
                    min_value=100,
                    max_value=10000,
                    step=100,
                    key=f"{key_prefix}_max_iter"
                )
            else:
                max_iter = None
            
            # Positive coefficients constraint
            positive = st.checkbox(
                "Positive Coefficients Only",
                value=self.positive,
                help="Force all coefficients to be non-negative",
                key=f"{key_prefix}_positive"
            )
            
            # Fit intercept
            fit_intercept = st.checkbox(
                "Fit Intercept",
                value=self.fit_intercept,
                help="Whether to calculate intercept term",
                key=f"{key_prefix}_fit_intercept"
            )
            
            # Random state
            random_state = st.number_input(
                "Random State:",
                value=int(self.random_state),
                min_value=0,
                max_value=1000,
                help="For reproducible results",
                key=f"{key_prefix}_random_state"
            )
        
        with tab3:
            st.markdown("**Algorithm Information**")
            st.info("""
            **Ridge Classifier** excels at:
            • Handling multicollinear features
            • Preventing overfitting with L2 regularization
            • Stable predictions on high-dimensional data
            • Fast training and prediction
            
            **L2 Regularization:** Shrinks coefficients towards zero,
            keeping all features but reducing their impact.
            """)
            
            # Regularization visualization
            if st.button("📊 Show Regularization Effect", key=f"{key_prefix}_show_reg"):
                st.markdown("**Regularization Strength Effect:**")
                st.markdown("• **Low α (0.01):** Minimal regularization, may overfit")
                st.markdown("• **Medium α (1.0):** Balanced bias-variance trade-off")
                st.markdown("• **High α (100):** Strong regularization, may underfit")
            
            if st.button("Show Algorithm Details", key=f"{key_prefix}_info"):
                info = self.get_algorithm_info()
                st.json(info)
        
        return {
            "alpha": alpha,
            "fit_intercept": fit_intercept,
            "copy_X": True,
            "max_iter": max_iter,
            "tol": tol,
            "class_weight": class_weight,
            "solver": solver,
            "positive": positive,
            "random_state": random_state
        }
    
    def create_model_instance(self, hyperparameters: Dict[str, Any]):
        """Create model instance with given hyperparameters"""
        return RidgeClassifierPlugin(
            alpha=hyperparameters.get("alpha", self.alpha),
            fit_intercept=hyperparameters.get("fit_intercept", self.fit_intercept),
            copy_X=hyperparameters.get("copy_X", self.copy_X),
            max_iter=hyperparameters.get("max_iter", self.max_iter),
            tol=hyperparameters.get("tol", self.tol),
            class_weight=hyperparameters.get("class_weight", self.class_weight),
            solver=hyperparameters.get("solver", self.solver),
            positive=hyperparameters.get("positive", self.positive),
            random_state=hyperparameters.get("random_state", self.random_state)
        )
    
    def preprocess_data(self, X, y):
        """
        Optional data preprocessing
        
        Ridge Classifier benefits from:
        1. Feature scaling (handled automatically)
        2. Removing constant features
        3. Handling multicollinearity (which Ridge naturally handles)
        """
        return X, y
    
    def is_compatible_with_data(self, df: pd.DataFrame, target_column: str) -> Tuple[bool, str]:
        """
        Check if algorithm is compatible with the data
        
        Parameters:
        -----------
        df : pd.DataFrame
            Input dataset
        target_column : str
            Name of target column
            
        Returns:
        --------
        is_compatible : bool
            Whether the algorithm can handle this data
        message : str
            Detailed compatibility message
        """
        try:
            # Check if target column exists
            if target_column not in df.columns:
                return False, f"Target column '{target_column}' not found in dataset"
            
            # Check dataset size
            n_samples, n_features = df.shape
            if n_samples < self._min_samples_required:
                return False, f"Minimum {self._min_samples_required} samples required, got {n_samples}"
            
            # Check for missing values
            if df.isnull().any().any():
                return False, "Ridge Classifier requires complete data (no missing values). Please handle missing values first."
            
            # Check target variable type
            target_values = df[target_column].unique()
            n_classes = len(target_values)
            
            if n_classes < 2:
                return False, "Need at least 2 classes for classification"
            
            if n_classes > 1000:
                return False, f"Too many classes ({n_classes}). Ridge Classifier works best with fewer classes."
            
            # Check feature types
            feature_columns = [col for col in df.columns if col != target_column]
            non_numeric_features = []
            
            for col in feature_columns:
                if not pd.api.types.is_numeric_dtype(df[col]):
                    non_numeric_features.append(col)
            
            if non_numeric_features:
                return False, f"Non-numeric features detected: {non_numeric_features}. Please encode categorical variables first."
            
            # Ridge-specific advantages and warnings
            advantages = []
            warnings = []
            
            # Check for multicollinearity (Ridge's strength)
            if n_features > 10:
                corr_matrix = df[feature_columns].corr()
                high_corr_pairs = np.where(np.abs(corr_matrix) > 0.8)
                high_corr_count = len(high_corr_pairs[0]) - n_features  # Subtract diagonal
                
                if high_corr_count > 0:
                    advantages.append(f"Ridge excels with correlated features (found {high_corr_count//2} high-correlation pairs)")
            
            # High-dimensional data
            if n_features > n_samples:
                advantages.append(f"Ridge handles high-dimensional data well ({n_features} features > {n_samples} samples)")
            
            if n_features > 100:
                advantages.append(f"Good for high-dimensional data ({n_features} features)")
            
            if n_samples > 100000:
                warnings.append(f"Large dataset ({n_samples} samples) - consider 'sag' or 'saga' solver for speed")
            
            if n_classes > 10:
                warnings.append(f"Many classes ({n_classes}) - may need more regularization")
            
            # Feature scaling note
            feature_ranges = []
            for col in feature_columns[:5]:  # Check first 5 features
                col_range = df[col].max() - df[col].min()
                if col_range > 0:
                    feature_ranges.append(col_range)
            
            if len(feature_ranges) > 1 and max(feature_ranges) / min(feature_ranges) > 100:
                advantages.append("Automatic feature scaling applied (features have different scales)")
            
            # Compatibility message
            message_parts = [f"✅ Compatible with {n_samples} samples, {n_features} features, {n_classes} classes"]
            
            if advantages:
                message_parts.append("🎯 Ridge advantages: " + "; ".join(advantages))
            
            if warnings:
                message_parts.append("⚠️ Notes: " + "; ".join(warnings))
            
            return True, " | ".join(message_parts)
            
        except Exception as e:
            return False, f"Compatibility check failed: {str(e)}"
    
    def get_model_params(self) -> Dict[str, Any]:
        """Get current model parameters"""
        return {
            'alpha': self.alpha,
            'fit_intercept': self.fit_intercept,
            'copy_X': self.copy_X,
            'max_iter': self.max_iter,
            'tol': self.tol,
            'class_weight': self.class_weight,
            'solver': self.solver,
            'positive': self.positive,
            'random_state': self.random_state
        }
    
    def get_training_info(self) -> Dict[str, Any]:
        """Get information about the trained model"""
        if not self.is_fitted_:
            return {"status": "Not fitted"}
        
        info = {
            "status": "Fitted",
            "n_features": self.n_features_in_,
            "n_classes": len(self.classes_),
            "classes": list(self.classes_),
            "feature_names": self.feature_names_,
            "regularization_strength": self.alpha,
            "solver_used": getattr(self.model_, 'solver', 'N/A'),
            "n_iterations": getattr(self.model_, 'n_iter_', 'N/A'),
            "coefficients_shape": self.model_.coef_.shape if hasattr(self.model_, 'coef_') else 'N/A',
            "intercept_shape": self.model_.intercept_.shape if hasattr(self.model_, 'intercept_') else 'N/A'
        }
        
        # Additional Ridge-specific info
        if hasattr(self.model_, 'coef_'):
            coef_stats = {
                "coefficient_l2_norm": np.linalg.norm(self.model_.coef_),
                "max_coefficient": np.max(np.abs(self.model_.coef_)),
                "mean_coefficient": np.mean(np.abs(self.model_.coef_)),
                "zero_coefficients": np.sum(np.abs(self.model_.coef_) < 1e-10)
            }
            info.update(coef_stats)
        
        return info

    def get_algorithm_specific_metrics(self, y_true: Optional[np.ndarray] = None, y_pred: Optional[np.ndarray] = None, y_proba: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        Retrieve algorithm-specific metrics for the Ridge Classifier model.

        These metrics are derived from the model's learned coefficients and intercept.
        Parameters y_true, y_pred, y_proba are kept for API consistency but are not
        directly used as metrics are sourced from the fitted model's attributes.

        Parameters:
        -----------
        y_true : np.ndarray, optional
            Actual target values.
        y_pred : np.ndarray, optional
            Predicted target values.
        y_proba : np.ndarray, optional
            Predicted probabilities (not directly produced by Ridge Classifier).

        Returns:
        --------
        Dict[str, Any]
            A dictionary containing algorithm-specific metrics.
        """
        if not self.is_fitted_ or not hasattr(self.model_, 'coef_'):
            return {"error": "Model not fitted or coefficients not available. Cannot retrieve Ridge Classifier specific metrics."}

        metrics = {}
        prefix = "ridge_" # Prefix for Ridge Classifier specific metrics

        coefficients = self.model_.coef_
        
        # Coefficient-based metrics
        metrics[f"{prefix}coefficient_l2_norm"] = float(np.linalg.norm(coefficients))
        metrics[f"{prefix}max_abs_coefficient"] = float(np.max(np.abs(coefficients)))
        metrics[f"{prefix}mean_abs_coefficient"] = float(np.mean(np.abs(coefficients)))
        
        # Count coefficients close to zero (e.g., absolute value < 1e-7)
        near_zero_threshold = 1e-7
        metrics[f"{prefix}num_near_zero_coefficients"] = int(np.sum(np.abs(coefficients) < near_zero_threshold))
        metrics[f"{prefix}percentage_near_zero_coefficients"] = float(100.0 * np.sum(np.abs(coefficients) < near_zero_threshold) / coefficients.size)


        # Intercept-based metrics
        if hasattr(self.model_, 'intercept_'):
            intercept = self.model_.intercept_
            if isinstance(intercept, (np.ndarray, list)):
                if len(intercept) == 1:
                    metrics[f"{prefix}intercept_value"] = float(intercept[0])
                else:
                    metrics[f"{prefix}mean_abs_intercept"] = float(np.mean(np.abs(intercept)))
                    # For multi-class, you might report all intercepts or a summary
                    # For simplicity, we'll report the mean absolute intercept here.
                    # Individual intercepts can be many, so not ideal for a flat metric dict.
            else: # Should be a scalar for binary or single-output regression underlying the classifier
                 metrics[f"{prefix}intercept_value"] = float(intercept)
        
        metrics[f"{prefix}regularization_alpha"] = float(self.alpha)
        metrics[f"{prefix}num_iterations"] = getattr(self.model_, 'n_iter_', 'N/A') # n_iter_ might be a list for multi-class sag/saga

        if not metrics: # Should not happen if fitted
            metrics['info'] = "No specific Ridge Classifier metrics were available."
            
        return metrics

def get_plugin():
    """Factory function to get plugin instance"""
    return RidgeClassifierPlugin()
