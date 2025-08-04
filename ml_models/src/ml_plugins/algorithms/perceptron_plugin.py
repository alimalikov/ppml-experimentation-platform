import streamlit as st
import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple, Optional, List
from sklearn.linear_model import Perceptron
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.utils.multiclass import unique_labels
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

class PerceptronPlugin(BaseEstimator, ClassifierMixin, MLPlugin):
    """
    Perceptron Plugin - Simple Linear Classifier
    
    The perceptron is a linear classifier that learns a decision boundary
    by iteratively updating weights based on misclassified examples.
    Excellent for linearly separable data and as a baseline classifier.
    """
    
    def __init__(self, 
                 penalty=None,
                 alpha=0.0001,
                 l1_ratio=0.15,
                 fit_intercept=True,
                 max_iter=1000,
                 tol=1e-3,
                 shuffle=True,
                 verbose=0,
                 eta0=1.0,
                 n_jobs=None,
                 random_state=42,
                 early_stopping=False,
                 validation_fraction=0.1,
                 n_iter_no_change=5,
                 class_weight=None,
                 warm_start=False):
        """
        Initialize Perceptron with comprehensive parameter support
        
        Parameters:
        -----------
        penalty : {'l1', 'l2', 'elasticnet'} or None, default=None
            The penalty (regularization term) to be used
        alpha : float, default=0.0001
            Constant that multiplies the regularization term
        l1_ratio : float, default=0.15
            The Elastic Net mixing parameter (only used if penalty='elasticnet')
        fit_intercept : bool, default=True
            Whether the intercept should be estimated
        max_iter : int, default=1000
            The maximum number of passes over the training data
        tol : float, default=1e-3
            The stopping criterion tolerance
        shuffle : bool, default=True
            Whether or not the training data should be shuffled after each epoch
        verbose : int, default=0
            The verbosity level
        eta0 : float, default=1.0
            Constant by which the updates are multiplied
        n_jobs : int, default=None
            The number of CPUs to use for computation
        random_state : int, default=42
            Random state for reproducibility
        early_stopping : bool, default=False
            Whether to use early stopping to terminate training
        validation_fraction : float, default=0.1
            The proportion of training data to set aside as validation set
        n_iter_no_change : int, default=5
            Number of iterations with no improvement to wait before early stopping
        class_weight : dict or 'balanced', default=None
            Weights associated with classes
        warm_start : bool, default=False
            When set to True, reuse the solution of the previous call to fit
        """
        super().__init__()
        
        # Algorithm parameters
        self.penalty = penalty
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.fit_intercept = fit_intercept
        self.max_iter = max_iter
        self.tol = tol
        self.shuffle = shuffle
        self.verbose = verbose
        self.eta0 = eta0
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.early_stopping = early_stopping
        self.validation_fraction = validation_fraction
        self.n_iter_no_change = n_iter_no_change
        self.class_weight = class_weight
        self.warm_start = warm_start
        
        # Plugin metadata
        self._name = "Perceptron"
        self._description = "Simple linear classifier using the perceptron algorithm. Fast, interpretable, and effective for linearly separable data."
        self._category = "Linear Models"
        self._algorithm_type = "Linear Classifier"
        self._paper_reference = "Rosenblatt, F. (1957). The perceptron: a perceiving and recognizing automaton. Cornell Aeronautical Laboratory."
        
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
        self._provides_probabilities = False  # Perceptron doesn't provide probabilities
        self._supports_online_learning = True
        
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
            "strengths": [
                "Simple and fast training",
                "Low memory requirements",
                "Good for linearly separable data",
                "Online learning capability",
                "Interpretable linear model",
                "Handles large datasets well",
                "No probabilistic assumptions"
            ],
            "weaknesses": [
                "Only works for linearly separable data",
                "No probability estimates",
                "Sensitive to feature scaling",
                "Can be unstable with non-separable data",
                "No regularization by default"
            ],
            "use_cases": [
                "Binary classification problems",
                "Linearly separable datasets",
                "Online learning scenarios",
                "Baseline classifier",
                "Large-scale text classification",
                "Stream processing applications"
            ],
            "complexity": {
                "training": "O(n × m × k)",
                "prediction": "O(m)",
                "space": "O(m)"
            },
            "note": "n=samples, m=features, k=iterations"
        }
    
    def fit(self, X, y, sample_weight=None):
        """
        Fit the Perceptron model
        
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
        
        # Scale features (recommended for perceptron)
        self.scaler_ = StandardScaler()
        X_scaled = self.scaler_.fit_transform(X)
        
        # Create and configure the perceptron model
        self.model_ = Perceptron(
            penalty=self.penalty,
            alpha=self.alpha,
            l1_ratio=self.l1_ratio,
            fit_intercept=self.fit_intercept,
            max_iter=self.max_iter,
            tol=self.tol,
            shuffle=self.shuffle,
            verbose=self.verbose,
            eta0=self.eta0,
            n_jobs=self.n_jobs,
            random_state=self.random_state,
            early_stopping=self.early_stopping,
            validation_fraction=self.validation_fraction,
            n_iter_no_change=self.n_iter_no_change,
            class_weight=self.class_weight,
            warm_start=self.warm_start
        )
        
        # Train the model
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)
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
        Confidence scores for samples
        
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
    
    def partial_fit(self, X, y, classes=None, sample_weight=None):
        """
        Perform one epoch of stochastic gradient descent on given samples
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Training data
        y : array-like, shape (n_samples,)
            Target values
        classes : array, shape (n_classes,), optional
            Classes across all calls to partial_fit
        sample_weight : array-like, shape (n_samples,), optional
            Sample weights
            
        Returns:
        --------
        self : object
        """
        if not self.is_fitted_:
            # First call to partial_fit
            X, y = check_X_y(X, y, accept_sparse=True, dtype=np.float64)
            
            self.n_features_in_ = X.shape[1]
            if hasattr(X, 'columns'):
                self.feature_names_ = list(X.columns)
            else:
                self.feature_names_ = [f"feature_{i}" for i in range(X.shape[1])]
            
            self.label_encoder_ = LabelEncoder()
            if classes is not None:
                self.label_encoder_.fit(classes)
                self.classes_ = classes
            else:
                self.label_encoder_.fit(y)
                self.classes_ = self.label_encoder_.classes_
            
            self.scaler_ = StandardScaler()
            X_scaled = self.scaler_.fit_transform(X)
            
            self.model_ = Perceptron(
                penalty=self.penalty,
                alpha=self.alpha,
                l1_ratio=self.l1_ratio,
                fit_intercept=self.fit_intercept,
                max_iter=self.max_iter,
                tol=self.tol,
                shuffle=self.shuffle,
                verbose=self.verbose,
                eta0=self.eta0,
                n_jobs=self.n_jobs,
                random_state=self.random_state,
                early_stopping=self.early_stopping,
                validation_fraction=self.validation_fraction,
                n_iter_no_change=self.n_iter_no_change,
                class_weight=self.class_weight,
                warm_start=self.warm_start
            )
            
            self.is_fitted_ = True
        else:
            # Subsequent calls
            X = check_array(X, accept_sparse=True, dtype=np.float64)
            X_scaled = self.scaler_.transform(X)
        
        y_encoded = self.label_encoder_.transform(y)
        
        # Update scaler incrementally (approximate)
        if not hasattr(self.scaler_, 'partial_fit'):
            # StandardScaler doesn't have partial_fit, so we approximate
            pass
        
        # Perform partial fit
        self.model_.partial_fit(X_scaled, y_encoded, classes=None, sample_weight=sample_weight)
        
        return self
    
    def get_feature_importance(self) -> Optional[np.ndarray]:
        """
        Get feature importance (coefficients for perceptron)
        
        Returns:
        --------
        importance : array, shape (n_features,)
            Feature importance scores
        """
        if not self.is_fitted_:
            return None
            
        # For perceptron, feature importance is the absolute value of coefficients
        if len(self.classes_) == 2:
            # Binary classification
            importance = np.abs(self.model_.coef_[0])
        else:
            # Multi-class: average absolute coefficients across all classes
            importance = np.mean(np.abs(self.model_.coef_), axis=0)
            
        return importance
    
    def get_hyperparameter_config(self, key_prefix: str) -> Dict[str, Any]:
        """Generate Streamlit UI for hyperparameters"""
        
        st.sidebar.markdown("### ⚡ Perceptron Configuration")
        
        # Create tabs for different parameter groups
        tab1, tab2, tab3 = st.sidebar.tabs(["Core", "Advanced", "Info"])
        
        with tab1:
            st.markdown("**Core Parameters**")
            
            # Learning rate (eta0)
            eta0 = st.number_input(
                "Learning Rate (η₀):",
                value=float(self.eta0),
                min_value=0.001,
                max_value=10.0,
                step=0.1,
                help="Constant by which updates are multiplied",
                key=f"{key_prefix}_eta0"
            )
            
            # Max iterations
            max_iter = st.number_input(
                "Max Iterations:",
                value=int(self.max_iter),
                min_value=10,
                max_value=10000,
                step=50,
                help="Maximum number of passes over training data",
                key=f"{key_prefix}_max_iter"
            )
            
            # Tolerance
            tol = st.number_input(
                "Tolerance:",
                value=float(self.tol),
                min_value=1e-6,
                max_value=1e-1,
                step=1e-6,
                format="%.2e",
                help="Stopping criterion tolerance",
                key=f"{key_prefix}_tol"
            )
            
            # Shuffle
            shuffle = st.checkbox(
                "Shuffle Data",
                value=self.shuffle,
                help="Shuffle training data after each epoch",
                key=f"{key_prefix}_shuffle"
            )
        
        with tab2:
            st.markdown("**Advanced Parameters**")
            
            # Penalty/Regularization
            penalty_options = ['None', 'l1', 'l2', 'elasticnet']
            penalty_index = 0 if self.penalty is None else penalty_options.index(self.penalty)
            penalty = st.selectbox(
                "Regularization:",
                options=penalty_options,
                index=penalty_index,
                help="Regularization type",
                key=f"{key_prefix}_penalty"
            )
            penalty = None if penalty == 'None' else penalty
            
            # Alpha (regularization strength)
            alpha = st.number_input(
                "Regularization Strength (α):",
                value=float(self.alpha),
                min_value=0.0,
                max_value=1.0,
                step=0.0001,
                format="%.4f",
                help="Regularization term multiplier",
                key=f"{key_prefix}_alpha"
            )
            
            # L1 ratio (for elastic net)
            l1_ratio = st.number_input(
                "L1 Ratio:",
                value=float(self.l1_ratio),
                min_value=0.0,
                max_value=1.0,
                step=0.05,
                help="Elastic Net mixing parameter (0=L2, 1=L1)",
                key=f"{key_prefix}_l1_ratio"
            )
            
            # Early stopping
            early_stopping = st.checkbox(
                "Early Stopping",
                value=self.early_stopping,
                help="Stop training when validation score stops improving",
                key=f"{key_prefix}_early_stopping"
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
            **Perceptron** is excellent for:
            • Simple linear classification
            • Fast training and prediction
            • Online learning scenarios
            • Baseline comparisons
            
            **Note:** Requires linearly separable data for convergence
            """)
            
            if st.button("Show Algorithm Details", key=f"{key_prefix}_info"):
                info = self.get_algorithm_info()
                st.json(info)
        
        return {
            "penalty": penalty,
            "alpha": alpha,
            "l1_ratio": l1_ratio,
            "fit_intercept": True,
            "max_iter": max_iter,
            "tol": tol,
            "shuffle": shuffle,
            "verbose": 0,
            "eta0": eta0,
            "n_jobs": None,
            "random_state": random_state,
            "early_stopping": early_stopping,
            "validation_fraction": 0.1,
            "n_iter_no_change": 5,
            "class_weight": class_weight,
            "warm_start": False
        }
    
    def create_model_instance(self, hyperparameters: Dict[str, Any]):
        """Create model instance with given hyperparameters"""
        return PerceptronPlugin(
            penalty=hyperparameters.get("penalty", self.penalty),
            alpha=hyperparameters.get("alpha", self.alpha),
            l1_ratio=hyperparameters.get("l1_ratio", self.l1_ratio),
            fit_intercept=hyperparameters.get("fit_intercept", self.fit_intercept),
            max_iter=hyperparameters.get("max_iter", self.max_iter),
            tol=hyperparameters.get("tol", self.tol),
            shuffle=hyperparameters.get("shuffle", self.shuffle),
            verbose=hyperparameters.get("verbose", self.verbose),
            eta0=hyperparameters.get("eta0", self.eta0),
            n_jobs=hyperparameters.get("n_jobs", self.n_jobs),
            random_state=hyperparameters.get("random_state", self.random_state),
            early_stopping=hyperparameters.get("early_stopping", self.early_stopping),
            validation_fraction=hyperparameters.get("validation_fraction", self.validation_fraction),
            n_iter_no_change=hyperparameters.get("n_iter_no_change", self.n_iter_no_change),
            class_weight=hyperparameters.get("class_weight", self.class_weight),
            warm_start=hyperparameters.get("warm_start", self.warm_start)
        )
    
    def preprocess_data(self, X, y):
        """
        Optional data preprocessing
        
        Perceptron benefits from:
        1. Feature scaling (handled automatically)
        2. Linearly separable data
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
                return False, "Perceptron requires complete data (no missing values). Please handle missing values first."
            
            # Check target variable type
            target_values = df[target_column].unique()
            n_classes = len(target_values)
            
            if n_classes < 2:
                return False, "Need at least 2 classes for classification"
            
            if n_classes > 100:
                return False, f"Too many classes ({n_classes}). Perceptron works best with fewer classes."
            
            # Check feature types
            feature_columns = [col for col in df.columns if col != target_column]
            non_numeric_features = []
            
            for col in feature_columns:
                if not pd.api.types.is_numeric_dtype(df[col]):
                    non_numeric_features.append(col)
            
            if non_numeric_features:
                return False, f"Non-numeric features detected: {non_numeric_features}. Please encode categorical variables first."
            
            # Warnings for perceptron-specific considerations
            warnings = []
            
            if n_features > 1000:
                warnings.append(f"High-dimensional data ({n_features} features) - Perceptron should handle this well")
            
            if n_samples > 50000:
                warnings.append(f"Large dataset ({n_samples} samples) - consider online learning with partial_fit")
            
            if n_classes > 10:
                warnings.append(f"Many classes ({n_classes}) - performance may vary")
            
            # Note about linear separability
            warnings.append("Perceptron requires linearly separable data for optimal performance")
            
            # Compatibility message
            message_parts = [f"✅ Compatible with {n_samples} samples, {n_features} features, {n_classes} classes"]
            
            if warnings:
                message_parts.append("⚠️ Notes: " + "; ".join(warnings))
            
            return True, " | ".join(message_parts)
            
        except Exception as e:
            return False, f"Compatibility check failed: {str(e)}"
    
    def get_model_params(self) -> Dict[str, Any]:
        """Get current model parameters"""
        return {
            'penalty': self.penalty,
            'alpha': self.alpha,
            'l1_ratio': self.l1_ratio,
            'fit_intercept': self.fit_intercept,
            'max_iter': self.max_iter,
            'tol': self.tol,
            'shuffle': self.shuffle,
            'eta0': self.eta0,
            'random_state': self.random_state,
            'early_stopping': self.early_stopping,
            'class_weight': self.class_weight
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
            "n_iterations": getattr(self.model_, 'n_iter_', 'N/A'),
            "converged": not hasattr(self.model_, 'n_iter_') or self.model_.n_iter_ < self.max_iter,
            "coefficients_shape": self.model_.coef_.shape if hasattr(self.model_, 'coef_') else 'N/A'
        }
        
        return info

    def get_algorithm_specific_metrics(self, y_true: Optional[np.ndarray] = None, y_pred: Optional[np.ndarray] = None, y_proba: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        Retrieve algorithm-specific metrics for the Perceptron model.

        Parameters:
        -----------
        y_true : np.ndarray, optional
            Actual target values. Not directly used for these specific metrics but kept for API consistency.
        y_pred : np.ndarray, optional
            Predicted target values. Not directly used for these specific metrics but kept for API consistency.
        y_proba : np.ndarray, optional
            Predicted probabilities. Not applicable for Perceptron.

        Returns:
        --------
        Dict[str, Any]
            A dictionary containing algorithm-specific metrics.
        """
        if not self.is_fitted_ or not hasattr(self.model_, 'n_iter_'):
            return {"error": "Model not fitted or n_iter_ attribute not available. Cannot retrieve Perceptron specific metrics."}

        metrics = {}
        prefix = "perceptron_"

        # Number of iterations
        n_iterations = getattr(self.model_, 'n_iter_', None)
        if n_iterations is not None:
            metrics[f"{prefix}n_iterations"] = int(n_iterations) # Ensure it's an int

            # Convergence status
            # Converged if n_iter_ is less than max_iter (assuming tol was met)
            # sklearn Perceptron stops if validation score is not improving (early_stopping)
            # or if loss is below tol for n_iter_no_change iterations.
            # A simpler check is if n_iter_ < max_iter.
            converged = n_iterations < self.max_iter
            metrics[f"{prefix}converged"] = bool(converged)
        else:
            metrics[f"{prefix}n_iterations"] = "N/A"
            metrics[f"{prefix}converged"] = "N/A"
            
        if not metrics:
            metrics['info'] = "No specific Perceptron metrics were available or calculated."
            
        return metrics

def get_plugin():
    """Factory function to get plugin instance"""
    return PerceptronPlugin()

