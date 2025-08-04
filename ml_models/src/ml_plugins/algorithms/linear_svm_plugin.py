import streamlit as st
import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple, Optional, List
from sklearn.svm import SVC, LinearSVC
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

class LinearSVMPlugin(BaseEstimator, ClassifierMixin, MLPlugin):
    """
    Linear Support Vector Machine Plugin
    
    Optimized for high-dimensional data with linear decision boundaries.
    Particularly effective for text classification, gene expression data,
    and sparse feature spaces.
    """
    
    def __init__(self, 
                 C=1.0,
                 penalty='l2',
                 loss='squared_hinge',
                 dual=True,
                 tol=1e-4,
                 max_iter=1000,
                 random_state=42,
                 class_weight=None,
                 verbose=0,
                 fit_intercept=True,
                 intercept_scaling=1.0,
                 multi_class='ovr'):
        """
        Initialize Linear SVM with comprehensive parameter support
        
        Parameters:
        -----------
        C : float, default=1.0
            Regularization parameter. Smaller values specify stronger regularization.
        penalty : {'l1', 'l2'}, default='l2'
            Norm used in the penalization
        loss : {'hinge', 'squared_hinge'}, default='squared_hinge'
            Loss function
        dual : bool, default=True
            Select the algorithm to solve the dual or primal optimization problem
        tol : float, default=1e-4
            Tolerance for stopping criterion
        max_iter : int, default=1000
            Maximum number of iterations
        random_state : int, default=42
            Random state for reproducibility
        class_weight : dict or 'balanced', default=None
            Weights associated with classes
        verbose : int, default=0
            Verbosity level
        fit_intercept : bool, default=True
            Whether to calculate the intercept
        intercept_scaling : float, default=1.0
            Scaling for the intercept
        multi_class : {'ovr', 'crammer_singer'}, default='ovr'
            Multi-class strategy
        """
        super().__init__()
        
        # Algorithm parameters
        self.C = C
        self.penalty = penalty
        self.loss = loss
        self.dual = dual
        self.tol = tol
        self.max_iter = max_iter
        self.random_state = random_state
        self.class_weight = class_weight
        self.verbose = verbose
        self.fit_intercept = fit_intercept
        self.intercept_scaling = intercept_scaling
        self.multi_class = multi_class
        
        # Plugin metadata
        self._name = "Linear SVM"
        self._description = "Linear Support Vector Machine optimized for high-dimensional data with fast training and excellent generalization"
        self._category = "Linear Models"
        self._algorithm_type = "Support Vector Machine"
        self._paper_reference = "Cortes, C., & Vapnik, V. (1995). Support-vector networks. Machine learning, 20(3), 273-297."
        
        # Capability flags
        self._supports_classification = True
        self._supports_regression = False
        self._supports_binary = True
        self._supports_multiclass = True
        self._min_samples_required = 20
        self._handles_missing_values = False
        self._requires_scaling = True
        self._supports_sparse = True
        self._is_linear = True
        self._provides_feature_importance = True
        self._provides_probabilities = False  # Linear SVM doesn't provide probabilities
        
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
                "Excellent for high-dimensional data",
                "Fast training and prediction",
                "Memory efficient",
                "Good generalization",
                "Robust to overfitting",
                "Handles sparse data well"
            ],
            "weaknesses": [
                "Assumes linear separability",
                "Sensitive to feature scaling",
                "No probability estimates",
                "Limited with non-linear patterns"
            ],
            "use_cases": [
                "Text classification",
                "Gene expression analysis",
                "High-dimensional sparse data",
                "Large datasets with many features",
                "When interpretability is important"
            ],
            "complexity": {
                "training": "O(n × m)",
                "prediction": "O(m)",
                "space": "O(m)"
            }
        }
    
    def fit(self, X, y, sample_weight=None):
        """
        Fit the Linear SVM model
        
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
        
        # Scale features (highly recommended for SVM)
        self.scaler_ = StandardScaler()
        X_scaled = self.scaler_.fit_transform(X)
        
        # Handle dual parameter based on data size
        # For large datasets, primal is often faster
        effective_dual = self.dual
        if X.shape[0] > X.shape[1] and self.penalty == 'l2':
            effective_dual = False
            
        # Configure model based on problem type
        if len(self.classes_) == 2:
            # Binary classification
            self.model_ = LinearSVC(
                C=self.C,
                penalty=self.penalty,
                loss=self.loss,
                dual=effective_dual,
                tol=self.tol,
                max_iter=self.max_iter,
                random_state=self.random_state,
                class_weight=self.class_weight,
                verbose=self.verbose,
                fit_intercept=self.fit_intercept,
                intercept_scaling=self.intercept_scaling
            )
        else:
            # Multi-class classification
            self.model_ = LinearSVC(
                C=self.C,
                penalty=self.penalty,
                loss=self.loss,
                dual=effective_dual,
                tol=self.tol,
                max_iter=self.max_iter,
                random_state=self.random_state,
                class_weight=self.class_weight,
                verbose=self.verbose,
                fit_intercept=self.fit_intercept,
                intercept_scaling=self.intercept_scaling,
                multi_class=self.multi_class
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
        Distance of the samples to the separating hyperplane
        
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
        Get feature importance (coefficients for linear SVM)
        
        Returns:
        --------
        importance : array, shape (n_features,)
            Feature importance scores
        """
        if not self.is_fitted_:
            return None
            
        # For linear SVM, feature importance is the absolute value of coefficients
        if len(self.classes_) == 2:
            # Binary classification
            importance = np.abs(self.model_.coef_[0])
        else:
            # Multi-class: average absolute coefficients across all classes
            importance = np.mean(np.abs(self.model_.coef_), axis=0)
            
        return importance
    
    def get_hyperparameter_config(self, key_prefix: str) -> Dict[str, Any]:
        """Generate Streamlit UI for hyperparameters"""
        
        st.sidebar.markdown("### 🔧 Linear SVM Configuration")
        
        # Create tabs for different parameter groups
        tab1, tab2, tab3 = st.sidebar.tabs(["Core", "Advanced", "Info"])
        
        with tab1:
            st.markdown("**Core Parameters**")
            
            # Regularization parameter C
            C = st.number_input(
                "Regularization (C):",
                value=float(self.C),
                min_value=0.001,
                max_value=1000.0,
                step=0.1,
                help="Smaller values = stronger regularization",
                key=f"{key_prefix}_C"
            )
            
            # Penalty type
            penalty = st.selectbox(
                "Penalty:",
                options=['l1', 'l2'],
                index=['l1', 'l2'].index(self.penalty),
                help="L1: Sparse solutions, L2: Dense solutions",
                key=f"{key_prefix}_penalty"
            )
            
            # Loss function
            loss = st.selectbox(
                "Loss Function:",
                options=['hinge', 'squared_hinge'],
                index=['hinge', 'squared_hinge'].index(self.loss),
                help="Hinge: Standard SVM, Squared Hinge: Differentiable",
                key=f"{key_prefix}_loss"
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
                help="Stopping criterion tolerance",
                key=f"{key_prefix}_tol"
            )
            
            # Max iterations
            max_iter = st.number_input(
                "Max Iterations:",
                value=int(self.max_iter),
                min_value=100,
                max_value=10000,
                step=100,
                help="Maximum number of iterations",
                key=f"{key_prefix}_max_iter"
            )
            
            # Multi-class strategy
            multi_class = st.selectbox(
                "Multi-class Strategy:",
                options=['ovr', 'crammer_singer'],
                index=['ovr', 'crammer_singer'].index(self.multi_class),
                help="OvR: One-vs-Rest, CS: Crammer-Singer",
                key=f"{key_prefix}_multi_class"
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
            **Linear SVM** is excellent for:
            • High-dimensional data
            • Text classification
            • Sparse datasets
            • When you need fast training
            
            **Note:** Automatically applies feature scaling
            """)
            
            if st.button("Show Algorithm Details", key=f"{key_prefix}_info"):
                info = self.get_algorithm_info()
                st.json(info)
        
        return {
            "C": C,
            "penalty": penalty,
            "loss": loss,
            "class_weight": class_weight,
            "tol": tol,
            "max_iter": max_iter,
            "multi_class": multi_class,
            "random_state": random_state,
            "dual": True,  # Will be auto-adjusted based on data
            "verbose": 0,
            "fit_intercept": True,
            "intercept_scaling": 1.0
        }
    
    def create_model_instance(self, hyperparameters: Dict[str, Any]):
        """Create model instance with given hyperparameters"""
        return LinearSVMPlugin(
            C=hyperparameters.get("C", self.C),
            penalty=hyperparameters.get("penalty", self.penalty),
            loss=hyperparameters.get("loss", self.loss),
            dual=hyperparameters.get("dual", self.dual),
            tol=hyperparameters.get("tol", self.tol),
            max_iter=hyperparameters.get("max_iter", self.max_iter),
            random_state=hyperparameters.get("random_state", self.random_state),
            class_weight=hyperparameters.get("class_weight", self.class_weight),
            verbose=hyperparameters.get("verbose", self.verbose),
            fit_intercept=hyperparameters.get("fit_intercept", self.fit_intercept),
            intercept_scaling=hyperparameters.get("intercept_scaling", self.intercept_scaling),
            multi_class=hyperparameters.get("multi_class", self.multi_class)
        )
    
    def preprocess_data(self, X, y):
        """
        Optional data preprocessing
        
        Linear SVM benefits from:
        1. Feature scaling (handled automatically)
        2. Removing highly correlated features
        3. Feature selection for very high-dimensional data
        """
        # Basic preprocessing is handled in fit() method
        # Additional preprocessing can be added here if needed
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
                return False, "Linear SVM requires complete data (no missing values). Please handle missing values first."
            
            # Check target variable type
            target_values = df[target_column].unique()
            n_classes = len(target_values)
            
            if n_classes < 2:
                return False, "Need at least 2 classes for classification"
            
            if n_classes > 1000:
                return False, f"Too many classes ({n_classes}). Linear SVM works best with fewer classes."
            
            # Check feature types
            feature_columns = [col for col in df.columns if col != target_column]
            non_numeric_features = []
            
            for col in feature_columns:
                if not pd.api.types.is_numeric_dtype(df[col]):
                    non_numeric_features.append(col)
            
            if non_numeric_features:
                return False, f"Non-numeric features detected: {non_numeric_features}. Please encode categorical variables first."
            
            # Performance warnings
            warnings = []
            
            if n_features > 10000:
                warnings.append(f"High-dimensional data ({n_features} features) - Linear SVM should handle this well")
            
            if n_samples > 100000:
                warnings.append(f"Large dataset ({n_samples} samples) - consider using dual=False for better performance")
            
            if n_classes > 10:
                warnings.append(f"Many classes ({n_classes}) - training might be slower")
            
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
            'C': self.C,
            'penalty': self.penalty,
            'loss': self.loss,
            'dual': self.dual,
            'tol': self.tol,
            'max_iter': self.max_iter,
            'random_state': self.random_state,
            'class_weight': self.class_weight,
            'multi_class': self.multi_class
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
            # n_iter_ is the number of iterations, not support vectors for LinearSVC
            "convergence_iterations": getattr(self.model_, 'n_iter_', [None])[0] if hasattr(self.model_, 'n_iter_') and self.model_.n_iter_ is not None else None,
        }
        
        return info

    def get_algorithm_specific_metrics(self, y_true: Optional[np.ndarray] = None, y_pred: Optional[np.ndarray] = None, y_proba: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        Retrieve algorithm-specific metrics for Linear SVM.

        Parameters:
        -----------
        y_true : np.ndarray, optional
            Actual target values. Not directly used for these specific metrics.
        y_pred : np.ndarray, optional
            Predicted target values. Not directly used for these specific metrics.
        y_proba : np.ndarray, optional
            Predicted probabilities. Not applicable for Linear SVM.

        Returns:
        --------
        Dict[str, Any]
            A dictionary containing algorithm-specific metrics.
        """
        if not self.is_fitted_ or not self.model_:
            return {"error": "Model not fitted. Cannot retrieve Linear SVM specific metrics."}

        metrics = {}

        # Number of iterations for convergence
        if hasattr(self.model_, 'n_iter_') and self.model_.n_iter_ is not None:
            # n_iter_ can be an array for multi-class with Crammer-Singer
            if isinstance(self.model_.n_iter_, (list, np.ndarray)):
                 metrics['lsvm_convergence_iterations'] = int(np.mean(self.model_.n_iter_)) # Average iterations
            else:
                 metrics['lsvm_convergence_iterations'] = int(self.model_.n_iter_)
        
        # Coefficient-related metrics
        if hasattr(self.model_, 'coef_'):
            coefficients = self.model_.coef_
            abs_coeffs = np.abs(coefficients)

            if coefficients.ndim == 1: # Binary classification or single set of coeffs
                metrics['lsvm_coefficient_norm_l2'] = float(np.linalg.norm(coefficients))
                metrics['lsvm_mean_abs_coefficient'] = float(np.mean(abs_coeffs))
                if self.penalty == 'l1':
                    metrics['lsvm_coefficient_sparsity'] = float(np.mean(coefficients == 0))
            elif coefficients.ndim == 2: # Multi-class OvR
                metrics['lsvm_coefficient_norm_l2_mean'] = float(np.mean(np.linalg.norm(coefficients, axis=1)))
                metrics['lsvm_mean_abs_coefficient'] = float(np.mean(abs_coeffs)) # Overall mean
                if self.penalty == 'l1':
                    metrics['lsvm_coefficient_sparsity'] = float(np.mean(coefficients == 0)) # Overall sparsity
            
            metrics['lsvm_num_features_with_non_zero_coeffs'] = int(np.sum(np.any(coefficients != 0, axis=0) if coefficients.ndim == 2 else coefficients != 0))


        # Intercept-related metrics
        if hasattr(self.model_, 'intercept_'):
            intercepts = self.model_.intercept_
            metrics['lsvm_mean_abs_intercept'] = float(np.mean(np.abs(intercepts)))
            if len(intercepts) == 1:
                metrics['lsvm_intercept_value'] = float(intercepts[0])
            else:
                # For multi-class, could report mean, or all, or norm. Let's report mean.
                metrics['lsvm_intercept_mean_value'] = float(np.mean(intercepts))

        if not metrics:
            metrics['info'] = "No specific Linear SVM metrics were available."
            
        return metrics


def get_plugin():
    """Factory function to get plugin instance"""
    return LinearSVMPlugin()
