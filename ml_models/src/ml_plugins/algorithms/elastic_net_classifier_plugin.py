import streamlit as st
import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple, Optional, List
from sklearn.linear_model import SGDClassifier, LogisticRegression
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

class ElasticNetClassifierPlugin(BaseEstimator, ClassifierMixin, MLPlugin):
    """
    Elastic Net Classifier Plugin - L1 + L2 Regularization
    
    Combines Ridge (L2) and Lasso (L1) regularization to benefit from both:
    - L1: Sparse solutions and automatic feature selection
    - L2: Handles multicollinearity and provides stability
    
    Particularly effective for high-dimensional data with grouped features.
    """
    
    def __init__(self, 
                 alpha=1.0,
                 l1_ratio=0.5,
                 fit_intercept=True,
                 max_iter=1000,
                 tol=1e-4,
                 shuffle=True,
                 verbose=0,
                 epsilon=0.1,
                 n_jobs=None,
                 random_state=42,
                 learning_rate='optimal',
                 eta0=0.01,
                 power_t=0.5,
                 early_stopping=False,
                 validation_fraction=0.1,
                 n_iter_no_change=5,
                 class_weight=None,
                 warm_start=False,
                 average=False):
        """
        Initialize Elastic Net Classifier with comprehensive parameter support
        
        Parameters:
        -----------
        alpha : float, default=1.0
            Constant that multiplies the regularization term
        l1_ratio : float, default=0.5
            The Elastic Net mixing parameter (0 <= l1_ratio <= 1)
            l1_ratio=0: L2 penalty (Ridge)
            l1_ratio=1: L1 penalty (Lasso)
            0 < l1_ratio < 1: Elastic Net
        fit_intercept : bool, default=True
            Whether the intercept should be estimated
        max_iter : int, default=1000
            Maximum number of iterations over the training data
        tol : float, default=1e-4
            The stopping criterion tolerance
        shuffle : bool, default=True
            Whether or not the training data should be shuffled after each epoch
        verbose : int, default=0
            The verbosity level
        epsilon : float, default=0.1
            Epsilon in the epsilon-insensitive loss functions
        n_jobs : int, default=None
            Number of CPUs to use during the cross-validation
        random_state : int, default=42
            Random state for reproducibility
        learning_rate : string, default='optimal'
            Learning rate schedule: 'constant', 'optimal', 'invscaling', 'adaptive'
        eta0 : float, default=0.01
            Initial learning rate for 'constant', 'invscaling' or 'adaptive'
        power_t : float, default=0.5
            The exponent for inverse scaling learning rate
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
        average : bool or int, default=False
            When set to True, computes the averaged SGD weights and stores
        """
        super().__init__()
        
        # Algorithm parameters
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.fit_intercept = fit_intercept
        self.max_iter = max_iter
        self.tol = tol
        self.shuffle = shuffle
        self.verbose = verbose
        self.epsilon = epsilon
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.learning_rate = learning_rate
        self.eta0 = eta0
        self.power_t = power_t
        self.early_stopping = early_stopping
        self.validation_fraction = validation_fraction
        self.n_iter_no_change = n_iter_no_change
        self.class_weight = class_weight
        self.warm_start = warm_start
        self.average = average
        
        # Plugin metadata
        self._name = "Elastic Net Classifier"
        self._description = "Linear classifier with combined L1 and L2 regularization. Performs automatic feature selection while handling multicollinearity."
        self._category = "Linear Models"
        self._algorithm_type = "Regularized Linear Classifier"
        self._paper_reference = "Zou, H., & Hastie, T. (2005). Regularization and variable selection via the elastic net. Journal of the royal statistical society, 67(2), 301-320."
        
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
        self._provides_probabilities = True
        self._supports_online_learning = True
        self._performs_feature_selection = True
        self._handles_multicollinearity = True
        self._regularization_type = "L1 + L2 (Elastic Net)"
        
        # Internal attributes
        self.model_ = None
        self.scaler_ = None
        self.label_encoder_ = None
        self.feature_names_ = None
        self.n_features_in_ = None
        self.classes_ = None
        self.is_fitted_ = False
        self.selected_features_ = None
        
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
                "Automatic feature selection (L1 component)",
                "Handles multicollinearity (L2 component)",
                "Prevents overfitting with dual regularization",
                "Maintains grouped feature selection",
                "Computationally efficient",
                "Works well with high-dimensional data",
                "Provides sparse solutions",
                "Online learning capability"
            ],
            "weaknesses": [
                "Requires feature scaling",
                "Two hyperparameters to tune (α, l1_ratio)",
                "Assumes linear relationship",
                "Can be sensitive to initialization",
                "May require more iterations to converge"
            ],
            "use_cases": [
                "High-dimensional feature selection",
                "Genomics and bioinformatics",
                "Text classification with feature selection",
                "Financial modeling with many predictors",
                "Image classification with pixel features",
                "Marketing analytics with many variables",
                "Any scenario with grouped correlated features"
            ],
            "l1_ratio_guide": {
                "0.0": "Pure Ridge (L2) - handles multicollinearity, no feature selection",
                "0.1-0.3": "Ridge-dominant - mild feature selection, strong multicollinearity handling",
                "0.4-0.6": "Balanced - moderate feature selection and multicollinearity handling",
                "0.7-0.9": "Lasso-dominant - strong feature selection, some multicollinearity handling",
                "1.0": "Pure Lasso (L1) - maximum feature selection, no multicollinearity handling"
            },
            "complexity": {
                "training": "O(n × m × k)",
                "prediction": "O(m_selected)",
                "space": "O(m)"
            },
            "mathematical_foundation": {
                "objective": "minimize ||Xw - y||² + α(l1_ratio||w||₁ + (1-l1_ratio)||w||²)",
                "l1_component": "α × l1_ratio × ||w||₁ (sparsity)",
                "l2_component": "α × (1-l1_ratio) × ||w||² (smoothness)",
                "feature_selection": "L1 drives coefficients to exactly zero"
            }
        }
    
    def fit(self, X, y, sample_weight=None):
        """
        Fit the Elastic Net Classifier model
        
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
        
        # Scale features (essential for Elastic Net)
        self.scaler_ = StandardScaler()
        X_scaled = self.scaler_.fit_transform(X)
        
        # Create and configure the Elastic Net classifier using SGDClassifier
        # SGDClassifier with loss='log' and penalty='elasticnet' gives us Elastic Net Logistic Regression
        self.model_ = SGDClassifier(
            loss='log',  # Logistic regression loss
            penalty='elasticnet',
            alpha=self.alpha,
            l1_ratio=self.l1_ratio,
            fit_intercept=self.fit_intercept,
            max_iter=self.max_iter,
            tol=self.tol,
            shuffle=self.shuffle,
            verbose=self.verbose,
            epsilon=self.epsilon,
            n_jobs=self.n_jobs,
            random_state=self.random_state,
            learning_rate=self.learning_rate,
            eta0=self.eta0,
            power_t=self.power_t,
            early_stopping=self.early_stopping,
            validation_fraction=self.validation_fraction,
            n_iter_no_change=self.n_iter_no_change,
            class_weight=self.class_weight,
            warm_start=self.warm_start,
            average=self.average
        )
        
        # Train the model
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)
            self.model_.fit(X_scaled, y_encoded)
        
        # Identify selected features (non-zero coefficients)
        if hasattr(self.model_, 'coef_'):
            if len(self.classes_) == 2:
                # Binary classification
                self.selected_features_ = np.where(np.abs(self.model_.coef_[0]) > 1e-10)[0]
            else:
                # Multi-class: feature is selected if non-zero in any class
                self.selected_features_ = np.where(np.any(np.abs(self.model_.coef_) > 1e-10, axis=0))[0]
        
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
    
    def predict_proba(self, X):
        """
        Predict class probabilities for samples in X
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Samples
            
        Returns:
        --------
        probabilities : array, shape (n_samples, n_classes)
            Class probabilities
        """
        check_is_fitted(self, 'is_fitted_')
        X = check_array(X, accept_sparse=True, dtype=np.float64)
        
        X_scaled = self.scaler_.transform(X)
        
        # SGDClassifier doesn't have predict_proba by default, so we use decision_function
        if hasattr(self.model_, 'predict_proba'):
            return self.model_.predict_proba(X_scaled)
        else:
            # Convert decision function to probabilities using sigmoid/softmax
            decision = self.model_.decision_function(X_scaled)
            if len(self.classes_) == 2:
                # Binary classification: use sigmoid
                prob_pos = 1 / (1 + np.exp(-decision))
                return np.vstack([1 - prob_pos, prob_pos]).T
            else:
                # Multi-class: use softmax
                exp_scores = np.exp(decision - np.max(decision, axis=1, keepdims=True))
                return exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
    
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
            # First call to partial_fit - initialize everything
            self.fit(X, y, sample_weight)
            return self
        
        # Subsequent calls - use SGDClassifier's partial_fit
        X = check_array(X, accept_sparse=True, dtype=np.float64)
        X_scaled = self.scaler_.transform(X)
        y_encoded = self.label_encoder_.transform(y)
        
        self.model_.partial_fit(X_scaled, y_encoded, sample_weight=sample_weight)
        
        # Update selected features
        if hasattr(self.model_, 'coef_'):
            if len(self.classes_) == 2:
                self.selected_features_ = np.where(np.abs(self.model_.coef_[0]) > 1e-10)[0]
            else:
                self.selected_features_ = np.where(np.any(np.abs(self.model_.coef_) > 1e-10, axis=0))[0]
        
        return self
    
    def get_feature_importance(self) -> Optional[np.ndarray]:
        """
        Get feature importance (coefficients for Elastic Net classifier)
        
        Returns:
        --------
        importance : array, shape (n_features,)
            Feature importance scores based on absolute coefficients
        """
        if not self.is_fitted_:
            return None
            
        # For Elastic Net, feature importance is based on coefficients
        if len(self.classes_) == 2:
            # Binary classification
            importance = np.abs(self.model_.coef_[0])
        else:
            # Multi-class: average absolute coefficients across all classes
            importance = np.mean(np.abs(self.model_.coef_), axis=0)
            
        return importance
    
    def get_selected_features(self) -> Optional[np.ndarray]:
        """
        Get indices of features selected by L1 regularization
        
        Returns:
        --------
        selected : array
            Indices of selected features (non-zero coefficients)
        """
        return self.selected_features_
    
    def get_feature_selection_info(self) -> Dict[str, Any]:
        """
        Get detailed information about feature selection
        
        Returns:
        --------
        info : dict
            Information about feature selection results
        """
        if not self.is_fitted_:
            return {"status": "Not fitted"}
        
        total_features = self.n_features_in_
        selected_count = len(self.selected_features_) if self.selected_features_ is not None else 0
        
        info = {
            "total_features": total_features,
            "selected_features": selected_count,
            "selection_ratio": selected_count / total_features if total_features > 0 else 0,
            "removed_features": total_features - selected_count,
            "l1_ratio": self.l1_ratio,
            "regularization_strength": self.alpha
        }
        
        if self.feature_names_ and self.selected_features_ is not None:
            info["selected_feature_names"] = [self.feature_names_[i] for i in self.selected_features_]
        
        return info
    
    def get_hyperparameter_config(self, key_prefix: str) -> Dict[str, Any]:
        """Generate Streamlit UI for hyperparameters"""
        
        st.sidebar.markdown("### 🔗 Elastic Net Classifier Configuration")
        
        # Create tabs for different parameter groups
        tab1, tab2, tab3 = st.sidebar.tabs(["Core", "Advanced", "Info"])
        
        with tab1:
            st.markdown("**Core Parameters**")
            
            # Alpha (overall regularization strength)
            alpha = st.number_input(
                "Regularization Strength (α):",
                value=float(self.alpha),
                min_value=0.0001,
                max_value=1000.0,
                step=0.1,
                help="Overall strength of regularization. Higher = more regularization",
                key=f"{key_prefix}_alpha"
            )
            
            # L1 ratio (balance between L1 and L2)
            l1_ratio = st.slider(
                "L1 Ratio (L1 vs L2 mix):",
                min_value=0.0,
                max_value=1.0,
                value=float(self.l1_ratio),
                step=0.05,
                help="0.0=Pure Ridge, 0.5=Balanced, 1.0=Pure Lasso",
                key=f"{key_prefix}_l1_ratio"
            )
            
            # Visual guide for L1 ratio
            if l1_ratio == 0.0:
                st.info("🏔️ Pure Ridge (L2): Handles multicollinearity, no feature selection")
            elif l1_ratio < 0.3:
                st.info("🏔️ Ridge-dominant: Mild feature selection, strong stability")
            elif l1_ratio < 0.7:
                st.info("⚖️ Balanced: Moderate feature selection & multicollinearity handling")
            elif l1_ratio < 1.0:
                st.info("🎯 Lasso-dominant: Strong feature selection, some stability")
            else:
                st.info("🎯 Pure Lasso (L1): Maximum feature selection")
            
            # Learning rate schedule
            learning_rate_options = ['optimal', 'constant', 'invscaling', 'adaptive']
            learning_rate = st.selectbox(
                "Learning Rate Schedule:",
                options=learning_rate_options,
                index=learning_rate_options.index(self.learning_rate),
                help="How learning rate changes over iterations",
                key=f"{key_prefix}_learning_rate"
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
            
            # Max iterations
            max_iter = st.number_input(
                "Max Iterations:",
                value=int(self.max_iter),
                min_value=100,
                max_value=10000,
                step=100,
                help="Maximum number of training iterations",
                key=f"{key_prefix}_max_iter"
            )
            
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
            
            # Initial learning rate (eta0)
            eta0 = st.number_input(
                "Initial Learning Rate (η₀):",
                value=float(self.eta0),
                min_value=0.001,
                max_value=1.0,
                step=0.001,
                help="Initial learning rate for constant/invscaling/adaptive schedules",
                key=f"{key_prefix}_eta0"
            )
            
            # Early stopping
            early_stopping = st.checkbox(
                "Early Stopping",
                value=self.early_stopping,
                help="Stop training when validation score stops improving",
                key=f"{key_prefix}_early_stopping"
            )
            
            if early_stopping:
                n_iter_no_change = st.number_input(
                    "Patience (iterations):",
                    value=int(self.n_iter_no_change),
                    min_value=2,
                    max_value=20,
                    help="Iterations to wait before stopping",
                    key=f"{key_prefix}_n_iter_no_change"
                )
            else:
                n_iter_no_change = self.n_iter_no_change
            
            # Averaging
            average = st.checkbox(
                "Average Weights",
                value=self.average,
                help="Compute averaged SGD weights for better stability",
                key=f"{key_prefix}_average"
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
            **Elastic Net** combines the best of both worlds:
            
            **L1 (Lasso) Component:**
            • Automatic feature selection
            • Sparse solutions
            • Handles irrelevant features
            
            **L2 (Ridge) Component:**
            • Handles multicollinearity
            • Stable solutions
            • Keeps grouped features together
            """)
            
            # Feature selection preview
            if st.button("🎯 Feature Selection Guide", key=f"{key_prefix}_selection_guide"):
                st.markdown("""
                **L1 Ratio Effects:**
                • **0.0-0.2:** Minimal feature selection, maximum stability
                • **0.3-0.4:** Conservative feature selection
                • **0.5-0.6:** Balanced feature selection
                • **0.7-0.8:** Aggressive feature selection
                • **0.9-1.0:** Maximum feature selection, minimal stability
                """)
            
            if st.button("Show Algorithm Details", key=f"{key_prefix}_info"):
                info = self.get_algorithm_info()
                st.json(info)
        
        return {
            "alpha": alpha,
            "l1_ratio": l1_ratio,
            "fit_intercept": True,
            "max_iter": max_iter,
            "tol": tol,
            "shuffle": True,
            "verbose": 0,
            "epsilon": 0.1,
            "n_jobs": None,
            "random_state": random_state,
            "learning_rate": learning_rate,
            "eta0": eta0,
            "power_t": 0.5,
            "early_stopping": early_stopping,
            "validation_fraction": 0.1,
            "n_iter_no_change": n_iter_no_change,
            "class_weight": class_weight,
            "warm_start": False,
            "average": average
        }
    
    def create_model_instance(self, hyperparameters: Dict[str, Any]):
        """Create model instance with given hyperparameters"""
        return ElasticNetClassifierPlugin(
            alpha=hyperparameters.get("alpha", self.alpha),
            l1_ratio=hyperparameters.get("l1_ratio", self.l1_ratio),
            fit_intercept=hyperparameters.get("fit_intercept", self.fit_intercept),
            max_iter=hyperparameters.get("max_iter", self.max_iter),
            tol=hyperparameters.get("tol", self.tol),
            shuffle=hyperparameters.get("shuffle", self.shuffle),
            verbose=hyperparameters.get("verbose", self.verbose),
            epsilon=hyperparameters.get("epsilon", self.epsilon),
            n_jobs=hyperparameters.get("n_jobs", self.n_jobs),
            random_state=hyperparameters.get("random_state", self.random_state),
            learning_rate=hyperparameters.get("learning_rate", self.learning_rate),
            eta0=hyperparameters.get("eta0", self.eta0),
            power_t=hyperparameters.get("power_t", self.power_t),
            early_stopping=hyperparameters.get("early_stopping", self.early_stopping),
            validation_fraction=hyperparameters.get("validation_fraction", self.validation_fraction),
            n_iter_no_change=hyperparameters.get("n_iter_no_change", self.n_iter_no_change),
            class_weight=hyperparameters.get("class_weight", self.class_weight),
            warm_start=hyperparameters.get("warm_start", self.warm_start),
            average=hyperparameters.get("average", self.average)
        )
    
    def preprocess_data(self, X, y):
        """
        Optional data preprocessing
        
        Elastic Net benefits from:
        1. Feature scaling (handled automatically)
        2. Removing constant features
        3. Feature engineering for grouped variables
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
                return False, "Elastic Net Classifier requires complete data (no missing values). Please handle missing values first."
            
            # Check target variable type
            target_values = df[target_column].unique()
            n_classes = len(target_values)
            
            if n_classes < 2:
                return False, "Need at least 2 classes for classification"
            
            if n_classes > 1000:
                return False, f"Too many classes ({n_classes}). Elastic Net works best with fewer classes."
            
            # Check feature types
            feature_columns = [col for col in df.columns if col != target_column]
            non_numeric_features = []
            
            for col in feature_columns:
                if not pd.api.types.is_numeric_dtype(df[col]):
                    non_numeric_features.append(col)
            
            if non_numeric_features:
                return False, f"Non-numeric features detected: {non_numeric_features}. Please encode categorical variables first."
            
            # Elastic Net specific advantages and considerations
            advantages = []
            considerations = []
            
            # High-dimensional data (Elastic Net's strength)
            if n_features > 50:
                advantages.append(f"Excellent for high-dimensional data ({n_features} features)")
            
            # Feature selection benefit
            if n_features > 20:
                advantages.append("Will perform automatic feature selection via L1 regularization")
            
            # Multicollinearity check
            if n_features > 10:
                try:
                    corr_matrix = df[feature_columns].corr()
                    high_corr_pairs = np.where(np.abs(corr_matrix) > 0.7)
                    high_corr_count = len(high_corr_pairs[0]) - n_features  # Subtract diagonal
                    
                    if high_corr_count > 0:
                        advantages.append(f"Handles correlated features well (found {high_corr_count//2} high-correlation pairs)")
                except:
                    pass
            
            # Dataset size considerations
            if n_samples > 100000:
                considerations.append(f"Large dataset ({n_samples} samples) - SGD will be efficient")
            
            if n_features > n_samples:
                advantages.append(f"Handles high-dimensional case well ({n_features} features > {n_samples} samples)")
            
            if n_classes > 10:
                considerations.append(f"Many classes ({n_classes}) - may need more regularization")
            
            # Feature scaling benefit
            feature_ranges = []
            for col in feature_columns[:5]:  # Check first 5 features
                try:
                    col_range = df[col].max() - df[col].min()
                    if col_range > 0:
                        feature_ranges.append(col_range)
                except:
                    pass
            
            if len(feature_ranges) > 1 and max(feature_ranges) / min(feature_ranges) > 100:
                advantages.append("Automatic feature scaling applied (features have different scales)")
            
            # Sparsity potential
            zero_ratio = (df[feature_columns] == 0).sum().sum() / (len(feature_columns) * len(df))
            if zero_ratio > 0.1:
                advantages.append(f"Sparse data detected ({zero_ratio:.1%} zeros) - L1 regularization will help")
            
            # Compatibility message
            message_parts = [f"✅ Compatible with {n_samples} samples, {n_features} features, {n_classes} classes"]
            
            if advantages:
                message_parts.append("🎯 Elastic Net advantages: " + "; ".join(advantages))
            
            if considerations:
                message_parts.append("💡 Considerations: " + "; ".join(considerations))
            
            return True, " | ".join(message_parts)
            
        except Exception as e:
            return False, f"Compatibility check failed: {str(e)}"
    
    def get_model_params(self) -> Dict[str, Any]:
        """Get current model parameters"""
        return {
            'alpha': self.alpha,
            'l1_ratio': self.l1_ratio,
            'fit_intercept': self.fit_intercept,
            'max_iter': self.max_iter,
            'tol': self.tol,
            'learning_rate': self.learning_rate,
            'eta0': self.eta0,
            'early_stopping': self.early_stopping,
            'class_weight': self.class_weight,
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
            "l1_ratio": self.l1_ratio,
            "n_iterations": getattr(self.model_, 'n_iter_', 'N/A'),
            "converged": not hasattr(self.model_, 'n_iter_') or self.model_.n_iter_ < self.max_iter
        }
        
        # Feature selection information
        selection_info = self.get_feature_selection_info()
        info.update(selection_info)
        
        # Coefficient statistics
        if hasattr(self.model_, 'coef_'):
            coef_stats = {
                "coefficient_l1_norm": np.sum(np.abs(self.model_.coef_)),
                "coefficient_l2_norm": np.linalg.norm(self.model_.coef_),
                "max_coefficient": np.max(np.abs(self.model_.coef_)),
                "mean_coefficient": np.mean(np.abs(self.model_.coef_)),
                "zero_coefficients": np.sum(np.abs(self.model_.coef_) < 1e-10),
                "sparsity_ratio": np.sum(np.abs(self.model_.coef_) < 1e-10) / self.model_.coef_.size
            }
            info.update(coef_stats)
        
        return info

    def get_algorithm_specific_metrics(self,
                                       y_true: np.ndarray,
                                       y_pred: np.ndarray,
                                       y_proba: Optional[np.ndarray] = None
                                       ) -> Dict[str, Any]:
        """
        Calculate Elastic Net Classifier-specific metrics based on the fitted model.

        These metrics describe the internal state of the model, such as
        coefficient sparsity and convergence. The y_true, y_pred, and y_proba
        parameters (typically for test set evaluation) are not used for these
        internal model-specific metrics.

        Args:
            y_true: Ground truth target values (not used for these metrics).
            y_pred: Predicted target values (not used for these metrics).
            y_proba: Predicted probabilities (not used for these metrics).

        Returns:
            A dictionary of Elastic Net Classifier-specific metrics.
        """
        metrics = {}
        if not self.is_fitted_ or self.model_ is None:
            metrics["status"] = "Model not fitted"
            return metrics

        # Metrics from feature selection
        selection_info = self.get_feature_selection_info()
        metrics['num_selected_features'] = selection_info.get('selected_features')
        metrics['feature_selection_ratio'] = selection_info.get('selection_ratio')

        # Metrics from coefficients
        if hasattr(self.model_, 'coef_') and self.model_.coef_ is not None:
            coef_array = self.model_.coef_
            metrics['coefficient_l1_norm'] = float(np.sum(np.abs(coef_array)))
            metrics['coefficient_l2_norm'] = float(np.linalg.norm(coef_array))
            metrics['max_abs_coefficient'] = float(np.max(np.abs(coef_array)))
            
            # Sparsity: proportion of zero coefficients
            num_zero_coefficients = np.sum(np.abs(coef_array) < 1e-10)
            total_coefficients = coef_array.size
            if total_coefficients > 0:
                metrics['coefficient_sparsity_ratio'] = float(num_zero_coefficients / total_coefficients)
            else:
                metrics['coefficient_sparsity_ratio'] = None # Or 0.0 if appropriate

        # Iteration and convergence information
        if hasattr(self.model_, 'n_iter_') and self.model_.n_iter_ is not None:
            metrics['num_iterations_to_converge'] = int(self.model_.n_iter_)
            if hasattr(self.model_, 'max_iter') and self.model_.max_iter is not None:
                 metrics['model_converged'] = bool(self.model_.n_iter_ < self.model_.max_iter)
            else: # Fallback if max_iter not directly on model_ but on self
                 metrics['model_converged'] = bool(self.model_.n_iter_ < self.max_iter)

        else:
            metrics['num_iterations_to_converge'] = None
            metrics['model_converged'] = None
            
        # Regularization parameters applied
        metrics['alpha_applied'] = self.alpha
        metrics['l1_ratio_applied'] = self.l1_ratio

        # Remove None values for cleaner output
        metrics = {k: v for k, v in metrics.items() if v is not None}

        return metrics    

def get_plugin():
    """Factory function to get plugin instance"""
    return ElasticNetClassifierPlugin()
