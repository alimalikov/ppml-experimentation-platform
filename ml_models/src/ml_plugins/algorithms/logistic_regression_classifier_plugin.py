import streamlit as st
import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple, Optional, List
from sklearn.linear_model import LogisticRegression
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import validation_curve
import matplotlib.pyplot as plt
import seaborn as sns
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

class LogisticRegressionClassifierPlugin(BaseEstimator, ClassifierMixin, MLPlugin):
    """
    Logistic Regression Classifier Plugin - Interpretable Baseline
    
    Logistic regression is a fundamental linear classifier that provides
    interpretable probability outputs and serves as an excellent baseline
    for classification tasks. It's particularly valuable for understanding
    feature relationships and providing explainable predictions.
    """
    
    def __init__(self, 
                 penalty='l2',
                 dual=False,
                 tol=1e-4,
                 C=1.0,
                 fit_intercept=True,
                 intercept_scaling=1.0,
                 class_weight=None,
                 random_state=42,
                 solver='lbfgs',
                 max_iter=100,
                 multi_class='auto',
                 verbose=0,
                 warm_start=False,
                 n_jobs=None,
                 l1_ratio=None):
        """
        Initialize Logistic Regression Classifier with comprehensive parameter support
        
        Parameters:
        -----------
        penalty : {'l1', 'l2', 'elasticnet', None}, default='l2'
            Regularization norm used in the penalization
        dual : bool, default=False
            Dual or primal formulation (prefer primal for n_samples > n_features)
        tol : float, default=1e-4
            Tolerance for stopping criteria
        C : float, default=1.0
            Inverse of regularization strength (smaller values = stronger regularization)
        fit_intercept : bool, default=True
            Whether to calculate the intercept for this model
        intercept_scaling : float, default=1.0
            When fit_intercept is True, instance vector x becomes [x, intercept_scaling]
        class_weight : dict or 'balanced', default=None
            Weights associated with classes
        random_state : int, default=42
            Random seed for reproducibility
        solver : {'newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'}, default='lbfgs'
            Algorithm to use in the optimization problem
        max_iter : int, default=100
            Maximum number of iterations for solvers to converge
        multi_class : {'auto', 'ovr', 'multinomial'}, default='auto'
            Strategy for handling multiclass classification
        verbose : int, default=0
            For the liblinear and lbfgs solvers set verbose to any positive number for verbosity
        warm_start : bool, default=False
            When set to True, reuse the solution of the previous call to fit
        n_jobs : int, default=None
            Number of CPU cores used when parallelizing over classes
        l1_ratio : float, default=None
            The Elastic-Net mixing parameter (only used if penalty='elasticnet')
        """
        super().__init__()
        
        # Algorithm parameters
        self.penalty = penalty
        self.dual = dual
        self.tol = tol
        self.C = C
        self.fit_intercept = fit_intercept
        self.intercept_scaling = intercept_scaling
        self.class_weight = class_weight
        self.random_state = random_state
        self.solver = solver
        self.max_iter = max_iter
        self.multi_class = multi_class
        self.verbose = verbose
        self.warm_start = warm_start
        self.n_jobs = n_jobs
        self.l1_ratio = l1_ratio
        
        # Plugin metadata
        self._name = "Logistic Regression"
        self._description = "Interpretable linear classifier that provides probability outputs and clear feature importance. Excellent baseline model."
        self._category = "Linear Models"
        self._algorithm_type = "Linear Classifier"
        self._paper_reference = "Cox, D. R. (1958). The regression analysis of binary sequences. Journal of the Royal Statistical Society, 20(2), 215-242."
        
        # Capability flags
        self._supports_classification = True
        self._supports_regression = False
        self._supports_binary = True
        self._supports_multiclass = True
        self._min_samples_required = 10
        self._handles_missing_values = False
        self._requires_scaling = True  # Important for logistic regression
        self._supports_sparse = True
        self._is_linear = True
        self._provides_feature_importance = True
        self._provides_probabilities = True
        self._highly_interpretable = True
        self._provides_coefficients = True
        self._supports_regularization = True
        
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
            "mathematical_foundation": {
                "model": "P(y=1|x) = 1 / (1 + exp(-(β₀ + β₁x₁ + ... + βₙxₙ)))",
                "loss_function": "Log-likelihood with optional regularization",
                "decision_boundary": "Linear hyperplane in feature space",
                "optimization": "Maximum likelihood estimation"
            },
            "strengths": [
                "Highly interpretable coefficients",
                "Provides calibrated probability estimates",
                "Fast training and prediction",
                "No assumptions about feature distributions",
                "Excellent baseline model",
                "Handles multiclass naturally (multinomial)",
                "Regularization prevents overfitting",
                "Well-established statistical theory",
                "Converges to global optimum",
                "Robust to outliers in features (not target)"
            ],
            "weaknesses": [
                "Assumes linear relationship between features and log-odds",
                "Sensitive to outliers in target variable",
                "Requires feature scaling for best performance",
                "Can struggle with complex non-linear patterns",
                "Assumes independence of observations",
                "May underfit complex datasets",
                "Sensitive to multicollinearity"
            ],
            "use_cases": [
                "Baseline model for any classification task",
                "When interpretability is crucial",
                "Binary classification problems",
                "Multiclass classification with clear boundaries",
                "Feature importance analysis",
                "Probability estimation tasks",
                "Medical diagnosis (explainable decisions)",
                "Marketing conversion prediction",
                "Risk assessment and scoring",
                "A/B test analysis",
                "Educational research (understanding relationships)"
            ],
            "interpretability_features": {
                "coefficients": "Direct feature impact on log-odds",
                "odds_ratios": "exp(coefficient) = odds ratio",
                "probability_outputs": "Well-calibrated probabilities",
                "feature_importance": "Absolute coefficient values",
                "statistical_significance": "P-values and confidence intervals"
            },
            "complexity": {
                "training": "O(n × m × k)",
                "prediction": "O(m)",
                "space": "O(m)"
            },
            "regularization_guide": {
                "l1_penalty": "Feature selection, sparse solutions",
                "l2_penalty": "Coefficient shrinkage, handles multicollinearity",
                "elasticnet": "Combination of L1 and L2",
                "C_parameter": "Higher C = less regularization"
            },
            "solver_guide": {
                "lbfgs": "Good default for small datasets",
                "liblinear": "Good for small datasets, supports L1",
                "newton-cg": "Handles multinomial loss well",
                "sag/saga": "Fast for large datasets"
            }
        }
    
    def fit(self, X, y, sample_weight=None):
        """
        Fit the Logistic Regression Classifier model
        
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
        
        # Scale features (essential for logistic regression)
        self.scaler_ = StandardScaler()
        X_scaled = self.scaler_.fit_transform(X)
        
        # Create and configure the Logistic Regression model
        self.model_ = LogisticRegression(
            penalty=self.penalty,
            dual=self.dual,
            tol=self.tol,
            C=self.C,
            fit_intercept=self.fit_intercept,
            intercept_scaling=self.intercept_scaling,
            class_weight=self.class_weight,
            random_state=self.random_state,
            solver=self.solver,
            max_iter=self.max_iter,
            multi_class=self.multi_class,
            verbose=self.verbose,
            warm_start=self.warm_start,
            n_jobs=self.n_jobs,
            l1_ratio=self.l1_ratio
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
        return self.model_.predict_proba(X_scaled)
    
    def predict_log_proba(self, X):
        """
        Predict log probabilities for samples in X
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Samples
            
        Returns:
        --------
        log_probabilities : array, shape (n_samples, n_classes)
            Log probabilities
        """
        check_is_fitted(self, 'is_fitted_')
        X = check_array(X, accept_sparse=True, dtype=np.float64)
        
        X_scaled = self.scaler_.transform(X)
        return self.model_.predict_log_proba(X_scaled)
    
    def decision_function(self, X):
        """
        Predict confidence scores for samples
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Samples
            
        Returns:
        --------
        scores : array, shape (n_samples,) or (n_samples, n_classes)
            Confidence scores
        """
        check_is_fitted(self, 'is_fitted_')
        X = check_array(X, accept_sparse=True, dtype=np.float64)
        
        X_scaled = self.scaler_.transform(X)
        return self.model_.decision_function(X_scaled)
    
    def get_feature_importance(self) -> Optional[np.ndarray]:
        """
        Get feature importance based on coefficient magnitudes
        
        Returns:
        --------
        importance : array, shape (n_features,)
            Feature importance scores based on absolute coefficients
        """
        if not self.is_fitted_:
            return None
            
        # For logistic regression, feature importance is based on coefficients
        if len(self.classes_) == 2:
            # Binary classification
            importance = np.abs(self.model_.coef_[0])
        else:
            # Multi-class: average absolute coefficients across all classes
            importance = np.mean(np.abs(self.model_.coef_), axis=0)
            
        return importance
    
    def get_coefficients(self) -> Optional[Dict[str, Any]]:
        """
        Get model coefficients and intercepts
        
        Returns:
        --------
        coefficients : dict
            Dictionary containing coefficients, intercepts, and interpretations
        """
        if not self.is_fitted_:
            return None
        
        coef_info = {
            "coefficients": self.model_.coef_.tolist(),
            "intercept": self.model_.intercept_.tolist(),
            "feature_names": self.feature_names_,
            "classes": self.classes_.tolist(),
            "n_classes": len(self.classes_)
        }
        
        # Add coefficient interpretations
        if len(self.classes_) == 2:
            # Binary classification
            coef_info["coefficient_interpretation"] = [
                {
                    "feature": self.feature_names_[i],
                    "coefficient": float(self.model_.coef_[0, i]),
                    "odds_ratio": float(np.exp(self.model_.coef_[0, i])),
                    "interpretation": self._interpret_coefficient(self.model_.coef_[0, i])
                }
                for i in range(len(self.feature_names_))
            ]
        else:
            # Multi-class
            coef_info["coefficient_interpretation"] = {}
            for class_idx, class_name in enumerate(self.classes_):
                coef_info["coefficient_interpretation"][str(class_name)] = [
                    {
                        "feature": self.feature_names_[i],
                        "coefficient": float(self.model_.coef_[class_idx, i]),
                        "odds_ratio": float(np.exp(self.model_.coef_[class_idx, i])),
                        "interpretation": self._interpret_coefficient(self.model_.coef_[class_idx, i])
                    }
                    for i in range(len(self.feature_names_))
                ]
        
        return coef_info
    
    def _interpret_coefficient(self, coef_value):
        """Interpret a coefficient value"""
        if abs(coef_value) < 0.01:
            return "Negligible effect"
        elif coef_value > 0:
            return f"Increases log-odds by {coef_value:.3f} (OR: {np.exp(coef_value):.3f})"
        else:
            return f"Decreases log-odds by {abs(coef_value):.3f} (OR: {np.exp(coef_value):.3f})"
    
    def get_odds_ratios(self) -> Optional[Dict[str, Any]]:
        """
        Get odds ratios for feature interpretability
        
        Returns:
        --------
        odds_ratios : dict
            Dictionary containing odds ratios and their interpretations
        """
        if not self.is_fitted_:
            return None
        
        coefficients = self.get_coefficients()
        if coefficients is None:
            return None
        
        if len(self.classes_) == 2:
            # Binary classification
            odds_ratios = {
                "interpretation": "Odds ratio = exp(coefficient). OR > 1: increases odds, OR < 1: decreases odds",
                "feature_odds_ratios": [
                    {
                        "feature": self.feature_names_[i],
                        "odds_ratio": float(np.exp(self.model_.coef_[0, i])),
                        "effect": self._interpret_odds_ratio(np.exp(self.model_.coef_[0, i]))
                    }
                    for i in range(len(self.feature_names_))
                ]
            }
        else:
            # Multi-class (odds ratios are more complex in multinomial case)
            odds_ratios = {
                "interpretation": "In multinomial logistic regression, coefficients represent log-odds relative to the reference class",
                "note": "Odds ratios are relative to the baseline class and other classes simultaneously"
            }
        
        return odds_ratios
    
    def _interpret_odds_ratio(self, odds_ratio):
        """Interpret an odds ratio value"""
        if 0.95 <= odds_ratio <= 1.05:
            return "No significant effect"
        elif odds_ratio > 1:
            return f"{((odds_ratio - 1) * 100):.1f}% increase in odds"
        else:
            return f"{((1 - odds_ratio) * 100):.1f}% decrease in odds"
    
    def plot_coefficients(self, top_n=20, figsize=(10, 8)):
        """
        Create a coefficient plot for feature importance visualization
        
        Parameters:
        -----------
        top_n : int, default=20
            Number of top features to display
        figsize : tuple, default=(10, 8)
            Figure size
            
        Returns:
        --------
        fig : matplotlib figure
            Coefficient plot
        """
        if not self.is_fitted_:
            return None
        
        coef_info = self.get_coefficients()
        if coef_info is None:
            return None
        
        fig, ax = plt.subplots(figsize=figsize)
        
        if len(self.classes_) == 2:
            # Binary classification
            coefficients = self.model_.coef_[0]
            feature_names = self.feature_names_
            
            # Get top features by absolute coefficient value
            indices = np.argsort(np.abs(coefficients))[::-1][:top_n]
            top_coefs = coefficients[indices]
            top_features = [feature_names[i] for i in indices]
            
            # Create horizontal bar plot
            colors = ['red' if coef < 0 else 'blue' for coef in top_coefs]
            bars = ax.barh(range(len(top_features)), top_coefs, color=colors, alpha=0.7)
            
            ax.set_yticks(range(len(top_features)))
            ax.set_yticklabels(top_features)
            ax.invert_yaxis()
            ax.set_xlabel('Coefficient Value')
            ax.set_title(f'Top {len(top_features)} Feature Coefficients - Logistic Regression')
            ax.axvline(x=0, color='black', linestyle='--', alpha=0.5)
            
            # Add value labels
            for i, bar in enumerate(bars):
                width = bar.get_width()
                ax.text(width + (0.01 if width >= 0 else -0.01), bar.get_y() + bar.get_height()/2, 
                       f'{width:.3f}', ha='left' if width >= 0 else 'right', va='center', fontsize=9)
        
        else:
            # Multi-class: show heatmap of coefficients
            coefficients = self.model_.coef_
            
            # Select top features based on maximum absolute coefficient across classes
            max_abs_coefs = np.max(np.abs(coefficients), axis=0)
            indices = np.argsort(max_abs_coefs)[::-1][:top_n]
            
            selected_coefs = coefficients[:, indices]
            selected_features = [self.feature_names_[i] for i in indices]
            
            # Create heatmap
            sns.heatmap(selected_coefs, 
                       xticklabels=selected_features, 
                       yticklabels=self.classes_,
                       annot=True, 
                       fmt='.3f', 
                       center=0, 
                       cmap='RdBu_r',
                       ax=ax)
            
            ax.set_title(f'Feature Coefficients by Class - Logistic Regression')
            plt.xticks(rotation=45, ha='right')
        
        plt.tight_layout()
        return fig
    
    def get_hyperparameter_config(self, key_prefix: str) -> Dict[str, Any]:
        """Generate Streamlit UI for hyperparameters"""
        
        st.sidebar.markdown("### 📊 Logistic Regression Configuration")
        
        # Create tabs for different parameter groups
        tab1, tab2, tab3 = st.sidebar.tabs(["Core", "Advanced", "Info"])
        
        with tab1:
            st.markdown("**Core Parameters**")
            
            # Regularization strength (C)
            C = st.number_input(
                "Regularization Strength (C):",
                value=float(self.C),
                min_value=0.001,
                max_value=1000.0,
                step=0.1,
                help="Inverse of regularization strength. Higher C = less regularization",
                key=f"{key_prefix}_C"
            )
            
            # Penalty type
            penalty_options = ['l1', 'l2', 'elasticnet', 'none']
            penalty = st.selectbox(
                "Penalty Type:",
                options=penalty_options,
                index=penalty_options.index(self.penalty) if self.penalty in penalty_options else 1,
                help="l1: Feature selection, l2: Coefficient shrinkage, elasticnet: Both",
                key=f"{key_prefix}_penalty"
            )
            
            # L1 ratio (only for elasticnet)
            if penalty == 'elasticnet':
                l1_ratio = st.slider(
                    "L1 Ratio (ElasticNet):",
                    min_value=0.0,
                    max_value=1.0,
                    value=float(self.l1_ratio) if self.l1_ratio else 0.5,
                    step=0.05,
                    help="L1 vs L2 penalty ratio. 0=Ridge, 1=Lasso",
                    key=f"{key_prefix}_l1_ratio"
                )
            else:
                l1_ratio = None
            
            # Solver
            solver_options = ['lbfgs', 'liblinear', 'newton-cg', 'sag', 'saga']
            
            # Filter solvers based on penalty
            if penalty == 'l1':
                available_solvers = ['liblinear', 'saga']
            elif penalty == 'elasticnet':
                available_solvers = ['saga']
            elif penalty == 'none':
                available_solvers = ['lbfgs', 'newton-cg', 'sag', 'saga']
            else:  # l2
                available_solvers = solver_options
            
            solver = st.selectbox(
                "Solver:",
                options=available_solvers,
                index=0 if self.solver not in available_solvers else available_solvers.index(self.solver),
                help="lbfgs: Good default, liblinear: Small datasets, saga: Large datasets",
                key=f"{key_prefix}_solver"
            )
            
            # Max iterations
            max_iter = st.slider(
                "Max Iterations:",
                min_value=50,
                max_value=1000,
                value=int(self.max_iter),
                step=50,
                help="Maximum number of iterations for solver convergence",
                key=f"{key_prefix}_max_iter"
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
            
            # Multi-class strategy
            multi_class = st.selectbox(
                "Multi-class Strategy:",
                options=['auto', 'ovr', 'multinomial'],
                index=['auto', 'ovr', 'multinomial'].index(self.multi_class),
                help="auto: Selects best, ovr: One-vs-Rest, multinomial: Multinomial loss",
                key=f"{key_prefix}_multi_class"
            )
            
            # Tolerance
            tol = st.number_input(
                "Tolerance:",
                value=float(self.tol),
                min_value=1e-6,
                max_value=1e-2,
                step=1e-6,
                format="%.2e",
                help="Tolerance for stopping criteria",
                key=f"{key_prefix}_tol"
            )
            
            # Fit intercept
            fit_intercept = st.checkbox(
                "Fit Intercept",
                value=self.fit_intercept,
                help="Whether to calculate the intercept (usually True)",
                key=f"{key_prefix}_fit_intercept"
            )
            
            # Warm start
            warm_start = st.checkbox(
                "Warm Start",
                value=self.warm_start,
                help="Reuse solution from previous fit as initialization",
                key=f"{key_prefix}_warm_start"
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
            **Logistic Regression** excels at:
            • Providing interpretable coefficients
            • Giving calibrated probability estimates
            • Serving as a strong baseline model
            • Feature importance analysis
            • Fast training and prediction
            
            **Mathematical Foundation:**
            • Linear decision boundary
            • Sigmoid activation function
            • Maximum likelihood estimation
            • Well-calibrated probabilities
            """)
            
            # Interpretability showcase
            if st.button("🔍 Interpretability Guide", key=f"{key_prefix}_interpretability"):
                st.markdown("""
                **Coefficient Interpretation:**
                • **Positive coefficient**: Increases probability of positive class
                • **Negative coefficient**: Decreases probability of positive class
                • **Magnitude**: Strength of the effect
                • **Odds ratio**: exp(coefficient) = multiplicative effect on odds
                """)
            
            # Regularization guide
            if st.button("⚖️ Regularization Guide", key=f"{key_prefix}_regularization"):
                st.markdown("""
                **Regularization Effects:**
                • **C → ∞**: No regularization (may overfit)
                • **C = 1**: Moderate regularization (good default)
                • **C → 0**: Strong regularization (may underfit)
                • **L1**: Feature selection, sparse solutions
                • **L2**: Coefficient shrinkage, handles multicollinearity
                """)
            
            if st.button("Show Algorithm Details", key=f"{key_prefix}_info"):
                info = self.get_algorithm_info()
                st.json(info)
        
        return {
            "penalty": penalty,
            "dual": False,  # Always False for most cases
            "tol": tol,
            "C": C,
            "fit_intercept": fit_intercept,
            "intercept_scaling": 1.0,
            "class_weight": class_weight,
            "random_state": random_state,
            "solver": solver,
            "max_iter": max_iter,
            "multi_class": multi_class,
            "verbose": 0,
            "warm_start": warm_start,
            "n_jobs": None,
            "l1_ratio": l1_ratio
        }
    
    def create_model_instance(self, hyperparameters: Dict[str, Any]):
        """Create model instance with given hyperparameters"""
        return LogisticRegressionClassifierPlugin(
            penalty=hyperparameters.get("penalty", self.penalty),
            dual=hyperparameters.get("dual", self.dual),
            tol=hyperparameters.get("tol", self.tol),
            C=hyperparameters.get("C", self.C),
            fit_intercept=hyperparameters.get("fit_intercept", self.fit_intercept),
            intercept_scaling=hyperparameters.get("intercept_scaling", self.intercept_scaling),
            class_weight=hyperparameters.get("class_weight", self.class_weight),
            random_state=hyperparameters.get("random_state", self.random_state),
            solver=hyperparameters.get("solver", self.solver),
            max_iter=hyperparameters.get("max_iter", self.max_iter),
            multi_class=hyperparameters.get("multi_class", self.multi_class),
            verbose=hyperparameters.get("verbose", self.verbose),
            warm_start=hyperparameters.get("warm_start", self.warm_start),
            n_jobs=hyperparameters.get("n_jobs", self.n_jobs),
            l1_ratio=hyperparameters.get("l1_ratio", self.l1_ratio)
        )
    
    def preprocess_data(self, X, y):
        """
        Optional data preprocessing
        
        Logistic regression benefits from:
        1. Feature scaling (handled automatically)
        2. Removing constant features
        3. Handling multicollinearity
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
                return False, "Logistic Regression requires complete data (no missing values). Please handle missing values first."
            
            # Check target variable type
            target_values = df[target_column].unique()
            n_classes = len(target_values)
            
            if n_classes < 2:
                return False, "Need at least 2 classes for classification"
            
            if n_classes > 1000:
                return False, f"Too many classes ({n_classes}). Logistic regression works better with fewer classes."
            
            # Check feature types
            feature_columns = [col for col in df.columns if col != target_column]
            non_numeric_features = []
            
            for col in feature_columns:
                if not pd.api.types.is_numeric_dtype(df[col]):
                    non_numeric_features.append(col)
            
            if non_numeric_features:
                return False, f"Non-numeric features detected: {non_numeric_features}. Please encode categorical variables first."
            
            # Logistic regression specific advantages and considerations
            advantages = []
            considerations = []
            
            # Binary vs multiclass
            if n_classes == 2:
                advantages.append("Perfect for binary classification with clear interpretation")
            elif n_classes <= 10:
                advantages.append(f"Good for multiclass ({n_classes} classes) with multinomial approach")
            else:
                considerations.append(f"Many classes ({n_classes}) - may need more regularization")
            
            # Dataset size considerations
            if n_features > n_samples:
                considerations.append(f"High-dimensional case ({n_features} features > {n_samples} samples) - use strong regularization")
            elif n_samples >= 1000:
                advantages.append(f"Good sample size ({n_samples}) for stable coefficient estimates")
            
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
            
            # Class balance
            if target_column in df.columns:
                class_counts = df[target_column].value_counts()
                min_class_size = class_counts.min()
                max_class_size = class_counts.max()
                
                if max_class_size / min_class_size > 10:
                    considerations.append("Imbalanced classes detected - consider class_weight='balanced'")
                else:
                    advantages.append("Well-balanced classes for stable training")
            
            # Linearity assumption
            if n_features <= 20:
                advantages.append("Moderate feature count - good for linear assumption interpretation")
            elif n_features > 100:
                considerations.append(f"Many features ({n_features}) - may need regularization or feature selection")
            
            # Baseline model advantage
            advantages.append("Excellent baseline model with interpretable results")
            
            # Compatibility message
            message_parts = [f"✅ Compatible with {n_samples} samples, {n_features} features, {n_classes} classes"]
            
            if advantages:
                message_parts.append("🎯 Logistic Regression advantages: " + "; ".join(advantages))
            
            if considerations:
                message_parts.append("💡 Considerations: " + "; ".join(considerations))
            
            return True, " | ".join(message_parts)
            
        except Exception as e:
            return False, f"Compatibility check failed: {str(e)}"
    
    def get_model_params(self) -> Dict[str, Any]:
        """Get current model parameters"""
        return {
            'penalty': self.penalty,
            'C': self.C,
            'solver': self.solver,
            'max_iter': self.max_iter,
            'multi_class': self.multi_class,
            'class_weight': self.class_weight,
            'random_state': self.random_state,
            'tol': self.tol,
            'fit_intercept': self.fit_intercept,
            'l1_ratio': self.l1_ratio
        }
    
    def get_training_info(self) -> Dict[str, Any]:
        """Get information about the trained model"""
        if not self.is_fitted_:
            return {"status": "Not fitted"}
        
        # Basic info
        info = {
            "status": "Fitted",
            "n_features": self.n_features_in_,
            "n_classes": len(self.classes_),
            "classes": list(self.classes_),
            "feature_names": self.feature_names_,
            "converged": hasattr(self.model_, 'n_iter_') and self.model_.n_iter_ < self.max_iter,
            "n_iterations": getattr(self.model_, 'n_iter_', 'N/A')
        }
        
        # Coefficient information
        coef_info = self.get_coefficients()
        if coef_info:
            info["coefficients"] = coef_info
        
        # Feature importance
        feature_importance = self.get_feature_importance()
        if feature_importance is not None:
            # Get top 5 most important features
            top_features_idx = np.argsort(feature_importance)[-5:][::-1]
            top_features = {
                "top_features": [
                    {
                        "feature": self.feature_names_[idx],
                        "importance": float(feature_importance[idx])
                    }
                    for idx in top_features_idx
                ]
            }
            info.update(top_features)
        
        return info

    def get_algorithm_specific_metrics(self, y_true: Optional[np.ndarray] = None, y_pred: Optional[np.ndarray] = None, y_proba: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        Retrieve algorithm-specific metrics for Logistic Regression.

        Parameters:
        -----------
        y_true : np.ndarray, optional
            Actual target values. Required for metrics like McFadden's R-squared.
        y_pred : np.ndarray, optional
            Predicted target values. Not directly used for these specific metrics.
        y_proba : np.ndarray, optional
            Predicted probabilities. Required for metrics like McFadden's R-squared.

        Returns:
        --------
        Dict[str, Any]
            A dictionary containing algorithm-specific metrics.
        """
        if not self.is_fitted_ or not self.model_:
            return {"error": "Model not fitted. Cannot retrieve Logistic Regression specific metrics."}

        metrics = {}

        # Number of iterations for convergence
        if hasattr(self.model_, 'n_iter_') and self.model_.n_iter_ is not None:
            # n_iter_ is an array of shape (n_classes,) or (1,)
            metrics['logreg_convergence_iterations'] = int(np.mean(self.model_.n_iter_))

        # Coefficient-related metrics
        if hasattr(self.model_, 'coef_'):
            coefficients = self.model_.coef_
            metrics['logreg_coefficient_norm_l2'] = float(np.linalg.norm(coefficients))
            metrics['logreg_mean_abs_coefficient'] = float(np.mean(np.abs(coefficients)))
            
            if self.penalty in ['l1', 'elasticnet']:
                metrics['logreg_coefficient_sparsity'] = float(np.mean(coefficients == 0))
            
            metrics['logreg_num_features_with_non_zero_coeffs'] = int(np.sum(np.any(coefficients != 0, axis=0) if coefficients.ndim == 2 and coefficients.shape[0] > 1 else coefficients != 0))


        # McFadden's Pseudo R-squared (requires y_true and y_proba)
        if y_true is not None and y_proba is not None:
            try:
                # Ensure y_true is in the same format as classes used for y_proba indexing
                if self.label_encoder_:
                    y_true_encoded = self.label_encoder_.transform(y_true)
                else: # Assume y_true is already 0, 1, ... n_classes-1
                    y_true_encoded = y_true 

                n_samples = len(y_true_encoded)
                
                # Log-likelihood of the full model
                # Ensure probabilities are clipped to avoid log(0)
                clipped_proba = np.clip(y_proba, 1e-15, 1 - 1e-15)
                
                # For binary or multi-class, select the probability of the true class
                if clipped_proba.shape[1] == 1: # Should not happen with predict_proba for classifier
                     # This case is unlikely for LogisticRegression.predict_proba
                     # but as a safeguard if it's binary and only prob of positive class is given
                    ll_model = np.sum(y_true_encoded * np.log(clipped_proba[:, 0]) + \
                                     (1 - y_true_encoded) * np.log(1 - clipped_proba[:, 0]))
                elif clipped_proba.shape[1] == 2 and len(self.classes_) == 2: # Binary case, y_proba has shape (n_samples, 2)
                    # y_true_encoded is 0 or 1. We need prob of class 1 if y_true_encoded is 1, prob of class 0 if y_true_encoded is 0.
                    # This is equivalent to selecting p if y=1 and 1-p if y=0, then taking log.
                    # Or, more generally, sum(log(proba_for_the_true_class_of_sample_i))
                    log_likelihoods_model = np.log(clipped_proba[np.arange(n_samples), y_true_encoded])
                    ll_model = np.sum(log_likelihoods_model)
                else: # Multi-class case
                    log_likelihoods_model = np.log(clipped_proba[np.arange(n_samples), y_true_encoded])
                    ll_model = np.sum(log_likelihoods_model)

                # Log-likelihood of the null model (intercept-only)
                class_counts = np.bincount(y_true_encoded)
                class_probas_null = class_counts / n_samples
                # Ensure no log(0) for classes not present in y_true (shouldn't happen if bincount is used correctly)
                ll_null = np.sum(class_counts * np.log(np.clip(class_probas_null[class_probas_null > 0], 1e-15, 1)))
                
                if ll_null == 0: # Avoid division by zero if all samples belong to one class (perfectly predicted by null)
                    metrics['logreg_mcfaddens_pseudo_r2'] = 1.0 if ll_model == 0 else 0.0
                elif ll_model > ll_null : # ll_model should not be greater than ll_null unless something is wrong or due to precision
                    metrics['logreg_mcfaddens_pseudo_r2'] = 0.0 
                else:
                    metrics['logreg_mcfaddens_pseudo_r2'] = float(1 - (ll_model / ll_null))
            except Exception as e:
                metrics['logreg_mcfaddens_pseudo_r2_error'] = str(e)

        if not metrics:
            metrics['info'] = "No specific Logistic Regression metrics were available (e.g., y_true/y_proba not provided for Pseudo R2)."
            
        return metrics

def get_plugin():
    """Factory function to get plugin instance"""
    return LogisticRegressionClassifierPlugin()
