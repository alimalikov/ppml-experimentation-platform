import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, Optional, Tuple, List
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_X_y, check_array
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.model_selection import cross_val_score, validation_curve
from sklearn.linear_model import Ridge as SklearnRidge, RidgeCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

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


class RidgeRegressionPlugin(BaseEstimator, RegressorMixin, MLPlugin):
    """
    Ridge Regression Plugin - L2 Regularized Linear Regression
    
    Ridge Regression implements L2 regularization to prevent overfitting by adding
    a penalty term proportional to the sum of squared coefficients. This plugin
    provides comprehensive regularization analysis, automatic alpha tuning, and
    advanced statistical interpretation.
    
    Key Features:
    - L2 regularization with tunable alpha parameter
    - Automatic regularization parameter selection via cross-validation
    - Regularization path analysis and visualization
    - Bias-variance tradeoff analysis
    - Feature shrinkage analysis and interpretation
    - Cross-validation performance curves
    - Comprehensive statistical inference
    - Robust handling of multicollinearity
    - Built-in feature scaling and preprocessing
    """
    
    def __init__(
        self,
        alpha=1.0,
        fit_intercept=True,
        normalize_features=True,
        copy_X=True,
        max_iter=None,
        tol=1e-3,
        solver='auto',
        positive=False,
        
        # Advanced regularization options
        auto_alpha=False,
        alpha_range=None,
        cv_folds=5,
        alpha_scale='log',
        
        # Polynomial features
        polynomial_degree=1,
        include_bias=True,
        
        # Analysis options
        compute_regularization_path=True,
        analyze_feature_shrinkage=True,
        perform_bias_variance_analysis=False,
        estimate_effective_dof=True,
        
        # Statistical inference
        bootstrap_inference=False,
        bootstrap_samples=1000,
        confidence_level=0.95,
        
        random_state=42
    ):
        super().__init__()
        
        # Core Ridge parameters
        self.alpha = alpha
        self.fit_intercept = fit_intercept
        self.normalize_features = normalize_features
        self.copy_X = copy_X
        self.max_iter = max_iter
        self.tol = tol
        self.solver = solver
        self.positive = positive
        
        # Advanced regularization parameters
        self.auto_alpha = auto_alpha
        self.alpha_range = alpha_range if alpha_range else np.logspace(-4, 2, 50)
        self.cv_folds = cv_folds
        self.alpha_scale = alpha_scale
        
        # Polynomial features
        self.polynomial_degree = polynomial_degree
        self.include_bias = include_bias
        
        # Analysis parameters
        self.compute_regularization_path = compute_regularization_path
        self.analyze_feature_shrinkage = analyze_feature_shrinkage
        self.perform_bias_variance_analysis = perform_bias_variance_analysis
        self.estimate_effective_dof = estimate_effective_dof
        
        # Statistical inference
        self.bootstrap_inference = bootstrap_inference
        self.bootstrap_samples = bootstrap_samples
        self.confidence_level = confidence_level
        
        self.random_state = random_state
        
        # Required plugin metadata
        self._name = "Ridge Regression"
        self._description = "L2 regularized linear regression with automatic hyperparameter tuning and comprehensive regularization analysis"
        self._category = "Linear Models"
        
        # Required capability flags
        self._supports_classification = False
        self._supports_regression = True
        self._min_samples_required = 10
        
        # Internal state
        self.is_fitted_ = False
        self.model_ = None
        self.scaler_ = None
        self.poly_features_ = None
        self.feature_names_ = None
        self.n_features_in_ = None
        
        # Analysis results
        self.regularization_analysis_ = {}
        self.cross_validation_results_ = {}
        self.feature_analysis_ = {}
        self.statistical_inference_ = {}
    
    def get_name(self) -> str:
        return self._name
    
    def get_description(self) -> str:
        return self._description
    
    def get_category(self) -> str:
        return self._category
    
    def fit(self, X, y, sample_weight=None):
        """
        Fit the Ridge Regression model with comprehensive regularization analysis
        
        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            Training data
        y : array-like of shape (n_samples,)
            Target values
        sample_weight : array-like of shape (n_samples,), optional
            Individual weights for each sample
        
        Returns:
        --------
        self : object
            Returns self for method chaining
        """
        # Store feature names before validation
        self.n_features_in_ = X.shape[1]
        if hasattr(X, 'columns'):
            self.feature_names_ = list(X.columns)
        else:
            self.feature_names_ = [f"feature_{i}" for i in range(X.shape[1])]
        
        # Validate input
        X, y = check_X_y(X, y, accept_sparse=False, y_numeric=True)
        
        # Store original data for analysis
        self.X_original_ = X.copy()
        self.y_original_ = y.copy()
        
        # Apply polynomial features if requested
        X_processed = X.copy()
        if self.polynomial_degree > 1:
            self.poly_features_ = PolynomialFeatures(
                degree=self.polynomial_degree,
                include_bias=self.include_bias,
                interaction_only=False
            )
            X_processed = self.poly_features_.fit_transform(X_processed)
            
            # Update feature names for polynomial features
            if hasattr(self.poly_features_, 'get_feature_names_out'):
                poly_names = self.poly_features_.get_feature_names_out(self.feature_names_)
                self.feature_names_processed_ = list(poly_names)
            else:
                self.feature_names_processed_ = [f"poly_feature_{i}" for i in range(X_processed.shape[1])]
        else:
            self.feature_names_processed_ = self.feature_names_.copy()
        
        # Apply feature scaling if requested (recommended for Ridge)
        if self.normalize_features:
            self.scaler_ = StandardScaler()
            X_processed = self.scaler_.fit_transform(X_processed)
        
        # Store processed training data
        self.X_processed_ = X_processed
        
        # Determine optimal alpha if auto_alpha is enabled
        if self.auto_alpha:
            self.optimal_alpha_ = self._find_optimal_alpha(X_processed, y, sample_weight)
            ridge_alpha = self.optimal_alpha_
        else:
            ridge_alpha = self.alpha
            self.optimal_alpha_ = ridge_alpha
        
        # Fit the Ridge regression model
        self.model_ = SklearnRidge(
            alpha=ridge_alpha,
            fit_intercept=self.fit_intercept,
            copy_X=self.copy_X,
            max_iter=self.max_iter,
            tol=self.tol,
            solver=self.solver,
            positive=self.positive,
            random_state=self.random_state
        )
        
        self.model_.fit(X_processed, y, sample_weight=sample_weight)
        
        # Perform comprehensive regularization analysis
        self._analyze_regularization_effects()
        self._analyze_cross_validation_performance()
        self._analyze_feature_effects()
        
        # Statistical inference if requested
        if self.bootstrap_inference:
            self._perform_bootstrap_inference()
        
        self.is_fitted_ = True
        return self
    
    def predict(self, X):
        """Make predictions using the fitted Ridge model"""
        if not self.is_fitted_:
            raise ValueError("Model must be fitted before making predictions")
        
        X = check_array(X, accept_sparse=False)
        
        # Apply same preprocessing as training
        X_processed = X.copy()
        
        # Apply polynomial features
        if self.poly_features_ is not None:
            X_processed = self.poly_features_.transform(X_processed)
        
        # Apply scaling
        if self.scaler_ is not None:
            X_processed = self.scaler_.transform(X_processed)
        
        return self.model_.predict(X_processed)
    
    def _find_optimal_alpha(self, X, y, sample_weight=None):
        """Find optimal alpha using cross-validation"""
        try:
            # Use RidgeCV for efficient alpha selection
            ridge_cv = RidgeCV(
                alphas=self.alpha_range,
                fit_intercept=self.fit_intercept,
                cv=self.cv_folds,
                scoring='neg_mean_squared_error',
                store_cv_values=True
            )
            
            ridge_cv.fit(X, y, sample_weight=sample_weight)
            
            # Store cross-validation results
            self.cv_alpha_scores_ = ridge_cv.cv_values_
            self.cv_alpha_mean_scores_ = np.mean(ridge_cv.cv_values_, axis=0)
            self.cv_alpha_std_scores_ = np.std(ridge_cv.cv_values_, axis=0)
            
            return ridge_cv.alpha_
            
        except Exception as e:
            print(f"Warning: Alpha optimization failed ({e}), using default alpha={self.alpha}")
            return self.alpha
    
    def _analyze_regularization_effects(self):
        """Analyze the effects of regularization on model coefficients"""
        if not self.compute_regularization_path:
            return
        
        try:
            # Compute regularization path
            alphas = self.alpha_range
            coefs = []
            scores = []
            
            for alpha in alphas:
                ridge_temp = SklearnRidge(
                    alpha=alpha,
                    fit_intercept=self.fit_intercept,
                    copy_X=self.copy_X,
                    solver=self.solver,
                    random_state=self.random_state
                )
                ridge_temp.fit(self.X_processed_, self.y_original_)
                coefs.append(ridge_temp.coef_)
                
                # Calculate training score
                y_pred = ridge_temp.predict(self.X_processed_)
                score = r2_score(self.y_original_, y_pred)
                scores.append(score)
            
            self.regularization_analysis_ = {
                'alphas': alphas,
                'coefficients_path': np.array(coefs),
                'training_scores': np.array(scores),
                'optimal_alpha': self.optimal_alpha_,
                'optimal_alpha_index': np.argmin(np.abs(alphas - self.optimal_alpha_)),
                'feature_names': self.feature_names_processed_
            }
            
            # Calculate effective degrees of freedom
            if self.estimate_effective_dof:
                self._calculate_effective_degrees_of_freedom()
            
        except Exception as e:
            print(f"Warning: Regularization path analysis failed: {e}")
            self.regularization_analysis_ = {}
    
    def _calculate_effective_degrees_of_freedom(self):
        """Calculate effective degrees of freedom for Ridge regression"""
        try:
            # For Ridge regression: df(λ) = tr(X(X'X + λI)^(-1)X')
            X = self.X_processed_
            n_samples, n_features = X.shape
            
            # Add intercept to design matrix if needed
            if self.fit_intercept:
                X_design = np.column_stack([np.ones(n_samples), X])
            else:
                X_design = X
            
            # Calculate effective degrees of freedom
            XtX = X_design.T @ X_design
            alpha_I = self.optimal_alpha_ * np.eye(X_design.shape[1])
            
            try:
                # df = tr(X(X'X + λI)^(-1)X')
                inv_term = np.linalg.inv(XtX + alpha_I)
                hat_matrix_trace = np.trace(X_design @ inv_term @ X_design.T)
                
                self.regularization_analysis_['effective_dof'] = hat_matrix_trace
                self.regularization_analysis_['max_dof'] = X_design.shape[1]
                self.regularization_analysis_['dof_reduction'] = X_design.shape[1] - hat_matrix_trace
                
            except np.linalg.LinAlgError:
                self.regularization_analysis_['effective_dof'] = np.nan
                
        except Exception as e:
            print(f"Warning: Effective DOF calculation failed: {e}")
    
    def _analyze_cross_validation_performance(self):
        """Perform comprehensive cross-validation analysis"""
        try:
            # Cross-validation scores for the optimal model
            cv_scores = cross_val_score(
                self.model_, 
                self.X_processed_, 
                self.y_original_,
                cv=self.cv_folds,
                scoring='r2',
                n_jobs=-1
            )
            
            cv_mse_scores = -cross_val_score(
                self.model_,
                self.X_processed_,
                self.y_original_,
                cv=self.cv_folds,
                scoring='neg_mean_squared_error',
                n_jobs=-1
            )
            
            self.cross_validation_results_ = {
                'r2_scores': cv_scores,
                'r2_mean': np.mean(cv_scores),
                'r2_std': np.std(cv_scores),
                'mse_scores': cv_mse_scores,
                'mse_mean': np.mean(cv_mse_scores),
                'mse_std': np.std(cv_mse_scores),
                'cv_folds': self.cv_folds
            }
            
            # Validation curve for alpha if auto_alpha was used
            if self.auto_alpha and hasattr(self, 'cv_alpha_scores_'):
                self.cross_validation_results_['alpha_validation'] = {
                    'alphas': self.alpha_range,
                    'mean_scores': self.cv_alpha_mean_scores_,
                    'std_scores': self.cv_alpha_std_scores_,
                    'optimal_alpha': self.optimal_alpha_
                }
                
        except Exception as e:
            print(f"Warning: Cross-validation analysis failed: {e}")
            self.cross_validation_results_ = {}
    
    def _analyze_feature_effects(self):
        """Analyze the effects of regularization on individual features"""
        if not self.analyze_feature_shrinkage:
            return
        
        try:
            # Get Ridge coefficients
            ridge_coefs = self.model_.coef_
            
            # Compare with unregularized linear regression if possible
            try:
                from sklearn.linear_model import LinearRegression
                linear_model = LinearRegression(fit_intercept=self.fit_intercept)
                linear_model.fit(self.X_processed_, self.y_original_)
                linear_coefs = linear_model.coef_
                
                # Calculate shrinkage
                shrinkage = np.abs(ridge_coefs) / (np.abs(linear_coefs) + 1e-8)  # Avoid division by zero
                shrinkage_percent = (1 - shrinkage) * 100
                
            except Exception:
                linear_coefs = None
                shrinkage = None
                shrinkage_percent = None
            
            # Feature importance based on absolute coefficients
            feature_importance = np.abs(ridge_coefs)
            feature_importance_normalized = feature_importance / np.sum(feature_importance) if np.sum(feature_importance) > 0 else feature_importance
            
            # Create feature analysis
            feature_analysis = {}
            for i, (name, ridge_coef, importance, norm_importance) in enumerate(
                zip(self.feature_names_processed_, ridge_coefs, feature_importance, feature_importance_normalized)
            ):
                analysis = {
                    'ridge_coefficient': ridge_coef,
                    'absolute_coefficient': importance,
                    'normalized_importance': norm_importance,
                    'rank': i + 1
                }
                
                if linear_coefs is not None:
                    analysis['linear_coefficient'] = linear_coefs[i]
                    analysis['shrinkage_factor'] = shrinkage[i] if shrinkage is not None else np.nan
                    analysis['shrinkage_percent'] = shrinkage_percent[i] if shrinkage_percent is not None else np.nan
                
                feature_analysis[name] = analysis
            
            # Sort by importance
            sorted_features = sorted(feature_analysis.items(), key=lambda x: x[1]['absolute_coefficient'], reverse=True)
            
            # Update ranks
            for rank, (name, info) in enumerate(sorted_features):
                feature_analysis[name]['rank'] = rank + 1
            
            self.feature_analysis_ = {
                'feature_analysis': feature_analysis,
                'sorted_features': [name for name, _ in sorted_features],
                'ridge_coefficients': ridge_coefs,
                'linear_coefficients': linear_coefs,
                'feature_names': self.feature_names_processed_,
                'regularization_strength': self.optimal_alpha_
            }
            
        except Exception as e:
            print(f"Warning: Feature analysis failed: {e}")
            self.feature_analysis_ = {}
    
    def _perform_bootstrap_inference(self):
        """Perform bootstrap inference for coefficient uncertainty"""
        if not self.bootstrap_inference:
            return
        
        try:
            n_samples = len(self.y_original_)
            bootstrap_coefs = []
            bootstrap_intercepts = []
            
            np.random.seed(self.random_state)
            
            for _ in range(self.bootstrap_samples):
                # Bootstrap sample
                indices = np.random.choice(n_samples, n_samples, replace=True)
                X_boot = self.X_processed_[indices]
                y_boot = self.y_original_[indices]
                
                # Fit Ridge model
                ridge_boot = SklearnRidge(
                    alpha=self.optimal_alpha_,
                    fit_intercept=self.fit_intercept,
                    random_state=self.random_state
                )
                ridge_boot.fit(X_boot, y_boot)
                
                bootstrap_coefs.append(ridge_boot.coef_)
                if self.fit_intercept:
                    bootstrap_intercepts.append(ridge_boot.intercept_)
            
            bootstrap_coefs = np.array(bootstrap_coefs)
            bootstrap_intercepts = np.array(bootstrap_intercepts) if bootstrap_intercepts else None
            
            # Calculate confidence intervals
            alpha_level = 1 - self.confidence_level
            lower_percentile = (alpha_level / 2) * 100
            upper_percentile = (1 - alpha_level / 2) * 100
            
            coef_ci_lower = np.percentile(bootstrap_coefs, lower_percentile, axis=0)
            coef_ci_upper = np.percentile(bootstrap_coefs, upper_percentile, axis=0)
            coef_std = np.std(bootstrap_coefs, axis=0)
            
            self.statistical_inference_ = {
                'bootstrap_coefficients': bootstrap_coefs,
                'coefficient_std': coef_std,
                'coefficient_ci_lower': coef_ci_lower,
                'coefficient_ci_upper': coef_ci_upper,
                'confidence_level': self.confidence_level,
                'bootstrap_samples': self.bootstrap_samples
            }
            
            if bootstrap_intercepts is not None:
                intercept_ci_lower = np.percentile(bootstrap_intercepts, lower_percentile)
                intercept_ci_upper = np.percentile(bootstrap_intercepts, upper_percentile)
                intercept_std = np.std(bootstrap_intercepts)
                
                self.statistical_inference_.update({
                    'bootstrap_intercepts': bootstrap_intercepts,
                    'intercept_std': intercept_std,
                    'intercept_ci_lower': intercept_ci_lower,
                    'intercept_ci_upper': intercept_ci_upper
                })
                
        except Exception as e:
            print(f"Warning: Bootstrap inference failed: {e}")
            self.statistical_inference_ = {}
    
    def get_hyperparameter_config(self, key_prefix: str) -> Dict[str, Any]:
        """Generate Streamlit UI for hyperparameters"""
        
        # Create tabs for different configuration aspects
        tab1, tab2, tab3, tab4 = st.tabs(["Core Parameters", "Regularization", "Advanced Options", "Algorithm Info"])
        
        with tab1:
            st.markdown("**Ridge Regression Configuration**")
            
            # Alpha parameter (most important)
            auto_alpha = st.checkbox(
                "Auto-tune Alpha (Recommended)",
                value=self.auto_alpha,
                help="Automatically find optimal regularization strength using cross-validation",
                key=f"{key_prefix}_auto_alpha"
            )
            
            if not auto_alpha:
                alpha = st.number_input(
                    "Alpha (Regularization Strength):",
                    value=float(self.alpha),
                    min_value=1e-6,
                    max_value=1000.0,
                    step=0.1,
                    format="%.6f",
                    help="Higher values = more regularization. Try 0.1, 1.0, 10.0",
                    key=f"{key_prefix}_alpha"
                )
                st.info(f"🎯 Current alpha: {alpha:.6f}")
            else:
                alpha = self.alpha
                st.info("🤖 Alpha will be automatically optimized using cross-validation")
            
            # Core regression parameters
            fit_intercept = st.checkbox(
                "Fit Intercept",
                value=self.fit_intercept,
                help="Whether to calculate the intercept for this model",
                key=f"{key_prefix}_fit_intercept"
            )
            
            normalize_features = st.checkbox(
                "Normalize Features (Highly Recommended)",
                value=self.normalize_features,
                help="Ridge regression is sensitive to feature scales - normalization is crucial",
                key=f"{key_prefix}_normalize_features"
            )
            
            if not normalize_features:
                st.warning("⚠️ Ridge regression performance depends heavily on feature scaling!")
            
            # Polynomial features
            polynomial_degree = st.selectbox(
                "Polynomial Degree:",
                options=[1, 2, 3, 4],
                index=[1, 2, 3, 4].index(self.polynomial_degree),
                help="Degree of polynomial features (1=linear, 2=quadratic, etc.)",
                key=f"{key_prefix}_polynomial_degree"
            )
            
            if polynomial_degree > 1:
                st.info(f"🔄 Using polynomial features of degree {polynomial_degree}")
                include_bias = st.checkbox(
                    "Include Bias Column",
                    value=self.include_bias,
                    help="Include bias column in polynomial features",
                    key=f"{key_prefix}_include_bias"
                )
            else:
                include_bias = self.include_bias
                st.info("📈 Using standard Ridge regression")
        
        with tab2:
            st.markdown("**Regularization Configuration**")
            
            if auto_alpha:
                cv_folds = st.selectbox(
                    "Cross-Validation Folds:",
                    options=[3, 5, 10],
                    index=[3, 5, 10].index(self.cv_folds),
                    help="Number of folds for alpha optimization",
                    key=f"{key_prefix}_cv_folds"
                )
                
                # Alpha range configuration
                st.markdown("**Alpha Search Range:**")
                col1, col2 = st.columns(2)
                
                with col1:
                    alpha_min = st.number_input(
                        "Min Alpha:",
                        value=1e-4,
                        min_value=1e-8,
                        max_value=1e-1,
                        format="%.2e",
                        help="Minimum alpha to test",
                        key=f"{key_prefix}_alpha_min"
                    )
                
                with col2:
                    alpha_max = st.number_input(
                        "Max Alpha:",
                        value=100.0,
                        min_value=1.0,
                        max_value=10000.0,
                        help="Maximum alpha to test",
                        key=f"{key_prefix}_alpha_max"
                    )
                
                alpha_points = st.slider(
                    "Number of Alpha Values:",
                    min_value=10,
                    max_value=100,
                    value=50,
                    help="Number of alpha values to test",
                    key=f"{key_prefix}_alpha_points"
                )
                
                alpha_range = np.logspace(np.log10(alpha_min), np.log10(alpha_max), alpha_points)
                st.info(f"📊 Testing {alpha_points} alpha values from {alpha_min:.2e} to {alpha_max:.2e}")
            else:
                cv_folds = self.cv_folds
                alpha_range = self.alpha_range
            
            # Regularization analysis options
            compute_regularization_path = st.checkbox(
                "Compute Regularization Path",
                value=self.compute_regularization_path,
                help="Analyze how coefficients change with regularization strength",
                key=f"{key_prefix}_regularization_path"
            )
            
            analyze_feature_shrinkage = st.checkbox(
                "Analyze Feature Shrinkage",
                value=self.analyze_feature_shrinkage,
                help="Compare Ridge coefficients with unregularized linear regression",
                key=f"{key_prefix}_feature_shrinkage"
            )
            
            estimate_effective_dof = st.checkbox(
                "Estimate Effective Degrees of Freedom",
                value=self.estimate_effective_dof,
                help="Calculate model complexity accounting for regularization",
                key=f"{key_prefix}_effective_dof"
            )
        
        with tab3:
            st.markdown("**Advanced Configuration**")
            
            # Solver options
            solver = st.selectbox(
                "Solver:",
                options=['auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag', 'saga'],
                index=['auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag', 'saga'].index(self.solver),
                help="Algorithm to use for optimization",
                key=f"{key_prefix}_solver"
            )
            
            if solver != 'auto':
                st.info(f"Using {solver} solver")
            
            # Technical parameters
            max_iter = st.number_input(
                "Max Iterations:",
                value=self.max_iter if self.max_iter else 1000,
                min_value=100,
                max_value=10000,
                step=100,
                help="Maximum number of iterations for iterative solvers",
                key=f"{key_prefix}_max_iter"
            )
            
            tol = st.number_input(
                "Tolerance:",
                value=float(self.tol),
                min_value=1e-6,
                max_value=1e-2,
                format="%.2e",
                help="Precision of the solution",
                key=f"{key_prefix}_tol"
            )
            
            positive = st.checkbox(
                "Positive Coefficients Only",
                value=self.positive,
                help="Constrain coefficients to be non-negative",
                key=f"{key_prefix}_positive"
            )
            
            # Statistical inference options
            bootstrap_inference = st.checkbox(
                "Bootstrap Confidence Intervals",
                value=self.bootstrap_inference,
                help="Estimate coefficient uncertainty using bootstrap sampling",
                key=f"{key_prefix}_bootstrap_inference"
            )
            
            if bootstrap_inference:
                bootstrap_samples = st.number_input(
                    "Bootstrap Samples:",
                    value=self.bootstrap_samples,
                    min_value=100,
                    max_value=5000,
                    step=100,
                    help="Number of bootstrap samples for inference",
                    key=f"{key_prefix}_bootstrap_samples"
                )
                
                confidence_level = st.slider(
                    "Confidence Level:",
                    min_value=0.80,
                    max_value=0.99,
                    value=self.confidence_level,
                    step=0.01,
                    help="Confidence level for bootstrap intervals",
                    key=f"{key_prefix}_confidence_level"
                )
            else:
                bootstrap_samples = self.bootstrap_samples
                confidence_level = self.confidence_level
            
            # Technical parameters
            copy_X = st.checkbox(
                "Copy Input Data",
                value=self.copy_X,
                help="Create copy of input data (safer but uses more memory)",
                key=f"{key_prefix}_copy_X"
            )
            
            random_state = st.number_input(
                "Random Seed:",
                value=int(self.random_state),
                min_value=0,
                max_value=1000,
                help="For reproducible results",
                key=f"{key_prefix}_random_state"
            )
        
        with tab4:
            st.markdown("**Algorithm Information**")
            
            st.info("""
            **Ridge Regression** - L2 Regularized Linear Regression:
            • 🎯 Adds L2 penalty: λ∑βᵢ² to the loss function
            • 📉 Shrinks coefficients towards zero (but not exactly zero)
            • 🛡️ Prevents overfitting in high-dimensional data
            • 🔧 Handles multicollinearity by coefficient shrinkage
            • 📊 Maintains all features (no feature selection)
            • ⚖️ Bias-variance tradeoff controlled by alpha parameter
            
            **Mathematical Foundation:**
            • Objective: min ||y - Xβ||² + α||β||²
            • Solution: β̂ = (X'X + αI)⁻¹X'y
            • Regularization strength: α (alpha parameter)
            """)
            
            # When to use Ridge regression
            if st.button("🎯 When to Use Ridge Regression", key=f"{key_prefix}_when_to_use"):
                st.markdown("""
                **Ideal Use Cases:**
                
                **Problem Characteristics:**
                • High-dimensional data (many features)
                • Multicollinearity between features
                • Overfitting in linear regression
                • Need all features retained (no feature selection)
                
                **Data Characteristics:**
                • More features than samples (p > n)
                • Highly correlated predictors
                • Noisy measurements
                • Continuous target variable
                
                **Advantages over Linear Regression:**
                • Better generalization performance
                • Numerical stability with correlated features
                • Reduced variance in coefficient estimates
                • Prevents overfitting automatically
                
                **Examples:**
                • Gene expression analysis (high-dimensional)
                • Financial modeling with correlated indicators
                • Image regression with pixel features
                • Econometric models with multicollinearity
                """)
            
            # Ridge vs other methods
            if st.button("⚖️ Ridge vs Other Methods", key=f"{key_prefix}_comparisons"):
                st.markdown("""
                **Ridge vs Linear Regression:**
                ✅ Better generalization (lower test error)
                ✅ Handles multicollinearity
                ✅ More stable coefficients
                ❌ Biased coefficient estimates
                ❌ Requires hyperparameter tuning
                
                **Ridge vs Lasso:**
                ✅ Keeps all features (no automatic selection)
                ✅ Better when all features are relevant
                ✅ More stable with grouped correlated features
                ❌ No automatic feature selection
                ❌ Less sparse solutions
                
                **Ridge vs Elastic Net:**
                ✅ Simpler (only one hyperparameter)
                ✅ More interpretable regularization
                ❌ No L1 penalty (no sparsity)
                ❌ Less flexible regularization
                """)
            
            # Best practices
            if st.button("🎯 Ridge Regression Best Practices", key=f"{key_prefix}_best_practices"):
                st.markdown("""
                **Ridge Regression Best Practices:**
                
                **Preprocessing:**
                1. **Always standardize features** - Ridge is scale-sensitive
                2. Check for multicollinearity (Ridge helps but awareness is key)
                3. Handle missing values appropriately
                4. Consider polynomial features for non-linear relationships
                
                **Hyperparameter Tuning:**
                1. Use cross-validation for alpha selection
                2. Try wide range: 10⁻⁴ to 10² or broader
                3. Use log scale for alpha search
                4. Consider nested CV for unbiased performance estimation
                
                **Model Interpretation:**
                1. Examine regularization path to understand feature importance
                2. Compare with unregularized coefficients
                3. Calculate effective degrees of freedom
                4. Use bootstrap for coefficient uncertainty
                
                **Validation:**
                1. Plot validation curves to check for over/under-regularization
                2. Examine residual plots for assumption validation
                3. Compare with simpler/more complex models
                4. Check generalization on holdout test set
                """)
            
            # Alpha selection guide
            if st.button("🔧 Alpha Selection Guide", key=f"{key_prefix}_alpha_guide"):
                st.markdown("""
                **Understanding Alpha (Regularization Strength):**
                
                **Alpha = 0:**
                • Equivalent to ordinary linear regression
                • No regularization, potential overfitting
                • Maximum likelihood estimates
                
                **Small Alpha (0.001 - 0.1):**
                • Light regularization
                • Coefficients close to unregularized values
                • Good when you have many samples vs features
                
                **Medium Alpha (0.1 - 10):**
                • Moderate regularization
                • Balanced bias-variance tradeoff
                • Good general-purpose starting range
                
                **Large Alpha (10 - 1000+):**
                • Strong regularization
                • Heavy coefficient shrinkage
                • Good for high-dimensional data with few samples
                
                **Alpha Selection Strategy:**
                1. Start with auto-tuning using cross-validation
                2. Examine validation curve for optimal region
                3. Consider domain knowledge about feature importance
                4. Balance training and validation performance
                """)
        
        return {
            "alpha": alpha,
            "auto_alpha": auto_alpha,
            "alpha_range": alpha_range if auto_alpha else self.alpha_range,
            "cv_folds": cv_folds,
            "fit_intercept": fit_intercept,
            "normalize_features": normalize_features,
            "polynomial_degree": polynomial_degree,
            "include_bias": include_bias,
            "solver": solver,
            "max_iter": max_iter if max_iter else None,
            "tol": tol,
            "positive": positive,
            "copy_X": copy_X,
            "compute_regularization_path": compute_regularization_path,
            "analyze_feature_shrinkage": analyze_feature_shrinkage,
            "estimate_effective_dof": estimate_effective_dof,
            "bootstrap_inference": bootstrap_inference,
            "bootstrap_samples": bootstrap_samples,
            "confidence_level": confidence_level,
            "random_state": random_state
        }
    
    def create_model_instance(self, hyperparameters: Dict[str, Any]):
        """Create model instance with given hyperparameters"""
        return RidgeRegressionPlugin(
            alpha=hyperparameters.get("alpha", self.alpha),
            auto_alpha=hyperparameters.get("auto_alpha", self.auto_alpha),
            alpha_range=hyperparameters.get("alpha_range", self.alpha_range),
            cv_folds=hyperparameters.get("cv_folds", self.cv_folds),
            fit_intercept=hyperparameters.get("fit_intercept", self.fit_intercept),
            normalize_features=hyperparameters.get("normalize_features", self.normalize_features),
            polynomial_degree=hyperparameters.get("polynomial_degree", self.polynomial_degree),
            include_bias=hyperparameters.get("include_bias", self.include_bias),
            solver=hyperparameters.get("solver", self.solver),
            max_iter=hyperparameters.get("max_iter", self.max_iter),
            tol=hyperparameters.get("tol", self.tol),
            positive=hyperparameters.get("positive", self.positive),
            copy_X=hyperparameters.get("copy_X", self.copy_X),
            compute_regularization_path=hyperparameters.get("compute_regularization_path", self.compute_regularization_path),
            analyze_feature_shrinkage=hyperparameters.get("analyze_feature_shrinkage", self.analyze_feature_shrinkage),
            estimate_effective_dof=hyperparameters.get("estimate_effective_dof", self.estimate_effective_dof),
            bootstrap_inference=hyperparameters.get("bootstrap_inference", self.bootstrap_inference),
            bootstrap_samples=hyperparameters.get("bootstrap_samples", self.bootstrap_samples),
            confidence_level=hyperparameters.get("confidence_level", self.confidence_level),
            random_state=hyperparameters.get("random_state", self.random_state)
        )
    
    def preprocess_data(self, X, y=None, training=True):
        """Preprocess data for Ridge Regression"""
        if hasattr(X, 'copy'):
            X_processed = X.copy()
        else:
            X_processed = np.array(X, copy=True)
        
        if training and y is not None:
            if hasattr(y, 'copy'):
                y_processed = y.copy()
            else:
                y_processed = np.array(y, copy=True)
            return X_processed, y_processed
        
        return X_processed
    
    def is_compatible_with_data(self, X, y=None) -> Tuple[bool, str]:
        """Check if Ridge Regression is compatible with the given data"""
        # Check minimum samples
        if X.shape[0] < self._min_samples_required:
            return False, f"Ridge regression requires at least {self._min_samples_required} samples, got {X.shape[0]}"
        
        # Check for regression targets
        if y is not None:
            if not np.issubdtype(y.dtype, np.number):
                return False, "Ridge regression requires continuous numerical target values"
            
            # Check for sufficient variance in target
            if np.var(y) == 0:
                return False, "Target variable has zero variance (all values are the same)"
            
            # Sample size considerations
            n_samples, n_features = X.shape
            effective_features = n_features * self.polynomial_degree if self.polynomial_degree > 1 else n_features
            
            advantages = []
            considerations = []
            
            # Sample size assessment
            if n_samples >= effective_features * 2:
                advantages.append(f"Good sample size ({n_samples} samples for {effective_features} features)")
            elif n_samples >= effective_features:
                advantages.append(f"Adequate sample size ({n_samples} samples for {effective_features} features)")
            else:
                advantages.append(f"High-dimensional regime ({n_samples} samples, {effective_features} features) - Ridge is ideal!")
            
            # Feature dimensionality - Ridge excels with many features
            if n_features >= n_samples:
                advantages.append("High-dimensional data - Ridge regression is particularly well-suited")
            elif n_features > 50:
                advantages.append("Medium-dimensional data - Ridge helps prevent overfitting")
            else:
                considerations.append("Low-dimensional data - consider if regularization is needed")
            
            # Check for potential multicollinearity
            try:
                correlation_matrix = np.corrcoef(X.T)
                max_correlation = 0
                for i in range(len(correlation_matrix)):
                    for j in range(i+1, len(correlation_matrix)):
                        abs_corr = abs(correlation_matrix[i, j])
                        if abs_corr > max_correlation:
                            max_correlation = abs_corr
                
                if max_correlation > 0.8:
                    advantages.append("High multicollinearity detected - Ridge will help stabilize coefficients")
                elif max_correlation > 0.5:
                    advantages.append("Moderate multicollinearity - Ridge provides coefficient stability")
                
            except Exception:
                pass
            
            # Target distribution
            target_skew = abs(stats.skew(y))
            if target_skew < 0.5:
                advantages.append("Target distribution is approximately normal")
            elif target_skew < 1.0:
                considerations.append("Target distribution is slightly skewed")
            else:
                considerations.append("Target distribution is heavily skewed - consider transformation")
            
            # Build compatibility message
            suitability = ("Excellent" if len(considerations) == 0 else "Very Good" if len(considerations) <= 1 else 
                          "Good" if len(considerations) <= 2 else "Fair")
            
            message_parts = [
                f"✅ Compatible with {n_samples} samples, {n_features} features",
                f"🎯 Suitability: {suitability}"
            ]
            
            if advantages:
                message_parts.append("🎯 Advantages: " + "; ".join(advantages))
            
            if considerations:
                message_parts.append("💡 Considerations: " + "; ".join(considerations))
            
            return True, " | ".join(message_parts)
        
        return True, f"Compatible with {X.shape[0]} samples and {X.shape[1]} features"
    
    def get_feature_importance(self) -> Optional[Dict[str, Any]]:
        """Get feature importance based on regularized coefficient magnitudes"""
        if not self.is_fitted_:
            return None
        
        # Get Ridge coefficients
        coefficients = self.model_.coef_
        feature_names = self.feature_names_processed_
        
        # Calculate importance as absolute coefficient values
        importance = np.abs(coefficients)
        
        # Normalize to sum to 1
        if np.sum(importance) > 0:
            importance_normalized = importance / np.sum(importance)
        else:
            importance_normalized = importance
        
        # Create feature importance dictionary with regularization info
        feature_importance = {}
        for i, (name, coef, imp, norm_imp) in enumerate(zip(feature_names, coefficients, importance, importance_normalized)):
            feature_importance[name] = {
                'ridge_coefficient': coef,
                'absolute_coefficient': imp,
                'normalized_importance': norm_imp,
                'rank': i + 1
            }
            
            # Add shrinkage information if available
            if hasattr(self, 'feature_analysis_') and self.feature_analysis_:
                feature_info = self.feature_analysis_.get('feature_analysis', {}).get(name, {})
                if 'shrinkage_percent' in feature_info:
                    feature_importance[name]['shrinkage_percent'] = feature_info['shrinkage_percent']
                if 'linear_coefficient' in feature_info:
                    feature_importance[name]['linear_coefficient'] = feature_info['linear_coefficient']
        
        # Sort by importance
        sorted_features = sorted(feature_importance.items(), key=lambda x: x[1]['absolute_coefficient'], reverse=True)
        
        # Update ranks
        for rank, (name, info) in enumerate(sorted_features):
            feature_importance[name]['rank'] = rank + 1
        
        return {
            'feature_importance': feature_importance,
            'sorted_features': [name for name, _ in sorted_features],
            'sorted_importance': [info['normalized_importance'] for _, info in sorted_features],
            'ridge_coefficients': coefficients,
            'feature_names': feature_names,
            'intercept': self.model_.intercept_ if self.fit_intercept else 0,
            'regularization_strength': self.optimal_alpha_,
            'interpretation': 'Feature importance based on L2-regularized coefficient magnitude'
        }
    
    def get_model_params(self) -> Dict[str, Any]:
        """Get current model parameters"""
        if not self.is_fitted_:
            return {}
        
        params = {
            "algorithm": "Ridge Regression",
            "regularization_type": "L2 (Ridge)",
            "n_features": self.n_features_in_,
            "feature_names": self.feature_names_,
            "processed_features": len(self.feature_names_processed_),
            "processed_feature_names": self.feature_names_processed_,
            "polynomial_degree": self.polynomial_degree,
            "fit_intercept": self.fit_intercept,
            "normalize_features": self.normalize_features,
            "ridge_coefficients": self.model_.coef_.tolist(),
            "intercept": self.model_.intercept_ if self.fit_intercept else 0,
            "alpha": self.optimal_alpha_,
            "solver": self.solver,
            "feature_scaling": self.scaler_ is not None
        }
        
        # Add effective degrees of freedom if calculated
        if self.regularization_analysis_ and 'effective_dof' in self.regularization_analysis_:
            params["effective_degrees_of_freedom"] = self.regularization_analysis_['effective_dof']
            params["max_degrees_of_freedom"] = self.regularization_analysis_['max_dof']
            params["dof_reduction"] = self.regularization_analysis_['dof_reduction']
        
        return params
    
    def get_regularization_analysis(self) -> Dict[str, Any]:
        """Get comprehensive regularization analysis results"""
        if not self.is_fitted_:
            return {"status": "Model not fitted"}
        
        return {
            "regularization_analysis": self.regularization_analysis_,
            "cross_validation_results": self.cross_validation_results_,
            "feature_analysis": self.feature_analysis_,
            "statistical_inference": self.statistical_inference_
        }
    
    def plot_regularization_path(self, max_features=15, figsize=(12, 8)):
        """Plot regularization path showing how coefficients change with alpha"""
        if not self.regularization_analysis_ or 'coefficients_path' not in self.regularization_analysis_:
            raise ValueError("Regularization path not computed. Set compute_regularization_path=True")
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=figsize)
        
        alphas = self.regularization_analysis_['alphas']
        coefs_path = self.regularization_analysis_['coefficients_path']
        feature_names = self.regularization_analysis_['feature_names']
        optimal_alpha = self.regularization_analysis_['optimal_alpha']
        
        # 1. Regularization path
        for i in range(min(max_features, coefs_path.shape[1])):
            ax1.plot(alphas, coefs_path[:, i], label=feature_names[i] if i < len(feature_names) else f'Feature {i}')
        
        ax1.axvline(optimal_alpha, color='red', linestyle='--', alpha=0.7, label=f'Optimal α={optimal_alpha:.2e}')
        ax1.set_xscale('log')
        ax1.set_xlabel('Alpha (Regularization Strength)')
        ax1.set_ylabel('Coefficient Value')
        ax1.set_title('Regularization Path')
        ax1.grid(True, alpha=0.3)
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # 2. Coefficient magnitude vs alpha
        coef_magnitudes = np.sqrt(np.sum(coefs_path**2, axis=1))
        ax2.plot(alphas, coef_magnitudes, 'b-', linewidth=2)
        ax2.axvline(optimal_alpha, color='red', linestyle='--', alpha=0.7)
        ax2.set_xscale('log')
        ax2.set_xlabel('Alpha (Regularization Strength)')
        ax2.set_ylabel('||β||₂ (L2 Norm of Coefficients)')
        ax2.set_title('Coefficient Shrinkage')
        ax2.grid(True, alpha=0.3)
        
        # 3. Training score vs alpha
        if 'training_scores' in self.regularization_analysis_:
            training_scores = self.regularization_analysis_['training_scores']
            ax3.plot(alphas, training_scores, 'g-', linewidth=2, label='Training R²')
            ax3.axvline(optimal_alpha, color='red', linestyle='--', alpha=0.7)
            ax3.set_xscale('log')
            ax3.set_xlabel('Alpha (Regularization Strength)')
            ax3.set_ylabel('R² Score')
            ax3.set_title('Training Performance vs Regularization')
            ax3.grid(True, alpha=0.3)
            ax3.legend()
        
        # 4. Cross-validation scores if available
        if self.cross_validation_results_ and 'alpha_validation' in self.cross_validation_results_:
            cv_results = self.cross_validation_results_['alpha_validation']
            cv_alphas = cv_results['alphas']
            cv_mean = cv_results['mean_scores']
            cv_std = cv_results['std_scores']
            
            ax4.errorbar(cv_alphas, cv_mean, yerr=cv_std, capsize=3, capthick=1)
            ax4.axvline(optimal_alpha, color='red', linestyle='--', alpha=0.7)
            ax4.set_xscale('log')
            ax4.set_xlabel('Alpha (Regularization Strength)')
            ax4.set_ylabel('Cross-Validation Score')
            ax4.set_title('Cross-Validation Performance')
            ax4.grid(True, alpha=0.3)
        else:
            ax4.text(0.5, 0.5, 'Cross-validation\nresults not available', 
                    ha='center', va='center', transform=ax4.transAxes)
            ax4.set_title('Cross-Validation Performance')
        
        plt.tight_layout()
        return fig
    
    def plot_feature_importance(self, max_features=15, figsize=(12, 6)):
        """Plot feature importance with shrinkage analysis"""
        importance_data = self.get_feature_importance()
        if not importance_data:
            raise ValueError("Model must be fitted to plot feature importance")
        
        # Get top features
        sorted_features = importance_data['sorted_features'][:max_features]
        ridge_coefs = [importance_data['feature_importance'][feat]['ridge_coefficient'] 
                      for feat in sorted_features]
        
        # Create plots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        
        # Plot 1: Ridge coefficients with sign
        colors = ['green' if coef > 0 else 'red' for coef in ridge_coefs]
        y_pos = np.arange(len(sorted_features))
        
        ax1.barh(y_pos, ridge_coefs, color=colors, alpha=0.7)
        ax1.set_yticks(y_pos)
        ax1.set_yticklabels(sorted_features)
        ax1.set_xlabel('Ridge Coefficient Value')
        ax1.set_title(f'Ridge Coefficients (α={importance_data["regularization_strength"]:.2e})')
        ax1.axvline(x=0, color='black', linestyle='-', alpha=0.3)
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Shrinkage comparison if available
        if 'feature_analysis' in self.feature_analysis_ and self.feature_analysis_['feature_analysis']:
            linear_coefs = []
            shrinkage_percents = []
            
            for feat in sorted_features:
                feat_info = self.feature_analysis_['feature_analysis'].get(feat, {})
                linear_coef = feat_info.get('linear_coefficient', np.nan)
                shrinkage = feat_info.get('shrinkage_percent', 0)
                linear_coefs.append(linear_coef)
                shrinkage_percents.append(shrinkage)
            
            if not all(np.isnan(linear_coefs)):
                # Shrinkage visualization
                ax2.barh(y_pos, shrinkage_percents, color='orange', alpha=0.7)
                ax2.set_yticks(y_pos)
                ax2.set_yticklabels(sorted_features)
                ax2.set_xlabel('Shrinkage Percentage (%)')
                ax2.set_title('Feature Shrinkage from Linear Regression')
                ax2.grid(True, alpha=0.3)
            else:
                # Absolute importance
                abs_importance = [importance_data['feature_importance'][feat]['normalized_importance'] 
                                for feat in sorted_features]
                ax2.barh(y_pos, abs_importance, color='steelblue', alpha=0.7)
                ax2.set_yticks(y_pos)
                ax2.set_yticklabels(sorted_features)
                ax2.set_xlabel('Normalized Importance')
                ax2.set_title('Feature Importance (Normalized)')
                ax2.grid(True, alpha=0.3)
        else:
            # Absolute importance fallback
            abs_importance = [importance_data['feature_importance'][feat]['normalized_importance'] 
                            for feat in sorted_features]
            ax2.barh(y_pos, abs_importance, color='steelblue', alpha=0.7)
            ax2.set_yticks(y_pos)
            ax2.set_yticklabels(sorted_features)
            ax2.set_xlabel('Normalized Importance')
            ax2.set_title('Feature Importance (Normalized)')
            ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def get_training_info(self) -> Dict[str, Any]:
        """Get information about the training process"""
        if not self.is_fitted_:
            return {"status": "Not fitted"}
        
        info = {
            "algorithm": "Ridge Regression",
            "type": "L2 Regularized Linear Regression",
            "training_completed": True,
            "regularization_characteristics": {
                "regularization_type": "L2 (Ridge)",
                "penalty_term": "α||β||²",
                "shrinks_coefficients": True,
                "performs_feature_selection": False,
                "handles_multicollinearity": True,
                "prevents_overfitting": True
            },
            "model_configuration": {
                "alpha": self.optimal_alpha_,
                "auto_alpha": self.auto_alpha,
                "fit_intercept": self.fit_intercept,
                "normalize_features": self.normalize_features,
                "polynomial_degree": self.polynomial_degree,
                "solver": self.solver,
                "n_original_features": self.n_features_in_,
                "n_processed_features": len(self.feature_names_processed_),
                "feature_scaling": self.scaler_ is not None
            },
            "analysis_performed": {
                "regularization_path": bool(self.regularization_analysis_),
                "cross_validation": bool(self.cross_validation_results_),
                "feature_shrinkage": bool(self.feature_analysis_),
                "bootstrap_inference": bool(self.statistical_inference_)
            }
        }
        
        # Add regularization analysis results
        if self.regularization_analysis_:
            info["regularization_effects"] = {
                "effective_dof": self.regularization_analysis_.get('effective_dof'),
                "max_dof": self.regularization_analysis_.get('max_dof'),
                "dof_reduction": self.regularization_analysis_.get('dof_reduction')
            }
        
        # Add cross-validation results
        if self.cross_validation_results_:
            info["cross_validation_performance"] = {
                "cv_r2_mean": self.cross_validation_results_.get('r2_mean'),
                "cv_r2_std": self.cross_validation_results_.get('r2_std'),
                "cv_folds": self.cross_validation_results_.get('cv_folds')
            }
        
        return info

    def get_algorithm_specific_metrics(self, y_true: Optional[np.ndarray] = None, y_pred: Optional[np.ndarray] = None, y_proba: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        Retrieve algorithm-specific metrics for the Ridge Regression model.

        These metrics are derived from the model's learned parameters, regularization analysis,
        and cross-validation results if performed.
        Parameters y_true, y_pred, y_proba are kept for API consistency but are not
        directly used as metrics are sourced from the fitted model's internal attributes.

        Parameters:
        -----------
        y_true : np.ndarray, optional
            Actual target values.
        y_pred : np.ndarray, optional
            Predicted target values.
        y_proba : np.ndarray, optional
            Predicted probabilities (not applicable for regression).

        Returns:
        --------
        Dict[str, Any]
            A dictionary containing algorithm-specific metrics.
        """
        if not self.is_fitted_ or not hasattr(self, 'model_') or self.model_ is None:
            return {"error": "Model not fitted. Cannot retrieve Ridge Regression specific metrics."}

        metrics = {}
        prefix = "ridge_reg_" # Prefix for Ridge Regression specific metrics

        # Optimal Alpha used
        if hasattr(self, 'optimal_alpha_'):
            metrics[f"{prefix}optimal_alpha_used"] = float(self.optimal_alpha_)
        else:
            metrics[f"{prefix}configured_alpha"] = float(self.alpha)

        # Coefficient-based metrics
        if hasattr(self.model_, 'coef_') and self.model_.coef_ is not None:
            coefficients = self.model_.coef_
            metrics[f"{prefix}coefficient_l2_norm"] = float(np.linalg.norm(coefficients))
            metrics[f"{prefix}mean_abs_coefficient"] = float(np.mean(np.abs(coefficients)))
            metrics[f"{prefix}max_abs_coefficient"] = float(np.max(np.abs(coefficients)))
            metrics[f"{prefix}num_coefficients"] = int(coefficients.size)
        
        # Intercept
        if hasattr(self.model_, 'intercept_') and self.model_.intercept_ is not None:
            metrics[f"{prefix}intercept_value"] = float(self.model_.intercept_)

        # Effective Degrees of Freedom
        if hasattr(self, 'regularization_analysis_') and self.regularization_analysis_:
            if 'effective_dof' in self.regularization_analysis_ and self.regularization_analysis_['effective_dof'] is not None:
                metrics[f"{prefix}effective_degrees_of_freedom"] = float(self.regularization_analysis_['effective_dof'])
            if 'dof_reduction' in self.regularization_analysis_ and self.regularization_analysis_['dof_reduction'] is not None:
                metrics[f"{prefix}degrees_of_freedom_reduction"] = float(self.regularization_analysis_['dof_reduction'])

        # Cross-validation performance for the optimal alpha (if auto_alpha was used)
        if self.auto_alpha and hasattr(self, 'cross_validation_results_') and self.cross_validation_results_:
            if 'r2_mean' in self.cross_validation_results_: # This is CV R2 for the final model
                 metrics[f"{prefix}cv_r2_mean_final_model"] = float(self.cross_validation_results_['r2_mean'])
            if 'mse_mean' in self.cross_validation_results_: # This is CV MSE for the final model
                 metrics[f"{prefix}cv_mse_mean_final_model"] = float(self.cross_validation_results_['mse_mean'])
            
            # Metrics from alpha tuning process
            alpha_val_results = self.cross_validation_results_.get('alpha_validation', {})
            if alpha_val_results and 'mean_scores' in alpha_val_results and 'optimal_alpha' in alpha_val_results:
                # Assuming mean_scores are neg_mean_squared_error from RidgeCV
                optimal_alpha_idx = np.argmin(np.abs(alpha_val_results['alphas'] - alpha_val_results['optimal_alpha']))
                if optimal_alpha_idx < len(alpha_val_results['mean_scores']):
                    # Stored as negative MSE, so negate back
                    metrics[f"{prefix}cv_neg_mse_at_optimal_alpha_tuning"] = float(alpha_val_results['mean_scores'][optimal_alpha_idx])


        # Feature Shrinkage Summary
        if hasattr(self, 'feature_analysis_') and self.feature_analysis_ and 'feature_analysis' in self.feature_analysis_:
            shrinkage_percentages = []
            for feat_name, analysis in self.feature_analysis_['feature_analysis'].items():
                if 'shrinkage_percent' in analysis and pd.notna(analysis['shrinkage_percent']):
                    shrinkage_percentages.append(analysis['shrinkage_percent'])
            if shrinkage_percentages:
                metrics[f"{prefix}mean_coefficient_shrinkage_percent"] = float(np.mean(shrinkage_percentages))
                metrics[f"{prefix}max_coefficient_shrinkage_percent"] = float(np.max(shrinkage_percentages))
        
        # Polynomial degree used
        metrics[f"{prefix}polynomial_degree_used"] = int(self.polynomial_degree)

        if not metrics:
            metrics['info'] = "No specific Ridge Regression metrics were available from internal analyses."
            
        return metrics

# Factory function
def get_plugin():
    """Factory function to get the plugin instance"""
    return RidgeRegressionPlugin()
