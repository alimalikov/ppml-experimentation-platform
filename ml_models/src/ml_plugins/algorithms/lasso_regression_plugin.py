import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, Optional, Tuple, List
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_X_y, check_array
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso, LassoCV, lasso_path
from sklearn.model_selection import cross_val_score, validation_curve
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


class LassoRegressionPlugin(BaseEstimator, RegressorMixin, MLPlugin):
    """
    Lasso Regression Plugin - L1 Regularization with Automatic Feature Selection
    
    This plugin implements comprehensive Lasso regression with L1 regularization for automatic
    feature selection, sparsity analysis, and regularization path exploration. Perfect for
    high-dimensional data and interpretable feature selection.
    
    Key Features:
    - L1 regularization for automatic feature selection
    - Cross-validated alpha selection (LassoCV)
    - Regularization path analysis
    - Sparsity analysis and feature ranking
    - Coefficient stability analysis
    - Advanced feature selection metrics
    - Cross-validation performance estimation
    - Comprehensive regularization diagnostics
    """
    
    def __init__(
        self,
        alpha=1.0,
        fit_intercept=True,
        normalize_features=True,
        precompute=False,
        copy_X=True,
        max_iter=1000,
        tol=1e-4,
        warm_start=False,
        positive=False,
        selection='cyclic',
        
        # Advanced options
        auto_alpha=True,
        alpha_selection_method='cv',
        cv_folds=5,
        n_alphas=100,
        alpha_min_ratio=1e-4,
        eps=1e-3,
        
        # Analysis options
        analyze_regularization_path=True,
        analyze_feature_stability=True,
        compute_feature_importance=True,
        sparsity_analysis=True,
        
        random_state=42
    ):
        super().__init__()
        
        # Core Lasso parameters
        self.alpha = alpha
        self.fit_intercept = fit_intercept
        self.normalize_features = normalize_features
        self.precompute = precompute
        self.copy_X = copy_X
        self.max_iter = max_iter
        self.tol = tol
        self.warm_start = warm_start
        self.positive = positive
        self.selection = selection
        
        # Advanced alpha selection
        self.auto_alpha = auto_alpha
        self.alpha_selection_method = alpha_selection_method
        self.cv_folds = cv_folds
        self.n_alphas = n_alphas
        self.alpha_min_ratio = alpha_min_ratio
        self.eps = eps
        
        # Analysis options
        self.analyze_regularization_path = analyze_regularization_path
        self.analyze_feature_stability = analyze_feature_stability
        self.compute_feature_importance = compute_feature_importance
        self.sparsity_analysis = sparsity_analysis
        
        self.random_state = random_state
        
        # Required plugin metadata
        self._name = "Lasso Regression"
        self._description = "L1 regularized regression with automatic feature selection and sparsity analysis"
        self._category = "Linear Models"
        
        # Required capability flags
        self._supports_classification = False
        self._supports_regression = True
        self._min_samples_required = 10
        
        # Internal state
        self.is_fitted_ = False
        self.model_ = None
        self.scaler_ = None
        self.feature_names_ = None
        self.n_features_in_ = None
        
        # Analysis results
        self.alpha_analysis_ = {}
        self.regularization_path_ = {}
        self.feature_selection_analysis_ = {}
        self.sparsity_analysis_ = {}
        self.stability_analysis_ = {}
    
    def get_name(self) -> str:
        return self._name
    
    def get_description(self) -> str:
        return self._description
    
    def get_category(self) -> str:
        return self._category
    
    def fit(self, X, y, sample_weight=None):
        """
        Fit the Lasso Regression model with comprehensive analysis
        
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
        
        # Apply feature scaling if requested
        X_processed = X.copy()
        if self.normalize_features:
            self.scaler_ = StandardScaler()
            X_processed = self.scaler_.fit_transform(X_processed)
        
        # Store processed training data
        self.X_processed_ = X_processed
        
        # Determine optimal alpha if auto_alpha is enabled
        if self.auto_alpha:
            optimal_alpha = self._find_optimal_alpha(X_processed, y, sample_weight)
            self.alpha_used_ = optimal_alpha
        else:
            self.alpha_used_ = self.alpha
        
        # Fit the Lasso regression model
        self.model_ = Lasso(
            alpha=self.alpha_used_,
            fit_intercept=self.fit_intercept,
            precompute=self.precompute,
            copy_X=self.copy_X,
            max_iter=self.max_iter,
            tol=self.tol,
            warm_start=self.warm_start,
            positive=self.positive,
            selection=self.selection,
            random_state=self.random_state
        )
        
        self.model_.fit(X_processed, y, sample_weight=sample_weight)
        
        # Perform comprehensive analysis
        self._analyze_regularization_path_detailed()
        self._analyze_feature_selection()
        self._analyze_sparsity()
        self._analyze_feature_stability()
        
        self.is_fitted_ = True
        return self
    
    def predict(self, X):
        """Make predictions using the fitted model"""
        if not self.is_fitted_:
            raise ValueError("Model must be fitted before making predictions")
        
        X = check_array(X, accept_sparse=False)
        
        # Apply same preprocessing as training
        X_processed = X.copy()
        if self.scaler_ is not None:
            X_processed = self.scaler_.transform(X_processed)
        
        return self.model_.predict(X_processed)
    
    def _find_optimal_alpha(self, X, y, sample_weight=None):
        """Find optimal alpha using cross-validation or other methods"""
        if self.alpha_selection_method == 'cv':
            # Use LassoCV for cross-validated alpha selection
            alphas = np.logspace(
                np.log10(self.alpha_min_ratio),
                np.log10(1.0),
                num=self.n_alphas
            )
            
            lasso_cv = LassoCV(
                alphas=alphas,
                cv=self.cv_folds,
                fit_intercept=self.fit_intercept,
                precompute=self.precompute,
                max_iter=self.max_iter,
                tol=self.tol,
                eps=self.eps,
                copy_X=self.copy_X,
                positive=self.positive,
                selection=self.selection,
                random_state=self.random_state
            )
            
            lasso_cv.fit(X, y, sample_weight=sample_weight)
            
            # Store CV results for analysis
            self.alpha_analysis_ = {
                'method': 'Cross-Validation',
                'alphas_tested': alphas,
                'cv_scores': lasso_cv.mse_path_,
                'optimal_alpha': lasso_cv.alpha_,
                'cv_folds': self.cv_folds,
                'alpha_std': np.std(lasso_cv.mse_path_, axis=1),
                'alpha_1se': self._calculate_1se_alpha(lasso_cv)
            }
            
            return lasso_cv.alpha_
            
        elif self.alpha_selection_method == 'aic':
            # Use AIC for alpha selection
            return self._find_alpha_by_aic(X, y, sample_weight)
        
        elif self.alpha_selection_method == 'bic':
            # Use BIC for alpha selection
            return self._find_alpha_by_ic(X, y, sample_weight, criterion='bic')
        
        else:
            # Default to provided alpha
            return self.alpha
    
    def _calculate_1se_alpha(self, lasso_cv):
        """Calculate 1-standard-error alpha for more parsimonious model"""
        mean_scores = np.mean(lasso_cv.mse_path_, axis=1)
        std_scores = np.std(lasso_cv.mse_path_, axis=1)
        
        # Find the minimum score and its standard error
        min_idx = np.argmin(mean_scores)
        min_score = mean_scores[min_idx]
        min_std = std_scores[min_idx]
        
        # Find the largest alpha within 1 SE of the minimum
        threshold = min_score + min_std
        valid_indices = np.where(mean_scores <= threshold)[0]
        
        if len(valid_indices) > 0:
            # Return the largest alpha (most regularized) within 1 SE
            return lasso_cv.alphas_[valid_indices[0]]
        else:
            return lasso_cv.alpha_
    
    def _find_alpha_by_ic(self, X, y, sample_weight=None, criterion='aic'):
        """Find optimal alpha using information criteria (AIC/BIC)"""
        alphas = np.logspace(
            np.log10(self.alpha_min_ratio),
            np.log10(1.0),
            num=self.n_alphas
        )
        
        ic_scores = []
        n_samples = X.shape[0]
        
        for alpha in alphas:
            lasso = Lasso(
                alpha=alpha,
                fit_intercept=self.fit_intercept,
                max_iter=self.max_iter,
                tol=self.tol,
                random_state=self.random_state
            )
            
            lasso.fit(X, y, sample_weight=sample_weight)
            y_pred = lasso.predict(X)
            
            # Calculate residual sum of squares
            rss = np.sum((y - y_pred) ** 2)
            
            # Count non-zero coefficients (effective parameters)
            n_params = np.sum(lasso.coef_ != 0) + (1 if self.fit_intercept else 0)
            
            # Calculate information criterion
            if criterion == 'aic':
                ic = n_samples * np.log(rss / n_samples) + 2 * n_params
            else:  # BIC
                ic = n_samples * np.log(rss / n_samples) + np.log(n_samples) * n_params
            
            ic_scores.append(ic)
        
        # Find alpha with minimum IC
        optimal_idx = np.argmin(ic_scores)
        optimal_alpha = alphas[optimal_idx]
        
        # Store analysis results
        self.alpha_analysis_ = {
            'method': criterion.upper(),
            'alphas_tested': alphas,
            'ic_scores': np.array(ic_scores),
            'optimal_alpha': optimal_alpha,
            'optimal_ic_score': ic_scores[optimal_idx]
        }
        
        return optimal_alpha
    
    def _analyze_regularization_path_detailed(self):
        """Analyze the complete regularization path"""
        if not self.analyze_regularization_path:
            return
        
        try:
            # Generate alphas for path analysis
            alphas = np.logspace(
                np.log10(self.alpha_min_ratio),
                np.log10(10.0),  # Extend range for better visualization
                num=min(100, self.n_alphas)
            )
            
            # Compute regularization path
            alphas_path, coefs_path, _ = lasso_path(
                self.X_processed_,
                self.y_original_,
                alphas=alphas,
                fit_intercept=self.fit_intercept,
                eps=self.eps,
                copy_X=True
            )
            
            # Analyze path characteristics
            n_features_path = np.sum(np.abs(coefs_path) > 1e-8, axis=0)
            max_coef_path = np.max(np.abs(coefs_path), axis=0)
            
            # Find feature entry points (when features first become non-zero)
            feature_entry_alphas = []
            for i in range(coefs_path.shape[0]):
                nonzero_indices = np.where(np.abs(coefs_path[i, :]) > 1e-8)[0]
                if len(nonzero_indices) > 0:
                    entry_alpha = alphas_path[nonzero_indices[-1]]  # Last alpha where it's non-zero
                    feature_entry_alphas.append(entry_alpha)
                else:
                    feature_entry_alphas.append(np.inf)
            
            # Store regularization path analysis
            self.regularization_path_ = {
                'alphas': alphas_path,
                'coefficients': coefs_path,
                'n_features_active': n_features_path,
                'max_coefficient_magnitude': max_coef_path,
                'feature_entry_alphas': np.array(feature_entry_alphas),
                'feature_names': self.feature_names_,
                'current_alpha': self.alpha_used_,
                'current_n_features': np.sum(np.abs(self.model_.coef_) > 1e-8)
            }
            
        except Exception as e:
            self.regularization_path_ = {
                'error': f'Could not compute regularization path: {str(e)}'
            }
    
    def _analyze_feature_selection(self):
        """Analyze feature selection characteristics"""
        if not self.compute_feature_importance:
            return
        
        # Get coefficients
        coefficients = self.model_.coef_
        
        # Identify selected features (non-zero coefficients)
        selected_mask = np.abs(coefficients) > 1e-8
        selected_features = np.where(selected_mask)[0]
        eliminated_features = np.where(~selected_mask)[0]
        
        # Calculate feature importance based on coefficient magnitudes
        importance_scores = np.abs(coefficients)
        
        # Normalize importance scores
        if np.sum(importance_scores) > 0:
            normalized_importance = importance_scores / np.sum(importance_scores)
        else:
            normalized_importance = importance_scores
        
        # Rank features by importance
        feature_ranking = np.argsort(importance_scores)[::-1]
        
        # Calculate selection stability if regularization path is available
        selection_stability = self._calculate_selection_stability()
        
        self.feature_selection_analysis_ = {
            'selected_features': {
                'indices': selected_features.tolist(),
                'names': [self.feature_names_[i] for i in selected_features],
                'coefficients': coefficients[selected_features].tolist(),
                'importance_scores': importance_scores[selected_features].tolist()
            },
            'eliminated_features': {
                'indices': eliminated_features.tolist(),
                'names': [self.feature_names_[i] for i in eliminated_features],
                'count': len(eliminated_features)
            },
            'feature_ranking': {
                'indices': feature_ranking.tolist(),
                'names': [self.feature_names_[i] for i in feature_ranking],
                'importance_scores': importance_scores[feature_ranking].tolist(),
                'normalized_importance': normalized_importance[feature_ranking].tolist()
            },
            'selection_statistics': {
                'total_features': len(coefficients),
                'selected_count': len(selected_features),
                'eliminated_count': len(eliminated_features),
                'selection_ratio': len(selected_features) / len(coefficients),
                'sparsity_level': len(eliminated_features) / len(coefficients)
            },
            'selection_stability': selection_stability
        }
    
    def _calculate_selection_stability(self):
        """Calculate feature selection stability across different alpha values"""
        if 'coefficients' not in self.regularization_path_:
            return None
        
        try:
            coefs_path = self.regularization_path_['coefficients']
            alphas_path = self.regularization_path_['alphas']
            
            # Create binary selection matrix
            selection_matrix = np.abs(coefs_path) > 1e-8
            
            # Calculate Jaccard similarity for consecutive alpha values
            jaccard_similarities = []
            for i in range(len(alphas_path) - 1):
                set1 = set(np.where(selection_matrix[:, i])[0])
                set2 = set(np.where(selection_matrix[:, i + 1])[0])
                
                if len(set1) == 0 and len(set2) == 0:
                    jaccard = 1.0
                elif len(set1) == 0 or len(set2) == 0:
                    jaccard = 0.0
                else:
                    intersection = len(set1.intersection(set2))
                    union = len(set1.union(set2))
                    jaccard = intersection / union if union > 0 else 0.0
                
                jaccard_similarities.append(jaccard)
            
            # Calculate stability metrics
            mean_stability = np.mean(jaccard_similarities)
            stability_variance = np.var(jaccard_similarities)
            
            return {
                'jaccard_similarities': jaccard_similarities,
                'mean_stability': mean_stability,
                'stability_variance': stability_variance,
                'interpretation': self._interpret_stability(mean_stability)
            }
            
        except Exception:
            return None
    
    def _interpret_stability(self, stability_score):
        """Interpret feature selection stability score"""
        if stability_score >= 0.8:
            return "Very stable feature selection"
        elif stability_score >= 0.6:
            return "Moderately stable feature selection"
        elif stability_score >= 0.4:
            return "Somewhat unstable feature selection"
        else:
            return "Unstable feature selection - consider different alpha"
    
    def _analyze_sparsity(self):
        """Analyze sparsity characteristics of the solution"""
        if not self.sparsity_analysis:
            return
        
        coefficients = self.model_.coef_
        
        # Basic sparsity metrics
        total_features = len(coefficients)
        zero_coefs = np.sum(np.abs(coefficients) < 1e-8)
        nonzero_coefs = total_features - zero_coefs
        sparsity_ratio = zero_coefs / total_features
        
        # Coefficient magnitude analysis
        nonzero_coef_values = coefficients[np.abs(coefficients) > 1e-8]
        if len(nonzero_coef_values) > 0:
            coef_stats = {
                'mean': np.mean(np.abs(nonzero_coef_values)),
                'std': np.std(np.abs(nonzero_coef_values)),
                'min': np.min(np.abs(nonzero_coef_values)),
                'max': np.max(np.abs(nonzero_coef_values)),
                'median': np.median(np.abs(nonzero_coef_values))
            }
        else:
            coef_stats = {
                'mean': 0, 'std': 0, 'min': 0, 'max': 0, 'median': 0
            }
        
        # Sparsity comparison with different alphas
        sparsity_comparison = self._analyze_sparsity_vs_alpha()
        
        self.sparsity_analysis_ = {
            'basic_metrics': {
                'total_features': total_features,
                'nonzero_coefficients': nonzero_coefs,
                'zero_coefficients': zero_coefs,
                'sparsity_ratio': sparsity_ratio,
                'density_ratio': 1 - sparsity_ratio
            },
            'coefficient_statistics': coef_stats,
            'sparsity_level': self._categorize_sparsity(sparsity_ratio),
            'sparsity_comparison': sparsity_comparison,
            'regularization_strength': {
                'alpha_used': self.alpha_used_,
                'interpretation': self._interpret_alpha_strength(self.alpha_used_)
            }
        }
    
    def _analyze_sparsity_vs_alpha(self):
        """Analyze how sparsity changes with different alpha values"""
        if 'coefficients' not in self.regularization_path_:
            return None
        
        try:
            coefs_path = self.regularization_path_['coefficients']
            alphas_path = self.regularization_path_['alphas']
            
            sparsity_ratios = []
            for i in range(coefs_path.shape[1]):
                coefs = coefs_path[:, i]
                zero_count = np.sum(np.abs(coefs) < 1e-8)
                sparsity_ratio = zero_count / len(coefs)
                sparsity_ratios.append(sparsity_ratio)
            
            return {
                'alphas': alphas_path.tolist(),
                'sparsity_ratios': sparsity_ratios,
                'current_alpha_sparsity': sparsity_ratios[np.argmin(np.abs(alphas_path - self.alpha_used_))]
            }
            
        except Exception:
            return None
    
    def _categorize_sparsity(self, sparsity_ratio):
        """Categorize sparsity level"""
        if sparsity_ratio >= 0.9:
            return "Very High Sparsity (>90% features eliminated)"
        elif sparsity_ratio >= 0.7:
            return "High Sparsity (70-90% features eliminated)"
        elif sparsity_ratio >= 0.5:
            return "Moderate Sparsity (50-70% features eliminated)"
        elif sparsity_ratio >= 0.2:
            return "Low Sparsity (20-50% features eliminated)"
        else:
            return "Very Low Sparsity (<20% features eliminated)"
    
    def _interpret_alpha_strength(self, alpha):
        """Interpret the strength of regularization"""
        if alpha >= 1.0:
            return "Strong regularization (high sparsity expected)"
        elif alpha >= 0.1:
            return "Moderate regularization (balanced sparsity)"
        elif alpha >= 0.01:
            return "Weak regularization (low sparsity)"
        else:
            return "Very weak regularization (minimal sparsity)"
    
    def _analyze_feature_stability(self):
        """Analyze stability of feature selection"""
        if not self.analyze_feature_stability:
            return
        
        if 'coefficients' not in self.regularization_path_:
            self.stability_analysis_ = {
                'status': 'Regularization path not available for stability analysis'
            }
            return
        
        try:
            coefs_path = self.regularization_path_['coefficients']
            alphas_path = self.regularization_path_['alphas']
            
            # Find current alpha position in path
            current_alpha_idx = np.argmin(np.abs(alphas_path - self.alpha_used_))
            
            # Analyze stability around current alpha
            stability_window = min(10, len(alphas_path) // 4)
            start_idx = max(0, current_alpha_idx - stability_window // 2)
            end_idx = min(len(alphas_path), current_alpha_idx + stability_window // 2)
            
            # Calculate feature selection frequency in the window
            window_coefs = coefs_path[:, start_idx:end_idx]
            selection_frequency = np.mean(np.abs(window_coefs) > 1e-8, axis=1)
            
            # Categorize features by stability
            highly_stable = np.where(selection_frequency >= 0.8)[0]
            moderately_stable = np.where((selection_frequency >= 0.4) & (selection_frequency < 0.8))[0]
            unstable = np.where(selection_frequency < 0.4)[0]
            
            self.stability_analysis_ = {
                'selection_frequency': selection_frequency.tolist(),
                'feature_names': self.feature_names_,
                'stability_categories': {
                    'highly_stable': {
                        'indices': highly_stable.tolist(),
                        'names': [self.feature_names_[i] for i in highly_stable],
                        'count': len(highly_stable)
                    },
                    'moderately_stable': {
                        'indices': moderately_stable.tolist(),
                        'names': [self.feature_names_[i] for i in moderately_stable],
                        'count': len(moderately_stable)
                    },
                    'unstable': {
                        'indices': unstable.tolist(),
                        'names': [self.feature_names_[i] for i in unstable],
                        'count': len(unstable)
                    }
                },
                'stability_window': {
                    'alpha_range': [alphas_path[start_idx], alphas_path[end_idx-1]],
                    'window_size': stability_window,
                    'current_alpha': self.alpha_used_
                },
                'overall_stability_score': np.mean(selection_frequency),
                'stability_interpretation': self._interpret_overall_stability(np.mean(selection_frequency))
            }
            
        except Exception as e:
            self.stability_analysis_ = {
                'error': f'Could not perform stability analysis: {str(e)}'
            }
    
    def _interpret_overall_stability(self, stability_score):
        """Interpret overall stability score"""
        if stability_score >= 0.8:
            return "Very stable - features consistently selected"
        elif stability_score >= 0.6:
            return "Good stability - most features consistently selected"
        elif stability_score >= 0.4:
            return "Moderate stability - some variation in feature selection"
        else:
            return "Poor stability - high variation in feature selection"
    
    def get_hyperparameter_config(self, key_prefix: str) -> Dict[str, Any]:
        """Generate Streamlit UI for hyperparameters"""
        
        # Create tabs for different configuration aspects
        tab1, tab2, tab3, tab4 = st.tabs(["Core Parameters", "Alpha Selection", "Advanced Options", "Algorithm Info"])
        
        with tab1:
            st.markdown("**Lasso Regression Configuration**")
            
            # Core parameters
            col1, col2 = st.columns(2)
            
            with col1:
                fit_intercept = st.checkbox(
                    "Fit Intercept",
                    value=self.fit_intercept,
                    help="Whether to calculate the intercept for this model",
                    key=f"{key_prefix}_fit_intercept"
                )
                
                normalize_features = st.checkbox(
                    "Normalize Features",
                    value=self.normalize_features,
                    help="Apply StandardScaler to features (recommended for Lasso)",
                    key=f"{key_prefix}_normalize_features"
                )
                
                positive = st.checkbox(
                    "Positive Coefficients Only",
                    value=self.positive,
                    help="Constrain coefficients to be non-negative",
                    key=f"{key_prefix}_positive"
                )
            
            with col2:
                max_iter = st.number_input(
                    "Maximum Iterations:",
                    value=self.max_iter,
                    min_value=100,
                    max_value=10000,
                    step=100,
                    help="Maximum number of iterations for convergence",
                    key=f"{key_prefix}_max_iter"
                )
                
                tol = st.number_input(
                    "Tolerance:",
                    value=self.tol,
                    min_value=1e-6,
                    max_value=1e-2,
                    step=1e-5,
                    format="%.1e",
                    help="Tolerance for optimization convergence",
                    key=f"{key_prefix}_tol"
                )
                
                selection = st.selectbox(
                    "Coordinate Selection:",
                    options=['cyclic', 'random'],
                    index=['cyclic', 'random'].index(self.selection),
                    help="Algorithm for coefficient updates",
                    key=f"{key_prefix}_selection"
                )
        
        with tab2:
            st.markdown("**Alpha (Regularization Strength) Configuration**")
            
            auto_alpha = st.checkbox(
                "Automatic Alpha Selection",
                value=self.auto_alpha,
                help="Automatically find optimal alpha using cross-validation or information criteria",
                key=f"{key_prefix}_auto_alpha"
            )
            
            if auto_alpha:
                col1, col2 = st.columns(2)
                
                with col1:
                    alpha_selection_method = st.selectbox(
                        "Alpha Selection Method:",
                        options=['cv', 'aic', 'bic'],
                        index=['cv', 'aic', 'bic'].index(self.alpha_selection_method),
                        help="Method for selecting optimal alpha",
                        key=f"{key_prefix}_alpha_selection_method"
                    )
                    
                    if alpha_selection_method == 'cv':
                        cv_folds = st.number_input(
                            "CV Folds:",
                            value=self.cv_folds,
                            min_value=3,
                            max_value=10,
                            step=1,
                            help="Number of cross-validation folds",
                            key=f"{key_prefix}_cv_folds"
                        )
                    else:
                        cv_folds = self.cv_folds
                
                with col2:
                    n_alphas = st.number_input(
                        "Number of Alphas to Test:",
                        value=self.n_alphas,
                        min_value=10,
                        max_value=200,
                        step=10,
                        help="Number of alpha values to test",
                        key=f"{key_prefix}_n_alphas"
                    )
                    
                    alpha_min_ratio = st.number_input(
                        "Alpha Min Ratio:",
                        value=self.alpha_min_ratio,
                        min_value=1e-6,
                        max_value=1e-2,
                        step=1e-5,
                        format="%.1e",
                        help="Ratio of smallest to largest alpha",
                        key=f"{key_prefix}_alpha_min_ratio"
                    )
                
                alpha = self.alpha  # Use default for automatic selection
                
                st.info(f"🔄 Using {alpha_selection_method.upper()} for automatic alpha selection")
                
            else:
                # Manual alpha selection
                col1, col2 = st.columns(2)
                
                with col1:
                    alpha = st.number_input(
                        "Alpha (Regularization Strength):",
                        value=self.alpha,
                        min_value=1e-6,
                        max_value=10.0,
                        step=0.01,
                        format="%.4f",
                        help="L1 regularization parameter (higher = more sparse)",
                        key=f"{key_prefix}_alpha"
                    )
                
                with col2:
                    # Alpha interpretation
                    if alpha >= 1.0:
                        alpha_desc = "Strong regularization (high sparsity)"
                        alpha_color = "🔴"
                    elif alpha >= 0.1:
                        alpha_desc = "Moderate regularization"
                        alpha_color = "🟡"
                    elif alpha >= 0.01:
                        alpha_desc = "Weak regularization"
                        alpha_color = "🟢"
                    else:
                        alpha_desc = "Very weak regularization"
                        alpha_color = "🔵"
                    
                    st.info(f"{alpha_color} {alpha_desc}")
                
                # Set auto_alpha parameters to defaults
                alpha_selection_method = self.alpha_selection_method
                cv_folds = self.cv_folds
                n_alphas = self.n_alphas
                alpha_min_ratio = self.alpha_min_ratio
        
        with tab3:
            st.markdown("**Advanced Configuration**")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Analysis Options:**")
                
                analyze_regularization_path = st.checkbox(
                    "Regularization Path Analysis",
                    value=self.analyze_regularization_path,
                    help="Analyze coefficient paths across different alpha values",
                    key=f"{key_prefix}_analyze_regularization_path"
                )
                
                analyze_feature_stability = st.checkbox(
                    "Feature Stability Analysis",
                    value=self.analyze_feature_stability,
                    help="Analyze stability of feature selection",
                    key=f"{key_prefix}_analyze_feature_stability"
                )
                
                compute_feature_importance = st.checkbox(
                    "Feature Importance Computation",
                    value=self.compute_feature_importance,
                    help="Compute and rank feature importance",
                    key=f"{key_prefix}_compute_feature_importance"
                )
                
                sparsity_analysis = st.checkbox(
                    "Sparsity Analysis",
                    value=self.sparsity_analysis,
                    help="Detailed analysis of solution sparsity",
                    key=f"{key_prefix}_sparsity_analysis"
                )
            
            with col2:
                st.markdown("**Technical Options:**")
                
                copy_X = st.checkbox(
                    "Copy Input Data",
                    value=self.copy_X,
                    help="Create copy of input data (safer but uses more memory)",
                    key=f"{key_prefix}_copy_X"
                )
                
                warm_start = st.checkbox(
                    "Warm Start",
                    value=self.warm_start,
                    help="Reuse solution from previous fit as initialization",
                    key=f"{key_prefix}_warm_start"
                )
                
                precompute = st.checkbox(
                    "Precompute Gram Matrix",
                    value=self.precompute,
                    help="Precompute Gram matrix (faster for n_samples > n_features)",
                    key=f"{key_prefix}_precompute"
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
            **Lasso Regression** - L1 Regularization with Feature Selection:
            • 🎯 Automatic feature selection through L1 penalty
            • 🔍 Sparse solutions (many coefficients = 0)
            • 📊 Built-in feature importance ranking
            • ⚡ Cross-validated alpha selection
            • 📈 Regularization path analysis
            • 🎪 Perfect for high-dimensional data
            
            **Mathematical Foundation:**
            • Objective: ||y - Xβ||² + α||β||₁
            • L1 penalty encourages sparsity
            • Automatic feature selection
            • Convex optimization problem
            """)
            
            # When to use Lasso
            if st.button("🎯 When to Use Lasso Regression", key=f"{key_prefix}_when_to_use"):
                st.markdown("""
                **Ideal Use Cases:**
                
                **Problem Characteristics:**
                • High-dimensional data (many features)
                • Need automatic feature selection
                • Sparse underlying model expected
                • Interpretability is important
                
                **Data Characteristics:**
                • More features than samples (p > n)
                • Many irrelevant or redundant features
                • Linear relationships with sparse coefficients
                • Features are somewhat correlated
                
                **Business Requirements:**
                • Feature selection for cost reduction
                • Model interpretability crucial
                • Regulatory compliance (explainable models)
                • Computational efficiency in prediction
                
                **Examples:**
                • Gene selection in bioinformatics
                • Text classification with many features
                • Financial risk modeling
                • Marketing attribution modeling
                """)
            
            # Advantages and limitations
            if st.button("⚖️ Advantages & Limitations", key=f"{key_prefix}_pros_cons"):
                st.markdown("""
                **Advantages:**
                ✅ Automatic feature selection
                ✅ Sparse, interpretable solutions
                ✅ Handles high-dimensional data well
                ✅ Built-in regularization prevents overfitting
                ✅ Computationally efficient
                ✅ Cross-validation for alpha selection
                ✅ Well-understood theoretical properties
                ✅ Reduces model complexity
                
                **Limitations:**
                ❌ May arbitrarily select among correlated features
                ❌ Can eliminate all correlated features
                ❌ Sensitive to feature scaling
                ❌ May underperform when most features are relevant
                ❌ Linear relationships only
                ❌ Alpha tuning can be sensitive
                ❌ May be unstable with small sample sizes
                """)
            
            # Alpha selection guide
            if st.button("🔧 Alpha Selection Guide", key=f"{key_prefix}_alpha_guide"):
                st.markdown("""
                **Understanding Alpha (Regularization Strength):**
                
                **Alpha = 0:**
                • No regularization (ordinary least squares)
                • May overfit with many features
                • All features retained
                
                **Small Alpha (0.001 - 0.01):**
                • Weak regularization
                • Most features retained
                • Similar to Ridge regression
                
                **Moderate Alpha (0.01 - 1.0):**
                • Balanced regularization
                • Some feature selection
                • Good starting point
                
                **Large Alpha (1.0+):**
                • Strong regularization
                • High sparsity (few features)
                • May underfit if too large
                
                **Selection Methods:**
                • **CV:** Cross-validation (most robust)
                • **AIC:** Akaike Information Criterion
                • **BIC:** Bayesian Information Criterion (more conservative)
                """)
            
            # Best practices
            if st.button("🎯 Best Practices", key=f"{key_prefix}_best_practices"):
                st.markdown("""
                **Lasso Regression Best Practices:**
                
                **Data Preparation:**
                1. **Always standardize features** (critical for Lasso)
                2. Remove constant and near-constant features
                3. Check for perfect multicollinearity
                4. Consider feature engineering for non-linear relationships
                
                **Alpha Selection:**
                1. Use cross-validation for robust alpha selection
                2. Consider 1-SE rule for more parsimonious models
                3. Plot regularization path to understand feature selection
                4. Validate alpha on independent test set
                
                **Model Interpretation:**
                1. Examine which features are selected vs eliminated
                2. Check stability of feature selection
                3. Validate biological/business relevance of selected features
                4. Consider elastic net if feature groups are important
                
                **Validation:**
                1. Use proper cross-validation
                2. Check prediction performance on test set
                3. Analyze residuals and model assumptions
                4. Compare with other regularization methods
                """)
        
        return {
            "alpha": alpha,
            "fit_intercept": fit_intercept,
            "normalize_features": normalize_features,
            "precompute": precompute,
            "copy_X": copy_X,
            "max_iter": max_iter,
            "tol": tol,
            "warm_start": warm_start,
            "positive": positive,
            "selection": selection,
            "auto_alpha": auto_alpha,
            "alpha_selection_method": alpha_selection_method,
            "cv_folds": cv_folds,
            "n_alphas": n_alphas,
            "alpha_min_ratio": alpha_min_ratio,
            "analyze_regularization_path": analyze_regularization_path,
            "analyze_feature_stability": analyze_feature_stability,
            "compute_feature_importance": compute_feature_importance,
            "sparsity_analysis": sparsity_analysis,
            "random_state": random_state
        }
    
    def create_model_instance(self, hyperparameters: Dict[str, Any]):
        """Create model instance with given hyperparameters"""
        return LassoRegressionPlugin(
            alpha=hyperparameters.get("alpha", self.alpha),
            fit_intercept=hyperparameters.get("fit_intercept", self.fit_intercept),
            normalize_features=hyperparameters.get("normalize_features", self.normalize_features),
            precompute=hyperparameters.get("precompute", self.precompute),
            copy_X=hyperparameters.get("copy_X", self.copy_X),
            max_iter=hyperparameters.get("max_iter", self.max_iter),
            tol=hyperparameters.get("tol", self.tol),
            warm_start=hyperparameters.get("warm_start", self.warm_start),
            positive=hyperparameters.get("positive", self.positive),
            selection=hyperparameters.get("selection", self.selection),
            auto_alpha=hyperparameters.get("auto_alpha", self.auto_alpha),
            alpha_selection_method=hyperparameters.get("alpha_selection_method", self.alpha_selection_method),
            cv_folds=hyperparameters.get("cv_folds", self.cv_folds),
            n_alphas=hyperparameters.get("n_alphas", self.n_alphas),
            alpha_min_ratio=hyperparameters.get("alpha_min_ratio", self.alpha_min_ratio),
            analyze_regularization_path=hyperparameters.get("analyze_regularization_path", self.analyze_regularization_path),
            analyze_feature_stability=hyperparameters.get("analyze_feature_stability", self.analyze_feature_stability),
            compute_feature_importance=hyperparameters.get("compute_feature_importance", self.compute_feature_importance),
            sparsity_analysis=hyperparameters.get("sparsity_analysis", self.sparsity_analysis),
            random_state=hyperparameters.get("random_state", self.random_state)
        )
    
    def preprocess_data(self, X, y=None, training=True):
        """Preprocess data for Lasso Regression"""
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
        """Check if Lasso Regression is compatible with the given data"""
        # Check minimum samples
        if X.shape[0] < self._min_samples_required:
            return False, f"Lasso regression requires at least {self._min_samples_required} samples, got {X.shape[0]}"
        
        # Check for regression targets
        if y is not None:
            if not np.issubdtype(y.dtype, np.number):
                return False, "Lasso regression requires continuous numerical target values"
            
            # Check for sufficient variance in target
            if np.var(y) == 0:
                return False, "Target variable has zero variance (all values are the same)"
            
            # Sample and feature analysis
            n_samples, n_features = X.shape
            
            advantages = []
            considerations = []
            
            # High-dimensional data assessment (Lasso's strength)
            if n_features > n_samples:
                advantages.append(f"High-dimensional data ({n_features} features, {n_samples} samples) - perfect for Lasso")
            elif n_features > n_samples * 0.5:
                advantages.append(f"Many features ({n_features}) relative to samples ({n_samples}) - good for Lasso")
            elif n_features > 50:
                advantages.append(f"Moderate feature count ({n_features}) - Lasso can provide feature selection")
            else:
                considerations.append(f"Low feature count ({n_features}) - feature selection benefit may be limited")
            
            # Sample size assessment
            if n_samples >= n_features * 10:
                advantages.append(f"Excellent sample size ({n_samples}) for reliable feature selection")
            elif n_samples >= n_features * 3:
                advantages.append(f"Good sample size ({n_samples}) for feature selection")
            elif n_samples >= n_features:
                considerations.append(f"Adequate sample size ({n_samples}) but feature selection may be unstable")
            else:
                considerations.append(f"Small sample size ({n_samples}) - feature selection may be very unstable")
            
            # Feature scaling check
            try:
                feature_scales = np.std(X, axis=0)
                max_scale_ratio = np.max(feature_scales) / np.min(feature_scales) if np.min(feature_scales) > 0 else np.inf
                
                if max_scale_ratio > 100:
                    considerations.append("Features have very different scales - standardization strongly recommended")
                elif max_scale_ratio > 10:
                    considerations.append("Features have different scales - standardization recommended")
                else:
                    advantages.append("Feature scales are similar")
            except:
                pass
            
            # Sparsity potential
            if n_features > 100:
                advantages.append("High feature count - Lasso can identify truly important features")
            
            # Build compatibility message
            if len(considerations) == 0:
                suitability = "Excellent"
            elif len(considerations) <= 1:
                suitability = "Very Good"
            elif len(considerations) <= 2:
                suitability = "Good"
            else:
                suitability = "Fair"
            
            message_parts = [
                f"✅ Compatible with {n_samples} samples, {n_features} features",
                f"📊 Suitability for Lasso: {suitability}"
            ]
            
            if advantages:
                message_parts.append("🎯 Advantages: " + "; ".join(advantages))
            
            if considerations:
                message_parts.append("💡 Considerations: " + "; ".join(considerations))
            
            return True, " | ".join(message_parts)
        
        return True, f"Compatible with {X.shape[0]} samples and {X.shape[1]} features"
    
    def get_feature_importance(self) -> Optional[Dict[str, Any]]:
        """Get feature importance based on coefficient magnitudes and selection"""
        if not self.is_fitted_:
            return None
        
        coefficients = self.model_.coef_
        feature_names = self.feature_names_
        
        # Calculate importance as absolute coefficient values
        importance = np.abs(coefficients)
        
        # Separate selected and eliminated features
        selected_mask = importance > 1e-8
        selected_indices = np.where(selected_mask)[0]
        eliminated_indices = np.where(~selected_mask)[0]
        
        # Normalize importance (only for selected features)
        selected_importance = importance[selected_mask]
        if np.sum(selected_importance) > 0:
            normalized_importance = np.zeros_like(importance)
            normalized_importance[selected_mask] = selected_importance / np.sum(selected_importance)
        else:
            normalized_importance = importance
        
        # Create feature importance dictionary
        feature_importance = {}
        for i, (name, coef, imp, norm_imp) in enumerate(zip(feature_names, coefficients, importance, normalized_importance)):
            feature_importance[name] = {
                'coefficient': coef,
                'absolute_coefficient': imp,
                'normalized_importance': norm_imp,
                'selected': selected_mask[i],
                'rank': i + 1
            }
        
        # Sort by importance (selected features first, then by magnitude)
        sorted_features = sorted(
            feature_importance.items(), 
            key=lambda x: (not x[1]['selected'], -x[1]['absolute_coefficient'])
        )
        
        # Update ranks
        for rank, (name, info) in enumerate(sorted_features):
            feature_importance[name]['rank'] = rank + 1
        
        return {
            'feature_importance': feature_importance,
            'selected_features': {
                'names': [feature_names[i] for i in selected_indices],
                'indices': selected_indices.tolist(),
                'coefficients': coefficients[selected_indices].tolist(),
                'importance': importance[selected_indices].tolist(),
                'count': len(selected_indices)
            },
            'eliminated_features': {
                'names': [feature_names[i] for i in eliminated_indices],
                'indices': eliminated_indices.tolist(),
                'count': len(eliminated_indices)
            },
            'sparsity_info': {
                'total_features': len(coefficients),
                'selected_count': len(selected_indices),
                'elimination_rate': len(eliminated_indices) / len(coefficients),
                'alpha_used': self.alpha_used_
            },
            'sorted_features': [name for name, _ in sorted_features],
            'sorted_importance': [info['normalized_importance'] for _, info in sorted_features],
            'interpretation': f'Feature selection via L1 regularization (α={self.alpha_used_:.4f})'
        }
    
    def get_model_params(self) -> Dict[str, Any]:
        """Get current model parameters"""
        if not self.is_fitted_:
            return {}
        
        selected_features = np.sum(np.abs(self.model_.coef_) > 1e-8)
        
        return {
            "algorithm": "Lasso Regression",
            "n_features": self.n_features_in_,
            "feature_names": self.feature_names_,
            "alpha_used": self.alpha_used_,
            "alpha_selection_method": self.alpha_selection_method if self.auto_alpha else "manual",
            "selected_features": selected_features,
            "eliminated_features": self.n_features_in_ - selected_features,
            "sparsity_ratio": (self.n_features_in_ - selected_features) / self.n_features_in_,
            "fit_intercept": self.fit_intercept,
            "normalize_features": self.normalize_features,
            "coefficients": self.model_.coef_.tolist(),
            "intercept": self.model_.intercept_ if self.fit_intercept else 0,
            "positive_constraint": self.positive,
            "max_iterations": self.max_iter,
            "convergence_tolerance": self.tol,
            "feature_scaling": self.scaler_ is not None,
            "regularization_analysis": self.analyze_regularization_path,
            "stability_analysis": self.analyze_feature_stability,
            "sparsity_analysis": self.sparsity_analysis
        }
    
    def get_regularization_analysis(self) -> Dict[str, Any]:
        """Get comprehensive regularization analysis results"""
        if not self.is_fitted_:
            return {"status": "Model not fitted"}
        
        results = {
            "alpha_selection": self.alpha_analysis_,
            "regularization_path": self.regularization_path_,
            "feature_selection": self.feature_selection_analysis_,
            "sparsity_analysis": self.sparsity_analysis_,
            "stability_analysis": self.stability_analysis_
        }
        
        return results
    
    def plot_regularization_path(self, figsize=(15, 10)):
        """Plot comprehensive regularization path analysis"""
        if not self.is_fitted_ or 'coefficients' not in self.regularization_path_:
            raise ValueError("Model must be fitted with regularization path analysis enabled")
        
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        
        alphas = self.regularization_path_['alphas']
        coefs = self.regularization_path_['coefficients']
        n_features_active = self.regularization_path_['n_features_active']
        
        # 1. Coefficient paths
        ax1 = axes[0, 0]
        for i in range(min(20, coefs.shape[0])):  # Limit to 20 features for readability
            ax1.plot(alphas, coefs[i, :], alpha=0.7, linewidth=1)
        
        ax1.axvline(x=self.alpha_used_, color='red', linestyle='--', alpha=0.8, label=f'Selected α={self.alpha_used_:.4f}')
        ax1.set_xscale('log')
        ax1.set_xlabel('Alpha')
        ax1.set_ylabel('Coefficients')
        ax1.set_title('Regularization Path - Coefficient Trajectories')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # 2. Number of active features vs alpha
        ax2 = axes[0, 1]
        ax2.plot(alphas, n_features_active, 'b-', linewidth=2, marker='o', markersize=3)
        ax2.axvline(x=self.alpha_used_, color='red', linestyle='--', alpha=0.8, 
                   label=f'Selected α: {self.regularization_path_["current_n_features"]} features')
        ax2.set_xscale('log')
        ax2.set_xlabel('Alpha')
        ax2.set_ylabel('Number of Active Features')
        ax2.set_title('Feature Selection vs Regularization Strength')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        # 3. Feature entry order
        ax3 = axes[1, 0]
        entry_alphas = self.regularization_path_['feature_entry_alphas']
        finite_entries = entry_alphas[np.isfinite(entry_alphas)]
        
        if len(finite_entries) > 0:
            ax3.hist(finite_entries, bins=min(20, len(finite_entries)), alpha=0.7, color='skyblue', edgecolor='black')
            ax3.axvline(x=self.alpha_used_, color='red', linestyle='--', alpha=0.8, label=f'Selected α')
            ax3.set_xscale('log')
            ax3.set_xlabel('Alpha')
            ax3.set_ylabel('Number of Features')
            ax3.set_title('Feature Entry Points Distribution')
            ax3.grid(True, alpha=0.3)
            ax3.legend()
        else:
            ax3.text(0.5, 0.5, 'No features selected\nat any alpha level', 
                    ha='center', va='center', transform=ax3.transAxes)
            ax3.set_title('Feature Entry Points Distribution')
        
        # 4. Cross-validation curve (if available)
        ax4 = axes[1, 1]
        if 'cv_scores' in self.alpha_analysis_:
            cv_scores = self.alpha_analysis_['cv_scores']
            alphas_cv = self.alpha_analysis_['alphas_tested']
            
            mean_scores = np.mean(cv_scores, axis=1)
            std_scores = np.std(cv_scores, axis=1)
            
            ax4.plot(alphas_cv, mean_scores, 'b-', linewidth=2, label='CV Score')
            ax4.fill_between(alphas_cv, mean_scores - std_scores, mean_scores + std_scores, 
                           alpha=0.3, color='blue')
            ax4.axvline(x=self.alpha_used_, color='red', linestyle='--', alpha=0.8, 
                       label=f'Selected α={self.alpha_used_:.4f}')
            
            if 'alpha_1se' in self.alpha_analysis_:
                ax4.axvline(x=self.alpha_analysis_['alpha_1se'], color='orange', linestyle='--', 
                           alpha=0.8, label=f'1-SE α={self.alpha_analysis_["alpha_1se"]:.4f}')
            
            ax4.set_xscale('log')
            ax4.set_xlabel('Alpha')
            ax4.set_ylabel('CV Score (MSE)')
            ax4.set_title('Cross-Validation Curve')
            ax4.grid(True, alpha=0.3)
            ax4.legend()
        else:
            ax4.text(0.5, 0.5, 'Cross-validation\nresults not available', 
                    ha='center', va='center', transform=ax4.transAxes)
            ax4.set_title('Cross-Validation Curve')
        
        plt.tight_layout()
        return fig
    
    def plot_feature_selection(self, max_features=20, figsize=(12, 8)):
        """Plot feature selection analysis"""
        importance_data = self.get_feature_importance()
        if not importance_data:
            raise ValueError("Model must be fitted to plot feature selection")
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        
        # Get selected and eliminated features
        selected_info = importance_data['selected_features']
        eliminated_info = importance_data['eliminated_features']
        
        # Plot 1: Selected features with coefficients
        if selected_info['count'] > 0:
            display_count = min(max_features, selected_info['count'])
            
            # Sort by absolute coefficient value
            coef_data = list(zip(selected_info['names'][:display_count], 
                               selected_info['coefficients'][:display_count]))
            coef_data.sort(key=lambda x: abs(x[1]), reverse=True)
            
            names, coefs = zip(*coef_data)
            colors = ['green' if c > 0 else 'red' for c in coefs]
            
            y_pos = np.arange(len(names))
            ax1.barh(y_pos, coefs, color=colors, alpha=0.7)
            ax1.set_yticks(y_pos)
            ax1.set_yticklabels(names)
            ax1.set_xlabel('Coefficient Value')
            ax1.set_title(f'Selected Features (α={self.alpha_used_:.4f})\n{selected_info["count"]} features selected')
            ax1.axvline(x=0, color='black', linestyle='-', alpha=0.3)
            ax1.grid(True, alpha=0.3)
        else:
            ax1.text(0.5, 0.5, f'No features selected\n(α={self.alpha_used_:.4f} too large)', 
                    ha='center', va='center', transform=ax1.transAxes)
            ax1.set_title('Selected Features')
        
        # Plot 2: Sparsity analysis
        total_features = importance_data['sparsity_info']['total_features']
        selected_count = importance_data['sparsity_info']['selected_count']
        eliminated_count = eliminated_info['count']
        
        labels = ['Selected', 'Eliminated']
        sizes = [selected_count, eliminated_count]
        colors = ['lightblue', 'lightcoral']
        explode = (0.05, 0)  # explode selected slice slightly
        
        wedges, texts, autotexts = ax2.pie(sizes, labels=labels, colors=colors, explode=explode, 
                                          autopct='%1.1f%%', shadow=True, startangle=90)
        
        # Add count information
        for i, (label, size) in enumerate(zip(labels, sizes)):
            texts[i].set_text(f'{label}\n({size} features)')
        
        ax2.set_title(f'Feature Selection Summary\nα={self.alpha_used_:.4f}')
        
        # Add sparsity information
        sparsity_ratio = eliminated_count / total_features
        ax2.text(0, -1.3, f'Sparsity: {sparsity_ratio:.1%}', ha='center', 
                transform=ax2.transAxes, fontsize=10, 
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        plt.tight_layout()
        return fig
    
    def plot_sparsity_analysis(self, figsize=(12, 6)):
        """Plot sparsity analysis across different alpha values"""
        if not self.is_fitted_ or 'sparsity_comparison' not in self.sparsity_analysis_:
            raise ValueError("Model must be fitted with sparsity analysis enabled")
        
        sparsity_data = self.sparsity_analysis_['sparsity_comparison']
        if sparsity_data is None:
            raise ValueError("Sparsity comparison data not available")
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        
        alphas = sparsity_data['alphas']
        sparsity_ratios = sparsity_data['sparsity_ratios']
        
        # Plot 1: Sparsity vs Alpha
        ax1.plot(alphas, sparsity_ratios, 'b-', linewidth=2, marker='o', markersize=4)
        ax1.axvline(x=self.alpha_used_, color='red', linestyle='--', alpha=0.8, 
                   label=f'Selected α={self.alpha_used_:.4f}')
        ax1.axhline(y=sparsity_data['current_alpha_sparsity'], color='red', linestyle=':', 
                   alpha=0.6, label=f'Current sparsity: {sparsity_data["current_alpha_sparsity"]:.1%}')
        
        ax1.set_xscale('log')
        ax1.set_xlabel('Alpha')
        ax1.set_ylabel('Sparsity Ratio')
        ax1.set_title('Sparsity vs Regularization Strength')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        ax1.set_ylim(0, 1)
        
        # Plot 2: Number of features vs Alpha
        n_features_active = [(1 - s) * len(self.feature_names_) for s in sparsity_ratios]
        ax2.plot(alphas, n_features_active, 'g-', linewidth=2, marker='s', markersize=4)
        ax2.axvline(x=self.alpha_used_, color='red', linestyle='--', alpha=0.8, 
                   label=f'Selected α')
        
        current_n_features = int((1 - sparsity_data['current_alpha_sparsity']) * len(self.feature_names_))
        ax2.axhline(y=current_n_features, color='red', linestyle=':', alpha=0.6, 
                   label=f'Current: {current_n_features} features')
        
        ax2.set_xscale('log')
        ax2.set_xlabel('Alpha')
        ax2.set_ylabel('Number of Active Features')
        ax2.set_title('Active Features vs Regularization Strength')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        plt.tight_layout()
        return fig
    
    def get_training_info(self) -> Dict[str, Any]:
        """Get information about the training process"""
        if not self.is_fitted_:
            return {"status": "Not fitted"}
        
        selected_features = np.sum(np.abs(self.model_.coef_) > 1e-8)
        total_features = len(self.model_.coef_)
        
        info = {
            "algorithm": "Lasso Regression",
            "type": "L1 Regularized Linear Regression",
            "training_completed": True,
            "regularization_characteristics": {
                "regularization_type": "L1 (Lasso)",
                "automatic_feature_selection": True,
                "sparsity_inducing": True,
                "handles_multicollinearity": "partially",
                "feature_selection_method": "coefficient shrinkage to zero"
            },
            "model_configuration": {
                "alpha_used": self.alpha_used_,
                "alpha_selection": "automatic" if self.auto_alpha else "manual",
                "alpha_method": self.alpha_selection_method if self.auto_alpha else "user_specified",
                "fit_intercept": self.fit_intercept,
                "normalize_features": self.normalize_features,
                "max_iterations": self.max_iter,
                "tolerance": self.tol,
                "positive_constraint": self.positive
            },
            "feature_selection_results": {
                "total_features": total_features,
                "selected_features": selected_features,
                "eliminated_features": total_features - selected_features,
                "sparsity_ratio": (total_features - selected_features) / total_features,
                "feature_reduction": f"{((total_features - selected_features) / total_features * 100):.1f}%"
            },
            "analysis_performed": {
                "regularization_path": self.analyze_regularization_path,
                "feature_stability": self.analyze_feature_stability,
                "sparsity_analysis": self.sparsity_analysis,
                "cross_validation": self.auto_alpha and self.alpha_selection_method == 'cv'
            }
        }
        
        # Add alpha selection results if available
        if self.alpha_analysis_:
            info["alpha_selection_results"] = {
                "method": self.alpha_analysis_.get('method'),
                "optimal_alpha": self.alpha_analysis_.get('optimal_alpha'),
                "cv_folds": self.alpha_analysis_.get('cv_folds'),
                "alphas_tested": len(self.alpha_analysis_.get('alphas_tested', []))
            }
        
        # Add sparsity analysis results
        if self.sparsity_analysis_:
            basic_metrics = self.sparsity_analysis_.get('basic_metrics', {})
            info["sparsity_characteristics"] = {
                "sparsity_level": self.sparsity_analysis_.get('sparsity_level'),
                "regularization_strength": self.sparsity_analysis_.get('regularization_strength', {}).get('interpretation'),
                "density_ratio": basic_metrics.get('density_ratio')
            }
        
        return info

    # ADD THE OVERRIDDEN METHOD HERE:
    def get_algorithm_specific_metrics(self,
                                       y_true: np.ndarray,
                                       y_pred: np.ndarray,
                                       y_proba: Optional[np.ndarray] = None
                                       ) -> Dict[str, Any]:
        """
        Calculate Lasso Regression-specific metrics based on the fitted model's
        internal analyses and characteristics.

        Note: Most metrics are derived from the extensive analyses performed during the fit method.
        The y_true, y_pred parameters (typically for test set evaluation)
        are not directly used for these internal model-specific metrics.
        y_proba is not applicable for regression.

        Args:
            y_true: Ground truth target values from a test set.
            y_pred: Predicted target values on a test set.
            y_proba: Not used for regression.

        Returns:
            A dictionary of Lasso Regression-specific metrics.
        """
        metrics = {}
        if not self.is_fitted_ or self.model_ is None:
            metrics["status"] = "Model not fitted"
            return metrics

        # Helper to safely extract nested dictionary values
        def safe_get(data_dict, path, default=np.nan):
            keys = path.split('.')
            current = data_dict
            for key in keys:
                if isinstance(current, dict) and key in current:
                    current = current[key]
                elif isinstance(current, list) and isinstance(key, int) and -len(current) <= key < len(current):
                    current = current[key]
                else:
                    return default
            # Avoid returning large lists/arrays if a scalar is expected
            if isinstance(current, (list, np.ndarray)) and len(current) > 10 and not path.endswith("_scores"):
                 return default
            return current if pd.notna(current) else default

        metrics['alpha_used'] = self.alpha_used_

        # --- Alpha Selection Analysis ---
        if hasattr(self, 'alpha_analysis_') and self.alpha_analysis_ and not self.alpha_analysis_.get('error'):
            alpha_an = self.alpha_analysis_
            metrics['alpha_selection_method'] = safe_get(alpha_an, 'method')
            metrics['alpha_optimal_selected'] = safe_get(alpha_an, 'optimal_alpha')
            if safe_get(alpha_an, 'method') == 'Cross-Validation':
                metrics['alpha_1se_rule'] = safe_get(alpha_an, 'alpha_1se')
            metrics['alpha_cv_folds_used'] = safe_get(alpha_an, 'cv_folds') # Will be NaN if not CV

        # --- Feature Selection & Sparsity Analysis ---
        if hasattr(self, 'feature_selection_analysis_') and self.feature_selection_analysis_ and not self.feature_selection_analysis_.get('error'):
            fs_an = self.feature_selection_analysis_
            metrics['fs_selected_features_count'] = safe_get(fs_an, 'selection_statistics.selected_count')
            metrics['fs_eliminated_features_count'] = safe_get(fs_an, 'selection_statistics.eliminated_count')
            metrics['fs_selection_ratio'] = safe_get(fs_an, 'selection_statistics.selection_ratio')
            # selection_stability is a dict, get mean_stability from it
            metrics['fs_selection_mean_stability_jaccard'] = safe_get(fs_an, 'selection_stability.mean_stability')

        if hasattr(self, 'sparsity_analysis_') and self.sparsity_analysis_ and not self.sparsity_analysis_.get('error'):
            sp_an = self.sparsity_analysis_
            metrics['sparsity_ratio'] = safe_get(sp_an, 'basic_metrics.sparsity_ratio')
            metrics['density_ratio'] = safe_get(sp_an, 'basic_metrics.density_ratio')
            metrics['sparsity_mean_abs_nonzero_coef'] = safe_get(sp_an, 'coefficient_statistics.mean')
            metrics['sparsity_median_abs_nonzero_coef'] = safe_get(sp_an, 'coefficient_statistics.median')
            metrics['sparsity_current_alpha_sparsity_on_path'] = safe_get(sp_an, 'sparsity_comparison.current_alpha_sparsity')


        # --- Feature Stability Analysis (more detailed) ---
        if hasattr(self, 'stability_analysis_') and self.stability_analysis_ and not self.stability_analysis_.get('error'):
            stab_an = self.stability_analysis_
            metrics['stability_overall_score'] = safe_get(stab_an, 'overall_stability_score')
            metrics['stability_highly_stable_features_count'] = safe_get(stab_an, 'stability_categories.highly_stable.count')
            metrics['stability_moderately_stable_features_count'] = safe_get(stab_an, 'stability_categories.moderately_stable.count')
            metrics['stability_unstable_features_count'] = safe_get(stab_an, 'stability_categories.unstable.count')

        # --- Regularization Path Information (related to selected alpha) ---
        if hasattr(self, 'regularization_path_') and self.regularization_path_ and not self.regularization_path_.get('error'):
            reg_path = self.regularization_path_
            metrics['reg_path_n_features_at_selected_alpha'] = safe_get(reg_path, 'current_n_features')


        # Remove NaN or None values for cleaner output
        metrics = {k: v for k, v in metrics.items() if pd.notna(v) and not (isinstance(v, float) and np.isinf(v))}

        # Convert numpy types to native python types for broader compatibility (e.g., JSON serialization)
        for key, value in metrics.items():
            if isinstance(value, (np.integer, np.int_)):
                metrics[key] = int(value)
            elif isinstance(value, (np.floating, np.float_)):
                metrics[key] = float(value)
            elif isinstance(value, np.bool_):
                metrics[key] = bool(value)

        return metrics

# Factory function
def get_plugin():
    """Factory function to get the plugin instance"""
    return LassoRegressionPlugin()


# Example usage and testing
if __name__ == "__main__":
    """
    Example usage of Lasso Regression Plugin
    """
    print("Testing Lasso Regression Plugin...")
    
    try:
        # Create sample high-dimensional regression data
        from sklearn.datasets import make_regression
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import mean_squared_error, r2_score
        
        # Generate high-dimensional dataset with some irrelevant features
        X, y = make_regression(
            n_samples=100,
            n_features=50,
            n_informative=10,  # Only 10 truly relevant features
            n_redundant=5,
            noise=0.1,
            random_state=42
        )
        
        # Create feature names
        feature_names = [f"feature_{i}" for i in range(X.shape[1])]
        X_df = pd.DataFrame(X, columns=feature_names)
        
        print(f"\n📊 High-Dimensional Dataset Info:")
        print(f"Shape: {X_df.shape}")
        print(f"Informative features: 10 out of {X.shape[1]}")
        print(f"Target range: [{y.min():.2f}, {y.max():.2f}]")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_df, y, test_size=0.2, random_state=42
        )
        
        # Create and test Lasso Regression plugin
        plugin = LassoRegressionPlugin(
            auto_alpha=True,
            alpha_selection_method='cv',
            cv_folds=5,
            normalize_features=True,
            analyze_regularization_path=True,
            analyze_feature_stability=True,
            compute_feature_importance=True,
            sparsity_analysis=True,
            random_state=42
        )
        
        print("\n🔍 Plugin Info:")
        print(f"Name: {plugin.get_name()}")
        print(f"Category: {plugin.get_category()}")
        print(f"Description: {plugin.get_description()}")
        
        # Check compatibility
        compatible, message = plugin.is_compatible_with_data(X_train, y_train)
        print(f"\n✅ Compatibility: {message}")
        
        if compatible:
            # Train Lasso model
            print("\n🚀 Training Lasso Regression with automatic alpha selection...")
            plugin.fit(X_train, y_train)
            
            # Make predictions
            y_pred = plugin.predict(X_test)
            
            # Evaluate performance
            mse = mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            r2 = r2_score(y_test, y_pred)
            
            print(f"\n📊 Lasso Regression Results:")
            print(f"MSE: {mse:.4f}")
            print(f"RMSE: {rmse:.4f}")
            print(f"R²: {r2:.4f}")
            
            # Get model parameters
            model_params = plugin.get_model_params()
            print(f"\n⚙️ Model Configuration:")
            print(f"Alpha used: {model_params['alpha_used']:.6f}")
            print(f"Selected features: {model_params['selected_features']}")
            print(f"Eliminated features: {model_params['eliminated_features']}")
            print(f"Sparsity ratio: {model_params['sparsity_ratio']:.2%}")
            
            # Get feature importance
            importance = plugin.get_feature_importance()
            print(f"\n🎯 Feature Selection Results:")
            print(f"Features selected: {importance['selected_features']['count']}")
            print(f"Features eliminated: {importance['eliminated_features']['count']}")
            print(f"Elimination rate: {importance['sparsity_info']['elimination_rate']:.1%}")
            
            if importance['selected_features']['count'] > 0:
                print(f"\nTop selected features:")
                for i, name in enumerate(importance['selected_features']['names'][:5]):
                    coef = importance['selected_features']['coefficients'][i]
                    print(f"  {name}: {coef:.4f}")
            
            print("\n✅ Lasso Regression Plugin test completed successfully!")
            
    except Exception as e:
        print(f"\n❌ Error testing Lasso Regression Plugin: {str(e)}")
        import traceback
        traceback.print_exc()