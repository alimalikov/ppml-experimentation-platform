import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, Optional, Tuple, List
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_X_y, check_array
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import ElasticNet, ElasticNetCV, enet_path
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


class ElasticNetRegressionPlugin(BaseEstimator, RegressorMixin, MLPlugin):
    """
    Elastic Net Regression Plugin - L1 + L2 Regularization for Optimal Feature Selection
    
    This plugin implements comprehensive Elastic Net regression that combines the benefits of
    both Ridge (L2) and Lasso (L1) regularization. Perfect for datasets with correlated features
    where you want both feature selection and grouped feature handling.
    
    Key Features:
    - Combined L1 + L2 regularization
    - Handles correlated features better than pure Lasso
    - Automatic alpha and l1_ratio optimization
    - Regularization path analysis for both parameters
    - Feature selection with stability analysis
    - Cross-validation for hyperparameter tuning
    - Comprehensive regularization diagnostics
    - Advanced coefficient stability analysis
    """
    
    def __init__(
        self,
        alpha=1.0,
        l1_ratio=0.5,
        fit_intercept=True,
        normalize_features=True,
        precompute=False,
        copy_X=True,
        max_iter=1000,
        tol=1e-4,
        warm_start=False,
        positive=False,
        selection='cyclic',
        
        # Advanced optimization options
        auto_alpha=True,
        auto_l1_ratio=True,
        alpha_selection_method='cv',
        cv_folds=5,
        n_alphas=100,
        n_l1_ratios=10,
        alpha_min_ratio=1e-4,
        l1_ratio_range=(0.1, 0.9),
        eps=1e-3,
        
        # Analysis options
        analyze_regularization_path=True,
        analyze_feature_stability=True,
        compute_feature_importance=True,
        regularization_analysis=True,
        l1_l2_balance_analysis=True,
        
        random_state=42
    ):
        super().__init__()
        
        # Core Elastic Net parameters
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.fit_intercept = fit_intercept
        self.normalize_features = normalize_features
        self.precompute = precompute
        self.copy_X = copy_X
        self.max_iter = max_iter
        self.tol = tol
        self.warm_start = warm_start
        self.positive = positive
        self.selection = selection
        
        # Advanced optimization
        self.auto_alpha = auto_alpha
        self.auto_l1_ratio = auto_l1_ratio
        self.alpha_selection_method = alpha_selection_method
        self.cv_folds = cv_folds
        self.n_alphas = n_alphas
        self.n_l1_ratios = n_l1_ratios
        self.alpha_min_ratio = alpha_min_ratio
        self.l1_ratio_range = l1_ratio_range
        self.eps = eps
        
        # Analysis options
        self.analyze_regularization_path = analyze_regularization_path
        self.analyze_feature_stability = analyze_feature_stability
        self.compute_feature_importance = compute_feature_importance
        self.regularization_analysis = regularization_analysis
        self.l1_l2_balance_analysis = l1_l2_balance_analysis
        
        self.random_state = random_state
        
        # Required plugin metadata
        self._name = "Elastic Net Regression"
        self._description = "L1+L2 regularized regression combining feature selection with correlated feature handling"
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
        self.hyperparameter_analysis_ = {}
        self.regularization_path_ = {}
        self.feature_selection_analysis_ = {}
        self.regularization_balance_ = {}
        self.stability_analysis_ = {}
    
    def get_name(self) -> str:
        return self._name
    
    def get_description(self) -> str:
        return self._description
    
    def get_category(self) -> str:
        return self._category
    
    def fit(self, X, y, sample_weight=None):
        """
        Fit the Elastic Net Regression model with comprehensive analysis
        
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
        
        # Determine optimal hyperparameters if auto-optimization is enabled
        if self.auto_alpha or self.auto_l1_ratio:
            optimal_alpha, optimal_l1_ratio = self._find_optimal_hyperparameters(
                X_processed, y, sample_weight
            )
            self.alpha_used_ = optimal_alpha
            self.l1_ratio_used_ = optimal_l1_ratio
        else:
            self.alpha_used_ = self.alpha
            self.l1_ratio_used_ = self.l1_ratio
        
        # Fit the Elastic Net regression model
        self.model_ = ElasticNet(
            alpha=self.alpha_used_,
            l1_ratio=self.l1_ratio_used_,
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
        self._analyze_regularization_path_comprehensive()
        self._analyze_feature_selection()
        self._analyze_regularization_balance()
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
    
    def _find_optimal_hyperparameters(self, X, y, sample_weight=None):
        """Find optimal alpha and l1_ratio using cross-validation or grid search"""
        if self.alpha_selection_method == 'cv':
            return self._cv_hyperparameter_search(X, y, sample_weight)
        elif self.alpha_selection_method == 'grid':
            return self._grid_hyperparameter_search(X, y, sample_weight)
        else:
            # ElasticNetCV for automatic selection
            return self._elasticnet_cv_search(X, y, sample_weight)
    
    def _elasticnet_cv_search(self, X, y, sample_weight=None):
        """Use ElasticNetCV for automatic hyperparameter selection"""
        # Generate alpha values
        alphas = np.logspace(
            np.log10(self.alpha_min_ratio),
            np.log10(1.0),
            num=self.n_alphas
        )
        
        # Generate l1_ratio values if auto_l1_ratio is enabled
        if self.auto_l1_ratio:
            l1_ratios = np.linspace(
                self.l1_ratio_range[0],
                self.l1_ratio_range[1],
                num=self.n_l1_ratios
            )
        else:
            l1_ratios = [self.l1_ratio]
        
        # Use ElasticNetCV
        elastic_cv = ElasticNetCV(
            l1_ratio=l1_ratios,
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
        
        elastic_cv.fit(X, y, sample_weight=sample_weight)
        
        # Store comprehensive CV results
        self.hyperparameter_analysis_ = {
            'method': 'ElasticNetCV',
            'alphas_tested': alphas,
            'l1_ratios_tested': l1_ratios,
            'cv_scores': elastic_cv.mse_path_,
            'optimal_alpha': elastic_cv.alpha_,
            'optimal_l1_ratio': elastic_cv.l1_ratio_,
            'cv_folds': self.cv_folds,
            'best_score': np.min(np.mean(elastic_cv.mse_path_, axis=-1)),
            'alpha_path': elastic_cv.alphas_,
            'l1_ratio_path': elastic_cv.l1_ratio
        }
        
        return elastic_cv.alpha_, elastic_cv.l1_ratio_
    
    def _grid_hyperparameter_search(self, X, y, sample_weight=None):
        """Manual grid search for hyperparameters"""
        alphas = np.logspace(
            np.log10(self.alpha_min_ratio),
            np.log10(1.0),
            num=self.n_alphas
        )
        
        if self.auto_l1_ratio:
            l1_ratios = np.linspace(
                self.l1_ratio_range[0],
                self.l1_ratio_range[1],
                num=self.n_l1_ratios
            )
        else:
            l1_ratios = [self.l1_ratio]
        
        best_score = float('inf')
        best_alpha = self.alpha
        best_l1_ratio = self.l1_ratio
        cv_results = []
        
        for alpha in alphas:
            for l1_ratio in l1_ratios:
                elastic = ElasticNet(
                    alpha=alpha,
                    l1_ratio=l1_ratio,
                    fit_intercept=self.fit_intercept,
                    max_iter=self.max_iter,
                    tol=self.tol,
                    random_state=self.random_state
                )
                
                # Cross-validation
                cv_scores = cross_val_score(
                    elastic, X, y,
                    cv=self.cv_folds,
                    scoring='neg_mean_squared_error',
                    fit_params={'sample_weight': sample_weight} if sample_weight is not None else None
                )
                
                mean_score = -np.mean(cv_scores)
                cv_results.append({
                    'alpha': alpha,
                    'l1_ratio': l1_ratio,
                    'cv_score': mean_score,
                    'cv_std': np.std(cv_scores)
                })
                
                if mean_score < best_score:
                    best_score = mean_score
                    best_alpha = alpha
                    best_l1_ratio = l1_ratio
        
        # Store grid search results
        self.hyperparameter_analysis_ = {
            'method': 'Grid Search',
            'alphas_tested': alphas,
            'l1_ratios_tested': l1_ratios,
            'cv_results': cv_results,
            'optimal_alpha': best_alpha,
            'optimal_l1_ratio': best_l1_ratio,
            'best_score': best_score,
            'cv_folds': self.cv_folds
        }
        
        return best_alpha, best_l1_ratio
    
    def _cv_hyperparameter_search(self, X, y, sample_weight=None):
        """Cross-validation based hyperparameter search"""
        return self._elasticnet_cv_search(X, y, sample_weight)
    
    def _analyze_regularization_path_comprehensive(self):
        """Analyze the complete regularization path for both alpha and l1_ratio"""
        if not self.analyze_regularization_path:
            return
        
        try:
            # Generate alpha values for path analysis
            alphas = np.logspace(
                np.log10(self.alpha_min_ratio),
                np.log10(10.0),
                num=min(50, self.n_alphas)
            )
            
            # Compute regularization path for current l1_ratio
            alphas_path, coefs_path, _ = enet_path(
                self.X_processed_,
                self.y_original_,
                l1_ratio=self.l1_ratio_used_,
                alphas=alphas,
                fit_intercept=self.fit_intercept,
                eps=self.eps,
                copy_X=True
            )
            
            # Analyze path characteristics
            n_features_path = np.sum(np.abs(coefs_path) > 1e-8, axis=0)
            max_coef_path = np.max(np.abs(coefs_path), axis=0)
            
            # Feature entry points
            feature_entry_alphas = []
            for i in range(coefs_path.shape[0]):
                nonzero_indices = np.where(np.abs(coefs_path[i, :]) > 1e-8)[0]
                if len(nonzero_indices) > 0:
                    entry_alpha = alphas_path[nonzero_indices[-1]]
                    feature_entry_alphas.append(entry_alpha)
                else:
                    feature_entry_alphas.append(np.inf)
            
            # Analyze different l1_ratios if enabled
            l1_ratio_comparison = None
            if self.l1_l2_balance_analysis:
                l1_ratio_comparison = self._analyze_l1_ratio_effects(alphas_path)
            
            # Store comprehensive regularization path analysis
            self.regularization_path_ = {
                'alphas': alphas_path,
                'coefficients': coefs_path,
                'n_features_active': n_features_path,
                'max_coefficient_magnitude': max_coef_path,
                'feature_entry_alphas': np.array(feature_entry_alphas),
                'feature_names': self.feature_names_,
                'current_alpha': self.alpha_used_,
                'current_l1_ratio': self.l1_ratio_used_,
                'current_n_features': np.sum(np.abs(self.model_.coef_) > 1e-8),
                'l1_ratio_comparison': l1_ratio_comparison
            }
            
        except Exception as e:
            self.regularization_path_ = {
                'error': f'Could not compute regularization path: {str(e)}'
            }
    
    def _analyze_l1_ratio_effects(self, alphas):
        """Analyze the effects of different l1_ratio values"""
        try:
            l1_ratios = [0.1, 0.3, 0.5, 0.7, 0.9]
            l1_ratio_results = {}
            
            for l1_ratio in l1_ratios:
                try:
                    _, coefs, _ = enet_path(
                        self.X_processed_,
                        self.y_original_,
                        l1_ratio=l1_ratio,
                        alphas=alphas,
                        fit_intercept=self.fit_intercept,
                        eps=self.eps,
                        copy_X=True
                    )
                    
                    # Analyze sparsity at current alpha
                    alpha_idx = np.argmin(np.abs(alphas - self.alpha_used_))
                    current_coefs = coefs[:, alpha_idx]
                    n_nonzero = np.sum(np.abs(current_coefs) > 1e-8)
                    sparsity = 1 - (n_nonzero / len(current_coefs))
                    
                    l1_ratio_results[l1_ratio] = {
                        'coefficients_at_current_alpha': current_coefs,
                        'n_nonzero_features': n_nonzero,
                        'sparsity': sparsity,
                        'max_coef_magnitude': np.max(np.abs(current_coefs)),
                        'coef_path': coefs
                    }
                    
                except Exception:
                    continue
            
            return l1_ratio_results
            
        except Exception:
            return None
    
    def _analyze_feature_selection(self):
        """Analyze feature selection characteristics"""
        if not self.compute_feature_importance:
            return
        
        coefficients = self.model_.coef_
        
        # Identify selected features
        selected_mask = np.abs(coefficients) > 1e-8
        selected_features = np.where(selected_mask)[0]
        eliminated_features = np.where(~selected_mask)[0]
        
        # Calculate feature importance
        importance_scores = np.abs(coefficients)
        
        # Normalize importance scores
        if np.sum(importance_scores) > 0:
            normalized_importance = importance_scores / np.sum(importance_scores)
        else:
            normalized_importance = importance_scores
        
        # Rank features by importance
        feature_ranking = np.argsort(importance_scores)[::-1]
        
        # Analyze regularization effect on features
        regularization_effect = self._analyze_regularization_effect_on_features()
        
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
            'regularization_effect': regularization_effect
        }
    
    def _analyze_regularization_effect_on_features(self):
        """Analyze how L1 and L2 regularization affects different features"""
        if 'l1_ratio_comparison' not in self.regularization_path_ or self.regularization_path_['l1_ratio_comparison'] is None:
            return None
        
        try:
            l1_ratio_comparison = self.regularization_path_['l1_ratio_comparison']
            current_coefs = self.model_.coef_
            
            # Analyze feature behavior across different l1_ratios
            feature_behavior = {}
            
            for i, feature_name in enumerate(self.feature_names_):
                behavior_data = {
                    'current_coefficient': current_coefs[i],
                    'l1_ratio_effects': {}
                }
                
                for l1_ratio, data in l1_ratio_comparison.items():
                    coef_at_ratio = data['coefficients_at_current_alpha'][i]
                    behavior_data['l1_ratio_effects'][l1_ratio] = coef_at_ratio
                
                # Categorize feature behavior
                coef_values = list(behavior_data['l1_ratio_effects'].values())
                coef_values = [c for c in coef_values if abs(c) > 1e-8]
                
                if len(coef_values) == 0:
                    category = "consistently_eliminated"
                elif len(coef_values) == len(behavior_data['l1_ratio_effects']):
                    category = "consistently_selected"
                elif abs(current_coefs[i]) > 1e-8:
                    category = "selected_with_regularization"
                else:
                    category = "eliminated_with_regularization"
                
                behavior_data['category'] = category
                feature_behavior[feature_name] = behavior_data
            
            return feature_behavior
            
        except Exception:
            return None
    
    def _analyze_regularization_balance(self):
        """Analyze the balance between L1 and L2 regularization"""
        if not self.regularization_analysis:
            return
        
        # Calculate effective L1 and L2 contributions
        l1_contribution = self.alpha_used_ * self.l1_ratio_used_
        l2_contribution = self.alpha_used_ * (1 - self.l1_ratio_used_)
        
        # Analyze coefficient characteristics
        coefficients = self.model_.coef_
        nonzero_coefs = coefficients[np.abs(coefficients) > 1e-8]
        
        # Sparsity analysis
        total_features = len(coefficients)
        selected_features = len(nonzero_coefs)
        sparsity_ratio = (total_features - selected_features) / total_features
        
        # Coefficient magnitude analysis
        if len(nonzero_coefs) > 0:
            coef_stats = {
                'mean_magnitude': np.mean(np.abs(nonzero_coefs)),
                'std_magnitude': np.std(np.abs(nonzero_coefs)),
                'max_magnitude': np.max(np.abs(nonzero_coefs)),
                'min_magnitude': np.min(np.abs(nonzero_coefs)),
                'coefficient_range': np.max(np.abs(nonzero_coefs)) - np.min(np.abs(nonzero_coefs))
            }
        else:
            coef_stats = {
                'mean_magnitude': 0, 'std_magnitude': 0, 'max_magnitude': 0,
                'min_magnitude': 0, 'coefficient_range': 0
            }
        
        # Regularization interpretation
        regularization_interpretation = self._interpret_regularization_balance()
        
        self.regularization_balance_ = {
            'hyperparameters': {
                'alpha': self.alpha_used_,
                'l1_ratio': self.l1_ratio_used_,
                'l1_contribution': l1_contribution,
                'l2_contribution': l2_contribution
            },
            'sparsity_metrics': {
                'total_features': total_features,
                'selected_features': selected_features,
                'sparsity_ratio': sparsity_ratio,
                'density_ratio': 1 - sparsity_ratio
            },
            'coefficient_statistics': coef_stats,
            'regularization_interpretation': regularization_interpretation,
            'balance_analysis': self._analyze_l1_l2_balance()
        }
    
    def _interpret_regularization_balance(self):
        """Interpret the current regularization balance"""
        l1_ratio = self.l1_ratio_used_
        alpha = self.alpha_used_
        
        # L1/L2 ratio interpretation
        if l1_ratio >= 0.8:
            ratio_desc = "Lasso-dominant (strong feature selection)"
        elif l1_ratio >= 0.6:
            ratio_desc = "L1-heavy (moderate feature selection)"
        elif l1_ratio >= 0.4:
            ratio_desc = "Balanced L1/L2 (feature selection + grouping)"
        elif l1_ratio >= 0.2:
            ratio_desc = "L2-heavy (feature grouping dominant)"
        else:
            ratio_desc = "Ridge-dominant (minimal feature selection)"
        
        # Alpha strength interpretation
        if alpha >= 1.0:
            alpha_desc = "Strong regularization"
        elif alpha >= 0.1:
            alpha_desc = "Moderate regularization"
        elif alpha >= 0.01:
            alpha_desc = "Weak regularization"
        else:
            alpha_desc = "Very weak regularization"
        
        return {
            'l1_ratio_description': ratio_desc,
            'alpha_description': alpha_desc,
            'overall_strategy': f"{alpha_desc} with {ratio_desc.lower()}"
        }
    
    def _analyze_l1_l2_balance(self):
        """Analyze the specific balance between L1 and L2 effects"""
        l1_ratio = self.l1_ratio_used_
        l2_ratio = 1 - l1_ratio
        
        # Calculate relative contributions
        l1_weight = l1_ratio
        l2_weight = l2_ratio
        
        # Expected effects
        expected_effects = {
            'feature_selection_strength': l1_weight,
            'feature_grouping_strength': l2_weight,
            'sparsity_tendency': l1_weight,
            'coefficient_shrinkage': l2_weight,
            'correlated_feature_handling': l2_weight
        }
        
        # Balance categorization
        if abs(l1_ratio - 0.5) <= 0.1:
            balance_category = "well_balanced"
        elif l1_ratio > 0.6:
            balance_category = "l1_dominant"
        else:
            balance_category = "l2_dominant"
        
        return {
            'l1_weight': l1_weight,
            'l2_weight': l2_weight,
            'expected_effects': expected_effects,
            'balance_category': balance_category,
            'recommended_use_case': self._get_recommended_use_case(balance_category)
        }
    
    def _get_recommended_use_case(self, balance_category):
        """Get recommended use case based on regularization balance"""
        use_cases = {
            'well_balanced': "Datasets with moderate feature correlation and need for both selection and grouping",
            'l1_dominant': "High-dimensional data with many irrelevant features requiring aggressive selection",
            'l2_dominant': "Datasets with highly correlated features requiring grouped selection"
        }
        return use_cases.get(balance_category, "General purpose regularized regression")
    
    def _analyze_feature_stability(self):
        """Analyze stability of feature selection across regularization parameters"""
        if not self.analyze_feature_stability:
            return
        
        if 'l1_ratio_comparison' not in self.regularization_path_ or self.regularization_path_['l1_ratio_comparison'] is None:
            self.stability_analysis_ = {
                'status': 'L1 ratio comparison not available for stability analysis'
            }
            return
        
        try:
            l1_ratio_comparison = self.regularization_path_['l1_ratio_comparison']
            
            # Calculate feature selection stability across l1_ratios
            stability_scores = []
            feature_selections = {}
            
            for l1_ratio, data in l1_ratio_comparison.items():
                coefs = data['coefficients_at_current_alpha']
                selected = np.abs(coefs) > 1e-8
                feature_selections[l1_ratio] = selected
            
            # Calculate stability for each feature
            feature_stability = {}
            for i, feature_name in enumerate(self.feature_names_):
                selections = [feature_selections[lr][i] for lr in feature_selections.keys()]
                stability_score = np.mean(selections)  # Proportion of times selected
                
                feature_stability[feature_name] = {
                    'stability_score': stability_score,
                    'selection_frequency': sum(selections),
                    'total_tests': len(selections),
                    'category': self._categorize_feature_stability(stability_score)
                }
            
            # Overall stability metrics
            all_stability_scores = [data['stability_score'] for data in feature_stability.values()]
            overall_stability = np.mean(all_stability_scores)
            
            self.stability_analysis_ = {
                'feature_stability': feature_stability,
                'overall_stability': overall_stability,
                'stability_interpretation': self._interpret_overall_stability(overall_stability),
                'l1_ratios_tested': list(feature_selections.keys()),
                'stability_distribution': {
                    'highly_stable': sum(1 for s in all_stability_scores if s >= 0.8),
                    'moderately_stable': sum(1 for s in all_stability_scores if 0.4 <= s < 0.8),
                    'unstable': sum(1 for s in all_stability_scores if s < 0.4)
                }
            }
            
        except Exception as e:
            self.stability_analysis_ = {
                'error': f'Could not perform stability analysis: {str(e)}'
            }
    
    def _categorize_feature_stability(self, stability_score):
        """Categorize individual feature stability"""
        if stability_score >= 0.8:
            return "highly_stable"
        elif stability_score >= 0.4:
            return "moderately_stable"
        else:
            return "unstable"
    
    def _interpret_overall_stability(self, stability_score):
        """Interpret overall stability score"""
        if stability_score >= 0.8:
            return "Very stable - features consistently selected across regularization settings"
        elif stability_score >= 0.6:
            return "Good stability - most features consistently selected"
        elif stability_score >= 0.4:
            return "Moderate stability - some variation in feature selection"
        else:
            return "Poor stability - high variation in feature selection"
    
    def get_hyperparameter_config(self, key_prefix: str) -> Dict[str, Any]:
        """Generate Streamlit UI for hyperparameters"""
        
        # Create tabs for different configuration aspects
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "Core Parameters", "Regularization", "Optimization", "Advanced Options", "Algorithm Info"
        ])
        
        with tab1:
            st.markdown("**Elastic Net Regression Configuration**")
            
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
                    help="Apply StandardScaler to features (recommended for Elastic Net)",
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
                    help="Algorithm for coordinate descent updates",
                    key=f"{key_prefix}_selection"
                )
        
        with tab2:
            st.markdown("**Regularization Configuration**")
            
            # Auto-optimization settings
            col1, col2 = st.columns(2)
            
            with col1:
                auto_alpha = st.checkbox(
                    "Automatic Alpha Selection",
                    value=self.auto_alpha,
                    help="Automatically find optimal alpha using cross-validation",
                    key=f"{key_prefix}_auto_alpha"
                )
                
                auto_l1_ratio = st.checkbox(
                    "Automatic L1 Ratio Selection",
                    value=self.auto_l1_ratio,
                    help="Automatically find optimal L1/L2 balance",
                    key=f"{key_prefix}_auto_l1_ratio"
                )
            
            with col2:
                if auto_alpha or auto_l1_ratio:
                    alpha_selection_method = st.selectbox(
                        "Optimization Method:",
                        options=['cv', 'grid'],
                        index=['cv', 'grid'].index(self.alpha_selection_method),
                        help="Method for hyperparameter optimization",
                        key=f"{key_prefix}_alpha_selection_method"
                    )
                else:
                    alpha_selection_method = self.alpha_selection_method
            
            # Manual parameter setting
            if not auto_alpha:
                alpha = st.number_input(
                    "Alpha (Regularization Strength):",
                    value=self.alpha,
                    min_value=1e-6,
                    max_value=10.0,
                    step=0.01,
                    format="%.4f",
                    help="Overall regularization parameter",
                    key=f"{key_prefix}_alpha"
                )
            else:
                alpha = self.alpha
            
            if not auto_l1_ratio:
                l1_ratio = st.slider(
                    "L1 Ratio (L1/(L1+L2)):",
                    min_value=0.0,
                    max_value=1.0,
                    value=self.l1_ratio,
                    step=0.05,
                    help="Balance between L1 (Lasso) and L2 (Ridge) regularization",
                    key=f"{key_prefix}_l1_ratio"
                )
                
                # L1 ratio interpretation
                if l1_ratio >= 0.8:
                    ratio_desc = "🔴 Lasso-dominant (strong feature selection)"
                elif l1_ratio >= 0.6:
                    ratio_desc = "🟡 L1-heavy (moderate feature selection)"
                elif l1_ratio >= 0.4:
                    ratio_desc = "🟢 Balanced (selection + grouping)"
                elif l1_ratio >= 0.2:
                    ratio_desc = "🔵 L2-heavy (feature grouping)"
                else:
                    ratio_desc = "🟣 Ridge-dominant (minimal selection)"
                
                st.info(f"{ratio_desc}")
                
            else:
                l1_ratio = self.l1_ratio
            
            # Regularization visualization
            if not auto_alpha and not auto_l1_ratio:
                l1_contribution = alpha * l1_ratio
                l2_contribution = alpha * (1 - l1_ratio)
                
                st.markdown("**Regularization Breakdown:**")
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("L1 Contribution", f"{l1_contribution:.4f}")
                with col2:
                    st.metric("L2 Contribution", f"{l2_contribution:.4f}")
        
        with tab3:
            st.markdown("**Optimization Configuration**")
            
            if auto_alpha or auto_l1_ratio:
                col1, col2 = st.columns(2)
                
                with col1:
                    cv_folds = st.number_input(
                        "CV Folds:",
                        value=self.cv_folds,
                        min_value=3,
                        max_value=10,
                        step=1,
                        help="Number of cross-validation folds",
                        key=f"{key_prefix}_cv_folds"
                    )
                    
                    n_alphas = st.number_input(
                        "Number of Alphas:",
                        value=self.n_alphas,
                        min_value=10,
                        max_value=200,
                        step=10,
                        help="Number of alpha values to test",
                        key=f"{key_prefix}_n_alphas"
                    )
                
                with col2:
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
                    
                    if auto_l1_ratio:
                        n_l1_ratios = st.number_input(
                            "Number of L1 Ratios:",
                            value=self.n_l1_ratios,
                            min_value=5,
                            max_value=20,
                            step=1,
                            help="Number of L1 ratio values to test",
                            key=f"{key_prefix}_n_l1_ratios"
                        )
                    else:
                        n_l1_ratios = self.n_l1_ratios
                
                if auto_l1_ratio:
                    st.markdown("**L1 Ratio Search Range:**")
                    l1_ratio_range = st.slider(
                        "L1 Ratio Range:",
                        min_value=0.0,
                        max_value=1.0,
                        value=self.l1_ratio_range,
                        step=0.05,
                        help="Range of L1 ratios to search",
                        key=f"{key_prefix}_l1_ratio_range"
                    )
                else:
                    l1_ratio_range = self.l1_ratio_range
                    n_l1_ratios = self.n_l1_ratios
                    
            else:
                # Default values when not auto-optimizing
                cv_folds = self.cv_folds
                n_alphas = self.n_alphas
                n_l1_ratios = self.n_l1_ratios
                alpha_min_ratio = self.alpha_min_ratio
                l1_ratio_range = self.l1_ratio_range
        
        with tab4:
            st.markdown("**Advanced Configuration**")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Analysis Options:**")
                
                analyze_regularization_path = st.checkbox(
                    "Regularization Path Analysis",
                    value=self.analyze_regularization_path,
                    help="Analyze coefficient paths across regularization parameters",
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
                
                regularization_analysis = st.checkbox(
                    "Regularization Analysis",
                    value=self.regularization_analysis,
                    help="Detailed analysis of regularization effects",
                    key=f"{key_prefix}_regularization_analysis"
                )
                
                l1_l2_balance_analysis = st.checkbox(
                    "L1/L2 Balance Analysis",
                    value=self.l1_l2_balance_analysis,
                    help="Analyze balance between L1 and L2 regularization",
                    key=f"{key_prefix}_l1_l2_balance_analysis"
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
        
        with tab5:
            st.markdown("**Algorithm Information**")
            
            st.info("""
            **Elastic Net Regression** - Best of Both Worlds (L1 + L2):
            • 🎯 Combines Lasso (L1) and Ridge (L2) regularization
            • 🔄 Handles correlated features better than pure Lasso
            • 📊 Automatic feature selection + feature grouping
            • ⚡ Cross-validated hyperparameter selection
            • 📈 Comprehensive regularization analysis
            • 🎪 Perfect for real-world datasets
            
            **Mathematical Foundation:**
            • Objective: ||y - Xβ||² + α(ρ||β||₁ + (1-ρ)||β||₂²)
            • ρ = l1_ratio controls L1/L2 balance
            • α = alpha controls overall regularization strength
            • Convex optimization with elastic penalty
            """)
            
            # When to use Elastic Net
            if st.button("🎯 When to Use Elastic Net", key=f"{key_prefix}_when_to_use"):
                st.markdown("""
                **Ideal Use Cases:**
                
                **Problem Characteristics:**
                • High-dimensional data with correlated features
                • Need both feature selection AND grouping
                • Real-world messy datasets
                • Feature multicollinearity present
                
                **Data Characteristics:**
                • Groups of correlated features
                • Mix of relevant and irrelevant features
                • Linear relationships with noise
                • Medium to high feature count
                
                **Business Requirements:**
                • Interpretable feature selection
                • Robust to feature correlation
                • Balanced bias-variance tradeoff
                • Stable feature selection
                
                **Examples:**
                • Genomics (gene expression data)
                • Finance (economic indicators)
                • Marketing (customer features)
                • Text analysis (word features)
                """)
            
            # Advantages and limitations
            if st.button("⚖️ Advantages & Limitations", key=f"{key_prefix}_pros_cons"):
                st.markdown("""
                **Advantages:**
                ✅ Best of Lasso + Ridge regularization
                ✅ Handles correlated features gracefully
                ✅ Automatic feature selection with grouping
                ✅ More stable than pure Lasso
                ✅ Cross-validation for hyperparameters
                ✅ Works well with real-world data
                ✅ Reduces overfitting effectively
                ✅ Interpretable results
                
                **Limitations:**
                ❌ More hyperparameters to tune (α, ρ)
                ❌ Computationally more expensive than Ridge/Lasso
                ❌ Still assumes linear relationships
                ❌ Feature scaling remains important
                ❌ May be overkill for simple datasets
                ❌ Hyperparameter selection can be complex
                """)
            
            # L1 vs L2 vs Elastic Net comparison
            if st.button("🔧 L1 vs L2 vs Elastic Net", key=f"{key_prefix}_comparison"):
                st.markdown("""
                **Regularization Method Comparison:**
                
                **Pure L1 (Lasso) - l1_ratio = 1.0:**
                • Strong feature selection (sparse solutions)
                • May arbitrarily choose among correlated features
                • Can eliminate entire groups of correlated features
                • Good for high-dimensional, sparse problems
                
                **Pure L2 (Ridge) - l1_ratio = 0.0:**
                • Shrinks coefficients toward zero
                • Keeps all features (no selection)
                • Handles correlated features by averaging
                • Good for many relevant features
                
                **Elastic Net - 0 < l1_ratio < 1:**
                • **l1_ratio = 0.1-0.3:** Ridge-heavy (grouping dominant)
                • **l1_ratio = 0.4-0.6:** Balanced approach
                • **l1_ratio = 0.7-0.9:** Lasso-heavy (selection dominant)
                • Combines benefits of both methods
                • Most robust for real-world data
                """)
            
            # Best practices
            if st.button("🎯 Best Practices", key=f"{key_prefix}_best_practices"):
                st.markdown("""
                **Elastic Net Best Practices:**
                
                **Data Preparation:**
                1. **Always standardize features** (critical for both L1 and L2)
                2. Check correlation structure of features
                3. Remove constant and near-constant features
                4. Consider feature engineering for interactions
                
                **Hyperparameter Selection:**
                1. Use cross-validation for both α and l1_ratio
                2. Start with balanced l1_ratio (0.5)
                3. Grid search over reasonable ranges
                4. Consider 1-SE rule for simpler models
                
                **Model Interpretation:**
                1. Examine selected vs eliminated features
                2. Check feature groups that are selected together
                3. Analyze regularization path for insights
                4. Validate feature relevance domain knowledge
                
                **Validation:**
                1. Use nested cross-validation for unbiased estimates
                2. Check stability across different random seeds
                3. Compare with pure Lasso and Ridge
                4. Validate on independent test set
                """)
        
        return {
            "alpha": alpha,
            "l1_ratio": l1_ratio,
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
            "auto_l1_ratio": auto_l1_ratio,
            "alpha_selection_method": alpha_selection_method,
            "cv_folds": cv_folds,
            "n_alphas": n_alphas,
            "n_l1_ratios": n_l1_ratios,
            "alpha_min_ratio": alpha_min_ratio,
            "l1_ratio_range": l1_ratio_range,
            "analyze_regularization_path": analyze_regularization_path,
            "analyze_feature_stability": analyze_feature_stability,
            "compute_feature_importance": compute_feature_importance,
            "regularization_analysis": regularization_analysis,
            "l1_l2_balance_analysis": l1_l2_balance_analysis,
            "random_state": random_state
        }
    
    def create_model_instance(self, hyperparameters: Dict[str, Any]):
        """Create model instance with given hyperparameters"""
        return ElasticNetRegressionPlugin(
            alpha=hyperparameters.get("alpha", self.alpha),
            l1_ratio=hyperparameters.get("l1_ratio", self.l1_ratio),
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
            auto_l1_ratio=hyperparameters.get("auto_l1_ratio", self.auto_l1_ratio),
            alpha_selection_method=hyperparameters.get("alpha_selection_method", self.alpha_selection_method),
            cv_folds=hyperparameters.get("cv_folds", self.cv_folds),
            n_alphas=hyperparameters.get("n_alphas", self.n_alphas),
            n_l1_ratios=hyperparameters.get("n_l1_ratios", self.n_l1_ratios),
            alpha_min_ratio=hyperparameters.get("alpha_min_ratio", self.alpha_min_ratio),
            l1_ratio_range=hyperparameters.get("l1_ratio_range", self.l1_ratio_range),
            analyze_regularization_path=hyperparameters.get("analyze_regularization_path", self.analyze_regularization_path),
            analyze_feature_stability=hyperparameters.get("analyze_feature_stability", self.analyze_feature_stability),
            compute_feature_importance=hyperparameters.get("compute_feature_importance", self.compute_feature_importance),
            regularization_analysis=hyperparameters.get("regularization_analysis", self.regularization_analysis),
            l1_l2_balance_analysis=hyperparameters.get("l1_l2_balance_analysis", self.l1_l2_balance_analysis),
            random_state=hyperparameters.get("random_state", self.random_state)
        )
    
    def preprocess_data(self, X, y=None, training=True):
        """Preprocess data for Elastic Net Regression"""
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
        """Check if Elastic Net Regression is compatible with the given data"""
        # Check minimum samples
        if X.shape[0] < self._min_samples_required:
            return False, f"Elastic Net regression requires at least {self._min_samples_required} samples, got {X.shape[0]}"
        
        # Check for regression targets
        if y is not None:
            if not np.issubdtype(y.dtype, np.number):
                return False, "Elastic Net regression requires continuous numerical target values"
            
            # Check for sufficient variance in target
            if np.var(y) == 0:
                return False, "Target variable has zero variance (all values are the same)"
            
            # Sample and feature analysis
            n_samples, n_features = X.shape
            
            advantages = []
            considerations = []
            
            # Correlation analysis (Elastic Net's strength)
            try:
                corr_matrix = np.corrcoef(X.T)
                max_corr = 0
                high_corr_pairs = 0
                
                for i in range(len(corr_matrix)):
                    for j in range(i+1, len(corr_matrix)):
                        abs_corr = abs(corr_matrix[i, j])
                        max_corr = max(max_corr, abs_corr)
                        if abs_corr > 0.7:
                            high_corr_pairs += 1
                
                if high_corr_pairs > 0:
                    advantages.append(f"Found {high_corr_pairs} highly correlated feature pairs - perfect for Elastic Net")
                elif max_corr > 0.5:
                    advantages.append(f"Moderate feature correlation detected - good for Elastic Net")
                else:
                    considerations.append("Low feature correlation - simple Lasso might suffice")
                    
            except:
                pass
            
            # Dimensionality assessment
            if n_features > n_samples:
                advantages.append(f"High-dimensional data ({n_features} features, {n_samples} samples) - excellent for Elastic Net")
            elif n_features > n_samples * 0.5:
                advantages.append(f"Many features ({n_features}) relative to samples ({n_samples}) - good for Elastic Net")
            elif n_features > 20:
                advantages.append(f"Moderate feature count ({n_features}) - Elastic Net provides robust regularization")
            else:
                considerations.append(f"Low feature count ({n_features}) - regularization benefit may be limited")
            
            # Sample size assessment
            if n_samples >= n_features * 10:
                advantages.append(f"Excellent sample size ({n_samples}) for reliable hyperparameter selection")
            elif n_samples >= n_features * 5:
                advantages.append(f"Good sample size ({n_samples}) for cross-validation")
            elif n_samples >= n_features:
                considerations.append(f"Adequate sample size ({n_samples}) but hyperparameter tuning may be noisy")
            else:
                considerations.append(f"Small sample size ({n_samples}) - hyperparameter selection may be unstable")
            
            # Feature scaling check
            try:
                feature_scales = np.std(X, axis=0)
                max_scale_ratio = np.max(feature_scales) / np.min(feature_scales) if np.min(feature_scales) > 0 else np.inf
                
                if max_scale_ratio > 100:
                    considerations.append("Features have very different scales - standardization critical")
                elif max_scale_ratio > 10:
                    considerations.append("Features have different scales - standardization recommended")
                else:
                    advantages.append("Feature scales are similar")
            except:
                pass
            
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
                f"📊 Suitability for Elastic Net: {suitability}"
            ]
            
            if advantages:
                message_parts.append("🎯 Advantages: " + "; ".join(advantages))
            
            if considerations:
                message_parts.append("💡 Considerations: " + "; ".join(considerations))
            
            return True, " | ".join(message_parts)
        
        return True, f"Compatible with {X.shape[0]} samples and {X.shape[1]} features"
    
    def get_feature_importance(self) -> Optional[Dict[str, Any]]:
        """Get feature importance based on coefficient magnitudes and regularization effects"""
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
            # Analyze regularization effect on this feature
            reg_effect = self._analyze_feature_regularization_effect(i, coef)
            
            feature_importance[name] = {
                'coefficient': coef,
                'absolute_coefficient': imp,
                'normalized_importance': norm_imp,
                'selected': selected_mask[i],
                'rank': i + 1,
                'regularization_effect': reg_effect
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
            'regularization_info': {
                'total_features': len(coefficients),
                'selected_count': len(selected_indices),
                'elimination_rate': len(eliminated_indices) / len(coefficients),
                'alpha_used': self.alpha_used_,
                'l1_ratio_used': self.l1_ratio_used_,
                'l1_contribution': self.alpha_used_ * self.l1_ratio_used_,
                'l2_contribution': self.alpha_used_ * (1 - self.l1_ratio_used_)
            },
            'sorted_features': [name for name, _ in sorted_features],
            'sorted_importance': [info['normalized_importance'] for _, info in sorted_features],
            'interpretation': f'Feature selection via Elastic Net (α={self.alpha_used_:.4f}, l1_ratio={self.l1_ratio_used_:.2f})'
        }
    
    def _analyze_feature_regularization_effect(self, feature_idx, coefficient):
        """Analyze how regularization affects a specific feature"""
        if 'l1_ratio_comparison' in self.regularization_path_ and self.regularization_path_['l1_ratio_comparison']:
            l1_ratio_comparison = self.regularization_path_['l1_ratio_comparison']
            
            # Check how this feature behaves across different l1_ratios
            feature_behavior = {}
            for l1_ratio, data in l1_ratio_comparison.items():
                coef_at_ratio = data['coefficients_at_current_alpha'][feature_idx]
                feature_behavior[l1_ratio] = coef_at_ratio
            
            # Analyze stability and effect
            coef_values = [abs(c) for c in feature_behavior.values() if abs(c) > 1e-8]
            
            if len(coef_values) == 0:
                effect = "consistently_eliminated"
            elif len(coef_values) == len(feature_behavior):
                effect = "consistently_selected"
            elif abs(coefficient) > 1e-8:
                effect = "selected_with_current_regularization"
            else:
                effect = "eliminated_with_current_regularization"
            
            return {
                'effect_category': effect,
                'behavior_across_l1_ratios': feature_behavior,
                'stability_score': len(coef_values) / len(feature_behavior)
            }
        
        return {
            'effect_category': 'selected' if abs(coefficient) > 1e-8 else 'eliminated',
            'behavior_across_l1_ratios': None,
            'stability_score': None
        }
    
    def get_model_params(self) -> Dict[str, Any]:
        """Get current model parameters"""
        if not self.is_fitted_:
            return {}
        
        selected_features = np.sum(np.abs(self.model_.coef_) > 1e-8)
        
        return {
            "algorithm": "Elastic Net Regression",
            "n_features": self.n_features_in_,
            "feature_names": self.feature_names_,
            "alpha_used": self.alpha_used_,
            "l1_ratio_used": self.l1_ratio_used_,
            "l1_contribution": self.alpha_used_ * self.l1_ratio_used_,
            "l2_contribution": self.alpha_used_ * (1 - self.l1_ratio_used_),
            "hyperparameter_method": "automatic" if (self.auto_alpha or self.auto_l1_ratio) else "manual",
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
            "regularization_analysis": self.regularization_analysis,
            "stability_analysis": self.analyze_feature_stability,
            "l1_l2_balance_analysis": self.l1_l2_balance_analysis
        }
    
    def get_regularization_analysis(self) -> Dict[str, Any]:
        """Get comprehensive regularization analysis results"""
        if not self.is_fitted_:
            return {"status": "Model not fitted"}
        
        results = {
            "hyperparameter_optimization": self.hyperparameter_analysis_,
            "regularization_path": self.regularization_path_,
            "feature_selection": self.feature_selection_analysis_,
            "regularization_balance": self.regularization_balance_,
            "stability_analysis": self.stability_analysis_
        }
        
        return results
    
    def plot_regularization_surface(self, figsize=(15, 12)):
        """Plot comprehensive regularization analysis"""
        if not self.is_fitted_:
            raise ValueError("Model must be fitted to plot regularization analysis")
        
        fig, axes = plt.subplots(2, 3, figsize=figsize)
        
        # Plot 1: Regularization path (alpha vs coefficients)
        ax1 = axes[0, 0]
        if 'coefficients' in self.regularization_path_:
            alphas = self.regularization_path_['alphas']
            coefs = self.regularization_path_['coefficients']
            
            for i in range(min(20, coefs.shape[0])):
                ax1.plot(alphas, coefs[i, :], alpha=0.7, linewidth=1)
            
            ax1.axvline(x=self.alpha_used_, color='red', linestyle='--', alpha=0.8, 
                       label=f'α={self.alpha_used_:.4f}')
            ax1.set_xscale('log')
            ax1.set_xlabel('Alpha')
            ax1.set_ylabel('Coefficients')
            ax1.set_title(f'Regularization Path (l1_ratio={self.l1_ratio_used_:.2f})')
            ax1.grid(True, alpha=0.3)
            ax1.legend()
        else:
            ax1.text(0.5, 0.5, 'Regularization path\nnot available', 
                    ha='center', va='center', transform=ax1.transAxes)
            ax1.set_title('Regularization Path')
        
        # Plot 2: L1 vs L2 contribution
        ax2 = axes[0, 1]
        l1_contrib = self.alpha_used_ * self.l1_ratio_used_
        l2_contrib = self.alpha_used_ * (1 - self.l1_ratio_used_)
        
        contributions = [l1_contrib, l2_contrib]
        labels = ['L1 (Lasso)', 'L2 (Ridge)']
        colors = ['lightcoral', 'lightblue']
        
        wedges, texts, autotexts = ax2.pie(contributions, labels=labels, colors=colors, 
                                          autopct='%1.1f%%', startangle=90)
        ax2.set_title(f'L1/L2 Balance\n(l1_ratio={self.l1_ratio_used_:.2f})')
        
        # Plot 3: Feature selection across l1_ratios
        ax3 = axes[0, 2]
        if ('l1_ratio_comparison' in self.regularization_path_ and 
            self.regularization_path_['l1_ratio_comparison']):
            
            l1_ratios = []
            n_features_selected = []
            
            for l1_ratio, data in self.regularization_path_['l1_ratio_comparison'].items():
                l1_ratios.append(l1_ratio)
                n_features_selected.append(data['n_nonzero_features'])
            
            ax3.plot(l1_ratios, n_features_selected, 'bo-', linewidth=2, markersize=6)
            ax3.axvline(x=self.l1_ratio_used_, color='red', linestyle='--', alpha=0.8,
                       label=f'Current: {np.sum(np.abs(self.model_.coef_) > 1e-8)} features')
            ax3.set_xlabel('L1 Ratio')
            ax3.set_ylabel('Number of Selected Features')
            ax3.set_title('Feature Selection vs L1/L2 Balance')
            ax3.grid(True, alpha=0.3)
            ax3.legend()
        else:
            ax3.text(0.5, 0.5, 'L1 ratio comparison\nnot available', 
                    ha='center', va='center', transform=ax3.transAxes)
            ax3.set_title('Feature Selection vs L1/L2 Balance')
        
        # Plot 4: Cross-validation results (if available)
        ax4 = axes[1, 0]
        if 'cv_scores' in self.hyperparameter_analysis_:
            # This would need specific implementation based on the CV method used
            ax4.text(0.5, 0.5, 'CV results available\n(implementation needed)', 
                    ha='center', va='center', transform=ax4.transAxes)
            ax4.set_title('Cross-Validation Results')
        else:
            ax4.text(0.5, 0.5, 'CV results\nnot available', 
                    ha='center', va='center', transform=ax4.transAxes)
            ax4.set_title('Cross-Validation Results')
        
        # Plot 5: Coefficient magnitude distribution
        ax5 = axes[1, 1]
        coefficients = self.model_.coef_
        nonzero_coefs = coefficients[np.abs(coefficients) > 1e-8]
        
        if len(nonzero_coefs) > 0:
            ax5.hist(nonzero_coefs, bins=min(20, len(nonzero_coefs)), alpha=0.7, 
                    color='skyblue', edgecolor='black')
            ax5.axvline(x=0, color='red', linestyle='-', alpha=0.5)
            ax5.set_xlabel('Coefficient Value')
            ax5.set_ylabel('Frequency')
            ax5.set_title(f'Coefficient Distribution\n({len(nonzero_coefs)} selected features)')
            ax5.grid(True, alpha=0.3)
        else:
            ax5.text(0.5, 0.5, 'No features selected', 
                    ha='center', va='center', transform=ax5.transAxes)
            ax5.set_title('Coefficient Distribution')
        
        # Plot 6: Regularization strength interpretation
        ax6 = axes[1, 2]
        
        # Create a visual representation of regularization balance
        l1_strength = self.l1_ratio_used_
        l2_strength = 1 - self.l1_ratio_used_
        alpha_strength = min(self.alpha_used_, 1.0)  # Cap for visualization
        
        # Create a simple bar chart
        categories = ['L1\n(Feature Selection)', 'L2\n(Feature Grouping)', 'Overall\n(Alpha)']
        values = [l1_strength, l2_strength, alpha_strength]
        colors = ['coral', 'lightblue', 'lightgreen']
        
        bars = ax6.bar(categories, values, color=colors, alpha=0.7)
        ax6.set_ylabel('Strength')
        ax6.set_title('Regularization Components')
        ax6.set_ylim(0, 1.1)
        
        # Add value labels on bars
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax6.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{value:.2f}', ha='center', va='bottom')
        
        plt.tight_layout()
        return fig
    
    def plot_l1_l2_comparison(self, figsize=(12, 8)):
        """Plot comparison of different L1/L2 ratios"""
        if not self.is_fitted_ or 'l1_ratio_comparison' not in self.regularization_path_:
            raise ValueError("Model must be fitted with L1/L2 balance analysis enabled")
        
        l1_ratio_comparison = self.regularization_path_['l1_ratio_comparison']
        if not l1_ratio_comparison:
            raise ValueError("L1/L2 comparison data not available")
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        
        # Plot 1: Sparsity vs L1 ratio
        l1_ratios = list(l1_ratio_comparison.keys())
        sparsities = [data['sparsity'] for data in l1_ratio_comparison.values()]
        n_features = [data['n_nonzero_features'] for data in l1_ratio_comparison.values()]
        
        ax1.plot(l1_ratios, sparsities, 'ro-', linewidth=2, markersize=8, label='Sparsity')
        ax1.axvline(x=self.l1_ratio_used_, color='blue', linestyle='--', alpha=0.8,
                   label=f'Current l1_ratio={self.l1_ratio_used_:.2f}')
        ax1.set_xlabel('L1 Ratio')
        ax1.set_ylabel('Sparsity (fraction of eliminated features)')
        ax1.set_title('Sparsity vs L1/L2 Balance')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        ax1.set_ylim(0, 1)
        
        # Plot 2: Number of features vs L1 ratio
        ax2.plot(l1_ratios, n_features, 'go-', linewidth=2, markersize=8, label='Selected Features')
        ax2.axvline(x=self.l1_ratio_used_, color='blue', linestyle='--', alpha=0.8,
                   label=f'Current: {np.sum(np.abs(self.model_.coef_) > 1e-8)} features')
        ax2.set_xlabel('L1 Ratio')
        ax2.set_ylabel('Number of Selected Features')
        ax2.set_title('Feature Count vs L1/L2 Balance')
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
            "algorithm": "Elastic Net Regression",
            "type": "L1+L2 Regularized Linear Regression",
            "training_completed": True,
            "regularization_characteristics": {
                "regularization_type": "L1 + L2 (Elastic Net)",
                "feature_selection": True,
                "feature_grouping": True,
                "handles_multicollinearity": "excellent",
                "correlated_feature_handling": "groups correlated features"
            },
            "model_configuration": {
                "alpha_used": self.alpha_used_,
                "l1_ratio_used": self.l1_ratio_used_,
                "l1_contribution": self.alpha_used_ * self.l1_ratio_used_,
                "l2_contribution": self.alpha_used_ * (1 - self.l1_ratio_used_),
                "hyperparameter_optimization": "automatic" if (self.auto_alpha or self.auto_l1_ratio) else "manual",
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
                "regularization_analysis": self.regularization_analysis,
                "l1_l2_balance_analysis": self.l1_l2_balance_analysis,
                "cross_validation": self.auto_alpha or self.auto_l1_ratio
            }
        }
        
        # Add hyperparameter optimization results if available
        if self.hyperparameter_analysis_:
            info["hyperparameter_optimization_results"] = {
                "method": self.hyperparameter_analysis_.get('method'),
                "optimal_alpha": self.hyperparameter_analysis_.get('optimal_alpha'),
                "optimal_l1_ratio": self.hyperparameter_analysis_.get('optimal_l1_ratio'),
                "cv_folds": self.hyperparameter_analysis_.get('cv_folds'),
                "best_score": self.hyperparameter_analysis_.get('best_score')
            }
        
        # Add regularization balance analysis
        if self.regularization_balance_:
            balance_info = self.regularization_balance_.get('regularization_interpretation', {})
            info["regularization_strategy"] = {
                "l1_ratio_description": balance_info.get('l1_ratio_description'),
                "alpha_description": balance_info.get('alpha_description'),
                "overall_strategy": balance_info.get('overall_strategy')
            }
        
        return info

    # ADD THE OVERRIDDEN METHOD HERE:
    def get_algorithm_specific_metrics(self,
                                       y_true: np.ndarray,
                                       y_pred: np.ndarray,
                                       y_proba: Optional[np.ndarray] = None
                                       ) -> Dict[str, Any]:
        """
        Calculate Elastic Net Regression-specific metrics based on the fitted model's
        internal analyses and characteristics.

        Note: Most metrics are derived from analyses performed during the fit method
        (e.g., hyperparameter tuning, coefficient analysis).
        The y_true, y_pred parameters (typically for test set evaluation)
        are not directly used for these internal model-specific metrics.
        y_proba is not applicable for regression.

        Args:
            y_true: Ground truth target values from a test set.
            y_pred: Predicted target values on a test set.
            y_proba: Not used for regression.

        Returns:
            A dictionary of Elastic Net Regression-specific metrics.
        """
        metrics = {}
        if not self.is_fitted_ or self.model_ is None:
            metrics["status"] = "Model not fitted"
            return metrics

        # --- Core Model Parameters & Coefficients ---
        metrics['alpha_applied'] = getattr(self, 'alpha_used_', self.alpha)
        metrics['l1_ratio_applied'] = getattr(self, 'l1_ratio_used_', self.l1_ratio)
        
        if hasattr(self.model_, 'coef_') and self.model_.coef_ is not None:
            coefs = self.model_.coef_
            metrics['num_selected_features'] = int(np.sum(np.abs(coefs) > 1e-8))
            if coefs.size > 0:
                metrics['coefficient_sparsity_ratio'] = float(np.sum(np.abs(coefs) <= 1e-8) / coefs.size)
            else:
                metrics['coefficient_sparsity_ratio'] = None
            metrics['coefficient_l1_norm'] = float(np.sum(np.abs(coefs)))
            metrics['coefficient_l2_norm'] = float(np.linalg.norm(coefs))
        
        if self.fit_intercept and hasattr(self.model_, 'intercept_'):
            metrics['intercept_value'] = float(self.model_.intercept_)

        # --- Hyperparameter Optimization Results ---
        if hasattr(self, 'hyperparameter_analysis_') and self.hyperparameter_analysis_:
            h_analysis = self.hyperparameter_analysis_
            metrics['optimal_alpha_from_cv'] = h_analysis.get('optimal_alpha')
            metrics['optimal_l1_ratio_from_cv'] = h_analysis.get('optimal_l1_ratio')
            metrics['best_cv_score_hyperparam_opt'] = h_analysis.get('best_score') # e.g., min MSE

        # --- Feature Selection Analysis ---
        if hasattr(self, 'feature_selection_analysis_') and self.feature_selection_analysis_ and \
           'selection_statistics' in self.feature_selection_analysis_:
            fs_stats = self.feature_selection_analysis_['selection_statistics']
            metrics['feature_selection_ratio'] = fs_stats.get('selection_ratio')

        # --- Regularization Balance Analysis ---
        if hasattr(self, 'regularization_balance_') and self.regularization_balance_:
            rb_analysis = self.regularization_balance_
            if 'hyperparameters' in rb_analysis:
                metrics['effective_l1_penalty'] = rb_analysis['hyperparameters'].get('l1_contribution')
                metrics['effective_l2_penalty'] = rb_analysis['hyperparameters'].get('l2_contribution')
            if 'coefficient_statistics' in rb_analysis:
                metrics['mean_abs_coefficient_selected'] = rb_analysis['coefficient_statistics'].get('mean_magnitude')
        
        # --- Feature Stability Analysis ---
        if hasattr(self, 'stability_analysis_') and self.stability_analysis_:
            s_analysis = self.stability_analysis_
            metrics['overall_feature_stability'] = s_analysis.get('overall_stability')

        # Remove None values for cleaner output
        metrics = {k: v for k, v in metrics.items() if v is not None}

        return metrics

# Factory function
def get_plugin():
    """Factory function to get the plugin instance"""
    return ElasticNetRegressionPlugin()


# Example usage and testing
if __name__ == "__main__":
    """
    Example usage of Elastic Net Regression Plugin
    """
    print("Testing Elastic Net Regression Plugin...")
    
    try:
        # Create sample data with correlated features
        from sklearn.datasets import make_regression
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import mean_squared_error, r2_score
        
        # Generate dataset with correlated features (perfect for Elastic Net)
        X, y = make_regression(
            n_samples=200,
            n_features=100,
            n_informative=20,
            n_redundant=20,  # Add redundant (correlated) features
            noise=0.1,
            random_state=42
        )
        
        # Create feature names
        feature_names = [f"feature_{i}" for i in range(X.shape[1])]
        X_df = pd.DataFrame(X, columns=feature_names)
        
        print(f"\n📊 Correlated High-Dimensional Dataset Info:")
        print(f"Shape: {X_df.shape}")
        print(f"Informative features: 20 out of {X.shape[1]}")
        print(f"Redundant (correlated) features: 20")
        print(f"Target range: [{y.min():.2f}, {y.max():.2f}]")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_df, y, test_size=0.2, random_state=42
        )
        
        # Create and test Elastic Net plugin
        plugin = ElasticNetRegressionPlugin(
            auto_alpha=True,
            auto_l1_ratio=True,
            alpha_selection_method='cv',
            cv_folds=5,
            normalize_features=True,
            analyze_regularization_path=True,
            analyze_feature_stability=True,
            compute_feature_importance=True,
            regularization_analysis=True,
            l1_l2_balance_analysis=True,
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
            # Train Elastic Net model
            print("\n🚀 Training Elastic Net with automatic hyperparameter optimization...")
            plugin.fit(X_train, y_train)
            
            # Make predictions
            y_pred = plugin.predict(X_test)
            
            # Evaluate performance
            mse = mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            r2 = r2_score(y_test, y_pred)
            
            print(f"\n📊 Elastic Net Regression Results:")
            print(f"MSE: {mse:.4f}")
            print(f"RMSE: {rmse:.4f}")
            print(f"R²: {r2:.4f}")
            
            # Get model parameters
            model_params = plugin.get_model_params()
            print(f"\n⚙️ Model Configuration:")
            print(f"Alpha used: {model_params['alpha_used']:.6f}")
            print(f"L1 ratio used: {model_params['l1_ratio_used']:.3f}")
            print(f"L1 contribution: {model_params['l1_contribution']:.6f}")
            print(f"L2 contribution: {model_params['l2_contribution']:.6f}")
            print(f"Selected features: {model_params['selected_features']}")
            print(f"Eliminated features: {model_params['eliminated_features']}")
            print(f"Sparsity ratio: {model_params['sparsity_ratio']:.2%}")
            
            # Get feature importance
            importance = plugin.get_feature_importance()
            print(f"\n🎯 Feature Selection Results:")
            print(f"Features selected: {importance['selected_features']['count']}")
            print(f"Features eliminated: {importance['eliminated_features']['count']}")
            print(f"Elimination rate: {importance['regularization_info']['elimination_rate']:.1%}")
            
            if importance['selected_features']['count'] > 0:
                print(f"\nTop selected features:")
                for i, name in enumerate(importance['selected_features']['names'][:5]):
                    coef = importance['selected_features']['coefficients'][i]
                    print(f"  {name}: {coef:.4f}")
            
            print("\n✅ Elastic Net Regression Plugin test completed successfully!")
            
    except Exception as e:
        print(f"\n❌ Error testing Elastic Net Regression Plugin: {str(e)}")
        import traceback
        traceback.print_exc()