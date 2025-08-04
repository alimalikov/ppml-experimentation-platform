import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, Optional, Tuple, List
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_X_y, check_array
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.pipeline import Pipeline
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


class PolynomialRegressionPlugin(BaseEstimator, RegressorMixin, MLPlugin):
    """
    Polynomial Regression Plugin - Capturing Non-Linear Relationships
    
    This plugin implements comprehensive polynomial regression that extends linear regression
    to capture non-linear relationships by creating polynomial features. Perfect for datasets
    where the relationship between features and target follows polynomial patterns.
    
    Key Features:
    - Automatic polynomial degree selection via cross-validation
    - Polynomial feature engineering with interaction terms
    - Ridge regularization to prevent overfitting
    - Comprehensive bias-variance analysis
    - Feature importance for polynomial terms
    - Overfitting detection and prevention
    - Advanced polynomial diagnostics
    - Curvature analysis and inflection point detection
    """
    
    def __init__(
        self,
        degree=2,
        include_bias=True,
        interaction_only=False,
        normalize_features=True,
        
        # Regularization options
        use_regularization=True,
        alpha=1.0,
        auto_alpha=True,
        alpha_selection_method='cv',
        cv_folds=5,
        
        # Polynomial configuration
        auto_degree=True,
        max_degree=10,
        degree_selection_method='cv',
        overfitting_threshold=0.1,
        
        # Feature engineering
        feature_selection=True,
        feature_selection_threshold=0.01,
        remove_low_variance=True,
        variance_threshold=1e-8,
        
        # Analysis options
        analyze_polynomial_terms=True,
        detect_overfitting=True,
        compute_feature_importance=True,
        bias_variance_analysis=True,
        curvature_analysis=True,
        
        random_state=42
    ):
        super().__init__()
        
        # Core polynomial parameters
        self.degree = degree
        self.include_bias = include_bias
        self.interaction_only = interaction_only
        self.normalize_features = normalize_features
        
        # Regularization
        self.use_regularization = use_regularization
        self.alpha = alpha
        self.auto_alpha = auto_alpha
        self.alpha_selection_method = alpha_selection_method
        self.cv_folds = cv_folds
        
        # Polynomial optimization
        self.auto_degree = auto_degree
        self.max_degree = max_degree
        self.degree_selection_method = degree_selection_method
        self.overfitting_threshold = overfitting_threshold
        
        # Feature engineering
        self.feature_selection = feature_selection
        self.feature_selection_threshold = feature_selection_threshold
        self.remove_low_variance = remove_low_variance
        self.variance_threshold = variance_threshold
        
        # Analysis options
        self.analyze_polynomial_terms = analyze_polynomial_terms
        self.detect_overfitting = detect_overfitting
        self.compute_feature_importance = compute_feature_importance
        self.bias_variance_analysis = bias_variance_analysis
        self.curvature_analysis = curvature_analysis
        
        self.random_state = random_state
        
        # Required plugin metadata
        self._name = "Polynomial Regression"
        self._description = "Non-linear regression through polynomial feature engineering with overfitting protection"
        self._category = "Linear Models"
        
        # Required capability flags
        self._supports_classification = False
        self._supports_regression = True
        self._min_samples_required = 15
        
        # Internal state
        self.is_fitted_ = False
        self.model_ = None
        self.scaler_ = None
        self.poly_features_ = None
        self.feature_names_ = None
        self.polynomial_feature_names_ = None
        self.n_features_in_ = None
        
        # Optimization results
        self.degree_used_ = None
        self.alpha_used_ = None
        
        # Analysis results
        self.degree_analysis_ = {}
        self.polynomial_analysis_ = {}
        self.overfitting_analysis_ = {}
        self.feature_importance_analysis_ = {}
        self.bias_variance_analysis_ = {}
        self.curvature_analysis_ = {}
    
    def get_name(self) -> str:
        return self._name
    
    def get_description(self) -> str:
        return self._description
    
    def get_category(self) -> str:
        return self._category
    
    def fit(self, X, y, sample_weight=None):
        """
        Fit the Polynomial Regression model with comprehensive analysis
        
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
        
        # Determine optimal polynomial degree if auto-optimization is enabled
        if self.auto_degree:
            optimal_degree = self._find_optimal_degree(X_processed, y, sample_weight)
            self.degree_used_ = optimal_degree
        else:
            self.degree_used_ = self.degree
        
        # Create polynomial features
        self.poly_features_ = PolynomialFeatures(
            degree=self.degree_used_,
            include_bias=self.include_bias,
            interaction_only=self.interaction_only
        )
        
        X_poly = self.poly_features_.fit_transform(X_processed)
        
        # Generate polynomial feature names
        self.polynomial_feature_names_ = self._generate_polynomial_feature_names()
        
        # Feature selection if enabled
        if self.feature_selection:
            X_poly = self._apply_feature_selection(X_poly, y)
        
        # Store processed polynomial features
        self.X_poly_ = X_poly
        
        # Determine optimal regularization if enabled
        if self.use_regularization:
            if self.auto_alpha:
                optimal_alpha = self._find_optimal_alpha(X_poly, y, sample_weight)
                self.alpha_used_ = optimal_alpha
            else:
                self.alpha_used_ = self.alpha
            
            # Use Ridge regression for regularization
            self.model_ = Ridge(alpha=self.alpha_used_, random_state=self.random_state)
        else:
            self.model_ = LinearRegression()
            self.alpha_used_ = 0.0
        
        # Fit the model
        self.model_.fit(X_poly, y, sample_weight=sample_weight)
        
        # Perform comprehensive analysis
        self._analyze_polynomial_terms_comprehensive()
        self._analyze_overfitting()
        self._analyze_feature_importance()
        self._analyze_bias_variance()
        self._analyze_curvature()
        
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
        
        # Transform to polynomial features
        X_poly = self.poly_features_.transform(X_processed)
        
        # Apply same feature selection as training
        if self.feature_selection and hasattr(self, 'selected_features_'):
            X_poly = X_poly[:, self.selected_features_]
        
        return self.model_.predict(X_poly)
    
    def _find_optimal_degree(self, X, y, sample_weight=None):
        """Find optimal polynomial degree using cross-validation"""
        degrees = range(1, min(self.max_degree + 1, min(X.shape[0] // 3, 15)))
        
        if self.degree_selection_method == 'cv':
            return self._cv_degree_search(X, y, degrees, sample_weight)
        elif self.degree_selection_method == 'validation_curve':
            return self._validation_curve_degree_search(X, y, degrees, sample_weight)
        else:
            return self._bias_variance_degree_search(X, y, degrees, sample_weight)
    
    def _cv_degree_search(self, X, y, degrees, sample_weight=None):
        """Cross-validation based degree selection"""
        best_score = float('-inf')
        best_degree = self.degree
        cv_results = []
        
        for degree in degrees:
            try:
                # Create polynomial features
                poly = PolynomialFeatures(
                    degree=degree,
                    include_bias=self.include_bias,
                    interaction_only=self.interaction_only
                )
                X_poly = poly.fit_transform(X)
                
                # Skip if too many features
                if X_poly.shape[1] > X.shape[0] * 0.8:
                    continue
                
                # Create model
                if self.use_regularization:
                    model = Ridge(alpha=self.alpha, random_state=self.random_state)
                else:
                    model = LinearRegression()
                
                # Cross-validation
                cv_scores = cross_val_score(
                    model, X_poly, y,
                    cv=self.cv_folds,
                    scoring='r2',
                    fit_params={'sample_weight': sample_weight} if sample_weight is not None else None
                )
                
                mean_score = np.mean(cv_scores)
                std_score = np.std(cv_scores)
                
                cv_results.append({
                    'degree': degree,
                    'cv_score': mean_score,
                    'cv_std': std_score,
                    'n_features': X_poly.shape[1]
                })
                
                if mean_score > best_score:
                    best_score = mean_score
                    best_degree = degree
                    
            except Exception:
                continue
        
        # Store degree analysis results
        self.degree_analysis_ = {
            'method': 'Cross-Validation',
            'degrees_tested': list(degrees),
            'cv_results': cv_results,
            'optimal_degree': best_degree,
            'best_score': best_score,
            'cv_folds': self.cv_folds
        }
        
        return best_degree
    
    def _validation_curve_degree_search(self, X, y, degrees, sample_weight=None):
        """Validation curve based degree selection with overfitting detection"""
        try:
            # Create a pipeline for validation curve
            if self.use_regularization:
                model = Ridge(alpha=self.alpha, random_state=self.random_state)
            else:
                model = LinearRegression()
            
            # We'll manually create polynomial features for each degree
            train_scores_list = []
            validation_scores_list = []
            valid_degrees = []
            
            for degree in degrees:
                try:
                    poly = PolynomialFeatures(
                        degree=degree,
                        include_bias=self.include_bias,
                        interaction_only=self.interaction_only
                    )
                    X_poly = poly.fit_transform(X)
                    
                    # Skip if too many features
                    if X_poly.shape[1] > X.shape[0] * 0.8:
                        continue
                    
                    # Manual cross-validation
                    from sklearn.model_selection import KFold
                    kf = KFold(n_splits=self.cv_folds, shuffle=True, random_state=self.random_state)
                    
                    train_scores = []
                    val_scores = []
                    
                    for train_idx, val_idx in kf.split(X_poly):
                        X_train_fold, X_val_fold = X_poly[train_idx], X_poly[val_idx]
                        y_train_fold, y_val_fold = y[train_idx], y[val_idx]
                        
                        model.fit(X_train_fold, y_train_fold)
                        
                        train_score = model.score(X_train_fold, y_train_fold)
                        val_score = model.score(X_val_fold, y_val_fold)
                        
                        train_scores.append(train_score)
                        val_scores.append(val_score)
                    
                    train_scores_list.append(train_scores)
                    validation_scores_list.append(val_scores)
                    valid_degrees.append(degree)
                    
                except Exception:
                    continue
            
            if not valid_degrees:
                return self.degree
            
            # Convert to arrays
            train_scores_array = np.array(train_scores_list)
            validation_scores_array = np.array(validation_scores_list)
            
            train_mean = np.mean(train_scores_array, axis=1)
            train_std = np.std(train_scores_array, axis=1)
            val_mean = np.mean(validation_scores_array, axis=1)
            val_std = np.std(validation_scores_array, axis=1)
            
            # Detect overfitting (train score much higher than validation score)
            overfitting_scores = train_mean - val_mean
            
            # Find best degree with overfitting consideration
            best_degree = valid_degrees[0]
            best_val_score = val_mean[0]
            
            for i, degree in enumerate(valid_degrees):
                # Check if overfitting is within threshold
                if overfitting_scores[i] <= self.overfitting_threshold:
                    if val_mean[i] > best_val_score:
                        best_val_score = val_mean[i]
                        best_degree = degree
            
            # Store validation curve analysis
            self.degree_analysis_ = {
                'method': 'Validation Curve',
                'degrees_tested': valid_degrees,
                'train_scores': train_mean,
                'train_std': train_std,
                'validation_scores': val_mean,
                'validation_std': val_std,
                'overfitting_scores': overfitting_scores,
                'optimal_degree': best_degree,
                'best_validation_score': best_val_score,
                'overfitting_threshold': self.overfitting_threshold
            }
            
            return best_degree
            
        except Exception:
            return self.degree
    
    def _bias_variance_degree_search(self, X, y, degrees, sample_weight=None):
        """Bias-variance tradeoff based degree selection"""
        return self._cv_degree_search(X, y, degrees, sample_weight)  # Fallback to CV
    
    def _find_optimal_alpha(self, X_poly, y, sample_weight=None):
        """Find optimal regularization parameter"""
        alphas = np.logspace(-6, 2, 50)
        
        best_score = float('-inf')
        best_alpha = self.alpha
        
        for alpha in alphas:
            try:
                model = Ridge(alpha=alpha, random_state=self.random_state)
                cv_scores = cross_val_score(
                    model, X_poly, y,
                    cv=self.cv_folds,
                    scoring='r2',
                    fit_params={'sample_weight': sample_weight} if sample_weight is not None else None
                )
                
                mean_score = np.mean(cv_scores)
                
                if mean_score > best_score:
                    best_score = mean_score
                    best_alpha = alpha
                    
            except Exception:
                continue
        
        return best_alpha
    
    def _generate_polynomial_feature_names(self):
        """Generate descriptive names for polynomial features"""
        feature_names = []
        
        if not hasattr(self.poly_features_, 'powers_'):
            # If powers_ not available, create generic names
            n_poly_features = self.poly_features_.n_output_features_
            for i in range(n_poly_features):
                feature_names.append(f"poly_feature_{i}")
            return feature_names
        
        powers = self.poly_features_.powers_
        
        for power_combo in powers:
            if np.sum(power_combo) == 0:  # Bias term
                feature_names.append("bias")
            elif np.sum(power_combo) == 1:  # Linear term
                idx = np.where(power_combo == 1)[0][0]
                feature_names.append(self.feature_names_[idx])
            else:  # Polynomial/interaction term
                term_parts = []
                for i, power in enumerate(power_combo):
                    if power > 0:
                        if power == 1:
                            term_parts.append(self.feature_names_[i])
                        else:
                            term_parts.append(f"{self.feature_names_[i]}^{power}")
                
                feature_names.append(" * ".join(term_parts))
        
        return feature_names
    
    def _apply_feature_selection(self, X_poly, y):
        """Apply feature selection to polynomial features"""
        # Remove low variance features
        if self.remove_low_variance:
            variances = np.var(X_poly, axis=0)
            high_variance_mask = variances > self.variance_threshold
        else:
            high_variance_mask = np.ones(X_poly.shape[1], dtype=bool)
        
        # Simple correlation-based feature selection
        selected_mask = high_variance_mask.copy()
        
        if self.feature_selection_threshold > 0:
            # Calculate correlation with target for remaining features
            correlations = []
            for i in range(X_poly.shape[1]):
                if selected_mask[i]:
                    corr = np.abs(np.corrcoef(X_poly[:, i], y)[0, 1])
                    if not np.isnan(corr):
                        correlations.append(corr)
                    else:
                        correlations.append(0)
                        selected_mask[i] = False
                else:
                    correlations.append(0)
            
            correlations = np.array(correlations)
            selected_mask = selected_mask & (correlations >= self.feature_selection_threshold)
        
        # Ensure at least one feature is selected
        if not np.any(selected_mask):
            # Select the feature with highest correlation
            best_feature = np.argmax(np.abs([np.corrcoef(X_poly[:, i], y)[0, 1] 
                                           for i in range(X_poly.shape[1])]))
            selected_mask[best_feature] = True
        
        self.selected_features_ = np.where(selected_mask)[0]
        return X_poly[:, selected_mask]
    
    def _analyze_polynomial_terms_comprehensive(self):
        """Analyze polynomial terms and their contributions"""
        if not self.analyze_polynomial_terms:
            return
        
        coefficients = self.model_.coef_
        
        # If bias is included in polynomial features but model doesn't fit intercept
        if hasattr(self.model_, 'intercept_') and self.model_.intercept_ != 0:
            all_coefficients = np.concatenate([[self.model_.intercept_], coefficients])
            all_feature_names = ['intercept'] + self.polynomial_feature_names_
        else:
            all_coefficients = coefficients
            all_feature_names = self.polynomial_feature_names_
        
        # Apply feature selection mapping if used
        if self.feature_selection and hasattr(self, 'selected_features_'):
            selected_names = [all_feature_names[i] for i in self.selected_features_]
            selected_coefficients = all_coefficients
        else:
            selected_names = all_feature_names
            selected_coefficients = all_coefficients
        
        # Categorize polynomial terms
        term_analysis = self._categorize_polynomial_terms(selected_names, selected_coefficients)
        
        # Calculate feature importance
        importance_scores = np.abs(selected_coefficients)
        normalized_importance = importance_scores / np.sum(importance_scores) if np.sum(importance_scores) > 0 else importance_scores
        
        # Rank terms by importance
        importance_ranking = np.argsort(importance_scores)[::-1]
        
        self.polynomial_analysis_ = {
            'total_polynomial_features': len(selected_coefficients),
            'original_features': self.n_features_in_,
            'degree_used': self.degree_used_,
            'feature_expansion_ratio': len(selected_coefficients) / self.n_features_in_,
            'term_categories': term_analysis,
            'coefficients': selected_coefficients,
            'feature_names': selected_names,
            'importance_scores': importance_scores,
            'normalized_importance': normalized_importance,
            'importance_ranking': importance_ranking,
            'top_terms': [(selected_names[i], selected_coefficients[i], importance_scores[i]) 
                         for i in importance_ranking[:10]]
        }
    
    def _categorize_polynomial_terms(self, feature_names, coefficients):
        """Categorize polynomial terms by type"""
        categories = {
            'bias': {'names': [], 'coefficients': [], 'count': 0},
            'linear': {'names': [], 'coefficients': [], 'count': 0},
            'quadratic': {'names': [], 'coefficients': [], 'count': 0},
            'cubic': {'names': [], 'coefficients': [], 'count': 0},
            'higher_order': {'names': [], 'coefficients': [], 'count': 0},
            'interactions': {'names': [], 'coefficients': [], 'count': 0}
        }
        
        for name, coef in zip(feature_names, coefficients):
            name_lower = name.lower()
            
            if 'bias' in name_lower or name_lower == 'intercept':
                categories['bias']['names'].append(name)
                categories['bias']['coefficients'].append(coef)
                categories['bias']['count'] += 1
            elif '*' in name and '^' not in name:
                # Interaction terms (multiple variables, no powers)
                categories['interactions']['names'].append(name)
                categories['interactions']['coefficients'].append(coef)
                categories['interactions']['count'] += 1
            elif '^2' in name and '*' not in name:
                # Pure quadratic terms
                categories['quadratic']['names'].append(name)
                categories['quadratic']['coefficients'].append(coef)
                categories['quadratic']['count'] += 1
            elif '^3' in name and '*' not in name:
                # Pure cubic terms
                categories['cubic']['names'].append(name)
                categories['cubic']['coefficients'].append(coef)
                categories['cubic']['count'] += 1
            elif '^' in name:
                # Higher order terms
                categories['higher_order']['names'].append(name)
                categories['higher_order']['coefficients'].append(coef)
                categories['higher_order']['count'] += 1
            else:
                # Linear terms (original features)
                categories['linear']['names'].append(name)
                categories['linear']['coefficients'].append(coef)
                categories['linear']['count'] += 1
        
        return categories
    
    def _analyze_overfitting(self):
        """Analyze potential overfitting in the polynomial model"""
        if not self.detect_overfitting:
            return
        
        try:
            # Training score
            train_predictions = self.predict(self.X_original_)
            train_r2 = r2_score(self.y_original_, train_predictions)
            train_mse = mean_squared_error(self.y_original_, train_predictions)
            
            # Cross-validation score
            cv_scores = cross_val_score(
                self, self.X_original_, self.y_original_,
                cv=self.cv_folds, scoring='r2'
            )
            cv_r2_mean = np.mean(cv_scores)
            cv_r2_std = np.std(cv_scores)
            
            # Overfitting metrics
            overfitting_gap = train_r2 - cv_r2_mean
            
            # Complexity metrics
            n_poly_features = len(self.model_.coef_)
            complexity_ratio = n_poly_features / len(self.y_original_)
            
            # Overfitting indicators
            is_overfitting = overfitting_gap > self.overfitting_threshold
            high_complexity = complexity_ratio > 0.1
            
            # Model complexity analysis
            complexity_analysis = self._analyze_model_complexity()
            
            self.overfitting_analysis_ = {
                'train_r2': train_r2,
                'cv_r2_mean': cv_r2_mean,
                'cv_r2_std': cv_r2_std,
                'overfitting_gap': overfitting_gap,
                'overfitting_threshold': self.overfitting_threshold,
                'is_overfitting': is_overfitting,
                'complexity_ratio': complexity_ratio,
                'high_complexity': high_complexity,
                'n_polynomial_features': n_poly_features,
                'n_samples': len(self.y_original_),
                'complexity_analysis': complexity_analysis,
                'overfitting_risk': self._assess_overfitting_risk(overfitting_gap, complexity_ratio),
                'recommendations': self._get_overfitting_recommendations(is_overfitting, high_complexity)
            }
            
        except Exception as e:
            self.overfitting_analysis_ = {
                'error': f'Could not perform overfitting analysis: {str(e)}'
            }
    
    def _analyze_model_complexity(self):
        """Analyze model complexity metrics"""
        coefficients = self.model_.coef_
        
        # Coefficient statistics
        coef_stats = {
            'mean_abs_coef': np.mean(np.abs(coefficients)),
            'std_coef': np.std(coefficients),
            'max_abs_coef': np.max(np.abs(coefficients)),
            'coef_range': np.max(coefficients) - np.min(coefficients),
            'effective_parameters': np.sum(np.abs(coefficients) > 1e-8)
        }
        
        # Polynomial degree impact
        degree_impact = {
            'degree_used': self.degree_used_,
            'theoretical_features': self._calculate_theoretical_features(),
            'actual_features': len(coefficients),
            'feature_reduction': 1 - (len(coefficients) / self._calculate_theoretical_features())
        }
        
        return {
            'coefficient_statistics': coef_stats,
            'degree_impact': degree_impact,
            'regularization_strength': self.alpha_used_ if self.use_regularization else 0
        }
    
    def _calculate_theoretical_features(self):
        """Calculate theoretical number of polynomial features"""
        from math import comb
        n = self.n_features_in_
        d = self.degree_used_
        
        if self.interaction_only:
            # Only interaction terms
            return sum(comb(n, k) for k in range(1, min(d + 1, n + 1)))
        else:
            # All polynomial terms up to degree d
            return comb(n + d, d)
    
    def _assess_overfitting_risk(self, overfitting_gap, complexity_ratio):
        """Assess overfitting risk level"""
        if overfitting_gap > 0.2 or complexity_ratio > 0.2:
            return "High"
        elif overfitting_gap > 0.1 or complexity_ratio > 0.1:
            return "Medium"
        else:
            return "Low"
    
    def _get_overfitting_recommendations(self, is_overfitting, high_complexity):
        """Get recommendations for overfitting prevention"""
        recommendations = []
        
        if is_overfitting:
            recommendations.append("Consider using regularization (Ridge/Lasso)")
            recommendations.append("Reduce polynomial degree")
            recommendations.append("Increase training data size")
            recommendations.append("Use cross-validation for model selection")
        
        if high_complexity:
            recommendations.append("Enable feature selection")
            recommendations.append("Use interaction_only=True to reduce features")
            recommendations.append("Consider ensemble methods")
        
        if not recommendations:
            recommendations.append("Model complexity appears appropriate")
        
        return recommendations
    
    def _analyze_feature_importance(self):
        """Analyze feature importance for polynomial terms"""
        if not self.compute_feature_importance:
            return
        
        coefficients = self.model_.coef_
        
        # Calculate importance as absolute coefficient values
        importance = np.abs(coefficients)
        
        # Normalize importance
        normalized_importance = importance / np.sum(importance) if np.sum(importance) > 0 else importance
        
        # Create feature importance dictionary
        feature_names = self.polynomial_feature_names_
        if self.feature_selection and hasattr(self, 'selected_features_'):
            feature_names = [feature_names[i] for i in self.selected_features_]
        
        feature_importance = {}
        for i, (name, coef, imp, norm_imp) in enumerate(zip(feature_names, coefficients, importance, normalized_importance)):
            feature_importance[name] = {
                'coefficient': coef,
                'absolute_coefficient': imp,
                'normalized_importance': norm_imp,
                'rank': i + 1,
                'term_type': self._classify_term_type(name)
            }
        
        # Sort by importance
        sorted_features = sorted(
            feature_importance.items(), 
            key=lambda x: x[1]['absolute_coefficient'], 
            reverse=True
        )
        
        # Update ranks
        for rank, (name, info) in enumerate(sorted_features):
            feature_importance[name]['rank'] = rank + 1
        
        # Analyze importance by term type
        importance_by_type = self._analyze_importance_by_term_type(feature_importance)
        
        self.feature_importance_analysis_ = {
            'feature_importance': feature_importance,
            'sorted_features': [name for name, _ in sorted_features],
            'sorted_importance': [info['normalized_importance'] for _, info in sorted_features],
            'importance_by_type': importance_by_type,
            'top_features': sorted_features[:10],
            'polynomial_contribution': self._analyze_polynomial_contribution(feature_importance)
        }
    
    def _classify_term_type(self, term_name):
        """Classify polynomial term type"""
        name_lower = term_name.lower()
        
        if 'bias' in name_lower or name_lower == 'intercept':
            return 'bias'
        elif '*' in term_name and '^' not in term_name:
            return 'interaction'
        elif '^2' in term_name:
            return 'quadratic'
        elif '^3' in term_name:
            return 'cubic'
        elif '^' in term_name:
            return 'higher_order'
        else:
            return 'linear'
    
    def _analyze_importance_by_term_type(self, feature_importance):
        """Analyze feature importance grouped by term type"""
        type_importance = {}
        
        for name, info in feature_importance.items():
            term_type = info['term_type']
            if term_type not in type_importance:
                type_importance[term_type] = {
                    'total_importance': 0,
                    'count': 0,
                    'terms': []
                }
            
            type_importance[term_type]['total_importance'] += info['normalized_importance']
            type_importance[term_type]['count'] += 1
            type_importance[term_type]['terms'].append(name)
        
        # Calculate average importance per type
        for term_type in type_importance:
            count = type_importance[term_type]['count']
            type_importance[term_type]['average_importance'] = (
                type_importance[term_type]['total_importance'] / count if count > 0 else 0
            )
        
        return type_importance
    
    def _analyze_polynomial_contribution(self, feature_importance):
        """Analyze overall contribution of polynomial vs linear terms"""
        linear_importance = 0
        polynomial_importance = 0
        
        for name, info in feature_importance.items():
            if info['term_type'] == 'linear':
                linear_importance += info['normalized_importance']
            else:
                polynomial_importance += info['normalized_importance']
        
        total_importance = linear_importance + polynomial_importance
        
        return {
            'linear_contribution': linear_importance,
            'polynomial_contribution': polynomial_importance,
            'linear_percentage': (linear_importance / total_importance * 100) if total_importance > 0 else 0,
            'polynomial_percentage': (polynomial_importance / total_importance * 100) if total_importance > 0 else 0,
            'polynomial_benefit': polynomial_importance > linear_importance
        }
    
    def _analyze_bias_variance(self):
        """Analyze bias-variance tradeoff"""
        if not self.bias_variance_analysis:
            return
        
        try:
            # This is a simplified bias-variance analysis
            # In practice, you'd need multiple bootstrap samples
                        
            # Simplified analysis using cross-validation variance
            cv_scores = cross_val_score(
                self, self.X_original_, self.y_original_,
                cv=self.cv_folds, scoring='r2'
            )
            
            cv_variance = np.var(cv_scores)
            cv_mean = np.mean(cv_scores)
            
            # Estimate bias (simplified)
            train_score = self.score(self.X_original_, self.y_original_)
            estimated_bias = abs(1.0 - train_score)  # Simplified bias estimate
            
            self.bias_variance_analysis_ = {
                'cv_scores': cv_scores,
                'cv_mean': cv_mean,
                'cv_variance': cv_variance,
                'estimated_bias': estimated_bias,
                'bias_variance_tradeoff': {
                    'degree': self.degree_used_,
                    'variance_indicator': cv_variance,
                    'bias_indicator': estimated_bias,
                    'balance_assessment': self._assess_bias_variance_balance(estimated_bias, cv_variance)
                }
            }
            
        except Exception as e:
            self.bias_variance_analysis_ = {
                'error': f'Could not perform bias-variance analysis: {str(e)}'
            }
    
    def _assess_bias_variance_balance(self, bias, variance):
        """Assess bias-variance balance"""
        if bias > 0.2 and variance < 0.01:
            return "High bias, low variance - consider higher degree"
        elif bias < 0.1 and variance > 0.05:
            return "Low bias, high variance - consider regularization or lower degree"
        elif bias < 0.1 and variance < 0.02:
            return "Good balance - model is well-tuned"
        else:
            return "Moderate bias-variance tradeoff"
    
    def _analyze_curvature(self):
        """Analyze curvature and non-linearity patterns"""
        if not self.curvature_analysis or self.n_features_in_ > 3:
            # Skip curvature analysis for high-dimensional data
            return
        
        try:
            # This analysis works best for 1D or 2D data
            if self.n_features_in_ == 1:
                self._analyze_1d_curvature()
            elif self.n_features_in_ == 2:
                self._analyze_2d_curvature()
            else:
                self.curvature_analysis_ = {
                    'note': 'Curvature analysis skipped for high-dimensional data'
                }
                
        except Exception as e:
            self.curvature_analysis_ = {
                'error': f'Could not perform curvature analysis: {str(e)}'
            }
    
    def _analyze_1d_curvature(self):
        """Analyze curvature for 1D polynomial"""
        X = self.X_original_[:, 0]
        y = self.y_original_
        
        # Create fine-grained prediction for curvature analysis
        X_range = np.linspace(np.min(X), np.max(X), 1000).reshape(-1, 1)
        y_pred = self.predict(X_range)
        
        # Calculate numerical derivatives
        dx = X_range[1, 0] - X_range[0, 0]
        dy_dx = np.gradient(y_pred, dx)
        d2y_dx2 = np.gradient(dy_dx, dx)
        
        # Find inflection points (where second derivative changes sign)
        inflection_points = []
        for i in range(1, len(d2y_dx2) - 1):
            if d2y_dx2[i-1] * d2y_dx2[i+1] < 0:
                inflection_points.append(X_range[i, 0])
        
        # Calculate curvature statistics
        curvature = np.abs(d2y_dx2) / (1 + dy_dx**2)**(3/2)
        
        self.curvature_analysis_ = {
            'dimension': '1D',
            'X_range': X_range.flatten(),
            'y_predictions': y_pred,
            'first_derivative': dy_dx,
            'second_derivative': d2y_dx2,
            'curvature': curvature,
            'inflection_points': inflection_points,
            'max_curvature': np.max(curvature),
            'mean_curvature': np.mean(curvature),
            'curvature_variation': np.std(curvature),
            'monotonic_regions': self._find_monotonic_regions(dy_dx),
            'concavity_changes': len(inflection_points)
        }
    
    def _analyze_2d_curvature(self):
        """Analyze curvature for 2D polynomial (simplified)"""
        # For 2D, we'll analyze curvature along each dimension separately
        self.curvature_analysis_ = {
            'dimension': '2D',
            'note': 'Simplified 2D curvature analysis - analyze each dimension separately'
        }
    
    def _find_monotonic_regions(self, derivative):
        """Find monotonic regions in the function"""
        regions = []
        current_region_start = 0
        current_sign = np.sign(derivative[0])
        
        for i, deriv in enumerate(derivative):
            if np.sign(deriv) != current_sign and deriv != 0:
                regions.append({
                    'start_idx': current_region_start,
                    'end_idx': i - 1,
                    'type': 'increasing' if current_sign > 0 else 'decreasing'
                })
                current_region_start = i
                current_sign = np.sign(deriv)
        
        # Add final region
        regions.append({
            'start_idx': current_region_start,
            'end_idx': len(derivative) - 1,
            'type': 'increasing' if current_sign > 0 else 'decreasing'
        })
        
        return regions
    
    def get_hyperparameter_config(self, key_prefix: str) -> Dict[str, Any]:
        """Generate Streamlit UI for hyperparameters"""
        
        # Create tabs for different configuration aspects
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "Polynomial Config", "Regularization", "Optimization", "Advanced Options", "Algorithm Info"
        ])
        
        with tab1:
            st.markdown("**Polynomial Regression Configuration**")
            
            col1, col2 = st.columns(2)
            
            with col1:
                auto_degree = st.checkbox(
                    "Automatic Degree Selection",
                    value=self.auto_degree,
                    help="Automatically find optimal polynomial degree using cross-validation",
                    key=f"{key_prefix}_auto_degree"
                )
                
                if not auto_degree:
                    degree = st.number_input(
                        "Polynomial Degree:",
                        value=self.degree,
                        min_value=1,
                        max_value=15,
                        step=1,
                        help="Degree of polynomial features",
                        key=f"{key_prefix}_degree"
                    )
                else:
                    degree = self.degree
                
                include_bias = st.checkbox(
                    "Include Bias Term",
                    value=self.include_bias,
                    help="Include bias (constant) term in polynomial features",
                    key=f"{key_prefix}_include_bias"
                )
                
                interaction_only = st.checkbox(
                    "Interaction Terms Only",
                    value=self.interaction_only,
                    help="Only include interaction terms (no pure powers)",
                    key=f"{key_prefix}_interaction_only"
                )
            
            with col2:
                normalize_features = st.checkbox(
                    "Normalize Features",
                    value=self.normalize_features,
                    help="Apply StandardScaler to features (highly recommended)",
                    key=f"{key_prefix}_normalize_features"
                )
                
                if auto_degree:
                    max_degree = st.number_input(
                        "Maximum Degree to Test:",
                        value=self.max_degree,
                        min_value=2,
                        max_value=20,
                        step=1,
                        help="Maximum polynomial degree to consider",
                        key=f"{key_prefix}_max_degree"
                    )
                    
                    degree_selection_method = st.selectbox(
                        "Degree Selection Method:",
                        options=['cv', 'validation_curve', 'bias_variance'],
                        index=['cv', 'validation_curve', 'bias_variance'].index(self.degree_selection_method),
                        help="Method for optimal degree selection",
                        key=f"{key_prefix}_degree_selection_method"
                    )
                else:
                    max_degree = self.max_degree
                    degree_selection_method = self.degree_selection_method
            
            # Polynomial complexity warning
            if not auto_degree and degree > 5:
                estimated_features = self._estimate_polynomial_features(degree)
                st.warning(f"⚠️ Degree {degree} may create ~{estimated_features} features. Consider enabling regularization!")
        
        with tab2:
            st.markdown("**Regularization Configuration**")
            
            col1, col2 = st.columns(2)
            
            with col1:
                use_regularization = st.checkbox(
                    "Enable Regularization",
                    value=self.use_regularization,
                    help="Use Ridge regularization to prevent overfitting",
                    key=f"{key_prefix}_use_regularization"
                )
                
                if use_regularization:
                    auto_alpha = st.checkbox(
                        "Automatic Alpha Selection",
                        value=self.auto_alpha,
                        help="Automatically find optimal regularization strength",
                        key=f"{key_prefix}_auto_alpha"
                    )
                    
                    if not auto_alpha:
                        alpha = st.number_input(
                            "Regularization Strength (Alpha):",
                            value=self.alpha,
                            min_value=1e-6,
                            max_value=100.0,
                            step=0.1,
                            format="%.6f",
                            help="Ridge regularization parameter",
                            key=f"{key_prefix}_alpha"
                        )
                    else:
                        alpha = self.alpha
                else:
                    auto_alpha = False
                    alpha = self.alpha
            
            with col2:
                overfitting_threshold = st.number_input(
                    "Overfitting Threshold:",
                    value=self.overfitting_threshold,
                    min_value=0.01,
                    max_value=0.5,
                    step=0.01,
                    help="Threshold for detecting overfitting (train - validation score)",
                    key=f"{key_prefix}_overfitting_threshold"
                )
                
                cv_folds = st.number_input(
                    "CV Folds:",
                    value=self.cv_folds,
                    min_value=3,
                    max_value=10,
                    step=1,
                    help="Number of cross-validation folds",
                    key=f"{key_prefix}_cv_folds"
                )
        
        with tab3:
            st.markdown("**Feature Engineering**")
            
            col1, col2 = st.columns(2)
            
            with col1:
                feature_selection = st.checkbox(
                    "Enable Feature Selection",
                    value=self.feature_selection,
                    help="Remove irrelevant polynomial features",
                    key=f"{key_prefix}_feature_selection"
                )
                
                if feature_selection:
                    feature_selection_threshold = st.number_input(
                        "Feature Selection Threshold:",
                        value=self.feature_selection_threshold,
                        min_value=0.0,
                        max_value=1.0,
                        step=0.01,
                        help="Minimum correlation with target to keep feature",
                        key=f"{key_prefix}_feature_selection_threshold"
                    )
                else:
                    feature_selection_threshold = self.feature_selection_threshold
            
            with col2:
                remove_low_variance = st.checkbox(
                    "Remove Low Variance Features",
                    value=self.remove_low_variance,
                    help="Remove features with very low variance",
                    key=f"{key_prefix}_remove_low_variance"
                )
                
                if remove_low_variance:
                    variance_threshold = st.number_input(
                        "Variance Threshold:",
                        value=self.variance_threshold,
                        min_value=1e-10,
                        max_value=1e-4,
                        step=1e-9,
                        format="%.1e",
                        help="Minimum variance to keep feature",
                        key=f"{key_prefix}_variance_threshold"
                    )
                else:
                    variance_threshold = self.variance_threshold
        
        with tab4:
            st.markdown("**Analysis Options**")
            
            col1, col2 = st.columns(2)
            
            with col1:
                analyze_polynomial_terms = st.checkbox(
                    "Analyze Polynomial Terms",
                    value=self.analyze_polynomial_terms,
                    help="Detailed analysis of polynomial term contributions",
                    key=f"{key_prefix}_analyze_polynomial_terms"
                )
                
                detect_overfitting = st.checkbox(
                    "Detect Overfitting",
                    value=self.detect_overfitting,
                    help="Analyze model for overfitting patterns",
                    key=f"{key_prefix}_detect_overfitting"
                )
                
                compute_feature_importance = st.checkbox(
                    "Compute Feature Importance",
                    value=self.compute_feature_importance,
                    help="Calculate importance of polynomial terms",
                    key=f"{key_prefix}_compute_feature_importance"
                )
            
            with col2:
                bias_variance_analysis = st.checkbox(
                    "Bias-Variance Analysis",
                    value=self.bias_variance_analysis,
                    help="Analyze bias-variance tradeoff",
                    key=f"{key_prefix}_bias_variance_analysis"
                )
                
                curvature_analysis = st.checkbox(
                    "Curvature Analysis",
                    value=self.curvature_analysis,
                    help="Analyze function curvature (works best for 1D/2D data)",
                    key=f"{key_prefix}_curvature_analysis"
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
            **Polynomial Regression** - Capturing Non-Linear Relationships:
            • 📈 Extends linear regression to capture polynomial patterns
            • 🔢 Creates polynomial features (x², x³, x₁x₂, etc.)
            • 🎯 Perfect for curved, non-linear relationships
            • 🛡️ Ridge regularization prevents overfitting
            • 📊 Automatic degree selection via cross-validation
            • 🔍 Comprehensive overfitting detection
            
            **Mathematical Foundation:**
            • y = β₀ + β₁x + β₂x² + β₃x³ + ... + βₙxⁿ + ε
            • Creates polynomial features up to specified degree
            • Uses least squares optimization (with optional regularization)
            """)
            
            # When to use Polynomial Regression
            if st.button("🎯 When to Use Polynomial Regression", key=f"{key_prefix}_when_to_use"):
                st.markdown("""
                **Ideal Use Cases:**
                
                **Problem Characteristics:**
                • Non-linear relationships between features and target
                • Curved patterns in scatter plots
                • Physical phenomena following polynomial laws
                • Growth/decay patterns
                
                **Data Characteristics:**
                • Continuous features and target
                • Clear non-linear trends
                • Not too high-dimensional (feature explosion risk)
                • Sufficient data for chosen polynomial degree
                
                **Examples:**
                • Physics: trajectory, acceleration, growth rates
                • Economics: diminishing returns, cost curves
                • Biology: population growth, dose-response
                • Engineering: stress-strain relationships
                """)
            
            # Advantages and limitations
            if st.button("⚖️ Advantages & Limitations", key=f"{key_prefix}_pros_cons"):
                st.markdown("""
                **Advantages:**
                ✅ Captures non-linear relationships naturally
                ✅ Still interpretable (polynomial terms have meaning)
                ✅ No assumptions about specific functional form
                ✅ Can model complex curves with higher degrees
                ✅ Built on solid linear algebra foundation
                ✅ Automatic degree selection available
                ✅ Regularization prevents overfitting
                
                **Limitations:**
                ❌ Feature explosion with high degrees/dimensions
                ❌ Prone to overfitting without regularization
                ❌ Extrapolation can be very poor
                ❌ Numerical instability with high degrees
                ❌ May not capture all types of non-linearity
                ❌ Curse of dimensionality in high dimensions
                """)
            
            # Polynomial degree guide
            if st.button("📊 Polynomial Degree Guide", key=f"{key_prefix}_degree_guide"):
                st.markdown("""
                **Choosing Polynomial Degree:**
                
                **Degree 1 (Linear):**
                • Straight line relationships
                • Use when no curvature is evident
                
                **Degree 2 (Quadratic):**
                • U-shaped or inverted U-shaped relationships
                • Parabolic patterns, optimization problems
                • Most common choice for mild non-linearity
                
                **Degree 3 (Cubic):**
                • S-shaped curves
                • Inflection points, growth curves
                • Good for biological/economic models
                
                **Degree 4+ (Higher Order):**
                • Complex multi-modal patterns
                • Use with caution (overfitting risk)
                • Requires large datasets
                • Consider regularization
                
                **Auto-Selection Tips:**
                • Use cross-validation for objective selection
                • Monitor overfitting gap (train vs validation)
                • Consider complexity vs performance trade-off
                """)
            
            # Best practices
            if st.button("🎯 Best Practices", key=f"{key_prefix}_best_practices"):
                st.markdown("""
                **Polynomial Regression Best Practices:**
                
                **Data Preparation:**
                1. **Always normalize/standardize features** (critical!)
                2. Plot data to visualize non-linear patterns
                3. Check for outliers (they heavily influence polynomials)
                4. Ensure sufficient data for chosen degree
                
                **Degree Selection:**
                1. Start with degree 2-3, increase gradually
                2. Use cross-validation for objective selection
                3. Monitor training vs validation performance
                4. Consider domain knowledge about relationships
                
                **Overfitting Prevention:**
                1. Use regularization (Ridge recommended)
                2. Enable feature selection for high degrees
                3. Use sufficient training data
                4. Validate on independent test set
                
                **Model Interpretation:**
                1. Examine polynomial term coefficients
                2. Plot fitted curves for visualization
                3. Identify important polynomial terms
                4. Check for reasonable extrapolation behavior
                """)
        
        return {
            "degree": degree,
            "include_bias": include_bias,
            "interaction_only": interaction_only,
            "normalize_features": normalize_features,
            "use_regularization": use_regularization,
            "alpha": alpha,
            "auto_alpha": auto_alpha,
            "auto_degree": auto_degree,
            "max_degree": max_degree,
            "degree_selection_method": degree_selection_method,
            "overfitting_threshold": overfitting_threshold,
            "cv_folds": cv_folds,
            "feature_selection": feature_selection,
            "feature_selection_threshold": feature_selection_threshold,
            "remove_low_variance": remove_low_variance,
            "variance_threshold": variance_threshold,
            "analyze_polynomial_terms": analyze_polynomial_terms,
            "detect_overfitting": detect_overfitting,
            "compute_feature_importance": compute_feature_importance,
            "bias_variance_analysis": bias_variance_analysis,
            "curvature_analysis": curvature_analysis,
            "random_state": random_state
        }
    
    def _estimate_polynomial_features(self, degree):
        """Estimate number of polynomial features"""
        from math import comb
        n = max(self.n_features_in_, 2)  # Assume at least 2 features for estimation
        
        if self.interaction_only:
            return sum(comb(n, k) for k in range(1, min(degree + 1, n + 1)))
        else:
            return comb(n + degree, degree)
    
    def create_model_instance(self, hyperparameters: Dict[str, Any]):
        """Create model instance with given hyperparameters"""
        return PolynomialRegressionPlugin(
            degree=hyperparameters.get("degree", self.degree),
            include_bias=hyperparameters.get("include_bias", self.include_bias),
            interaction_only=hyperparameters.get("interaction_only", self.interaction_only),
            normalize_features=hyperparameters.get("normalize_features", self.normalize_features),
            use_regularization=hyperparameters.get("use_regularization", self.use_regularization),
            alpha=hyperparameters.get("alpha", self.alpha),
            auto_alpha=hyperparameters.get("auto_alpha", self.auto_alpha),
            auto_degree=hyperparameters.get("auto_degree", self.auto_degree),
            max_degree=hyperparameters.get("max_degree", self.max_degree),
            degree_selection_method=hyperparameters.get("degree_selection_method", self.degree_selection_method),
            overfitting_threshold=hyperparameters.get("overfitting_threshold", self.overfitting_threshold),
            cv_folds=hyperparameters.get("cv_folds", self.cv_folds),
            feature_selection=hyperparameters.get("feature_selection", self.feature_selection),
            feature_selection_threshold=hyperparameters.get("feature_selection_threshold", self.feature_selection_threshold),
            remove_low_variance=hyperparameters.get("remove_low_variance", self.remove_low_variance),
            variance_threshold=hyperparameters.get("variance_threshold", self.variance_threshold),
            analyze_polynomial_terms=hyperparameters.get("analyze_polynomial_terms", self.analyze_polynomial_terms),
            detect_overfitting=hyperparameters.get("detect_overfitting", self.detect_overfitting),
            compute_feature_importance=hyperparameters.get("compute_feature_importance", self.compute_feature_importance),
            bias_variance_analysis=hyperparameters.get("bias_variance_analysis", self.bias_variance_analysis),
            curvature_analysis=hyperparameters.get("curvature_analysis", self.curvature_analysis),
            random_state=hyperparameters.get("random_state", self.random_state)
        )
    
    def preprocess_data(self, X, y=None, training=True):
        """Preprocess data for Polynomial Regression"""
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
        """Check if Polynomial Regression is compatible with the given data"""
        # Check minimum samples
        if X.shape[0] < self._min_samples_required:
            return False, f"Polynomial regression requires at least {self._min_samples_required} samples, got {X.shape[0]}"
        
        # Check for regression targets
        if y is not None:
            if not np.issubdtype(y.dtype, np.number):
                return False, "Polynomial regression requires continuous numerical target values"
            
            # Check for sufficient variance in target
            if np.var(y) == 0:
                return False, "Target variable has zero variance (all values are the same)"
            
            n_samples, n_features = X.shape
            
            advantages = []
            considerations = []
            
            # Feature explosion analysis
            max_test_degree = min(self.max_degree if self.auto_degree else self.degree, 10)
            estimated_features = self._estimate_polynomial_features_static(n_features, max_test_degree)
            
            if estimated_features > n_samples:
                considerations.append(f"Polynomial features ({estimated_features}) > samples ({n_samples}) - high overfitting risk")
            elif estimated_features > n_samples * 0.5:
                considerations.append(f"Many polynomial features ({estimated_features}) - regularization recommended")
            else:
                advantages.append(f"Reasonable feature expansion ({estimated_features} features)")
            
            # Sample size assessment
            if n_samples >= estimated_features * 10:
                advantages.append(f"Excellent sample size ({n_samples}) for polynomial degree {max_test_degree}")
            elif n_samples >= estimated_features * 3:
                advantages.append(f"Good sample size ({n_samples}) for polynomial modeling")
            elif n_samples >= estimated_features:
                considerations.append(f"Adequate sample size ({n_samples}) but overfitting risk exists")
            else:
                considerations.append(f"Small sample size ({n_samples}) for polynomial degree {max_test_degree}")
            
            # Dimensionality assessment
            if n_features > 10:
                considerations.append(f"High dimensionality ({n_features}) - polynomial explosion risk")
            elif n_features > 5:
                considerations.append(f"Moderate dimensionality ({n_features}) - monitor feature count")
            else:
                advantages.append(f"Low dimensionality ({n_features}) - suitable for polynomial expansion")
            
            # Non-linearity detection (simplified)
            try:
                # Simple check for potential non-linear patterns
                if n_features == 1:
                    # For 1D, check if linear fit is poor
                    from sklearn.linear_model import LinearRegression
                    linear_model = LinearRegression()
                    linear_model.fit(X, y)
                    linear_r2 = linear_model.score(X, y)
                    
                    if linear_r2 < 0.7:
                        advantages.append(f"Poor linear fit (R²={linear_r2:.2f}) - polynomial may help")
                    elif linear_r2 < 0.9:
                        advantages.append(f"Moderate linear fit (R²={linear_r2:.2f}) - polynomial could improve")
                    else:
                        considerations.append(f"Strong linear fit (R²={linear_r2:.2f}) - polynomial may not be needed")
                else:
                    advantages.append("Multi-dimensional data - polynomial can capture interactions")
                    
            except:
                pass
            
            # Feature scaling check
            try:
                feature_scales = np.std(X, axis=0)
                max_scale_ratio = np.max(feature_scales) / np.min(feature_scales) if np.min(feature_scales) > 0 else np.inf
                
                if max_scale_ratio > 100:
                    considerations.append("Features have very different scales - normalization critical for polynomials")
                elif max_scale_ratio > 10:
                    considerations.append("Features have different scales - normalization recommended")
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
                f"📊 Suitability for Polynomial Regression: {suitability}"
            ]
            
            if advantages:
                message_parts.append("🎯 Advantages: " + "; ".join(advantages))
            
            if considerations:
                message_parts.append("💡 Considerations: " + "; ".join(considerations))
            
            return True, " | ".join(message_parts)
        
        return True, f"Compatible with {X.shape[0]} samples and {X.shape[1]} features"
    
    def _estimate_polynomial_features_static(self, n_features, degree):
        """Static method to estimate polynomial features"""
        from math import comb
        
        if self.interaction_only:
            return sum(comb(n_features, k) for k in range(1, min(degree + 1, n_features + 1)))
        else:
            return comb(n_features + degree, degree)
    
    def get_feature_importance(self) -> Optional[Dict[str, Any]]:
        """Get feature importance based on polynomial term coefficients"""
        if not self.is_fitted_:
            return None
        
        coefficients = self.model_.coef_
        
        # Get polynomial feature names
        feature_names = self.polynomial_feature_names_
        if self.feature_selection and hasattr(self, 'selected_features_'):
            feature_names = [feature_names[i] for i in self.selected_features_]
        
        # Calculate importance as absolute coefficient values
        importance = np.abs(coefficients)
        
        # Normalize importance
        normalized_importance = importance / np.sum(importance) if np.sum(importance) > 0 else importance
        
        # Create feature importance dictionary
        feature_importance = {}
        for i, (name, coef, imp, norm_imp) in enumerate(zip(feature_names, coefficients, importance, normalized_importance)):
            term_type = self._classify_term_type(name)
            
            feature_importance[name] = {
                'coefficient': coef,
                'absolute_coefficient': imp,
                'normalized_importance': norm_imp,
                'term_type': term_type,
                'polynomial_degree': self._get_term_degree(name),
                'rank': i + 1
            }
        
        # Sort by importance
        sorted_features = sorted(
            feature_importance.items(), 
            key=lambda x: x[1]['absolute_coefficient'], 
            reverse=True
        )
        
        # Update ranks
        for rank, (name, info) in enumerate(sorted_features):
            feature_importance[name]['rank'] = rank + 1
        
        # Analyze contribution by polynomial degree
        degree_contribution = self._analyze_degree_contribution(feature_importance)
        
        return {
            'feature_importance': feature_importance,
            'sorted_features': [name for name, _ in sorted_features],
            'sorted_importance': [info['normalized_importance'] for _, info in sorted_features],
            'degree_contribution': degree_contribution,
            'polynomial_info': {
                'degree_used': self.degree_used_,
                'total_polynomial_features': len(coefficients),
                'original_features': self.n_features_in_,
                'feature_expansion_ratio': len(coefficients) / self.n_features_in_,
                'regularization_used': self.use_regularization,
                'alpha_used': self.alpha_used_
            },
            'top_linear_terms': self._get_top_terms_by_type(feature_importance, 'linear'),
            'top_polynomial_terms': self._get_top_terms_by_type(feature_importance, ['quadratic', 'cubic', 'higher_order']),
            'top_interaction_terms': self._get_top_terms_by_type(feature_importance, 'interaction'),
            'interpretation': f'Polynomial feature importance (degree {self.degree_used_}, {len(coefficients)} features)'
        }
    
    def _get_term_degree(self, term_name):
        """Get the degree of a polynomial term"""
        if 'bias' in term_name.lower() or term_name.lower() == 'intercept':
            return 0
        elif '*' not in term_name and '^' not in term_name:
            return 1  # Linear term
        elif '^' in term_name:
            # Extract highest power
            import re
            powers = re.findall(r'\^(\d+)', term_name)
            if powers:
                return max(int(p) for p in powers)
            return 1
        elif '*' in term_name:
            # Interaction term - count variables
            return len(term_name.split(' * '))
        else:
            return 1
    
    def _analyze_degree_contribution(self, feature_importance):
        """Analyze contribution by polynomial degree"""
        degree_contrib = {}
        
        for name, info in feature_importance.items():
            degree = info['polynomial_degree']
            if degree not in degree_contrib:
                degree_contrib[degree] = {
                    'total_importance': 0,
                    'count': 0,
                    'terms': []
                }
            
            degree_contrib[degree]['total_importance'] += info['normalized_importance']
            degree_contrib[degree]['count'] += 1
            degree_contrib[degree]['terms'].append(name)
        
        # Calculate percentages and average importance
        for degree in degree_contrib:
            count = degree_contrib[degree]['count']
            degree_contrib[degree]['percentage'] = degree_contrib[degree]['total_importance'] * 100
            degree_contrib[degree]['average_importance'] = (
                degree_contrib[degree]['total_importance'] / count if count > 0 else 0
            )
        
        return degree_contrib
    
    def _get_top_terms_by_type(self, feature_importance, term_types):
        """Get top terms of specific types"""
        if isinstance(term_types, str):
            term_types = [term_types]
        
        filtered_terms = [
            (name, info) for name, info in feature_importance.items() 
            if info['term_type'] in term_types
        ]
        
        # Sort by importance
        filtered_terms.sort(key=lambda x: x[1]['absolute_coefficient'], reverse=True)
        
        return [(name, info['coefficient'], info['normalized_importance']) for name, info in filtered_terms[:5]]
    
    def get_model_params(self) -> Dict[str, Any]:
        """Get current model parameters"""
        if not self.is_fitted_:
            return {}
        
        return {
            "algorithm": "Polynomial Regression",
            "degree_used": self.degree_used_,
            "n_features": self.n_features_in_,
            "n_polynomial_features": len(self.model_.coef_),
            "feature_expansion_ratio": len(self.model_.coef_) / self.n_features_in_,
            "regularization_used": self.use_regularization,
            "alpha_used": self.alpha_used_,
            "include_bias": self.include_bias,
            "interaction_only": self.interaction_only,
            "normalize_features": self.normalize_features,
            "coefficients": self.model_.coef_.tolist(),
            "intercept": self.model_.intercept_ if hasattr(self.model_, 'intercept_') else 0,
            "feature_selection_used": self.feature_selection,
            "selected_features_count": len(self.selected_features_) if hasattr(self, 'selected_features_') else len(self.model_.coef_),
            "overfitting_threshold": self.overfitting_threshold,
            "auto_degree_selection": self.auto_degree,
            "degree_selection_method": self.degree_selection_method if self.auto_degree else "manual"
        }
    
    def get_polynomial_analysis(self) -> Dict[str, Any]:
        """Get comprehensive polynomial analysis results"""
        if not self.is_fitted_:
            return {"status": "Model not fitted"}
        
        return {
            "degree_analysis": self.degree_analysis_,
            "polynomial_analysis": self.polynomial_analysis_,
            "overfitting_analysis": self.overfitting_analysis_,
            "feature_importance_analysis": self.feature_importance_analysis_,
            "bias_variance_analysis": self.bias_variance_analysis_,
            "curvature_analysis": self.curvature_analysis_
        }
    
    def plot_polynomial_analysis(self, figsize=(15, 12)):
        """Plot comprehensive polynomial analysis"""
        if not self.is_fitted_:
            raise ValueError("Model must be fitted to plot polynomial analysis")
        
        fig, axes = plt.subplots(2, 3, figsize=figsize)
        
        # Plot 1: Degree selection (if available)
        ax1 = axes[0, 0]
        if 'cv_results' in self.degree_analysis_:
            cv_results = self.degree_analysis_['cv_results']
            degrees = [r['degree'] for r in cv_results]
            scores = [r['cv_score'] for r in cv_results]
            
            ax1.plot(degrees, scores, 'bo-', linewidth=2, markersize=6)
            ax1.axvline(x=self.degree_used_, color='red', linestyle='--', alpha=0.8,
                       label=f'Selected degree: {self.degree_used_}')
            ax1.set_xlabel('Polynomial Degree')
            ax1.set_ylabel('CV Score (R²)')
            ax1.set_title('Degree Selection via Cross-Validation')
            ax1.grid(True, alpha=0.3)
            ax1.legend()
        else:
            ax1.text(0.5, 0.5, f'Degree: {self.degree_used_}\n(Fixed)', 
                    ha='center', va='center', transform=ax1.transAxes)
            ax1.set_title('Polynomial Degree')
        
        # Plot 2: Feature importance by type
        ax2 = axes[0, 1]
        if self.feature_importance_analysis_:
            importance_by_type = self.feature_importance_analysis_['importance_by_type']
            types = list(importance_by_type.keys())
            importances = [importance_by_type[t]['total_importance'] for t in types]
            
            colors = plt.cm.Set3(np.linspace(0, 1, len(types)))
            wedges, texts, autotexts = ax2.pie(importances, labels=types, colors=colors,
                                              autopct='%1.1f%%', startangle=90)
            ax2.set_title('Feature Importance by Polynomial Type')
        else:
            ax2.text(0.5, 0.5, 'Feature importance\nnot available', 
                    ha='center', va='center', transform=ax2.transAxes)
            ax2.set_title('Feature Importance by Type')
        
        # Plot 3: Overfitting analysis
        ax3 = axes[0, 2]
        if 'train_r2' in self.overfitting_analysis_:
            train_r2 = self.overfitting_analysis_['train_r2']
            cv_r2 = self.overfitting_analysis_['cv_r2_mean']
            cv_std = self.overfitting_analysis_['cv_r2_std']
            
            x = ['Training', 'Cross-Validation']
            y = [train_r2, cv_r2]
            yerr = [0, cv_std]
            
            bars = ax3.bar(x, y, yerr=yerr, capsize=5, alpha=0.7, 
                          color=['lightcoral', 'lightblue'])
            ax3.set_ylabel('R² Score')
            ax3.set_title('Training vs Cross-Validation Performance')
            ax3.set_ylim(0, 1)
            
            # Add overfitting gap annotation
            gap = self.overfitting_analysis_['overfitting_gap']
            ax3.annotate(f'Gap: {gap:.3f}', xy=(0.5, max(y) + 0.05), 
                        ha='center', fontsize=10)
            
            # Color bars based on overfitting
            if self.overfitting_analysis_['is_overfitting']:
                bars[0].set_color('red')
                bars[0].set_alpha(0.8)
        else:
            ax3.text(0.5, 0.5, 'Overfitting analysis\nnot available', 
                    ha='center', va='center', transform=ax3.transAxes)
            ax3.set_title('Overfitting Analysis')
        
        # Plot 4: Polynomial coefficients
        ax4 = axes[1, 0]
        coefficients = self.model_.coef_
        feature_names = self.polynomial_feature_names_
        if self.feature_selection and hasattr(self, 'selected_features_'):
            feature_names = [feature_names[i] for i in self.selected_features_]
        
        # Show top 10 coefficients
        abs_coefs = np.abs(coefficients)
        top_indices = np.argsort(abs_coefs)[-10:][::-1]
        
        top_coefs = coefficients[top_indices]
        top_names = [feature_names[i] if i < len(feature_names) else f'term_{i}' 
                    for i in top_indices]
        
        colors = ['red' if c < 0 else 'blue' for c in top_coefs]
        bars = ax4.barh(range(len(top_coefs)), top_coefs, color=colors, alpha=0.7)
        ax4.set_yticks(range(len(top_coefs)))
        ax4.set_yticklabels(top_names, fontsize=8)
        ax4.set_xlabel('Coefficient Value')
        ax4.set_title('Top 10 Polynomial Coefficients')
        ax4.grid(True, alpha=0.3)
        
        # Plot 5: Feature expansion visualization
        ax5 = axes[1, 1]
        expansion_data = {
            'Original Features': self.n_features_in_,
            'Polynomial Features': len(coefficients),
            'Selected Features': len(self.selected_features_) if hasattr(self, 'selected_features_') else len(coefficients)
        }
        
        x = list(expansion_data.keys())
        y = list(expansion_data.values())
        colors = ['lightgreen', 'orange', 'lightblue']
        
        bars = ax5.bar(x, y, color=colors, alpha=0.7)
        ax5.set_ylabel('Number of Features')
        ax5.set_title('Feature Expansion Process')
        
        # Add value labels on bars
        for bar, value in zip(bars, y):
            height = bar.get_height()
            ax5.text(bar.get_x() + bar.get_width()/2., height + max(y) * 0.01,
                    f'{value}', ha='center', va='bottom')
        
        # Plot 6: Curvature analysis (if available and 1D)
        ax6 = axes[1, 2]
        if ('X_range' in self.curvature_analysis_ and 
            'y_predictions' in self.curvature_analysis_):
            
            X_range = self.curvature_analysis_['X_range']
            y_pred = self.curvature_analysis_['y_predictions']
            
            ax6.plot(X_range, y_pred, 'b-', linewidth=2, label='Polynomial Fit')
            
            # Plot original data points if 1D
            if self.X_original_.shape[1] == 1:
                ax6.scatter(self.X_original_[:, 0], self.y_original_, 
                           alpha=0.6, color='red', s=20, label='Data')
            
            # Mark inflection points if available
            if 'inflection_points' in self.curvature_analysis_:
                inflection_points = self.curvature_analysis_['inflection_points']
                for point in inflection_points:
                    ax6.axvline(x=point, color='green', linestyle='--', alpha=0.7)
            
            ax6.set_xlabel('Feature Value')
            ax6.set_ylabel('Target Value')
            ax6.set_title('Polynomial Function Shape')
            ax6.legend()
            ax6.grid(True, alpha=0.3)
        else:
            ax6.text(0.5, 0.5, 'Curvature analysis\nnot available', 
                    ha='center', va='center', transform=ax6.transAxes)
            ax6.set_title('Curvature Analysis')
        
        plt.tight_layout()
        return fig
    
    def plot_polynomial_surface(self, feature_indices=(0, 1), resolution=50, figsize=(12, 8)):
        """Plot polynomial surface for 2D visualization"""
        if self.n_features_in_ < 2:
            raise ValueError("Need at least 2 features for surface plot")
        
        if not self.is_fitted_:
            raise ValueError("Model must be fitted to plot polynomial surface")
        
        # Create meshgrid for surface
        X = self.X_original_
        feature_0 = X[:, feature_indices[0]]
        feature_1 = X[:, feature_indices[1]]
        
        x_min, x_max = feature_0.min() - 0.1, feature_0.max() + 0.1
        y_min, y_max = feature_1.min() - 0.1, feature_1.max() + 0.1
        
        xx, yy = np.meshgrid(
            np.linspace(x_min, x_max, resolution),
            np.linspace(y_min, y_max, resolution)
        )
        
        # Create prediction grid
        grid_points = np.zeros((resolution * resolution, self.n_features_in_))
        grid_points[:, feature_indices[0]] = xx.ravel()
        grid_points[:, feature_indices[1]] = yy.ravel()
        
        # Set other features to their mean values
        for i in range(self.n_features_in_):
            if i not in feature_indices:
                grid_points[:, i] = np.mean(X[:, i])
        
        # Make predictions
        Z = self.predict(grid_points).reshape(xx.shape)
        
        # Create 3D surface plot
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection='3d')
        
        # Plot surface
        surf = ax.plot_surface(xx, yy, Z, alpha=0.6, cmap='viridis')
        
        # Plot original data points
        ax.scatter(feature_0, feature_1, self.y_original_, 
                  color='red', s=50, alpha=0.8, label='Data')
        
        ax.set_xlabel(f'Feature {feature_indices[0]} ({self.feature_names_[feature_indices[0]]})')
        ax.set_ylabel(f'Feature {feature_indices[1]} ({self.feature_names_[feature_indices[1]]})')
        ax.set_zlabel('Target')
        ax.set_title(f'Polynomial Surface (Degree {self.degree_used_})')
        
        # Add colorbar
        fig.colorbar(surf, shrink=0.5, aspect=5)
        
        return fig
    
    def get_training_info(self) -> Dict[str, Any]:
        """Get information about the training process"""
        if not self.is_fitted_:
            return {"status": "Not fitted"}
        
        info = {
            "algorithm": "Polynomial Regression",
            "type": "Non-linear regression via polynomial feature engineering",
            "training_completed": True,
            "polynomial_characteristics": {
                "captures_non_linearity": True,
                "feature_engineering": "Polynomial features up to degree " + str(self.degree_used_),
                "handles_interactions": not self.interaction_only,
                "overfitting_protection": "Ridge regularization" if self.use_regularization else "None"
            },
            "model_configuration": {
                "degree_used": self.degree_used_,
                "degree_selection": "automatic" if self.auto_degree else "manual",
                "alpha_used": self.alpha_used_,
                "regularization_used": self.use_regularization,
                "include_bias": self.include_bias,
                "interaction_only": self.interaction_only,
                "normalize_features": self.normalize_features,
                "feature_selection": self.feature_selection
            },
            "feature_engineering_results": {
                "original_features": self.n_features_in_,
                "polynomial_features": len(self.model_.coef_),
                "selected_features": len(self.selected_features_) if hasattr(self, 'selected_features_') else len(self.model_.coef_),
                "feature_expansion_ratio": len(self.model_.coef_) / self.n_features_in_,
                "theoretical_max_features": self._calculate_theoretical_features()
            },
            "overfitting_assessment": {
                "overfitting_detected": self.overfitting_analysis_.get('is_overfitting', False),
                "overfitting_gap": self.overfitting_analysis_.get('overfitting_gap', 'N/A'),
                "complexity_ratio": self.overfitting_analysis_.get('complexity_ratio', 'N/A'),
                "overfitting_risk": self.overfitting_analysis_.get('overfitting_risk', 'Unknown')
            }
        }
        
        # Add degree analysis if available
        if self.degree_analysis_:
            info["degree_optimization"] = {
                "method": self.degree_analysis_.get('method'),
                "degrees_tested": self.degree_analysis_.get('degrees_tested'),
                "optimal_degree": self.degree_analysis_.get('optimal_degree'),
                "best_score": self.degree_analysis_.get('best_score')
            }
        
        return info

    def get_algorithm_specific_metrics(self, y_true: Optional[np.ndarray] = None, y_pred: Optional[np.ndarray] = None, y_proba: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        Retrieve algorithm-specific metrics for the Polynomial Regression model.

        Parameters:
        -----------
        y_true : np.ndarray, optional
            Actual target values. Required for metrics like Adjusted R-squared.
        y_pred : np.ndarray, optional
            Predicted target values. Required for metrics like Adjusted R-squared.
        y_proba : np.ndarray, optional
            Not used for regression models.

        Returns:
        --------
        Dict[str, Any]
            A dictionary containing algorithm-specific metrics.
        """
        if not self.is_fitted_:
            return {"error": "Model not fitted. Cannot retrieve Polynomial Regression specific metrics."}

        metrics = {}
        prefix = "polyreg_"

        # Degree and Feature Expansion
        metrics[f"{prefix}degree_used"] = self.degree_used_
        if self.auto_degree and self.degree_analysis_ and 'optimal_degree' in self.degree_analysis_:
            metrics[f"{prefix}optimal_degree_selected"] = self.degree_analysis_['optimal_degree']
        
        num_poly_features = len(self.model_.coef_)
        metrics[f"{prefix}num_polynomial_features"] = num_poly_features
        if self.n_features_in_ > 0:
            metrics[f"{prefix}feature_expansion_ratio"] = num_poly_features / self.n_features_in_

        # Regularization
        if self.use_regularization:
            metrics[f"{prefix}alpha_used"] = self.alpha_used_
            metrics[f"{prefix}regularization_type"] = "Ridge" # Currently only Ridge is implemented
        else:
            metrics[f"{prefix}regularization_type"] = "None"

        # Overfitting Metrics
        if self.detect_overfitting and self.overfitting_analysis_:
            if 'overfitting_gap' in self.overfitting_analysis_:
                metrics[f"{prefix}overfitting_gap_r2"] = self.overfitting_analysis_['overfitting_gap']
            if 'complexity_ratio' in self.overfitting_analysis_:
                metrics[f"{prefix}complexity_ratio"] = self.overfitting_analysis_['complexity_ratio']
            if 'is_overfitting' in self.overfitting_analysis_:
                 metrics[f"{prefix}overfitting_detected"] = self.overfitting_analysis_['is_overfitting']

        # Bias-Variance Indicators
        if self.bias_variance_analysis and self.bias_variance_analysis_:
            if 'estimated_bias' in self.bias_variance_analysis_:
                metrics[f"{prefix}estimated_bias_indicator"] = self.bias_variance_analysis_['estimated_bias']
            if 'cv_variance' in self.bias_variance_analysis_:
                metrics[f"{prefix}cv_score_variance"] = self.bias_variance_analysis_['cv_variance']

        # Curvature Metrics (1D specific)
        if self.curvature_analysis and self.curvature_analysis_.get('dimension') == '1D':
            if 'inflection_points' in self.curvature_analysis_:
                metrics[f"{prefix}num_inflection_points"] = len(self.curvature_analysis_['inflection_points'])
            if 'mean_curvature' in self.curvature_analysis_:
                metrics[f"{prefix}mean_curvature"] = self.curvature_analysis_['mean_curvature']
            if 'concavity_changes' in self.curvature_analysis_:
                metrics[f"{prefix}concavity_changes"] = self.curvature_analysis_['concavity_changes']


        # Polynomial Term Importance Contribution
        if self.compute_feature_importance and self.feature_importance_analysis_ and 'polynomial_contribution' in self.feature_importance_analysis_:
            poly_contrib = self.feature_importance_analysis_['polynomial_contribution']
            if 'linear_percentage' in poly_contrib:
                metrics[f"{prefix}linear_term_importance_pct"] = poly_contrib['linear_percentage']
            if 'polynomial_percentage' in poly_contrib: # Non-linear polynomial terms
                metrics[f"{prefix}nonlinear_term_importance_pct"] = poly_contrib['polynomial_percentage']

        # Adjusted R-squared
        if y_true is not None and y_pred is not None:
            n_samples = len(y_true)
            # Number of predictors is the number of polynomial features
            # If intercept is fitted by LinearRegression/Ridge, it's not in model_.coef_
            # PolynomialFeatures can include a bias term which becomes a feature.
            # If self.poly_features_.include_bias is True, one of the X_poly columns is the bias.
            # If self.model_ is LinearRegression and fit_intercept=True (default), it handles intercept separately.
            # If self.model_ is Ridge, fit_intercept=True is default.
            
            # n_predictors should be the number of terms in the model excluding the intercept if the model fits it separately.
            # self.model_.coef_ gives coefficients for features passed to it.
            # self.X_poly_ contains the features including a bias column if self.poly_features_.include_bias=True.
            # If self.poly_features_.include_bias is True, and the model (e.g. Ridge) also fits an intercept,
            # the bias column in X_poly might be redundant or lead to collinearity if not handled.
            # However, scikit-learn's Ridge/LinearRegression handles this.
            # The number of features used by the model is len(self.model_.coef_).
            n_predictors = num_poly_features
            
            if n_samples > n_predictors + 1: # Check to avoid division by zero or negative in sqrt
                r2 = r2_score(y_true, y_pred)
                adj_r2 = 1 - (1 - r2) * (n_samples - 1) / (n_samples - n_predictors - 1)
                metrics[f"{prefix}adjusted_r_squared"] = float(adj_r2)
            else:
                metrics[f"{prefix}adjusted_r_squared_info"] = "Not enough samples to calculate Adjusted R-squared reliably."
        
        if not metrics:
            metrics['info'] = "No specific Polynomial Regression metrics were available or calculated."
            
        return metrics

# Factory function
def get_plugin():
    """Factory function to get the plugin instance"""
    return PolynomialRegressionPlugin()


# Example usage and testing
if __name__ == "__main__":
    """
    Example usage of Polynomial Regression Plugin
    """
    print("Testing Polynomial Regression Plugin...")
    
    try:
        # Create sample non-linear data
        np.random.seed(42)
        
        # Generate 1D polynomial data
        X_1d = np.linspace(-2, 2, 100).reshape(-1, 1)
        y_1d = 0.5 * X_1d.ravel()**3 - 2 * X_1d.ravel()**2 + X_1d.ravel() + 1 + np.random.normal(0, 0.2, 100)
        
        # Generate 2D polynomial data with interactions
        from sklearn.datasets import make_regression
        X_2d, y_2d = make_regression(
            n_samples=200,
            n_features=2,
            noise=0.1,
            random_state=42
        )
        
        # Add polynomial relationships
        y_2d = y_2d + 0.5 * X_2d[:, 0]**2 + 0.3 * X_2d[:, 1]**2 + 0.2 * X_2d[:, 0] * X_2d[:, 1]
        
        print(f"\n📊 Test Datasets:")
        print(f"1D Dataset: {X_1d.shape} - Cubic polynomial with noise")
        print(f"2D Dataset: {X_2d.shape} - Quadratic with interactions")
        
        # Test 1D polynomial regression
        print(f"\n🧪 Testing 1D Polynomial Regression...")
        
        plugin_1d = PolynomialRegressionPlugin(
            auto_degree=True,
            max_degree=6,
            use_regularization=True,
            auto_alpha=True,
            normalize_features=True,
            detect_overfitting=True,
            curvature_analysis=True,
            random_state=42
        )
        
        # Check compatibility
        compatible, message = plugin_1d.is_compatible_with_data(X_1d, y_1d)
        print(f"✅ Compatibility: {message}")
        
        if compatible:
            # Train model
            plugin_1d.fit(X_1d, y_1d)
            
            # Make predictions
            y_pred_1d = plugin_1d.predict(X_1d)
            
            # Evaluate
            from sklearn.metrics import r2_score, mean_squared_error
            r2_1d = r2_score(y_1d, y_pred_1d)
            mse_1d = mean_squared_error(y_1d, y_pred_1d)
            
            print(f"\n📊 1D Polynomial Results:")
            print(f"R²: {r2_1d:.4f}")
            print(f"MSE: {mse_1d:.4f}")
            
            # Get model info
            model_params = plugin_1d.get_model_params()
            print(f"Degree used: {model_params['degree_used']}")
            print(f"Polynomial features: {model_params['n_polynomial_features']}")
            print(f"Regularization: α = {model_params['alpha_used']:.6f}")
            
            # Test feature importance
            importance = plugin_1d.get_feature_importance()
            if importance:
                print(f"\n🎯 Top Polynomial Terms:")
                for i, (name, coef, imp) in enumerate(importance['top_polynomial_terms'][:3]):
                    print(f"  {i+1}. {name}: {coef:.4f} (importance: {imp:.3f})")
        
        # Test 2D polynomial regression
        print(f"\n🧪 Testing 2D Polynomial Regression...")
        
        # Create DataFrame for 2D test
        X_2d_df = pd.DataFrame(X_2d, columns=['feature_0', 'feature_1'])
        
        plugin_2d = PolynomialRegressionPlugin(
            degree=3,
            auto_degree=False,
            use_regularization=True,
            alpha=0.1,
            interaction_only=False,
            feature_selection=True,
            normalize_features=True,
            detect_overfitting=True,
            random_state=42
        )
        
        # Check compatibility
        compatible, message = plugin_2d.is_compatible_with_data(X_2d_df, y_2d)
        print(f"✅ Compatibility: {message}")
        
        if compatible:
            # Train model
            plugin_2d.fit(X_2d_df, y_2d)
            
            # Make predictions
            y_pred_2d = plugin_2d.predict(X_2d_df)
            
            # Evaluate
            r2_2d = r2_score(y_2d, y_pred_2d)
            mse_2d = mean_squared_error(y_2d, y_pred_2d)
            
            print(f"\n📊 2D Polynomial Results:")
            print(f"R²: {r2_2d:.4f}")
            print(f"MSE: {mse_2d:.4f}")
            
            # Get model info
            model_params = plugin_2d.get_model_params()
            print(f"Degree used: {model_params['degree_used']}")
            print(f"Polynomial features: {model_params['n_polynomial_features']}")
            print(f"Feature expansion ratio: {model_params['feature_expansion_ratio']:.1f}x")
            
            # Get overfitting analysis
            overfitting = plugin_2d.overfitting_analysis_
            if 'is_overfitting' in overfitting:
                print(f"Overfitting detected: {overfitting['is_overfitting']}")
                print(f"Overfitting gap: {overfitting['overfitting_gap']:.4f}")
                print(f"Overfitting risk: {overfitting['overfitting_risk']}")
        
        print("\n✅ Polynomial Regression Plugin test completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Error testing Polynomial Regression Plugin: {str(e)}")
        import traceback
        traceback.print_exc()