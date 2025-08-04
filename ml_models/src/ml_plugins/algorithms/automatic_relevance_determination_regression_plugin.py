import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, Optional, Tuple, List, Union
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_X_y, check_array
from sklearn.linear_model import ARDRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy import stats
from scipy.special import gamma, digamma
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


class ARDRegressionPlugin(BaseEstimator, RegressorMixin, MLPlugin):
    """
    ARD (Automatic Relevance Determination) Regression Plugin
    
    This plugin implements Automatic Relevance Determination Regression, a Bayesian 
    approach to linear regression that automatically determines feature relevance by 
    learning individual precision parameters (alpha) for each feature. Features with 
    high precision (low relevance) are effectively removed from the model, providing 
    automatic feature selection and sparsity.
    
    Key Features:
    - Automatic feature selection through relevance determination
    - Bayesian uncertainty quantification
    - Individual precision parameters for each feature
    - Sparse solutions with irrelevant features pruned
    - Hierarchical Bayesian modeling with hyperpriors
    - Evidence-based model selection
    - Robust to overfitting and multicollinearity
    - Advanced diagnostics and feature relevance analysis
    """
    
    def __init__(
        self,
        # ARD-specific hyperparameters
        n_iter=300,
        tol=1e-3,
        alpha_1=1e-6,
        alpha_2=1e-6,
        lambda_1=1e-6,
        lambda_2=1e-6,
        
        # Model configuration
        alpha_init=None,
        lambda_init=None,
        compute_score=True,
        fit_intercept=True,
        normalize_features=True,
        
        # ARD-specific options
        threshold_lambda=10000.0,  # Threshold for feature pruning
        copy_X=True,
        verbose=False,
        
        # Uncertainty quantification
        compute_prediction_intervals=True,
        confidence_level=0.95,
        n_posterior_samples=1000,
        
        # Analysis options
        convergence_analysis=True,
        evidence_analysis=True,
        feature_selection_analysis=True,
        relevance_evolution_analysis=True,
        uncertainty_decomposition=True,
        sparsity_analysis=True,
        
        # Advanced ARD options
        track_alpha_evolution=True,
        automatic_pruning=True,
        relevance_threshold=1e-3,
        
        random_state=42
    ):
        super().__init__()
        
        # Core ARD parameters
        self.n_iter = n_iter
        self.tol = tol
        self.alpha_1 = alpha_1
        self.alpha_2 = alpha_2
        self.lambda_1 = lambda_1
        self.lambda_2 = lambda_2
        
        # Model configuration
        self.alpha_init = alpha_init
        self.lambda_init = lambda_init
        self.compute_score = compute_score
        self.fit_intercept = fit_intercept
        self.normalize_features = normalize_features
        self.copy_X = copy_X
        self.verbose = verbose
        
        # ARD-specific options
        self.threshold_lambda = threshold_lambda
        
        # Uncertainty quantification
        self.compute_prediction_intervals = compute_prediction_intervals
        self.confidence_level = confidence_level
        self.n_posterior_samples = n_posterior_samples
        
        # Analysis options
        self.convergence_analysis = convergence_analysis
        self.evidence_analysis = evidence_analysis
        self.feature_selection_analysis = feature_selection_analysis
        self.relevance_evolution_analysis = relevance_evolution_analysis
        self.uncertainty_decomposition = uncertainty_decomposition
        self.sparsity_analysis = sparsity_analysis
        
        # Advanced ARD options
        self.track_alpha_evolution = track_alpha_evolution
        self.automatic_pruning = automatic_pruning
        self.relevance_threshold = relevance_threshold
        
        self.random_state = random_state
        
        # Required plugin metadata
        self._name = "ARD Regression"
        self._description = "Automatic Relevance Determination - Bayesian regression with automatic feature selection"
        self._category = "Bayesian Models"
        
        # Required capability flags
        self._supports_classification = False
        self._supports_regression = True
        self._min_samples_required = 15
        
        # Internal state
        self.is_fitted_ = False
        self.model_ = None
        self.scaler_ = None
        self.feature_names_ = None
        self.n_features_in_ = None
        
        # ARD analysis results
        self.convergence_analysis_ = {}
        self.evidence_analysis_ = {}
        self.feature_selection_analysis_ = {}
        self.relevance_evolution_analysis_ = {}
        self.uncertainty_analysis_ = {}
        self.sparsity_analysis_ = {}
        self.alpha_evolution_analysis_ = {}
    
    def get_name(self) -> str:
        return self._name
    
    def get_description(self) -> str:
        return self._description
    
    def get_category(self) -> str:
        return self._category
    
    def fit(self, X, y, sample_weight=None):
        """
        Fit the ARD Regression model with comprehensive analysis
        
        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            Training data
        y : array-like of shape (n_samples,)
            Target values
        sample_weight : array-like of shape (n_samples,), optional
            Individual weights for each sample (not directly supported by ARDRegression)
        
        Returns:
        --------
        self : object
            Returns self for method chaining
        """
        # Store feature information
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
        
        # Create and configure ARD model
        self.model_ = ARDRegression(
            n_iter=self.n_iter,
            tol=self.tol,
            alpha_1=self.alpha_1,
            alpha_2=self.alpha_2,
            lambda_1=self.lambda_1,
            lambda_2=self.lambda_2,
            alpha_init=self.alpha_init,
            lambda_init=self.lambda_init,
            compute_score=self.compute_score,
            threshold_lambda=self.threshold_lambda,
            fit_intercept=self.fit_intercept,
            copy_X=self.copy_X,
            verbose=self.verbose
        )
        
        # Fit the model
        self.model_.fit(X_processed, y)
        
        # Add cross-validation metrics
        if hasattr(self, 'compute_cv_metrics') and self.compute_cv_metrics:
            cv_scores = cross_val_score(self.model_, X_processed, y, cv=5, scoring='r2')
            self.cv_metrics_ = {
                'cv_r2_mean': np.mean(cv_scores),
                'cv_r2_std': np.std(cv_scores),
                'cv_scores': cv_scores
            }
        
        # Perform comprehensive ARD analysis
        self._analyze_convergence()
        self._analyze_evidence()
        self._analyze_feature_selection()
        self._analyze_relevance_evolution()
        self._analyze_uncertainty()
        self._analyze_sparsity()
        self._analyze_alpha_evolution()
        
        self.is_fitted_ = True
        return self
    
    def predict(self, X, return_std=False):
        """
        Make predictions using the fitted ARD model
        
        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            Input data for prediction
        return_std : bool, default=False
            If True, returns prediction standard deviations
        
        Returns:
        --------
        y_pred : array of shape (n_samples,)
            Predicted values
        y_std : array of shape (n_samples,), optional
            Standard deviation of predictions (if return_std=True)
        """
        if not self.is_fitted_:
            raise ValueError("Model must be fitted before making predictions")
        
        X = check_array(X, accept_sparse=False)
        
        # Apply same preprocessing as training
        X_processed = X.copy()
        if self.scaler_ is not None:
            X_processed = self.scaler_.transform(X_processed)
        
        if return_std:
            return self.model_.predict(X_processed, return_std=True)
        else:
            return self.model_.predict(X_processed)
    
    def predict_with_uncertainty(self, X, confidence_level=None):
        """
        Make predictions with uncertainty quantification
        
        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            Input data for prediction
        confidence_level : float, optional
            Confidence level for prediction intervals
        
        Returns:
        --------
        results : dict
            Dictionary containing predictions, standard deviations, and intervals
        """
        if not self.is_fitted_:
            raise ValueError("Model must be fitted before making predictions")
        
        if confidence_level is None:
            confidence_level = self.confidence_level
        
        X = check_array(X, accept_sparse=False)
        
        # Apply same preprocessing as training
        X_processed = X.copy()
        if self.scaler_ is not None:
            X_processed = self.scaler_.transform(X_processed)
        
        # Get predictions and standard deviations
        y_pred, y_std = self.model_.predict(X_processed, return_std=True)
        
        # Calculate prediction intervals
        alpha = 1 - confidence_level
        z_score = stats.norm.ppf(1 - alpha/2)
        
        y_lower = y_pred - z_score * y_std
        y_upper = y_pred + z_score * y_std
        
        # Decompose uncertainty sources
        uncertainty_decomposition = self._decompose_uncertainty(X_processed, y_pred, y_std)
        
        return {
            'predictions': y_pred,
            'std_dev': y_std,
            'lower_bound': y_lower,
            'upper_bound': y_upper,
            'confidence_level': confidence_level,
            'uncertainty_decomposition': uncertainty_decomposition,
            'feature_relevance_info': self._get_prediction_relevance_info(X_processed)
        }
    
    def _analyze_convergence(self):
        """Analyze convergence of the ARD inference"""
        if not self.convergence_analysis:
            return
        
        # Extract convergence information
        if hasattr(self.model_, 'scores_') and self.model_.scores_ is not None:
            scores = np.array(self.model_.scores_)
            n_iterations = len(scores)
            
            # Analyze convergence characteristics
            final_score = scores[-1] if len(scores) > 0 else None
            
            # Check for convergence
            if len(scores) > 10:
                recent_scores = scores[-10:]
                score_variance = np.var(recent_scores)
                score_trend = np.polyfit(range(len(recent_scores)), recent_scores, 1)[0]
                converged = score_variance < self.tol and abs(score_trend) < self.tol
            else:
                score_variance = np.var(scores) if len(scores) > 1 else 0
                score_trend = 0
                converged = n_iterations < self.n_iter
            
            # Calculate convergence rate
            if len(scores) > 1:
                convergence_rate = self._estimate_convergence_rate(scores)
            else:
                convergence_rate = None
            
            # ARD-specific convergence analysis
            alpha_convergence = self._analyze_alpha_convergence()
            
            self.convergence_analysis_ = {
                'scores': scores,
                'n_iterations': n_iterations,
                'max_iterations': self.n_iter,
                'final_score': final_score,
                'converged': converged,
                'convergence_tolerance': self.tol,
                'score_variance': score_variance,
                'score_trend': score_trend,
                'convergence_rate': convergence_rate,
                'early_stopping': n_iterations < self.n_iter,
                'alpha_convergence': alpha_convergence
            }
        else:
            self.convergence_analysis_ = {
                'error': 'Convergence scores not available (compute_score=False)'
            }
    
    def _analyze_alpha_convergence(self):
        """Analyze convergence of alpha parameters"""
        try:
            # Get final alpha values
            alpha_final = self.model_.alpha_
            
            # Identify converged features (those not pruned)
            active_features = alpha_final < self.threshold_lambda
            n_active = np.sum(active_features)
            n_pruned = len(alpha_final) - n_active
            
            # Calculate alpha statistics
            alpha_stats = {
                'n_active_features': n_active,
                'n_pruned_features': n_pruned,
                'pruning_ratio': n_pruned / len(alpha_final),
                'active_alpha_mean': np.mean(alpha_final[active_features]) if n_active > 0 else 0,
                'active_alpha_std': np.std(alpha_final[active_features]) if n_active > 0 else 0,
                'pruned_alpha_mean': np.mean(alpha_final[~active_features]) if n_pruned > 0 else 0,
                'alpha_range': np.max(alpha_final) - np.min(alpha_final),
                'alpha_sparsity_level': np.sum(alpha_final > self.threshold_lambda) / len(alpha_final)
            }
            
            return alpha_stats
            
        except Exception as e:
            return {'error': f'Could not analyze alpha convergence: {str(e)}'}
    
    def _estimate_convergence_rate(self, scores):
        """Estimate convergence rate from score evolution"""
        if len(scores) < 5:
            return None
        
        try:
            # Take differences to estimate rate
            score_diffs = np.abs(np.diff(scores[-20:]))  # Last 20 iterations
            score_diffs = score_diffs[score_diffs > 0]  # Remove zeros
            
            if len(score_diffs) > 1:
                # Fit exponential decay
                t = np.arange(len(score_diffs))
                log_diffs = np.log(score_diffs)
                rate = -np.polyfit(t, log_diffs, 1)[0]
                return max(0, rate)  # Ensure positive rate
            else:
                return None
        except:
            return None
    
    def _analyze_evidence(self):
        """Analyze model evidence (marginal likelihood)"""
        if not self.evidence_analysis:
            return
        
        try:
            # Calculate log marginal likelihood (evidence)
            alpha = self.model_.alpha_
            lambda_ = self.model_.lambda_
            
            X_processed = self.X_original_.copy()
            if self.scaler_ is not None:
                X_processed = self.scaler_.transform(X_processed)
            
            # Add intercept column if needed
            if self.fit_intercept:
                X_design = np.column_stack([np.ones(X_processed.shape[0]), X_processed])
                # Adjust alpha for intercept (add very small value for intercept)
                alpha_full = np.concatenate([[1e-12], alpha])
            else:
                X_design = X_processed
                alpha_full = alpha
            
            # Calculate evidence components
            evidence_components = self._calculate_ard_evidence_components(
                X_design, self.y_original_, alpha_full, lambda_
            )
            
            # Model complexity penalty (ARD-specific)
            effective_parameters = self._calculate_ard_effective_parameters(X_design, alpha_full, lambda_)
            
            # Information criteria
            n_samples = len(self.y_original_)
            log_likelihood = evidence_components['log_likelihood']
            
            aic = 2 * effective_parameters - 2 * log_likelihood
            bic = effective_parameters * np.log(n_samples) - 2 * log_likelihood
            
            # ARD-specific metrics
            active_features = alpha < self.threshold_lambda
            sparsity_ratio = 1 - np.sum(active_features) / len(alpha)
            
            self.evidence_analysis_ = {
                'log_marginal_likelihood': evidence_components['log_marginal_likelihood'],
                'log_likelihood': log_likelihood,
                'complexity_penalty': evidence_components['complexity_penalty'],
                'effective_parameters': effective_parameters,
                'actual_parameters': np.sum(active_features),
                'sparsity_ratio': sparsity_ratio,
                'aic': aic,
                'bic': bic,
                'alpha_geometric_mean': np.exp(np.mean(np.log(alpha[active_features]))) if np.sum(active_features) > 0 else 0,
                'lambda': lambda_,
                'evidence_components': evidence_components,
                'ard_specific_metrics': {
                    'feature_selection_quality': self._assess_feature_selection_quality(),
                    'sparsity_vs_fit_tradeoff': self._assess_sparsity_fit_tradeoff()
                }
            }
            
        except Exception as e:
            self.evidence_analysis_ = {
                'error': f'Could not compute evidence: {str(e)}'
            }
    
    def _calculate_ard_evidence_components(self, X, y, alpha, lambda_):
        """Calculate components of the log marginal likelihood for ARD"""
        n_samples, n_features = X.shape
        
        # Posterior covariance matrix with individual alpha values
        A = np.diag(alpha)
        S_inv = A + lambda_ * X.T @ X
        
        try:
            S = np.linalg.inv(S_inv)
            log_det_S_inv = np.linalg.slogdet(S_inv)[1]
        except:
            # Use pseudo-inverse for numerical stability
            S = np.linalg.pinv(S_inv)
            eigenvals = np.linalg.eigvals(S_inv)
            eigenvals = eigenvals[eigenvals > 1e-12]
            log_det_S_inv = np.sum(np.log(eigenvals))
        
        # Posterior mean
        mu = lambda_ * S @ X.T @ y
        
        # Residuals
        residuals = y - X @ mu
        
        # Log likelihood
        log_likelihood = -0.5 * lambda_ * np.sum(residuals**2)
        log_likelihood -= 0.5 * n_samples * np.log(2 * np.pi / lambda_)
        
        # Prior term (ARD-specific with individual alphas)
        prior_term = -0.5 * np.sum(alpha * mu**2)
        prior_term -= 0.5 * np.sum(np.log(2 * np.pi / alpha))
        
        # Complexity penalty
        complexity_penalty = 0.5 * log_det_S_inv
        
        # Total log marginal likelihood
        log_marginal_likelihood = log_likelihood + prior_term + complexity_penalty
        
        return {
            'log_marginal_likelihood': log_marginal_likelihood,
            'log_likelihood': log_likelihood,
            'prior_term': prior_term,
            'complexity_penalty': complexity_penalty,
            'posterior_mean': mu,
            'posterior_covariance': S
        }
    
    def _calculate_ard_effective_parameters(self, X, alpha, lambda_):
        """Calculate effective number of parameters for ARD"""
        try:
            # For ARD: effective parameters = tr(lambda * X @ S @ X.T) where S is posterior covariance
            A = np.diag(alpha)
            S_inv = A + lambda_ * X.T @ X
            S = np.linalg.inv(S_inv)
            
            # Effective degrees of freedom
            H = lambda_ * X @ S @ X.T
            effective_params = np.trace(H)
            return effective_params
        except:
            # Fallback: count non-pruned features
            active_features = alpha < self.threshold_lambda
            return np.sum(active_features)
    
    def _assess_feature_selection_quality(self):
        """Assess quality of automatic feature selection"""
        try:
            alpha = self.model_.alpha_
            active_features = alpha < self.threshold_lambda
            
            # Calculate feature selection metrics
            n_selected = np.sum(active_features)
            n_total = len(alpha)
            selection_ratio = n_selected / n_total
            
            # Alpha ratio analysis
            if n_selected > 0:
                alpha_active = alpha[active_features]
                alpha_pruned = alpha[~active_features]
                
                alpha_separation = np.min(alpha_pruned) / np.max(alpha_active) if len(alpha_active) > 0 and len(alpha_pruned) > 0 else 1
                alpha_consistency = np.std(np.log(alpha_active)) if len(alpha_active) > 1 else 0
            else:
                alpha_separation = 0
                alpha_consistency = 0
            
            return {
                'n_selected_features': n_selected,
                'selection_ratio': selection_ratio,
                'alpha_separation_ratio': alpha_separation,
                'alpha_consistency': alpha_consistency,
                'selection_quality_score': min(alpha_separation, 10) * (1 - alpha_consistency)
            }
            
        except Exception as e:
            return {'error': f'Could not assess feature selection: {str(e)}'}
    
    def _assess_sparsity_fit_tradeoff(self):
        """Assess trade-off between sparsity and model fit"""
        try:
            # Get model performance
            X_processed = self.X_original_.copy()
            if self.scaler_ is not None:
                X_processed = self.scaler_.transform(X_processed)
            
            y_pred = self.model_.predict(X_processed)
            r2 = r2_score(self.y_original_, y_pred)
            
            # Get sparsity metrics
            alpha = self.model_.alpha_
            active_features = alpha < self.threshold_lambda
            sparsity_ratio = 1 - np.sum(active_features) / len(alpha)
            
            # Calculate trade-off score
            tradeoff_score = r2 * (1 + sparsity_ratio)  # Reward both fit and sparsity
            
            return {
                'r2_score': r2,
                'sparsity_ratio': sparsity_ratio,
                'n_active_features': np.sum(active_features),
                'tradeoff_score': tradeoff_score,
                'sparsity_benefit': sparsity_ratio * 0.1  # Quantify sparsity benefit
            }
            
        except Exception as e:
            return {'error': f'Could not assess tradeoff: {str(e)}'}
    
    def _analyze_feature_selection(self):
        """Analyze automatic feature selection results"""
        if not self.feature_selection_analysis:
            return
        
        try:
            alpha = self.model_.alpha_
            active_features = alpha < self.threshold_lambda
            
            # Feature selection statistics
            n_selected = np.sum(active_features)
            n_total = len(alpha)
            
            # Rank features by relevance (inverse of alpha)
            relevance_scores = 1.0 / (alpha + 1e-12)  # Add small epsilon to avoid division by zero
            feature_ranking = np.argsort(relevance_scores)[::-1]
            
            # Identify feature groups
            selected_features = np.where(active_features)[0]
            pruned_features = np.where(~active_features)[0]
            
            # Feature importance based on coefficient magnitude and relevance
            coef_abs = np.abs(self.model_.coef_)
            feature_importance = coef_abs * relevance_scores / np.max(coef_abs * relevance_scores)
            
            # Alpha distribution analysis
            alpha_stats = self._analyze_alpha_distribution(alpha, active_features)
            
            self.feature_selection_analysis_ = {
                'alpha_values': alpha,
                'active_features': active_features,
                'relevance_scores': relevance_scores,
                'feature_ranking': feature_ranking,
                'feature_importance': feature_importance,
                'n_selected_features': n_selected,
                'n_pruned_features': n_total - n_selected,
                'selection_ratio': n_selected / n_total,
                'selected_feature_indices': selected_features,
                'pruned_feature_indices': pruned_features,
                'selected_feature_names': [self.feature_names_[i] for i in selected_features],
                'pruned_feature_names': [self.feature_names_[i] for i in pruned_features],
                'alpha_statistics': alpha_stats,
                'threshold_lambda': self.threshold_lambda,
                'top_features': [
                    (self.feature_names_[i], alpha[i], relevance_scores[i], feature_importance[i], active_features[i])
                    for i in feature_ranking[:min(10, len(feature_ranking))]
                ]
            }
            
        except Exception as e:
            self.feature_selection_analysis_ = {
                'error': f'Could not analyze feature selection: {str(e)}'
            }
    
    def _analyze_alpha_distribution(self, alpha, active_features):
        """Analyze distribution of alpha parameters"""
        try:
            alpha_active = alpha[active_features]
            alpha_pruned = alpha[~active_features]
            
            stats_dict = {
                'all_alpha_stats': {
                    'mean': np.mean(alpha),
                    'std': np.std(alpha),
                    'min': np.min(alpha),
                    'max': np.max(alpha),
                    'median': np.median(alpha),
                    'range': np.max(alpha) - np.min(alpha)
                }
            }
            
            if len(alpha_active) > 0:
                stats_dict['active_alpha_stats'] = {
                    'mean': np.mean(alpha_active),
                    'std': np.std(alpha_active),
                    'min': np.min(alpha_active),
                    'max': np.max(alpha_active),
                    'median': np.median(alpha_active),
                    'geometric_mean': np.exp(np.mean(np.log(alpha_active)))
                }
            
            if len(alpha_pruned) > 0:
                stats_dict['pruned_alpha_stats'] = {
                    'mean': np.mean(alpha_pruned),
                    'std': np.std(alpha_pruned),
                    'min': np.min(alpha_pruned),
                    'max': np.max(alpha_pruned),
                    'median': np.median(alpha_pruned)
                }
            
            # Separation quality
            if len(alpha_active) > 0 and len(alpha_pruned) > 0:
                separation_gap = np.min(alpha_pruned) - np.max(alpha_active)
                separation_ratio = np.min(alpha_pruned) / np.max(alpha_active)
                stats_dict['separation_metrics'] = {
                    'gap': separation_gap,
                    'ratio': separation_ratio,
                    'well_separated': separation_ratio > 10
                }
            
            return stats_dict
            
        except Exception as e:
            return {'error': f'Could not analyze alpha distribution: {str(e)}'}
    
    def _analyze_relevance_evolution(self):
        """Analyze evolution of feature relevance during training"""
        if not self.relevance_evolution_analysis:
            return
        
        try:
            # Final alpha values
            alpha_final = self.model_.alpha_
            
            # Calculate relevance metrics
            relevance_final = 1.0 / (alpha_final + 1e-12)
            relevance_normalized = relevance_final / np.max(relevance_final)
            
            # Feature relevance categories
            high_relevance = relevance_normalized > 0.5
            medium_relevance = (relevance_normalized > 0.1) & (relevance_normalized <= 0.5)
            low_relevance = relevance_normalized <= 0.1
            
            # Evolution summary
            evolution_summary = {
                'final_alpha': alpha_final,
                'final_relevance': relevance_final,
                'normalized_relevance': relevance_normalized,
                'high_relevance_features': np.where(high_relevance)[0],
                'medium_relevance_features': np.where(medium_relevance)[0],
                'low_relevance_features': np.where(low_relevance)[0],
                'relevance_categories': {
                    'high': np.sum(high_relevance),
                    'medium': np.sum(medium_relevance),
                    'low': np.sum(low_relevance)
                },
                'relevance_distribution': {
                    'mean': np.mean(relevance_normalized),
                    'std': np.std(relevance_normalized),
                    'entropy': -np.sum(relevance_normalized * np.log(relevance_normalized + 1e-12))
                }
            }
            
            # Feature relevance ranking with names
            relevance_ranking = np.argsort(relevance_normalized)[::-1]
            ranked_features = [
                (self.feature_names_[i], relevance_normalized[i], alpha_final[i])
                for i in relevance_ranking
            ]
            
            evolution_summary['ranked_features'] = ranked_features
            
            self.relevance_evolution_analysis_ = evolution_summary
            
        except Exception as e:
            self.relevance_evolution_analysis_ = {
                'error': f'Could not analyze relevance evolution: {str(e)}'
            }
    
    def _analyze_uncertainty(self):
        """Analyze uncertainty sources and prediction reliability"""
        if not self.uncertainty_decomposition:
            return
        
        try:
            # Sample predictions on training data to analyze uncertainty
            X_processed = self.X_original_.copy()
            if self.scaler_ is not None:
                X_processed = self.scaler_.transform(X_processed)
            
            y_pred, y_std = self.model_.predict(X_processed, return_std=True)
            
            # Decompose uncertainty sources
            uncertainty_decomp = self._decompose_uncertainty(X_processed, y_pred, y_std)
            
            # Calculate uncertainty statistics
            uncertainty_stats = {
                'mean_uncertainty': np.mean(y_std),
                'std_uncertainty': np.std(y_std),
                'min_uncertainty': np.min(y_std),
                'max_uncertainty': np.max(y_std),
                'uncertainty_range': np.max(y_std) - np.min(y_std),
                'relative_uncertainty': np.mean(y_std) / np.std(self.y_original_)
            }
            
            # Identify high uncertainty regions
            uncertainty_threshold = np.percentile(y_std, 90)
            high_uncertainty_mask = y_std > uncertainty_threshold
            
            # ARD-specific uncertainty analysis
            ard_uncertainty_analysis = self._analyze_ard_specific_uncertainty(
                X_processed, y_pred, y_std
            )
            
            self.uncertainty_analysis_ = {
                'prediction_uncertainties': y_std,
                'uncertainty_statistics': uncertainty_stats,
                'uncertainty_decomposition': uncertainty_decomp,
                'high_uncertainty_indices': np.where(high_uncertainty_mask)[0],
                'uncertainty_threshold': uncertainty_threshold,
                'uncertainty_calibration': self._assess_uncertainty_calibration(
                    self.y_original_, y_pred, y_std
                ),
                'ard_specific_uncertainty': ard_uncertainty_analysis
            }
            
        except Exception as e:
            self.uncertainty_analysis_ = {
                'error': f'Could not analyze uncertainty: {str(e)}'
            }
    
    def _analyze_ard_specific_uncertainty(self, X, y_pred, y_std):
        """Analyze ARD-specific uncertainty contributions"""
        try:
            alpha = self.model_.alpha_
            active_features = alpha < self.threshold_lambda
            
            # Uncertainty contribution from active vs pruned features
            X_active = X[:, active_features] if np.any(active_features) else np.array([]).reshape(X.shape[0], 0)
            
            if X_active.shape[1] > 0:
                # Approximate uncertainty from active features only
                # This is a simplified analysis - full implementation would require posterior sampling
                feature_contributions = np.abs(X_active @ self.model_.coef_[active_features])
                uncertainty_from_active = feature_contributions * np.mean(y_std)
            else:
                uncertainty_from_active = np.zeros(X.shape[0])
            
            # Feature-wise uncertainty contribution
            feature_uncertainty_contributions = {}
            if np.any(active_features):
                for i, feature_idx in enumerate(np.where(active_features)[0]):
                    feature_uncertainty_contributions[self.feature_names_[feature_idx]] = {
                        'alpha': alpha[feature_idx],
                        'coefficient': self.model_.coef_[feature_idx],
                        'avg_contribution': np.mean(np.abs(X[:, feature_idx] * self.model_.coef_[feature_idx]))
                    }
            
            return {
                'n_active_features_in_uncertainty': np.sum(active_features),
                'uncertainty_from_feature_selection': np.std(uncertainty_from_active),
                'feature_uncertainty_contributions': feature_uncertainty_contributions,
                'total_vs_active_uncertainty_ratio': np.mean(y_std) / (np.mean(uncertainty_from_active) + 1e-12)
            }
            
        except Exception as e:
            return {'error': f'Could not analyze ARD-specific uncertainty: {str(e)}'}
    
    def _decompose_uncertainty(self, X, y_pred, y_std):
        """Decompose uncertainty into aleatoric and epistemic components"""
        try:
            # Aleatoric uncertainty (irreducible noise)
            noise_variance = 1.0 / self.model_.lambda_
            aleatoric_std = np.sqrt(noise_variance)
            
            # Epistemic uncertainty (model uncertainty)
            epistemic_variance = y_std**2 - noise_variance
            epistemic_variance = np.maximum(epistemic_variance, 0)  # Ensure non-negative
            epistemic_std = np.sqrt(epistemic_variance)
            
            # ARD-specific: uncertainty from feature selection
            alpha = self.model_.alpha_
            active_features = alpha < self.threshold_lambda
            selection_uncertainty = self._estimate_selection_uncertainty(X, active_features)
            
            return {
                'aleatoric_uncertainty': aleatoric_std,
                'epistemic_uncertainty': epistemic_std,
                'selection_uncertainty': selection_uncertainty,
                'total_uncertainty': y_std,
                'aleatoric_percentage': (aleatoric_std**2 / y_std**2) * 100,
                'epistemic_percentage': (epistemic_variance / y_std**2) * 100,
                'uncertainty_decomposition_valid': np.allclose(y_std, 
                    np.sqrt(aleatoric_std**2 + epistemic_std**2), rtol=0.1)
            }
            
        except Exception as e:
            return {'error': f'Could not decompose uncertainty: {str(e)}'}
    
    def _estimate_selection_uncertainty(self, X, active_features):
        """Estimate uncertainty contribution from feature selection"""
        try:
            # Simplified estimation of uncertainty due to feature selection
            # Full implementation would require variational inference or sampling
            
            if np.sum(active_features) == 0:
                return 0.0
            
            # Use coefficient variance as proxy for selection uncertainty
            active_coefs = self.model_.coef_[active_features]
            selection_uncertainty = np.std(active_coefs) * np.mean(np.abs(X[:, active_features]), axis=1)
            
            return np.mean(selection_uncertainty)
            
        except:
            return 0.0
    
    def _assess_uncertainty_calibration(self, y_true, y_pred, y_std):
        """Assess how well uncertainty estimates are calibrated"""
        try:
            # Calculate normalized residuals
            normalized_residuals = (y_true - y_pred) / y_std
            
            # Test if residuals follow standard normal distribution
            statistic, p_value = stats.normaltest(normalized_residuals)
            
            # Calculate empirical coverage for different confidence levels
            confidence_levels = [0.68, 0.95, 0.99]
            empirical_coverage = []
            
            for conf_level in confidence_levels:
                z_score = stats.norm.ppf((1 + conf_level) / 2)
                within_interval = np.abs(normalized_residuals) <= z_score
                empirical_coverage.append(np.mean(within_interval))
            
            # Calibration assessment
            coverage_errors = [abs(emp - theory) for emp, theory in zip(empirical_coverage, confidence_levels)]
            well_calibrated = all(error < 0.1 for error in coverage_errors)
            
            return {
                'normalized_residuals': normalized_residuals,
                'normality_test_statistic': statistic,
                'normality_test_p_value': p_value,
                'confidence_levels': confidence_levels,
                'empirical_coverage': empirical_coverage,
                'coverage_errors': coverage_errors,
                'well_calibrated': well_calibrated,
                'calibration_assessment': 'Well calibrated' if well_calibrated else 'Poorly calibrated'
            }
            
        except Exception as e:
            return {'error': f'Could not assess calibration: {str(e)}'}
    
    def _analyze_sparsity(self):
        """Analyze sparsity characteristics of the ARD solution"""
        if not self.sparsity_analysis:
            return
        
        try:
            alpha = self.model_.alpha_
            active_features = alpha < self.threshold_lambda
            coef = self.model_.coef_
            
            # Basic sparsity metrics
            n_total = len(alpha)
            n_active = np.sum(active_features)
            n_pruned = n_total - n_active
            sparsity_ratio = n_pruned / n_total
            
            # Coefficient sparsity
            coef_nonzero = np.abs(coef) > 1e-10
            effective_sparsity = 1 - np.sum(coef_nonzero) / len(coef)
            
            # Alpha-based sparsity analysis
            alpha_log_range = np.log10(np.max(alpha)) - np.log10(np.min(alpha) + 1e-12)
            alpha_separation_quality = self._assess_alpha_separation(alpha, active_features)
            
            # Sparsity quality metrics
            sparsity_quality = self._assess_sparsity_quality(alpha, coef, active_features)
            
            self.sparsity_analysis_ = {
                'n_total_features': n_total,
                'n_active_features': n_active,
                'n_pruned_features': n_pruned,
                'sparsity_ratio': sparsity_ratio,
                'effective_sparsity': effective_sparsity,
                'alpha_log_range': alpha_log_range,
                'alpha_separation_quality': alpha_separation_quality,
                'sparsity_quality_metrics': sparsity_quality,
                'threshold_lambda': self.threshold_lambda,
                'sparsity_interpretation': self._interpret_sparsity_level(sparsity_ratio),
                'feature_pruning_summary': {
                    'pruned_feature_names': [self.feature_names_[i] for i in range(n_total) if not active_features[i]],
                    'active_feature_names': [self.feature_names_[i] for i in range(n_total) if active_features[i]],
                    'pruning_effectiveness': 'High' if sparsity_ratio > 0.3 else 'Moderate' if sparsity_ratio > 0.1 else 'Low'
                }
            }
            
        except Exception as e:
            self.sparsity_analysis_ = {
                'error': f'Could not analyze sparsity: {str(e)}'
            }
    
    def _assess_alpha_separation(self, alpha, active_features):
        """Assess quality of separation between active and pruned features"""
        try:
            if np.sum(active_features) == 0 or np.sum(~active_features) == 0:
                return {'separation_exists': False, 'quality': 'N/A'}
            
            alpha_active = alpha[active_features]
            alpha_pruned = alpha[~active_features]
            
            # Calculate separation metrics
            max_active = np.max(alpha_active)
            min_pruned = np.min(alpha_pruned)
            
            separation_gap = min_pruned - max_active
            separation_ratio = min_pruned / max_active if max_active > 0 else float('inf')
            
            # Quality assessment
            if separation_ratio > 100:
                quality = 'Excellent'
            elif separation_ratio > 10:
                quality = 'Good'
            elif separation_ratio > 2:
                quality = 'Fair'
            else:
                quality = 'Poor'
            
            return {
                'separation_exists': separation_gap > 0,
                'separation_gap': separation_gap,
                'separation_ratio': separation_ratio,
                'quality': quality,
                'max_active_alpha': max_active,
                'min_pruned_alpha': min_pruned
            }
            
        except Exception as e:
            return {'error': f'Could not assess separation: {str(e)}'}
    
    def _assess_sparsity_quality(self, alpha, coef, active_features):
        """Assess overall quality of the sparse solution"""
        try:
            # Sparsity consistency (alpha and coefficient alignment)
            alpha_coef_correlation = stats.spearmanr(alpha, np.abs(coef))[0] if len(alpha) > 2 else 0
            
            # Active feature quality
            if np.sum(active_features) > 0:
                active_coef_magnitude = np.mean(np.abs(coef[active_features]))
                active_alpha_consistency = np.std(alpha[active_features]) / np.mean(alpha[active_features])
            else:
                active_coef_magnitude = 0
                active_alpha_consistency = 0
            
            # Pruning decisiveness
            pruning_decisiveness = np.mean(alpha[~active_features]) / (np.mean(alpha[active_features]) + 1e-12) if np.sum(active_features) > 0 else 0
            
            return {
                'alpha_coef_correlation': alpha_coef_correlation,
                'active_coef_magnitude': active_coef_magnitude,
                'active_alpha_consistency': active_alpha_consistency,
                'pruning_decisiveness': pruning_decisiveness,
                'overall_quality_score': (
                    abs(alpha_coef_correlation) * 0.3 +
                    min(pruning_decisiveness / 10, 1) * 0.4 +
                    max(0, 1 - active_alpha_consistency) * 0.3
                )
            }
            
        except Exception as e:
            return {'error': f'Could not assess sparsity quality: {str(e)}'}
    
    def _interpret_sparsity_level(self, sparsity_ratio):
        """Provide interpretation of sparsity level"""
        if sparsity_ratio > 0.7:
            return "Very high sparsity - most features deemed irrelevant"
        elif sparsity_ratio > 0.5:
            return "High sparsity - significant feature pruning"
        elif sparsity_ratio > 0.3:
            return "Moderate sparsity - selective feature pruning"
        elif sparsity_ratio > 0.1:
            return "Low sparsity - minor feature pruning"
        else:
            return "Minimal sparsity - most features retained"
    
    def _analyze_alpha_evolution(self):
        """Analyze evolution of alpha parameters (if tracking enabled)"""
        if not self.track_alpha_evolution:
            return
        
        try:
            # Final alpha values analysis
            alpha_final = self.model_.alpha_
            
            # Alpha evolution characteristics
            evolution_characteristics = {
                'final_alpha_values': alpha_final,
                'alpha_range': np.max(alpha_final) - np.min(alpha_final),
                'alpha_geometric_mean': np.exp(np.mean(np.log(alpha_final + 1e-12))),
                'alpha_log_std': np.std(np.log(alpha_final + 1e-12)),
                'dominant_alpha': np.max(alpha_final),
                'smallest_alpha': np.min(alpha_final),
                'alpha_concentration': np.sum(alpha_final > self.threshold_lambda) / len(alpha_final)
            }
            
            # Feature evolution summary
            active_features = alpha_final < self.threshold_lambda
            evolution_summary = []
            
            for i, (name, alpha_val, is_active) in enumerate(zip(
                self.feature_names_, alpha_final, active_features
            )):
                evolution_summary.append({
                    'feature_name': name,
                    'feature_index': i,
                    'final_alpha': alpha_val,
                    'is_active': is_active,
                    'relevance_score': 1.0 / (alpha_val + 1e-12),
                    'status': 'Active' if is_active else 'Pruned'
                })
            
            # Sort by relevance
            evolution_summary.sort(key=lambda x: x['relevance_score'], reverse=True)
            
            self.alpha_evolution_analysis_ = {
                'evolution_characteristics': evolution_characteristics,
                'feature_evolution_summary': evolution_summary,
                'convergence_pattern': self._analyze_alpha_convergence_pattern(alpha_final),
                'pruning_summary': {
                    'total_pruned': np.sum(~active_features),
                    'pruning_ratio': np.sum(~active_features) / len(active_features),
                    'threshold_effectiveness': np.sum(alpha_final > self.threshold_lambda * 10) / len(alpha_final)
                }
            }
            
        except Exception as e:
            self.alpha_evolution_analysis_ = {
                'error': f'Could not analyze alpha evolution: {str(e)}'
            }
    
    def _analyze_alpha_convergence_pattern(self, alpha_final):
        """Analyze the pattern of alpha convergence"""
        try:
            # Categorize alpha values
            very_large = alpha_final > self.threshold_lambda * 10
            large = (alpha_final > self.threshold_lambda) & (alpha_final <= self.threshold_lambda * 10)
            medium = (alpha_final > 1.0) & (alpha_final <= self.threshold_lambda)
            small = alpha_final <= 1.0
            
            pattern_analysis = {
                'very_large_alpha_count': np.sum(very_large),
                'large_alpha_count': np.sum(large),
                'medium_alpha_count': np.sum(medium),
                'small_alpha_count': np.sum(small),
                'convergence_pattern': 'Decisive' if np.sum(very_large) > len(alpha_final) * 0.1 else 'Gradual',
                'alpha_clustering': self._assess_alpha_clustering(alpha_final)
            }
            
            return pattern_analysis
            
        except Exception as e:
            return {'error': f'Could not analyze convergence pattern: {str(e)}'}
    
    def _assess_alpha_clustering(self, alpha_final):
        """Assess clustering of alpha values"""
        try:
            # Simple clustering assessment using log-scale gaps
            log_alpha = np.log(alpha_final + 1e-12)
            log_alpha_sorted = np.sort(log_alpha)
            
            # Find gaps in log-alpha space
            gaps = np.diff(log_alpha_sorted)
            large_gaps = gaps > np.std(gaps) * 2
            
            return {
                'n_clusters_estimated': np.sum(large_gaps) + 1,
                'largest_gap': np.max(gaps),
                'gap_variance': np.var(gaps),
                'clustering_quality': 'High' if np.sum(large_gaps) > 0 else 'Low'
            }
            
        except Exception as e:
            return {'error': f'Could not assess clustering: {str(e)}'}
    
    def _get_prediction_relevance_info(self, X):
        """Get feature relevance information for predictions"""
        try:
            alpha = self.model_.alpha_
            active_features = alpha < self.threshold_lambda
            
            if np.sum(active_features) == 0:
                return {'message': 'No active features found'}
            
            # Feature contributions to predictions
            feature_contributions = {}
            active_indices = np.where(active_features)[0]
            
            for i in active_indices:
                feature_contributions[self.feature_names_[i]] = {
                    'alpha': alpha[i],
                    'coefficient': self.model_.coef_[i],
                    'relevance_score': 1.0 / (alpha[i] + 1e-12),
                    'avg_input_magnitude': np.mean(np.abs(X[:, i]))
                }
            
            return {
                'n_active_features': np.sum(active_features),
                'feature_contributions': feature_contributions,
                'sparsity_level': 1 - np.sum(active_features) / len(alpha)
            }
            
        except Exception as e:
            return {'error': f'Could not get relevance info: {str(e)}'}
    
    def get_hyperparameter_config(self, key_prefix: str) -> Dict[str, Any]:
        """Generate Streamlit UI for hyperparameters"""
        
        # Create tabs for different configuration aspects
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "ARD Config", "Prior Specification", "Feature Selection", "Analysis Options", "Algorithm Info"
        ])
        
        with tab1:
            st.markdown("**ARD Regression Configuration**")
            
            col1, col2 = st.columns(2)
            
            with col1:
                n_iter = st.number_input(
                    "Maximum Iterations:",
                    value=self.n_iter,
                    min_value=10,
                    max_value=1000,
                    step=10,
                    help="Maximum number of iterations for ARD inference",
                    key=f"{key_prefix}_n_iter"
                )
                
                tol = st.number_input(
                    "Convergence Tolerance:",
                    value=self.tol,
                    min_value=1e-6,
                    max_value=1e-1,
                    step=1e-6,
                    format="%.1e",
                    help="Tolerance for convergence criterion",
                    key=f"{key_prefix}_tol"
                )
                
                fit_intercept = st.checkbox(
                    "Fit Intercept",
                    value=self.fit_intercept,
                    help="Whether to fit intercept term",
                    key=f"{key_prefix}_fit_intercept"
                )
                
                normalize_features = st.checkbox(
                    "Normalize Features",
                    value=self.normalize_features,
                    help="Apply StandardScaler to features",
                    key=f"{key_prefix}_normalize_features"
                )
            
            with col2:
                compute_score = st.checkbox(
                    "Compute Evidence Scores",
                    value=self.compute_score,
                    help="Compute log marginal likelihood at each iteration",
                    key=f"{key_prefix}_compute_score"
                )
                
                verbose = st.checkbox(
                    "Verbose Output",
                    value=self.verbose,
                    help="Print convergence information",
                    key=f"{key_prefix}_verbose"
                )
                
                copy_X = st.checkbox(
                    "Copy Input Data",
                    value=self.copy_X,
                    help="Whether to copy input data (memory vs safety trade-off)",
                    key=f"{key_prefix}_copy_X"
                )
        
        with tab2:
            st.markdown("**Prior Specification**")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Precision Priors (Gamma Distribution)**")
                
                alpha_1 = st.number_input(
                    "Alpha₁ (weight precision shape):",
                    value=self.alpha_1,
                    min_value=1e-10,
                    max_value=1e2,
                    step=1e-7,
                    format="%.1e",
                    help="Shape parameter for individual weight precision priors Gamma(α₁, α₂)",
                    key=f"{key_prefix}_alpha_1"
                )
                
                alpha_2 = st.number_input(
                    "Alpha₂ (weight precision rate):",
                    value=self.alpha_2,
                    min_value=1e-10,
                    max_value=1e2,
                    step=1e-7,
                    format="%.1e",
                    help="Rate parameter for individual weight precision priors Gamma(α₁, α₂)",
                    key=f"{key_prefix}_alpha_2"
                )
                
                lambda_1 = st.number_input(
                    "Lambda₁ (noise precision shape):",
                    value=self.lambda_1,
                    min_value=1e-10,
                    max_value=1e2,
                    step=1e-7,
                    format="%.1e",
                    help="Shape parameter for noise precision prior Gamma(λ₁, λ₂)",
                    key=f"{key_prefix}_lambda_1"
                )
                
                lambda_2 = st.number_input(
                    "Lambda₂ (noise precision rate):",
                    value=self.lambda_2,
                    min_value=1e-10,
                    max_value=1e2,
                    step=1e-7,
                    format="%.1e",
                    help="Rate parameter for noise precision prior Gamma(λ₁, λ₂)",
                    key=f"{key_prefix}_lambda_2"
                )
            
            with col2:
                st.markdown("**Initial Values (Optional)**")
                
                use_custom_init = st.checkbox(
                    "Use Custom Initial Values",
                    value=self.alpha_init is not None or self.lambda_init is not None,
                    help="Specify initial values for hyperparameters",
                    key=f"{key_prefix}_use_custom_init"
                )
                
                if use_custom_init:
                    alpha_init = st.number_input(
                        "Initial Alpha (individual precisions):",
                        value=self.alpha_init if self.alpha_init is not None else 1.0,
                        min_value=1e-10,
                        max_value=1e10,
                        step=0.1,
                        format="%.6f",
                        help="Initial value for individual weight precisions",
                        key=f"{key_prefix}_alpha_init"
                    )
                    
                    lambda_init = st.number_input(
                        "Initial Lambda (noise precision):",
                        value=self.lambda_init if self.lambda_init is not None else 1.0,
                        min_value=1e-10,
                        max_value=1e10,
                        step=0.1,
                        format="%.6f",
                        help="Initial value for noise precision",
                        key=f"{key_prefix}_lambda_init"
                    )
                else:
                    alpha_init = None
                    lambda_init = None
                
                st.markdown("**ARD Prior Information:**")
                st.info("""
                ARD uses **individual precision parameters** for each feature:
                • Each feature gets its own α_i parameter
                • High α_i → feature is irrelevant (pruned)
                • Low α_i → feature is relevant (kept)
                • Automatic feature selection emerges naturally
                """)
        
        with tab3:
            st.markdown("**Feature Selection Configuration**")
            
            col1, col2 = st.columns(2)
            
            with col1:
                threshold_lambda = st.number_input(
                    "Pruning Threshold (λ):",
                    value=self.threshold_lambda,
                    min_value=1.0,
                    max_value=1e6,
                    step=1000.0,
                    format="%.1f",
                    help="Threshold for feature pruning - features with α > threshold are pruned",
                    key=f"{key_prefix}_threshold_lambda"
                )
                
                automatic_pruning = st.checkbox(
                    "Automatic Pruning",
                    value=self.automatic_pruning,
                    help="Automatically prune irrelevant features during inference",
                    key=f"{key_prefix}_automatic_pruning"
                )
                
                relevance_threshold = st.number_input(
                    "Relevance Threshold:",
                    value=self.relevance_threshold,
                    min_value=1e-6,
                    max_value=1.0,
                    step=1e-3,
                    format="%.1e",
                    help="Threshold for determining feature relevance",
                    key=f"{key_prefix}_relevance_threshold"
                )
            
            with col2:
                track_alpha_evolution = st.checkbox(
                    "Track Alpha Evolution",
                    value=self.track_alpha_evolution,
                    help="Track evolution of individual alpha parameters",
                    key=f"{key_prefix}_track_alpha_evolution"
                )
                
                st.markdown("**Feature Selection Mechanism:**")
                st.info("""
                **ARD Feature Selection:**
                • Each feature gets precision parameter α_i
                • Model learns α_i automatically
                • α_i >> threshold → feature pruned
                • α_i << threshold → feature kept
                • Sparse solutions emerge naturally
                """)
                
                st.markdown("**Threshold Guidelines:**")
                st.success("""
                • **Conservative**: 1,000 - 10,000
                • **Moderate**: 10,000 - 100,000  
                • **Aggressive**: > 100,000
                """)
        
        with tab4:
            st.markdown("**Analysis Options**")
            
            col1, col2 = st.columns(2)
            
            with col1:
                convergence_analysis = st.checkbox(
                    "Convergence Analysis",
                    value=self.convergence_analysis,
                    help="Analyze convergence of ARD inference",
                    key=f"{key_prefix}_convergence_analysis"
                )
                
                evidence_analysis = st.checkbox(
                    "Evidence Analysis",
                    value=self.evidence_analysis,
                    help="Compute model evidence (marginal likelihood)",
                    key=f"{key_prefix}_evidence_analysis"
                )
                
                feature_selection_analysis = st.checkbox(
                    "Feature Selection Analysis",
                    value=self.feature_selection_analysis,
                    help="Analyze automatic feature selection results",
                    key=f"{key_prefix}_feature_selection_analysis"
                )
                
                sparsity_analysis = st.checkbox(
                    "Sparsity Analysis",
                    value=self.sparsity_analysis,
                    help="Analyze sparsity characteristics",
                    key=f"{key_prefix}_sparsity_analysis"
                )
            
            with col2:
                relevance_evolution_analysis = st.checkbox(
                    "Relevance Evolution Analysis",
                    value=self.relevance_evolution_analysis,
                    help="Track evolution of feature relevance during training",
                    key=f"{key_prefix}_relevance_evolution_analysis"
                )
                
                uncertainty_decomposition = st.checkbox(
                    "Uncertainty Decomposition",
                    value=self.uncertainty_decomposition,
                    help="Separate aleatoric and epistemic uncertainty",
                    key=f"{key_prefix}_uncertainty_decomposition"
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
            **ARD (Automatic Relevance Determination) Regression**:
            • 🎯 Automatic feature selection via individual precision parameters
            • 📊 Bayesian uncertainty quantification
            • ✂️ Sparse solutions with irrelevant features pruned
            • 🔄 Hierarchical Bayesian modeling with feature-specific priors
            • 📈 Evidence-based model selection
            • 🎲 Individual α_i parameters for each feature
            
            **Mathematical Foundation:**
            • Likelihood: p(y|X,w,α,λ) = N(Xw, λ⁻¹I)
            • Individual Priors: p(w_i|α_i) = N(0, α_i⁻¹)
            • Hyperpriors: p(α_i) = Gamma(α₁,α₂) for each feature i
            • Automatic Pruning: α_i >> threshold → feature i irrelevant
            """)
            
            # When to use ARD
            if st.button("🎯 When to Use ARD Regression", key=f"{key_prefix}_when_to_use"):
                st.markdown("""
                **Ideal Use Cases:**
                
                **Problem Characteristics:**
                • High-dimensional data with many potentially irrelevant features
                • Need automatic feature selection
                • Want uncertainty quantification
                • Suspected sparse underlying relationships
                • Need interpretable feature importance
                
                **Data Characteristics:**
                • More features than samples (p > n)
                • Multicollinear features
                • Noisy measurements
                • Mixed relevant/irrelevant features
                • Need robust feature selection
                
                **Examples:**
                • Gene expression analysis (thousands of genes, few relevant)
                • Financial modeling (many economic indicators)
                • Image/signal processing (many pixels/frequencies)
                • Scientific discovery (identify key factors)
                • Feature screening in machine learning pipelines
                """)
            
            # Advantages and limitations
            if st.button("⚖️ Advantages & Limitations", key=f"{key_prefix}_pros_cons"):
                st.markdown("""
                **Advantages:**
                ✅ Automatic feature selection (no manual tuning)
                ✅ Handles high-dimensional data (p >> n)
                ✅ Uncertainty quantification for both predictions and feature selection
                ✅ Robust to multicollinearity
                ✅ Interpretable feature relevance scores
                ✅ Principled Bayesian approach
                ✅ Sparse solutions reduce overfitting
                ✅ Evidence-based model comparison
                
                **Limitations:**
                ❌ Limited to linear relationships
                ❌ Assumes Gaussian noise and priors
                ❌ Computational cost scales with number of features
                ❌ May be overly aggressive in pruning
                ❌ Requires understanding of Bayesian concepts
                ❌ Sensitive to prior specification for small datasets
                """)
            
            # ARD vs other methods
            if st.button("🔍 ARD vs Other Methods", key=f"{key_prefix}_comparison"):
                st.markdown("""
                **ARD vs Other Feature Selection Methods:**
                
                **ARD vs LASSO:**
                • ARD: Bayesian, automatic λ tuning, uncertainty quantification
                • LASSO: Frequentist, requires cross-validation for λ
                • ARD: Smoother feature selection, less aggressive
                • LASSO: Can be more aggressive, exact zeros
                
                **ARD vs Ridge:**
                • ARD: Automatic feature selection
                • Ridge: No feature selection, shrinks all coefficients
                • ARD: Individual precision per feature
                • Ridge: Single regularization parameter
                
                **ARD vs Elastic Net:**
                • ARD: Automatic hyperparameter tuning
                • Elastic Net: Requires tuning α and λ
                • ARD: Uncertainty quantification
                • Elastic Net: Point estimates only
                
                **ARD vs Bayesian Ridge:**
                • ARD: Individual α_i per feature (feature selection)
                • Bayesian Ridge: Single α for all features (no selection)
                • ARD: Sparse solutions
                • Bayesian Ridge: Dense solutions
                """)
            
            # Best practices
            if st.button("🎯 Best Practices", key=f"{key_prefix}_best_practices"):
                st.markdown("""
                **ARD Regression Best Practices:**
                
                **Data Preparation:**
                1. **Normalize features** (essential for ARD)
                2. Remove constant or near-constant features
                3. Check for extreme multicollinearity
                4. Consider feature transformations if needed
                
                **Hyperparameter Selection:**
                1. Use weakly informative priors (defaults work well)
                2. Adjust threshold_lambda based on desired sparsity:
                   - Conservative: 1,000-10,000
                   - Moderate: 10,000-100,000
                   - Aggressive: >100,000
                3. Monitor convergence diagnostics
                
                **Model Validation:**
                1. Check feature selection stability across runs
                2. Validate selected features make domain sense
                3. Assess uncertainty calibration
                4. Compare evidence with alternative models
                5. Examine α parameter separation quality
                
                **Interpretation:**
                1. Focus on selected features (α_i < threshold)
                2. Use relevance scores for feature ranking
                3. Report uncertainty intervals
                4. Analyze sparsity patterns
                5. Consider biological/domain plausibility of selections
                """)
            
            # Advanced usage
            if st.button("🚀 Advanced Usage", key=f"{key_prefix}_advanced"):
                st.markdown("""
                **Advanced ARD Techniques:**
                
                **Hierarchical Feature Groups:**
                • Group related features with shared hyperpriors
                • Useful for structured data (e.g., gene pathways)
                • Implement custom group-wise ARD
                
                **Multi-task ARD:**
                • Share relevance patterns across related tasks
                • Useful for related prediction problems
                • Implement joint ARD across tasks
                
                **Non-linear ARD:**
                • Combine with kernel methods
                • ARD for feature selection + kernel for non-linearity
                • Gaussian Process with ARD covariance
                
                **Temporal ARD:**
                • Time-varying feature relevance
                • Useful for dynamic systems
                • Implement state-space ARD models
                
                **Robust ARD:**
                • Heavy-tailed noise models
                • Robust to outliers
                • Student-t likelihood instead of Gaussian
                """)
        
        return {
            "n_iter": n_iter,
            "tol": tol,
            "alpha_1": alpha_1,
            "alpha_2": alpha_2,
            "lambda_1": lambda_1,
            "lambda_2": lambda_2,
            "alpha_init": alpha_init,
            "lambda_init": lambda_init,
            "compute_score": compute_score,
            "fit_intercept": fit_intercept,
            "normalize_features": normalize_features,
            "copy_X": copy_X,
            "verbose": verbose,
            "threshold_lambda": threshold_lambda,
            "automatic_pruning": automatic_pruning,
            "relevance_threshold": relevance_threshold,
            "track_alpha_evolution": track_alpha_evolution,
            "convergence_analysis": convergence_analysis,
            "evidence_analysis": evidence_analysis,
            "feature_selection_analysis": feature_selection_analysis,
            "relevance_evolution_analysis": relevance_evolution_analysis,
            "uncertainty_decomposition": uncertainty_decomposition,
            "sparsity_analysis": sparsity_analysis,
            "random_state": random_state
        }
    
    def create_model_instance(self, hyperparameters: Dict[str, Any]):
        """Create model instance with given hyperparameters"""
        return ARDRegressionPlugin(
            n_iter=hyperparameters.get("n_iter", self.n_iter),
            tol=hyperparameters.get("tol", self.tol),
            alpha_1=hyperparameters.get("alpha_1", self.alpha_1),
            alpha_2=hyperparameters.get("alpha_2", self.alpha_2),
            lambda_1=hyperparameters.get("lambda_1", self.lambda_1),
            lambda_2=hyperparameters.get("lambda_2", self.lambda_2),
            alpha_init=hyperparameters.get("alpha_init", self.alpha_init),
            lambda_init=hyperparameters.get("lambda_init", self.lambda_init),
            compute_score=hyperparameters.get("compute_score", self.compute_score),
            fit_intercept=hyperparameters.get("fit_intercept", self.fit_intercept),
            normalize_features=hyperparameters.get("normalize_features", self.normalize_features),
            copy_X=hyperparameters.get("copy_X", self.copy_X),
            verbose=hyperparameters.get("verbose", self.verbose),
            threshold_lambda=hyperparameters.get("threshold_lambda", self.threshold_lambda),
            automatic_pruning=hyperparameters.get("automatic_pruning", self.automatic_pruning),
            relevance_threshold=hyperparameters.get("relevance_threshold", self.relevance_threshold),
            track_alpha_evolution=hyperparameters.get("track_alpha_evolution", self.track_alpha_evolution),
            convergence_analysis=hyperparameters.get("convergence_analysis", self.convergence_analysis),
            evidence_analysis=hyperparameters.get("evidence_analysis", self.evidence_analysis),
            feature_selection_analysis=hyperparameters.get("feature_selection_analysis", self.feature_selection_analysis),
            relevance_evolution_analysis=hyperparameters.get("relevance_evolution_analysis", self.relevance_evolution_analysis),
            uncertainty_decomposition=hyperparameters.get("uncertainty_decomposition", self.uncertainty_decomposition),
            sparsity_analysis=hyperparameters.get("sparsity_analysis", self.sparsity_analysis),
            random_state=hyperparameters.get("random_state", self.random_state)
        )
    
    def preprocess_data(self, X, y=None, training=True):
        """Preprocess data for ARD Regression"""
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
        """Check if ARD Regression is compatible with the given data"""
        # Check minimum samples
        if X.shape[0] < self._min_samples_required:
            return False, f"ARD requires at least {self._min_samples_required} samples, got {X.shape[0]}"
        
        # Check for regression targets
        if y is not None:
            if not np.issubdtype(y.dtype, np.number):
                return False, "ARD requires continuous numerical target values"
            
            # Check for sufficient variance in target
            if np.var(y) == 0:
                return False, "Target variable has zero variance (all values are the same)"
            
            n_samples, n_features = X.shape
            
            advantages = []
            considerations = []
            
            # High-dimensional data assessment (ARD's strength)
            if n_features > n_samples:
                advantages.append(f"High-dimensional data ({n_features} > {n_samples}) - ARD excels here")
            elif n_features > n_samples * 0.5:
                advantages.append(f"Moderate dimensionality ({n_features}) - good for ARD feature selection")
            else:
                considerations.append(f"Low dimensionality ({n_features}) - simpler methods might suffice")
            
            # Sample size assessment
            if n_samples >= 50:
                advantages.append(f"Good sample size ({n_samples}) for reliable ARD inference")
            elif n_samples >= 20:
                considerations.append(f"Moderate sample size ({n_samples}) - monitor convergence carefully")
            else:
                considerations.append(f"Small sample size ({n_samples}) - use informative priors")
            
            # Feature selection potential
            if n_features > 10:
                advantages.append(f"Many features ({n_features}) - automatic selection valuable")
                advantages.append("Individual feature precision parameters enable sparse solutions")
            else:
                considerations.append(f"Few features ({n_features}) - may not need automatic selection")
            
            # Multicollinearity assessment
            try:
                if n_features > 1 and n_samples > n_features:
                    corr_matrix = np.corrcoef(X.T)
                    high_corr = np.any(np.abs(corr_matrix - np.eye(n_features)) > 0.8)
                    
                    if high_corr:
                        advantages.append("High feature correlation - ARD handles multicollinearity well")
                    else:
                        advantages.append("Features relatively independent - good for ARD")
            except:
                pass
            
            # ARD-specific advantages
            advantages.append("Automatic feature selection (no hyperparameter tuning)")
            advantages.append("Uncertainty quantification for predictions and feature relevance")
            advantages.append("Robust to overfitting through sparsity")
            
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
                f"📊 Suitability for ARD: {suitability}"
            ]
            
            if advantages:
                message_parts.append("🎯 Advantages: " + "; ".join(advantages))
            
            if considerations:
                message_parts.append("💡 Considerations: " + "; ".join(considerations))
            
            return True, " | ".join(message_parts)
        
        return True, f"Compatible with {X.shape[0]} samples and {X.shape[1]} features"
    
    def get_feature_importance(self) -> Optional[Dict[str, Any]]:
        """Get ARD feature importance with automatic selection results"""
        if not self.is_fitted_:
            return None
        
        if not self.feature_selection_analysis_:
            return None
        
        analysis = self.feature_selection_analysis_
        
        if 'error' in analysis:
            return {'error': analysis['error']}
        
        # Extract ARD-specific importance information
        alpha_values = analysis['alpha_values']
        active_features = analysis['active_features']
        relevance_scores = analysis['relevance_scores']
        feature_importance = analysis['feature_importance']
        
        # Create feature importance dictionary
        feature_importance_dict = {}
        for i, name in enumerate(self.feature_names_):
            feature_importance_dict[name] = {
                'alpha_value': alpha_values[i],
                'relevance_score': relevance_scores[i],
                'importance_score': feature_importance[i],
                'is_selected': active_features[i],
                'coefficient': self.model_.coef_[i] if hasattr(self.model_, 'coef_') else 0,
                'selection_status': 'Selected' if active_features[i] else 'Pruned'
            }
        
        # Get top features
        top_features = analysis['top_features']
        
        return {
            'feature_importance': feature_importance_dict,
            'top_features': top_features,
            'selection_summary': {
                'n_selected': analysis['n_selected_features'],
                'n_pruned': analysis['n_pruned_features'],
                'selection_ratio': analysis['selection_ratio'],
                'threshold_lambda': analysis['threshold_lambda']
            },
            'ard_info': {
                'automatic_selection': True,
                'individual_alphas': True,
                'sparse_solution': True,
                'uncertainty_quantified': True
            },
            'interpretation': 'ARD feature importance with automatic relevance determination'
        }
    
    def get_model_params(self) -> Dict[str, Any]:
        """Get current model parameters"""
        if not self.is_fitted_:
            return {}
        
        # Get feature selection info
        alpha_values = self.model_.alpha_
        active_features = alpha_values < self.threshold_lambda
        
        return {
            "algorithm": "ARD Regression",
            "n_features_original": self.n_features_in_,
            "n_features_selected": np.sum(active_features),
            "sparsity_ratio": 1 - np.sum(active_features) / len(alpha_values),
            "coefficients": self.model_.coef_.tolist(),
            "intercept": self.model_.intercept_ if hasattr(self.model_, 'intercept_') else 0,
            "alpha_values": alpha_values.tolist(),
            "lambda": self.model_.lambda_,
            "threshold_lambda": self.threshold_lambda,
            "selected_features": np.where(active_features)[0].tolist(),
            "pruned_features": np.where(~active_features)[0].tolist(),
            "n_iterations": self.n_iter,
            "convergence_tolerance": self.tol,
            "fit_intercept": self.fit_intercept,
            "normalize_features": self.normalize_features,
            "ard_characteristics": {
                "automatic_feature_selection": True,
                "individual_precision_parameters": True,
                "sparse_solutions": True,
                "uncertainty_quantification": True,
                "bayesian_evidence_based": True
            }
        }
    
    def get_algorithm_specific_metrics(self,
                                       y_true: Union[pd.Series, np.ndarray],
                                       y_pred: Union[pd.Series, np.ndarray],
                                       y_proba: Optional[np.ndarray] = None  # y_proba is not used for regressors
                                       ) -> Dict[str, Any]:
        """
        Calculate ARD Regression-specific metrics related to its Bayesian properties,
        feature relevance, and sparsity.

        Args:
            y_true: Ground truth target values from the test set.
            y_pred: Predicted target values on the test set.
            y_proba: Predicted probabilities (not used for regressors).

        Returns:
            A dictionary of ARD Regression-specific metrics.
        """
        metrics = {}
        if not self.is_fitted_ or self.model_ is None:
            metrics["status"] = "Model not fitted or not available"
            return metrics

        # --- Core ARD Model Parameters ---
        if hasattr(self.model_, 'lambda_'):
            metrics['noise_precision_lambda'] = float(self.model_.lambda_)
            metrics['estimated_noise_variance'] = 1.0 / float(self.model_.lambda_) if self.model_.lambda_ > 0 else float('inf')

        if hasattr(self.model_, 'alpha_'):
            alpha_values = self.model_.alpha_
            metrics['mean_feature_precision_alpha'] = float(np.mean(alpha_values))
            metrics['std_feature_precision_alpha'] = float(np.std(alpha_values))
            metrics['min_feature_precision_alpha'] = float(np.min(alpha_values))
            metrics['max_feature_precision_alpha'] = float(np.max(alpha_values))
            # Relevance is inverse of alpha
            relevance_scores = 1.0 / (alpha_values + 1e-12) # Add epsilon for stability
            metrics['mean_relevance_score'] = float(np.mean(relevance_scores))


        # --- Metrics from Feature Selection Analysis ---
        if hasattr(self, 'feature_selection_analysis_') and self.feature_selection_analysis_:
            fs_analysis = self.feature_selection_analysis_
            if 'error' not in fs_analysis:
                metrics['n_selected_features'] = fs_analysis.get('n_selected_features')
                metrics['n_pruned_features'] = fs_analysis.get('n_pruned_features')
                metrics['selection_ratio'] = fs_analysis.get('selection_ratio') # n_selected / n_total
                if fs_analysis.get('alpha_statistics') and 'separation_metrics' in fs_analysis['alpha_statistics']:
                    sep_metrics = fs_analysis['alpha_statistics']['separation_metrics']
                    metrics['alpha_separation_ratio'] = sep_metrics.get('ratio')
                    metrics['alpha_separation_quality'] = sep_metrics.get('quality')

        # --- Metrics from Evidence Analysis ---
        if hasattr(self, 'evidence_analysis_') and self.evidence_analysis_:
            ev_analysis = self.evidence_analysis_
            if 'error' not in ev_analysis:
                metrics['log_marginal_likelihood'] = ev_analysis.get('log_marginal_likelihood')
                metrics['effective_parameters_in_model'] = ev_analysis.get('effective_parameters')
                metrics['aic_from_evidence'] = ev_analysis.get('aic')
                metrics['bic_from_evidence'] = ev_analysis.get('bic')
                if ev_analysis.get('ard_specific_metrics'):
                    ard_spec = ev_analysis['ard_specific_metrics']
                    if ard_spec.get('feature_selection_quality'):
                        metrics['feature_selection_quality_score'] = ard_spec['feature_selection_quality'].get('selection_quality_score')
                    if ard_spec.get('sparsity_vs_fit_tradeoff'):
                        metrics['sparsity_fit_tradeoff_score'] = ard_spec['sparsity_vs_fit_tradeoff'].get('tradeoff_score')


        # --- Metrics from Sparsity Analysis ---
        if hasattr(self, 'sparsity_analysis_') and self.sparsity_analysis_:
            sp_analysis = self.sparsity_analysis_
            if 'error' not in sp_analysis:
                metrics['sparsity_ratio_alpha_based'] = sp_analysis.get('sparsity_ratio') # (n_pruned / n_total)
                metrics['effective_sparsity_coef_based'] = sp_analysis.get('effective_sparsity') # (1 - n_nonzero_coef / n_total)
                if sp_analysis.get('sparsity_quality_metrics'):
                    qual_metrics = sp_analysis['sparsity_quality_metrics']
                    metrics['sparsity_quality_overall_score'] = qual_metrics.get('overall_quality_score')
                    metrics['sparsity_alpha_coef_correlation'] = qual_metrics.get('alpha_coef_correlation')
                metrics['sparsity_interpretation'] = sp_analysis.get('sparsity_interpretation')

        # --- Metrics from Convergence Analysis ---
        if hasattr(self, 'convergence_analysis_') and self.convergence_analysis_:
            conv_analysis = self.convergence_analysis_
            if 'error' not in conv_analysis:
                metrics['n_iterations_to_converge'] = conv_analysis.get('n_iterations')
                metrics['converged_in_training'] = conv_analysis.get('converged')
                metrics['final_convergence_score'] = conv_analysis.get('final_score') # e.g. log marginal likelihood
                if conv_analysis.get('alpha_convergence'):
                     metrics['n_active_features_at_convergence'] = conv_analysis['alpha_convergence'].get('n_active_features')


        # --- Coefficients related metrics (sparsity context) ---
        if hasattr(self.model_, 'coef_'):
            coefs = self.model_.coef_
            metrics['n_non_zero_coefficients'] = int(np.sum(np.abs(coefs) > 1e-10)) # Practical non-zero
            metrics['mean_abs_coefficient_selected'] = float(np.mean(np.abs(coefs[self.model_.alpha_ < self.threshold_lambda]))) if np.any(self.model_.alpha_ < self.threshold_lambda) else 0.0

        return metrics
    
    def get_ard_analysis(self) -> Dict[str, Any]:
        """Get comprehensive ARD analysis results"""
        if not self.is_fitted_:
            return {"status": "Model not fitted"}
        
        return {
            "convergence_analysis": self.convergence_analysis_,
            "evidence_analysis": self.evidence_analysis_,
            "feature_selection_analysis": self.feature_selection_analysis_,
            "relevance_evolution_analysis": self.relevance_evolution_analysis_,
            "uncertainty_analysis": self.uncertainty_analysis_,
            "sparsity_analysis": self.sparsity_analysis_,
            "alpha_evolution_analysis": self.alpha_evolution_analysis_
        }


# Factory function
def get_plugin():
    """Factory function to get the plugin instance"""
    return ARDRegressionPlugin()


# Example usage and testing
if __name__ == "__main__":
    """
    Example usage of ARD Regression Plugin
    """
    print("Testing ARD Regression Plugin...")
    
    try:
        # Create sample high-dimensional data
        np.random.seed(42)
        
        # Generate synthetic regression data with many irrelevant features
        from sklearn.datasets import make_regression
        X, y = make_regression(
            n_samples=100,
            n_features=50,  # Many features
            n_informative=5,  # Only 5 are actually relevant
            n_redundant=10,
            noise=0.1,
            random_state=42
        )
        
        print(f"\n📊 Test Dataset:")
        print(f"Samples: {X.shape[0]}, Features: {X.shape[1]}")
        print(f"True informative features: 5")
        print(f"Target variance: {np.var(y):.3f}")
        
        # Test ARD regression
        print(f"\n🧪 Testing ARD Regression...")
        
        # Create DataFrame for proper feature names
        feature_names = [f'feature_{i}' for i in range(X.shape[1])]
        X_df = pd.DataFrame(X, columns=feature_names)
        
        plugin = ARDRegressionPlugin(
            n_iter=300,
            tol=1e-3,
            compute_score=True,
            threshold_lambda=10000.0,
            normalize_features=True,
            automatic_pruning=True,
            feature_selection_analysis=True,
            relevance_evolution_analysis=True,
            sparsity_analysis=True,
            track_alpha_evolution=True,
            random_state=42
        )
        
        # Check compatibility
        compatible, message = plugin.is_compatible_with_data(X_df, y)
        print(f"✅ Compatibility: {message}")
        
        if compatible:
            # Train model
            plugin.fit(X_df, y)
            
            # Make predictions with uncertainty
            y_pred = plugin.predict(X_df)
            uncertainty_results = plugin.predict_with_uncertainty(X_df)
            
            # Evaluate
            from sklearn.metrics import r2_score, mean_squared_error
            r2 = r2_score(y, y_pred)
            mse = mean_squared_error(y, y_pred)
            
            print(f"\n📊 ARD Results:")
            print(f"R²: {r2:.4f}")
            print(f"MSE: {mse:.4f}")
            print(f"Mean prediction uncertainty: {np.mean(uncertainty_results['std_dev']):.4f}")
            
            # Get model parameters
            model_params = plugin.get_model_params()
            print(f"\nFeature Selection Results:")
            print(f"Original features: {model_params['n_features_original']}")
            print(f"Selected features: {model_params['n_features_selected']}")
            print(f"Sparsity ratio: {model_params['sparsity_ratio']:.3f}")
            
            # Test feature importance
            importance = plugin.get_feature_importance()
            if importance and 'top_features' in importance:
                print(f"\n🎯 Top Selected Features:")
                for i, (name, alpha, relevance, imp, selected) in enumerate(importance['top_features'][:5]):
                    status = "✅ Selected" if selected else "❌ Pruned"
                    print(f"  {i+1}. {name}: α={alpha:.2e}, relevance={relevance:.3f} {status}")
            
            # Get ARD analysis
            analysis = plugin.get_ard_analysis()
            
            # Check feature selection quality
            if 'feature_selection_analysis' in analysis:
                fs = analysis['feature_selection_analysis']
                if 'alpha_statistics' in fs:
                    alpha_stats = fs['alpha_statistics']
                    if 'separation_metrics' in alpha_stats:
                        sep = alpha_stats['separation_metrics']
                        print(f"\nFeature Selection Quality:")
                        print(f"Alpha separation ratio: {sep.get('ratio', 'N/A'):.1f}")
                        print(f"Well separated: {sep.get('well_separated', False)}")
            
            # Check sparsity analysis
            if 'sparsity_analysis' in analysis:
                sparse = analysis['sparsity_analysis']
                if 'sparsity_interpretation' in sparse:
                    print(f"Sparsity level: {sparse['sparsity_interpretation']}")
        
        print("\n✅ ARD Regression Plugin test completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Error testing ARD Plugin: {str(e)}")
        import traceback
        traceback.print_exc()