import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, Optional, Tuple, List
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_X_y, check_array
from sklearn.linear_model import BayesianRidge
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


class BayesianRidgePlugin(BaseEstimator, RegressorMixin, MLPlugin):
    """
    Bayesian Ridge Regression Plugin - Probabilistic Linear Regression
    
    This plugin implements Bayesian Ridge Regression, which provides a probabilistic
    approach to linear regression with automatic regularization parameter estimation
    and uncertainty quantification. Unlike classical ridge regression, it treats
    model parameters as probability distributions and provides prediction intervals.
    
    Key Features:
    - Automatic hyperparameter estimation via Bayesian inference
    - Uncertainty quantification with prediction intervals
    - Robust to overfitting through Bayesian regularization
    - Evidence-based model selection
    - Conjugate prior distributions for tractable inference
    - Advanced diagnostics and convergence analysis
    - Hierarchical Bayesian modeling capabilities
    """
    
    def __init__(
        self,
        # Bayesian hyperparameters
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
        
        # Advanced Bayesian options
        copy_X=True,
        verbose=False,
        
        # Uncertainty quantification
        compute_prediction_intervals=True,
        confidence_level=0.95,
        n_posterior_samples=1000,
        
        # Analysis options
        convergence_analysis=True,
        evidence_analysis=True,
        hyperparameter_evolution=True,
        uncertainty_decomposition=True,
        feature_relevance_analysis=True,
        
        # Prior specification
        use_informative_priors=False,
        prior_mean=None,
        prior_precision=None,
        
        random_state=42
    ):
        super().__init__()
        
        # Core Bayesian parameters
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
        
        # Uncertainty quantification
        self.compute_prediction_intervals = compute_prediction_intervals
        self.confidence_level = confidence_level
        self.n_posterior_samples = n_posterior_samples
        
        # Analysis options
        self.convergence_analysis = convergence_analysis
        self.evidence_analysis = evidence_analysis
        self.hyperparameter_evolution = hyperparameter_evolution
        self.uncertainty_decomposition = uncertainty_decomposition
        self.feature_relevance_analysis = feature_relevance_analysis
        
        # Prior specification
        self.use_informative_priors = use_informative_priors
        self.prior_mean = prior_mean
        self.prior_precision = prior_precision
        
        self.random_state = random_state
        
        # Required plugin metadata
        self._name = "Bayesian Ridge Regression"
        self._description = "Probabilistic linear regression with automatic regularization and uncertainty quantification"
        self._category = "Bayesian Models"
        
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
        
        # Bayesian analysis results
        self.convergence_analysis_ = {}
        self.evidence_analysis_ = {}
        self.hyperparameter_evolution_ = {}
        self.uncertainty_analysis_ = {}
        self.feature_relevance_analysis_ = {}
        self.posterior_analysis_ = {}
    
    def get_name(self) -> str:
        return self._name
    
    def get_description(self) -> str:
        return self._description
    
    def get_category(self) -> str:
        return self._category
    
    def fit(self, X, y, sample_weight=None):
        """
        Fit the Bayesian Ridge Regression model with comprehensive Bayesian analysis
        
        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            Training data
        y : array-like of shape (n_samples,)
            Target values
        sample_weight : array-like of shape (n_samples,), optional
            Individual weights for each sample (not directly supported by BayesianRidge)
        
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
        
        # Set up prior parameters if using informative priors
        if self.use_informative_priors:
            alpha_init, lambda_init = self._setup_informative_priors(X_processed, y)
        else:
            alpha_init = self.alpha_init
            lambda_init = self.lambda_init
        
        # Create and configure Bayesian Ridge model
        self.model_ = BayesianRidge(
            n_iter=self.n_iter,
            tol=self.tol,
            alpha_1=self.alpha_1,
            alpha_2=self.alpha_2,
            lambda_1=self.lambda_1,
            lambda_2=self.lambda_2,
            alpha_init=alpha_init,
            lambda_init=lambda_init,
            compute_score=self.compute_score,
            fit_intercept=self.fit_intercept,
            copy_X=self.copy_X,
            verbose=self.verbose
        )
        
        # Fit the model
        self.model_.fit(X_processed, y)
        
        # Perform comprehensive Bayesian analysis
        self._analyze_convergence()
        self._analyze_evidence()
        self._analyze_hyperparameter_evolution()
        self._analyze_uncertainty()
        self._analyze_feature_relevance()
        self._analyze_posterior_distribution()
        
        self.is_fitted_ = True
        return self
    
    def predict(self, X, return_std=False):
        """
        Make predictions using the fitted Bayesian model
        
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
            'uncertainty_decomposition': uncertainty_decomposition
        }
    
    def _setup_informative_priors(self, X, y):
        """Set up informative priors based on data or user specification"""
        if self.prior_mean is not None and self.prior_precision is not None:
            # Use user-specified priors
            alpha_init = self.prior_precision
            lambda_init = 1.0 / np.var(y - np.mean(y))
        else:
            # Set empirical priors based on data
            # Use small fraction of data variance for noise precision
            lambda_init = 1.0 / (0.1 * np.var(y))
            # Use regularization that keeps coefficients reasonable
            alpha_init = 1.0 / (0.1 * np.var(y) / X.shape[1])
        
        return alpha_init, lambda_init
    
    def _analyze_convergence(self):
        """Analyze convergence of the Bayesian inference"""
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
                'early_stopping': n_iterations < self.n_iter
            }
        else:
            self.convergence_analysis_ = {
                'error': 'Convergence scores not available (compute_score=False)'
            }
    
    def _estimate_convergence_rate(self, scores):
        """Estimate convergence rate from score evolution"""
        if len(scores) < 5:
            return None
        
        # Use exponential decay model: score[t] = a * exp(-b * t) + c
        # Linear fit in log space for last part of convergence
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
            else:
                X_design = X_processed
            
            # Calculate evidence components
            evidence_components = self._calculate_evidence_components(
                X_design, self.y_original_, alpha, lambda_
            )
            
            # Model complexity penalty
            effective_parameters = self._calculate_effective_parameters(X_design, alpha, lambda_)
            
            # Information criteria
            n_samples = len(self.y_original_)
            log_likelihood = evidence_components['log_likelihood']
            
            aic = 2 * effective_parameters - 2 * log_likelihood
            bic = effective_parameters * np.log(n_samples) - 2 * log_likelihood
            
            self.evidence_analysis_ = {
                'log_marginal_likelihood': evidence_components['log_marginal_likelihood'],
                'log_likelihood': log_likelihood,
                'complexity_penalty': evidence_components['complexity_penalty'],
                'effective_parameters': effective_parameters,
                'aic': aic,
                'bic': bic,
                'alpha': alpha,
                'lambda': lambda_,
                'evidence_components': evidence_components
            }
            
        except Exception as e:
            self.evidence_analysis_ = {
                'error': f'Could not compute evidence: {str(e)}'
            }
    
    def _calculate_evidence_components(self, X, y, alpha, lambda_):
        """Calculate components of the log marginal likelihood"""
        n_samples, n_features = X.shape
        
        # Posterior covariance matrix
        S_inv = alpha * np.eye(n_features) + lambda_ * X.T @ X
        try:
            S = np.linalg.inv(S_inv)
            log_det_S_inv = np.linalg.slogdet(S_inv)[1]
        except:
            # Use pseudo-inverse for numerical stability
            S = np.linalg.pinv(S_inv)
            log_det_S_inv = np.sum(np.log(np.linalg.eigvals(S_inv) + 1e-12))
        
        # Posterior mean
        mu = lambda_ * S @ X.T @ y
        
        # Residuals
        residuals = y - X @ mu
        
        # Log likelihood
        log_likelihood = -0.5 * lambda_ * np.sum(residuals**2)
        log_likelihood -= 0.5 * n_samples * np.log(2 * np.pi / lambda_)
        
        # Prior term
        prior_term = -0.5 * alpha * np.sum(mu**2)
        prior_term -= 0.5 * n_features * np.log(2 * np.pi / alpha)
        
        # Complexity penalty (negative log determinant of posterior covariance)
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
    
    def _calculate_effective_parameters(self, X, alpha, lambda_):
        """Calculate effective number of parameters"""
        try:
            # Effective degrees of freedom = tr(H) where H is the hat matrix
            # For Bayesian ridge: H = X @ (alpha*I + lambda*X.T@X)^(-1) @ lambda*X.T
            S_inv = alpha * np.eye(X.shape[1]) + lambda_ * X.T @ X
            H = X @ np.linalg.solve(S_inv, lambda_ * X.T)
            effective_params = np.trace(H)
            return effective_params
        except:
            # Fallback: use number of features
            return X.shape[1]
    
    def _analyze_hyperparameter_evolution(self):
        """Analyze evolution of hyperparameters during inference"""
        if not self.hyperparameter_evolution:
            return
        
        # Get final hyperparameter values
        alpha_final = self.model_.alpha_
        lambda_final = self.model_.lambda_
        
        # Calculate hyperparameter ratios and interpretations
        noise_precision = lambda_final
        weight_precision = alpha_final
        
        # Signal-to-noise ratio
        snr = weight_precision / noise_precision
        
        # Effective regularization strength
        effective_alpha = alpha_final / lambda_final
        
        self.hyperparameter_evolution_ = {
            'alpha_initial': self.alpha_init,
            'lambda_initial': self.lambda_init,
            'alpha_final': alpha_final,
            'lambda_final': lambda_final,
            'alpha_change_factor': alpha_final / (self.alpha_init or 1.0),
            'lambda_change_factor': lambda_final / (self.lambda_init or 1.0),
            'noise_precision': noise_precision,
            'weight_precision': weight_precision,
            'signal_to_noise_ratio': snr,
            'effective_regularization': effective_alpha,
            'hyperparameter_interpretation': self._interpret_hyperparameters(
                alpha_final, lambda_final
            )
        }
    
    def _interpret_hyperparameters(self, alpha, lambda_):
        """Provide interpretation of learned hyperparameters"""
        interpretation = []
        
        # Weight precision interpretation
        if alpha > 1.0:
            interpretation.append("Strong weight regularization - prefers sparse solutions")
        elif alpha > 0.1:
            interpretation.append("Moderate weight regularization")
        else:
            interpretation.append("Weak weight regularization - prefers complex models")
        
        # Noise precision interpretation
        if lambda_ > 100:
            interpretation.append("Low noise assumption - data fits model well")
        elif lambda_ > 1:
            interpretation.append("Moderate noise level")
        else:
            interpretation.append("High noise assumption - uncertain predictions")
        
        # Relative interpretation
        ratio = alpha / lambda_
        if ratio > 1:
            interpretation.append("Model complexity penalty dominates")
        elif ratio > 0.1:
            interpretation.append("Balanced complexity vs fit trade-off")
        else:
            interpretation.append("Data fit dominates over complexity penalty")
        
        return interpretation
    
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
            
            self.uncertainty_analysis_ = {
                'prediction_uncertainties': y_std,
                'uncertainty_statistics': uncertainty_stats,
                'uncertainty_decomposition': uncertainty_decomp,
                'high_uncertainty_indices': np.where(high_uncertainty_mask)[0],
                'uncertainty_threshold': uncertainty_threshold,
                'uncertainty_calibration': self._assess_uncertainty_calibration(
                    self.y_original_, y_pred, y_std
                )
            }
            
        except Exception as e:
            self.uncertainty_analysis_ = {
                'error': f'Could not analyze uncertainty: {str(e)}'
            }
    
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
            
            # Total uncertainty check
            total_uncertainty_check = np.sqrt(aleatoric_std**2 + epistemic_std**2)
            
            return {
                'aleatoric_uncertainty': aleatoric_std,
                'epistemic_uncertainty': epistemic_std,
                'total_uncertainty': y_std,
                'aleatoric_percentage': (aleatoric_std**2 / y_std**2) * 100,
                'epistemic_percentage': (epistemic_variance / y_std**2) * 100,
                'uncertainty_decomposition_valid': np.allclose(y_std, total_uncertainty_check, rtol=0.1)
            }
            
        except Exception as e:
            return {'error': f'Could not decompose uncertainty: {str(e)}'}
    
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
    
    def _analyze_feature_relevance(self):
        """Analyze feature relevance from Bayesian perspective"""
        if not self.feature_relevance_analysis:
            return
        
        try:
            # Get posterior mean and covariance
            coef_mean = self.model_.coef_
            
            # Calculate feature relevance based on coefficient magnitude and uncertainty
            coef_abs = np.abs(coef_mean)
            
            # Estimate coefficient uncertainties (simplified)
            # In full Bayesian treatment, we'd use posterior covariance
            alpha = self.model_.alpha_
            lambda_ = self.model_.lambda_
            
            X_processed = self.X_original_.copy()
            if self.scaler_ is not None:
                X_processed = self.scaler_.transform(X_processed)
            
            # Posterior covariance approximation
            try:
                if self.fit_intercept:
                    X_design = np.column_stack([np.ones(X_processed.shape[0]), X_processed])
                    coef_uncertainties = self._estimate_coefficient_uncertainties(X_design, alpha, lambda_)
                    if len(coef_uncertainties) > 1:
                        coef_uncertainties = coef_uncertainties[1:]  # Remove intercept
                else:
                    X_design = X_processed
                    coef_uncertainties = self._estimate_coefficient_uncertainties(X_design, alpha, lambda_)
            except:
                coef_uncertainties = np.ones(len(coef_mean)) * np.std(coef_mean)
            
            # Signal-to-noise ratio for each coefficient
            coef_snr = coef_abs / (coef_uncertainties + 1e-10)
            
            # Feature relevance score
            relevance_scores = coef_snr / np.max(coef_snr)
            
            # Rank features
            feature_ranking = np.argsort(relevance_scores)[::-1]
            
            # Identify significant features (high signal-to-noise ratio)
            significance_threshold = 2.0  # Approximately 95% confidence
            significant_features = coef_snr > significance_threshold
            
            self.feature_relevance_analysis_ = {
                'coefficient_means': coef_mean,
                'coefficient_uncertainties': coef_uncertainties,
                'coefficient_snr': coef_snr,
                'relevance_scores': relevance_scores,
                'feature_ranking': feature_ranking,
                'significant_features': significant_features,
                'significance_threshold': significance_threshold,
                'feature_names': self.feature_names_,
                'top_features': [
                    (self.feature_names_[i], coef_mean[i], coef_uncertainties[i], relevance_scores[i])
                    for i in feature_ranking[:10]
                ]
            }
            
        except Exception as e:
            self.feature_relevance_analysis_ = {
                'error': f'Could not analyze feature relevance: {str(e)}'
            }
    
    def _estimate_coefficient_uncertainties(self, X, alpha, lambda_):
        """Estimate uncertainties of coefficient estimates"""
        try:
            # Posterior covariance matrix
            S_inv = alpha * np.eye(X.shape[1]) + lambda_ * X.T @ X
            S = np.linalg.inv(S_inv)
            
            # Standard deviations are square roots of diagonal elements
            coef_uncertainties = np.sqrt(np.diag(S))
            return coef_uncertainties
            
        except:
            # Fallback: simple approximation
            return np.ones(X.shape[1]) * np.sqrt(1.0 / alpha)
    
    def _analyze_posterior_distribution(self):
        """Analyze posterior distribution characteristics"""
        try:
            # Get model parameters
            alpha = self.model_.alpha_
            lambda_ = self.model_.lambda_
            coef_mean = self.model_.coef_
            
            # Posterior statistics
            if self.fit_intercept:
                intercept = self.model_.intercept_
                n_params = len(coef_mean) + 1
            else:
                intercept = 0.0
                n_params = len(coef_mean)
            
            # Calculate posterior credible intervals for coefficients
            try:
                X_processed = self.X_original_.copy()
                if self.scaler_ is not None:
                    X_processed = self.scaler_.transform(X_processed)
                
                if self.fit_intercept:
                    X_design = np.column_stack([np.ones(X_processed.shape[0]), X_processed])
                else:
                    X_design = X_processed
                
                coef_uncertainties = self._estimate_coefficient_uncertainties(X_design, alpha, lambda_)
                
                if self.fit_intercept and len(coef_uncertainties) > 1:
                    intercept_uncertainty = coef_uncertainties[0]
                    coef_uncertainties = coef_uncertainties[1:]
                else:
                    intercept_uncertainty = 0.0
                    
            except:
                coef_uncertainties = np.ones(len(coef_mean)) * np.sqrt(1.0 / alpha)
                intercept_uncertainty = np.sqrt(1.0 / alpha)
            
            # 95% credible intervals
            z_score = stats.norm.ppf(0.975)
            coef_lower = coef_mean - z_score * coef_uncertainties
            coef_upper = coef_mean + z_score * coef_uncertainties
            
            if self.fit_intercept:
                intercept_lower = intercept - z_score * intercept_uncertainty
                intercept_upper = intercept + z_score * intercept_uncertainty
            else:
                intercept_lower = intercept_upper = intercept
            
            self.posterior_analysis_ = {
                'coefficient_posterior': {
                    'means': coef_mean,
                    'std_devs': coef_uncertainties,
                    'lower_95': coef_lower,
                    'upper_95': coef_upper,
                    'feature_names': self.feature_names_
                },
                'intercept_posterior': {
                    'mean': intercept,
                    'std_dev': intercept_uncertainty,
                    'lower_95': intercept_lower,
                    'upper_95': intercept_upper
                },
                'hyperparameter_posterior': {
                    'alpha': alpha,
                    'lambda': lambda_,
                    'noise_variance': 1.0 / lambda_,
                    'weight_precision': alpha
                },
                'model_statistics': {
                    'n_parameters': n_params,
                    'effective_parameters': self._calculate_effective_parameters(X_design, alpha, lambda_),
                    'posterior_log_likelihood': self._calculate_posterior_log_likelihood()
                }
            }
            
        except Exception as e:
            self.posterior_analysis_ = {
                'error': f'Could not analyze posterior distribution: {str(e)}'
            }
    
    def _calculate_posterior_log_likelihood(self):
        """Calculate posterior log likelihood"""
        try:
            # Make predictions on training data
            X_processed = self.X_original_.copy()
            if self.scaler_ is not None:
                X_processed = self.scaler_.transform(X_processed)
            
            y_pred = self.model_.predict(X_processed)
            residuals = self.y_original_ - y_pred
            
            # Log likelihood under Gaussian noise model
            lambda_ = self.model_.lambda_
            log_likelihood = -0.5 * lambda_ * np.sum(residuals**2)
            log_likelihood -= 0.5 * len(residuals) * np.log(2 * np.pi / lambda_)
            
            return log_likelihood
            
        except:
            return None
    
    def get_hyperparameter_config(self, key_prefix: str) -> Dict[str, Any]:
        """Generate Streamlit UI for hyperparameters"""
        
        # Create tabs for different configuration aspects
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "Bayesian Config", "Prior Specification", "Uncertainty Options", "Analysis Options", "Algorithm Info"
        ])
        
        with tab1:
            st.markdown("**Bayesian Ridge Regression Configuration**")
            
            col1, col2 = st.columns(2)
            
            with col1:
                n_iter = st.number_input(
                    "Maximum Iterations:",
                    value=self.n_iter,
                    min_value=10,
                    max_value=1000,
                    step=10,
                    help="Maximum number of iterations for Bayesian inference",
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
                    help="Shape parameter for weight precision prior Gamma(α₁, α₂)",
                    key=f"{key_prefix}_alpha_1"
                )
                
                alpha_2 = st.number_input(
                    "Alpha₂ (weight precision rate):",
                    value=self.alpha_2,
                    min_value=1e-10,
                    max_value=1e2,
                    step=1e-7,
                    format="%.1e",
                    help="Rate parameter for weight precision prior Gamma(α₁, α₂)",
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
                        "Initial Alpha (weight precision):",
                        value=self.alpha_init if self.alpha_init is not None else 1.0,
                        min_value=1e-10,
                        max_value=1e10,
                        step=0.1,
                        format="%.6f",
                        help="Initial value for weight precision",
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
                
                use_informative_priors = st.checkbox(
                    "Use Informative Priors",
                    value=self.use_informative_priors,
                    help="Set priors based on data characteristics",
                    key=f"{key_prefix}_use_informative_priors"
                )
        
        with tab3:
            st.markdown("**Uncertainty Quantification**")
            
            col1, col2 = st.columns(2)
            
            with col1:
                compute_prediction_intervals = st.checkbox(
                    "Compute Prediction Intervals",
                    value=self.compute_prediction_intervals,
                    help="Calculate uncertainty intervals for predictions",
                    key=f"{key_prefix}_compute_prediction_intervals"
                )
                
                confidence_level = st.slider(
                    "Confidence Level:",
                    min_value=0.80,
                    max_value=0.99,
                    value=self.confidence_level,
                    step=0.01,
                    help="Confidence level for prediction intervals",
                    key=f"{key_prefix}_confidence_level"
                )
                
                n_posterior_samples = st.number_input(
                    "Posterior Samples:",
                    value=self.n_posterior_samples,
                    min_value=100,
                    max_value=10000,
                    step=100,
                    help="Number of samples for posterior analysis",
                    key=f"{key_prefix}_n_posterior_samples"
                )
            
            with col2:
                uncertainty_decomposition = st.checkbox(
                    "Decompose Uncertainty",
                    value=self.uncertainty_decomposition,
                    help="Separate aleatoric and epistemic uncertainty",
                    key=f"{key_prefix}_uncertainty_decomposition"
                )
                
                st.markdown("**Uncertainty Types:**")
                st.info("""
                • **Aleatoric**: Irreducible noise in data
                • **Epistemic**: Model uncertainty (reducible with more data)
                • **Total**: Combined uncertainty
                """)
        
        with tab4:
            st.markdown("**Analysis Options**")
            
            col1, col2 = st.columns(2)
            
            with col1:
                convergence_analysis = st.checkbox(
                    "Convergence Analysis",
                    value=self.convergence_analysis,
                    help="Analyze convergence of Bayesian inference",
                    key=f"{key_prefix}_convergence_analysis"
                )
                
                evidence_analysis = st.checkbox(
                    "Evidence Analysis",
                    value=self.evidence_analysis,
                    help="Compute model evidence (marginal likelihood)",
                    key=f"{key_prefix}_evidence_analysis"
                )
                
                hyperparameter_evolution = st.checkbox(
                    "Hyperparameter Evolution",
                    value=self.hyperparameter_evolution,
                    help="Track hyperparameter changes during inference",
                    key=f"{key_prefix}_hyperparameter_evolution"
                )
            
            with col2:
                feature_relevance_analysis = st.checkbox(
                    "Feature Relevance Analysis",
                    value=self.feature_relevance_analysis,
                    help="Bayesian feature importance analysis",
                    key=f"{key_prefix}_feature_relevance_analysis"
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
            **Bayesian Ridge Regression** - Probabilistic Linear Regression:
            • 🎯 Automatic regularization via Bayesian inference
            • 📊 Uncertainty quantification with prediction intervals
            • 🔄 Conjugate priors for tractable inference
            • 📈 Evidence-based model selection
            • 🎲 Hierarchical Bayesian modeling
            • 🔍 Robust to overfitting
            
            **Mathematical Foundation:**
            • Likelihood: p(y|X,w,α,λ) = N(Xw, λ⁻¹I)
            • Prior: p(w|α) = N(0, α⁻¹I)
            • Hyperpriors: p(α) = Gamma(α₁,α₂), p(λ) = Gamma(λ₁,λ₂)
            • Posterior: p(w|X,y,α,λ) = N(μ, Σ)
            """)
            
            # When to use Bayesian Ridge
            if st.button("🎯 When to Use Bayesian Ridge", key=f"{key_prefix}_when_to_use"):
                st.markdown("""
                **Ideal Use Cases:**
                
                **Problem Characteristics:**
                • Need uncertainty quantification
                • Small to medium datasets
                • Linear or near-linear relationships
                • Want automatic regularization
                • Need probabilistic predictions
                
                **Data Characteristics:**
                • Continuous target variable
                • Potentially noisy measurements
                • Limited training data
                • Need to quantify prediction confidence
                
                **Examples:**
                • Scientific modeling with measurement uncertainty
                • Medical diagnosis with confidence intervals
                • Financial risk assessment
                • Engineering design with safety margins
                • A/B testing with statistical significance
                """)
            
            # Advantages and limitations
            if st.button("⚖️ Advantages & Limitations", key=f"{key_prefix}_pros_cons"):
                st.markdown("""
                **Advantages:**
                ✅ Automatic regularization (no hyperparameter tuning)
                ✅ Uncertainty quantification built-in
                ✅ Robust to overfitting
                ✅ Principled Bayesian approach
                ✅ No cross-validation needed for regularization
                ✅ Handles small datasets well
                ✅ Interpretable probabilistic predictions
                
                **Limitations:**
                ❌ Limited to linear relationships
                ❌ Assumes Gaussian noise and priors
                ❌ Computational overhead for large datasets
                ❌ May be overkill for simple problems
                ❌ Requires understanding of Bayesian concepts
                ❌ Less flexible than non-parametric methods
                """)
            
            # Bayesian concepts guide
            if st.button("📚 Bayesian Concepts Guide", key=f"{key_prefix}_bayesian_guide"):
                st.markdown("""
                **Key Bayesian Concepts:**
                
                **Prior Distributions:**
                • Express beliefs before seeing data
                • Gamma priors for precision parameters
                • Conjugate priors enable analytical solutions
                
                **Posterior Inference:**
                • Updates priors with data likelihood
                • Provides full probability distributions
                • Enables uncertainty quantification
                
                **Hyperparameter Learning:**
                • Automatic estimation via empirical Bayes
                • Balances model complexity and data fit
                • No manual hyperparameter tuning needed
                
                **Evidence (Marginal Likelihood):**
                • Model comparison criterion
                • Automatically penalizes complexity
                • Used for Bayesian model selection
                
                **Prediction Intervals:**
                • Account for both parameter and noise uncertainty
                • More honest than point predictions
                • Critical for decision making under uncertainty
                """)
            
            # Best practices
            if st.button("🎯 Best Practices", key=f"{key_prefix}_best_practices"):
                st.markdown("""
                **Bayesian Ridge Best Practices:**
                
                **Data Preparation:**
                1. **Normalize features** (highly recommended)
                2. Check for linear relationships
                3. Examine noise characteristics
                4. Consider outlier treatment
                
                **Prior Selection:**
                1. Use weakly informative priors (default is good)
                2. Consider data-driven priors for informative setting
                3. Check prior sensitivity if domain knowledge exists
                4. Monitor hyperparameter evolution
                
                **Model Validation:**
                1. Check convergence diagnostics
                2. Validate uncertainty calibration
                3. Compare evidence across models
                4. Examine residual patterns
                
                **Interpretation:**
                1. Report prediction intervals, not just point estimates
                2. Decompose uncertainty sources
                3. Identify features with credible effects
                4. Use posterior distributions for decisions
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
            "compute_prediction_intervals": compute_prediction_intervals,
            "confidence_level": confidence_level,
            "n_posterior_samples": n_posterior_samples,
            "convergence_analysis": convergence_analysis,
            "evidence_analysis": evidence_analysis,
            "hyperparameter_evolution": hyperparameter_evolution,
            "uncertainty_decomposition": uncertainty_decomposition,
            "feature_relevance_analysis": feature_relevance_analysis,
            "use_informative_priors": use_informative_priors,
            "random_state": random_state
        }
    
    def create_model_instance(self, hyperparameters: Dict[str, Any]):
        """Create model instance with given hyperparameters"""
        return BayesianRidgePlugin(
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
            compute_prediction_intervals=hyperparameters.get("compute_prediction_intervals", self.compute_prediction_intervals),
            confidence_level=hyperparameters.get("confidence_level", self.confidence_level),
            n_posterior_samples=hyperparameters.get("n_posterior_samples", self.n_posterior_samples),
            convergence_analysis=hyperparameters.get("convergence_analysis", self.convergence_analysis),
            evidence_analysis=hyperparameters.get("evidence_analysis", self.evidence_analysis),
            hyperparameter_evolution=hyperparameters.get("hyperparameter_evolution", self.hyperparameter_evolution),
            uncertainty_decomposition=hyperparameters.get("uncertainty_decomposition", self.uncertainty_decomposition),
            feature_relevance_analysis=hyperparameters.get("feature_relevance_analysis", self.feature_relevance_analysis),
            use_informative_priors=hyperparameters.get("use_informative_priors", self.use_informative_priors),
            random_state=hyperparameters.get("random_state", self.random_state)
        )
    
    def preprocess_data(self, X, y=None, training=True):
        """Preprocess data for Bayesian Ridge Regression"""
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
        """Check if Bayesian Ridge Regression is compatible with the given data"""
        # Check minimum samples
        if X.shape[0] < self._min_samples_required:
            return False, f"Bayesian Ridge requires at least {self._min_samples_required} samples, got {X.shape[0]}"
        
        # Check for regression targets
        if y is not None:
            if not np.issubdtype(y.dtype, np.number):
                return False, "Bayesian Ridge requires continuous numerical target values"
            
            # Check for sufficient variance in target
            if np.var(y) == 0:
                return False, "Target variable has zero variance (all values are the same)"
            
            n_samples, n_features = X.shape
            
            advantages = []
            considerations = []
            
            # Sample size assessment
            if n_samples >= n_features * 10:
                advantages.append(f"Excellent sample size ({n_samples}) for {n_features} features")
            elif n_samples >= n_features * 3:
                advantages.append(f"Good sample size ({n_samples}) for Bayesian inference")
            elif n_samples >= n_features:
                considerations.append(f"Adequate sample size ({n_samples}) but uncertainty may be high")
            else:
                considerations.append(f"Small sample size ({n_samples}) - Bayesian approach especially beneficial")
            
            # Dimensionality assessment
            if n_features > n_samples:
                advantages.append(f"High-dimensional data ({n_features} features) - Bayesian regularization helpful")
            elif n_features > n_samples * 0.1:
                considerations.append(f"Moderate dimensionality ({n_features}) - automatic regularization beneficial")
            else:
                advantages.append(f"Low dimensionality ({n_features}) - ideal for Bayesian Ridge")
            
            # Noise level assessment
            try:
                # Simple linear fit to estimate noise level
                from sklearn.linear_model import LinearRegression
                if n_samples > n_features:
                    lr = LinearRegression()
                    lr.fit(X, y)
                    y_pred = lr.predict(X)
                    residual_std = np.std(y - y_pred)
                    target_std = np.std(y)
                    noise_ratio = residual_std / target_std
                    
                    if noise_ratio > 0.3:
                        advantages.append(f"Noisy data (noise ratio: {noise_ratio:.2f}) - uncertainty quantification valuable")
                    elif noise_ratio > 0.1:
                        advantages.append(f"Moderate noise (ratio: {noise_ratio:.2f}) - Bayesian approach helpful")
                    else:
                        considerations.append(f"Low noise (ratio: {noise_ratio:.2f}) - simpler methods might suffice")
            except:
                pass
            
            # Feature correlation assessment
            try:
                if n_features > 1 and n_samples > n_features:
                    corr_matrix = np.corrcoef(X.T)
                    high_corr = np.any(np.abs(corr_matrix - np.eye(n_features)) > 0.8)
                    
                    if high_corr:
                        advantages.append("High feature correlation detected - Bayesian regularization beneficial")
                    else:
                        advantages.append("Features relatively independent - good for linear modeling")
            except:
                pass
            
            # Uncertainty quantification value
            advantages.append("Provides prediction uncertainty intervals")
            advantages.append("Automatic regularization (no hyperparameter tuning)")
            
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
                f"📊 Suitability for Bayesian Ridge: {suitability}"
            ]
            
            if advantages:
                message_parts.append("🎯 Advantages: " + "; ".join(advantages))
            
            if considerations:
                message_parts.append("💡 Considerations: " + "; ".join(considerations))
            
            return True, " | ".join(message_parts)
        
        return True, f"Compatible with {X.shape[0]} samples and {X.shape[1]} features"
    
    def get_feature_importance(self) -> Optional[Dict[str, Any]]:
        """Get Bayesian feature importance with uncertainty"""
        if not self.is_fitted_:
            return None
        
        if not self.feature_relevance_analysis_:
            return None
        
        analysis = self.feature_relevance_analysis_
        
        if 'error' in analysis:
            return {'error': analysis['error']}
        
        # Extract importance information
        coef_means = analysis['coefficient_means']
        coef_uncertainties = analysis['coefficient_uncertainties']
        relevance_scores = analysis['relevance_scores']
        feature_ranking = analysis['feature_ranking']
        
        # Create feature importance dictionary
        feature_importance = {}
        for i, (name, coef, uncertainty, relevance) in enumerate(zip(
            self.feature_names_, coef_means, coef_uncertainties, relevance_scores
        )):
            feature_importance[name] = {
                'coefficient_mean': coef,
                'coefficient_uncertainty': uncertainty,
                'relevance_score': relevance,
                'signal_to_noise_ratio': abs(coef) / (uncertainty + 1e-10),
                'significant': analysis['significant_features'][i],
                'rank': np.where(feature_ranking == i)[0][0] + 1 if i in feature_ranking else len(feature_ranking) + 1
            }
        
        # Get top features
        top_features = analysis['top_features']
        
        return {
            'feature_importance': feature_importance,
            'top_features': top_features,
            'significance_threshold': analysis['significance_threshold'],
            'bayesian_info': {
                'alpha': self.model_.alpha_,
                'lambda': self.model_.lambda_,
                'automatic_regularization': True,
                'uncertainty_quantified': True
            },
            'interpretation': 'Bayesian feature importance with uncertainty quantification'
        }
    
    def get_model_params(self) -> Dict[str, Any]:
        """Get current model parameters"""
        if not self.is_fitted_:
            return {}
        
        return {
            "algorithm": "Bayesian Ridge Regression",
            "n_features": self.n_features_in_,
            "coefficients": self.model_.coef_.tolist(),
            "intercept": self.model_.intercept_ if hasattr(self.model_, 'intercept_') else 0,
            "alpha": self.model_.alpha_,
            "lambda": self.model_.lambda_,
            "noise_variance": 1.0 / self.model_.lambda_,
            "weight_precision": self.model_.alpha_,
            "effective_regularization": self.model_.alpha_ / self.model_.lambda_,
            "n_iterations": self.n_iter,
            "convergence_tolerance": self.tol,
            "fit_intercept": self.fit_intercept,
            "normalize_features": self.normalize_features,
            "bayesian_characteristics": {
                "automatic_regularization": True,
                "uncertainty_quantification": True,
                "conjugate_priors": True,
                "evidence_based_selection": True
            }
        }
    
    def get_bayesian_analysis(self) -> Dict[str, Any]:
        """Get comprehensive Bayesian analysis results"""
        if not self.is_fitted_:
            return {"status": "Model not fitted"}
        
        return {
            "convergence_analysis": self.convergence_analysis_,
            "evidence_analysis": self.evidence_analysis_,
            "hyperparameter_evolution": self.hyperparameter_evolution_,
            "uncertainty_analysis": self.uncertainty_analysis_,
            "feature_relevance_analysis": self.feature_relevance_analysis_,
            "posterior_analysis": self.posterior_analysis_
        }
    
    def plot_bayesian_analysis(self, figsize=(16, 12)):
        """Plot comprehensive Bayesian analysis"""
        if not self.is_fitted_:
            raise ValueError("Model must be fitted to plot Bayesian analysis")
        
        fig, axes = plt.subplots(3, 3, figsize=figsize)
        axes = axes.ravel()
        
        # Plot 1: Convergence analysis
        ax1 = axes[0]
        if 'scores' in self.convergence_analysis_:
            scores = self.convergence_analysis_['scores']
            iterations = range(1, len(scores) + 1)
            
            ax1.plot(iterations, scores, 'b-', linewidth=2, marker='o', markersize=4)
            ax1.set_xlabel('Iteration')
            ax1.set_ylabel('Log Marginal Likelihood')
            ax1.set_title('Convergence Analysis')
            ax1.grid(True, alpha=0.3)
            
            # Mark convergence point
            if self.convergence_analysis_['converged']:
                ax1.axvline(x=len(scores), color='red', linestyle='--', alpha=0.7,
                           label=f'Converged at iteration {len(scores)}')
                ax1.legend()
        else:
            ax1.text(0.5, 0.5, 'Convergence data\nnot available', 
                    ha='center', va='center', transform=ax1.transAxes)
            ax1.set_title('Convergence Analysis')
        
        # Plot 2: Hyperparameter evolution
        ax2 = axes[1]
        if self.hyperparameter_evolution_:
            hparam_data = self.hyperparameter_evolution_
            
            # Create bar plot for initial vs final values
            categories = ['Alpha\n(Weight Precision)', 'Lambda\n(Noise Precision)']
            initial_values = [hparam_data.get('alpha_initial', 0), hparam_data.get('lambda_initial', 0)]
            final_values = [hparam_data['alpha_final'], hparam_data['lambda_final']]
            
            x = np.arange(len(categories))
            width = 0.35
            
            bars1 = ax2.bar(x - width/2, initial_values, width, label='Initial', alpha=0.7, color='lightblue')
            bars2 = ax2.bar(x + width/2, final_values, width, label='Final', alpha=0.7, color='darkblue')
            
            ax2.set_xlabel('Hyperparameters')
            ax2.set_ylabel('Value')
            ax2.set_title('Hyperparameter Evolution')
            ax2.set_xticks(x)
            ax2.set_xticklabels(categories)
            ax2.legend()
            ax2.set_yscale('log')
            
            # Add value labels
            for bar in bars1 + bars2:
                height = bar.get_height()
                if height > 0:
                    ax2.text(bar.get_x() + bar.get_width()/2., height * 1.1,
                            f'{height:.2e}', ha='center', va='bottom', fontsize=8)
        else:
            ax2.text(0.5, 0.5, 'Hyperparameter\nevolution not available', 
                    ha='center', va='center', transform=ax2.transAxes)
            ax2.set_title('Hyperparameter Evolution')
        
        # Plot 3: Uncertainty analysis
        ax3 = axes[2]
        if 'prediction_uncertainties' in self.uncertainty_analysis_:
            uncertainties = self.uncertainty_analysis_['prediction_uncertainties']
            
            ax3.hist(uncertainties, bins=30, alpha=0.7, color='green', edgecolor='black')
            ax3.axvline(x=np.mean(uncertainties), color='red', linestyle='--', 
                       label=f'Mean: {np.mean(uncertainties):.3f}')
            ax3.set_xlabel('Prediction Standard Deviation')
            ax3.set_ylabel('Frequency')
            ax3.set_title('Prediction Uncertainty Distribution')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
        else:
            ax3.text(0.5, 0.5, 'Uncertainty analysis\nnot available', 
                    ha='center', va='center', transform=ax3.transAxes)
            ax3.set_title('Uncertainty Distribution')
        
        # Plot 4: Feature relevance
        ax4 = axes[3]
        if 'top_features' in self.feature_relevance_analysis_:
            top_features = self.feature_relevance_analysis_['top_features'][:8]  # Top 8 features
            
            if top_features:
                feature_names = [f[0] for f in top_features]
                coefficients = [f[1] for f in top_features]
                uncertainties = [f[2] for f in top_features]
                
                # Truncate long feature names
                feature_names = [name[:15] + '...' if len(name) > 15 else name for name in feature_names]
                
                y_pos = np.arange(len(feature_names))
                colors = ['red' if c < 0 else 'blue' for c in coefficients]
                
                bars = ax4.barh(y_pos, coefficients, xerr=uncertainties, 
                              color=colors, alpha=0.7, capsize=3)
                ax4.set_yticks(y_pos)
                ax4.set_yticklabels(feature_names, fontsize=8)
                ax4.set_xlabel('Coefficient Value')
                ax4.set_title('Top Features with Uncertainty')
                ax4.grid(True, alpha=0.3)
        else:
            ax4.text(0.5, 0.5, 'Feature relevance\nanalysis not available', 
                    ha='center', va='center', transform=ax4.transAxes)
            ax4.set_title('Feature Relevance')
        
        # Plot 5: Evidence components
        ax5 = axes[4]
        if 'evidence_components' in self.evidence_analysis_:
            evidence = self.evidence_analysis_['evidence_components']
            
            components = ['Log Likelihood', 'Prior Term', 'Complexity Penalty']
            values = [
                evidence.get('log_likelihood', 0),
                evidence.get('prior_term', 0),
                evidence.get('complexity_penalty', 0)
            ]
            
            colors = ['lightcoral', 'lightblue', 'lightgreen']
            bars = ax5.bar(components, values, color=colors, alpha=0.7)
            ax5.set_ylabel('Value')
            ax5.set_title('Evidence Components')
            ax5.tick_params(axis='x', rotation=45)
            
            # Add value labels
            for bar, val in zip(bars, values):
                height = bar.get_height()
                ax5.text(bar.get_x() + bar.get_width()/2., height + abs(height) * 0.01,
                        f'{val:.2f}', ha='center', va='bottom', fontsize=9)
        else:
            ax5.text(0.5, 0.5, 'Evidence analysis\nnot available', 
                    ha='center', va='center', transform=ax5.transAxes)
            ax5.set_title('Evidence Components')
        
        # Plot 6: Posterior coefficient intervals
        ax6 = axes[5]
        if 'coefficient_posterior' in self.posterior_analysis_:
            coef_posterior = self.posterior_analysis_['coefficient_posterior']
            
            means = coef_posterior['means'][:10]  # Top 10 coefficients
            lower = coef_posterior['lower_95'][:10]
            upper = coef_posterior['upper_95'][:10]
            names = coef_posterior['feature_names'][:10]
            
            # Truncate names
            names = [name[:12] + '...' if len(name) > 12 else name for name in names]
            
            y_pos = np.arange(len(means))
            
            # Plot confidence intervals
            ax6.errorbar(means, y_pos, xerr=[means - lower, upper - means], 
                        fmt='o', capsize=5, capthick=2, alpha=0.8)
            
            # Color points by sign
            colors = ['red' if m < 0 else 'blue' for m in means]
            ax6.scatter(means, y_pos, c=colors, s=50, alpha=0.8, zorder=5)
            
            ax6.set_yticks(y_pos)
            ax6.set_yticklabels(names, fontsize=8)
            ax6.set_xlabel('Coefficient Value')
            ax6.set_title('Posterior Coefficient Intervals (95%)')
            ax6.axvline(x=0, color='black', linestyle='--', alpha=0.5)
            ax6.grid(True, alpha=0.3)
        else:
            ax6.text(0.5, 0.5, 'Posterior analysis\nnot available', 
                    ha='center', va='center', transform=ax6.transAxes)
            ax6.set_title('Posterior Intervals')
        
        # Plot 7: Uncertainty decomposition
        ax7 = axes[6]
        if ('uncertainty_decomposition' in self.uncertainty_analysis_ and 
            'aleatoric_uncertainty' in self.uncertainty_analysis_['uncertainty_decomposition']):
            
            decomp = self.uncertainty_analysis_['uncertainty_decomposition']
            
            try:
                aleatoric_pct = decomp['aleatoric_percentage']
                epistemic_pct = decomp['epistemic_percentage']
                
                labels = ['Aleatoric\n(Data Noise)', 'Epistemic\n(Model Uncertainty)']
                sizes = [aleatoric_pct, epistemic_pct]
                colors = ['lightcoral', 'lightblue']
                
                wedges, texts, autotexts = ax7.pie(sizes, labels=labels, colors=colors,
                                                  autopct='%1.1f%%', startangle=90)
                ax7.set_title('Uncertainty Decomposition')
            except:
                ax7.text(0.5, 0.5, 'Uncertainty\ndecomposition failed', 
                        ha='center', va='center', transform=ax7.transAxes)
                ax7.set_title('Uncertainty Decomposition')
        else:
            ax7.text(0.5, 0.5, 'Uncertainty\ndecomposition not available', 
                    ha='center', va='center', transform=ax7.transAxes)
            ax7.set_title('Uncertainty Decomposition')
        
        # Plot 8: Residuals vs predictions with uncertainty
        ax8 = axes[7]
        try:
            # Make predictions on training data
            X_processed = self.X_original_.copy()
            if self.scaler_ is not None:
                X_processed = self.scaler_.transform(X_processed)
            
            y_pred, y_std = self.model_.predict(X_processed, return_std=True)
            residuals = self.y_original_ - y_pred
            
            # Plot residuals with uncertainty bands
            ax8.scatter(y_pred, residuals, alpha=0.6, c=y_std, cmap='viridis', s=30)
            
            # Add zero line
            ax8.axhline(y=0, color='red', linestyle='--', alpha=0.8)
            
            # Add uncertainty bands (approximate)
            sorted_indices = np.argsort(y_pred)
            y_pred_sorted = y_pred[sorted_indices]
            y_std_sorted = y_std[sorted_indices]
            
            ax8.fill_between(y_pred_sorted, -2*y_std_sorted, 2*y_std_sorted, 
                           alpha=0.2, color='gray', label='±2σ bands')
            
            ax8.set_xlabel('Predicted Values')
            ax8.set_ylabel('Residuals')
            ax8.set_title('Residuals vs Predictions (colored by uncertainty)')
            ax8.legend()
            ax8.grid(True, alpha=0.3)
            
            # Add colorbar
            cbar = plt.colorbar(ax8.collections[0], ax=ax8, shrink=0.8)
            cbar.set_label('Prediction Std Dev', rotation=270, labelpad=15)
            
        except Exception as e:
            ax8.text(0.5, 0.5, f'Residuals plot\nfailed: {str(e)[:20]}...', 
                    ha='center', va='center', transform=ax8.transAxes, fontsize=8)
            ax8.set_title('Residuals vs Predictions')
        
        # Plot 9: Model comparison metrics
        ax9 = axes[8]
        if self.evidence_analysis_:
            metrics = {
                'Log Evidence': self.evidence_analysis_.get('log_marginal_likelihood', 0),
                'AIC': -self.evidence_analysis_.get('aic', 0),  # Negative for "higher is better"
                'BIC': -self.evidence_analysis_.get('bic', 0),
                'Effective\nParameters': self.evidence_analysis_.get('effective_parameters', 0)
            }
            
            # Normalize metrics for visualization (except effective parameters)
            normalized_metrics = {}
            for key, value in metrics.items():
                if 'Parameters' in key:
                    normalized_metrics[key] = value
                else:
                    normalized_metrics[key] = value / abs(value) if value != 0 else 0
            
            labels = list(normalized_metrics.keys())
            values = list(normalized_metrics.values())
            
            colors = ['lightgreen', 'lightblue', 'lightcoral', 'lightyellow']
            bars = ax9.bar(labels, values, color=colors, alpha=0.7)
            ax9.set_ylabel('Normalized Value')
            ax9.set_title('Model Quality Metrics')
            ax9.tick_params(axis='x', rotation=45)
            
            # Add value labels
            for bar, val in zip(bars, values):
                height = bar.get_height()
                ax9.text(bar.get_x() + bar.get_width()/2., height + abs(height) * 0.01,
                        f'{val:.2f}', ha='center', va='bottom', fontsize=9)
        else:
            ax9.text(0.5, 0.5, 'Model metrics\nnot available', 
                    ha='center', va='center', transform=ax9.transAxes)
            ax9.set_title('Model Quality Metrics')
        
        plt.tight_layout()
        return fig
    
    def plot_prediction_intervals(self, X_test=None, y_test=None, n_points=100, figsize=(12, 8)):
        """Plot predictions with uncertainty intervals"""
        if not self.is_fitted_:
            raise ValueError("Model must be fitted to plot prediction intervals")
        
        # Use test data if provided, otherwise sample from training data range
        if X_test is None:
            if self.X_original_.shape[1] > 1:
                raise ValueError("For multi-dimensional data, please provide X_test")
            
            X_min, X_max = self.X_original_[:, 0].min(), self.X_original_[:, 0].max()
            X_range = X_max - X_min
            X_test = np.linspace(X_min - 0.1*X_range, X_max + 0.1*X_range, n_points).reshape(-1, 1)
        
        # Get predictions with uncertainty
        results = self.predict_with_uncertainty(X_test)
        
        y_pred = results['predictions']
        y_lower = results['lower_bound']
        y_upper = results['upper_bound']
        confidence_level = results['confidence_level']
        
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot for 1D case
        if X_test.shape[1] == 1:
            X_plot = X_test[:, 0]
            
            # Sort for smooth plotting
            sort_indices = np.argsort(X_plot)
            X_plot_sorted = X_plot[sort_indices]
            y_pred_sorted = y_pred[sort_indices]
            y_lower_sorted = y_lower[sort_indices]
            y_upper_sorted = y_upper[sort_indices]
            
            # Plot prediction intervals
            ax.fill_between(X_plot_sorted, y_lower_sorted, y_upper_sorted, 
                           alpha=0.3, color='lightblue', 
                           label=f'{confidence_level*100:.0f}% Prediction Interval')
            
            # Plot mean prediction
            ax.plot(X_plot_sorted, y_pred_sorted, 'b-', linewidth=2, label='Mean Prediction')
            
            # Plot training data
            ax.scatter(self.X_original_[:, 0], self.y_original_, 
                      alpha=0.6, color='red', s=30, label='Training Data')
            
            # Plot test data if provided
            if y_test is not None:
                ax.scatter(X_plot, y_test, alpha=0.8, color='green', s=30, 
                          marker='s', label='Test Data')
            
            ax.set_xlabel('Feature Value')
            ax.set_ylabel('Target Value')
            ax.set_title(f'Bayesian Ridge Predictions with {confidence_level*100:.0f}% Intervals')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
        else:
            # For multi-dimensional case, plot prediction vs actual
            ax.scatter(y_pred, y_pred, alpha=0.1, s=1, color='gray', label='Perfect Prediction')
            
            # Plot error bars for uncertainty
            if y_test is not None:
                ax.errorbar(y_test, y_pred, yerr=[y_pred - y_lower, y_upper - y_pred], 
                           fmt='o', alpha=0.6, capsize=3, capthick=1, label='Predictions with Uncertainty')
                ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 
                       'r--', alpha=0.8, label='Perfect Prediction')
            else:
                # Just show predictions with uncertainty
                ax.errorbar(range(len(y_pred)), y_pred, 
                           yerr=[y_pred - y_lower, y_upper - y_pred], 
                           fmt='o', alpha=0.6, capsize=3, capthick=1)
            
            ax.set_xlabel('Actual Values' if y_test is not None else 'Sample Index')
            ax.set_ylabel('Predicted Values')
            ax.set_title(f'Predictions with {confidence_level*100:.0f}% Uncertainty Intervals')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        return fig
    
    def get_training_info(self) -> Dict[str, Any]:
        """Get information about the training process"""
        if not self.is_fitted_:
            return {"status": "Not fitted"}
        
        info = {
            "algorithm": "Bayesian Ridge Regression",
            "type": "Probabilistic linear regression with automatic regularization",
            "training_completed": True,
            "bayesian_characteristics": {
                "automatic_regularization": True,
                "uncertainty_quantification": True,
                "conjugate_priors": True,
                "evidence_computation": self.compute_score,
                "hierarchical_modeling": True
            },
            "model_configuration": {
                "n_iterations": self.n_iter,
                "convergence_tolerance": self.tol,
                "fit_intercept": self.fit_intercept,
                "normalize_features": self.normalize_features,
                "compute_score": self.compute_score
            },
            "learned_hyperparameters": {
                "alpha": self.model_.alpha_,
                "lambda": self.model_.lambda_,
                "noise_variance": 1.0 / self.model_.lambda_,
                "weight_precision": self.model_.alpha_,
                "effective_regularization": self.model_.alpha_ / self.model_.lambda_
            },
            "convergence_info": {
                "converged": self.convergence_analysis_.get('converged', 'Unknown'),
                "n_iterations_used": self.convergence_analysis_.get('n_iterations', 'Unknown'),
                "final_score": self.convergence_analysis_.get('final_score', 'Unknown')
            }
        }
        
        # Add evidence information if available
        if self.evidence_analysis_:
            info["evidence_analysis"] = {
                "log_marginal_likelihood": self.evidence_analysis_.get('log_marginal_likelihood'),
                "aic": self.evidence_analysis_.get('aic'),
                "bic": self.evidence_analysis_.get('bic'),
                "effective_parameters": self.evidence_analysis_.get('effective_parameters')
            }
        
        return info

    def get_algorithm_specific_metrics(self,
                                        y_true: np.ndarray,
                                        y_pred: np.ndarray,
                                        y_proba: Optional[np.ndarray] = None
                                        ) -> Dict[str, Any]:
        """
        Calculate Bayesian Ridge Regression-specific metrics.

        This includes learned hyperparameters (alpha, lambda), model evidence metrics
        (log marginal likelihood, AIC, BIC), effective number of parameters,
        and if prediction standard deviations (y_proba) are provided for the test set,
        metrics related to prediction uncertainty.

        Args:
            y_true: Ground truth target values from the test set.
            y_pred: Predicted target values on the test set.
            y_proba: Optional. For regression, this can be interpreted as the
                        standard deviations of the predictions (y_pred_std) on the test set.

        Returns:
            A dictionary of Bayesian Ridge Regression-specific metrics.
        """
        metrics = {}
        if not self.is_fitted_ or self.model_ is None:
            metrics["status"] = "Model not fitted or not available"
            return metrics

        # --- Metrics from the fitted scikit-learn BayesianRidge model ---
        metrics['learned_alpha_precision'] = float(self.model_.alpha_)
        metrics['learned_lambda_precision'] = float(self.model_.lambda_)
        metrics['noise_variance_estimate'] = 1.0 / float(self.model_.lambda_) if self.model_.lambda_ > 0 else np.inf
        metrics['weight_variance_estimate'] = 1.0 / float(self.model_.alpha_) if self.model_.alpha_ > 0 else np.inf
        metrics['iterations_to_converge'] = int(self.model_.n_iter_)

        if hasattr(self.model_, 'scores_') and self.model_.scores_ is not None and len(self.model_.scores_) > 0:
            metrics['log_marginal_likelihood_final_iter'] = float(self.model_.scores_[-1])
        else:
            metrics['log_marginal_likelihood_final_iter'] = None

        # --- Metrics from internal Bayesian analysis dictionaries ---
        if hasattr(self, 'convergence_analysis_') and self.convergence_analysis_:
            metrics['converged_status'] = self.convergence_analysis_.get('converged')

        if hasattr(self, 'evidence_analysis_') and self.evidence_analysis_:
            if 'error' not in self.evidence_analysis_:
                metrics['evidence_log_marginal_likelihood'] = self.evidence_analysis_.get('log_marginal_likelihood')
                metrics['evidence_aic'] = self.evidence_analysis_.get('aic')
                metrics['evidence_bic'] = self.evidence_analysis_.get('bic')
                metrics['evidence_effective_parameters'] = self.evidence_analysis_.get('effective_parameters')
            else:
                metrics['evidence_analysis_error'] = self.evidence_analysis_.get('error')

        if hasattr(self, 'hyperparameter_evolution_') and self.hyperparameter_evolution_:
            if 'error' not in self.hyperparameter_evolution_:
                metrics['hyperparam_snr_final'] = self.hyperparameter_evolution_.get('signal_to_noise_ratio')
                metrics['hyperparam_effective_regularization_final'] = self.hyperparameter_evolution_.get('effective_regularization')

        # --- Test-set specific uncertainty metrics (if y_proba is provided as y_pred_std) ---
        if y_proba is not None:
            y_pred_std_test = np.asarray(y_proba)
            if y_pred_std_test.ndim == 1 and len(y_pred_std_test) == len(y_true):
                metrics['mean_test_prediction_std_dev'] = float(np.mean(y_pred_std_test))
                metrics['median_test_prediction_std_dev'] = float(np.median(y_pred_std_test))
                metrics['std_dev_of_test_prediction_std_devs'] = float(np.std(y_pred_std_test))

                # Calculate empirical coverage for 95% confidence interval
                z_score_95 = 1.96
                lower_bound = y_pred - z_score_95 * y_pred_std_test
                upper_bound = y_pred + z_score_95 * y_pred_std_test
                covered = np.sum((y_true >= lower_bound) & (y_true <= upper_bound))
                metrics['empirical_coverage_95_test'] = float(covered / len(y_true))
            else:
                metrics['y_proba_format_warning'] = "y_proba (for y_pred_std) was not in expected 1D array format matching y_true."
        else:
            metrics['test_prediction_std_dev_status'] = "y_pred_std (via y_proba) not provided for test set."
            
        return metrics
                    
# Factory function
def get_plugin():
    """Factory function to get the plugin instance"""
    return BayesianRidgePlugin()


# Example usage and testing
if __name__ == "__main__":
    """
    Example usage of Bayesian Ridge Regression Plugin
    """
    print("Testing Bayesian Ridge Regression Plugin...")
    
    try:
        # Create sample data with noise
        np.random.seed(42)
        
        # Generate synthetic regression data
        from sklearn.datasets import make_regression
        X, y = make_regression(
            n_samples=100,
            n_features=5,
            noise=0.1,
            random_state=42
        )
        
        # Add some correlated features to test regularization
        X_corr = np.column_stack([X, X[:, 0] + np.random.normal(0, 0.1, X.shape[0])])
        
        print(f"\n📊 Test Dataset:")
        print(f"Samples: {X_corr.shape[0]}, Features: {X_corr.shape[1]}")
        print(f"Target variance: {np.var(y):.3f}")
        
        # Test Bayesian Ridge regression
        print(f"\n🧪 Testing Bayesian Ridge Regression...")
        
        # Create DataFrame for proper feature names
        feature_names = [f'feature_{i}' for i in range(X_corr.shape[1])]
        X_df = pd.DataFrame(X_corr, columns=feature_names)
        
        plugin = BayesianRidgePlugin(
            n_iter=300,
            tol=1e-3,
            compute_score=True,
            normalize_features=True,
            compute_prediction_intervals=True,
            confidence_level=0.95,
            convergence_analysis=True,
            evidence_analysis=True,
            hyperparameter_evolution=True,
            uncertainty_decomposition=True,
            feature_relevance_analysis=True,
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
            
            print(f"\n📊 Bayesian Ridge Results:")
            print(f"R²: {r2:.4f}")
            print(f"MSE: {mse:.4f}")
            print(f"Mean prediction uncertainty: {np.mean(uncertainty_results['std_dev']):.4f}")
            
            # Get model parameters
            model_params = plugin.get_model_params()
            print(f"\nLearned hyperparameters:")
            print(f"Alpha (weight precision): {model_params['alpha']:.6f}")
            print(f"Lambda (noise precision): {model_params['lambda']:.6f}")
            print(f"Noise variance: {model_params['noise_variance']:.6f}")
            
            # Test feature importance
            importance = plugin.get_feature_importance()
            if importance and 'top_features' in importance:
                print(f"\n🎯 Top Features with Uncertainty:")
                for i, (name, coef, uncertainty, relevance) in enumerate(importance['top_features'][:3]):
                    print(f"  {i+1}. {name}: {coef:.4f} ± {uncertainty:.4f} (relevance: {relevance:.3f})")
            
            # Get Bayesian analysis
            analysis = plugin.get_bayesian_analysis()
            
            # Check convergence
            if 'convergence_analysis' in analysis:
                conv = analysis['convergence_analysis']
                if 'converged' in conv:
                    print(f"\nConvergence: {'✅ Converged' if conv['converged'] else '❌ Not converged'}")
                    print(f"Iterations used: {conv.get('n_iterations', 'Unknown')}")
            
            # Check evidence
            if 'evidence_analysis' in analysis:
                ev = analysis['evidence_analysis']
                if 'log_marginal_likelihood' in ev:
                    print(f"Log marginal likelihood: {ev['log_marginal_likelihood']:.2f}")
                    print(f"Effective parameters: {ev.get('effective_parameters', 'Unknown'):.1f}")
            
            # Test uncertainty decomposition
            if 'uncertainty_analysis' in analysis:
                unc = analysis['uncertainty_analysis']
                if 'uncertainty_decomposition' in unc:
                    decomp = unc['uncertainty_decomposition']
                    if 'aleatoric_percentage' in decomp:
                        print(f"\nUncertainty decomposition:")
                        print(f"Aleatoric (data noise): {decomp['aleatoric_percentage']:.1f}%")
                        print(f"Epistemic (model uncertainty): {decomp['epistemic_percentage']:.1f}%")
        
        print("\n✅ Bayesian Ridge Regression Plugin test completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Error testing Bayesian Ridge Plugin: {str(e)}")
        import traceback
        traceback.print_exc()