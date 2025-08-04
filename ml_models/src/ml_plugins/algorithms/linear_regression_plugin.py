import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, Optional, Tuple
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_X_y, check_array
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression as SklearnLinearRegression
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

# Try to import statsmodels for VIF calculation - optional dependency
try:
    from statsmodels.stats.outliers_influence import variance_inflation_factor
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False
    variance_inflation_factor = None


class LinearRegressionPlugin(BaseEstimator, RegressorMixin, MLPlugin):
    """
    Linear Regression Plugin - The Foundation of Regression Analysis
    
    This plugin implements comprehensive linear regression with advanced statistical analysis,
    assumption checking, and interpretability features. Perfect as a baseline model and
    for understanding linear relationships in data.
    
    Key Features:
    - OLS (Ordinary Least Squares) regression
    - Statistical inference (p-values, confidence intervals)
    - Assumption validation (linearity, normality, homoscedasticity)
    - Feature importance and coefficient interpretation
    - Polynomial feature expansion
    - Comprehensive residual analysis
    - Outlier detection and influence analysis
    - Cross-validation performance estimation
    """
    
    def __init__(
        self,
        fit_intercept=True,
        normalize_features=True,
        polynomial_degree=1,
        include_bias=True,
        alpha_level=0.05,
        copy_X=True,
        positive=False,
        estimate_statistical_properties=True,
        detect_outliers=True,
        validate_assumptions=True,
        random_state=42
    ):
        super().__init__()
        
        # Core regression parameters
        self.fit_intercept = fit_intercept
        self.normalize_features = normalize_features
        self.polynomial_degree = polynomial_degree
        self.include_bias = include_bias
        self.alpha_level = alpha_level
        self.copy_X = copy_X
        self.positive = positive
        
        # Analysis parameters
        self.estimate_statistical_properties = estimate_statistical_properties
        self.detect_outliers = detect_outliers
        self.validate_assumptions = validate_assumptions
        self.random_state = random_state
        
        # Required plugin metadata
        self._name = "Linear Regression"
        self._description = "Ordinary Least Squares regression with comprehensive statistical analysis and assumption validation"
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
        
        # Statistical properties
        self.statistical_properties_ = {}
        self.assumption_tests_ = {}
        self.outlier_analysis_ = {}
        self.residual_analysis_ = {}
    
    def get_name(self) -> str:
        return self._name
    
    def get_description(self) -> str:
        return self._description
    
    def get_category(self) -> str:
        return self._category
    
    def fit(self, X, y, sample_weight=None):
        """
        Fit the Linear Regression model with comprehensive analysis
        
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
        # 🎯 STORE FEATURE NAMES BEFORE VALIDATION!
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
                # Fallback for older sklearn versions
                self.feature_names_processed_ = [f"poly_feature_{i}" for i in range(X_processed.shape[1])]
        else:
            self.feature_names_processed_ = self.feature_names_.copy()
        
        # Apply feature scaling if requested
        if self.normalize_features:
            self.scaler_ = StandardScaler()
            X_processed = self.scaler_.fit_transform(X_processed)
        
        # Fit the linear regression model
        self.model_ = SklearnLinearRegression(
            fit_intercept=self.fit_intercept,
            copy_X=self.copy_X,
            positive=self.positive
        )
        
        self.model_.fit(X_processed, y, sample_weight=sample_weight)
        
        # Store processed training data for analysis
        self.X_processed_ = X_processed
        
        # Perform comprehensive analysis
        self._perform_statistical_analysis()
        self._validate_regression_assumptions()
        self._analyze_outliers_and_influence()
        self._analyze_residuals()
        
        self.is_fitted_ = True
        return self
    
    def predict(self, X):
        """Make predictions using the fitted model"""
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
    
    def _perform_statistical_analysis(self):
        """Perform comprehensive statistical analysis of the regression"""
        if not self.estimate_statistical_properties:
            return
        
        # Get predictions and residuals
        y_pred = self.model_.predict(self.X_processed_)
        residuals = self.y_original_ - y_pred
        
        # Basic statistics
        n_samples, n_features = self.X_processed_.shape
        
        # Degrees of freedom
        df_model = n_features
        df_residual = n_samples - n_features - (1 if self.fit_intercept else 0)
        
        # Sum of squares
        ss_total = np.sum((self.y_original_ - np.mean(self.y_original_)) ** 2)
        ss_residual = np.sum(residuals ** 2)
        ss_model = ss_total - ss_residual
        
        # Mean squared errors
        mse_model = ss_model / df_model if df_model > 0 else 0
        mse_residual = ss_residual / df_residual if df_residual > 0 else 0
        
        # R-squared and adjusted R-squared
        r2 = r2_score(self.y_original_, y_pred)
        adj_r2 = 1 - (1 - r2) * (n_samples - 1) / df_residual if df_residual > 0 else r2
        
        # F-statistic and p-value
        f_statistic = mse_model / mse_residual if mse_residual > 0 else np.inf
        f_p_value = 1 - stats.f.cdf(f_statistic, df_model, df_residual) if df_residual > 0 else 0
        
        # Standard errors of coefficients
        try:
            # Calculate covariance matrix
            X_design = self.X_processed_
            if self.fit_intercept:
                X_design = np.column_stack([np.ones(len(X_design)), X_design])
            
            # Variance-covariance matrix
            if df_residual > 0:
                var_covar_matrix = mse_residual * np.linalg.inv(X_design.T @ X_design)
                coef_std_errors = np.sqrt(np.diag(var_covar_matrix))
                
                # Separate intercept and coefficient standard errors
                if self.fit_intercept:
                    intercept_se = coef_std_errors[0]
                    coef_se = coef_std_errors[1:]
                else:
                    intercept_se = 0
                    coef_se = coef_std_errors
                
                # T-statistics and p-values for coefficients
                if self.fit_intercept:
                    intercept_t = self.model_.intercept_ / intercept_se if intercept_se > 0 else np.inf
                    intercept_p = 2 * (1 - stats.t.cdf(np.abs(intercept_t), df_residual))
                else:
                    intercept_t = np.nan
                    intercept_p = np.nan
                
                coef_t = self.model_.coef_ / coef_se
                coef_p = 2 * (1 - stats.t.cdf(np.abs(coef_t), df_residual))
                
                # Confidence intervals
                t_critical = stats.t.ppf(1 - self.alpha_level/2, df_residual)
                
                if self.fit_intercept:
                    intercept_ci = [
                        self.model_.intercept_ - t_critical * intercept_se,
                        self.model_.intercept_ + t_critical * intercept_se
                    ]
                else:
                    intercept_ci = [np.nan, np.nan]
                
                coef_ci_lower = self.model_.coef_ - t_critical * coef_se
                coef_ci_upper = self.model_.coef_ + t_critical * coef_se
                
            else:
                coef_se = np.full(len(self.model_.coef_), np.nan)
                intercept_se = np.nan
                coef_t = np.full(len(self.model_.coef_), np.nan)
                intercept_t = np.nan
                coef_p = np.full(len(self.model_.coef_), np.nan)
                intercept_p = np.nan
                intercept_ci = [np.nan, np.nan]
                coef_ci_lower = np.full(len(self.model_.coef_), np.nan)
                coef_ci_upper = np.full(len(self.model_.coef_), np.nan)
        
        except np.linalg.LinAlgError:
            # Handle singular matrix (perfect multicollinearity)
            coef_se = np.full(len(self.model_.coef_), np.nan)
            intercept_se = np.nan
            coef_t = np.full(len(self.model_.coef_), np.nan)
            intercept_t = np.nan
            coef_p = np.full(len(self.model_.coef_), np.nan)
            intercept_p = np.nan
            intercept_ci = [np.nan, np.nan]
            coef_ci_lower = np.full(len(self.model_.coef_), np.nan)
            coef_ci_upper = np.full(len(self.model_.coef_), np.nan)
        
        # Store statistical properties
        self.statistical_properties_ = {
            'model_summary': {
                'n_samples': n_samples,
                'n_features': len(self.model_.coef_),
                'df_model': df_model,
                'df_residual': df_residual,
                'r_squared': r2,
                'adj_r_squared': adj_r2,
                'f_statistic': f_statistic,
                'f_p_value': f_p_value,
                'aic': n_samples * np.log(ss_residual/n_samples) + 2 * (n_features + 1),
                'bic': n_samples * np.log(ss_residual/n_samples) + np.log(n_samples) * (n_features + 1)
            },
            'sum_of_squares': {
                'total': ss_total,
                'model': ss_model,
                'residual': ss_residual,
                'mse_model': mse_model,
                'mse_residual': mse_residual
            },
            'coefficients': {
                'intercept': self.model_.intercept_ if self.fit_intercept else 0,
                'coefficients': self.model_.coef_,
                'feature_names': self.feature_names_processed_,
                'intercept_se': intercept_se,
                'coef_se': coef_se,
                'intercept_t': intercept_t,
                'coef_t': coef_t,
                'intercept_p': intercept_p,
                'coef_p': coef_p,
                'intercept_ci': intercept_ci,
                'coef_ci_lower': coef_ci_lower,
                'coef_ci_upper': coef_ci_upper,
                'confidence_level': 1 - self.alpha_level
            }
        }
    
    def _validate_regression_assumptions(self):
        """Validate key assumptions of linear regression"""
        if not self.validate_assumptions:
            return
        
        y_pred = self.model_.predict(self.X_processed_)
        residuals = self.y_original_ - y_pred
        standardized_residuals = residuals / np.std(residuals)
        
        # 1. Linearity (already assumed in linear regression)
        linearity_test = {
            'description': 'Linear relationship assumed in model specification',
            'check': 'Visual inspection of residuals vs fitted values recommended'
        }
        
        # 2. Independence (cannot be tested without additional information)
        independence_test = {
            'description': 'Independence of observations cannot be statistically tested',
            'check': 'Ensure proper experimental design and no temporal correlation'
        }
        
        # 3. Homoscedasticity (constant variance) - Breusch-Pagan test
        try:
            from scipy.stats import pearsonr
            
            # Simple test: correlation between |residuals| and fitted values
            abs_residuals = np.abs(residuals)
            correlation, p_value = pearsonr(abs_residuals, y_pred)
            
            homoscedasticity_test = {
                'test_name': 'Breusch-Pagan (simplified)',
                'correlation': correlation,
                'p_value': p_value,
                'null_hypothesis': 'Homoscedasticity (constant variance)',
                'interpretation': 'Fail to reject null' if p_value > self.alpha_level else 'Reject null (heteroscedasticity detected)',
                'assumptions_met': p_value > self.alpha_level
            }
        except Exception:
            homoscedasticity_test = {
                'test_name': 'Breusch-Pagan (simplified)',
                'error': 'Could not perform test',
                'assumptions_met': None
            }
        
        # 4. Normality of residuals - Shapiro-Wilk test
        try:
            if len(residuals) <= 5000:  # Shapiro-Wilk has sample size limitations
                shapiro_stat, shapiro_p = stats.shapiro(residuals)
                normality_test = {
                    'test_name': 'Shapiro-Wilk',
                    'statistic': shapiro_stat,
                    'p_value': shapiro_p,
                    'null_hypothesis': 'Residuals are normally distributed',
                    'interpretation': 'Fail to reject null' if shapiro_p > self.alpha_level else 'Reject null (non-normal residuals)',
                    'assumptions_met': shapiro_p > self.alpha_level
                }
            else:
                # Use Kolmogorov-Smirnov test for large samples
                ks_stat, ks_p = stats.kstest(standardized_residuals, 'norm')
                normality_test = {
                    'test_name': 'Kolmogorov-Smirnov',
                    'statistic': ks_stat,
                    'p_value': ks_p,
                    'null_hypothesis': 'Residuals are normally distributed',
                    'interpretation': 'Fail to reject null' if ks_p > self.alpha_level else 'Reject null (non-normal residuals)',
                    'assumptions_met': ks_p > self.alpha_level
                }
        except Exception:
            normality_test = {
                'test_name': 'Normality test',
                'error': 'Could not perform test',
                'assumptions_met': None
            }
        
        # 5. No multicollinearity - Variance Inflation Factor (VIF)
        try:
            if self.X_processed_.shape[1] > 1:
                # Use statsmodels if available
                if STATSMODELS_AVAILABLE and variance_inflation_factor is not None:
                    vif_values = []
                    for i in range(self.X_processed_.shape[1]):
                        vif = variance_inflation_factor(self.X_processed_, i)
                        vif_values.append(vif)
                    
                    max_vif = max(vif_values)
                    multicollinearity_test = {
                        'test_name': 'Variance Inflation Factor (VIF)',
                        'vif_values': vif_values,
                        'feature_names': self.feature_names_processed_,
                        'max_vif': max_vif,
                        'interpretation': 'Low multicollinearity' if max_vif < 5 else 'Moderate multicollinearity' if max_vif < 10 else 'High multicollinearity',
                        'assumptions_met': max_vif < 10
                    }
                else:
                    # Fallback: use correlation-based multicollinearity detection
                    correlation_matrix = np.corrcoef(self.X_processed_.T)
                    max_correlation = 0
                    
                    # Find maximum absolute correlation between features
                    for i in range(len(correlation_matrix)):
                        for j in range(i+1, len(correlation_matrix)):
                            abs_corr = abs(correlation_matrix[i, j])
                            if abs_corr > max_correlation:
                                max_correlation = abs_corr
                    
                    # Approximate VIF interpretation from correlation
                    # High correlation (>0.9) suggests high multicollinearity
                    assumptions_met = max_correlation < 0.8
                    interpretation = ('Low multicollinearity' if max_correlation < 0.5 else 
                                    'Moderate multicollinearity' if max_correlation < 0.8 else 
                                    'High multicollinearity')
                    
                    multicollinearity_test = {
                        'test_name': 'Correlation-based Multicollinearity Check',
                        'max_correlation': max_correlation,
                        'feature_names': self.feature_names_processed_,
                        'interpretation': f'{interpretation} (max correlation: {max_correlation:.3f})',
                        'assumptions_met': assumptions_met,
                        'note': 'Using correlation as VIF approximation (install statsmodels for precise VIF)'
                    }
            else:
                multicollinearity_test = {
                    'test_name': 'Variance Inflation Factor (VIF)',
                    'interpretation': 'Single feature - no multicollinearity',
                    'assumptions_met': True
                }
        except Exception:
            multicollinearity_test = {
                'test_name': 'Multicollinearity Check',
                'error': 'Could not perform multicollinearity analysis',
                'assumptions_met': None
            }
        
        self.assumption_tests_ = {
            'linearity': linearity_test,
            'independence': independence_test,
            'homoscedasticity': homoscedasticity_test,
            'normality': normality_test,
            'multicollinearity': multicollinearity_test,
            'overall_assessment': self._assess_overall_assumptions()
        }
    
    def _assess_overall_assumptions(self):
        """Assess overall assumption compliance"""
        testable_assumptions = ['homoscedasticity', 'normality', 'multicollinearity']
        
        met_count = 0
        total_count = 0
        
        for assumption in testable_assumptions:
            if assumption in self.assumption_tests_:
                test_result = self.assumption_tests_[assumption]
                if 'assumptions_met' in test_result and test_result['assumptions_met'] is not None:
                    total_count += 1
                    if test_result['assumptions_met']:
                        met_count += 1
        
        if total_count == 0:
            return {
                'score': None,
                'interpretation': 'Could not assess assumptions',
                'recommendation': 'Manually check assumption plots'
            }
        
        score = met_count / total_count
        
        if score >= 0.8:
            interpretation = 'Excellent - Most assumptions satisfied'
            recommendation = 'Linear regression is appropriate for this data'
        elif score >= 0.6:
            interpretation = 'Good - Most assumptions satisfied'
            recommendation = 'Linear regression should work well, monitor residuals'
        elif score >= 0.4:
            interpretation = 'Fair - Some assumption violations detected'
            recommendation = 'Consider data transformation or alternative models'
        else:
            interpretation = 'Poor - Multiple assumption violations'
            recommendation = 'Linear regression may not be appropriate, try non-linear models'
        
        return {
            'score': score,
            'met_assumptions': met_count,
            'total_testable': total_count,
            'interpretation': interpretation,
            'recommendation': recommendation
        }
    
    def _analyze_outliers_and_influence(self):
        """Analyze outliers and influential observations"""
        if not self.detect_outliers:
            return
        
        y_pred = self.model_.predict(self.X_processed_)
        residuals = self.y_original_ - y_pred
        
        # Standardized residuals
        residual_std = np.std(residuals)
        standardized_residuals = residuals / residual_std if residual_std > 0 else residuals
        
        # Outlier detection based on standardized residuals
        outlier_threshold = 2.5
        outlier_indices = np.where(np.abs(standardized_residuals) > outlier_threshold)[0]
        
        # Cook's distance (measure of influence)
        try:
            n_samples, n_features = self.X_processed_.shape
            
            # Leverage (hat values)
            X_design = self.X_processed_
            if self.fit_intercept:
                X_design = np.column_stack([np.ones(len(X_design)), X_design])
            
            try:
                hat_matrix = X_design @ np.linalg.inv(X_design.T @ X_design) @ X_design.T
                leverage = np.diag(hat_matrix)
                
                # Cook's distance
                mse = np.mean(residuals ** 2)
                cooks_distance = (standardized_residuals ** 2 / n_features) * (leverage / (1 - leverage) ** 2)
                
                # High influence threshold
                influence_threshold = 4 / n_samples
                influential_indices = np.where(cooks_distance > influence_threshold)[0]
                
            except np.linalg.LinAlgError:
                leverage = np.full(n_samples, np.nan)
                cooks_distance = np.full(n_samples, np.nan)
                influential_indices = np.array([])
            
        except Exception:
            leverage = np.full(len(self.y_original_), np.nan)
            cooks_distance = np.full(len(self.y_original_), np.nan)
            influential_indices = np.array([])
        
        self.outlier_analysis_ = {
            'outlier_detection': {
                'method': 'Standardized Residuals',
                'threshold': outlier_threshold,
                'outlier_indices': outlier_indices.tolist(),
                'n_outliers': len(outlier_indices),
                'outlier_percentage': len(outlier_indices) / len(self.y_original_) * 100
            },
            'influence_analysis': {
                'method': "Cook's Distance",
                'leverage': leverage,
                'cooks_distance': cooks_distance,
                'influence_threshold': influence_threshold if 'influence_threshold' in locals() else np.nan,
                'influential_indices': influential_indices.tolist(),
                'n_influential': len(influential_indices)
            },
            'diagnostics': {
                'standardized_residuals': standardized_residuals,
                'residuals': residuals,
                'fitted_values': y_pred
            }
        }
    
    def _analyze_residuals(self):
        """Comprehensive residual analysis"""
        y_pred = self.model_.predict(self.X_processed_)
        residuals = self.y_original_ - y_pred
        
        # Basic residual statistics
        residual_stats = {
            'mean': np.mean(residuals),
            'std': np.std(residuals),
            'min': np.min(residuals),
            'max': np.max(residuals),
            'q25': np.percentile(residuals, 25),
            'median': np.median(residuals),
            'q75': np.percentile(residuals, 75),
            'skewness': stats.skew(residuals),
            'kurtosis': stats.kurtosis(residuals)
        }
        
        # Residual patterns
        self.residual_analysis_ = {
            'statistics': residual_stats,
            'residuals': residuals,
            'fitted_values': y_pred,
            'standardized_residuals': residuals / np.std(residuals) if np.std(residuals) > 0 else residuals
        }
    
    def get_hyperparameter_config(self, key_prefix: str) -> Dict[str, Any]:
        """Generate Streamlit UI for hyperparameters"""
        
        # Create tabs for different configuration aspects
        tab1, tab2, tab3, tab4 = st.tabs(["Core Parameters", "Advanced Options", "Analysis & Diagnostics", "Algorithm Info"])
        
        with tab1:
            st.markdown("**Linear Regression Configuration**")
            
            # Core regression parameters
            fit_intercept = st.checkbox(
                "Fit Intercept",
                value=self.fit_intercept,
                help="Whether to calculate the intercept (y-axis offset) for this model",
                key=f"{key_prefix}_fit_intercept"
            )
            
            normalize_features = st.checkbox(
                "Normalize Features",
                value=self.normalize_features,
                help="Apply StandardScaler to features for better numerical stability",
                key=f"{key_prefix}_normalize_features"
            )
            
            positive = st.checkbox(
                "Positive Coefficients Only",
                value=self.positive,
                help="Constrain coefficients to be non-negative",
                key=f"{key_prefix}_positive"
            )
            
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
                st.info("📈 Using standard linear regression")
        
        with tab2:
            st.markdown("**Advanced Configuration**")
            
            # Statistical analysis options
            estimate_statistical_properties = st.checkbox(
                "Statistical Analysis",
                value=self.estimate_statistical_properties,
                help="Compute p-values, confidence intervals, F-statistics, etc.",
                key=f"{key_prefix}_statistical_analysis"
            )
            
            if estimate_statistical_properties:
                alpha_level = st.slider(
                    "Significance Level (α):",
                    min_value=0.01,
                    max_value=0.10,
                    value=self.alpha_level,
                    step=0.01,
                    help="Significance level for hypothesis tests and confidence intervals",
                    key=f"{key_prefix}_alpha_level"
                )
                st.info(f"📊 Confidence level: {(1-alpha_level)*100:.0f}%")
            else:
                alpha_level = self.alpha_level
            
            # Diagnostic options
            validate_assumptions = st.checkbox(
                "Assumption Validation",
                value=self.validate_assumptions,
                help="Test regression assumptions (normality, homoscedasticity, etc.)",
                key=f"{key_prefix}_validate_assumptions"
            )
            
            detect_outliers = st.checkbox(
                "Outlier Detection",
                value=self.detect_outliers,
                help="Detect outliers and influential observations",
                key=f"{key_prefix}_detect_outliers"
            )
            
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
        
        with tab3:
            st.markdown("**Linear Regression Analysis Features**")
            
            st.info("""
            **Statistical Analysis:**
            • R² and Adjusted R² calculation
            • F-statistic and model significance
            • Coefficient significance tests (t-tests)
            • Confidence intervals for coefficients
            • AIC/BIC information criteria
            
            **Assumption Testing:**
            • Homoscedasticity (Breusch-Pagan test)
            • Normality of residuals (Shapiro-Wilk/KS test)
            • Multicollinearity detection (VIF)
            • Overall assumption assessment
            
            **Diagnostic Analysis:**
            • Outlier detection via standardized residuals
            • Cook's distance for influence analysis
            • Leverage analysis
            • Comprehensive residual analysis
            """)
            
            # Diagnostic explanations
            if st.button("📊 Understanding Linear Regression Diagnostics", key=f"{key_prefix}_diagnostics_help"):
                st.markdown("""
                **R-squared (R²):**
                - Proportion of variance in target explained by features
                - Range: 0 to 1 (higher is better)
                - Adjusted R² penalizes for additional features
                
                **F-statistic:**
                - Tests overall model significance
                - High F-value with low p-value indicates significant model
                
                **Coefficient Tests:**
                - t-statistics test individual coefficient significance
                - p-values < 0.05 indicate significant features
                - Confidence intervals show coefficient uncertainty
                
                **Assumption Tests:**
                - Homoscedasticity: Constant variance of residuals
                - Normality: Residuals should be normally distributed
                - Multicollinearity: Features shouldn't be highly correlated
                
                **Outlier Detection:**
                - Standardized residuals > 2.5 indicate potential outliers
                - Cook's distance > 4/n indicates influential observations
                - High leverage points have unusual feature values
                """)
            
            # Model interpretation guide
            if st.button("🔍 Coefficient Interpretation Guide", key=f"{key_prefix}_interpretation_guide"):
                st.markdown("""
                **Linear Regression Coefficients:**
                
                **For continuous features:**
                - Coefficient = change in target per unit change in feature
                - Positive coefficient = positive relationship
                - Negative coefficient = negative relationship
                
                **Statistical Significance:**
                - p-value < 0.05: statistically significant
                - Confidence interval not containing 0: significant
                - Large t-statistic: strong evidence for effect
                
                **Effect Size:**
                - Large |coefficient|: strong effect
                - Small |coefficient|: weak effect
                - Compare standardized coefficients for relative importance
                
                **Intercept Interpretation:**
                - Expected target value when all features = 0
                - May not be meaningful if 0 is outside feature ranges
                """)
        
        with tab4:
            st.markdown("**Algorithm Information**")
            
            st.info("""
            **Linear Regression** - The Foundation of Regression:
            • 📈 Ordinary Least Squares (OLS) estimation
            • 🎯 Minimizes sum of squared residuals
            • 📊 Comprehensive statistical inference
            • 🔍 Assumption validation and diagnostics
            • 🎪 Perfect baseline for regression tasks
            • 📐 Linear relationship modeling
            
            **Mathematical Foundation:**
            • Model: y = β₀ + β₁x₁ + β₂x₂ + ... + βₚxₚ + ε
            • Estimation: β̂ = (X'X)⁻¹X'y
            • Assumptions: Linearity, Independence, Homoscedasticity, Normality
            """)
            
            # When to use linear regression
            if st.button("🎯 When to Use Linear Regression", key=f"{key_prefix}_when_to_use"):
                st.markdown("""
                **Ideal Use Cases:**
                
                **Problem Characteristics:**
                • Continuous target variable
                • Linear relationships between features and target
                • Need for interpretable coefficients
                • Baseline model for comparison
                
                **Data Characteristics:**
                • Medium to large sample sizes
                • Low to moderate feature dimensionality
                • Features not highly multicollinear
                • Target roughly normally distributed
                
                **Business Requirements:**
                • Need statistical inference (p-values, confidence intervals)
                • Model interpretability is crucial
                • Fast training and prediction required
                • Regulatory compliance (explainable models)
                
                **Examples:**
                • Predicting house prices based on features
                • Sales forecasting with market variables
                • Scientific studies requiring statistical inference
                • Risk modeling in finance
                """)
            
            # Advantages and limitations
            if st.button("⚖️ Advantages & Limitations", key=f"{key_prefix}_pros_cons"):
                st.markdown("""
                **Advantages:**
                ✅ Highly interpretable coefficients
                ✅ Fast training and prediction
                ✅ No hyperparameter tuning needed
                ✅ Statistical inference available
                ✅ Well-understood mathematical properties
                ✅ Excellent baseline model
                ✅ Memory efficient
                ✅ Stable and robust
                
                **Limitations:**
                ❌ Assumes linear relationships
                ❌ Sensitive to outliers
                ❌ Requires assumption validation
                ❌ Poor with multicollinearity
                ❌ Cannot capture complex patterns
                ❌ May underfit complex data
                ❌ Assumes constant variance
                """)
            
            # Best practices
            if st.button("🎯 Best Practices", key=f"{key_prefix}_best_practices"):
                st.markdown("""
                **Linear Regression Best Practices:**
                
                **Data Preparation:**
                1. Check for and handle outliers
                2. Validate linear relationships (scatter plots)
                3. Scale features if needed
                4. Check for multicollinearity (VIF < 10)
                
                **Model Building:**
                1. Start with simple linear model
                2. Check assumption diagnostics
                3. Consider polynomial features if needed
                4. Use cross-validation for performance estimation
                
                **Interpretation:**
                1. Always check statistical significance
                2. Report confidence intervals
                3. Validate assumptions before interpreting
                4. Consider practical significance vs statistical significance
                
                **Diagnostics:**
                1. Examine residual plots
                2. Test normality of residuals
                3. Check for heteroscedasticity
                4. Identify influential observations
                """)
        
        return {
            "fit_intercept": fit_intercept,
            "normalize_features": normalize_features,
            "polynomial_degree": polynomial_degree,
            "include_bias": include_bias,
            "alpha_level": alpha_level,
            "copy_X": copy_X,
            "positive": positive,
            "estimate_statistical_properties": estimate_statistical_properties,
            "detect_outliers": detect_outliers,
            "validate_assumptions": validate_assumptions,
            "random_state": random_state
        }
    
    def create_model_instance(self, hyperparameters: Dict[str, Any]):
        """Create model instance with given hyperparameters"""
        return LinearRegressionPlugin(
            fit_intercept=hyperparameters.get("fit_intercept", self.fit_intercept),
            normalize_features=hyperparameters.get("normalize_features", self.normalize_features),
            polynomial_degree=hyperparameters.get("polynomial_degree", self.polynomial_degree),
            include_bias=hyperparameters.get("include_bias", self.include_bias),
            alpha_level=hyperparameters.get("alpha_level", self.alpha_level),
            copy_X=hyperparameters.get("copy_X", self.copy_X),
            positive=hyperparameters.get("positive", self.positive),
            estimate_statistical_properties=hyperparameters.get("estimate_statistical_properties", self.estimate_statistical_properties),
            detect_outliers=hyperparameters.get("detect_outliers", self.detect_outliers),
            validate_assumptions=hyperparameters.get("validate_assumptions", self.validate_assumptions),
            random_state=hyperparameters.get("random_state", self.random_state)
        )
    
    def preprocess_data(self, X, y=None, training=True):
        """Preprocess data for Linear Regression"""
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
        """Check if Linear Regression is compatible with the given data"""
        # Check minimum samples
        if X.shape[0] < self._min_samples_required:
            return False, f"Linear regression requires at least {self._min_samples_required} samples, got {X.shape[0]}"
        
        # Check for regression targets
        if y is not None:
            if not np.issubdtype(y.dtype, np.number):
                return False, "Linear regression requires continuous numerical target values"
            
            # Check for sufficient variance in target
            if np.var(y) == 0:
                return False, "Target variable has zero variance (all values are the same)"
            
            # Sample size considerations
            n_samples, n_features = X.shape
            effective_features = n_features * self.polynomial_degree if self.polynomial_degree > 1 else n_features
            
            # Rule of thumb: at least 10-20 samples per feature
            min_recommended = max(20, effective_features * 10)
            
            advantages = []
            considerations = []
            
            # Sample size assessment
            if n_samples >= min_recommended:
                advantages.append(f"Excellent sample size ({n_samples} samples for {effective_features} features)")
            elif n_samples >= effective_features * 5:
                advantages.append(f"Good sample size ({n_samples} samples for {effective_features} features)")
            elif n_samples >= effective_features * 2:
                considerations.append(f"Adequate but small sample size ({n_samples} samples for {effective_features} features)")
            else:
                considerations.append(f"Very small sample size ({n_samples} samples for {effective_features} features) - results may be unreliable")
            
            # Feature dimensionality
            if n_features <= 20:
                advantages.append("Low-dimensional feature space (good for interpretation)")
            elif n_features <= 100:
                advantages.append("Moderate feature dimensionality")
            else:
                considerations.append(f"High-dimensional data ({n_features} features) - may need regularization")
            
            # Target distribution
            target_skew = abs(stats.skew(y))
            if target_skew < 0.5:
                advantages.append("Target distribution is approximately normal")
            elif target_skew < 1.0:
                considerations.append("Target distribution is slightly skewed")
            else:
                considerations.append("Target distribution is heavily skewed - consider transformation")
            
            # Polynomial features warning
            if self.polynomial_degree > 1:
                poly_features = n_features * (self.polynomial_degree ** 2)  # Rough estimate
                considerations.append(f"Polynomial degree {self.polynomial_degree} creates ~{poly_features} features")
            
            # Build compatibility message
            suitability = ("Excellent" if len(considerations) == 0 else "Good" if len(considerations) <= 1 else 
                          "Fair" if len(considerations) <= 2 else "Poor")
            
            message_parts = [
                f"✅ Compatible with {n_samples} samples, {n_features} features",
                f"📊 Suitability: {suitability}"
            ]
            
            if advantages:
                message_parts.append("🎯 Advantages: " + "; ".join(advantages))
            
            if considerations:
                message_parts.append("💡 Considerations: " + "; ".join(considerations))
            
            return True, " | ".join(message_parts)
        
        return True, f"Compatible with {X.shape[0]} samples and {X.shape[1]} features"
    
    def get_feature_importance(self) -> Optional[Dict[str, Any]]:
        """Get feature importance based on coefficient magnitudes"""
        if not self.is_fitted_:
            return None
        
        # Get coefficients
        coefficients = self.model_.coef_
        feature_names = self.feature_names_processed_
        
        # Calculate importance as absolute coefficient values
        importance = np.abs(coefficients)
        
        # Normalize to sum to 1
        if np.sum(importance) > 0:
            importance_normalized = importance / np.sum(importance)
        else:
            importance_normalized = importance
        
        # Create feature importance dictionary
        feature_importance = {}
        for i, (name, coef, imp, norm_imp) in enumerate(zip(feature_names, coefficients, importance, importance_normalized)):
            feature_importance[name] = {
                'coefficient': coef,
                'absolute_coefficient': imp,
                'normalized_importance': norm_imp,
                'rank': i + 1
            }
        
        # Sort by importance
        sorted_features = sorted(feature_importance.items(), key=lambda x: x[1]['absolute_coefficient'], reverse=True)
        
        # Update ranks
        for rank, (name, info) in enumerate(sorted_features):
            feature_importance[name]['rank'] = rank + 1
        
        return {
            'feature_importance': feature_importance,
            'sorted_features': [name for name, _ in sorted_features],
            'sorted_importance': [info['normalized_importance'] for _, info in sorted_features],
            'coefficients': coefficients,
            'feature_names': feature_names,
            'intercept': self.model_.intercept_ if self.fit_intercept else 0,
            'interpretation': 'Feature importance based on absolute coefficient magnitude'
        }
    
    def get_model_params(self) -> Dict[str, Any]:
        """Get current model parameters"""
        if not self.is_fitted_:
            return {}
        
        return {
            "n_features": self.n_features_in_,
            "feature_names": self.feature_names_,
            "processed_features": len(self.feature_names_processed_),
            "processed_feature_names": self.feature_names_processed_,
            "polynomial_degree": self.polynomial_degree,
            "fit_intercept": self.fit_intercept,
            "normalize_features": self.normalize_features,
            "coefficients": self.model_.coef_.tolist(),
            "intercept": self.model_.intercept_ if self.fit_intercept else 0,
            "positive_constraint": self.positive,
            "feature_scaling": self.scaler_ is not None,
            "statistical_analysis": self.estimate_statistical_properties,
            "assumption_validation": self.validate_assumptions,
            "outlier_detection": self.detect_outliers
        }
    
    def get_statistical_summary(self) -> Dict[str, Any]:
        """Get comprehensive statistical analysis results"""
        if not self.is_fitted_ or not self.estimate_statistical_properties:
            return {"status": "Statistical analysis not performed"}
        
        return {
            "model_summary": self.statistical_properties_.get('model_summary', {}),
            "coefficient_analysis": self.statistical_properties_.get('coefficients', {}),
            "assumption_tests": self.assumption_tests_,
            "outlier_analysis": self.outlier_analysis_,
            "residual_analysis": self.residual_analysis_
        }
    
    def plot_diagnostics(self, figsize=(15, 12)):
        """Create comprehensive diagnostic plots"""
        if not self.is_fitted_:
            raise ValueError("Model must be fitted before creating diagnostic plots")
        
        fig, axes = plt.subplots(2, 3, figsize=figsize)
        
        # Get data for plots
        y_pred = self.model_.predict(self.X_processed_)
        residuals = self.y_original_ - y_pred
        standardized_residuals = residuals / np.std(residuals) if np.std(residuals) > 0 else residuals
        
        # 1. Residuals vs Fitted
        axes[0, 0].scatter(y_pred, residuals, alpha=0.6, color='steelblue')
        axes[0, 0].axhline(y=0, color='red', linestyle='--', alpha=0.7)
        axes[0, 0].set_xlabel('Fitted Values')
        axes[0, 0].set_ylabel('Residuals')
        axes[0, 0].set_title('Residuals vs Fitted')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Add smoother line
        try:
            from scipy.interpolate import make_interp_spline
            if len(y_pred) > 3:
                x_smooth = np.linspace(y_pred.min(), y_pred.max(), 100)
                # Sort for interpolation
                sorted_indices = np.argsort(y_pred)
                y_pred_sorted = y_pred[sorted_indices]
                residuals_sorted = residuals[sorted_indices]
                
                # Create smoother
                spl = make_interp_spline(y_pred_sorted, residuals_sorted, k=min(3, len(y_pred_sorted)-1))
                y_smooth = spl(x_smooth)
                axes[0, 0].plot(x_smooth, y_smooth, 'red', linewidth=2, alpha=0.7)
        except:
            pass
        
        # 2. Q-Q Plot for normality
        try:
            stats.probplot(residuals, dist="norm", plot=axes[0, 1])
            axes[0, 1].set_title('Q-Q Plot (Normality Test)')
            axes[0, 1].grid(True, alpha=0.3)
        except:
            axes[0, 1].hist(residuals, bins=20, alpha=0.7, color='steelblue')
            axes[0, 1].set_title('Residual Distribution')
            axes[0, 1].set_xlabel('Residuals')
            axes[0, 1].set_ylabel('Frequency')
        
        # 3. Scale-Location (Homoscedasticity)
        sqrt_abs_residuals = np.sqrt(np.abs(standardized_residuals))
        axes[0, 2].scatter(y_pred, sqrt_abs_residuals, alpha=0.6, color='steelblue')
        axes[0, 2].set_xlabel('Fitted Values')
        axes[0, 2].set_ylabel('√|Standardized Residuals|')
        axes[0, 2].set_title('Scale-Location Plot')
        axes[0, 2].grid(True, alpha=0.3)
        
        # 4. Leverage vs Residuals (Cook's Distance)
        if 'leverage' in self.outlier_analysis_.get('influence_analysis', {}):
            leverage = self.outlier_analysis_['influence_analysis']['leverage']
            if not np.isnan(leverage).all():
                axes[1, 0].scatter(leverage, standardized_residuals, alpha=0.6, color='steelblue')
                axes[1, 0].set_xlabel('Leverage')
                axes[1, 0].set_ylabel('Standardized Residuals')
                axes[1, 0].set_title('Leverage vs Standardized Residuals')
                axes[1, 0].grid(True, alpha=0.3)
                
                # Add Cook's distance contours
                if 'cooks_distance' in self.outlier_analysis_['influence_analysis']:
                    cooks_d = self.outlier_analysis_['influence_analysis']['cooks_distance']
                    if not np.isnan(cooks_d).all():
                        # Color points by Cook's distance
                        scatter = axes[1, 0].scatter(leverage, standardized_residuals, 
                                                   c=cooks_d, cmap='viridis', alpha=0.6)
                        plt.colorbar(scatter, ax=axes[1, 0], label="Cook's Distance")
            else:
                axes[1, 0].text(0.5, 0.5, 'Leverage analysis\nnot available', 
                              ha='center', va='center', transform=axes[1, 0].transAxes)
                axes[1, 0].set_title('Leverage Analysis')
        else:
            axes[1, 0].text(0.5, 0.5, 'Leverage analysis\nnot available', 
                          ha='center', va='center', transform=axes[1, 0].transAxes)
            axes[1, 0].set_title('Leverage Analysis')
        
        # 5. Actual vs Predicted
        min_val = min(self.y_original_.min(), y_pred.min())
        max_val = max(self.y_original_.max(), y_pred.max())
        axes[1, 1].scatter(self.y_original_, y_pred, alpha=0.6, color='steelblue')
        axes[1, 1].plot([min_val, max_val], [min_val, max_val], 'red', linestyle='--', alpha=0.7)
        axes[1, 1].set_xlabel('Actual Values')
        axes[1, 1].set_ylabel('Predicted Values')
        axes[1, 1].set_title('Actual vs Predicted')
        axes[1, 1].grid(True, alpha=0.3)
        
        # Add R² to the plot
        r2 = r2_score(self.y_original_, y_pred)
        axes[1, 1].text(0.05, 0.95, f'R² = {r2:.3f}', transform=axes[1, 1].transAxes, 
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # 6. Residual histogram
        axes[1, 2].hist(residuals, bins=min(20, len(residuals)//5), alpha=0.7, color='steelblue', density=True)
        axes[1, 2].set_xlabel('Residuals')
        axes[1, 2].set_ylabel('Density')
        axes[1, 2].set_title('Residual Distribution')
        
        # Overlay normal distribution
        x_norm = np.linspace(residuals.min(), residuals.max(), 100)
        y_norm = stats.norm.pdf(x_norm, residuals.mean(), residuals.std())
        axes[1, 2].plot(x_norm, y_norm, 'red', linewidth=2, alpha=0.7, label='Normal')
        axes[1, 2].legend()
        axes[1, 2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def plot_feature_importance(self, max_features=15, figsize=(10, 6)):
        """Plot feature importance based on coefficient magnitudes"""
        importance_data = self.get_feature_importance()
        if not importance_data:
            raise ValueError("Model must be fitted to plot feature importance")
        
        # Get top features
        sorted_features = importance_data['sorted_features'][:max_features]
        sorted_importance = importance_data['sorted_importance'][:max_features]
        coefficients = importance_data['coefficients'][:max_features]
        
        # Create horizontal bar plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        
        # Plot 1: Importance (absolute coefficients)
        colors = ['green' if coef > 0 else 'red' for coef in coefficients]
        y_pos = np.arange(len(sorted_features))
        
        ax1.barh(y_pos, sorted_importance, color=colors, alpha=0.7)
        ax1.set_yticks(y_pos)
        ax1.set_yticklabels(sorted_features)
        ax1.set_xlabel('Normalized Importance')
        ax1.set_title('Feature Importance\n(Absolute Coefficients)')
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Actual coefficients with sign
        actual_coefs = [importance_data['feature_importance'][feat]['coefficient'] 
                       for feat in sorted_features]
        colors = ['green' if coef > 0 else 'red' for coef in actual_coefs]
        
        ax2.barh(y_pos, actual_coefs, color=colors, alpha=0.7)
        ax2.set_yticks(y_pos)
        ax2.set_yticklabels(sorted_features)
        ax2.set_xlabel('Coefficient Value')
        ax2.set_title('Feature Coefficients\n(with Sign)')
        ax2.axvline(x=0, color='black', linestyle='-', alpha=0.3)
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def get_training_info(self) -> Dict[str, Any]:
        """Get information about the training process"""
        if not self.is_fitted_:
            return {"status": "Not fitted"}
        
        info = {
            "algorithm": "Linear Regression",
            "type": "Ordinary Least Squares (OLS)",
            "training_completed": True,
            "regression_characteristics": {
                "linear_model": True,
                "parametric": True,
                "assumes_linearity": True,
                "provides_inference": True,
                "interpretable": True,
                "fast_training": True
            },
            "model_configuration": {
                "fit_intercept": self.fit_intercept,
                "normalize_features": self.normalize_features,
                "polynomial_degree": self.polynomial_degree,
                "positive_constraint": self.positive,
                "n_original_features": self.n_features_in_,
                "n_processed_features": len(self.feature_names_processed_),
                "feature_scaling": self.scaler_ is not None
            },
            "analysis_performed": {
                "statistical_inference": self.estimate_statistical_properties,
                "assumption_validation": self.validate_assumptions,
                "outlier_detection": self.detect_outliers,
                "residual_analysis": True
            }
        }
        
        # Add statistical results if available
        if self.statistical_properties_:
            model_summary = self.statistical_properties_.get('model_summary', {})
            info["performance_metrics"] = {
                "r_squared": model_summary.get('r_squared'),
                "adj_r_squared": model_summary.get('adj_r_squared'),
                "f_statistic": model_summary.get('f_statistic'),
                "f_p_value": model_summary.get('f_p_value'),
                "aic": model_summary.get('aic'),
                "bic": model_summary.get('bic')
            }
        
        # Add assumption test results
        if self.assumption_tests_:
            overall_assessment = self.assumption_tests_.get('overall_assessment', {})
            info["assumption_validation"] = {
                "overall_score": overall_assessment.get('score'),
                "interpretation": overall_assessment.get('interpretation'),
                "recommendation": overall_assessment.get('recommendation')
            }
        
        # Add outlier analysis results
        if self.outlier_analysis_:
            outlier_info = self.outlier_analysis_.get('outlier_detection', {})
            info["outlier_analysis"] = {
                "n_outliers": outlier_info.get('n_outliers'),
                "outlier_percentage": outlier_info.get('outlier_percentage')
            }
        
        return info

    def get_algorithm_specific_metrics(self, y_true: Optional[np.ndarray] = None, y_pred: Optional[np.ndarray] = None, y_proba: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        Retrieve algorithm-specific statistical metrics for Linear Regression.

        Parameters:
        -----------
        y_true : np.ndarray, optional
            Actual target values. Not directly used for these specific metrics as they are
            derived from the fitted model's internal state.
        y_pred : np.ndarray, optional
            Predicted target values. Not directly used for these specific metrics.
        y_proba : np.ndarray, optional
            Predicted probabilities. Not applicable for regressors.

        Returns:
        --------
        Dict[str, Any]
            A dictionary containing algorithm-specific statistical metrics.
        """
        if not self.is_fitted_:
            return {"error": "Model not fitted. Cannot retrieve specific metrics."}

        metrics = {}

        # From statistical_properties_['model_summary']
        if self.statistical_properties_ and 'model_summary' in self.statistical_properties_:
            model_summary = self.statistical_properties_['model_summary']
            metrics['lr_f_statistic'] = model_summary.get('f_statistic')
            metrics['lr_f_p_value'] = model_summary.get('f_p_value')
            metrics['lr_aic'] = model_summary.get('aic')
            metrics['lr_bic'] = model_summary.get('bic')
            metrics['lr_adj_r_squared'] = model_summary.get('adj_r_squared')
            metrics['lr_df_model'] = model_summary.get('df_model')
            metrics['lr_df_residual'] = model_summary.get('df_residual')

        # Number of significant coefficients
        if self.statistical_properties_ and 'coefficients' in self.statistical_properties_:
            coeffs_data = self.statistical_properties_['coefficients']
            coef_p_values = coeffs_data.get('coef_p')
            if coef_p_values is not None:
                significant_coeffs = np.sum(np.array(coef_p_values) < self.alpha_level)
                metrics['lr_num_significant_coeffs'] = int(significant_coeffs)
            
            intercept_p_value = coeffs_data.get('intercept_p')
            if intercept_p_value is not None and self.fit_intercept:
                metrics['lr_intercept_significant'] = bool(intercept_p_value < self.alpha_level)


        # From assumption_tests_
        if self.assumption_tests_:
            if 'normality' in self.assumption_tests_:
                normality_test = self.assumption_tests_['normality']
                metrics['lr_residuals_normality_p_value'] = normality_test.get('p_value')
                metrics['lr_residuals_normality_test_name'] = normality_test.get('test_name')
            
            if 'homoscedasticity' in self.assumption_tests_:
                homoscedasticity_test = self.assumption_tests_['homoscedasticity']
                metrics['lr_homoscedasticity_p_value'] = homoscedasticity_test.get('p_value')
                metrics['lr_homoscedasticity_test_name'] = homoscedasticity_test.get('test_name')

            if 'multicollinearity' in self.assumption_tests_:
                multicollinearity_test = self.assumption_tests_['multicollinearity']
                if 'max_vif' in multicollinearity_test:
                    metrics['lr_max_vif'] = multicollinearity_test.get('max_vif')
                elif 'max_correlation' in multicollinearity_test: # Fallback
                    metrics['lr_max_feature_correlation'] = multicollinearity_test.get('max_correlation')
                metrics['lr_multicollinearity_test_name'] = multicollinearity_test.get('test_name')
        
        if not metrics:
            metrics['info'] = "No specific Linear Regression metrics were available (e.g., statistical analysis not enabled or model not fitted)."
            
        return metrics


# Factory function
def get_plugin():
    """Factory function to get the plugin instance"""
    return LinearRegressionPlugin()
