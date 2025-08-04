import streamlit as st
import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple, Optional, List, Union
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from scipy import stats

# Try to import Naive Bayes with graceful fallback
try:
    from sklearn.naive_bayes import GaussianNB
    from sklearn.model_selection import cross_val_score
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    GaussianNB = None

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

class GaussianNaiveBayesPlugin(BaseEstimator, ClassifierMixin, MLPlugin):
    """
    Gaussian Naive Bayes Classifier Plugin - Fast Probabilistic Learning
    
    Gaussian Naive Bayes assumes features follow a normal distribution and are
    conditionally independent given the class. Despite the "naive" independence
    assumption, it often performs surprisingly well in practice and is extremely
    fast to train and predict.
    """
    
    def __init__(self,
                 priors=None,
                 var_smoothing=1e-9,
                 # Advanced parameters
                 feature_independence_test=False,
                 normality_test=False,
                 class_balance_analysis=True,
                 probability_calibration=False,
                 calibration_method='isotonic',
                 cv_folds=5,
                 # Preprocessing options
                 auto_scaling=False,
                 scaling_method='standard',
                 outlier_detection=False,
                 outlier_threshold=3.0):
        """
        Initialize Gaussian Naive Bayes Classifier with comprehensive parameter support
        
        Parameters:
        -----------
        priors : array-like, optional
            Prior probabilities of the classes
        var_smoothing : float, default=1e-9
            Portion of the largest variance added to variances for numerical stability
        feature_independence_test : bool, default=False
            Test feature independence assumption
        normality_test : bool, default=False
            Test Gaussian distribution assumption for features
        class_balance_analysis : bool, default=True
            Analyze class balance and its impact
        probability_calibration : bool, default=False
            Enable probability calibration for better probability estimates
        calibration_method : str, default='isotonic'
            Calibration method ('isotonic' or 'sigmoid')
        cv_folds : int, default=5
            Number of cross-validation folds
        auto_scaling : bool, default=False
            Whether to automatically scale features (usually not needed for NB)
        scaling_method : str, default='standard'
            Scaling method if auto_scaling is enabled
        outlier_detection : bool, default=False
            Enable outlier detection and analysis
        outlier_threshold : float, default=3.0
            Z-score threshold for outlier detection
        """
        super().__init__()
        
        # Core Naive Bayes parameters
        self.priors = priors
        self.var_smoothing = var_smoothing
        
        # Analysis parameters
        self.feature_independence_test = feature_independence_test
        self.normality_test = normality_test
        self.class_balance_analysis = class_balance_analysis
        self.probability_calibration = probability_calibration
        self.calibration_method = calibration_method
        self.cv_folds = cv_folds
        
        # Preprocessing parameters
        self.auto_scaling = auto_scaling
        self.scaling_method = scaling_method
        self.outlier_detection = outlier_detection
        self.outlier_threshold = outlier_threshold
        
        # Plugin metadata
        self._name = "Gaussian Naive Bayes"
        self._description = "Fast probabilistic classifier that assumes features are normally distributed and conditionally independent."
        self._category = "Probabilistic Models"
        self._algorithm_type = "Probabilistic Classifier"
        self._paper_reference = "Hand, D. J., & Yu, K. (2001). Idiot's Bayes—not so stupid after all?"
        
        # Capability flags
        self._supports_classification = True
        self._supports_regression = False
        self._supports_binary = True
        self._supports_multiclass = True
        self._min_samples_required = 10  # Very small sample friendly
        self._handles_missing_values = False
        self._requires_scaling = False  # Scale-invariant due to normalization
        self._supports_sparse = False
        self._is_linear = False  # Decision boundary can be non-linear
        self._provides_feature_importance = False  # No direct feature importance
        self._provides_probabilities = True  # Excellent probability estimates
        self._handles_categorical = False
        self._ensemble_method = False
        self._probabilistic = True
        self._fast_training = True
        self._fast_prediction = True
        self._interpretable = True
        self._handles_small_data = True
        self._baseline_algorithm = True
        self._assumption_dependent = True  # Strong assumptions about data
        
        # Internal attributes
        self.model_ = None
        self.calibrated_model_ = None
        self.scaler_ = None
        self.label_encoder_ = None
        self.feature_names_ = None
        self.n_features_in_ = None
        self.classes_ = None
        self.is_fitted_ = False
        self.training_data_ = None
        self.training_labels_ = None
        self.feature_stats_ = None
        self.class_stats_ = None
        self.independence_test_results_ = None
        self.normality_test_results_ = None
        self.outlier_analysis_ = None
        self.class_balance_analysis_ = None
        
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
            "framework": "scikit-learn",
            "year_introduced": 1960,
            "key_assumptions": {
                "feature_independence": "Features are conditionally independent given class",
                "gaussian_distribution": "Each feature follows normal distribution within each class",
                "class_separability": "Classes have different feature distributions"
            },
            "algorithm_mechanics": {
                "training_process": [
                    "Calculate class priors P(C_k)",
                    "For each feature and class, estimate μ and σ²",
                    "Store Gaussian parameters for each feature-class combination"
                ],
                "prediction_process": [
                    "For new instance x, calculate likelihood for each class",
                    "P(x|C_k) = ∏ P(x_i|C_k) (independence assumption)",
                    "Apply Bayes' theorem: P(C_k|x) ∝ P(x|C_k) * P(C_k)",
                    "Predict class with highest posterior probability"
                ],
                "bayes_theorem": "P(C|X) = P(X|C) * P(C) / P(X)",
                "gaussian_likelihood": "P(x_i|C_k) = (1/√(2π*σ²)) * exp(-(x_i-μ)²/(2σ²))",
                "log_probabilities": "Used to avoid numerical underflow in practice"
            },
            "strengths": [
                "Extremely fast training and prediction",
                "Works well with small datasets",
                "Naturally handles multi-class problems",
                "Provides calibrated probability estimates",
                "No hyperparameters to tune (almost parameter-free)",
                "Not sensitive to irrelevant features",
                "Good baseline algorithm",
                "Scale-invariant (doesn't require feature scaling)",
                "Simple to understand and implement",
                "Robust to noise in training data",
                "Works well when independence assumption holds",
                "Low memory requirements",
                "Can handle streaming data easily",
                "Good performance on text classification",
                "Handles missing values naturally (with proper implementation)"
            ],
            "weaknesses": [
                "Strong independence assumption rarely holds in practice",
                "Assumes Gaussian distribution for all features",
                "Can be outperformed by more sophisticated methods",
                "Poor performance when features are highly correlated",
                "Sensitive to skewed data distributions",
                "May struggle with complex feature interactions",
                "Decision boundaries limited by distributional assumptions",
                "Can be biased if training data is not representative",
                "Zero-frequency problem (handled by smoothing)",
                "Not suitable for highly non-linear relationships"
            ],
            "ideal_use_cases": [
                "Text classification and sentiment analysis",
                "Spam filtering and email classification",
                "Small datasets with limited training samples",
                "Real-time prediction systems",
                "Multi-class classification problems",
                "Baseline model for comparison",
                "Medical diagnosis with continuous features",
                "Document categorization",
                "Feature selection preprocessing",
                "Streaming/online learning scenarios",
                "When training time is critical",
                "When interpretability is important",
                "Datasets with many irrelevant features",
                "Problems where features are approximately independent"
            ],
            "assumption_analysis": {
                "independence_assumption": {
                    "reality": "Features are rarely truly independent",
                    "impact": "Algorithm often works despite violation",
                    "mitigation": "Feature selection, dimensionality reduction",
                    "testing": "Statistical independence tests available"
                },
                "gaussian_assumption": {
                    "reality": "Real data often deviates from normal distribution",
                    "impact": "Can reduce performance significantly",
                    "mitigation": "Data transformation, other NB variants",
                    "testing": "Normality tests (Shapiro-Wilk, Anderson-Darling)"
                },
                "why_it_works_anyway": [
                    "Robust to assumption violations in practice",
                    "Only requires correct ranking of probabilities",
                    "Large margin between classes helps",
                    "Feature averaging effect reduces individual violations"
                ]
            },
            "variants_and_extensions": {
                "multinomial_nb": {
                    "use_case": "Text data, count features",
                    "assumption": "Multinomial distribution",
                    "example": "Bag-of-words text classification"
                },
                "bernoulli_nb": {
                    "use_case": "Binary features",
                    "assumption": "Bernoulli distribution",
                    "example": "Binary text features, presence/absence"
                },
                "complement_nb": {
                    "use_case": "Imbalanced datasets",
                    "advantage": "Better handling of class imbalance",
                    "method": "Uses complement of each class"
                },
                "categorical_nb": {
                    "use_case": "Categorical features",
                    "assumption": "Categorical distribution",
                    "method": "Handles discrete categorical data"
                }
            },
            "hyperparameter_guide": {
                "var_smoothing": {
                    "purpose": "Numerical stability and zero-variance handling",
                    "default": "1e-9 (usually works well)",
                    "tuning": "Increase if experiencing numerical issues",
                    "range": "1e-12 to 1e-6 typically"
                },
                "priors": {
                    "purpose": "Class prior probabilities",
                    "default": "Estimated from training data",
                    "custom": "Use domain knowledge for imbalanced data",
                    "uniform": "Set to None for uniform priors"
                }
            },
            "performance_characteristics": {
                "training_complexity": "O(n*d) - linear in samples and features",
                "prediction_complexity": "O(d*k) - linear in features and classes",
                "memory_usage": "O(d*k) - stores means and variances",
                "scalability": "Excellent - handles large datasets well",
                "parallelization": "Naturally parallelizable",
                "incremental_learning": "Easy to implement online version"
            },
            "comparison_with_other_methods": {
                "vs_logistic_regression": {
                    "assumptions": "NB: independence, LR: linear separability",
                    "probabilities": "NB: generative, LR: discriminative",
                    "performance": "LR often better with correlated features",
                    "speed": "NB faster, especially for training"
                },
                "vs_knn": {
                    "assumptions": "NB: distributional, KNN: local similarity",
                    "training_time": "NB: instant, KNN: no training",
                    "prediction_time": "NB: constant, KNN: linear in training size",
                    "interpretability": "Both interpretable but differently"
                },
                "vs_decision_trees": {
                    "feature_interactions": "NB: ignores, DT: captures naturally",
                    "overfitting": "NB: less prone, DT: more prone",
                    "categorical_features": "DT handles better",
                    "continuous_features": "NB handles more naturally"
                },
                "vs_svm": {
                    "small_data": "NB: excellent, SVM: can struggle",
                    "high_dimensions": "Both can handle well",
                    "interpretability": "NB: probabilistic, SVM: margin-based",
                    "kernel_trick": "SVM: yes, NB: not directly applicable"
                }
            },
            "practical_tips": {
                "data_preprocessing": [
                    "Check for normality in features (optional but good)",
                    "Handle outliers carefully - they affect Gaussian estimates",
                    "Feature scaling not required but won't hurt",
                    "Consider log-transform for skewed features"
                ],
                "feature_engineering": [
                    "Remove highly correlated features",
                    "Consider feature selection methods",
                    "Create features that are more likely to be independent",
                    "Avoid too many irrelevant features"
                ],
                "model_evaluation": [
                    "Use cross-validation for small datasets",
                    "Check probability calibration",
                    "Analyze feature contributions",
                    "Test assumptions when possible"
                ],
                "troubleshooting": [
                    "Poor performance: check feature correlations",
                    "Probability issues: increase var_smoothing",
                    "Class imbalance: adjust priors or use stratified sampling",
                    "Skewed features: consider transformations"
                ]
            }
        }
    
    def fit(self, X, y, 
            analyze_assumptions=None,
            store_training_data=True):
        """
        Fit the Gaussian Naive Bayes Classifier model
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Training data
        y : array-like, shape (n_samples,)
            Target values
        analyze_assumptions : bool, optional
            Whether to analyze model assumptions
        store_training_data : bool, default=True
            Whether to store training data for analysis
            
        Returns:
        --------
        self : object
        """
        if not SKLEARN_AVAILABLE:
            raise ImportError("scikit-learn is not installed. Please install with: pip install scikit-learn")
        
        # Validate input
        X, y = check_X_y(X, y, accept_sparse=False)
        
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
        n_classes = len(self.classes_)
        
        # Store original data for analysis
        if store_training_data:
            self.training_data_ = X.copy() if hasattr(X, 'copy') else np.array(X, copy=True)
            self.training_labels_ = y.copy() if hasattr(y, 'copy') else np.array(y, copy=True)
        
        # Feature scaling (optional for Naive Bayes)
        if self.auto_scaling:
            if self.scaling_method == 'standard':
                self.scaler_ = StandardScaler()
            else:
                from sklearn.preprocessing import MinMaxScaler, RobustScaler
                if self.scaling_method == 'minmax':
                    self.scaler_ = MinMaxScaler()
                elif self.scaling_method == 'robust':
                    self.scaler_ = RobustScaler()
                else:
                    self.scaler_ = StandardScaler()
            
            X_scaled = self.scaler_.fit_transform(X)
        else:
            X_scaled = X
            self.scaler_ = None
        
        # Create and fit Naive Bayes model
        self.model_ = GaussianNB(
            priors=self.priors,
            var_smoothing=self.var_smoothing
        )
        
        self.model_.fit(X_scaled, y_encoded)
        
        # Probability calibration if requested
        if self.probability_calibration:
            from sklearn.calibration import CalibratedClassifierCV
            self.calibrated_model_ = CalibratedClassifierCV(
                self.model_, 
                method=self.calibration_method,
                cv=self.cv_folds
            )
            self.calibrated_model_.fit(X_scaled, y_encoded)
        
        # Analyze model assumptions and characteristics
        if analyze_assumptions is None:
            analyze_assumptions = (self.feature_independence_test or 
                                 self.normality_test or 
                                 self.class_balance_analysis or 
                                 self.outlier_detection)
        
        if analyze_assumptions:
            self._analyze_model_assumptions(X_scaled, y_encoded)
        
        # Calculate feature statistics
        self._calculate_feature_statistics(X_scaled, y_encoded)
        
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
        X = check_array(X, accept_sparse=False)
        
        if not SKLEARN_AVAILABLE:
            raise ImportError("scikit-learn is not installed. Please install with: pip install scikit-learn")
        
        # Apply scaling if fitted
        if self.scaler_ is not None:
            X_scaled = self.scaler_.transform(X)
        else:
            X_scaled = X
        
        # Use calibrated model if available
        if self.calibrated_model_ is not None:
            y_pred_encoded = self.calibrated_model_.predict(X_scaled)
        else:
            y_pred_encoded = self.model_.predict(X_scaled)
        
        # Convert back to original labels
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
        X = check_array(X, accept_sparse=False)
        
        if not SKLEARN_AVAILABLE:
            raise ImportError("scikit-learn is not installed. Please install with: pip install scikit-learn")
        
        # Apply scaling if fitted
        if self.scaler_ is not None:
            X_scaled = self.scaler_.transform(X)
        else:
            X_scaled = X
        
        # Use calibrated model if available
        if self.calibrated_model_ is not None:
            probabilities = self.calibrated_model_.predict_proba(X_scaled)
        else:
            probabilities = self.model_.predict_proba(X_scaled)
        
        return probabilities
    
    def predict_log_proba(self, X):
        """
        Predict log class probabilities for samples in X
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Samples
            
        Returns:
        --------
        log_probabilities : array, shape (n_samples, n_classes)
            Log class probabilities
        """
        check_is_fitted(self, 'is_fitted_')
        X = check_array(X, accept_sparse=False)
        
        # Apply scaling if fitted
        if self.scaler_ is not None:
            X_scaled = self.scaler_.transform(X)
        else:
            X_scaled = X
        
        # Use base model for log probabilities (calibrated doesn't support this)
        return self.model_.predict_log_proba(X_scaled)
    
    def _analyze_model_assumptions(self, X, y):
        """
        Analyze the key assumptions of Naive Bayes
        
        Parameters:
        -----------
        X : array-like
            Scaled training features
        y : array-like
            Encoded training targets
        """
        n_samples, n_features = X.shape
        
        # Initialize results
        results = {
            'independence_tests': {},
            'normality_tests': {},
            'outlier_analysis': {},
            'class_balance': {}
        }
        
        # Feature independence tests
        if self.feature_independence_test and n_features > 1:
            from scipy.stats import chi2_contingency, pearsonr
            
            independence_results = []
            for i in range(n_features):
                for j in range(i + 1, n_features):
                    try:
                        # Pearson correlation
                        corr, p_value = pearsonr(X[:, i], X[:, j])
                        independence_results.append({
                            'feature_pair': (i, j),
                            'correlation': corr,
                            'p_value': p_value,
                            'independent': p_value > 0.05,
                            'feature_names': (self.feature_names_[i], self.feature_names_[j])
                        })
                    except:
                        pass
            
            results['independence_tests'] = {
                'performed': True,
                'results': independence_results,
                'summary': {
                    'total_pairs': len(independence_results),
                    'independent_pairs': sum(1 for r in independence_results if r['independent']),
                    'violation_rate': 1 - (sum(1 for r in independence_results if r['independent']) / max(1, len(independence_results)))
                }
            }
        
        # Normality tests
        if self.normality_test:
            from scipy.stats import shapiro, normaltest
            
            normality_results = []
            for i in range(n_features):
                for class_label in np.unique(y):
                    class_data = X[y == class_label, i]
                    if len(class_data) >= 3:  # Minimum for normality test
                        try:
                            # Shapiro-Wilk test (for smaller samples)
                            if len(class_data) <= 5000:
                                stat, p_value = shapiro(class_data)
                                test_name = 'Shapiro-Wilk'
                            else:
                                # D'Agostino's test (for larger samples)
                                stat, p_value = normaltest(class_data)
                                test_name = "D'Agostino"
                            
                            normality_results.append({
                                'feature': i,
                                'feature_name': self.feature_names_[i],
                                'class': class_label,
                                'class_name': self.label_encoder_.classes_[class_label],
                                'test': test_name,
                                'statistic': stat,
                                'p_value': p_value,
                                'normal': p_value > 0.05,
                                'sample_size': len(class_data)
                            })
                        except:
                            pass
            
            results['normality_tests'] = {
                'performed': True,
                'results': normality_results,
                'summary': {
                    'total_tests': len(normality_results),
                    'normal_distributions': sum(1 for r in normality_results if r['normal']),
                    'violation_rate': 1 - (sum(1 for r in normality_results if r['normal']) / max(1, len(normality_results)))
                }
            }
        
        # Outlier analysis
        if self.outlier_detection:
            outlier_results = []
            for i in range(n_features):
                z_scores = np.abs(stats.zscore(X[:, i]))
                outliers = z_scores > self.outlier_threshold
                outlier_results.append({
                    'feature': i,
                    'feature_name': self.feature_names_[i],
                    'outlier_count': np.sum(outliers),
                    'outlier_percentage': np.mean(outliers) * 100,
                    'outlier_indices': np.where(outliers)[0].tolist()
                })
            
            results['outlier_analysis'] = {
                'performed': True,
                'threshold': self.outlier_threshold,
                'results': outlier_results,
                'summary': {
                    'total_outliers': sum(r['outlier_count'] for r in outlier_results),
                    'outlier_rate': np.mean([r['outlier_percentage'] for r in outlier_results])
                }
            }
        
        # Class balance analysis
        if self.class_balance_analysis:
            class_counts = np.bincount(y)
            class_proportions = class_counts / len(y)
            
            results['class_balance'] = {
                'performed': True,
                'class_counts': class_counts.tolist(),
                'class_proportions': class_proportions.tolist(),
                'class_names': self.label_encoder_.classes_.tolist(),
                'balance_ratio': np.max(class_proportions) / np.min(class_proportions),
                'is_balanced': np.max(class_proportions) / np.min(class_proportions) < 3.0,
                'entropy': -np.sum(class_proportions * np.log2(class_proportions + 1e-10))
            }
        
        # Store results
        self.independence_test_results_ = results.get('independence_tests')
        self.normality_test_results_ = results.get('normality_tests')
        self.outlier_analysis_ = results.get('outlier_analysis')
        self.class_balance_analysis_ = results.get('class_balance')
    
    def _calculate_feature_statistics(self, X, y):
        """Calculate detailed feature statistics per class"""
        n_features = X.shape[1]
        n_classes = len(np.unique(y))
        
        self.feature_stats_ = {
            'class_means': {},
            'class_variances': {},
            'class_stds': {},
            'global_stats': {}
        }
        
        # Per-class statistics
        for class_idx in range(n_classes):
            class_mask = y == class_idx
            class_data = X[class_mask]
            
            class_name = self.label_encoder_.classes_[class_idx]
            
            self.feature_stats_['class_means'][class_name] = np.mean(class_data, axis=0)
            self.feature_stats_['class_variances'][class_name] = np.var(class_data, axis=0)
            self.feature_stats_['class_stds'][class_name] = np.std(class_data, axis=0)
        
        # Global statistics
        self.feature_stats_['global_stats'] = {
            'means': np.mean(X, axis=0),
            'variances': np.var(X, axis=0),
            'stds': np.std(X, axis=0),
            'mins': np.min(X, axis=0),
            'maxs': np.max(X, axis=0)
        }
        
        # Class statistics
        self.class_stats_ = {
            'class_priors': self.model_.class_prior_,
            'class_counts': self.model_.class_count_,
            'total_samples': len(y)
        }
    
    def get_model_analysis(self) -> Dict[str, Any]:
        """
        Get comprehensive analysis of the Naive Bayes model
        
        Returns:
        --------
        analysis_info : dict
            Detailed information about model characteristics and assumptions
        """
        if not self.is_fitted_:
            return {"status": "Not fitted"}
        
        analysis = {
            "algorithm_config": {
                "var_smoothing": self.var_smoothing,
                "custom_priors": self.priors is not None,
                "probability_calibration": self.calibrated_model_ is not None,
                "feature_scaling": self.scaler_ is not None
            },
            "model_parameters": {
                "n_features": self.n_features_in_,
                "n_classes": len(self.classes_),
                "class_names": self.classes_.tolist(),
                "feature_names": self.feature_names_
            }
        }
        
        # Add class statistics
        if self.class_stats_:
            analysis["class_statistics"] = {
                "class_priors": self.class_stats_['class_priors'].tolist(),
                "class_counts": self.class_stats_['class_counts'].tolist(),
                "total_samples": self.class_stats_['total_samples'],
                "class_balance": dict(zip(self.classes_, self.class_stats_['class_priors']))
            }
        
        # Add feature statistics summary
        if self.feature_stats_:
            analysis["feature_statistics"] = {
                "global_means": self.feature_stats_['global_stats']['means'].tolist(),
                "global_stds": self.feature_stats_['global_stats']['stds'].tolist(),
                "class_means_available": bool(self.feature_stats_['class_means']),
                "class_variances_available": bool(self.feature_stats_['class_variances'])
            }
        
        # Add assumption analysis results
        if self.independence_test_results_:
            independence = self.independence_test_results_
            analysis["independence_analysis"] = {
                "tests_performed": independence.get('performed', False),
                "violation_rate": independence.get('summary', {}).get('violation_rate', 0),
                "independence_assessment": self._assess_independence_violation(
                    independence.get('summary', {}).get('violation_rate', 0)
                )
            }
        
        if self.normality_test_results_:
            normality = self.normality_test_results_
            analysis["normality_analysis"] = {
                "tests_performed": normality.get('performed', False),
                "violation_rate": normality.get('summary', {}).get('violation_rate', 0),
                "normality_assessment": self._assess_normality_violation(
                    normality.get('summary', {}).get('violation_rate', 0)
                )
            }
        
        if self.outlier_analysis_:
            outliers = self.outlier_analysis_
            analysis["outlier_analysis"] = {
                "detection_performed": outliers.get('performed', False),
                "outlier_rate": outliers.get('summary', {}).get('outlier_rate', 0),
                "outlier_impact": self._assess_outlier_impact(
                    outliers.get('summary', {}).get('outlier_rate', 0)
                )
            }
        
        if self.class_balance_analysis_:
            balance = self.class_balance_analysis_
            analysis["class_balance_analysis"] = {
                "is_balanced": balance.get('is_balanced', True),
                "balance_ratio": balance.get('balance_ratio', 1.0),
                "entropy": balance.get('entropy', 0),
                "balance_assessment": self._assess_class_balance(
                    balance.get('balance_ratio', 1.0)
                )
            }
        
        return analysis
    
    def _assess_independence_violation(self, violation_rate):
        """Assess impact of independence assumption violation"""
        if violation_rate < 0.2:
            return "Low violation - Independence assumption reasonably satisfied"
        elif violation_rate < 0.5:
            return "Moderate violation - Some feature correlation, but NB should still work"
        else:
            return "High violation - Strong feature correlations may hurt performance"
    
    def _assess_normality_violation(self, violation_rate):
        """Assess impact of normality assumption violation"""
        if violation_rate < 0.3:
            return "Low violation - Features are approximately normal"
        elif violation_rate < 0.6:
            return "Moderate violation - Some non-normal distributions present"
        else:
            return "High violation - Many features deviate from normality"
    
    def _assess_outlier_impact(self, outlier_rate):
        """Assess impact of outliers on model"""
        if outlier_rate < 5:
            return "Low impact - Few outliers detected"
        elif outlier_rate < 15:
            return "Moderate impact - Some outliers may affect Gaussian estimates"
        else:
            return "High impact - Many outliers may distort class distributions"
    
    def _assess_class_balance(self, balance_ratio):
        """Assess class balance impact"""
        if balance_ratio < 2.0:
            return "Well balanced - Classes have similar representation"
        elif balance_ratio < 5.0:
            return "Moderately imbalanced - Consider adjusting priors"
        else:
            return "Highly imbalanced - Strong impact on performance expected"
    
    def plot_model_analysis(self, figsize=(16, 12)):
        """
        Create comprehensive Naive Bayes model analysis visualization
        
        Parameters:
        -----------
        figsize : tuple, default=(16, 12)
            Figure size
            
        Returns:
        --------
        fig : matplotlib figure
            Model analysis visualization
        """
        if not self.is_fitted_:
            return None
        
        # Determine number of subplots based on available analyses
        n_plots = 2  # Always have class distribution and feature means
        if self.feature_stats_ and len(self.feature_names_) <= 20:  # Feature distributions
            n_plots += 1
        if self.independence_test_results_:  # Independence heatmap
            n_plots += 1
        
        # Create subplots
        rows = (n_plots + 1) // 2
        fig, axes = plt.subplots(rows, 2, figsize=figsize)
        if rows == 1:
            axes = axes.reshape(1, -1)
        axes = axes.flatten()
        
        plot_idx = 0
        
        # 1. Class Distribution
        if self.class_stats_:
            ax = axes[plot_idx]
            class_counts = self.class_stats_['class_counts']
            class_names = [str(name) for name in self.classes_]
            
            bars = ax.bar(class_names, class_counts, alpha=0.7, color='skyblue', edgecolor='black')
            ax.set_xlabel('Class')
            ax.set_ylabel('Sample Count')
            ax.set_title('Class Distribution in Training Data')
            ax.grid(True, alpha=0.3)
            
            # Add count labels on bars
            for bar, count in zip(bars, class_counts):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + max(class_counts) * 0.01,
                       f'{int(count)}', ha='center', va='bottom')
            
            plot_idx += 1
        
        # 2. Feature Means by Class
        if self.feature_stats_ and self.feature_stats_['class_means']:
            ax = axes[plot_idx]
            
            # Select top features for visualization
            n_features_to_show = min(10, len(self.feature_names_))
            feature_indices = range(n_features_to_show)
            
            x = np.arange(n_features_to_show)
            width = 0.8 / len(self.classes_)
            
            for i, class_name in enumerate(self.classes_):
                means = self.feature_stats_['class_means'][class_name]
                offset = (i - len(self.classes_) / 2) * width + width / 2
                ax.bar(x + offset, means[feature_indices], width, 
                      label=str(class_name), alpha=0.7)
            
            ax.set_xlabel('Features')
            ax.set_ylabel('Mean Value')
            ax.set_title(f'Feature Means by Class (Top {n_features_to_show} Features)')
            ax.set_xticks(x)
            ax.set_xticklabels([self.feature_names_[i][:10] + '...' if len(self.feature_names_[i]) > 10 
                               else self.feature_names_[i] for i in feature_indices], rotation=45)
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            plot_idx += 1
        
        # 3. Feature Distributions (if not too many features)
        if (self.feature_stats_ and len(self.feature_names_) <= 20 and 
            self.training_data_ is not None and plot_idx < len(axes)):
            ax = axes[plot_idx]
            
            # Show distribution of first feature as example
            feature_idx = 0
            for class_idx, class_name in enumerate(self.classes_):
                class_mask = self.training_labels_ == class_name
                if hasattr(self.training_data_, 'iloc'):
                    feature_data = self.training_data_.iloc[class_mask, feature_idx]
                else:
                    feature_data = self.training_data_[class_mask, feature_idx]
                
                ax.hist(feature_data, alpha=0.6, label=str(class_name), bins=20)
            
            ax.set_xlabel(f'{self.feature_names_[feature_idx]} Value')
            ax.set_ylabel('Frequency')
            ax.set_title(f'Distribution of {self.feature_names_[feature_idx]} by Class')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            plot_idx += 1
        
        # 4. Feature Independence Heatmap
        if (self.independence_test_results_ and 
            self.independence_test_results_.get('performed') and 
            plot_idx < len(axes)):
            ax = axes[plot_idx]
            
            # Create correlation matrix
            independence_results = self.independence_test_results_['results']
            n_features = len(self.feature_names_)
            
            if n_features <= 20 and independence_results:  # Only for reasonable number of features
                corr_matrix = np.eye(n_features)
                
                for result in independence_results:
                    i, j = result['feature_pair']
                    corr = abs(result['correlation'])
                    corr_matrix[i, j] = corr
                    corr_matrix[j, i] = corr
                
                im = ax.imshow(corr_matrix, cmap='Reds', aspect='auto')
                ax.set_xticks(range(n_features))
                ax.set_yticks(range(n_features))
                ax.set_xticklabels([name[:8] + '...' if len(name) > 8 else name 
                                   for name in self.feature_names_], rotation=45)
                ax.set_yticklabels([name[:8] + '...' if len(name) > 8 else name 
                                   for name in self.feature_names_])
                ax.set_title('Feature Correlation Matrix\n(Independence Assumption Check)')
                
                # Add colorbar
                cbar = plt.colorbar(im, ax=ax)
                cbar.set_label('Absolute Correlation')
                
                plot_idx += 1
        
        # Hide unused subplots
        for i in range(plot_idx, len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        return fig
    
    def plot_feature_distributions(self, max_features=6, figsize=(15, 10)):
        """
        Plot feature distributions by class for detailed analysis
        
        Parameters:
        -----------
        max_features : int, default=6
            Maximum number of features to plot
        figsize : tuple, default=(15, 10)
            Figure size
            
        Returns:
        --------
        fig : matplotlib figure
            Feature distribution plots
        """
        if not self.is_fitted_ or self.training_data_ is None:
            return None
        
        n_features_to_plot = min(max_features, len(self.feature_names_))
        rows = (n_features_to_plot + 2) // 3
        cols = min(3, n_features_to_plot)
        
        fig, axes = plt.subplots(rows, cols, figsize=figsize)
        if rows == 1:
            axes = axes.reshape(1, -1)
        axes = axes.flatten()
        
        for i in range(n_features_to_plot):
            ax = axes[i]
            
            # Plot distribution for each class
            for class_name in self.classes_:
                class_mask = self.training_labels_ == class_name
                if hasattr(self.training_data_, 'iloc'):
                    feature_data = self.training_data_.iloc[class_mask, i]
                else:
                    feature_data = self.training_data_[class_mask, i]
                
                ax.hist(feature_data, alpha=0.6, label=str(class_name), bins=20, density=True)
                
                # Overlay fitted Gaussian
                if self.feature_stats_:
                    mean = self.feature_stats_['class_means'][class_name][i]
                    std = self.feature_stats_['class_stds'][class_name][i]
                    x_range = np.linspace(feature_data.min(), feature_data.max(), 100)
                    gaussian = stats.norm.pdf(x_range, mean, std)
                    ax.plot(x_range, gaussian, '--', linewidth=2, 
                           label=f'{class_name} Gaussian fit')
            
            ax.set_xlabel(f'{self.feature_names_[i]} Value')
            ax.set_ylabel('Density')
            ax.set_title(f'Distribution: {self.feature_names_[i]}')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        # Hide unused subplots
        for i in range(n_features_to_plot, len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        return fig
    
    def get_hyperparameter_config(self, key_prefix: str) -> Dict[str, Any]:
        """Generate Streamlit UI for hyperparameters"""
        
        st.sidebar.markdown("### 🎯 Gaussian Naive Bayes Configuration")
        
        # Create tabs for different parameter groups
        tab1, tab2, tab3, tab4, tab5 = st.sidebar.tabs(["Core", "Analysis", "Calibration", "Preprocessing", "Info"])
        
        with tab1:
            st.markdown("**Core Parameters**")
            
            # Variance smoothing
            var_smoothing = st.number_input(
                "Variance Smoothing:",
                value=float(self.var_smoothing),
                min_value=1e-12,
                max_value=1e-6,
                step=1e-10,
                format="%.2e",
                help="Numerical stability parameter - portion of largest variance added to all variances",
                key=f"{key_prefix}_var_smoothing"
            )
            
            # Custom priors
            use_custom_priors = st.checkbox(
                "Use Custom Class Priors",
                value=self.priors is not None,
                help="Specify custom prior probabilities for classes",
                key=f"{key_prefix}_use_custom_priors"
            )
            
            if use_custom_priors:
                st.info("Custom priors will be set based on training data class distribution")
                priors = "auto"  # Will be set during training
            else:
                priors = None
        
        with tab2:
            st.markdown("**Assumption Analysis**")
            
            # Feature independence test
            feature_independence_test = st.checkbox(
                "Test Feature Independence",
                value=self.feature_independence_test,
                help="Test the independence assumption between features",
                key=f"{key_prefix}_feature_independence_test"
            )
            
            # Normality test
            normality_test = st.checkbox(
                "Test Gaussian Distribution",
                value=self.normality_test,
                help="Test if features follow normal distribution within each class",
                key=f"{key_prefix}_normality_test"
            )
            
            # Class balance analysis
            class_balance_analysis = st.checkbox(
                "Analyze Class Balance",
                value=self.class_balance_analysis,
                help="Analyze class distribution and balance",
                key=f"{key_prefix}_class_balance_analysis"
            )
            
            # Outlier detection
            outlier_detection = st.checkbox(
                "Detect Outliers",
                value=self.outlier_detection,
                help="Detect outliers that may affect Gaussian parameter estimation",
                key=f"{key_prefix}_outlier_detection"
            )
            
            if outlier_detection:
                outlier_threshold = st.slider(
                    "Outlier Z-Score Threshold:",
                    min_value=2.0,
                    max_value=4.0,
                    value=float(self.outlier_threshold),
                    step=0.1,
                    help="Z-score threshold for outlier detection",
                    key=f"{key_prefix}_outlier_threshold"
                )
            else:
                outlier_threshold = self.outlier_threshold
        
        with tab3:
            st.markdown("**Probability Calibration**")
            
            # Probability calibration
            probability_calibration = st.checkbox(
                "Enable Probability Calibration",
                value=self.probability_calibration,
                help="Calibrate predicted probabilities for better reliability",
                key=f"{key_prefix}_probability_calibration"
            )
            
            if probability_calibration:
                calibration_method = st.selectbox(
                    "Calibration Method:",
                    options=['isotonic', 'sigmoid'],
                    index=['isotonic', 'sigmoid'].index(self.calibration_method),
                    help="isotonic: non-parametric, sigmoid: parametric (Platt scaling)",
                    key=f"{key_prefix}_calibration_method"
                )
                
                cv_folds = st.slider(
                    "Calibration CV Folds:",
                    min_value=3,
                    max_value=10,
                    value=int(self.cv_folds),
                    help="Number of folds for cross-validation calibration",
                    key=f"{key_prefix}_cv_folds"
                )
            else:
                calibration_method = self.calibration_method
                cv_folds = self.cv_folds
        
        with tab4:
            st.markdown("**Preprocessing Options**")
            
            # Feature scaling (usually not needed for NB)
            auto_scaling = st.checkbox(
                "Feature Scaling",
                value=self.auto_scaling,
                help="Usually not needed for Naive Bayes, but can help with outliers",
                key=f"{key_prefix}_auto_scaling"
            )
            
            if auto_scaling:
                scaling_method = st.selectbox(
                    "Scaling Method:",
                    options=['standard', 'minmax', 'robust'],
                    index=['standard', 'minmax', 'robust'].index(self.scaling_method),
                    help="standard: z-score, minmax: [0,1], robust: median/IQR",
                    key=f"{key_prefix}_scaling_method"
                )
            else:
                scaling_method = self.scaling_method
                
            if not auto_scaling:
                st.info("💡 Naive Bayes doesn't usually need feature scaling")
        
        with tab5:
            st.markdown("**Algorithm Information**")
            
            if SKLEARN_AVAILABLE:
                st.success("✅ scikit-learn is available")
            else:
                st.error("❌ scikit-learn not installed. Run: pip install scikit-learn")
            
            st.info("""
            **Gaussian Naive Bayes** - Fast Probabilistic Learning:
            • ⚡ Extremely fast training and prediction
            • 📊 Excellent probability estimates
            • 🎯 Works well with small datasets
            • 🔢 Assumes features are normally distributed
            • 🚫 Assumes feature independence
            • 📈 Great baseline algorithm
            
            **Key Advantages:**
            • No hyperparameters to tune
            • Naturally handles multi-class
            • Scale-invariant
            • Interpretable results
            """)
            
            # Algorithm mechanics
            if st.button("🧮 How Naive Bayes Works", key=f"{key_prefix}_how_it_works"):
                st.markdown("""
                **Naive Bayes Algorithm:**
                
                **Training Phase:**
                1. Calculate class priors: P(C_k) = count(C_k) / total
                2. For each feature and class, estimate μ and σ²
                3. Store Gaussian parameters (μ, σ²) for each feature-class pair
                
                **Prediction Phase:**
                1. For new instance x, calculate likelihood for each class:
                   P(x|C_k) = ∏ P(x_i|C_k) (independence assumption)
                2. P(x_i|C_k) = (1/√(2πσ²)) * exp(-(x_i-μ)²/(2σ²))
                3. Apply Bayes' theorem: P(C_k|x) ∝ P(x|C_k) * P(C_k)
                4. Predict class with highest posterior probability
                """)
            
            # Assumptions guide
            if st.button("📋 Key Assumptions", key=f"{key_prefix}_assumptions"):
                st.markdown("""
                **Naive Bayes Assumptions:**
                
                **1. Feature Independence:**
                - Features are conditionally independent given class
                - Reality: Rarely true, but algorithm often works anyway
                - Impact: Violation can reduce performance
                
                **2. Gaussian Distribution:**
                - Features follow normal distribution within each class
                - Reality: Many real features are not normal
                - Mitigation: Data transformation, other NB variants
                
                **3. Same Variance Assumption:**
                - All classes have same feature variance (in some implementations)
                - Gaussian NB allows different variances per class
                
                **Why It Still Works:**
                - Only needs correct probability ranking
                - Robust to moderate assumption violations
                - Large margins between classes help
                """)
            
            # Hyperparameter tuning guide
            if st.button("🎯 Tuning Strategy", key=f"{key_prefix}_tuning_strategy"):
                st.markdown("""
                **Naive Bayes Tuning Strategy:**
                
                **Step 1: Data Quality**
                - Check for outliers (can affect Gaussian estimates)
                - Test normality assumption (optional)
                - Handle missing values
                
                **Step 2: Core Parameters**
                - var_smoothing: Usually 1e-9 works well
                - Increase if getting numerical issues
                - Custom priors: Use for imbalanced data
                
                **Step 3: Advanced Options**
                - Probability calibration for better probabilities
                - Feature independence testing for diagnostics
                - Consider feature transformation if normality violated
                
                **Step 4: Alternative Variants**
                - Multinomial NB for count data
                - Bernoulli NB for binary features
                - Complement NB for imbalanced data
                """)
            
            if st.button("Show Algorithm Details", key=f"{key_prefix}_info"):
                info = self.get_algorithm_info()
                st.json(info)
        
        return {
            "var_smoothing": var_smoothing,
            "priors": priors,
            "feature_independence_test": feature_independence_test,
            "normality_test": normality_test,
            "class_balance_analysis": class_balance_analysis,
            "outlier_detection": outlier_detection,
            "outlier_threshold": outlier_threshold,
            "probability_calibration": probability_calibration,
            "calibration_method": calibration_method,
            "cv_folds": cv_folds,
            "auto_scaling": auto_scaling,
            "scaling_method": scaling_method,
            "_ui_options": {
                "show_model_analysis": True,
                "show_feature_distributions": True,
                "show_assumption_tests": feature_independence_test or normality_test,
                "analyze_assumptions": True
            }
        }
    
    def create_model_instance(self, hyperparameters: Dict[str, Any]):
        """Create model instance with given hyperparameters"""
        # Handle custom priors
        priors = hyperparameters.get("priors")
        if priors == "auto":
            priors = None  # Will be estimated from data
        
        return GaussianNaiveBayesPlugin(
            priors=priors,
            var_smoothing=hyperparameters.get("var_smoothing", self.var_smoothing),
            feature_independence_test=hyperparameters.get("feature_independence_test", self.feature_independence_test),
            normality_test=hyperparameters.get("normality_test", self.normality_test),
            class_balance_analysis=hyperparameters.get("class_balance_analysis", self.class_balance_analysis),
            outlier_detection=hyperparameters.get("outlier_detection", self.outlier_detection),
            outlier_threshold=hyperparameters.get("outlier_threshold", self.outlier_threshold),
            probability_calibration=hyperparameters.get("probability_calibration", self.probability_calibration),
            calibration_method=hyperparameters.get("calibration_method", self.calibration_method),
            cv_folds=hyperparameters.get("cv_folds", self.cv_folds),
            auto_scaling=hyperparameters.get("auto_scaling", self.auto_scaling),
            scaling_method=hyperparameters.get("scaling_method", self.scaling_method)
        )
    
    def preprocess_data(self, X, y=None, training=True):
        """
        Preprocess data for Gaussian Naive Bayes
        
        Naive Bayes typically requires minimal preprocessing.
        """
        if hasattr(X, 'copy'):
            X_processed = X.copy()
        else:
            X_processed = np.array(X, copy=True)
        
        # Naive Bayes handles missing values poorly
        if np.any(pd.isna(X_processed)):
            warnings.warn("Naive Bayes doesn't handle missing values well. Consider imputation.")
        
        if training and y is not None:
            if hasattr(y, 'copy'):
                y_processed = y.copy()
            else:
                y_processed = np.array(y, copy=True)
            return X_processed, y_processed
        
        return X_processed
    
    def is_compatible_with_data(self, X, y=None) -> Tuple[bool, str]:
        """
        Check if Gaussian Naive Bayes is compatible with the given data
        
        Returns:
        --------
        compatible : bool
            Whether the algorithm is compatible
        message : str
            Explanation message
        """
        if not SKLEARN_AVAILABLE:
            return False, "scikit-learn is not installed. Install with: pip install scikit-learn"
        
        # Check minimum samples
        if X.shape[0] < self._min_samples_required:
            return False, f"Gaussian Naive Bayes requires at least {self._min_samples_required} samples, got {X.shape[0]}"
        
        # Check for missing values
        if np.any(pd.isna(X)):
            return False, "Gaussian Naive Bayes doesn't handle missing values well. Please impute missing values first."
        
        # Check for non-numeric data
        try:
            X_numeric = np.array(X, dtype=float)
        except (ValueError, TypeError):
            return False, "Gaussian Naive Bayes requires numeric features only."
        
        # Check for classification targets
        if y is not None:
            unique_values = np.unique(y)
            if len(unique_values) < 2:
                return False, "Need at least 2 classes for classification"
            
            # Check for sufficient samples per class
            min_samples_per_class = 3
            for class_val in unique_values:
                class_count = np.sum(y == class_val)
                if class_count < min_samples_per_class:
                    return True, f"Warning: Class '{class_val}' has only {class_count} samples. Consider more data."
        
        return True, "Gaussian Naive Bayes is compatible with this data"
    
    def get_model_params(self) -> Dict[str, Any]:
        """Get current model parameters"""
        if not self.is_fitted_:
            return {}
        
        return {
            "n_features": self.n_features_in_,
            "n_classes": len(self.classes_) if self.classes_ is not None else None,
            "feature_names": self.feature_names_,
            "var_smoothing": self.var_smoothing,
            "class_priors": self.model_.class_prior_.tolist() if hasattr(self.model_, 'class_prior_') else None,
            "probability_calibration": self.calibrated_model_ is not None,
            "scaling_applied": self.scaler_ is not None,
            "scaling_method": self.scaling_method if self.scaler_ is not None else None
        }
    
    def get_training_info(self) -> Dict[str, Any]:
        """Get information about the training process"""
        if not self.is_fitted_:
            return {"status": "Not fitted"}
        
        info = {
            "algorithm": "Gaussian Naive Bayes",
            "training_completed": True,
            "naive_bayes_characteristics": {
                "probabilistic": True,
                "fast_training": True,
                "fast_prediction": True,
                "feature_independence_assumption": True,
                "gaussian_assumption": True,
                "handles_small_data": True,
                "naturally_multiclass": True
            },
            "model_configuration": {
                "var_smoothing": self.var_smoothing,
                "custom_priors": self.priors is not None,
                "probability_calibration": self.calibrated_model_ is not None,
                "feature_scaling": self.scaler_ is not None
            },
            "model_analysis": self.get_model_analysis(),
            "performance_characteristics": {
                "training_complexity": "O(n*d) - linear in samples and features",
                "prediction_complexity": "O(d*k) - linear in features and classes",
                "memory_usage": "O(d*k) - stores means and variances",
                "scalability": "Excellent for large datasets",
                "parameter_free": "Almost no hyperparameters to tune"
            },
            "assumptions_status": {
                "independence_tested": self.independence_test_results_ is not None,
                "normality_tested": self.normality_test_results_ is not None,
                "outliers_analyzed": self.outlier_analysis_ is not None,
                "class_balance_analyzed": self.class_balance_analysis_ is not None
            }
        }
        
        return info

    def get_algorithm_specific_metrics(self, y_true: Optional[np.ndarray] = None, y_pred: Optional[np.ndarray] = None, y_proba: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        Retrieve algorithm-specific metrics for Gaussian Naive Bayes.

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
            return {"error": "Model not fitted. Cannot retrieve Gaussian Naive Bayes specific metrics."}

        metrics = {}
        prefix = "gnb_" # Gaussian Naive Bayes

        # Core model parameters
        metrics[f"{prefix}var_smoothing"] = self.var_smoothing
        if hasattr(self.model_, 'n_features_in_'):
            metrics[f"{prefix}num_features_in_model"] = self.model_.n_features_in_
        if self.classes_ is not None:
            metrics[f"{prefix}num_classes"] = len(self.classes_)

        # From class_stats_
        if self.class_stats_ and 'class_priors' in self.class_stats_:
            class_priors = np.array(self.class_stats_['class_priors'])
            metrics[f"{prefix}mean_class_prior"] = float(np.mean(class_priors))
            metrics[f"{prefix}std_class_prior"] = float(np.std(class_priors))
            # Individual class priors can be many, so mean/std is a summary.
            # For specific priors, one would look at get_model_params or get_model_analysis

        # From feature_stats_ (summaries of means and variances)
        if self.feature_stats_:
            if 'class_means' in self.feature_stats_ and self.feature_stats_['class_means']:
                # Calculate overall mean of feature means across all classes and features
                all_means = np.concatenate(list(self.feature_stats_['class_means'].values()))
                if all_means.size > 0:
                    metrics[f"{prefix}overall_mean_of_feature_means"] = float(np.mean(all_means))
            
            if 'class_variances' in self.feature_stats_ and self.feature_stats_['class_variances']:
                # Calculate overall mean of feature variances
                all_vars = np.concatenate(list(self.feature_stats_['class_variances'].values()))
                if all_vars.size > 0:
                    metrics[f"{prefix}overall_mean_of_feature_variances"] = float(np.mean(all_vars))

        # From assumption analysis results
        if self.independence_test_results_ and self.independence_test_results_.get('performed'):
            summary = self.independence_test_results_.get('summary', {})
            metrics[f"{prefix}independence_violation_rate"] = summary.get('violation_rate')

        if self.normality_test_results_ and self.normality_test_results_.get('performed'):
            summary = self.normality_test_results_.get('summary', {})
            metrics[f"{prefix}normality_violation_rate"] = summary.get('violation_rate')
            
        if self.outlier_analysis_ and self.outlier_analysis_.get('performed'):
            summary = self.outlier_analysis_.get('summary', {})
            metrics[f"{prefix}outlier_rate"] = summary.get('outlier_rate')

        if self.class_balance_analysis_ and self.class_balance_analysis_.get('performed'):
            metrics[f"{prefix}class_balance_ratio"] = self.class_balance_analysis_.get('balance_ratio')
            metrics[f"{prefix}class_balance_entropy"] = self.class_balance_analysis_.get('entropy')
            
        # McFadden's Pseudo R-squared
        if y_true is not None and y_proba is not None and self.label_encoder_ is not None and self.classes_ is not None:
            try:
                y_true_encoded = self.label_encoder_.transform(y_true)
                n_samples = len(y_true_encoded)
                n_classes_model = len(self.classes_)

                clipped_proba = np.clip(y_proba, 1e-15, 1 - 1e-15)
                log_likelihoods_model = np.log(clipped_proba[np.arange(n_samples), y_true_encoded])
                ll_model = np.sum(log_likelihoods_model)

                class_counts = np.bincount(y_true_encoded, minlength=n_classes_model)
                class_probas_null = class_counts / n_samples
                
                ll_null = 0
                for k_idx in range(n_classes_model):
                    if class_counts[k_idx] > 0 and class_probas_null[k_idx] > 0:
                         ll_null += class_counts[k_idx] * np.log(np.clip(class_probas_null[k_idx], 1e-15, 1))
                
                if ll_null == 0:
                    metrics[f"{prefix}mcfaddens_pseudo_r2"] = 1.0 if ll_model == 0 else 0.0
                elif ll_model > ll_null:
                     metrics[f"{prefix}mcfaddens_pseudo_r2"] = 0.0
                else:
                    metrics[f"{prefix}mcfaddens_pseudo_r2"] = float(1 - (ll_model / ll_null))
            except Exception as e:
                metrics[f"{prefix}mcfaddens_pseudo_r2_error"] = str(e)

        if not metrics:
            metrics['info'] = "No specific Gaussian Naive Bayes metrics were available (e.g., model not fitted, analysis not performed, or y_true/y_proba not provided for Pseudo R2)."
            
        return metrics


# Factory function
def get_plugin():
    """Factory function to get the plugin instance"""
    return GaussianNaiveBayesPlugin()