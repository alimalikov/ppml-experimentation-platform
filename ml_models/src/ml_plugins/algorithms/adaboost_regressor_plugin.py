import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, Optional, Tuple, List, Union
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_X_y, check_array
from sklearn.ensemble import AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score, validation_curve, learning_curve
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

# Import for plugin system
try:
    from src.ml_plugins.base_ml_plugin import MLPlugin
except ImportError:
    import sys
    import os
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))
    sys.path.append(project_root)
    from src.ml_plugins.base_ml_plugin import MLPlugin


class AdaBoostRegressorPlugin(BaseEstimator, RegressorMixin, MLPlugin):
    """
    AdaBoost Regressor Plugin - Adaptive Boosting for Regression
    
    AdaBoost (Adaptive Boosting) is an ensemble method that combines multiple 
    weak learners (typically decision trees) by sequentially training them,
    where each new learner focuses on the errors made by previous learners.
    
    Key Features:
    - 🎯 Adaptive Learning: Each model focuses on previous errors
    - 🌳 Weak Learner Combination: Combines simple models into strong predictor
    - 📈 Sequential Training: Models trained one after another
    - ⚖️ Weighted Predictions: Combines predictions with learned weights
    - 🎨 Versatile Base Estimators: Works with various weak learners
    - 🛡️ Robust to Overfitting: Generally less prone to overfitting than individual trees
    - 📊 Feature Importance: Provides feature importance through base estimators
    - 🔄 Iterative Improvement: Performance improves with each iteration
    - 🎛️ Loss Function Flexibility: Multiple loss functions available
    - 📉 Error Reduction: Systematically reduces prediction errors
    """
    
    def __init__(
        self,
        # Core AdaBoost parameters
        estimator=None,                # Base estimator (default: DecisionTreeRegressor)
        n_estimators=50,               # Number of estimators in ensemble
        learning_rate=1.0,             # Learning rate shrinks contribution of each estimator
        loss='linear',                 # Loss function ('linear', 'square', 'exponential')
        
        # Base estimator parameters (for DecisionTree)
        max_depth=3,                   # Maximum depth of base estimators
        min_samples_split=2,           # Minimum samples to split
        min_samples_leaf=1,            # Minimum samples in leaf
        
        # Advanced parameters
        random_state=None,             # Random state for reproducibility
        
        # Data preprocessing
        auto_scale=False,              # Automatic feature scaling (usually not needed for trees)
        scaler_type='standard',        # 'standard', 'minmax', 'robust'
        
        # Analysis options
        compute_feature_importance=True,
        boosting_analysis=True,
        base_estimator_analysis=True,
        learning_curve_analysis=True,
        cross_validation_analysis=True,
        
        # Advanced analysis
        loss_function_analysis=True,
        estimator_comparison_analysis=True,
        convergence_analysis=True,
        prediction_variance_analysis=True,
        error_evolution_analysis=True,
        
        # Visualization options
        plot_boosting_stages=True,
        plot_feature_importance=True,
        plot_learning_curves=True,
        plot_prediction_intervals=True,
        max_plots=5,
        
        # Performance monitoring
        cv_folds=5,
        memory_efficient=True,
        
        # Comparison analysis
        compare_with_single_estimator=True,
        compare_with_random_forest=True,
        performance_profiling=True
    ):
        super().__init__()
        
        # Core AdaBoost parameters
        self.estimator = estimator
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.loss = loss
        
        # Base estimator parameters
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        
        # Advanced parameters
        self.random_state = random_state
        
        # Data preprocessing
        self.auto_scale = auto_scale
        self.scaler_type = scaler_type
        
        # Analysis options
        self.compute_feature_importance = compute_feature_importance
        self.boosting_analysis = boosting_analysis
        self.base_estimator_analysis = base_estimator_analysis
        self.learning_curve_analysis = learning_curve_analysis
        self.cross_validation_analysis = cross_validation_analysis
        
        # Advanced analysis
        self.loss_function_analysis = loss_function_analysis
        self.estimator_comparison_analysis = estimator_comparison_analysis
        self.convergence_analysis = convergence_analysis
        self.prediction_variance_analysis = prediction_variance_analysis
        self.error_evolution_analysis = error_evolution_analysis
        
        # Visualization options
        self.plot_boosting_stages = plot_boosting_stages
        self.plot_feature_importance = plot_feature_importance
        self.plot_learning_curves = plot_learning_curves
        self.plot_prediction_intervals = plot_prediction_intervals
        self.max_plots = max_plots
        
        # Performance monitoring
        self.cv_folds = cv_folds
        self.memory_efficient = memory_efficient
        
        # Comparison analysis
        self.compare_with_single_estimator = compare_with_single_estimator
        self.compare_with_random_forest = compare_with_random_forest
        self.performance_profiling = performance_profiling
        
        # Required plugin metadata
        self._name = "AdaBoost Regressor"
        self._description = "Adaptive boosting ensemble method for regression"
        self._category = "Ensemble"
        
        # Required capability flags - THESE ARE ESSENTIAL!
        self._supports_classification = False
        self._supports_regression = True
        self._min_samples_required = 20
        
        # Internal state
        self.is_fitted_ = False
        self.model_ = None
        self.scaler_ = None
        self.feature_names_ = None
        self.n_features_in_ = None
        self.X_train_scaled_ = None
        self.y_train_ = None
        self.base_estimator_ = None
        
        # Analysis results storage
        self.feature_importance_analysis_ = {}
        self.boosting_analysis_ = {}
        self.base_estimator_analysis_ = {}
        self.learning_curve_analysis_ = {}
        self.cross_validation_analysis_ = {}
        self.loss_function_analysis_ = {}
        self.estimator_comparison_analysis_ = {}
        self.convergence_analysis_ = {}
        self.prediction_variance_analysis_ = {}
        self.error_evolution_analysis_ = {}
        self.performance_profile_ = {}

    def get_name(self) -> str:
        """Get the plugin name"""
        return self._name
    
    def get_description(self) -> str:
        """Get the plugin description"""
        return self._description
    
    def get_category(self) -> str:
        """Get the plugin category"""
        return self._category
    
    def supports_task(self, task_type: str) -> bool:
        """
        Check if the plugin supports a specific task type
        
        Parameters:
        -----------
        task_type : str
            The task type to check ('regression', 'classification', 'clustering', etc.)
        
        Returns:
        --------
        bool
            True if the task type is supported, False otherwise
        """
        task_type = task_type.lower()
        
        if task_type in ['regression', 'regressor']:
            return self._supports_regression
        elif task_type in ['classification', 'classifier']:
            return self._supports_classification
        else:
            return False

    def create_model_instance(self, **kwargs):
        """
        Create a new instance of the AdaBoostRegressor model
        
        Parameters:
        -----------
        **kwargs : dict
            Additional parameters to override default settings
        
        Returns:
        --------
        AdaBoostRegressor
            A new instance of the sklearn AdaBoostRegressor
        """
        # Create base estimator if not provided
        base_estimator = kwargs.get('estimator', self.estimator)
        if base_estimator is None:
            # Create default DecisionTreeRegressor with current parameters
            base_estimator = DecisionTreeRegressor(
                max_depth=kwargs.get('max_depth', self.max_depth),
                min_samples_split=kwargs.get('min_samples_split', self.min_samples_split),
                min_samples_leaf=kwargs.get('min_samples_leaf', self.min_samples_leaf),
                random_state=kwargs.get('random_state', self.random_state)
            )
        
        # Create AdaBoost model parameters
        model_params = {
            'estimator': base_estimator,
            'n_estimators': kwargs.get('n_estimators', self.n_estimators),
            'learning_rate': kwargs.get('learning_rate', self.learning_rate),
            'loss': kwargs.get('loss', self.loss),
            'random_state': kwargs.get('random_state', self.random_state)
        }
        
        return AdaBoostRegressor(**model_params)

    def get_hyperparameter_config(self, unique_key_prefix=None) -> Dict[str, Any]:
        """
        Get the hyperparameter configuration for this algorithm
        
        Parameters:
        -----------
        unique_key_prefix : str, optional
            Prefix for unique keys in Streamlit components (unused but required by interface)
        
        Returns:
        --------
        Dict[str, Any]
            Dictionary containing hyperparameter definitions with their types,
            ranges, and default values for optimization
        """
        return {
            'n_estimators': {
                'type': 'int',
                'range': [10, 500],
                'default': 50,
                'description': 'Number of estimators in the ensemble',
                'scale': 'linear'
            },
            'learning_rate': {
                'type': 'float',
                'range': [0.01, 2.0],
                'default': 1.0,
                'description': 'Learning rate shrinks the contribution of each estimator',
                'scale': 'log'
            },
            'loss': {
                'type': 'categorical',
                'choices': ['linear', 'square', 'exponential'],
                'default': 'linear',
                'description': 'Loss function to use when updating weights'
            },
            'max_depth': {
                'type': 'int',
                'range': [1, 20],
                'default': 3,
                'description': 'Maximum depth of the base decision tree estimators',
                'conditional': {'estimator': 'default'}
            },
            'min_samples_split': {
                'type': 'int',
                'range': [2, 50],
                'default': 2,
                'description': 'Minimum samples required to split an internal node',
                'conditional': {'estimator': 'default'}
            },
            'min_samples_leaf': {
                'type': 'int',
                'range': [1, 20],
                'default': 1,
                'description': 'Minimum samples required to be at a leaf node',
                'conditional': {'estimator': 'default'}
            },
            'auto_scale': {
                'type': 'boolean',
                'default': False,
                'description': 'Whether to automatically scale features (usually not needed for tree-based methods)'
            },
            'scaler_type': {
                'type': 'categorical',
                'choices': ['standard', 'minmax', 'robust'],
                'default': 'standard',
                'description': 'Type of feature scaling to apply',
                'conditional': {'auto_scale': True}
            },
            'random_state': {
                'type': 'int',
                'range': [0, 1000],
                'default': 42,
                'description': 'Random state for reproducibility (optional)'
            }
        }
        
    def fit(self, X, y, sample_weight=None):
        """
        Fit the AdaBoost Regressor with comprehensive boosting analysis
        
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
        
        # Handle data preprocessing
        if self.auto_scale:
            self.scaler_ = self._create_scaler()
            X_scaled = self.scaler_.fit_transform(X)
        else:
            X_scaled = X.copy()
            self.scaler_ = None
        
        # Store scaled training data
        self.X_train_scaled_ = X_scaled
        self.y_train_ = y
        
        # Create base estimator if not provided
        if self.estimator is None:
            self.base_estimator_ = DecisionTreeRegressor(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
                random_state=self.random_state
            )
        else:
            self.base_estimator_ = self.estimator
        
        # Create and configure the AdaBoost model
        self.model_ = AdaBoostRegressor(
            estimator=self.base_estimator_,
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            loss=self.loss,
            random_state=self.random_state
        )
        
        # Fit the model
        if sample_weight is not None:
            self.model_.fit(X_scaled, y, sample_weight=sample_weight)
        else:
            self.model_.fit(X_scaled, y)
        
        # Perform comprehensive analysis
        if self.boosting_analysis:
            self._analyze_boosting_process()
        
        if self.base_estimator_analysis:
            self._analyze_base_estimators()
        
        if self.compute_feature_importance:
            self._analyze_feature_importance()
        
        if self.convergence_analysis:
            self._analyze_convergence()
        
        if self.error_evolution_analysis:
            self._analyze_error_evolution()
        
        if self.loss_function_analysis:
            self._analyze_loss_function_impact()
        
        if self.prediction_variance_analysis:
            self._analyze_prediction_variance()
        
        if self.learning_curve_analysis:
            self._analyze_learning_curves()
        
        if self.cross_validation_analysis:
            self._analyze_cross_validation()
        
        if self.estimator_comparison_analysis:
            self._analyze_estimator_comparison()
        
        if self.performance_profiling:
            self._profile_performance()
        
        if self.compare_with_single_estimator:
            self._compare_with_single_estimator()
        
        if self.compare_with_random_forest:
            self._compare_with_random_forest()
        
        self.is_fitted_ = True
        return self
    
    def predict(self, X):
        """
        Make predictions using the fitted AdaBoost model
        
        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            Input data for prediction
        
        Returns:
        --------
        y_pred : array of shape (n_samples,)
            Predicted values
        """
        if not self.is_fitted_:
            raise ValueError("Model must be fitted before making predictions")
        
        X = check_array(X, accept_sparse=False)
        
        # Apply same scaling as training data if scaler was used
        if self.scaler_ is not None:
            X_scaled = self.scaler_.transform(X)
        else:
            X_scaled = X
        
        # Make predictions
        predictions = self.model_.predict(X_scaled)
        
        return predictions
    
    def predict_with_stages(self, X):
        """
        Make predictions with detailed stage-wise information
        
        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            Input data for prediction
        
        Returns:
        --------
        results : dict
            Dictionary containing predictions and stage analysis
        """
        if not self.is_fitted_:
            raise ValueError("Model must be fitted before making predictions")
        
        X = check_array(X, accept_sparse=False)
        
        # Apply scaling if used during training
        if self.scaler_ is not None:
            X_scaled = self.scaler_.transform(X)
        else:
            X_scaled = X
        
        # Get final predictions
        final_predictions = self.model_.predict(X_scaled)
        
        # Get staged predictions (predictions at each boosting stage)
        staged_predictions = list(self.model_.staged_predict(X_scaled))
        
        # Calculate prediction confidence based on stage convergence
        confidence_scores = self._calculate_prediction_confidence(staged_predictions)
        
        # Calculate prediction variance across stages
        prediction_variance = self._calculate_prediction_variance(staged_predictions)
        
        # Identify stable vs unstable predictions
        stability_scores = self._calculate_prediction_stability(staged_predictions)
        
        # Get estimator weights
        estimator_weights = self.model_.estimator_weights_
        
        # Calculate weighted contribution of each estimator
        estimator_contributions = self._calculate_estimator_contributions(X_scaled, estimator_weights)
        
        return {
            'predictions': final_predictions,
            'staged_predictions': staged_predictions,
            'confidence_scores': confidence_scores,
            'prediction_variance': prediction_variance,
            'stability_scores': stability_scores,
            'estimator_weights': estimator_weights,
            'estimator_contributions': estimator_contributions,
            'n_estimators_used': len(estimator_weights),
            'convergence_info': {
                'final_stage': len(staged_predictions),
                'mean_confidence': np.mean(confidence_scores),
                'mean_stability': np.mean(stability_scores),
                'prediction_consistency': 1.0 - np.mean(prediction_variance)
            }
        }

    def _create_scaler(self):
        """Create appropriate scaler based on scaler_type"""
        if self.scaler_type == 'standard':
            return StandardScaler()
        elif self.scaler_type == 'minmax':
            return MinMaxScaler()
        elif self.scaler_type == 'robust':
            return RobustScaler()
        else:
            raise ValueError(f"Unknown scaler type: {self.scaler_type}")  
        
    def _calculate_prediction_confidence(self, staged_predictions):
        """
        Calculate prediction confidence based on stage convergence
        
        Parameters:
        -----------
        staged_predictions : list
            List of predictions at each boosting stage
        
        Returns:
        --------
        confidence_scores : array
            Confidence scores for each prediction
        """
        try:
            staged_predictions = np.array(staged_predictions)
            n_stages, n_samples = staged_predictions.shape
            
            confidence_scores = []
            
            for sample_idx in range(n_samples):
                sample_predictions = staged_predictions[:, sample_idx]
                
                # Calculate convergence stability (how much predictions change in later stages)
                if n_stages > 10:
                    # Look at variance in last 25% of stages
                    last_quarter = sample_predictions[-max(1, n_stages//4):]
                    convergence_variance = np.var(last_quarter)
                    
                    # Lower variance = higher confidence
                    confidence = 1.0 / (1.0 + convergence_variance)
                else:
                    # For fewer stages, use overall variance
                    confidence = 1.0 / (1.0 + np.var(sample_predictions))
                
                confidence_scores.append(confidence)
            
            return np.array(confidence_scores)
            
        except Exception as e:
            # Fallback: uniform confidence
            return np.ones(len(staged_predictions[0]) if staged_predictions else 1) * 0.5

    def _calculate_prediction_variance(self, staged_predictions):
        """
        Calculate prediction variance across boosting stages
        
        Parameters:
        -----------
        staged_predictions : list
            List of predictions at each boosting stage
        
        Returns:
        --------
        variance_scores : array
            Variance scores for each prediction
        """
        try:
            staged_predictions = np.array(staged_predictions)
            
            # Calculate variance across stages for each sample
            variance_scores = np.var(staged_predictions, axis=0)
            
            return variance_scores
            
        except Exception as e:
            # Fallback: zero variance
            return np.zeros(len(staged_predictions[0]) if staged_predictions else 1)

    def _calculate_prediction_stability(self, staged_predictions):
        """
        Calculate prediction stability based on how much predictions change across stages
        
        Parameters:
        -----------
        staged_predictions : list
            List of predictions at each boosting stage
        
        Returns:
        --------
        stability_scores : array
            Stability scores for each prediction (higher = more stable)
        """
        try:
            staged_predictions = np.array(staged_predictions)
            n_stages, n_samples = staged_predictions.shape
            
            if n_stages < 2:
                return np.ones(n_samples)
            
            stability_scores = []
            
            for sample_idx in range(n_samples):
                sample_predictions = staged_predictions[:, sample_idx]
                
                # Calculate the coefficient of variation (std/mean) as instability measure
                mean_pred = np.mean(sample_predictions)
                std_pred = np.std(sample_predictions)
                
                if abs(mean_pred) > 1e-8:  # Avoid division by zero
                    coefficient_of_variation = std_pred / abs(mean_pred)
                    # Convert to stability score (lower CV = higher stability)
                    stability = 1.0 / (1.0 + coefficient_of_variation)
                else:
                    # If mean is near zero, use inverse of std
                    stability = 1.0 / (1.0 + std_pred)
                
                stability_scores.append(stability)
            
            return np.array(stability_scores)
            
        except Exception as e:
            # Fallback: uniform stability
            return np.ones(len(staged_predictions[0]) if staged_predictions else 1) * 0.5

    def _calculate_estimator_contributions(self, X_scaled, estimator_weights):
        """
        Calculate the weighted contribution of each base estimator to the final predictions
        
        Parameters:
        -----------
        X_scaled : array
            Scaled input features
        estimator_weights : array
            Weights of each estimator in the ensemble
        
        Returns:
        --------
        contributions : dict
            Dictionary containing contribution analysis
        """
        try:
            # Get individual estimator predictions
            estimator_predictions = []
            
            for estimator in self.model_.estimators_:
                pred = estimator.predict(X_scaled)
                estimator_predictions.append(pred)
            
            estimator_predictions = np.array(estimator_predictions)
            
            # Calculate weighted contributions
            weighted_contributions = estimator_predictions * estimator_weights[:, np.newaxis]
            
            # Calculate relative importance of each estimator
            total_weight = np.sum(np.abs(estimator_weights))
            relative_importance = np.abs(estimator_weights) / total_weight if total_weight > 0 else np.ones_like(estimator_weights) / len(estimator_weights)
            
            # Calculate contribution variance (how much each estimator varies in its contribution)
            contribution_variance = np.var(weighted_contributions, axis=1)
            
            # Identify most and least influential estimators
            most_influential_idx = np.argmax(np.abs(estimator_weights))
            least_influential_idx = np.argmin(np.abs(estimator_weights))
            
            return {
                'individual_predictions': estimator_predictions,
                'weighted_contributions': weighted_contributions,
                'relative_importance': relative_importance,
                'contribution_variance': contribution_variance,
                'total_contribution': np.sum(weighted_contributions, axis=0),
                'estimator_statistics': {
                    'most_influential_estimator': most_influential_idx,
                    'least_influential_estimator': least_influential_idx,
                    'weight_distribution': {
                        'mean': np.mean(estimator_weights),
                        'std': np.std(estimator_weights),
                        'min': np.min(estimator_weights),
                        'max': np.max(estimator_weights),
                        'positive_weights': np.sum(estimator_weights > 0),
                        'negative_weights': np.sum(estimator_weights < 0)
                    }
                },
                'diversity_metrics': {
                    'prediction_diversity': np.mean(np.std(estimator_predictions, axis=0)),
                    'weight_entropy': self._calculate_weight_entropy(estimator_weights),
                    'contribution_balance': self._calculate_contribution_balance(weighted_contributions)
                }
            }
            
        except Exception as e:
            return {
                'error': str(e),
                'individual_predictions': [],
                'weighted_contributions': [],
                'relative_importance': [],
                'contribution_variance': [],
                'total_contribution': np.zeros(len(X_scaled))
            }

    def _calculate_weight_entropy(self, weights):
        """Calculate entropy of estimator weights to measure diversity"""
        try:
            # Normalize weights to probabilities
            abs_weights = np.abs(weights)
            if np.sum(abs_weights) > 0:
                probabilities = abs_weights / np.sum(abs_weights)
                # Calculate entropy
                entropy = -np.sum(probabilities * np.log2(probabilities + 1e-10))
                return entropy
            else:
                return 0.0
        except Exception:
            return 0.0

    def _calculate_contribution_balance(self, weighted_contributions):
        """Calculate how balanced the contributions are across estimators"""
        try:
            # Calculate the coefficient of variation of absolute contributions
            abs_contributions = np.abs(weighted_contributions)
            mean_contributions = np.mean(abs_contributions, axis=1)
            
            if len(mean_contributions) > 1:
                balance = 1.0 / (1.0 + np.std(mean_contributions) / (np.mean(mean_contributions) + 1e-10))
                return balance
            else:
                return 1.0
        except Exception:
            return 0.5    
        
    def _analyze_feature_importance(self):
        """
        Analyze feature importance from the ensemble of base estimators
        """
        try:
            # Initialize feature importance analysis
            self.feature_importance_analysis_ = {
                'individual_estimator_importance': [],
                'ensemble_importance': None,
                'importance_consistency': None,
                'feature_stability': None,
                'importance_evolution': [],
                'top_features': [],
                'feature_rankings': [],
                'statistical_significance': {}
            }
            
            # Get feature importance from each base estimator
            individual_importances = []
            for i, estimator in enumerate(self.model_.estimators_):
                if hasattr(estimator, 'feature_importances_'):
                    importance = estimator.feature_importances_
                    individual_importances.append(importance)
                    self.feature_importance_analysis_['individual_estimator_importance'].append({
                        'estimator_index': i,
                        'importance': importance,
                        'weight': self.model_.estimator_weights_[i] if i < len(self.model_.estimator_weights_) else 0.0
                    })
            
            if individual_importances:
                individual_importances = np.array(individual_importances)
                
                # Calculate weighted ensemble importance
                estimator_weights = self.model_.estimator_weights_[:len(individual_importances)]
                weighted_importances = individual_importances * estimator_weights[:, np.newaxis]
                ensemble_importance = np.sum(weighted_importances, axis=0)
                
                # Normalize to sum to 1
                if np.sum(ensemble_importance) > 0:
                    ensemble_importance = ensemble_importance / np.sum(ensemble_importance)
                
                self.feature_importance_analysis_['ensemble_importance'] = ensemble_importance
                
                # Calculate importance consistency (how much individual estimators agree)
                importance_std = np.std(individual_importances, axis=0)
                importance_mean = np.mean(individual_importances, axis=0)
                consistency = 1.0 - (importance_std / (importance_mean + 1e-10))
                self.feature_importance_analysis_['importance_consistency'] = consistency
                
                # Calculate feature stability (coefficient of variation)
                cv = importance_std / (importance_mean + 1e-10)
                stability = 1.0 / (1.0 + cv)
                self.feature_importance_analysis_['feature_stability'] = stability
                
                # Track importance evolution through boosting stages
                cumulative_importance = np.zeros(len(ensemble_importance))
                cumulative_weight = 0.0
                
                for i, (importance, weight) in enumerate(zip(individual_importances, estimator_weights)):
                    cumulative_importance += importance * weight
                    cumulative_weight += weight
                    
                    if cumulative_weight > 0:
                        normalized_importance = cumulative_importance / cumulative_weight
                    else:
                        normalized_importance = cumulative_importance
                    
                    self.feature_importance_analysis_['importance_evolution'].append({
                        'stage': i + 1,
                        'cumulative_importance': normalized_importance.copy(),
                        'stage_contribution': importance * weight,
                        'cumulative_weight': cumulative_weight
                    })
                
                # Identify top features
                top_indices = np.argsort(ensemble_importance)[::-1]
                self.feature_importance_analysis_['top_features'] = [
                    {
                        'feature_name': self.feature_names_[idx],
                        'feature_index': idx,
                        'importance': ensemble_importance[idx],
                        'consistency': consistency[idx],
                        'stability': stability[idx],
                        'rank': rank + 1
                    }
                    for rank, idx in enumerate(top_indices)
                ]
                
                # Calculate feature rankings across estimators
                feature_rankings = []
                for importance in individual_importances:
                    ranking = np.argsort(np.argsort(importance)[::-1])
                    feature_rankings.append(ranking)
                
                self.feature_importance_analysis_['feature_rankings'] = np.array(feature_rankings)
                
                # Statistical significance testing (simplified)
                self.feature_importance_analysis_['statistical_significance'] = {
                    'mean_importance': np.mean(individual_importances, axis=0),
                    'std_importance': np.std(individual_importances, axis=0),
                    'min_importance': np.min(individual_importances, axis=0),
                    'max_importance': np.max(individual_importances, axis=0),
                    'significant_features': np.where(ensemble_importance > np.mean(ensemble_importance) + np.std(ensemble_importance))[0]
                }
                
        except Exception as e:
            self.feature_importance_analysis_['error'] = str(e)

    def _analyze_boosting_process(self):
        """
        Analyze the boosting process including weight evolution and error reduction
        """
        try:
            # Initialize boosting analysis
            self.boosting_analysis_ = {
                'estimator_weights': self.model_.estimator_weights_.copy(),
                'weight_evolution': [],
                'error_evolution': [],
                'staged_scores': [],
                'boosting_efficiency': {},
                'convergence_metrics': {},
                'diversity_analysis': {},
                'weight_distribution': {}
            }
            
            # Get staged predictions for training data
            staged_predictions = list(self.model_.staged_predict(self.X_train_scaled_))
            
            # Calculate error evolution
            train_errors = []
            for stage_pred in staged_predictions:
                mse = mean_squared_error(self.y_train_, stage_pred)
                mae = mean_absolute_error(self.y_train_, stage_pred)
                r2 = r2_score(self.y_train_, stage_pred)
                
                train_errors.append({
                    'mse': mse,
                    'mae': mae,
                    'r2': r2,
                    'rmse': np.sqrt(mse)
                })
            
            self.boosting_analysis_['error_evolution'] = train_errors
            
            # Calculate staged scores
            self.boosting_analysis_['staged_scores'] = [
                {
                    'stage': i + 1,
                    'r2_score': score['r2'],
                    'mse': score['mse'],
                    'improvement': score['r2'] - train_errors[0]['r2'] if i > 0 else 0.0
                }
                for i, score in enumerate(train_errors)
            ]
            
            # Analyze weight evolution
            cumulative_weight = 0.0
            for i, weight in enumerate(self.model_.estimator_weights_):
                cumulative_weight += abs(weight)
                self.boosting_analysis_['weight_evolution'].append({
                    'stage': i + 1,
                    'weight': weight,
                    'abs_weight': abs(weight),
                    'cumulative_weight': cumulative_weight,
                    'relative_contribution': abs(weight) / cumulative_weight if cumulative_weight > 0 else 0.0
                })
            
            # Calculate boosting efficiency
            if len(train_errors) > 1:
                initial_error = train_errors[0]['mse']
                final_error = train_errors[-1]['mse']
                error_reduction = (initial_error - final_error) / initial_error if initial_error > 0 else 0.0
                
                self.boosting_analysis_['boosting_efficiency'] = {
                    'error_reduction_rate': error_reduction,
                    'convergence_rate': self._calculate_convergence_rate(train_errors),
                    'effectiveness_per_estimator': error_reduction / len(self.model_.estimators_),
                    'diminishing_returns': self._calculate_diminishing_returns(train_errors)
                }
            
            # Convergence metrics
            self.boosting_analysis_['convergence_metrics'] = {
                'converged': self._check_convergence(train_errors),
                'convergence_stage': self._find_convergence_stage(train_errors),
                'stability_window': 5,
                'final_performance': train_errors[-1] if train_errors else {}
            }
            
            # Diversity analysis
            self.boosting_analysis_['diversity_analysis'] = {
                'weight_diversity': self._calculate_weight_diversity(),
                'prediction_diversity': self._calculate_prediction_diversity(staged_predictions),
                'complementarity': self._calculate_estimator_complementarity()
            }
            
            # Weight distribution analysis
            weights = self.model_.estimator_weights_
            self.boosting_analysis_['weight_distribution'] = {
                'mean_weight': np.mean(weights),
                'std_weight': np.std(weights),
                'min_weight': np.min(weights),
                'max_weight': np.max(weights),
                'weight_range': np.max(weights) - np.min(weights),
                'positive_weights': np.sum(weights > 0),
                'negative_weights': np.sum(weights < 0),
                'zero_weights': np.sum(np.abs(weights) < 1e-10),
                'weight_entropy': self._calculate_weight_entropy(weights),
                'effective_estimators': np.sum(np.abs(weights) > 0.01 * np.max(np.abs(weights)))
            }
            
        except Exception as e:
            self.boosting_analysis_['error'] = str(e)

    def _analyze_base_estimators(self):
        """
        Analyze individual base estimators and their characteristics
        """
        try:
            # Initialize base estimator analysis
            self.base_estimator_analysis_ = {
                'estimator_characteristics': [],
                'performance_analysis': {},
                'complexity_analysis': {},
                'diversity_metrics': {},
                'correlation_analysis': {},
                'ensemble_contribution': {}
            }
            
            # Analyze each base estimator
            individual_predictions = []
            estimator_info = []
            
            for i, estimator in enumerate(self.model_.estimators_):
                # Get estimator predictions
                pred = estimator.predict(self.X_train_scaled_)
                individual_predictions.append(pred)
                
                # Calculate individual performance
                mse = mean_squared_error(self.y_train_, pred)
                mae = mean_absolute_error(self.y_train_, pred)
                r2 = r2_score(self.y_train_, pred)
                
                # Get estimator characteristics
                estimator_chars = {
                    'index': i,
                    'weight': self.model_.estimator_weights_[i] if i < len(self.model_.estimator_weights_) else 0.0,
                    'performance': {
                        'mse': mse,
                        'mae': mae,
                        'r2': r2,
                        'rmse': np.sqrt(mse)
                    }
                }
                
                # Add tree-specific characteristics if it's a decision tree
                if hasattr(estimator, 'tree_'):
                    estimator_chars['tree_characteristics'] = {
                        'n_nodes': estimator.tree_.node_count,
                        'n_leaves': estimator.tree_.n_leaves,
                        'max_depth': estimator.tree_.max_depth,
                        'n_features_used': len(np.unique(estimator.tree_.feature[estimator.tree_.feature >= 0]))
                    }
                
                # Feature importance if available
                if hasattr(estimator, 'feature_importances_'):
                    estimator_chars['feature_importance'] = estimator.feature_importances_
                    estimator_chars['top_features'] = np.argsort(estimator.feature_importances_)[-5:][::-1]
                
                estimator_info.append(estimator_chars)
            
            self.base_estimator_analysis_['estimator_characteristics'] = estimator_info
            
            # Performance analysis
            performances = [info['performance'] for info in estimator_info]
            weights = [info['weight'] for info in estimator_info]
            
            self.base_estimator_analysis_['performance_analysis'] = {
                'mean_performance': {
                    'mse': np.mean([p['mse'] for p in performances]),
                    'mae': np.mean([p['mae'] for p in performances]),
                    'r2': np.mean([p['r2'] for p in performances])
                },
                'performance_std': {
                    'mse': np.std([p['mse'] for p in performances]),
                    'mae': np.std([p['mae'] for p in performances]),
                    'r2': np.std([p['r2'] for p in performances])
                },
                'best_estimator': np.argmax([p['r2'] for p in performances]),
                'worst_estimator': np.argmin([p['r2'] for p in performances]),
                'performance_weight_correlation': np.corrcoef([p['r2'] for p in performances], weights)[0, 1] if len(weights) > 1 else 0.0
            }
            
            # Complexity analysis
            if any('tree_characteristics' in info for info in estimator_info):
                tree_infos = [info['tree_characteristics'] for info in estimator_info if 'tree_characteristics' in info]
                
                self.base_estimator_analysis_['complexity_analysis'] = {
                    'mean_nodes': np.mean([t['n_nodes'] for t in tree_infos]),
                    'mean_leaves': np.mean([t['n_leaves'] for t in tree_infos]),
                    'mean_depth': np.mean([t['max_depth'] for t in tree_infos]),
                    'mean_features_used': np.mean([t['n_features_used'] for t in tree_infos]),
                    'complexity_diversity': np.std([t['n_nodes'] for t in tree_infos]),
                    'depth_diversity': np.std([t['max_depth'] for t in tree_infos])
                }
            
            # Diversity metrics
            if len(individual_predictions) > 1:
                individual_predictions = np.array(individual_predictions)
                
                # Calculate pairwise correlations
                correlations = np.corrcoef(individual_predictions)
                
                self.base_estimator_analysis_['diversity_metrics'] = {
                    'mean_correlation': np.mean(correlations[np.triu_indices_from(correlations, k=1)]),
                    'correlation_matrix': correlations,
                    'diversity_score': 1.0 - np.mean(correlations[np.triu_indices_from(correlations, k=1)]),
                    'prediction_variance': np.var(individual_predictions, axis=0),
                    'ensemble_variance': np.mean(np.var(individual_predictions, axis=0))
                }
            
            # Ensemble contribution analysis
            self.base_estimator_analysis_['ensemble_contribution'] = {
                'weight_contribution': [abs(w) / np.sum(np.abs(weights)) for w in weights],
                'performance_contribution': self._calculate_performance_contribution(individual_predictions, weights),
                'marginal_contribution': self._calculate_marginal_contributions(individual_predictions, weights)
            }
            
        except Exception as e:
            self.base_estimator_analysis_['error'] = str(e)

    def _calculate_convergence_rate(self, error_evolution):
        """Calculate the rate of convergence based on error reduction"""
        try:
            if len(error_evolution) < 2:
                return 0.0
            
            # Calculate exponential decay rate
            errors = [e['mse'] for e in error_evolution]
            if errors[0] <= 0:
                return 0.0
            
            # Fit exponential decay model: error = initial * exp(-rate * stage)
            stages = np.arange(1, len(errors) + 1)
            log_errors = np.log(np.maximum(errors, 1e-10))
            
            # Simple linear regression on log scale
            slope = np.polyfit(stages, log_errors, 1)[0]
            return -slope  # Negative slope means decay
            
        except Exception:
            return 0.0
        
    def _calculate_diminishing_returns(self, error_evolution):
        """Calculate diminishing returns in error reduction"""
        try:
            if len(error_evolution) < 3:
                return 0.0
            
            # Calculate the rate of improvement change
            errors = [e['mse'] for e in error_evolution]
            improvements = []
            
            for i in range(1, len(errors)):
                if errors[i-1] > 0:
                    improvement = (errors[i-1] - errors[i]) / errors[i-1]
                    improvements.append(improvement)
                else:
                    improvements.append(0.0)
            
            if len(improvements) < 2:
                return 0.0
            
            # Calculate how much improvement rate decreases
            improvement_changes = []
            for i in range(1, len(improvements)):
                change = improvements[i-1] - improvements[i]
                improvement_changes.append(change)
            
            # Average rate of diminishing returns
            return np.mean(improvement_changes) if improvement_changes else 0.0
            
        except Exception:
            return 0.0

    def _check_convergence(self, error_evolution, window_size=5, threshold=0.001):
        """Check if the boosting process has converged"""
        try:
            if len(error_evolution) < window_size:
                return False
            
            # Check if error has stabilized in the last window_size stages
            recent_errors = [e['mse'] for e in error_evolution[-window_size:]]
            error_variance = np.var(recent_errors)
            error_mean = np.mean(recent_errors)
            
            # Convergence if relative variance is below threshold
            relative_variance = error_variance / (error_mean + 1e-10)
            return relative_variance < threshold
            
        except Exception:
            return False

    def _find_convergence_stage(self, error_evolution, window_size=5, threshold=0.001):
        """Find the stage where convergence occurred"""
        try:
            if len(error_evolution) < window_size:
                return None
            
            # Check each possible convergence point
            for i in range(window_size, len(error_evolution) + 1):
                window_errors = [e['mse'] for e in error_evolution[i-window_size:i]]
                error_variance = np.var(window_errors)
                error_mean = np.mean(window_errors)
                
                relative_variance = error_variance / (error_mean + 1e-10)
                if relative_variance < threshold:
                    return i - window_size + 1  # Return the start of convergence window
            
            return None
            
        except Exception:
            return None

    def _calculate_weight_diversity(self):
        """Calculate diversity metrics for estimator weights"""
        try:
            weights = self.model_.estimator_weights_
            
            # Normalize weights to calculate entropy
            abs_weights = np.abs(weights)
            if np.sum(abs_weights) > 0:
                probabilities = abs_weights / np.sum(abs_weights)
                entropy = -np.sum(probabilities * np.log2(probabilities + 1e-10))
                max_entropy = np.log2(len(weights))
                normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0.0
            else:
                normalized_entropy = 0.0
            
            # Calculate coefficient of variation
            weight_cv = np.std(weights) / (np.mean(np.abs(weights)) + 1e-10)
            
            # Calculate effective number of estimators (based on weight distribution)
            effective_estimators = 1.0 / np.sum(probabilities**2) if np.sum(abs_weights) > 0 else 1.0
            
            return {
                'entropy': entropy,
                'normalized_entropy': normalized_entropy,
                'coefficient_of_variation': weight_cv,
                'effective_estimators': effective_estimators,
                'weight_concentration': np.max(probabilities) if np.sum(abs_weights) > 0 else 1.0,
                'diversity_score': 1.0 - np.max(probabilities) if np.sum(abs_weights) > 0 else 0.0
            }
            
        except Exception:
            return {
                'entropy': 0.0,
                'normalized_entropy': 0.0,
                'coefficient_of_variation': 0.0,
                'effective_estimators': 1.0,
                'weight_concentration': 1.0,
                'diversity_score': 0.0
            }

    def _calculate_prediction_diversity(self, staged_predictions):
        """Calculate prediction diversity across boosting stages"""
        try:
            if len(staged_predictions) < 2:
                return {
                    'mean_variance': 0.0,
                    'variance_evolution': [],
                    'diversity_trend': 0.0,
                    'final_diversity': 0.0
                }
            
            staged_predictions = np.array(staged_predictions)
            
            # Calculate variance at each stage (diversity among predictions)
            variance_evolution = []
            for i in range(1, len(staged_predictions)):
                current_predictions = staged_predictions[:i+1]
                stage_variance = np.mean(np.var(current_predictions, axis=0))
                variance_evolution.append(stage_variance)
            
            # Calculate trend in diversity
            if len(variance_evolution) > 1:
                # Linear regression to find trend
                stages = np.arange(len(variance_evolution))
                trend_slope = np.polyfit(stages, variance_evolution, 1)[0]
            else:
                trend_slope = 0.0
            
            return {
                'mean_variance': np.mean(variance_evolution) if variance_evolution else 0.0,
                'variance_evolution': variance_evolution,
                'diversity_trend': trend_slope,
                'final_diversity': variance_evolution[-1] if variance_evolution else 0.0,
                'diversity_range': np.max(variance_evolution) - np.min(variance_evolution) if variance_evolution else 0.0
            }
            
        except Exception:
            return {
                'mean_variance': 0.0,
                'variance_evolution': [],
                'diversity_trend': 0.0,
                'final_diversity': 0.0
            }

    def _calculate_estimator_complementarity(self):
        """Calculate how well estimators complement each other"""
        try:
            if not hasattr(self.model_, 'estimators_') or len(self.model_.estimators_) < 2:
                return {
                    'pairwise_complementarity': 0.0,
                    'error_complementarity': 0.0,
                    'prediction_complementarity': 0.0,
                    'overall_complementarity': 0.0
                }
            
            # Get individual predictions
            individual_predictions = []
            individual_errors = []
            
            for estimator in self.model_.estimators_:
                pred = estimator.predict(self.X_train_scaled_)
                error = np.abs(pred - self.y_train_)
                individual_predictions.append(pred)
                individual_errors.append(error)
            
            individual_predictions = np.array(individual_predictions)
            individual_errors = np.array(individual_errors)
            
            # Calculate pairwise correlation of predictions (lower = more complementary)
            pred_correlations = np.corrcoef(individual_predictions)
            mean_pred_correlation = np.mean(pred_correlations[np.triu_indices_from(pred_correlations, k=1)])
            prediction_complementarity = 1.0 - abs(mean_pred_correlation)
            
            # Calculate error complementarity (negative correlation of errors is good)
            error_correlations = np.corrcoef(individual_errors)
            mean_error_correlation = np.mean(error_correlations[np.triu_indices_from(error_correlations, k=1)])
            error_complementarity = 1.0 - max(0, mean_error_correlation)
            
            # Calculate ensemble vs individual performance improvement
            ensemble_pred = self.model_.predict(self.X_train_scaled_)
            ensemble_mse = mean_squared_error(self.y_train_, ensemble_pred)
            individual_mses = [mean_squared_error(self.y_train_, pred) for pred in individual_predictions]
            mean_individual_mse = np.mean(individual_mses)
            
            performance_improvement = (mean_individual_mse - ensemble_mse) / mean_individual_mse if mean_individual_mse > 0 else 0.0
            
            # Overall complementarity score
            overall_complementarity = (prediction_complementarity + error_complementarity + performance_improvement) / 3.0
            
            return {
                'pairwise_complementarity': prediction_complementarity,
                'error_complementarity': error_complementarity,
                'prediction_complementarity': prediction_complementarity,
                'performance_improvement': performance_improvement,
                'overall_complementarity': overall_complementarity,
                'mean_prediction_correlation': mean_pred_correlation,
                'mean_error_correlation': mean_error_correlation
            }
            
        except Exception:
            return {
                'pairwise_complementarity': 0.0,
                'error_complementarity': 0.0,
                'prediction_complementarity': 0.0,
                'overall_complementarity': 0.0
            }

    def _calculate_performance_contribution(self, individual_predictions, weights):
        """Calculate how each estimator contributes to overall performance"""
        try:
            if len(individual_predictions) == 0:
                return []
            
            individual_predictions = np.array(individual_predictions)
            weights = np.array(weights)
            
            # Calculate individual performance
            individual_performances = []
            for pred in individual_predictions:
                r2 = r2_score(self.y_train_, pred)
                individual_performances.append(r2)
            
            # Calculate weighted ensemble performance
            weighted_pred = np.average(individual_predictions, axis=0, weights=np.abs(weights))
            ensemble_r2 = r2_score(self.y_train_, weighted_pred)
            
            # Calculate marginal contribution of each estimator
            contributions = []
            for i in range(len(individual_predictions)):
                # Remove estimator i and recalculate performance
                remaining_indices = list(range(len(individual_predictions)))
                remaining_indices.remove(i)
                
                if remaining_indices:
                    remaining_predictions = individual_predictions[remaining_indices]
                    remaining_weights = weights[remaining_indices]
                    
                    if np.sum(np.abs(remaining_weights)) > 0:
                        remaining_weights = remaining_weights / np.sum(np.abs(remaining_weights))
                        without_i_pred = np.average(remaining_predictions, axis=0, weights=np.abs(remaining_weights))
                        without_i_r2 = r2_score(self.y_train_, without_i_pred)
                        
                        # Contribution is the difference
                        contribution = ensemble_r2 - without_i_r2
                    else:
                        contribution = individual_performances[i]
                else:
                    contribution = individual_performances[i]
                
                contributions.append(contribution)
            
            return contributions
            
        except Exception:
            return [0.0] * len(individual_predictions) if len(individual_predictions) > 0 else []

    def _calculate_marginal_contributions(self, individual_predictions, weights):
        """Calculate marginal contribution of each estimator to ensemble performance"""
        try:
            if len(individual_predictions) == 0:
                return []
            
            individual_predictions = np.array(individual_predictions)
            weights = np.array(weights)
            
            # Calculate baseline (empty ensemble) - use mean prediction
            baseline_pred = np.mean(self.y_train_)
            baseline_mse = mean_squared_error(self.y_train_, np.full_like(self.y_train_, baseline_pred))
            
            marginal_contributions = []
            cumulative_pred = np.zeros_like(self.y_train_)
            cumulative_weight = 0.0
            
            # Add estimators one by one and calculate marginal improvement
            for i, (pred, weight) in enumerate(zip(individual_predictions, weights)):
                # Add current estimator
                old_cumulative_pred = cumulative_pred.copy()
                old_cumulative_weight = cumulative_weight
                
                cumulative_pred += pred * weight
                cumulative_weight += abs(weight)
                
                if cumulative_weight > 0:
                    current_ensemble_pred = cumulative_pred / cumulative_weight
                    current_mse = mean_squared_error(self.y_train_, current_ensemble_pred)
                else:
                    current_mse = baseline_mse
                
                # Calculate marginal contribution
                if i == 0:
                    marginal_contribution = baseline_mse - current_mse
                else:
                    if old_cumulative_weight > 0:
                        old_ensemble_pred = old_cumulative_pred / old_cumulative_weight
                        old_mse = mean_squared_error(self.y_train_, old_ensemble_pred)
                    else:
                        old_mse = baseline_mse
                    
                    marginal_contribution = old_mse - current_mse
                
                marginal_contributions.append(marginal_contribution)
            
            return marginal_contributions
            
        except Exception:
            return [0.0] * len(individual_predictions) if len(individual_predictions) > 0 else []
        
    def _analyze_convergence(self):
        """
        Comprehensive convergence analysis of the AdaBoost training process
        """
        try:
            # Initialize convergence analysis
            self.convergence_analysis_ = {
                'convergence_detected': False,
                'convergence_stage': None,
                'convergence_metrics': {},
                'stability_analysis': {},
                'plateau_detection': {},
                'early_stopping_recommendations': {},
                'convergence_quality': {}
            }
            
            # Get staged predictions and errors
            staged_predictions = list(self.model_.staged_predict(self.X_train_scaled_))
            
            # Calculate staged errors
            staged_errors = []
            for pred in staged_predictions:
                mse = mean_squared_error(self.y_train_, pred)
                mae = mean_absolute_error(self.y_train_, pred)
                r2 = r2_score(self.y_train_, pred)
                staged_errors.append({'mse': mse, 'mae': mae, 'r2': r2})
            
            # Detect convergence using multiple criteria
            convergence_detected = self._check_convergence(staged_errors)
            convergence_stage = self._find_convergence_stage(staged_errors)
            
            self.convergence_analysis_['convergence_detected'] = convergence_detected
            self.convergence_analysis_['convergence_stage'] = convergence_stage
            
            # Calculate convergence metrics
            if len(staged_errors) > 1:
                mse_values = [e['mse'] for e in staged_errors]
                r2_values = [e['r2'] for e in staged_errors]
                
                # Rate of improvement
                improvement_rate = self._calculate_improvement_rate(mse_values)
                
                # Stability metrics
                stability_window = min(10, len(mse_values) // 4)
                if stability_window > 0:
                    recent_mse = mse_values[-stability_window:]
                    stability_variance = np.var(recent_mse)
                    stability_trend = np.polyfit(range(len(recent_mse)), recent_mse, 1)[0]
                else:
                    stability_variance = 0.0
                    stability_trend = 0.0
                
                self.convergence_analysis_['convergence_metrics'] = {
                    'improvement_rate': improvement_rate,
                    'final_mse': mse_values[-1],
                    'best_mse': min(mse_values),
                    'total_improvement': (mse_values[0] - mse_values[-1]) / mse_values[0] if mse_values[0] > 0 else 0.0,
                    'convergence_efficiency': improvement_rate / len(staged_errors) if len(staged_errors) > 0 else 0.0
                }
                
                self.convergence_analysis_['stability_analysis'] = {
                    'stability_variance': stability_variance,
                    'stability_trend': stability_trend,
                    'is_stable': abs(stability_trend) < 0.001 and stability_variance < 0.001,
                    'oscillation_detected': self._detect_oscillations(mse_values),
                    'overfitting_risk': self._assess_overfitting_risk(mse_values, r2_values)
                }
            
            # Plateau detection
            plateau_info = self._detect_plateau(staged_errors)
            self.convergence_analysis_['plateau_detection'] = plateau_info
            
            # Early stopping recommendations
            optimal_stage = self._recommend_early_stopping(staged_errors)
            self.convergence_analysis_['early_stopping_recommendations'] = {
                'optimal_n_estimators': optimal_stage,
                'potential_savings': max(0, len(staged_errors) - optimal_stage) if optimal_stage else 0,
                'performance_at_optimal': staged_errors[optimal_stage - 1] if optimal_stage and optimal_stage <= len(staged_errors) else None
            }
            
            # Convergence quality assessment
            quality_score = self._assess_convergence_quality(staged_errors, convergence_detected, convergence_stage)
            self.convergence_analysis_['convergence_quality'] = quality_score
            
        except Exception as e:
            self.convergence_analysis_['error'] = str(e)

    def _analyze_error_evolution(self):
        """
        Analyze how errors evolve through the boosting process
        """
        try:
            # Initialize error evolution analysis
            self.error_evolution_analysis_ = {
                'training_error_evolution': [],
                'error_trends': {},
                'error_decomposition': {},
                'bias_variance_analysis': {},
                'residual_analysis': {},
                'error_distribution_changes': {}
            }
            
            # Get staged predictions
            staged_predictions = list(self.model_.staged_predict(self.X_train_scaled_))
            
            # Calculate detailed error evolution
            for i, pred in enumerate(staged_predictions):
                # Basic errors
                mse = mean_squared_error(self.y_train_, pred)
                mae = mean_absolute_error(self.y_train_, pred)
                r2 = r2_score(self.y_train_, pred)
                
                # Residuals
                residuals = self.y_train_ - pred
                
                # Error statistics
                error_stats = {
                    'stage': i + 1,
                    'mse': mse,
                    'mae': mae,
                    'r2': r2,
                    'rmse': np.sqrt(mse),
                    'residual_mean': np.mean(residuals),
                    'residual_std': np.std(residuals),
                    'residual_skewness': self._calculate_skewness(residuals),
                    'residual_kurtosis': self._calculate_kurtosis(residuals),
                    'max_error': np.max(np.abs(residuals)),
                    'median_error': np.median(np.abs(residuals)),
                    'error_quantiles': {
                        '25%': np.percentile(np.abs(residuals), 25),
                        '75%': np.percentile(np.abs(residuals), 75),
                        '90%': np.percentile(np.abs(residuals), 90),
                        '95%': np.percentile(np.abs(residuals), 95)
                    }
                }
                
                self.error_evolution_analysis_['training_error_evolution'].append(error_stats)
            
            # Analyze error trends
            if len(self.error_evolution_analysis_['training_error_evolution']) > 1:
                mse_values = [e['mse'] for e in self.error_evolution_analysis_['training_error_evolution']]
                mae_values = [e['mae'] for e in self.error_evolution_analysis_['training_error_evolution']]
                r2_values = [e['r2'] for e in self.error_evolution_analysis_['training_error_evolution']]
                
                # Calculate trends
                stages = np.arange(1, len(mse_values) + 1)
                mse_trend = np.polyfit(stages, mse_values, 1)[0]
                mae_trend = np.polyfit(stages, mae_values, 1)[0]
                r2_trend = np.polyfit(stages, r2_values, 1)[0]
                
                self.error_evolution_analysis_['error_trends'] = {
                    'mse_trend': mse_trend,
                    'mae_trend': mae_trend,
                    'r2_trend': r2_trend,
                    'overall_improvement': mse_values[0] - mse_values[-1],
                    'relative_improvement': (mse_values[0] - mse_values[-1]) / mse_values[0] if mse_values[0] > 0 else 0.0,
                    'improvement_consistency': self._calculate_improvement_consistency(mse_values)
                }
            
            # Bias-Variance decomposition approximation
            if len(staged_predictions) > 10:
                bias_variance = self._approximate_bias_variance_decomposition(staged_predictions)
                self.error_evolution_analysis_['bias_variance_analysis'] = bias_variance
            
            # Residual analysis
            final_residuals = self.y_train_ - staged_predictions[-1]
            residual_analysis = self._analyze_residuals(final_residuals)
            self.error_evolution_analysis_['residual_analysis'] = residual_analysis
            
            # Error distribution changes
            error_distribution_changes = self._analyze_error_distribution_changes(staged_predictions)
            self.error_evolution_analysis_['error_distribution_changes'] = error_distribution_changes
            
        except Exception as e:
            self.error_evolution_analysis_['error'] = str(e)

    def _analyze_learning_curves(self):
        """
        Analyze learning curves with different training set sizes
        """
        try:
            # Initialize learning curve analysis
            self.learning_curve_analysis_ = {
                'train_sizes': [],
                'train_scores': [],
                'validation_scores': [],
                'learning_curve_metrics': {},
                'overfitting_analysis': {},
                'data_efficiency': {},
                'convergence_by_size': {}
            }
            
            # Define training sizes
            train_sizes = np.linspace(0.1, 1.0, 10)
            
            # Calculate learning curves
            train_sizes_abs, train_scores, val_scores = learning_curve(
                self.model_,
                self.X_train_scaled_,
                self.y_train_,
                train_sizes=train_sizes,
                cv=min(self.cv_folds, 5),  # Limit CV folds for efficiency
                scoring='neg_mean_squared_error',
                n_jobs=1,  # Single job to avoid issues
                random_state=self.random_state
            )
            
            # Convert to positive MSE scores
            train_scores = -train_scores
            val_scores = -val_scores
            
            # Store results
            self.learning_curve_analysis_['train_sizes'] = train_sizes_abs
            self.learning_curve_analysis_['train_scores'] = train_scores
            self.learning_curve_analysis_['validation_scores'] = val_scores
            
            # Calculate learning curve metrics
            train_means = np.mean(train_scores, axis=1)
            train_stds = np.std(train_scores, axis=1)
            val_means = np.mean(val_scores, axis=1)
            val_stds = np.std(val_scores, axis=1)
            
            # Gap analysis (overfitting indicator)
            gaps = val_means - train_means
            
            self.learning_curve_analysis_['learning_curve_metrics'] = {
                'train_score_means': train_means,
                'train_score_stds': train_stds,
                'val_score_means': val_means,
                'val_score_stds': val_stds,
                'generalization_gaps': gaps,
                'final_gap': gaps[-1],
                'max_gap': np.max(gaps),
                'gap_trend': np.polyfit(train_sizes_abs, gaps, 1)[0]
            }
            
            # Overfitting analysis
            overfitting_score = self._analyze_overfitting_from_curves(train_means, val_means, gaps)
            self.learning_curve_analysis_['overfitting_analysis'] = overfitting_score
            
            # Data efficiency analysis
            data_efficiency = self._analyze_data_efficiency(train_sizes_abs, val_means)
            self.learning_curve_analysis_['data_efficiency'] = data_efficiency
            
            # Analyze convergence behavior by training size
            convergence_analysis = self._analyze_convergence_by_size(train_sizes_abs, val_means, val_stds)
            self.learning_curve_analysis_['convergence_by_size'] = convergence_analysis
            
        except Exception as e:
            self.learning_curve_analysis_['error'] = str(e)

    def _calculate_improvement_rate(self, error_values):
        """Calculate the rate of improvement in error values"""
        try:
            if len(error_values) < 2:
                return 0.0
            
            # Calculate relative improvements
            improvements = []
            for i in range(1, len(error_values)):
                if error_values[i-1] > 0:
                    improvement = (error_values[i-1] - error_values[i]) / error_values[i-1]
                    improvements.append(improvement)
            
            return np.mean(improvements) if improvements else 0.0
            
        except Exception:
            return 0.0

    def _detect_oscillations(self, values, window_size=5):
        """Detect oscillations in a sequence of values"""
        try:
            if len(values) < window_size * 2:
                return False
            
            # Calculate direction changes
            direction_changes = 0
            for i in range(window_size, len(values) - window_size):
                window = values[i-window_size:i+window_size+1]
                if values[i] == max(window) or values[i] == min(window):
                    direction_changes += 1
            
            # Oscillation if too many direction changes
            oscillation_threshold = len(values) // (window_size * 2)
            return direction_changes > oscillation_threshold
            
        except Exception:
            return False

    def _assess_overfitting_risk(self, mse_values, r2_values):
        """Assess the risk of overfitting based on error evolution"""
        try:
            if len(mse_values) < 10:
                return 'insufficient_data'
            
            # Check if errors start increasing (overfitting indicator)
            recent_portion = len(mse_values) // 4
            recent_mse = mse_values[-recent_portion:]
            recent_trend = np.polyfit(range(len(recent_mse)), recent_mse, 1)[0]
            
            # Check R2 degradation
            recent_r2 = r2_values[-recent_portion:]
            r2_trend = np.polyfit(range(len(recent_r2)), recent_r2, 1)[0]
            
            if recent_trend > 0.001 or r2_trend < -0.001:
                return 'high'
            elif recent_trend > 0.0001 or r2_trend < -0.0001:
                return 'medium'
            else:
                return 'low'
                
        except Exception:
            return 'unknown'        
 
    def _analyze_cross_validation(self):
        """
        Perform cross-validation analysis of the AdaBoost model
        """
        try:
            # Initialize cross-validation analysis
            self.cross_validation_analysis_ = {
                'cv_scores': [],
                'cv_statistics': {},
                'fold_analysis': [],
                'stability_metrics': {},
                'hyperparameter_sensitivity': {},
                'performance_consistency': {}
            }
            
            # Perform cross-validation with multiple metrics
            cv_scores_mse = cross_val_score(
                self.model_, self.X_train_scaled_, self.y_train_,
                cv=self.cv_folds, scoring='neg_mean_squared_error'
            )
            cv_scores_mse = -cv_scores_mse  # Convert to positive MSE
            
            cv_scores_r2 = cross_val_score(
                self.model_, self.X_train_scaled_, self.y_train_,
                cv=self.cv_folds, scoring='r2'
            )
            
            cv_scores_mae = cross_val_score(
                self.model_, self.X_train_scaled_, self.y_train_,
                cv=self.cv_folds, scoring='neg_mean_absolute_error'
            )
            cv_scores_mae = -cv_scores_mae  # Convert to positive MAE
            
            # Store CV scores
            self.cross_validation_analysis_['cv_scores'] = {
                'mse': cv_scores_mse,
                'r2': cv_scores_r2,
                'mae': cv_scores_mae
            }
            
            # Calculate CV statistics
            self.cross_validation_analysis_['cv_statistics'] = {
                'mse': {
                    'mean': np.mean(cv_scores_mse),
                    'std': np.std(cv_scores_mse),
                    'min': np.min(cv_scores_mse),
                    'max': np.max(cv_scores_mse),
                    'cv': np.std(cv_scores_mse) / np.mean(cv_scores_mse) if np.mean(cv_scores_mse) > 0 else 0
                },
                'r2': {
                    'mean': np.mean(cv_scores_r2),
                    'std': np.std(cv_scores_r2),
                    'min': np.min(cv_scores_r2),
                    'max': np.max(cv_scores_r2),
                    'cv': np.std(cv_scores_r2) / np.mean(cv_scores_r2) if np.mean(cv_scores_r2) > 0 else 0
                },
                'mae': {
                    'mean': np.mean(cv_scores_mae),
                    'std': np.std(cv_scores_mae),
                    'min': np.min(cv_scores_mae),
                    'max': np.max(cv_scores_mae),
                    'cv': np.std(cv_scores_mae) / np.mean(cv_scores_mae) if np.mean(cv_scores_mae) > 0 else 0
                }
            }
            
            # Analyze individual folds
            for i, (mse, r2, mae) in enumerate(zip(cv_scores_mse, cv_scores_r2, cv_scores_mae)):
                fold_analysis = {
                    'fold': i + 1,
                    'mse': mse,
                    'r2': r2,
                    'mae': mae,
                    'performance_rank': {
                        'mse_rank': np.argsort(cv_scores_mse)[i] + 1,
                        'r2_rank': np.argsort(-cv_scores_r2)[i] + 1,  # Negative for descending order
                        'mae_rank': np.argsort(cv_scores_mae)[i] + 1
                    }
                }
                self.cross_validation_analysis_['fold_analysis'].append(fold_analysis)
            
            # Stability metrics
            self.cross_validation_analysis_['stability_metrics'] = {
                'performance_stability': 1.0 - np.std(cv_scores_r2) / (np.mean(cv_scores_r2) + 1e-10),
                'error_stability': 1.0 - np.std(cv_scores_mse) / (np.mean(cv_scores_mse) + 1e-10),
                'consistency_score': self._calculate_cv_consistency(cv_scores_r2),
                'outlier_folds': self._identify_outlier_folds(cv_scores_r2)
            }
            
            # Performance consistency analysis
            self.cross_validation_analysis_['performance_consistency'] = {
                'coefficient_of_variation_r2': np.std(cv_scores_r2) / (np.mean(cv_scores_r2) + 1e-10),
                'range_normalized_r2': (np.max(cv_scores_r2) - np.min(cv_scores_r2)) / (np.mean(cv_scores_r2) + 1e-10),
                'robust_std_r2': np.percentile(cv_scores_r2, 75) - np.percentile(cv_scores_r2, 25),
                'consistency_rating': self._rate_cv_consistency(cv_scores_r2)
            }
            
        except Exception as e:
            self.cross_validation_analysis_['error'] = str(e)

    def _analyze_loss_function_impact(self):
        """
        Analyze the impact of different loss functions on model performance
        """
        try:
            # Initialize loss function analysis
            self.loss_function_analysis_ = {
                'loss_comparisons': {},
                'convergence_differences': {},
                'performance_impact': {},
                'robustness_analysis': {},
                'recommendations': {}
            }
            
            # Test different loss functions
            loss_functions = ['linear', 'square', 'exponential']
            loss_results = {}
            
            for loss_func in loss_functions:
                try:
                    # Create model with different loss function
                    test_model = AdaBoostRegressor(
                        estimator=self.base_estimator_,
                        n_estimators=min(self.n_estimators, 20),  # Limit for efficiency
                        learning_rate=self.learning_rate,
                        loss=loss_func,
                        random_state=self.random_state
                    )
                    
                    # Fit and evaluate
                    test_model.fit(self.X_train_scaled_, self.y_train_)
                    
                    # Get staged predictions for convergence analysis
                    staged_preds = list(test_model.staged_predict(self.X_train_scaled_))
                    final_pred = test_model.predict(self.X_train_scaled_)
                    
                    # Calculate metrics
                    mse = mean_squared_error(self.y_train_, final_pred)
                    r2 = r2_score(self.y_train_, final_pred)
                    mae = mean_absolute_error(self.y_train_, final_pred)
                    
                    # Convergence analysis
                    mse_evolution = [mean_squared_error(self.y_train_, pred) for pred in staged_preds]
                    convergence_rate = self._calculate_convergence_rate([{'mse': mse} for mse in mse_evolution])
                    
                    # Robustness metrics
                    residuals = self.y_train_ - final_pred
                    robustness_score = self._calculate_robustness_score(residuals)
                    
                    loss_results[loss_func] = {
                        'final_performance': {'mse': mse, 'r2': r2, 'mae': mae},
                        'convergence_rate': convergence_rate,
                        'mse_evolution': mse_evolution,
                        'robustness_score': robustness_score,
                        'estimator_weights': test_model.estimator_weights_,
                        'weight_distribution': {
                            'mean': np.mean(test_model.estimator_weights_),
                            'std': np.std(test_model.estimator_weights_),
                            'range': np.max(test_model.estimator_weights_) - np.min(test_model.estimator_weights_)
                        }
                    }
                    
                except Exception as e:
                    loss_results[loss_func] = {'error': str(e)}
            
            self.loss_function_analysis_['loss_comparisons'] = loss_results
            
            # Compare convergence differences
            convergence_comparison = {}
            for loss_func, results in loss_results.items():
                if 'error' not in results:
                    convergence_comparison[loss_func] = {
                        'convergence_rate': results['convergence_rate'],
                        'final_mse': results['final_performance']['mse'],
                        'improvement_efficiency': results['convergence_rate'] / len(results['mse_evolution']) if len(results['mse_evolution']) > 0 else 0
                    }
            
            self.loss_function_analysis_['convergence_differences'] = convergence_comparison
            
            # Performance impact analysis
            if len(loss_results) > 1:
                performance_analysis = self._compare_loss_function_performance(loss_results)
                self.loss_function_analysis_['performance_impact'] = performance_analysis
            
            # Generate recommendations
            recommendations = self._generate_loss_function_recommendations(loss_results)
            self.loss_function_analysis_['recommendations'] = recommendations
            
        except Exception as e:
            self.loss_function_analysis_['error'] = str(e)

    def _analyze_prediction_variance(self):
        """
        Analyze prediction variance and uncertainty estimation
        """
        try:
            # Initialize prediction variance analysis
            self.prediction_variance_analysis_ = {
                'staged_variance': [],
                'prediction_uncertainty': {},
                'confidence_intervals': {},
                'stability_analysis': {},
                'ensemble_diversity': {}
            }
            
            # Get staged predictions
            staged_predictions = list(self.model_.staged_predict(self.X_train_scaled_))
            staged_predictions = np.array(staged_predictions)
            
            # Calculate variance at each stage
            for i in range(len(staged_predictions)):
                stage_preds = staged_predictions[:i+1]
                
                # Variance across stages for each sample
                sample_variances = np.var(stage_preds, axis=0)
                
                # Summary statistics
                variance_stats = {
                    'stage': i + 1,
                    'mean_variance': np.mean(sample_variances),
                    'std_variance': np.std(sample_variances),
                    'max_variance': np.max(sample_variances),
                    'min_variance': np.min(sample_variances),
                    'variance_distribution': {
                        '25%': np.percentile(sample_variances, 25),
                        '50%': np.percentile(sample_variances, 50),
                        '75%': np.percentile(sample_variances, 75),
                        '90%': np.percentile(sample_variances, 90)
                    }
                }
                
                self.prediction_variance_analysis_['staged_variance'].append(variance_stats)
            
            # Final prediction uncertainty
            final_variances = np.var(staged_predictions, axis=0)
            final_predictions = staged_predictions[-1]
            
            # Calculate confidence intervals (approximation)
            confidence_intervals = self._calculate_prediction_confidence_intervals(
                final_predictions, final_variances
            )
            
            self.prediction_variance_analysis_['confidence_intervals'] = confidence_intervals
            
            # Prediction uncertainty metrics
            uncertainty_metrics = {
                'mean_prediction_variance': np.mean(final_variances),
                'prediction_std': np.sqrt(np.mean(final_variances)),
                'uncertainty_coefficient': np.std(final_predictions) / (np.mean(np.abs(final_predictions)) + 1e-10),
                'high_uncertainty_samples': np.sum(final_variances > np.percentile(final_variances, 90)),
                'low_uncertainty_samples': np.sum(final_variances < np.percentile(final_variances, 10)),
                'uncertainty_distribution': {
                    'skewness': self._calculate_skewness(final_variances),
                    'kurtosis': self._calculate_kurtosis(final_variances)
                }
            }
            
            self.prediction_variance_analysis_['prediction_uncertainty'] = uncertainty_metrics
            
            # Stability analysis
            variance_evolution = [stage['mean_variance'] for stage in self.prediction_variance_analysis_['staged_variance']]
            stability_analysis = {
                'variance_trend': np.polyfit(range(len(variance_evolution)), variance_evolution, 1)[0] if len(variance_evolution) > 1 else 0,
                'variance_stability': 1.0 - (np.std(variance_evolution[-5:]) / (np.mean(variance_evolution[-5:]) + 1e-10)) if len(variance_evolution) >= 5 else 0,
                'converged_variance': abs(variance_evolution[-1] - variance_evolution[-5]) < 0.01 if len(variance_evolution) >= 5 else False,
                'oscillation_detected': self._detect_oscillations(variance_evolution)
            }
            
            self.prediction_variance_analysis_['stability_analysis'] = stability_analysis
            
            # Ensemble diversity analysis
            diversity_metrics = self._analyze_ensemble_prediction_diversity(staged_predictions)
            self.prediction_variance_analysis_['ensemble_diversity'] = diversity_metrics
            
        except Exception as e:
            self.prediction_variance_analysis_['error'] = str(e)

    def _analyze_estimator_comparison(self):
        """
        Compare AdaBoost with alternative ensemble methods
        """
        try:
            # Initialize estimator comparison analysis
            self.estimator_comparison_analysis_ = {
                'method_comparisons': {},
                'performance_ranking': {},
                'convergence_comparison': {},
                'computational_analysis': {},
                'recommendation_summary': {}
            }
            
            # Test different ensemble methods for comparison
            comparison_methods = {
                'AdaBoost': self.model_,
                'Single_DecisionTree': DecisionTreeRegressor(
                    max_depth=self.max_depth,
                    min_samples_split=self.min_samples_split,
                    min_samples_leaf=self.min_samples_leaf,
                    random_state=self.random_state
                )
            }
            
            # Add RandomForest if sklearn is available
            try:
                from sklearn.ensemble import RandomForestRegressor
                comparison_methods['RandomForest'] = RandomForestRegressor(
                    n_estimators=min(self.n_estimators, 50),
                    max_depth=self.max_depth,
                    min_samples_split=self.min_samples_split,
                    min_samples_leaf=self.min_samples_leaf,
                    random_state=self.random_state
                )
            except ImportError:
                pass
            
            method_results = {}
            
            for method_name, model in comparison_methods.items():
                try:
                    if method_name != 'AdaBoost':
                        # Fit the comparison model
                        model.fit(self.X_train_scaled_, self.y_train_)
                    
                    # Make predictions
                    predictions = model.predict(self.X_train_scaled_)
                    
                    # Calculate metrics
                    mse = mean_squared_error(self.y_train_, predictions)
                    r2 = r2_score(self.y_train_, predictions)
                    mae = mean_absolute_error(self.y_train_, predictions)
                    
                    # Cross-validation performance
                    cv_scores = cross_val_score(model, self.X_train_scaled_, self.y_train_, 
                                              cv=min(self.cv_folds, 5), scoring='r2')
                    
                    # Feature importance (if available)
                    feature_importance = None
                    if hasattr(model, 'feature_importances_'):
                        feature_importance = model.feature_importances_
                    
                    method_results[method_name] = {
                        'training_performance': {'mse': mse, 'r2': r2, 'mae': mae},
                        'cv_performance': {
                            'mean_r2': np.mean(cv_scores),
                            'std_r2': np.std(cv_scores),
                            'cv_scores': cv_scores
                        },
                        'feature_importance': feature_importance,
                        'model_complexity': self._assess_model_complexity(model),
                        'prediction_time': self._measure_prediction_time(model)
                    }
                    
                except Exception as e:
                    method_results[method_name] = {'error': str(e)}
            
            self.estimator_comparison_analysis_['method_comparisons'] = method_results
            
            # Performance ranking
            performance_ranking = self._rank_methods_by_performance(method_results)
            self.estimator_comparison_analysis_['performance_ranking'] = performance_ranking
            
            # Generate recommendations
            recommendations = self._generate_method_recommendations(method_results)
            self.estimator_comparison_analysis_['recommendation_summary'] = recommendations
            
        except Exception as e:
            self.estimator_comparison_analysis_['error'] = str(e)

    def _profile_performance(self):
        """
        Profile computational performance of the AdaBoost model
        """
        try:
            import time
            
            # Initialize performance profiling
            self.performance_profile_ = {
                'training_time': {},
                'prediction_time': {},
                'memory_usage': {},
                'scalability_analysis': {},
                'efficiency_metrics': {}
            }
            
            # Training time analysis
            start_time = time.time()
            
            # Test different numbers of estimators
            estimator_counts = [10, 25, 50, 100] if self.n_estimators >= 100 else [5, 10, 20, self.n_estimators]
            training_times = []
            
            for n_est in estimator_counts:
                if n_est <= self.n_estimators:
                    test_start = time.time()
                    
                    test_model = AdaBoostRegressor(
                        estimator=self.base_estimator_,
                        n_estimators=n_est,
                        learning_rate=self.learning_rate,
                        loss=self.loss,
                        random_state=self.random_state
                    )
                    test_model.fit(self.X_train_scaled_, self.y_train_)
                    
                    training_time = time.time() - test_start
                    training_times.append({'n_estimators': n_est, 'time': training_time})
            
            self.performance_profile_['training_time'] = {
                'estimator_scaling': training_times,
                'time_per_estimator': np.mean([t['time']/t['n_estimators'] for t in training_times]) if training_times else 0
            }
            
            # Prediction time analysis
            n_samples_test = [100, 500, 1000] if len(self.X_train_scaled_) >= 1000 else [len(self.X_train_scaled_)]
            prediction_times = []
            
            for n_samples in n_samples_test:
                if n_samples <= len(self.X_train_scaled_):
                    test_data = self.X_train_scaled_[:n_samples]
                    
                    pred_start = time.time()
                    _ = self.model_.predict(test_data)
                    pred_time = time.time() - pred_start
                    
                    prediction_times.append({
                        'n_samples': n_samples,
                        'time': pred_time,
                        'time_per_sample': pred_time / n_samples
                    })
            
            self.performance_profile_['prediction_time'] = {
                'sample_scaling': prediction_times,
                'average_time_per_sample': np.mean([p['time_per_sample'] for p in prediction_times]) if prediction_times else 0
            }
            
            # Efficiency metrics
            efficiency_metrics = {
                'training_efficiency': self._calculate_training_efficiency(),
                'prediction_efficiency': self._calculate_prediction_efficiency(),
                'memory_efficiency': self._estimate_memory_usage(),
                'overall_efficiency_score': 0.0  # Will be calculated based on above metrics
            }
            
            # Calculate overall efficiency score
            if all(v is not None for v in [efficiency_metrics['training_efficiency'], 
                                         efficiency_metrics['prediction_efficiency']]):
                efficiency_metrics['overall_efficiency_score'] = (
                    efficiency_metrics['training_efficiency'] * 0.4 +
                    efficiency_metrics['prediction_efficiency'] * 0.6
                )
            
            self.performance_profile_['efficiency_metrics'] = efficiency_metrics
            
        except Exception as e:
            self.performance_profile_['error'] = str(e)
        
    def _calculate_cv_consistency(self, cv_scores):
        """Calculate consistency score for cross-validation results"""
        try:
            if len(cv_scores) < 2:
                return 1.0
            
            # Calculate coefficient of variation (lower is more consistent)
            cv_coeff = np.std(cv_scores) / (np.mean(cv_scores) + 1e-10)
            
            # Convert to consistency score (0-1, higher is better)
            consistency = 1.0 / (1.0 + cv_coeff)
            return consistency
            
        except Exception:
            return 0.5

    def _identify_outlier_folds(self, cv_scores):
        """Identify outlier folds in cross-validation"""
        try:
            if len(cv_scores) < 3:
                return []
            
            # Use IQR method to identify outliers
            q25 = np.percentile(cv_scores, 25)
            q75 = np.percentile(cv_scores, 75)
            iqr = q75 - q25
            
            lower_bound = q25 - 1.5 * iqr
            upper_bound = q75 + 1.5 * iqr
            
            outlier_indices = []
            for i, score in enumerate(cv_scores):
                if score < lower_bound or score > upper_bound:
                    outlier_indices.append(i + 1)  # 1-based indexing
            
            return outlier_indices
            
        except Exception:
            return []

    def _rate_cv_consistency(self, cv_scores):
        """Rate the consistency of cross-validation scores"""
        try:
            cv_coeff = np.std(cv_scores) / (np.mean(cv_scores) + 1e-10)
            
            if cv_coeff < 0.05:
                return 'excellent'
            elif cv_coeff < 0.10:
                return 'good'
            elif cv_coeff < 0.20:
                return 'fair'
            else:
                return 'poor'
                
        except Exception:
            return 'unknown'

    def _calculate_robustness_score(self, residuals):
        """Calculate robustness score based on residual analysis"""
        try:
            # Check for outliers in residuals
            q75 = np.percentile(residuals, 75)
            q25 = np.percentile(residuals, 25)
            iqr = q75 - q25
            
            outlier_threshold = 1.5 * iqr
            outliers = np.sum(np.abs(residuals - np.median(residuals)) > outlier_threshold)
            outlier_rate = outliers / len(residuals)
            
            # Robustness based on outlier rate and residual distribution
            robustness = 1.0 - outlier_rate
            
            # Adjust for residual normality
            skewness = self._calculate_skewness(residuals)
            kurtosis = self._calculate_kurtosis(residuals)
            
            normality_penalty = min(0.2, abs(skewness) / 10 + abs(kurtosis - 3) / 20)
            robustness = max(0.0, robustness - normality_penalty)
            
            return robustness
            
        except Exception:
            return 0.5

    def _compare_loss_function_performance(self, loss_results):
        """Compare performance across different loss functions"""
        try:
            comparison = {}
            
            # Extract performance metrics
            performances = {}
            for loss_func, results in loss_results.items():
                if 'error' not in results:
                    performances[loss_func] = results['final_performance']
            
            if len(performances) < 2:
                return comparison
            
            # Find best performer for each metric
            best_mse = min(performances.values(), key=lambda x: x['mse'])
            best_r2 = max(performances.values(), key=lambda x: x['r2'])
            best_mae = min(performances.values(), key=lambda x: x['mae'])
            
            # Calculate relative performance
            for loss_func, perf in performances.items():
                comparison[loss_func] = {
                    'mse_relative': perf['mse'] / best_mse['mse'] if best_mse['mse'] > 0 else 1.0,
                    'r2_relative': perf['r2'] / best_r2['r2'] if best_r2['r2'] > 0 else 1.0,
                    'mae_relative': perf['mae'] / best_mae['mae'] if best_mae['mae'] > 0 else 1.0,
                    'overall_rank': 0  # Will be calculated
                }
            
            # Calculate overall rankings
            rankings = []
            for loss_func in comparison.keys():
                mse_rank = sorted(performances.keys(), key=lambda x: performances[x]['mse']).index(loss_func) + 1
                r2_rank = sorted(performances.keys(), key=lambda x: performances[x]['r2'], reverse=True).index(loss_func) + 1
                mae_rank = sorted(performances.keys(), key=lambda x: performances[x]['mae']).index(loss_func) + 1
                
                avg_rank = (mse_rank + r2_rank + mae_rank) / 3
                comparison[loss_func]['overall_rank'] = avg_rank
                rankings.append((loss_func, avg_rank))
            
            # Sort by overall rank
            rankings.sort(key=lambda x: x[1])
            comparison['ranking_order'] = [func for func, _ in rankings]
            
            return comparison
            
        except Exception:
            return {}

    def _generate_loss_function_recommendations(self, loss_results):
        """Generate recommendations for loss function selection"""
        try:
            recommendations = {
                'best_overall': None,
                'best_for_robustness': None,
                'best_for_convergence': None,
                'recommendations': []
            }
            
            valid_results = {k: v for k, v in loss_results.items() if 'error' not in v}
            
            if not valid_results:
                return recommendations
            
            # Find best for different criteria
            best_r2 = max(valid_results.items(), key=lambda x: x[1]['final_performance']['r2'])
            best_convergence = max(valid_results.items(), key=lambda x: x[1]['convergence_rate'])
            best_robustness = max(valid_results.items(), key=lambda x: x[1]['robustness_score'])
            
            recommendations['best_overall'] = best_r2[0]
            recommendations['best_for_convergence'] = best_convergence[0]
            recommendations['best_for_robustness'] = best_robustness[0]
            
            # Generate specific recommendations
            if best_r2[1]['final_performance']['r2'] > 0.8:
                recommendations['recommendations'].append(
                    f"Use '{best_r2[0]}' loss function for best overall performance (R² = {best_r2[1]['final_performance']['r2']:.3f})"
                )
            
            if best_convergence[1]['convergence_rate'] > 0.1:
                recommendations['recommendations'].append(
                    f"Use '{best_convergence[0]}' loss function for fastest convergence"
                )
            
            if best_robustness[1]['robustness_score'] > 0.7:
                recommendations['recommendations'].append(
                    f"Use '{best_robustness[0]}' loss function for best robustness to outliers"
                )
            
            return recommendations
            
        except Exception:
            return recommendations

    def _calculate_prediction_confidence_intervals(self, predictions, variances, confidence_level=0.95):
        """Calculate prediction confidence intervals"""
        try:
            from scipy import stats
            
            # Use t-distribution for confidence intervals
            df = len(predictions) - 1  # degrees of freedom
            t_value = stats.t.ppf((1 + confidence_level) / 2, df)
            
            std_errors = np.sqrt(variances)
            margin_of_error = t_value * std_errors
            
            lower_bounds = predictions - margin_of_error
            upper_bounds = predictions + margin_of_error
            
            return {
                'lower_bounds': lower_bounds,
                'upper_bounds': upper_bounds,
                'margin_of_error': margin_of_error,
                'confidence_level': confidence_level,
                'mean_interval_width': np.mean(upper_bounds - lower_bounds)
            }
            
        except ImportError:
            # Fallback without scipy
            std_errors = np.sqrt(variances)
            margin_of_error = 1.96 * std_errors  # Approximate 95% CI
            
            return {
                'lower_bounds': predictions - margin_of_error,
                'upper_bounds': predictions + margin_of_error,
                'margin_of_error': margin_of_error,
                'confidence_level': 0.95,
                'mean_interval_width': np.mean(2 * margin_of_error)
            }
        except Exception:
            return {
                'lower_bounds': predictions,
                'upper_bounds': predictions,
                'margin_of_error': np.zeros_like(predictions),
                'confidence_level': confidence_level,
                'mean_interval_width': 0.0
            }

    def _analyze_ensemble_prediction_diversity(self, staged_predictions):
        """Analyze diversity of predictions across ensemble stages"""
        try:
            staged_predictions = np.array(staged_predictions)
            
            # Calculate pairwise correlations between stages
            correlations = np.corrcoef(staged_predictions)
            
            # Diversity metrics
            mean_correlation = np.mean(correlations[np.triu_indices_from(correlations, k=1)])
            diversity_score = 1.0 - mean_correlation
            
            # Evolution of diversity
            diversity_evolution = []
            for i in range(2, len(staged_predictions) + 1):
                subset = staged_predictions[:i]
                subset_corr = np.corrcoef(subset)
                subset_diversity = 1.0 - np.mean(subset_corr[np.triu_indices_from(subset_corr, k=1)])
                diversity_evolution.append(subset_diversity)
            
            return {
                'overall_diversity': diversity_score,
                'mean_correlation': mean_correlation,
                'diversity_evolution': diversity_evolution,
                'final_diversity': diversity_evolution[-1] if diversity_evolution else diversity_score,
                'correlation_matrix': correlations,
                'diversity_trend': np.polyfit(range(len(diversity_evolution)), diversity_evolution, 1)[0] if len(diversity_evolution) > 1 else 0
            }
            
        except Exception:
            return {
                'overall_diversity': 0.0,
                'mean_correlation': 1.0,
                'diversity_evolution': [],
                'final_diversity': 0.0,
                'correlation_matrix': None,
                'diversity_trend': 0.0
            }

    def _assess_model_complexity(self, model):
        """Assess the complexity of a model"""
        try:
            complexity = {
                'model_type': type(model).__name__,
                'n_parameters': 0,
                'complexity_score': 0.0
            }
            
            # For tree-based models
            if hasattr(model, 'tree_'):
                complexity['n_parameters'] = model.tree_.node_count
                complexity['complexity_score'] = model.tree_.node_count / 100.0  # Normalize
            elif hasattr(model, 'estimators_'):
                # For ensemble models
                total_nodes = sum(est.tree_.node_count for est in model.estimators_ if hasattr(est, 'tree_'))
                complexity['n_parameters'] = total_nodes
                complexity['complexity_score'] = total_nodes / 1000.0  # Normalize
            elif hasattr(model, 'coef_'):
                # For linear models
                complexity['n_parameters'] = len(model.coef_.flatten())
                complexity['complexity_score'] = len(model.coef_.flatten()) / 100.0
            
            return complexity
            
        except Exception:
            return {
                'model_type': 'unknown',
                'n_parameters': 0,
                'complexity_score': 0.0
            }

    def _measure_prediction_time(self, model):
        """Measure prediction time for a model"""
        try:
            import time
            
            # Use a subset of training data for timing
            test_size = min(100, len(self.X_train_scaled_))
            test_data = self.X_train_scaled_[:test_size]
            
            # Warm-up run
            _ = model.predict(test_data)
            
            # Actual timing
            start_time = time.time()
            _ = model.predict(test_data)
            prediction_time = time.time() - start_time
            
            return {
                'total_time': prediction_time,
                'time_per_sample': prediction_time / test_size,
                'samples_tested': test_size
            }
            
        except Exception:
            return {
                'total_time': 0.0,
                'time_per_sample': 0.0,
                'samples_tested': 0
            }

    def _rank_methods_by_performance(self, method_results):
        """Rank methods by their performance metrics"""
        try:
            rankings = {}
            
            valid_methods = {k: v for k, v in method_results.items() if 'error' not in v}
            
            if not valid_methods:
                return rankings
            
            # Rank by training R²
            train_r2_ranking = sorted(valid_methods.items(), 
                                    key=lambda x: x[1]['training_performance']['r2'], 
                                    reverse=True)
            
            # Rank by CV R²
            cv_r2_ranking = sorted(valid_methods.items(), 
                                 key=lambda x: x[1]['cv_performance']['mean_r2'], 
                                 reverse=True)
            
            # Rank by training MSE
            train_mse_ranking = sorted(valid_methods.items(), 
                                     key=lambda x: x[1]['training_performance']['mse'])
            
            rankings = {
                'training_r2_ranking': [method for method, _ in train_r2_ranking],
                'cv_r2_ranking': [method for method, _ in cv_r2_ranking],
                'training_mse_ranking': [method for method, _ in train_mse_ranking],
                'best_overall': cv_r2_ranking[0][0] if cv_r2_ranking else None,
                'performance_summary': {
                    method: {
                        'train_r2_rank': train_r2_ranking.index((method, data)) + 1,
                        'cv_r2_rank': cv_r2_ranking.index((method, data)) + 1,
                        'train_mse_rank': train_mse_ranking.index((method, data)) + 1
                    }
                    for method, data in valid_methods.items()
                }
            }
            
            return rankings
            
        except Exception:
            return {}

    def _generate_method_recommendations(self, method_results):
        """Generate recommendations for method selection"""
        try:
            recommendations = {
                'primary_recommendation': None,
                'alternative_recommendation': None,
                'reasoning': [],
                'trade_offs': []
            }
            
            valid_methods = {k: v for k, v in method_results.items() if 'error' not in v}
            
            if not valid_methods:
                return recommendations
            
            # Find best performing method by CV R²
            best_method = max(valid_methods.items(), 
                            key=lambda x: x[1]['cv_performance']['mean_r2'])
            
            recommendations['primary_recommendation'] = best_method[0]
            recommendations['reasoning'].append(
                f"{best_method[0]} shows the best cross-validation performance (R² = {best_method[1]['cv_performance']['mean_r2']:.3f})"
            )
            
            # Check for overfitting
            train_r2 = best_method[1]['training_performance']['r2']
            cv_r2 = best_method[1]['cv_performance']['mean_r2']
            overfitting_gap = train_r2 - cv_r2
            
            if overfitting_gap > 0.1:
                recommendations['reasoning'].append(
                    f"Warning: {best_method[0]} shows signs of overfitting (train R² = {train_r2:.3f}, CV R² = {cv_r2:.3f})"
                )
                
                # Find alternative with less overfitting
                alternatives = []
                for method, data in valid_methods.items():
                    if method != best_method[0]:
                        alt_gap = data['training_performance']['r2'] - data['cv_performance']['mean_r2']
                        alternatives.append((method, alt_gap, data['cv_performance']['mean_r2']))
                
                if alternatives:
                    # Choose alternative with smallest gap but still good performance
                    alternatives.sort(key=lambda x: (x[1], -x[2]))  # Sort by gap, then by performance
                    recommendations['alternative_recommendation'] = alternatives[0][0]
                    recommendations['reasoning'].append(
                        f"Consider {alternatives[0][0]} as alternative with less overfitting"
                    )
            
            return recommendations
            
        except Exception:
            return recommendations

    def _calculate_training_efficiency(self):
        """Calculate training efficiency score"""
        try:
            # Simple efficiency based on performance vs number of estimators
            if hasattr(self, 'boosting_analysis_') and 'error_evolution' in self.boosting_analysis_:
                error_evolution = self.boosting_analysis_['error_evolution']
                if error_evolution:
                    initial_error = error_evolution[0]['mse']
                    final_error = error_evolution[-1]['mse']
                    error_reduction = (initial_error - final_error) / initial_error if initial_error > 0 else 0
                    
                    # Efficiency = performance improvement per estimator
                    efficiency = error_reduction / len(error_evolution) if len(error_evolution) > 0 else 0
                    return min(1.0, efficiency * 10)  # Scale to 0-1
                    
            return 0.5  # Default neutral score
            
        except Exception:
            return 0.5

    def _calculate_prediction_efficiency(self):
        """Calculate prediction efficiency score"""
        try:
            # Efficiency based on prediction speed relative to model complexity
            if hasattr(self, 'performance_profile_') and 'prediction_time' in self.performance_profile_:
                pred_times = self.performance_profile_['prediction_time']['sample_scaling']
                if pred_times:
                    avg_time_per_sample = np.mean([pt['time_per_sample'] for pt in pred_times])
                    
                    # Lower time per sample = higher efficiency
                    # Normalize assuming 1ms per sample is baseline
                    efficiency = 1.0 / (1.0 + avg_time_per_sample * 1000)
                    return efficiency
                    
            return 0.5  # Default neutral score
            
        except Exception:
            return 0.5

    def _estimate_memory_usage(self):
        """Estimate memory usage of the model"""
        try:
            import sys
            
            # Estimate based on model components
            memory_estimate = 0
            
            # Base estimators
            if hasattr(self.model_, 'estimators_'):
                for estimator in self.model_.estimators_:
                    if hasattr(estimator, 'tree_'):
                        # Estimate tree memory (rough approximation)
                        memory_estimate += estimator.tree_.node_count * 64  # bytes per node
            
            # Feature data
            if hasattr(self, 'X_train_scaled_'):
                memory_estimate += self.X_train_scaled_.nbytes
            
            # Weights and other arrays
            if hasattr(self.model_, 'estimator_weights_'):
                memory_estimate += self.model_.estimator_weights_.nbytes
            
            # Convert to MB
            memory_mb = memory_estimate / (1024 * 1024)
            
            return {
                'estimated_memory_mb': memory_mb,
                'memory_efficiency_score': 1.0 / (1.0 + memory_mb / 100)  # Normalize
            }
            
        except Exception:
            return {
                'estimated_memory_mb': 0.0,
                'memory_efficiency_score': 0.5
            }

    def _calculate_skewness(self, data):
        """Calculate skewness of data"""
        try:
            n = len(data)
            if n < 3:
                return 0.0
            
            mean_val = np.mean(data)
            std_val = np.std(data)
            
            if std_val == 0:
                return 0.0
            
            skewness = np.mean(((data - mean_val) / std_val) ** 3)
            return skewness
            
        except Exception:
            return 0.0

    def _calculate_kurtosis(self, data):
        """Calculate kurtosis of data"""
        try:
            n = len(data)
            if n < 4:
                return 3.0  # Normal distribution kurtosis
            
            mean_val = np.mean(data)
            std_val = np.std(data)
            
            if std_val == 0:
                return 3.0
            
            kurtosis = np.mean(((data - mean_val) / std_val) ** 4)
            return kurtosis
            
        except Exception:
            return 3.0

    def _calculate_improvement_consistency(self, error_values):
        """Calculate how consistently errors improve"""
        try:
            if len(error_values) < 3:
                return 0.0
            
            # Calculate consecutive improvements
            improvements = []
            for i in range(1, len(error_values)):
                if error_values[i-1] > 0:
                    improvement = (error_values[i-1] - error_values[i]) / error_values[i-1]
                    improvements.append(improvement)
            
            if not improvements:
                return 0.0
            
            # Consistency = 1 - coefficient of variation of improvements
            mean_improvement = np.mean(improvements)
            std_improvement = np.std(improvements)
            
            if abs(mean_improvement) < 1e-10:
                return 0.0
            
            cv = std_improvement / abs(mean_improvement)
            consistency = 1.0 / (1.0 + cv)
            
            return consistency
            
        except Exception:
            return 0.0        
        
    def _detect_plateau(self, staged_errors, window_size=10, threshold=0.005):
        """Detect performance plateau in staged errors"""
        try:
            if len(staged_errors) < window_size:
                return {
                    'plateau_detected': False,
                    'plateau_start': None,
                    'plateau_length': 0,
                    'plateau_improvement': 0.0
                }
            
            mse_values = [e['mse'] for e in staged_errors]
            
            # Check for plateau by looking for periods of minimal improvement
            for i in range(window_size, len(mse_values)):
                window = mse_values[i-window_size:i]
                
                # Calculate improvement over window
                improvement = (window[0] - window[-1]) / window[0] if window[0] > 0 else 0.0
                
                if improvement < threshold:
                    return {
                        'plateau_detected': True,
                        'plateau_start': i - window_size + 1,
                        'plateau_length': window_size,
                        'plateau_improvement': improvement,
                        'plateau_mse_range': max(window) - min(window),
                        'plateau_stability': 1.0 - (np.std(window) / (np.mean(window) + 1e-10))
                    }
            
            return {
                'plateau_detected': False,
                'plateau_start': None,
                'plateau_length': 0,
                'plateau_improvement': 0.0
            }
            
        except Exception:
            return {
                'plateau_detected': False,
                'plateau_start': None,
                'plateau_length': 0,
                'plateau_improvement': 0.0
            }

    def _recommend_early_stopping(self, staged_errors, patience=10, min_improvement=0.001):
        """Recommend optimal early stopping point"""
        try:
            if len(staged_errors) < patience:
                return len(staged_errors)
            
            mse_values = [e['mse'] for e in staged_errors]
            
            # Find the best performance point
            best_mse = min(mse_values)
            best_stage = mse_values.index(best_mse) + 1
            
            # Check if performance hasn't improved for 'patience' stages after best
            if best_stage + patience < len(mse_values):
                subsequent_values = mse_values[best_stage:best_stage + patience]
                min_subsequent = min(subsequent_values)
                
                # If no significant improvement after best point
                if (best_mse - min_subsequent) / best_mse < min_improvement:
                    return best_stage
            
            # Alternative: find first point where improvement stops
            for i in range(patience, len(mse_values)):
                recent_window = mse_values[i-patience:i]
                older_window = mse_values[i-patience*2:i-patience] if i >= patience*2 else mse_values[:i-patience]
                
                if older_window:
                    recent_avg = np.mean(recent_window)
                    older_avg = np.mean(older_window)
                    
                    improvement = (older_avg - recent_avg) / older_avg if older_avg > 0 else 0.0
                    
                    if improvement < min_improvement:
                        return i - patience + 1
            
            # Default to full training if no early stopping point found
            return len(staged_errors)
            
        except Exception:
            return len(staged_errors) if staged_errors else 50

    def _assess_convergence_quality(self, staged_errors, convergence_detected, convergence_stage):
        """Assess the quality of convergence"""
        try:
            if not staged_errors:
                return {
                    'quality_score': 0.0,
                    'quality_rating': 'unknown',
                    'quality_factors': {}
                }
            
            mse_values = [e['mse'] for e in staged_errors]
            r2_values = [e['r2'] for e in staged_errors]
            
            quality_factors = {}
            
            # Factor 1: Final performance quality
            final_r2 = r2_values[-1]
            if final_r2 > 0.9:
                performance_score = 1.0
            elif final_r2 > 0.8:
                performance_score = 0.8
            elif final_r2 > 0.6:
                performance_score = 0.6
            else:
                performance_score = 0.4
            
            quality_factors['performance_quality'] = performance_score
            
            # Factor 2: Convergence speed
            if convergence_stage and convergence_stage < len(staged_errors) * 0.5:
                speed_score = 1.0
            elif convergence_stage and convergence_stage < len(staged_errors) * 0.7:
                speed_score = 0.8
            else:
                speed_score = 0.6
            
            quality_factors['convergence_speed'] = speed_score
            
            # Factor 3: Stability (low oscillation)
            oscillation_detected = self._detect_oscillations(mse_values)
            stability_score = 0.4 if oscillation_detected else 1.0
            quality_factors['stability'] = stability_score
            
            # Factor 4: Improvement consistency
            improvement_consistency = self._calculate_improvement_consistency(mse_values)
            quality_factors['consistency'] = improvement_consistency
            
            # Factor 5: Convergence detection reliability
            detection_score = 1.0 if convergence_detected else 0.5
            quality_factors['detection_reliability'] = detection_score
            
            # Overall quality score
            weights = [0.3, 0.2, 0.2, 0.2, 0.1]  # Weights for each factor
            factors = [performance_score, speed_score, stability_score, improvement_consistency, detection_score]
            
            quality_score = sum(w * f for w, f in zip(weights, factors))
            
            # Quality rating
            if quality_score > 0.8:
                quality_rating = 'excellent'
            elif quality_score > 0.6:
                quality_rating = 'good'
            elif quality_score > 0.4:
                quality_rating = 'fair'
            else:
                quality_rating = 'poor'
            
            return {
                'quality_score': quality_score,
                'quality_rating': quality_rating,
                'quality_factors': quality_factors,
                'weighted_factors': dict(zip(['performance', 'speed', 'stability', 'consistency', 'detection'], factors))
            }
            
        except Exception:
            return {
                'quality_score': 0.5,
                'quality_rating': 'unknown',
                'quality_factors': {}
            }

    def _approximate_bias_variance_decomposition(self, staged_predictions):
        """Approximate bias-variance decomposition using staged predictions"""
        try:
            staged_predictions = np.array(staged_predictions)
            
            # Use different subsets of estimators to approximate bootstrap samples
            n_stages = len(staged_predictions)
            n_samples = len(self.y_train_)
            
            # Create "bootstrap-like" predictions by using different stage subsets
            bootstrap_predictions = []
            for i in range(min(10, n_stages)):  # Limit to 10 bootstrap samples
                # Use every i-th stage to create diversity
                step = max(1, n_stages // 10)
                subset_indices = list(range(i, n_stages, step))
                if subset_indices:
                    subset_preds = staged_predictions[subset_indices]
                    mean_pred = np.mean(subset_preds, axis=0)
                    bootstrap_predictions.append(mean_pred)
            
            if len(bootstrap_predictions) < 2:
                return {
                    'bias_squared': 0.0,
                    'variance': 0.0,
                    'noise': 0.0,
                    'total_error': 0.0,
                    'bias_variance_ratio': 0.0
                }
            
            bootstrap_predictions = np.array(bootstrap_predictions)
            
            # Calculate main prediction (average of all bootstrap predictions)
            main_prediction = np.mean(bootstrap_predictions, axis=0)
            
            # Bias squared: squared difference between main prediction and true values
            bias_squared = np.mean((main_prediction - self.y_train_) ** 2)
            
            # Variance: average variance of bootstrap predictions
            variance = np.mean(np.var(bootstrap_predictions, axis=0))
            
            # Noise: irreducible error (approximated)
            final_pred = staged_predictions[-1]
            total_mse = mean_squared_error(self.y_train_, final_pred)
            noise = max(0.0, total_mse - bias_squared - variance)
            
            # Bias-variance ratio
            bias_variance_ratio = bias_squared / (variance + 1e-10)
            
            return {
                'bias_squared': bias_squared,
                'variance': variance,
                'noise': noise,
                'total_error': bias_squared + variance + noise,
                'bias_variance_ratio': bias_variance_ratio,
                'bias_percentage': bias_squared / (total_mse + 1e-10) * 100,
                'variance_percentage': variance / (total_mse + 1e-10) * 100,
                'noise_percentage': noise / (total_mse + 1e-10) * 100
            }
            
        except Exception:
            return {
                'bias_squared': 0.0,
                'variance': 0.0,
                'noise': 0.0,
                'total_error': 0.0,
                'bias_variance_ratio': 0.0
            }

    def _analyze_residuals(self, residuals):
        """Analyze residual patterns and properties"""
        try:
            analysis = {
                'basic_stats': {
                    'mean': np.mean(residuals),
                    'std': np.std(residuals),
                    'min': np.min(residuals),
                    'max': np.max(residuals),
                    'median': np.median(residuals),
                    'range': np.max(residuals) - np.min(residuals)
                },
                'distribution_properties': {
                    'skewness': self._calculate_skewness(residuals),
                    'kurtosis': self._calculate_kurtosis(residuals),
                    'is_normal': self._test_normality(residuals)
                },
                'outlier_analysis': {
                    'outlier_count': 0,
                    'outlier_percentage': 0.0,
                    'outlier_threshold': 0.0
                },
                'pattern_analysis': {
                    'autocorrelation': self._calculate_autocorrelation(residuals),
                    'heteroscedasticity': self._test_heteroscedasticity(residuals),
                    'trend': np.polyfit(range(len(residuals)), residuals, 1)[0] if len(residuals) > 1 else 0.0
                }
            }
            
            # Outlier analysis using IQR method
            q25 = np.percentile(residuals, 25)
            q75 = np.percentile(residuals, 75)
            iqr = q75 - q25
            outlier_threshold = 1.5 * iqr
            
            outliers = np.abs(residuals - np.median(residuals)) > outlier_threshold
            analysis['outlier_analysis']['outlier_count'] = np.sum(outliers)
            analysis['outlier_analysis']['outlier_percentage'] = np.mean(outliers) * 100
            analysis['outlier_analysis']['outlier_threshold'] = outlier_threshold
            
            return analysis
            
        except Exception:
            return {
                'basic_stats': {},
                'distribution_properties': {},
                'outlier_analysis': {},
                'pattern_analysis': {}
            }

    def _analyze_error_distribution_changes(self, staged_predictions):
        """Analyze how error distributions change across boosting stages"""
        try:
            n_stages = len(staged_predictions)
            if n_stages < 3:
                return {
                    'error_evolution': [],
                    'distribution_changes': [],
                    'convergence_pattern': 'insufficient_data'
                }
            
            error_evolution = []
            distribution_changes = []
            
            for i, pred in enumerate(staged_predictions):
                residuals = self.y_train_ - pred
                
                # Error distribution statistics
                error_stats = {
                    'stage': i + 1,
                    'mean_absolute_error': np.mean(np.abs(residuals)),
                    'error_std': np.std(residuals),
                    'error_skewness': self._calculate_skewness(residuals),
                    'error_kurtosis': self._calculate_kurtosis(residuals),
                    'error_range': np.max(residuals) - np.min(residuals),
                    'outlier_percentage': self._calculate_outlier_percentage(residuals)
                }
                
                error_evolution.append(error_stats)
                
                # Compare with previous stage if available
                if i > 0:
                    prev_residuals = self.y_train_ - staged_predictions[i-1]
                    
                    distribution_change = {
                        'stage_transition': f"{i} to {i+1}",
                        'mean_error_change': np.mean(np.abs(residuals)) - np.mean(np.abs(prev_residuals)),
                        'std_change': np.std(residuals) - np.std(prev_residuals),
                        'skewness_change': self._calculate_skewness(residuals) - self._calculate_skewness(prev_residuals),
                        'distribution_similarity': self._calculate_distribution_similarity(residuals, prev_residuals)
                    }
                    
                    distribution_changes.append(distribution_change)
            
            # Analyze convergence pattern
            mean_errors = [e['mean_absolute_error'] for e in error_evolution]
            error_stds = [e['error_std'] for e in error_evolution]
            
            if len(mean_errors) > 5:
                # Check for monotonic decrease
                decreasing_trend = all(mean_errors[i] >= mean_errors[i+1] for i in range(len(mean_errors)-1))
                
                # Check for stabilization
                recent_errors = mean_errors[-5:]
                stabilized = np.std(recent_errors) / np.mean(recent_errors) < 0.05
                
                if decreasing_trend and stabilized:
                    convergence_pattern = 'smooth_convergence'
                elif decreasing_trend:
                    convergence_pattern = 'monotonic_decrease'
                elif stabilized:
                    convergence_pattern = 'stabilized'
                else:
                    convergence_pattern = 'irregular'
            else:
                convergence_pattern = 'insufficient_data'
            
            return {
                'error_evolution': error_evolution,
                'distribution_changes': distribution_changes,
                'convergence_pattern': convergence_pattern,
                'final_error_properties': error_evolution[-1] if error_evolution else {},
                'overall_improvement': {
                    'initial_mae': error_evolution[0]['mean_absolute_error'] if error_evolution else 0,
                    'final_mae': error_evolution[-1]['mean_absolute_error'] if error_evolution else 0,
                    'relative_improvement': ((error_evolution[0]['mean_absolute_error'] - error_evolution[-1]['mean_absolute_error']) / 
                                           error_evolution[0]['mean_absolute_error']) if error_evolution and error_evolution[0]['mean_absolute_error'] > 0 else 0
                }
            }
            
        except Exception:
            return {
                'error_evolution': [],
                'distribution_changes': [],
                'convergence_pattern': 'error'
            }

    def _analyze_overfitting_from_curves(self, train_means, val_means, gaps):
        """Analyze overfitting from learning curves"""
        try:
            overfitting_analysis = {
                'overfitting_score': 0.0,
                'overfitting_severity': 'none',
                'gap_analysis': {},
                'recommendations': []
            }
            
            # Calculate overfitting score based on gap
            final_gap = gaps[-1]
            max_gap = np.max(gaps)
            gap_trend = np.polyfit(range(len(gaps)), gaps, 1)[0] if len(gaps) > 1 else 0
            
            # Normalize gaps relative to validation error
            relative_gap = final_gap / (val_means[-1] + 1e-10)
            
            overfitting_analysis['gap_analysis'] = {
                'final_gap': final_gap,
                'max_gap': max_gap,
                'relative_gap': relative_gap,
                'gap_trend': gap_trend,
                'gap_stability': np.std(gaps[-3:]) if len(gaps) >= 3 else 0
            }
            
            # Overfitting severity assessment
            if relative_gap > 0.3:
                severity = 'severe'
                score = 0.9
            elif relative_gap > 0.15:
                severity = 'moderate'
                score = 0.6
            elif relative_gap > 0.05:
                severity = 'mild'
                score = 0.3
            else:
                severity = 'none'
                score = 0.1
            
            overfitting_analysis['overfitting_score'] = score
            overfitting_analysis['overfitting_severity'] = severity
            
            # Generate recommendations
            if severity == 'severe':
                overfitting_analysis['recommendations'] = [
                    "Consider reducing model complexity (fewer estimators, higher learning rate)",
                    "Implement early stopping",
                    "Add regularization",
                    "Increase training data if possible"
                ]
            elif severity == 'moderate':
                overfitting_analysis['recommendations'] = [
                    "Consider early stopping",
                    "Monitor validation performance during training",
                    "Reduce number of estimators slightly"
                ]
            elif severity == 'mild':
                overfitting_analysis['recommendations'] = [
                    "Current overfitting level is acceptable",
                    "Monitor with larger datasets"
                ]
            
            return overfitting_analysis
            
        except Exception:
            return {
                'overfitting_score': 0.0,
                'overfitting_severity': 'unknown',
                'gap_analysis': {},
                'recommendations': []
            }

    def _analyze_data_efficiency(self, train_sizes, val_means):
        """Analyze data efficiency from learning curves"""
        try:
            efficiency_analysis = {
                'efficiency_score': 0.0,
                'optimal_data_size': None,
                'diminishing_returns_point': None,
                'data_utilization': {},
                'recommendations': []
            }
            
            if len(train_sizes) < 3 or len(val_means) < 3:
                return efficiency_analysis
            
            # Calculate improvement rate at each point
            improvements = []
            for i in range(1, len(val_means)):
                if val_means[i-1] > 0:
                    improvement = (val_means[i-1] - val_means[i]) / val_means[i-1]
                    improvements.append(improvement)
                else:
                    improvements.append(0.0)
            
            # Find diminishing returns point (where improvement rate drops significantly)
            if len(improvements) > 2:
                improvement_changes = []
                for i in range(1, len(improvements)):
                    change = improvements[i-1] - improvements[i]
                    improvement_changes.append(change)
                
                # Find point where improvement change becomes small
                threshold = np.mean(improvement_changes) * 0.5
                for i, change in enumerate(improvement_changes):
                    if change < threshold:
                        efficiency_analysis['diminishing_returns_point'] = train_sizes[i+2]
                        break
            
            # Calculate overall efficiency (performance per data point)
            final_performance = 1.0 / (val_means[-1] + 1e-10)  # Higher is better
            data_efficiency = final_performance / train_sizes[-1]
            efficiency_analysis['efficiency_score'] = min(1.0, data_efficiency * 1000)  # Normalize
            
            # Data utilization analysis
            efficiency_analysis['data_utilization'] = {
                'performance_at_50pct': val_means[len(val_means)//2] if len(val_means) > 2 else val_means[-1],
                'performance_at_100pct': val_means[-1],
                'relative_benefit_full_data': (val_means[len(val_means)//2] - val_means[-1]) / val_means[len(val_means)//2] if len(val_means) > 2 and val_means[len(val_means)//2] > 0 else 0,
                'optimal_size_estimate': efficiency_analysis['diminishing_returns_point'] or train_sizes[-1]
            }
            
            # Generate recommendations
            if efficiency_analysis['diminishing_returns_point'] and efficiency_analysis['diminishing_returns_point'] < train_sizes[-1] * 0.8:
                efficiency_analysis['recommendations'].append(
                    f"Consider using approximately {int(efficiency_analysis['diminishing_returns_point'])} samples for optimal efficiency"
                )
            
            if efficiency_analysis['efficiency_score'] < 0.3:
                efficiency_analysis['recommendations'].append(
                    "Model shows low data efficiency - consider simpler model or feature engineering"
                )
            elif efficiency_analysis['efficiency_score'] > 0.7:
                efficiency_analysis['recommendations'].append(
                    "Model shows good data efficiency"
                )
            
            return efficiency_analysis
            
        except Exception:
            return {
                'efficiency_score': 0.0,
                'optimal_data_size': None,
                'diminishing_returns_point': None,
                'data_utilization': {},
                'recommendations': []
            }

    def _analyze_convergence_by_size(self, train_sizes, val_means, val_stds):
        """Analyze how convergence behavior changes with training set size"""
        try:
            convergence_analysis = {
                'size_convergence_relationship': [],
                'optimal_size_range': None,
                'convergence_stability': {},
                'scalability_assessment': {}
            }
            
            # Analyze convergence at each training size
            for i, (size, mean_val, std_val) in enumerate(zip(train_sizes, val_means, val_stds)):
                size_analysis = {
                    'training_size': size,
                    'validation_performance': mean_val,
                    'performance_stability': 1.0 / (1.0 + std_val / (mean_val + 1e-10)),
                    'convergence_quality': self._assess_size_convergence_quality(mean_val, std_val)
                }
                convergence_analysis['size_convergence_relationship'].append(size_analysis)
            
            # Find optimal size range (balance between performance and stability)
            stability_scores = [s['performance_stability'] for s in convergence_analysis['size_convergence_relationship']]
            performance_scores = [1.0 / (1.0 + s['validation_performance']) for s in convergence_analysis['size_convergence_relationship']]  # Lower validation error is better
            
            # Combined score (weighted average of performance and stability)
            combined_scores = [0.7 * perf + 0.3 * stab for perf, stab in zip(performance_scores, stability_scores)]
            best_idx = np.argmax(combined_scores)
            
            convergence_analysis['optimal_size_range'] = {
                'optimal_size': train_sizes[best_idx],
                'optimal_performance': val_means[best_idx],
                'optimal_stability': stability_scores[best_idx],
                'combined_score': combined_scores[best_idx]
            }
            
            # Convergence stability assessment
            convergence_analysis['convergence_stability'] = {
                'stability_trend': np.polyfit(train_sizes, stability_scores, 1)[0] if len(stability_scores) > 1 else 0,
                'performance_trend': np.polyfit(train_sizes, val_means, 1)[0] if len(val_means) > 1 else 0,
                'stability_consistency': 1.0 - np.std(stability_scores) / (np.mean(stability_scores) + 1e-10)
            }
            
            # Scalability assessment
            if len(train_sizes) > 2:
                size_ratios = [train_sizes[i] / train_sizes[i-1] for i in range(1, len(train_sizes))]
                perf_ratios = [val_means[i-1] / val_means[i] for i in range(1, len(val_means)) if val_means[i] > 0]
                
                if perf_ratios:
                    convergence_analysis['scalability_assessment'] = {
                        'average_size_ratio': np.mean(size_ratios),
                        'average_performance_ratio': np.mean(perf_ratios),
                        'scalability_efficiency': np.mean(perf_ratios) / np.mean(size_ratios),
                        'scalability_rating': 'good' if np.mean(perf_ratios) > np.mean(size_ratios) else 'poor'
                    }
            
            return convergence_analysis
            
        except Exception:
            return {
                'size_convergence_relationship': [],
                'optimal_size_range': None,
                'convergence_stability': {},
                'scalability_assessment': {}
            }

    def _test_normality(self, data, alpha=0.05):
        """Simple normality test using skewness and kurtosis"""
        try:
            skewness = self._calculate_skewness(data)
            kurtosis = self._calculate_kurtosis(data)
            
            # Simple heuristic: normal if |skewness| < 1 and |kurtosis - 3| < 1
            is_normal = abs(skewness) < 1.0 and abs(kurtosis - 3.0) < 1.0
            
            return {
                'is_normal': is_normal,
                'skewness': skewness,
                'kurtosis': kurtosis,
                'test_method': 'heuristic'
            }
            
        except Exception:
            return {
                'is_normal': False,
                'skewness': 0.0,
                'kurtosis': 3.0,
                'test_method': 'failed'
            }

    def _calculate_autocorrelation(self, data, lag=1):
        """Calculate autocorrelation of residuals"""
        try:
            if len(data) <= lag:
                return 0.0
            
            # Simple lag-1 autocorrelation
            data_shifted = data[:-lag]
            data_lag = data[lag:]
            
            correlation = np.corrcoef(data_shifted, data_lag)[0, 1]
            return correlation if not np.isnan(correlation) else 0.0
            
        except Exception:
            return 0.0

    def _test_heteroscedasticity(self, residuals):
        """Simple test for heteroscedasticity"""
        try:
            # Split data into halves and compare variances
            n = len(residuals)
            if n < 10:
                return {
                    'heteroscedastic': False,
                    'variance_ratio': 1.0,
                    'test_method': 'insufficient_data'
                }
            
            mid = n // 2
            first_half = residuals[:mid]
            second_half = residuals[mid:]
            
            var1 = np.var(first_half)
            var2 = np.var(second_half)
            
            variance_ratio = max(var1, var2) / (min(var1, var2) + 1e-10)
            
            # Simple threshold: ratio > 2 suggests heteroscedasticity
            heteroscedastic = variance_ratio > 2.0
            
            return {
                'heteroscedastic': heteroscedastic,
                'variance_ratio': variance_ratio,
                'first_half_var': var1,
                'second_half_var': var2,
                'test_method': 'variance_comparison'
            }
            
        except Exception:
            return {
                'heteroscedastic': False,
                'variance_ratio': 1.0,
                'test_method': 'failed'
            }

    def _calculate_outlier_percentage(self, data):
        """Calculate percentage of outliers using IQR method"""
        try:
            q25 = np.percentile(data, 25)
            q75 = np.percentile(data, 75)
            iqr = q75 - q25
            
            outlier_threshold = 1.5 * iqr
            outliers = np.abs(data - np.median(data)) > outlier_threshold
            
            return np.mean(outliers) * 100
            
        except Exception:
            return 0.0

    def _calculate_distribution_similarity(self, data1, data2):
        """Calculate similarity between two distributions using KL divergence approximation"""
        try:
            # Simple approximation using histograms
            min_val = min(np.min(data1), np.min(data2))
            max_val = max(np.max(data1), np.max(data2))
            
            bins = np.linspace(min_val, max_val, 20)
            
            hist1, _ = np.histogram(data1, bins=bins, density=True)
            hist2, _ = np.histogram(data2, bins=bins, density=True)
            
            # Add small epsilon to avoid log(0)
            epsilon = 1e-10
            hist1 = hist1 + epsilon
            hist2 = hist2 + epsilon
            
            # Normalize
            hist1 = hist1 / np.sum(hist1)
            hist2 = hist2 / np.sum(hist2)
            
            # Jensen-Shannon divergence (symmetric version of KL divergence)
            m = 0.5 * (hist1 + hist2)
            js_div = 0.5 * np.sum(hist1 * np.log(hist1 / m)) + 0.5 * np.sum(hist2 * np.log(hist2 / m))
            
            # Convert to similarity (0 = identical, 1 = completely different)
            similarity = 1.0 - min(1.0, js_div)
            
            return similarity
            
        except Exception:
            return 0.5

    def _assess_size_convergence_quality(self, mean_val, std_val):
        """Assess convergence quality for a specific training size"""
        try:
            # Lower validation error and lower std = higher quality
            performance_score = 1.0 / (1.0 + mean_val)
            stability_score = 1.0 / (1.0 + std_val)
            
            # Combined quality score
            quality = 0.7 * performance_score + 0.3 * stability_score
            
            return min(1.0, quality)
            
        except Exception:
            return 0.5

    def _compare_with_single_estimator(self):
        """Compare AdaBoost performance with a single base estimator"""
        try:
            # This is already partially implemented in _analyze_estimator_comparison
            # We can extract the single tree comparison from there
            if hasattr(self, 'estimator_comparison_analysis_'):
                single_tree_results = self.estimator_comparison_analysis_.get('method_comparisons', {}).get('Single_DecisionTree', {})
                
                self.single_estimator_comparison_ = {
                    'single_estimator_performance': single_tree_results,
                    'ensemble_vs_single': {
                        'performance_improvement': 'calculated_in_estimator_comparison',
                        'complexity_increase': 'calculated_in_estimator_comparison'
                    }
                }
            
        except Exception as e:
            self.single_estimator_comparison_ = {'error': str(e)}

    def _compare_with_random_forest(self):
        """Compare AdaBoost performance with Random Forest"""
        try:
            # This is already partially implemented in _analyze_estimator_comparison
            # We can extract the Random Forest comparison from there
            if hasattr(self, 'estimator_comparison_analysis_'):
                rf_results = self.estimator_comparison_analysis_.get('method_comparisons', {}).get('RandomForest', {})
                
                self.random_forest_comparison_ = {
                    'random_forest_performance': rf_results,
                    'adaboost_vs_rf': {
                        'performance_comparison': 'calculated_in_estimator_comparison',
                        'trade_offs': 'calculated_in_estimator_comparison'
                    }
                }
            
        except Exception as e:
            self.random_forest_comparison_ = {'error': str(e)}
           
# ...existing code...

    def get_analysis_results(self) -> Dict[str, Any]:
        """
        Get comprehensive analysis results from all performed analyses
        
        Returns:
        --------
        Dict[str, Any]
            Dictionary containing all analysis results with structured information
        """
        if not self.is_fitted_:
            return {'error': 'Model must be fitted before getting analysis results'}
        
        analysis_results = {
            'model_info': {
                'algorithm_name': self._name,
                'algorithm_category': self._category,
                'n_estimators': self.n_estimators,
                'learning_rate': self.learning_rate,
                'loss_function': self.loss,
                'base_estimator_type': type(self.base_estimator_).__name__ if self.base_estimator_ else 'DecisionTreeRegressor',
                'n_features': self.n_features_in_,
                'feature_names': self.feature_names_,
                'is_fitted': self.is_fitted_,
                'preprocessing': {
                    'scaling_applied': self.scaler_ is not None,
                    'scaler_type': self.scaler_type if self.scaler_ is not None else None
                }
            },
            
            'performance_summary': {
                'training_performance': self._get_training_performance_summary(),
                'cross_validation_performance': self._get_cv_performance_summary(),
                'convergence_summary': self._get_convergence_summary(),
                'generalization_assessment': self._get_generalization_assessment()
            },
            
            'detailed_analyses': {
                'feature_importance': self.feature_importance_analysis_,
                'boosting_analysis': self.boosting_analysis_,
                'base_estimator_analysis': self.base_estimator_analysis_,
                'convergence_analysis': self.convergence_analysis_,
                'error_evolution_analysis': self.error_evolution_analysis_,
                'learning_curve_analysis': self.learning_curve_analysis_,
                'cross_validation_analysis': self.cross_validation_analysis_,
                'loss_function_analysis': self.loss_function_analysis_,
                'prediction_variance_analysis': self.prediction_variance_analysis_,
                'estimator_comparison_analysis': self.estimator_comparison_analysis_,
                'performance_profile': self.performance_profile_
            },
            
            'recommendations': self._generate_comprehensive_recommendations(),
            'insights': self._extract_key_insights(),
            'warnings': self._identify_potential_issues(),
            
            'visualization_data': self._prepare_visualization_data(),
            'export_summary': self._create_export_summary()
        }
        
        return analysis_results

    def display_results(self, show_plots=True, max_plots=None):
        """
        Display comprehensive analysis results in Streamlit interface
        
        Parameters:
        -----------
        show_plots : bool, default=True
            Whether to display visualization plots
        max_plots : int, optional
            Maximum number of plots to display
        """
        if not self.is_fitted_:
            st.error("❌ Model must be fitted before displaying results")
            return
        
        # Set max_plots from instance variable if not provided
        if max_plots is None:
            max_plots = self.max_plots
        
        st.header(f"🎯 {self._name} - Analysis Results")
        
        # Performance Overview
        self._display_performance_overview()
        
        # Boosting Process Analysis
        if self.boosting_analysis and hasattr(self, 'boosting_analysis_') and self.boosting_analysis_:
            st.subheader("🔄 Boosting Process Analysis")
            self._display_boosting_analysis()
        
        # Feature Importance Analysis
        if self.compute_feature_importance and hasattr(self, 'feature_importance_analysis_') and self.feature_importance_analysis_:
            st.subheader("📊 Feature Importance Analysis")
            self._display_feature_importance_analysis()
        
        # Convergence Analysis
        if self.convergence_analysis and hasattr(self, 'convergence_analysis_') and self.convergence_analysis_:
            st.subheader("📈 Convergence Analysis")
            self._display_convergence_analysis()
        
        # Error Evolution Analysis
        if self.error_evolution_analysis and hasattr(self, 'error_evolution_analysis_') and self.error_evolution_analysis_:
            st.subheader("📉 Error Evolution Analysis")
            self._display_error_evolution_analysis()
        
        # Learning Curves
        if self.learning_curve_analysis and hasattr(self, 'learning_curve_analysis_') and self.learning_curve_analysis_:
            st.subheader("📚 Learning Curve Analysis")
            self._display_learning_curve_analysis()
        
        # Cross-Validation Analysis
        if self.cross_validation_analysis and hasattr(self, 'cross_validation_analysis_') and self.cross_validation_analysis_:
            st.subheader("🔄 Cross-Validation Analysis")
            self._display_cross_validation_analysis()
        
        # Base Estimator Analysis
        if self.base_estimator_analysis and hasattr(self, 'base_estimator_analysis_') and self.base_estimator_analysis_:
            st.subheader("🌳 Base Estimator Analysis")
            self._display_base_estimator_analysis()
        
        # Prediction Variance Analysis
        if self.prediction_variance_analysis and hasattr(self, 'prediction_variance_analysis_') and self.prediction_variance_analysis_:
            st.subheader("🎲 Prediction Variance Analysis")
            self._display_prediction_variance_analysis()
        
        # Loss Function Analysis
        if self.loss_function_analysis and hasattr(self, 'loss_function_analysis_') and self.loss_function_analysis_:
            st.subheader("⚖️ Loss Function Analysis")
            self._display_loss_function_analysis()
        
        # Estimator Comparison
        if self.estimator_comparison_analysis and hasattr(self, 'estimator_comparison_analysis_') and self.estimator_comparison_analysis_:
            st.subheader("🔀 Model Comparison Analysis")
            self._display_estimator_comparison_analysis()
        
        # Performance Profiling
        if self.performance_profiling and hasattr(self, 'performance_profile_') and self.performance_profile_:
            st.subheader("⚡ Performance Profiling")
            self._display_performance_profiling()
        
        # Comprehensive Recommendations
        st.subheader("💡 Recommendations & Insights")
        self._display_recommendations_and_insights()

    def create_visualizations(self, plot_types=None, save_plots=False, plot_dir=None):
        """
        Create comprehensive visualizations for AdaBoost analysis
        
        Parameters:
        -----------
        plot_types : list, optional
            Specific types of plots to create. If None, creates all available plots
        save_plots : bool, default=False
            Whether to save plots to files
        plot_dir : str, optional
            Directory to save plots (if save_plots=True)
        
        Returns:
        --------
        Dict[str, Any]
            Dictionary containing plot objects and metadata
        """
        if not self.is_fitted_:
            return {'error': 'Model must be fitted before creating visualizations'}
        
        # Available plot types
        available_plots = {
            'boosting_stages': self._plot_boosting_stages,
            'feature_importance': self._plot_feature_importance,
            'error_evolution': self._plot_error_evolution,
            'convergence_analysis': self._plot_convergence_analysis,
            'learning_curves': self._plot_learning_curves,
            'prediction_variance': self._plot_prediction_variance,
            'residual_analysis': self._plot_residual_analysis,
            'base_estimator_analysis': self._plot_base_estimator_analysis,
            'loss_function_comparison': self._plot_loss_function_comparison,
            'estimator_comparison': self._plot_estimator_comparison,
            'performance_profile': self._plot_performance_profile,
            'weight_evolution': self._plot_weight_evolution,
            'prediction_intervals': self._plot_prediction_intervals,
            'cross_validation_analysis': self._plot_cross_validation_analysis,
            'bias_variance_analysis': self._plot_bias_variance_analysis
        }
        
        # Determine which plots to create
        if plot_types is None:
            plots_to_create = available_plots
        else:
            plots_to_create = {k: v for k, v in available_plots.items() if k in plot_types}
        
        # Create plots
        created_plots = {}
        plot_count = 0
        
        for plot_name, plot_function in plots_to_create.items():
            if plot_count >= self.max_plots:
                break
                
            try:
                fig = plot_function()
                if fig is not None:
                    created_plots[plot_name] = {
                        'figure': fig,
                        'title': plot_name.replace('_', ' ').title(),
                        'description': self._get_plot_description(plot_name)
                    }
                    
                    # Save plot if requested
                    if save_plots and plot_dir:
                        self._save_plot(fig, plot_name, plot_dir)
                    
                    plot_count += 1
                    
            except Exception as e:
                created_plots[plot_name] = {'error': str(e)}
        
        return {
            'plots': created_plots,
            'total_plots_created': plot_count,
            'max_plots_limit': self.max_plots,
            'available_plot_types': list(available_plots.keys())
        }        

    def _get_training_performance_summary(self):
        """Get summary of training performance metrics"""
        try:
            if not hasattr(self, 'model_') or self.model_ is None:
                return {'error': 'Model not fitted'}
            
            # Get training predictions
            train_pred = self.model_.predict(self.X_train_scaled_)
            
            # Calculate basic metrics
            train_mse = mean_squared_error(self.y_train_, train_pred)
            train_r2 = r2_score(self.y_train_, train_pred)
            train_mae = mean_absolute_error(self.y_train_, train_pred)
            
            summary = {
                'mse': train_mse,
                'rmse': np.sqrt(train_mse),
                'mae': train_mae,
                'r2': train_r2,
                'n_estimators_used': len(self.model_.estimators_),
                'final_training_score': train_r2
            }
            
            # Add boosting-specific metrics if available
            if hasattr(self, 'boosting_analysis_') and self.boosting_analysis_:
                boosting_info = self.boosting_analysis_.get('boosting_efficiency', {})
                summary.update({
                    'error_reduction_rate': boosting_info.get('error_reduction_rate', 0.0),
                    'convergence_efficiency': boosting_info.get('effectiveness_per_estimator', 0.0),
                    'boosting_effectiveness': boosting_info.get('effectiveness_per_estimator', 0.0)
                })
            
            # Add convergence info if available
            if hasattr(self, 'convergence_analysis_') and self.convergence_analysis_:
                conv_info = self.convergence_analysis_.get('convergence_metrics', {})
                summary.update({
                    'convergence_detected': self.convergence_analysis_.get('convergence_detected', False),
                    'final_improvement': conv_info.get('total_improvement', 0.0),
                    'convergence_stage': self.convergence_analysis_.get('convergence_stage')
                })
            
            return summary
            
        except Exception as e:
            return {'error': str(e)}

    def _get_cv_performance_summary(self):
        """Get summary of cross-validation performance"""
        try:
            if not hasattr(self, 'cross_validation_analysis_') or not self.cross_validation_analysis_:
                return {'error': 'Cross-validation analysis not performed'}
            
            cv_stats = self.cross_validation_analysis_.get('cv_statistics', {})
            
            summary = {
                'cv_r2_mean': cv_stats.get('r2', {}).get('mean', 0.0),
                'cv_r2_std': cv_stats.get('r2', {}).get('std', 0.0),
                'cv_mse_mean': cv_stats.get('mse', {}).get('mean', 0.0),
                'cv_mse_std': cv_stats.get('mse', {}).get('std', 0.0),
                'cv_mae_mean': cv_stats.get('mae', {}).get('mean', 0.0),
                'cv_mae_std': cv_stats.get('mae', {}).get('std', 0.0)
            }
            
            # Add stability metrics
            stability_metrics = self.cross_validation_analysis_.get('stability_metrics', {})
            summary.update({
                'performance_stability': stability_metrics.get('performance_stability', 0.0),
                'consistency_score': stability_metrics.get('consistency_score', 0.0),
                'consistency_rating': self.cross_validation_analysis_.get('performance_consistency', {}).get('consistency_rating', 'unknown')
            })
            
            # Calculate generalization gap
            train_summary = self._get_training_performance_summary()
            if 'error' not in train_summary:
                summary['generalization_gap'] = train_summary['r2'] - summary['cv_r2_mean']
            
            return summary
            
        except Exception as e:
            return {'error': str(e)}

    def _get_convergence_summary(self):
        """Get summary of convergence analysis"""
        try:
            if not hasattr(self, 'convergence_analysis_') or not self.convergence_analysis_:
                return {'error': 'Convergence analysis not performed'}
            
            conv_analysis = self.convergence_analysis_
            
            summary = {
                'convergence_detected': conv_analysis.get('convergence_detected', False),
                'convergence_stage': conv_analysis.get('convergence_stage'),
                'total_stages': self.n_estimators,
                'convergence_efficiency': 0.0
            }
            
            # Add convergence metrics
            conv_metrics = conv_analysis.get('convergence_metrics', {})
            summary.update({
                'improvement_rate': conv_metrics.get('improvement_rate', 0.0),
                'total_improvement': conv_metrics.get('total_improvement', 0.0),
                'final_mse': conv_metrics.get('final_mse', 0.0),
                'best_mse': conv_metrics.get('best_mse', 0.0)
            })
            
            # Calculate convergence efficiency
            if summary['convergence_stage'] and summary['total_stages'] > 0:
                summary['convergence_efficiency'] = summary['convergence_stage'] / summary['total_stages']
            
            # Add stability analysis
            stability = conv_analysis.get('stability_analysis', {})
            summary.update({
                'is_stable': stability.get('is_stable', False),
                'oscillation_detected': stability.get('oscillation_detected', False),
                'overfitting_risk': stability.get('overfitting_risk', 'unknown')
            })
            
            # Add early stopping recommendations
            early_stop = conv_analysis.get('early_stopping_recommendations', {})
            summary.update({
                'optimal_n_estimators': early_stop.get('optimal_n_estimators'),
                'potential_savings': early_stop.get('potential_savings', 0)
            })
            
            # Convergence quality
            quality = conv_analysis.get('convergence_quality', {})
            summary.update({
                'convergence_quality_score': quality.get('quality_score', 0.0),
                'convergence_quality_rating': quality.get('quality_rating', 'unknown')
            })
            
            return summary
            
        except Exception as e:
            return {'error': str(e)}
    # ADD THE OVERRIDDEN METHOD HERE:
    def get_algorithm_specific_metrics(self, 
                                       y_true: Union[pd.Series, np.ndarray], 
                                       y_pred: Union[pd.Series, np.ndarray], 
                                       y_proba: Optional[np.ndarray] = None  # y_proba is not used for regressors
                                       ) -> Dict[str, Any]:
        """
        Calculate AdaBoost Regressor-specific metrics related to its training and ensemble structure.

        Args:
            y_true: Ground truth target values from the test set.
            y_pred: Predicted target values on the test set.
            y_proba: Predicted probabilities (not used for regressors).

        Returns:
            A dictionary of AdaBoost Regressor-specific metrics.
        """
        metrics = {}
        if not self.is_fitted_ or self.model_ is None:
            metrics["status"] = "Model not fitted or not available"
            return metrics

        # Metrics from the underlying scikit-learn AdaBoostRegressor model
        if hasattr(self.model_, 'estimator_weights_') and self.model_.estimator_weights_ is not None:
            metrics['mean_estimator_weight'] = float(np.mean(self.model_.estimator_weights_))
            metrics['std_estimator_weight'] = float(np.std(self.model_.estimator_weights_))
            metrics['min_estimator_weight'] = float(np.min(self.model_.estimator_weights_))
            metrics['max_estimator_weight'] = float(np.max(self.model_.estimator_weights_))
        
        if hasattr(self.model_, 'estimator_errors_') and self.model_.estimator_errors_ is not None:
            metrics['mean_estimator_median_loss'] = float(np.mean(self.model_.estimator_errors_)) # Median loss for each estimator
            metrics['std_estimator_median_loss'] = float(np.std(self.model_.estimator_errors_))
            if len(self.model_.estimator_errors_) > 0:
                metrics['final_estimator_median_loss'] = float(self.model_.estimator_errors_[-1])
        
        if hasattr(self.model_, 'estimators_'):
            metrics['n_estimators_trained'] = len(self.model_.estimators_)

        if hasattr(self.model_, 'feature_importances_') and self.model_.feature_importances_ is not None:
            metrics['mean_feature_importance'] = float(np.mean(self.model_.feature_importances_))
            # Storing all feature importances can be verbose; consider top N or if plots are primary.
            # metrics['feature_importances'] = self.model_.feature_importances_.tolist() 

        if hasattr(self.model_, 'loss'):
            metrics['loss_function_used'] = self.model_.loss
        
        # Metrics from the plugin's internal analysis (if available and populated)
        if hasattr(self, 'boosting_analysis_') and self.boosting_analysis_ and 'boosting_efficiency' in self.boosting_analysis_:
            boosting_efficiency = self.boosting_analysis_['boosting_efficiency']
            if 'error_reduction_rate' in boosting_efficiency:
                metrics['training_error_reduction_rate'] = boosting_efficiency['error_reduction_rate']
            if 'effectiveness_per_estimator' in boosting_efficiency:
                 metrics['training_effectiveness_per_estimator'] = boosting_efficiency['effectiveness_per_estimator']

        if hasattr(self, 'convergence_analysis_') and self.convergence_analysis_:
            if 'convergence_detected' in self.convergence_analysis_:
                metrics['convergence_detected_on_train'] = self.convergence_analysis_['convergence_detected']
            if 'convergence_stage' in self.convergence_analysis_:
                metrics['convergence_stage_on_train'] = self.convergence_analysis_['convergence_stage']
            if 'early_stopping_recommendations' in self.convergence_analysis_ and \
               'optimal_n_estimators' in self.convergence_analysis_['early_stopping_recommendations']:
                metrics['recommended_n_estimators_from_train'] = self.convergence_analysis_['early_stopping_recommendations']['optimal_n_estimators']
            if 'convergence_quality' in self.convergence_analysis_ and \
               'quality_rating' in self.convergence_analysis_['convergence_quality']:
                metrics['train_convergence_quality_rating'] = self.convergence_analysis_['convergence_quality']['quality_rating']


        # Example: Calculate Mean Absolute Percentage Error (MAPE) using y_true, y_pred
        # This is a standard metric, but demonstrates using the passed arguments
        # Ensure y_true does not contain zeros for MAPE calculation
        # if len(y_true) > 0 and len(y_pred) == len(y_true):
        #     y_true_safe = np.where(y_true == 0, 1e-8, y_true) # Avoid division by zero
        #     try:
        #         mape = np.mean(np.abs((y_true_safe - y_pred) / y_true_safe)) * 100
        #         metrics['MAPE_test_set'] = float(mape)
        #     except Exception as e:
        #         metrics['MAPE_test_set_error'] = str(e)
                
        return metrics

def get_plugin():
    """Factory function to get the plugin instance"""
    return AdaBoostRegressorPlugin()

        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        